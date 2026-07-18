//! Finite Scalar Quantization: the compressor's bottleneck.
//!
//! Mentzer et al., ["Finite Scalar Quantization: VQ-VAE Made
//! Simple"](https://arxiv.org/abs/2309.15505) (ICLR 2024). Project each latent
//! slot down to `d` scalars, squash each into a bounded range, and round it to
//! one of `L_i` levels. The implicit codebook is the product grid
//! `prod(L_i)`; there is no codebook *parameter*, no commitment loss, no EMA
//! update, no dead-code reseeding, and no entropy penalty.
//!
//! Two reasons this and not VQ-VAE, in the order they matter here.
//!
//! **1. It makes the compression ratio a fact rather than a claim.** A latent
//! of `K` slots holding floats is not a compressed anything -- one fp16 vector
//! of width 128 is 2048 bits, which at `log2(8192) = 13` bits per token is more
//! than 157 tokens' worth of channel. That is why "N tokens into K vectors =
//! N/K compression" is a *sequence-length* claim, not an information one, and
//! why the literature's headline ratios are mostly unfalsifiable
//! ([Kuratov et al., 2502.13063](https://arxiv.org/abs/2502.13063), which
//! states the capacity bound `L <= d*b / log2|V|` this module exposes as
//! [`Fsq::token_capacity`]). With FSQ the rate is `K * sum(log2 L_i)` bits.
//! Exactly. Countable before a single step of training.
//!
//! **2. There is no codebook to collapse.** VQ's failure mode is worst exactly
//! where this project lives: ["Representation Collapsing Problems in Vector
//! Quantization"](https://arxiv.org/abs/2411.16550) identifies limited encoder
//! capacity as a trigger, and the FSQ paper measures VQ using under half its
//! codebook above `2^11` entries while FSQ stays near 100% by construction. A
//! 3-20M model cannot afford to spend its capacity keeping a codebook alive.
//!
//! The one concession the FSQ paper makes is that VQ is *marginally* better
//! below roughly `2^10`-`2^11` codes. [`Fsq::default_levels`] sits above that
//! crossover deliberately.

use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};

/// Squashing margin. `tanh` only reaches its asymptote at infinity, so without
/// a margin the two outermost levels would be reachable only by unbounded
/// pre-activations and would sit unused -- the codebook-usage problem FSQ
/// exists to avoid, reintroduced by arithmetic. Shrinking the range by 0.1%
/// puts the extremes a finite distance away. This is the FSQ paper's `eps`.
const BOUND_EPS: f64 = 1e-3;

/// A finite scalar quantizer over `levels.len()` dimensions.
///
/// Holds no parameters and no device state, so it is a plain value rather than
/// a [`Module`](burn::module::Module): it never appears in a checkpoint, and a
/// run that changes `levels` changes the *rate*, which must not be silently
/// restorable from an old record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fsq {
    levels: Vec<u32>,
}

impl Fsq {
    /// The default level vector: `[7, 5, 5, 5, 5]`, i.e. 5 dimensions and
    /// 4,375 implicit codes -- 12.0948 bits per slot.
    ///
    /// Table 1 of the FSQ paper, their `2^12` row. Chosen over the smaller rows
    /// because it clears the `~2^10`-`2^11` band where the paper concedes VQ is
    /// marginally better, and over the larger ones for a reason specific to
    /// this project: at vocabulary 8192 a raw token id is `log2(8192) = 13`
    /// bits, so a wider grid would make one slot able to hold *more* than one
    /// token. A bottleneck whose every slot can carry a whole token is not a
    /// bottleneck; it just moves the representation sideways. The next row up,
    /// `[8, 8, 8, 6, 5]`, is 15,360 codes and 13.9 bits -- over that line.
    ///
    /// At 12.0948 bits the slot sits just under one token
    /// ([`Self::token_capacity`] = 0.930), so every bit of compression has to
    /// come from there being fewer slots than tokens, and none of it from
    /// arithmetic sleight of hand. That is the honest framing, and it is
    /// visible in the numbers rather than asserted in a README.
    pub fn default_levels() -> Self {
        Self::new(vec![7, 5, 5, 5, 5])
    }

    /// # Panics
    ///
    /// If `levels` is empty or any level is below 2. A level of 1 is a constant
    /// dimension: it carries zero bits while still costing a projection column,
    /// which is always a configuration mistake rather than an intent.
    pub fn new(levels: Vec<u32>) -> Self {
        match Self::try_new(levels) {
            Ok(fsq) => fsq,
            Err(errs) => panic!("invalid Fsq:\n  - {}", errs.join("\n  - ")),
        }
    }

    /// The fallible form, for callers validating a whole configuration and
    /// wanting to report every problem at once rather than the first one.
    pub fn try_new(levels: Vec<u32>) -> Result<Self, Vec<String>> {
        let fsq = Self { levels };
        fsq.validate().map(|()| fsq)
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        if self.levels.is_empty() {
            errs.push("levels must not be empty".to_string());
        }
        for (i, &l) in self.levels.iter().enumerate() {
            if l < 2 {
                errs.push(format!(
                    "levels[{i}] = {l}: a dimension needs at least 2 levels to carry any bits"
                ));
            }
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    pub fn levels(&self) -> &[u32] {
        &self.levels
    }

    /// Width of the quantized latent: one scalar per level entry.
    pub fn dim(&self) -> usize {
        self.levels.len()
    }

    /// Size of the implicit codebook, `prod(L_i)`.
    pub fn codebook_size(&self) -> u64 {
        self.levels.iter().map(|&l| l as u64).product()
    }

    /// Bits carried by one slot: `sum(log2 L_i)`.
    ///
    /// Fractional, and correctly so -- a 5-level dimension is 2.32 bits, and
    /// rounding that up to 3 would overstate the cost of every slot. An actual
    /// serializer reaches this bound with arithmetic coding over the mixed
    /// radix; see [`Fsq::pack_bits`] for the whole-sequence figure.
    pub fn bits_per_slot(&self) -> f64 {
        self.levels.iter().map(|&l| (l as f64).log2()).sum()
    }

    /// Bits to store `n_slots` slots, as an integer count of a real payload.
    ///
    /// The mixed-radix integer of `n_slots` slots has `codebook_size^n_slots`
    /// possible values, so it needs `ceil(n_slots * bits_per_slot)` bits. This
    /// is the numerator of every compression ratio this crate reports; the
    /// denominator comes from [`ShardMeta`](crate::data::shard::ShardMeta),
    /// which already records `n_bytes` for exactly this purpose.
    pub fn pack_bits(&self, n_slots: usize) -> usize {
        (n_slots as f64 * self.bits_per_slot()).ceil() as usize
    }

    /// The most tokens one slot could carry *even in principle*, at a given
    /// vocabulary: `bits_per_slot / log2(vocab_size)`.
    ///
    /// This is the capacity bound of [Kuratov et al.,
    /// 2502.13063](https://arxiv.org/abs/2502.13063) restated for a discrete
    /// bottleneck, where it is exact rather than an estimate of what a float
    /// vector might hold. A compression ratio above this number is not
    /// ambitious, it is impossible: it asks a channel to carry more distinct
    /// messages than it has states, so reconstruction *must* fail on almost
    /// every input regardless of architecture, data, or training budget.
    ///
    /// [`CompressConfig::validate`](crate::compress::config::CompressConfig::validate)
    /// refuses to build a model that crosses it, which is the cheapest
    /// impossible experiment this crate can decline to run.
    pub fn token_capacity(&self, vocab_size: usize) -> f64 {
        self.bits_per_slot() / (vocab_size as f64).log2()
    }

    /// Per-dimension `(half_l, offset, shift)`.
    ///
    /// `half_l` scales `tanh`'s `(-1, 1)` onto a window `L` wide; `offset`
    /// half-shifts even level counts so the grid straddles zero symmetrically
    /// (`L = 4` gives `{-2, -1, 0, 1}`, `L = 3` gives `{-1, 0, 1}`); `shift`
    /// pre-compensates so that a zero pre-activation still lands on zero, which
    /// matters because a freshly initialized projection outputs approximately
    /// zero and should therefore start at the middle of the grid rather than
    /// half a level off it.
    ///
    /// `shift = tan(offset / half_l)`, as the published reference implementation
    /// writes it. Solving `tanh(shift) * half_l - offset = 0` exactly would give
    /// `atanh` instead, and for the level counts in Table 1 the two agree to
    /// four decimals (`atanh(1/3) = 0.34657` vs `tan(1/3) = 0.34625`) -- but
    /// `atanh` is only defined below 1, and `offset / half_l` reaches
    /// `0.5 / 0.4995 > 1` at `L = 2`. The tidier-looking form silently produces
    /// `NaN` for the smallest legal level count, and a `NaN` here propagates
    /// through the whole latent. `tan` is total on this input, which is
    /// presumably why the reference chose it; the exact centering was never the
    /// point, since the result is rounded to the grid regardless.
    fn constants(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut half_l = Vec::with_capacity(self.dim());
        let mut offset = Vec::with_capacity(self.dim());
        let mut shift = Vec::with_capacity(self.dim());
        for &l in &self.levels {
            let h = (l as f64 - 1.0) * (1.0 - BOUND_EPS) / 2.0;
            let o = if l % 2 == 0 { 0.5 } else { 0.0 };
            let x = o / h;
            half_l.push(h as f32);
            offset.push(o as f32);
            shift.push(x.tan() as f32);
        }
        (half_l, offset, shift)
    }

    /// `[batch, slots, dim]` pre-activations -> quantized codes in `[-1, 1]`.
    ///
    /// The straight-through estimator: the forward value is rounded, the
    /// backward gradient is the identity, because `round` has a derivative of
    /// zero almost everywhere and would otherwise stop training dead at the
    /// bottleneck. `detach` on the correction term is what splits the two.
    ///
    /// The result is divided by the half-width so every dimension lands in
    /// `[-1, 1]` whatever its level count. Without that, an `L = 8` dimension
    /// would arrive at the decoder roughly three times the scale of an `L = 3`
    /// one and the decoder's input projection would have to spend capacity
    /// undoing a choice of level vector.
    pub fn quantize<B: Backend>(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, d] = z.dims();
        assert_eq!(
            d,
            self.dim(),
            "Fsq has {} levels but was handed width {d}",
            self.dim()
        );

        let device = z.device();
        let bounded = self.bound(z);
        // Forward: round(bounded). Backward: d/d bounded = 1.
        let quantized = bounded.clone() + (bounded.clone().round() - bounded).detach();
        quantized / self.half_width_tensor::<B>(&device)
    }

    /// The squashing step alone, exposed for tests that need to check the grid
    /// bounds without the rounding on top.
    fn bound<B: Backend>(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = z.device();
        let (half_l, offset, shift) = self.constants();
        let row = |v: Vec<f32>| {
            Tensor::<B, 1>::from_floats(v.as_slice(), &device).reshape([1, 1, self.dim()])
        };
        (z + row(shift)).tanh() * row(half_l) - row(offset)
    }

    /// Half-width per dimension, `L / 2` rounded **down**: the normalizer that
    /// maps the integer grid onto `[-1, 1]`.
    ///
    /// Integer division, and not `(L - 1) / 2`, which is the tempting reading
    /// and is wrong for even `L`. An even dimension's grid is asymmetric --
    /// `L = 4` gives `{-2, -1, 0, 1}`, which reaches 2 below zero but only 1
    /// above -- so the normalizer has to be the larger half, `2`, or the
    /// bottom of the grid lands outside `[-1, 1]`. At `L = 2` the difference is
    /// not cosmetic: `(L - 1) / 2 = 0.5` maps `{-1, 0}` to `{-2, 0}`, putting
    /// the latent at double the range the decoder's input projection is
    /// initialized for. This matches the reference implementation's
    /// `half_width = levels // 2`.
    fn half_width_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let hw: Vec<f32> = self.levels.iter().map(|&l| (l / 2) as f32).collect();
        Tensor::<B, 1>::from_floats(hw.as_slice(), device).reshape([1, 1, self.dim()])
    }

    /// Quantized codes -> per-dimension integer indices in `0..L_i`.
    ///
    /// The serializable form of the latent, and the only form in which a
    /// compression ratio can be quoted. Inverse of the normalization in
    /// [`Fsq::quantize`]: undo the half-width, then shift the symmetric grid
    /// (`{-2, -1, 0, 1}`) onto the natural numbers (`{0, 1, 2, 3}`).
    pub fn code_indices<B: Backend>(&self, zq: Tensor<B, 3>) -> Tensor<B, 3, Int> {
        let device = zq.device();
        let half_width = self.half_width_tensor::<B>(&device);
        (zq * half_width.clone() + half_width).round().int()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::{Device, Distribution};

    fn device() -> Device<TestBackend> {
        Default::default()
    }

    /// A wide sweep of pre-activations, far past where `tanh` saturates, so
    /// every level is reachable and none of the checks below can pass by
    /// exercising only the middle of the grid.
    ///
    /// A deterministic ramp, not random samples. The narrowest rounding bin of
    /// an even-level dimension covers barely 1% of this range, so a random
    /// sweep misses a level every few runs -- and a test that is flaky about
    /// the single property it exists to check is worse than no test.
    fn sweep(d: usize) -> Tensor<TestBackend, 3> {
        const N: usize = 2001;
        let ramp = Tensor::<TestBackend, 1, Int>::arange(0..N as i64, &device())
            .float()
            .div_scalar(N as f32 - 1.0)
            .mul_scalar(24.0)
            .sub_scalar(12.0);
        ramp.reshape([N, 1, 1]).expand([N, 1, d])
    }

    /// The rate claim, checked against the definition rather than restated.
    /// If this drifts, every compression ratio this crate prints is wrong.
    #[test]
    fn the_default_level_vector_carries_the_bits_it_advertises() {
        let fsq = Fsq::default_levels();
        assert_eq!(fsq.dim(), 5);
        assert_eq!(fsq.codebook_size(), 7 * 5 * 5 * 5 * 5);
        assert_eq!(fsq.codebook_size(), 4_375);
        assert!((fsq.bits_per_slot() - 4_375f64.log2()).abs() < 1e-12);
        assert!((fsq.bits_per_slot() - 12.0948).abs() < 1e-3);
        // 32 slots is a payload, not a hand-wave: 388 bits.
        assert_eq!(fsq.pack_bits(32), 388);
    }

    /// The capacity bound. At vocab 8192 one slot is worth slightly under one
    /// token, so any configuration compressing more than ~0.99 tokens per slot
    /// is over budget -- which is to say every interesting configuration is,
    /// and the bound is doing real work rather than passing vacuously.
    #[test]
    fn token_capacity_is_just_under_one_token_per_slot_at_vocab_8192() {
        let fsq = Fsq::default_levels();
        let cap = fsq.token_capacity(8192);
        assert!((cap - 12.0948 / 13.0).abs() < 1e-3, "capacity {cap}");
        assert!(cap < 1.0);
        // A larger vocabulary is a tighter bound: each token costs more bits.
        assert!(fsq.token_capacity(32_000) < cap);
    }

    /// Quantization must land on exactly `L_i` distinct values in dimension
    /// `i` -- not fewer (a dimension that cannot reach its extremes is paying
    /// for bits it does not deliver) and not more (a grid wider than declared
    /// makes `bits_per_slot` an undercount, i.e. an overstated ratio).
    #[test]
    fn each_dimension_uses_exactly_its_declared_number_of_levels() {
        for levels in [vec![8, 8, 8, 6, 5], vec![2, 3], vec![4], vec![5, 5, 5, 5]] {
            let fsq = Fsq::new(levels.clone());
            let codes = fsq.code_indices(fsq.quantize(sweep(fsq.dim())));
            let flat: Vec<i64> = codes.into_data().to_vec().unwrap();
            let d = fsq.dim();
            for (i, &l) in levels.iter().enumerate() {
                let seen: std::collections::BTreeSet<i64> =
                    flat.iter().skip(i).step_by(d).copied().collect();
                assert_eq!(
                    seen,
                    (0..l as i64).collect(),
                    "levels {levels:?}, dim {i}: wrong set of codes"
                );
            }
        }
    }

    /// A binary dimension must produce numbers, not `NaN`.
    ///
    /// Regression test. `L = 2` is the smallest level count `validate` accepts,
    /// and it is the one where `offset / half_l` exceeds 1 -- the input on
    /// which the exact `atanh` centering is undefined. A single `NaN` in the
    /// shift constant would silently poison every latent that dimension
    /// touches, and it would surface as a training run that never converges
    /// rather than as an error.
    #[test]
    fn a_binary_dimension_is_finite() {
        let fsq = Fsq::new(vec![2, 2, 2]);
        let z = sweep(fsq.dim());
        let out: Vec<f32> = fsq.quantize(z).into_data().to_vec().unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "quantizer produced NaN");
        // Both of a binary dimension's two states are reached, and both are
        // inside the normalized range the decoder expects.
        assert!(
            out.iter().all(|&v| (-1.0..=1.0).contains(&v)),
            "outside [-1, 1]"
        );
        let seen: std::collections::BTreeSet<i64> = fsq
            .code_indices(fsq.quantize(sweep(fsq.dim())))
            .into_data()
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(seen, (0..2).collect());
    }

    /// The straight-through estimator, stated as the two halves it is made of.
    ///
    /// Forward: the output is on the grid, so the bottleneck is real and the
    /// decoder never sees an unquantized value. Backward: the gradient is not
    /// annihilated, which is the whole reason for the `detach` trick -- a
    /// plain `round` would hand every parameter below the bottleneck a zero
    /// gradient and the encoder would never train at all.
    #[test]
    fn quantization_is_exact_forward_and_transparent_backward() {
        type Ad = burn::backend::Autodiff<TestBackend>;

        let fsq = Fsq::default_levels();
        let d = fsq.dim();

        // Forward: on-grid. Undo the normalization and every value is an
        // integer to within float error.
        let zq = fsq.quantize(sweep(d));
        let raw: Vec<f32> = (zq.clone() * fsq.half_width_tensor::<TestBackend>(&device()))
            .into_data()
            .to_vec()
            .unwrap();
        for v in &raw {
            assert!(
                (v - v.round()).abs() < 1e-4,
                "{v} is not on the quantization grid"
            );
        }

        // Backward: a gradient reaches the pre-activation.
        let z = Tensor::<Ad, 3>::random([2, 8, d], Distribution::Uniform(-2.0, 2.0), &device())
            .require_grad();
        let loss = fsq.quantize(z.clone()).sum();
        let grads = loss.backward();
        let g = z
            .grad(&grads)
            .expect("no gradient reached the encoder side");
        let total: f32 = g.abs().sum().into_scalar();
        assert!(total > 0.0, "the straight-through path is broken");
    }

    /// A zero pre-activation sits on a grid point, not between two.
    ///
    /// This is what `shift` buys, and it matters at exactly one moment: step
    /// zero, when the encoder's projection outputs near-zero everywhere. Half
    /// a level off, and half the dimensions would round arbitrarily on the
    /// first batches.
    ///
    /// The tolerance is a tenth of a level rather than machine epsilon,
    /// because `tan` centres approximately -- see [`Fsq::constants`] for why
    /// the exact form is not usable. What has to hold is that zero rounds to
    /// the middle of the grid unambiguously, not that it hits it to the last
    /// bit.
    #[test]
    fn a_zero_pre_activation_lands_on_a_grid_point() {
        for levels in [vec![7, 5, 5, 5, 5], vec![2, 3, 4, 8]] {
            let fsq = Fsq::new(levels.clone());
            let z = Tensor::<TestBackend, 3>::zeros([1, 1, fsq.dim()], &device());
            let bounded: Vec<f32> = fsq.bound(z).into_data().to_vec().unwrap();
            for (i, v) in bounded.iter().enumerate() {
                assert!(
                    (v - v.round()).abs() < 0.1,
                    "levels {levels:?}, dim {i}: bound(0) = {v} is between grid points"
                );
            }
        }
    }

    /// Saturating inputs must stay inside the grid. `tanh` cannot exceed 1, so
    /// this is really a check that `BOUND_EPS` and the offset compose without
    /// pushing the extreme level out past its own rounding bin.
    #[test]
    fn extreme_inputs_stay_within_the_declared_grid() {
        let fsq = Fsq::new(vec![8, 8, 8, 6, 5]);
        let d = fsq.dim();
        let big = Tensor::<TestBackend, 3>::full([2, 4, d], 1e4, &device());
        for z in [big.clone(), big.neg()] {
            let codes: Vec<i64> = fsq
                .code_indices(fsq.quantize(z))
                .into_data()
                .to_vec()
                .unwrap();
            for (i, &c) in codes.iter().enumerate() {
                let l = fsq.levels()[i % d] as i64;
                assert!((0..l).contains(&c), "code {c} out of range for L = {l}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "at least 2 levels")]
    fn a_one_level_dimension_is_rejected() {
        Fsq::new(vec![8, 1, 5]);
    }
}
