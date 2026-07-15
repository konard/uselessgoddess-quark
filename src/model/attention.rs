//! Causal grouped-query attention with rotary position embeddings.
//!
//! This is hand-rolled rather than built on `burn::nn::attention::
//! MultiHeadAttention` because that module cannot express what we need:
//!
//!  * it has no grouped/multi-query support -- all of Q, K, V are
//!    `Linear(d_model, d_model)`, so K and V cost full width. At a 3.0M budget
//!    that is ~0.2M parameters we would rather spend on the FFN.
//!  * it offers no hook to apply RoPE, because the Q/K projections happen
//!    inside its `forward`.
//!  * its `MhaCache` requires re-feeding the whole sequence each decode step,
//!    and the incremental `TensorCache` API is `pub(crate)`.
//!  * it applies attention dropout to the *scores*, before the softmax, which
//!    is not the standard placement.
//!
//! Everything below uses only `Linear`, `RotaryEncoding` and tensor ops.

use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig},
    prelude::Backend,
    tensor::{activation::softmax, Tensor},
};

/// Cached keys and values for incremental decoding.
///
/// Holds `[batch, n_kv_heads, seq_so_far, d_head]` with RoPE already applied,
/// which is valid precisely because RoPE is applied at absolute positions: a
/// cached key's rotation never needs revisiting.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for KvCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KvCache<B> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Number of positions currently cached; also the absolute position of the
    /// next token, which is what RoPE needs as its offset.
    pub fn len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

#[derive(Config, Debug)]
pub struct GroupedQueryAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    /// `1` = multi-query, `n_heads` = full multi-head, else grouped-query.
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    rope: RotaryEncoding<B>,
    dropout: Dropout,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
}

impl GroupedQueryAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GroupedQueryAttention<B> {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model {} not divisible by n_heads {}",
            self.d_model,
            self.n_heads
        );
        assert!(
            self.n_kv_heads > 0 && self.n_heads % self.n_kv_heads == 0,
            "n_heads {} must be a positive multiple of n_kv_heads {}",
            self.n_heads,
            self.n_kv_heads
        );
        let d_head = self.d_model / self.n_heads;
        assert_eq!(d_head % 2, 0, "d_head {d_head} must be even for RoPE");

        let kv_width = self.n_kv_heads * d_head;
        // No biases: they cost parameters and buy nothing measurable in a
        // pre-norm transformer (LLaMA, PaLM and friends all drop them).
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);

        GroupedQueryAttention {
            wq: lin(self.d_model, self.n_heads * d_head),
            wk: lin(self.d_model, kv_width),
            wv: lin(self.d_model, kv_width),
            wo: lin(self.n_heads * d_head, self.d_model),
            rope: RotaryEncodingConfig::new(self.max_seq_len, d_head)
                .with_theta(self.rope_theta)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            d_head,
        }
    }
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Training forward: full causal self-attention over `[batch, seq,
    /// d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_inner(x, None, 0)
    }

    /// Incremental forward for decoding. `x` is usually a single token; the
    /// cache is extended in place.
    pub fn forward_cached(&self, x: Tensor<B, 3>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let pos = cache.len();
        self.forward_inner(x, Some(cache), pos)
    }

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        cache: Option<&mut KvCache<B>>,
        pos_offset: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();

        // Project, then split into heads: [B, T, H*Dh] -> [B, H, T, Dh].
        // RoPE wants the rotated axis last and the position axis second-to-last,
        // which this layout gives us for free.
        let to_heads = |t: Tensor<B, 3>, n: usize| -> Tensor<B, 4> {
            t.reshape([batch, seq, n, self.d_head]).swap_dims(1, 2)
        };
        let q = to_heads(self.wq.forward(x.clone()), self.n_heads);
        let k = to_heads(self.wk.forward(x.clone()), self.n_kv_heads);
        let v = to_heads(self.wv.forward(x), self.n_kv_heads);

        // Rotate at ABSOLUTE positions. During decoding `pos_offset` is the
        // cache length, so token N is rotated as position N whether it arrived
        // in a full prompt or one token at a time.
        let q = self.rope.apply(q, pos_offset);
        let k = self.rope.apply(k, pos_offset);

        // Extend the cache before expanding K/V to full head count: caching the
        // grouped (pre-expansion) tensors is what makes GQA save decode memory.
        let (k, v) = match cache {
            Some(cache) => {
                let k = match cache.k.take() {
                    Some(prev) => Tensor::cat(vec![prev, k], 2),
                    None => k,
                };
                let v = match cache.v.take() {
                    Some(prev) => Tensor::cat(vec![prev, v], 2),
                    None => v,
                };
                cache.k = Some(k.clone());
                cache.v = Some(v.clone());
                (k, v)
            }
            None => (k, v),
        };

        let kv_seq = k.dims()[2];
        let k = self.expand_kv(k);
        let v = self.expand_kv(v);

        // [B, H, T, S]
        let scores = q.matmul(k.swap_dims(2, 3)) / (self.d_head as f64).sqrt();

        // Causal mask. With a cache, the query at local index i sits at absolute
        // position pos_offset + i and may attend to keys 0..=pos_offset+i, so we
        // need `true` (fill with -inf) exactly where j > i + pos_offset. When
        // seq == 1 (single-step decode) this correctly masks nothing.
        //
        // `tril_mask`, NOT `triu_mask`: burn builds `matrix = i - j + offset`
        // and `tril_mask` returns `matrix < 0`, i.e. true where j > i + offset --
        // which is this condition exactly. `triu_mask` returns `matrix > 0`, true
        // where j < i + offset, which masks the PAST and leaves the future
        // visible; its last row is entirely true, so softmax over an all -inf row
        // yields NaN. burn's own `generate_autoregressive_mask` likewise uses
        // `tril_mask(.., 0)`.
        let mask = Tensor::<B, 2, burn::tensor::Bool>::tril_mask(
            [seq, kv_seq],
            pos_offset as i64,
            &scores.device(),
        )
        .unsqueeze::<4>()
        .expand([batch, self.n_heads, seq, kv_seq]);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = self.dropout.forward(softmax(scores, 3));

        // [B, H, T, Dh] -> [B, T, H*Dh]
        let ctx =
            weights
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq, self.n_heads * self.d_head]);
        self.wo.forward(ctx)
    }

    /// Broadcast `n_kv_heads` K/V heads up to `n_heads` query heads.
    ///
    /// Grouping is contiguous -- query head `i` reads KV head `i / n_rep`, so
    /// the expansion must produce `[kv0, kv0, kv0, kv1, kv1, kv1]`. A plain
    /// `repeat_dim` would tile instead (`[kv0, kv1, kv0, kv1, ...]`), which is
    /// a *different* head-to-group assignment. Both train fine from scratch
    /// since head order is arbitrary, but only this one matches the reference
    /// GQA semantics that external checkpoints assume.
    fn expand_kv(&self, kv: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            return kv;
        }
        let [batch, n_kv, seq, d_head] = kv.dims();
        kv.unsqueeze_dim::<5>(2)
            .expand([batch, n_kv, n_rep, seq, d_head])
            .reshape([batch, n_kv * n_rep, seq, d_head])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, TestBackend};
    use burn::tensor::{Device, Distribution, TensorData};

    fn device() -> Device<TestBackend> {
        Default::default()
    }

    fn cfg(n_heads: usize, n_kv_heads: usize) -> GroupedQueryAttentionConfig {
        GroupedQueryAttentionConfig::new(32, n_heads, n_kv_heads, 64)
    }

    #[test]
    fn output_shape_is_preserved() {
        let d = device();
        let attn = cfg(4, 2).init::<TestBackend>(&d);
        let x = Tensor::<TestBackend, 3>::random([2, 7, 32], Distribution::Default, &d);
        assert_eq!(attn.forward(x).dims(), [2, 7, 32]);
    }

    #[test]
    fn mqa_gqa_and_mha_all_run() {
        let d = device();
        for n_kv in [1, 2, 4] {
            let attn = cfg(4, n_kv).init::<TestBackend>(&d);
            let x = Tensor::<TestBackend, 3>::random([1, 5, 32], Distribution::Default, &d);
            assert_eq!(attn.forward(x).dims(), [1, 5, 32]);
        }
    }

    /// The property that makes a causal LM well-posed: output at position `t`
    /// must not depend on inputs after `t`. This is the single most valuable
    /// test in the file -- a mask off by one silently leaks the answer and
    /// shows up only as a suspiciously good loss curve.
    #[test]
    fn attention_is_causal() {
        let d = device();
        let attn = cfg(4, 2).init::<TestBackend>(&d);
        let seq = 6;

        let x = Tensor::<TestBackend, 3>::random([1, seq, 32], Distribution::Default, &d);
        let base = attn.forward(x.clone());

        // Perturb the LAST position only.
        let noise = Tensor::<TestBackend, 3>::random([1, 1, 32], Distribution::Default, &d) * 100.0;
        let perturbed = x.clone().slice_assign(
            [0..1, seq - 1..seq, 0..32],
            x.clone().slice([0..1, seq - 1..seq, 0..32]) + noise,
        );
        let after = attn.forward(perturbed);

        // Positions 0..seq-1 must be untouched.
        let a = base.clone().slice([0..1, 0..seq - 1, 0..32]);
        let b = after.clone().slice([0..1, 0..seq - 1, 0..32]);
        assert_close(a, b, 1e-4);

        // Sanity: the last position DID change, else the test proves nothing.
        let la = base.slice([0..1, seq - 1..seq, 0..32]);
        let lb = after.slice([0..1, seq - 1..seq, 0..32]);
        let delta: f32 = (la - lb).abs().sum().into_scalar();
        assert!(delta > 1e-3, "perturbation had no effect; test is vacuous");
    }

    /// Incremental decoding with a KV cache must reproduce the full forward
    /// pass exactly. If RoPE offsets or the cache-aware mask are wrong, this
    /// diverges -- and nothing else would catch it, since both paths look
    /// plausible in isolation.
    #[test]
    fn cached_decoding_matches_full_forward() {
        let d = device();
        let attn = cfg(4, 2).init::<TestBackend>(&d);
        let seq = 5;
        let x = Tensor::<TestBackend, 3>::random([1, seq, 32], Distribution::Default, &d);

        let full = attn.forward(x.clone());

        let mut cache = KvCache::new();
        let mut steps = Vec::new();
        for t in 0..seq {
            let step = x.clone().slice([0..1, t..t + 1, 0..32]);
            steps.push(attn.forward_cached(step, &mut cache));
        }
        let incremental = Tensor::cat(steps, 1);

        assert_eq!(cache.len(), seq);
        assert_close(full, incremental, 1e-4);
    }

    /// Prefill-then-decode: feed a prompt in one shot, then continue one token
    /// at a time. This is the path generation actually takes.
    #[test]
    fn prefill_then_step_matches_full_forward() {
        let d = device();
        let attn = cfg(4, 1).init::<TestBackend>(&d);
        let seq = 6;
        let prefill = 4;
        let x = Tensor::<TestBackend, 3>::random([1, seq, 32], Distribution::Default, &d);

        let full = attn.forward(x.clone());

        let mut cache = KvCache::new();
        let mut outs =
            vec![attn.forward_cached(x.clone().slice([0..1, 0..prefill, 0..32]), &mut cache)];
        for t in prefill..seq {
            outs.push(attn.forward_cached(x.clone().slice([0..1, t..t + 1, 0..32]), &mut cache));
        }
        assert_close(full, Tensor::cat(outs, 1), 1e-4);
    }

    /// GQA must map query heads onto KV groups contiguously. We verify it
    /// structurally: expanding `[[0],[1]]` with n_rep=2 has to give
    /// `[0, 0, 1, 1]`, not the `[0, 1, 0, 1]` that `repeat_dim` would produce.
    #[test]
    fn kv_expansion_groups_contiguously() {
        let d = device();
        let attn = cfg(4, 2).init::<TestBackend>(&d);
        // [batch=1, n_kv=2, seq=1, d_head=8], head 0 all 0.0, head 1 all 1.0.
        let kv = Tensor::<TestBackend, 4>::from_data(
            TensorData::new([[0.0f32; 8], [1.0f32; 8]].concat(), [1usize, 2, 1, 8]),
            &d,
        );
        let expanded = attn.expand_kv(kv);
        assert_eq!(expanded.dims(), [1, 4, 1, 8]);
        let got: Vec<f32> = expanded
            .slice([0..1, 0..4, 0..1, 0..1])
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(
            got,
            vec![0.0, 0.0, 1.0, 1.0],
            "GQA grouping must be contiguous"
        );
    }

    #[test]
    fn mqa_uses_fewer_parameters_than_mha() {
        let d = device();
        let mqa = cfg(4, 1).init::<TestBackend>(&d);
        let mha = cfg(4, 4).init::<TestBackend>(&d);
        assert!(
            mqa.num_params() < mha.num_params(),
            "MQA {} should be cheaper than MHA {}",
            mqa.num_params(),
            mha.num_params()
        );
    }
}
