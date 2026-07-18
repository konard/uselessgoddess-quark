//! The compressor: a bidirectional slot encoder, an FSQ bottleneck, and a
//! prefix-conditioned causal decoder.
//!
//! ```text
//!   x_1..x_N  ->  [ embed | slot queries ]  bidirectional stack  ->  last K
//!                                                                      |
//!                                                            to_latent + FSQ
//!                                                                      |
//!   x^_1..x^_N <-  causal stack  <-  [ from_latent(z) | bos, x_1..x_{N-1} ]
//! ```
//!
//! **Why memory slots and a prefix, rather than cross-attention.** This is what
//! ICAE ([2307.06945](https://arxiv.org/abs/2307.06945)), Gist Tokens
//! ([2304.08467](https://arxiv.org/abs/2304.08467)) and 500xCompressor
//! ([2408.03094](https://arxiv.org/abs/2408.03094)) all do, and it is also the
//! choice that keeps this feature out of the rest of the crate. A
//! cross-attending decoder would need a second attention module inside
//! [`Block`], which means a new field, which means every existing checkpoint
//! stops loading. Conditioning on a prefix instead needs *nothing* new: the
//! decoder is an ordinary causal stack that happens to begin with K positions
//! that are not tokens. The only change this whole feature makes to the shared
//! model code is [`Attend`], which selects a mask and adds no parameters.
//!
//! **What is deliberately not here.** No cross-attention, no learned positional
//! table (RoPE already extrapolates, and a table is what caps the reference
//! implementation in issue #12 at 128 tokens), no padding mask (spans are dense
//! fixed-length windows out of a token shard, so there is nothing to pad), and
//! no second embedding table (the encoder, the decoder and the output head all
//! read the same one).

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm,
        RmsNormConfig,
    },
    prelude::Backend,
    tensor::{Distribution, Int, Tensor},
};

use crate::{
    compress::{config::CompressConfig, quantize::Fsq},
    model::{attention::KvCache, block::Block, Attend},
};

/// Standard deviation for the token table and the slot queries.
///
/// GPT-2's constant, and here for the reason spelled out at length in
/// [`crate::model::lm`]: the table is also the output matrix, so a unit-variance
/// initialization would make the model's first job undoing its own start. The
/// slot queries get the same treatment because they enter the same residual
/// stream at the same scale.
const INIT_STD: f64 = 0.02;

#[derive(Module, Debug)]
pub struct Compressor<B: Backend> {
    /// Shared by the encoder, the decoder and the output head.
    token_embedding: Embedding<B>,
    encoder_embed_proj: Linear<B>,
    decoder_embed_proj: Linear<B>,
    /// `[K, d_model]`. Learned queries appended to the encoder's input; the
    /// encoder's output at those positions is the compressed representation.
    slot_queries: Param<Tensor<B, 2>>,
    encoder: Vec<Block<B>>,
    encoder_norm: RmsNorm<B>,
    to_latent: Linear<B>,
    from_latent: Linear<B>,
    decoder: Vec<Block<B>>,
    decoder_norm: RmsNorm<B>,
    unembed_proj: Linear<B>,
    latent_dropout: Dropout,
    config: CompressConfig,
    /// Micro-batches burn sums into one optimizer step. Skipped by the `Module`
    /// derive (no backend generic), exactly as on
    /// [`QuarkLm`](crate::model::QuarkLm), so it never reaches a checkpoint.
    grad_accumulation: usize,
}

impl<B: Backend> Compressor<B> {
    pub fn new(config: CompressConfig, device: &B::Device) -> Self {
        if let Err(errs) = config.validate() {
            panic!("invalid CompressConfig:\n  - {}", errs.join("\n  - "));
        }

        let m = &config.model;
        let n_layers = m.n_layer_applications();
        let d = config.fsq().dim();
        let linear = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);

        Self {
            token_embedding: EmbeddingConfig::new(m.vocab_size, m.d_emb)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: INIT_STD,
                })
                .init(device),
            encoder_embed_proj: linear(m.d_emb, m.d_model),
            decoder_embed_proj: linear(m.d_emb, m.d_model),
            slot_queries: Param::from_tensor(Tensor::random(
                [config.n_slots, m.d_model],
                Distribution::Normal(0.0, INIT_STD),
                device,
            )),
            encoder: (0..n_layers).map(|_| Block::new(m, device)).collect(),
            encoder_norm: RmsNormConfig::new(m.d_model)
                .with_epsilon(m.norm_eps)
                .init(device),
            to_latent: linear(m.d_model, d),
            from_latent: linear(d, m.d_model),
            decoder: (0..n_layers).map(|_| Block::new(m, device)).collect(),
            decoder_norm: RmsNormConfig::new(m.d_model)
                .with_epsilon(m.norm_eps)
                .init(device),
            unembed_proj: linear(m.d_model, m.d_emb),
            latent_dropout: DropoutConfig::new(config.latent_dropout).init(),
            config,
            grad_accumulation: 1,
        }
    }

    pub fn config(&self) -> &CompressConfig {
        &self.config
    }

    pub fn grad_accumulation(&self) -> usize {
        self.grad_accumulation
    }

    /// See [`QuarkLm::with_grad_accumulation`](crate::model::QuarkLm::with_grad_accumulation).
    pub fn with_grad_accumulation(mut self, n: usize) -> Self {
        assert!(n >= 1, "grad_accumulation must be at least 1, got {n}");
        self.grad_accumulation = n;
        self
    }

    /// `[batch, N]` tokens -> `[batch, K, d]` quantized latents in `[-1, 1]`.
    ///
    /// The slot queries are appended *after* the span rather than prepended,
    /// and read bidirectionally: a query at position `N + k` can see every
    /// token, and every token can see every other. Forbidding the encoder to
    /// look right would cost information for nothing -- it is summarizing a
    /// span it holds in full, not predicting the next word.
    pub fn encode(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let h = self.encode_hidden(tokens);
        self.config.fsq().quantize(self.to_latent.forward(h))
    }

    /// The encoder's output at the slot positions, before `to_latent` and
    /// before quantization. Split out so a test can watch a continuous signal:
    /// past the quantizer, a small perturbation is rounded away and "did this
    /// slot see that token" stops being answerable.
    fn encode_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, n] = tokens.dims();
        assert_eq!(
            n, self.config.span_len,
            "encode expects exactly span_len = {} tokens",
            self.config.span_len
        );
        let d_model = self.config.model.d_model;
        let k = self.config.n_slots;

        let x = self
            .encoder_embed_proj
            .forward(self.token_embedding.forward(tokens));
        let slots = self
            .slot_queries
            .val()
            .reshape([1, k, d_model])
            .expand([batch, k, d_model]);

        let mut h = Tensor::cat(vec![x, slots], 1);
        for block in &self.encoder {
            h = block.forward_as(h, Attend::Bidirectional);
        }
        let h = self.encoder_norm.forward(h);

        h.slice([0..batch, n..n + k, 0..d_model])
    }

    /// `[batch, K, d]` latents and `[batch, N]` decoder inputs -> `[batch, N,
    /// V]` logits.
    ///
    /// The latents occupy the first K positions of an otherwise ordinary causal
    /// stack, so token `i` attends over every latent and every earlier token,
    /// and the logits are read off the last N positions.
    pub fn decode(&self, zq: Tensor<B, 3>, inputs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, n] = inputs.dims();
        let k = self.config.n_slots;
        let d_model = self.config.model.d_model;

        let mem = self.from_latent.forward(self.latent_dropout.forward(zq));
        let x = self
            .decoder_embed_proj
            .forward(self.token_embedding.forward(inputs));

        let mut h = Tensor::cat(vec![mem, x], 1);
        for block in &self.decoder {
            h = block.forward(h);
        }
        let h = h.slice([0..batch, k..k + n, 0..d_model]);
        self.logits(self.decoder_norm.forward(h))
    }

    /// Teacher-forced reconstruction: `[batch, N]` tokens -> `[batch, N, V]`
    /// logits predicting the *same* tokens.
    ///
    /// The decoder input is the span shifted right by one and started with
    /// `bos_id`, then corrupted by [`Self::corrupt`]. Position `i` therefore
    /// predicts `x_i` from the latents and `x_{<i}` -- which is why the
    /// corruption matters: without it the prefix alone answers most of the
    /// question and the bottleneck can stay empty.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let zq = self.encode(tokens.clone());
        self.decode(zq, self.corrupt(self.shift_right(tokens)))
    }

    /// `[x_1, .., x_N]` -> `[bos, x_1, .., x_{N-1}]`.
    pub fn shift_right(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let [batch, n] = tokens.dims();
        let bos =
            Tensor::<B, 2, Int>::full([batch, 1], self.config.bos_id as i64, &tokens.device());
        Tensor::cat(vec![bos, tokens.slice([0..batch, 0..n - 1])], 1)
    }

    /// Replace a `token_dropout` fraction of decoder inputs with `bos_id`.
    ///
    /// Note *replace*, not zero, and no `1/(1-p)` rescaling: this is input
    /// corruption in the sense of DAAE
    /// ([1905.12777](https://arxiv.org/abs/1905.12777)) and CALM's CBOW-style
    /// token dropout, not the variance-preserving activation dropout that
    /// [`Dropout`] implements. Rescaling token *ids* would be meaningless.
    ///
    /// Gated on `ad_enabled` for the same reason burn's own dropout is: an
    /// evaluation pass must be deterministic, or the reconstruction metric
    /// would measure the noise as much as the model.
    pub fn corrupt(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let device = tokens.device();
        if self.config.token_dropout == 0.0 || !B::ad_enabled(&device) {
            return tokens;
        }
        let keep = Tensor::<B, 2>::random(
            tokens.dims(),
            Distribution::Bernoulli(1.0 - self.config.token_dropout),
            &device,
        )
        .int();
        let drop = keep.clone().neg() + 1;
        tokens * keep + drop * self.config.bos_id as i64
    }

    /// Free-running greedy reconstruction: `[batch, N]` tokens -> `[batch, N]`
    /// tokens, decoded from the latent alone.
    ///
    /// **This, not the teacher-forced loss, is the headline number.** A
    /// teacher-forced decoder is handed the correct prefix at every step, so it
    /// reports what the model would do if it never made a mistake -- an upper
    /// bound, and one that hides exposure bias (Ranzato et al.,
    /// [1511.06732](https://arxiv.org/abs/1511.06732)). Free-running decoding
    /// feeds the model its own output, which is the only setting in which
    /// "the latent reconstructs the text" is a claim about the latent.
    ///
    /// Uses the ordinary [`KvCache`] path, so it inherits the equivalence that
    /// `cached_decoding_matches_full_forward` already pins in
    /// [`crate::model::attention`].
    pub fn reconstruct(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let n = tokens.dims()[1];
        self.decode_greedy(self.encode(tokens), n)
    }

    /// Greedy decoding of `len` tokens from `[batch, K, d]` latents.
    pub fn decode_greedy(&self, zq: Tensor<B, 3>, len: usize) -> Tensor<B, 2, Int> {
        let device = zq.device();
        let batch = zq.dims()[0];
        let vocab = self.config.model.vocab_size;

        let mut caches: Vec<KvCache<B>> = self.decoder.iter().map(|_| KvCache::new()).collect();
        let mem = self.from_latent.forward(self.latent_dropout.forward(zq));
        let bos = Tensor::<B, 2, Int>::full([batch, 1], self.config.bos_id as i64, &device);

        // Prefill: the K latent positions plus the start token, in one pass.
        let mut step = Tensor::cat(
            vec![
                mem,
                self.decoder_embed_proj
                    .forward(self.token_embedding.forward(bos)),
            ],
            1,
        );

        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let mut h = step;
            for (block, cache) in self.decoder.iter().zip(caches.iter_mut()) {
                h = block.forward_cached(h, cache);
            }
            let t = h.dims()[1] - 1;
            let last = h.slice([0..batch, t..t + 1, 0..self.config.model.d_model]);
            let logits = self.logits(self.decoder_norm.forward(last));
            let next = logits.argmax(2).reshape([batch, 1]);
            out.push(next.clone());
            step = self
                .decoder_embed_proj
                .forward(self.token_embedding.forward(next));
        }
        let _ = vocab;
        Tensor::cat(out, 1)
    }

    /// Tied output head, identical in form to
    /// [`QuarkLm`](crate::model::QuarkLm)'s: project `H -> E`, then multiply by
    /// the transposed embedding table.
    ///
    /// Tied because the encoder, the decoder and the head are all talking about
    /// the same 8192 tokens. The reference implementation in issue #12 gives
    /// the encoder and decoder separate tables and an untied head -- three
    /// copies of the same knowledge, which at `vocab x hidden` is a large
    /// fraction of a 13-17M budget spent on redundancy.
    fn logits(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.unembed_proj.forward(h);
        let [batch, seq, d_emb] = h.dims();
        let w = self.token_embedding.weight.val();
        h.reshape([batch * seq, d_emb])
            .matmul(w.transpose())
            .reshape([batch, seq, self.config.model.vocab_size])
    }

    pub fn fsq(&self) -> Fsq {
        self.config.fsq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::Device;

    fn device() -> Device<TestBackend> {
        Default::default()
    }

    fn ids(cfg: &CompressConfig, batch: usize) -> Tensor<TestBackend, 2, Int> {
        Tensor::<TestBackend, 2, Int>::random(
            [batch, cfg.span_len],
            Distribution::Uniform(0.0, cfg.model.vocab_size as f64 - 1.0),
            &device(),
        )
    }

    /// The analytic budget in `CompressConfig::budget` must equal the model
    /// that actually gets built. This is the check that lets a 15M-parameter
    /// configuration be sized on paper, which is what the issue means by
    /// verifying the code logically instead of by training it.
    #[test]
    fn parameter_count_matches_the_analytic_budget() {
        for cfg in [CompressConfig::tiny(), CompressConfig::compressor_15m()] {
            let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
            assert_eq!(
                model.num_params(),
                cfg.param_count(),
                "budget disagrees with the module for {} slots / {} span:\n{}",
                cfg.n_slots,
                cfg.span_len,
                cfg.budget_table()
            );
        }
    }

    #[test]
    fn shapes_line_up_end_to_end() {
        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let x = ids(&cfg, 3);

        let zq = model.encode(x.clone());
        assert_eq!(zq.dims(), [3, cfg.n_slots, cfg.fsq().dim()]);

        let logits = model.forward(x.clone());
        assert_eq!(logits.dims(), [3, cfg.span_len, cfg.model.vocab_size]);

        assert_eq!(model.reconstruct(x).dims(), [3, cfg.span_len]);
    }

    /// The latent is on the quantization grid, so what reaches the decoder is
    /// `n_slots * bits_per_slot` bits and nothing more. If this ever fails, the
    /// bottleneck has been bypassed and every rate figure in the docs is void.
    #[test]
    fn the_latent_is_discrete() {
        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let codes = cfg.fsq().code_indices(model.encode(ids(&cfg, 4)));
        let flat: Vec<i64> = codes.into_data().to_vec().unwrap();
        let d = cfg.fsq().dim();
        for (i, &c) in flat.iter().enumerate() {
            let l = cfg.fsq().levels()[i % d] as i64;
            assert!((0..l).contains(&c), "code {c} outside 0..{l}");
        }
    }

    /// The property the whole design rests on: **the decoder cannot see the
    /// token it is predicting.**
    ///
    /// If it could, the reconstruction loss would be trivially near zero and
    /// the entire experiment would be measuring nothing -- the single most
    /// expensive way for this feature to be silently wrong. So: change the
    /// decoder input at position `t`, and no logit *before* `t` may move.
    ///
    /// Before, not at: under the shift convention of [`Self::shift_right`],
    /// decoder input `t` holds `x_{t-1}` and output `t` predicts `x_t`, so
    /// output `t` is supposed to depend on input `t`. It is output `t` seeing
    /// input `t+1` -- that is, `x_t` itself -- that would be the leak. The
    /// encoder is bypassed with a fixed latent to isolate the decoder, since
    /// the encoder is bidirectional on purpose.
    #[test]
    fn the_decoder_cannot_read_ahead() {
        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let batch = 1;
        let n = cfg.span_len;
        let zq = Tensor::<TestBackend, 3>::zeros([batch, cfg.n_slots, cfg.fsq().dim()], &device());

        let inputs = ids(&cfg, batch);
        let base = model.decode(zq.clone(), inputs.clone());

        // Flip the token at position `t`.
        let t = n / 2;
        let bumped = inputs.clone().slice_assign(
            [0..batch, t..t + 1],
            inputs.clone().slice([0..batch, t..t + 1]) + 1,
        );
        let after = model.decode(zq, bumped);

        let v = cfg.model.vocab_size;
        let unchanged: f32 = (base.clone().slice([0..batch, 0..t, 0..v])
            - after.clone().slice([0..batch, 0..t, 0..v]))
        .abs()
        .sum()
        .into_scalar();
        assert!(unchanged < 1e-4, "positions 0..{t} saw a future token");

        let changed: f32 = (base.slice([0..batch, t..n, 0..v])
            - after.slice([0..batch, t..n, 0..v]))
        .abs()
        .sum()
        .into_scalar();
        assert!(changed > 1e-4, "the change never propagated forward at all");
    }

    /// The mirror property: the *encoder* must be bidirectional, so a token at
    /// the end of the span has to influence every slot. A causal encoder would
    /// leave the first slots summarizing a prefix, which is a strictly worse
    /// summary bought for no saving.
    #[test]
    fn every_slot_sees_the_whole_span() {
        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let batch = 1;
        let n = cfg.span_len;
        let x = ids(&cfg, batch);

        let base = model.to_latent.forward(model.encode_hidden(x.clone()));
        let bumped = x.clone().slice_assign(
            [0..batch, n - 1..n],
            x.clone().slice([0..batch, n - 1..n]) + 1,
        );
        let after = model.to_latent.forward(model.encode_hidden(bumped));

        let d = cfg.fsq().dim();
        for k in 0..cfg.n_slots {
            let delta: f32 = (base.clone().slice([0..batch, k..k + 1, 0..d])
                - after.clone().slice([0..batch, k..k + 1, 0..d]))
            .abs()
            .sum()
            .into_scalar();
            assert!(delta > 1e-6, "slot {k} did not see the last token");
        }
    }

    /// A freshly built model must start near a uniform distribution: an initial
    /// loss of about `ln(V)`. Anything much higher means the model begins by
    /// undoing its own initialization, which at this scale is a large fraction
    /// of the training budget spent on nothing.
    ///
    /// This is the closest thing to "training will work" that can be checked
    /// without training, and it is exactly the check the issue asks for.
    #[test]
    fn initial_loss_is_near_uniform() {
        use crate::train::output::masked_cross_entropy;

        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let x = ids(&cfg, 8);
        let mask = Tensor::<TestBackend, 2>::ones([8, cfg.span_len], &device());

        let out = masked_cross_entropy(model.forward(x.clone()), x, mask);
        let loss: f32 = out.loss.into_scalar();
        let uniform = (cfg.model.vocab_size as f32).ln();
        assert!(
            (loss - uniform).abs() < 0.15,
            "initial loss {loss} is far from ln(V) = {uniform}"
        );
    }

    /// Greedy decoding must agree with the teacher-forced forward pass when the
    /// teacher's tokens *are* the greedy ones. This is what makes the two
    /// reported accuracies comparable rather than two unrelated numbers, and it
    /// re-checks the KV-cache path through a second, independent stack.
    #[test]
    fn greedy_decoding_agrees_with_the_full_forward_pass() {
        let cfg = CompressConfig::tiny();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device());
        let batch = 2;
        let x = ids(&cfg, batch);
        let zq = model.encode(x);

        let greedy = model.decode_greedy(zq.clone(), cfg.span_len);
        // Feed the greedy output back in as the teacher, shifted.
        let logits = model.decode(zq, model.shift_right(greedy.clone()));
        let argmax = logits.argmax(2).reshape([batch, cfg.span_len]);

        let a: Vec<i64> = greedy.into_data().to_vec().unwrap();
        let b: Vec<i64> = argmax.into_data().to_vec().unwrap();
        assert_eq!(a, b);
    }
}
