//! The Quark causal language model.
//!
//! Three parameter-saving mechanisms compose here, and each has a cost that is
//! worth naming rather than burying:
//!
//! 1. **Factorized embeddings** (`V x E`, then `E -> H`). Cuts the embedding
//!    table by `H/E`. Cost: with tying, it caps the output distribution's rank
//!    at `E` (the softmax bottleneck).
//! 2. **Tied embeddings**. Reuses the `V x E` table as the output projection.
//!    Cost: input and output token spaces are forced to coincide.
//! 3. **Cross-layer sharing**. One parameter set applied `n_loops` times.
//!    Cost: no compute saving and no activation-memory saving -- only storage.
//!    A 12-loop model costs 12 layers to train.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use crate::{
    config::{LayerSchedule, ModelConfig},
    model::{attention::KvCache, block::Block},
};

#[derive(Module, Debug)]
pub struct QuarkLm<B: Backend> {
    token_embedding: Embedding<B>,
    embed_proj: Linear<B>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    /// Present iff `tie_embeddings`: projects `H -> E` so the tied `V x E`
    /// table can be applied as the output matrix.
    unembed_proj: Option<Linear<B>>,
    /// Present iff `!tie_embeddings`: a full `H -> V` head.
    lm_head: Option<Linear<B>>,
    config: ModelConfig,
}

impl<B: Backend> QuarkLm<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Self {
        if let Err(errs) = config.validate() {
            panic!("invalid ModelConfig:\n  - {}", errs.join("\n  - "));
        }

        let blocks = (0..config.n_unique_layers)
            .map(|_| Block::new(&config, device))
            .collect();

        let (unembed_proj, lm_head) = if config.tie_embeddings {
            (
                Some(
                    LinearConfig::new(config.d_model, config.d_emb)
                        .with_bias(false)
                        .init(device),
                ),
                None,
            )
        } else {
            (
                None,
                Some(
                    LinearConfig::new(config.d_model, config.vocab_size)
                        .with_bias(false)
                        .init(device),
                ),
            )
        };

        Self {
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.d_emb).init(device),
            embed_proj: LinearConfig::new(config.d_emb, config.d_model)
                .with_bias(false)
                .init(device),
            blocks,
            final_norm: RmsNormConfig::new(config.d_model)
                .with_epsilon(config.norm_eps)
                .init(device),
            unembed_proj,
            lm_head,
            config,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Index of the unique layer used at application `i`.
    fn layer_at(&self, i: usize) -> usize {
        match self.config.layer_schedule {
            LayerSchedule::Cycle => i % self.config.n_unique_layers,
            LayerSchedule::Blocked => i / self.config.n_loops,
        }
    }

    /// `[batch, seq]` token ids -> `[batch, seq, vocab]` logits.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut h = self
            .embed_proj
            .forward(self.token_embedding.forward(tokens));
        for i in 0..self.config.n_layer_applications() {
            h = self.blocks[self.layer_at(i)].forward(h);
        }
        self.logits(self.final_norm.forward(h))
    }

    /// Incremental forward for generation. One cache per *layer application*,
    /// not per unique layer: a shared layer sees different activations on each
    /// loop, so their keys and values are different and cannot be pooled.
    pub fn forward_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        caches: &mut [KvCache<B>],
    ) -> Tensor<B, 3> {
        assert_eq!(
            caches.len(),
            self.config.n_layer_applications(),
            "need one KV cache per layer application"
        );
        let mut h = self
            .embed_proj
            .forward(self.token_embedding.forward(tokens));
        for (i, cache) in caches.iter_mut().enumerate() {
            h = self.blocks[self.layer_at(i)].forward_cached(h, cache);
        }
        self.logits(self.final_norm.forward(h))
    }

    pub fn new_caches(&self) -> Vec<KvCache<B>> {
        (0..self.config.n_layer_applications())
            .map(|_| KvCache::new())
            .collect()
    }

    fn logits(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        match (&self.unembed_proj, &self.lm_head) {
            (Some(unembed), None) => {
                // Tied: project H -> E, then multiply by the transposed
                // embedding table. Autodiff routes gradients from both the
                // input and output paths into the same table, which is the
                // point of tying.
                let h = unembed.forward(h); // [B, T, E]
                let [batch, seq, d_emb] = h.dims();
                let w = self.token_embedding.weight.val(); // [V, E]
                h.reshape([batch * seq, d_emb])
                    .matmul(w.transpose()) // [B*T, V]
                    .reshape([batch, seq, self.config.vocab_size])
            }
            (None, Some(head)) => head.forward(h),
            _ => unreachable!("exactly one of unembed_proj / lm_head is constructed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, TestBackend};
    use burn::tensor::{Distribution, TensorData};

    fn ids<B: Backend>(dims: [usize; 2], vocab: usize, device: &B::Device) -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::random(dims, Distribution::Uniform(0.0, vocab as f64 - 1.0), device)
    }

    #[test]
    fn forward_produces_vocab_logits() {
        let d = Default::default();
        let cfg = ModelConfig::tiny();
        let m = QuarkLm::<TestBackend>::new(cfg.clone(), &d);
        let t = ids::<TestBackend>([2, 8], cfg.vocab_size, &d);
        assert_eq!(m.forward(t).dims(), [2, 8, cfg.vocab_size]);
    }

    /// The budget in `config.rs` is what we plan against and what we report.
    /// This is the test that keeps it honest against the real module.
    #[test]
    fn analytic_budget_matches_the_real_module() {
        let d = Default::default();
        for cfg in [
            ModelConfig::quark_3m(),
            ModelConfig::quark_3m_deep(),
            ModelConfig::quark_3m_dense(),
            ModelConfig::tiny(),
            ModelConfig {
                tie_embeddings: false,
                ..ModelConfig::tiny()
            },
        ] {
            let m = QuarkLm::<TestBackend>::new(cfg.clone(), &d);
            assert_eq!(
                m.num_params(),
                cfg.param_count(),
                "analytic budget disagrees with the module for {cfg:?}"
            );
        }
    }

    #[test]
    fn reference_model_is_under_three_million_params() {
        let d = Default::default();
        let m = QuarkLm::<TestBackend>::new(ModelConfig::quark_3m(), &d);
        assert!(m.num_params() <= 3_000_000, "got {}", m.num_params());
    }

    /// Sharing must actually share: a 12-loop model has to hold the same
    /// parameters as a 1-loop model of the same shape.
    #[test]
    fn looping_a_shared_layer_adds_no_parameters() {
        let d = Default::default();
        let one = ModelConfig {
            n_loops: 1,
            ..ModelConfig::tiny()
        };
        let twelve = ModelConfig {
            n_loops: 12,
            ..ModelConfig::tiny()
        };
        let a = QuarkLm::<TestBackend>::new(one, &d);
        let b = QuarkLm::<TestBackend>::new(twelve, &d);
        assert_eq!(a.num_params(), b.num_params());
    }

    #[test]
    fn layer_schedules_visit_the_expected_layers() {
        let d = Default::default();
        let cycle = QuarkLm::<TestBackend>::new(
            ModelConfig {
                n_unique_layers: 2,
                n_loops: 3,
                layer_schedule: LayerSchedule::Cycle,
                ..ModelConfig::tiny()
            },
            &d,
        );
        let visited: Vec<_> = (0..6).map(|i| cycle.layer_at(i)).collect();
        assert_eq!(visited, vec![0, 1, 0, 1, 0, 1]);

        let blocked = QuarkLm::<TestBackend>::new(
            ModelConfig {
                n_unique_layers: 2,
                n_loops: 3,
                layer_schedule: LayerSchedule::Blocked,
                ..ModelConfig::tiny()
            },
            &d,
        );
        let visited: Vec<_> = (0..6).map(|i| blocked.layer_at(i)).collect();
        assert_eq!(visited, vec![0, 0, 0, 1, 1, 1]);
    }

    /// End-to-end causality at the model level, not just inside attention: the
    /// embedding, loop and head must not smuggle in future information either.
    #[test]
    fn model_is_causal() {
        let d = Default::default();
        let cfg = ModelConfig::tiny();
        let m = QuarkLm::<TestBackend>::new(cfg.clone(), &d);
        let seq = 6;

        let a = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [1usize, seq]);
        let b = TensorData::new(vec![1i32, 2, 3, 4, 5, 99], [1usize, seq]);
        let la = m.forward(Tensor::<TestBackend, 2, Int>::from_data(a, &d));
        let lb = m.forward(Tensor::<TestBackend, 2, Int>::from_data(b, &d));

        // Changing only the final token must leave every earlier logit intact.
        assert_close(
            la.slice([0..1, 0..seq - 1, 0..cfg.vocab_size]),
            lb.slice([0..1, 0..seq - 1, 0..cfg.vocab_size]),
            1e-4,
        );
    }

    #[test]
    fn cached_generation_matches_full_forward() {
        let d = Default::default();
        let cfg = ModelConfig::tiny();
        let m = QuarkLm::<TestBackend>::new(cfg.clone(), &d);
        let seq = 5;
        let t = ids::<TestBackend>([1, seq], cfg.vocab_size, &d);

        let full = m.forward(t.clone());

        let mut caches = m.new_caches();
        let mut steps = Vec::new();
        for i in 0..seq {
            steps.push(m.forward_cached(t.clone().slice([0..1, i..i + 1]), &mut caches));
        }
        assert_close(full, Tensor::cat(steps, 1), 1e-4);
    }

    #[test]
    fn untied_model_also_works() {
        let d = Default::default();
        let cfg = ModelConfig {
            tie_embeddings: false,
            ..ModelConfig::tiny()
        };
        let m = QuarkLm::<TestBackend>::new(cfg.clone(), &d);
        let t = ids::<TestBackend>([2, 4], cfg.vocab_size, &d);
        assert_eq!(m.forward(t).dims(), [2, 4, cfg.vocab_size]);
    }

    #[test]
    #[should_panic(expected = "invalid ModelConfig")]
    fn constructing_an_invalid_config_panics_loudly() {
        let d = Default::default();
        let _ = QuarkLm::<TestBackend>::new(
            ModelConfig {
                n_heads: 5,
                ..ModelConfig::quark_3m()
            },
            &d,
        );
    }
}
