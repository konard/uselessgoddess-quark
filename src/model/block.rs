//! A single transformer block: attention and FFN, each on a residual branch.

use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::Tensor,
};

use crate::{
    config::{ModelConfig, NormPlacement},
    model::{
        attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, KvCache},
        ffn::{SwiGluFeedForward, SwiGluFeedForwardConfig},
    },
};

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    attn: GroupedQueryAttention<B>,
    ffn: SwiGluFeedForward<B>,
    attn_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,
    placement: NormPlacement,
}

impl<B: Backend> Block<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self {
            attn: GroupedQueryAttentionConfig::new(
                cfg.d_model,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.max_seq_len,
            )
            .with_rope_theta(cfg.rope_theta)
            .with_dropout(cfg.dropout)
            .init(device),
            ffn: SwiGluFeedForwardConfig::new(cfg.d_model, cfg.d_ff)
                .with_dropout(cfg.dropout)
                .init(device),
            attn_norm: RmsNormConfig::new(cfg.d_model)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            ffn_norm: RmsNormConfig::new(cfg.d_model)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            placement: cfg.norm_placement,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_inner(x, None)
    }

    pub fn forward_cached(&self, x: Tensor<B, 3>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        self.forward_inner(x, Some(cache))
    }

    fn forward_inner(&self, x: Tensor<B, 3>, cache: Option<&mut KvCache<B>>) -> Tensor<B, 3> {
        let attn = |t: Tensor<B, 3>| match cache {
            Some(c) => self.attn.forward_cached(t, c),
            None => self.attn.forward(t),
        };
        match self.placement {
            NormPlacement::Pre => {
                let x = x.clone() + attn(self.attn_norm.forward(x));
                let h = self.ffn.forward(self.ffn_norm.forward(x.clone()));
                x + h
            }
            NormPlacement::Post => {
                let x = self.attn_norm.forward(x.clone() + attn(x));
                let h = self.ffn.forward(x.clone());
                self.ffn_norm.forward(x + h)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::Distribution;

    #[test]
    fn both_norm_placements_preserve_shape() {
        let d = Default::default();
        for placement in [NormPlacement::Pre, NormPlacement::Post] {
            let cfg = ModelConfig {
                norm_placement: placement,
                ..ModelConfig::tiny()
            };
            let block = Block::<TestBackend>::new(&cfg, &d);
            let x =
                Tensor::<TestBackend, 3>::random([2, 6, cfg.d_model], Distribution::Default, &d);
            assert_eq!(block.forward(x).dims(), [2, 6, cfg.d_model]);
        }
    }

    /// Pins the per-layer term of the analytic budget in `config.rs`.
    #[test]
    fn parameter_count_matches_analytic_budget() {
        let d = Default::default();
        let cfg = ModelConfig::quark_3m();
        let block = Block::<TestBackend>::new(&cfg, &d);
        let expected = cfg
            .budget()
            .iter()
            .find(|e| e.name == "layers")
            .unwrap()
            .params;
        assert_eq!(block.num_params(), expected / cfg.n_unique_layers);
    }
}
