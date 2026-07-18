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
        attention::{Attend, GroupedQueryAttention, GroupedQueryAttentionConfig, KvCache},
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
            .with_qk_norm(cfg.qk_norm)
            .with_norm_eps(cfg.norm_eps)
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
        self.forward_inner(x, None, Attend::Causal)
    }

    /// The block under an explicit [`Attend`] mode.
    ///
    /// Only the attention mask changes; norms, FFN, residuals and parameter
    /// count are identical. That is what lets an encoder and a decoder be the
    /// same `Block` type built from the same [`ModelConfig`].
    pub fn forward_as(&self, x: Tensor<B, 3>, attend: Attend) -> Tensor<B, 3> {
        self.forward_inner(x, None, attend)
    }

    pub fn forward_cached(&self, x: Tensor<B, 3>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        self.forward_inner(x, Some(cache), Attend::Causal)
    }

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        cache: Option<&mut KvCache<B>>,
        attend: Attend,
    ) -> Tensor<B, 3> {
        let attn = |t: Tensor<B, 3>| match cache {
            Some(c) => self.attn.forward_cached(t, c),
            None => self.attn.forward_as(t, attend),
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

    /// Adding [`Attend`] must not have moved the existing path. Every caller in
    /// the crate uses `forward`, so if `forward` and `forward_as(.., Causal)`
    /// ever diverge, the language model changed while nobody was looking at it.
    #[test]
    fn the_causal_mode_is_exactly_what_forward_always_did() {
        let d = Default::default();
        let cfg = ModelConfig::tiny();
        let block = Block::<TestBackend>::new(&cfg, &d);
        let x = Tensor::<TestBackend, 3>::random([2, 6, cfg.d_model], Distribution::Default, &d);

        let want: Vec<f32> = block.forward(x.clone()).into_data().to_vec().unwrap();
        let got: Vec<f32> = block
            .forward_as(x, Attend::Causal)
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(want, got);
    }

    /// ...and the bidirectional mode must actually differ, or the argument is
    /// decorative. Both placements, because `Post` routes the residual
    /// differently and could in principle wash the difference out.
    #[test]
    fn the_bidirectional_mode_actually_changes_the_output() {
        let d = Default::default();
        for placement in [NormPlacement::Pre, NormPlacement::Post] {
            let cfg = ModelConfig {
                norm_placement: placement,
                ..ModelConfig::tiny()
            };
            let block = Block::<TestBackend>::new(&cfg, &d);
            let x =
                Tensor::<TestBackend, 3>::random([2, 6, cfg.d_model], Distribution::Default, &d);

            let causal = block.forward_as(x.clone(), Attend::Causal);
            let bidi = block.forward_as(x, Attend::Bidirectional);
            let delta: f32 = (causal - bidi).abs().sum().into_scalar();
            assert!(delta > 1e-4, "{placement:?}: the mask made no difference");
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

    /// The budget has to track the `qk_norm` flag, not just the default. A
    /// config field the analytic count ignores is a budget that quietly lies
    /// about any config that sets it.
    #[test]
    fn the_analytic_budget_follows_qk_norm() {
        let d = Default::default();
        let cfg = ModelConfig {
            qk_norm: true,
            ..ModelConfig::quark_3m()
        };
        let block = Block::<TestBackend>::new(&cfg, &d);
        let expected = cfg
            .budget()
            .iter()
            .find(|e| e.name == "layers")
            .unwrap()
            .params;
        assert_eq!(block.num_params(), expected / cfg.n_unique_layers);

        let off = Block::<TestBackend>::new(&ModelConfig::quark_3m(), &d);
        assert_eq!(block.num_params() - off.num_params(), 2 * cfg.d_head());
    }
}
