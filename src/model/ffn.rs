//! SwiGLU feed-forward block.
//!
//! `burn::nn::SwiGlu` implements only the gated projection
//! `silu(W_gate x) * W_up x` -- it has no down projection, so a LLaMA-style FFN
//! needs an explicit `Linear(d_ff, d_model)` after it. Parameter cost is
//! `3 * d_model * d_ff`, versus `2 * d_model * d_ff` for a plain GELU MLP; the
//! usual convention is to shrink `d_ff` to hold the count constant, which is
//! why the reference config pairs `d_model = 384` with `d_ff = 1152` (3x)
//! rather than the GPT-2 4x.

use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, SwiGlu, SwiGluConfig},
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Config, Debug)]
pub struct SwiGluFeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct SwiGluFeedForward<B: Backend> {
    gate_up: SwiGlu<B>,
    down: Linear<B>,
    dropout: Dropout,
}

impl SwiGluFeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFeedForward<B> {
        SwiGluFeedForward {
            gate_up: SwiGluConfig::new(self.d_model, self.d_ff)
                .with_bias(false)
                .init(device),
            down: LinearConfig::new(self.d_ff, self.d_model)
                .with_bias(false)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> SwiGluFeedForward<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.dropout
            .forward(self.down.forward(self.gate_up.forward(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::Distribution;

    #[test]
    fn preserves_shape() {
        let d = Default::default();
        let ffn = SwiGluFeedForwardConfig::new(16, 48).init::<TestBackend>(&d);
        let x = Tensor::<TestBackend, 3>::random([2, 5, 16], Distribution::Default, &d);
        assert_eq!(ffn.forward(x).dims(), [2, 5, 16]);
    }

    /// The analytic budget in `config.rs` assumes exactly `3 * d_model * d_ff`.
    /// If burn's SwiGlu ever grows a bias or a third projection, the budget
    /// silently becomes a lie -- so pin it here.
    #[test]
    fn parameter_count_is_three_matrices() {
        let d = Default::default();
        let (d_model, d_ff) = (16, 48);
        let ffn = SwiGluFeedForwardConfig::new(d_model, d_ff).init::<TestBackend>(&d);
        assert_eq!(ffn.num_params(), 3 * d_model * d_ff);
    }
}
