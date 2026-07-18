//! Quark: a 3.0M-parameter language model family and training harness on burn.
//!
//! See `docs/DESIGN.md` for the feasibility analysis and the reasoning behind
//! the architecture. The short version:
//!
//! * The 3.0M budget is spent on embedding *rank* rather than width, because
//!   tied+factorized embeddings cap the output distribution's rank at `d_emb`.
//! * Cross-layer weight sharing buys parameters, not compute -- the reference
//!   model stores 2.87M parameters but costs ~20.6M to train.
//! * Perplexity is never compared across tokenizers. The harness reports bits
//!   per byte and word-level perplexity, both tokenizer-independent, and it
//!   evaluates the GPT-2 baseline with the same code path.

pub mod compress;
pub mod config;
pub mod data;
pub mod eval;
pub mod model;
pub mod train;

#[cfg(test)]
mod test_util;

pub use config::{LayerSchedule, ModelConfig, NormPlacement};
pub use model::QuarkLm;
pub use train::TrainConfig;
