//! An optional text compressor: a discrete-bottleneck autoencoder built out of
//! the pieces the language model already has.
//!
//! See `docs/COMPRESSION.md` for the survey this design comes out of, the
//! choice of base configuration, and the metric.
//!
//! Everything specific to compression lives under this module. The rest of the
//! crate learns about it through exactly two things: one line in `lib.rs`, and
//! [`Attend`](crate::model::Attend), a mask selector on the existing attention
//! module that adds no parameters and leaves the causal path bit-identical.
//! Nothing here is reachable from a normal training run.

pub mod config;
pub mod model;
pub mod quantize;
pub mod train;

pub use config::CompressConfig;
pub use model::Compressor;
pub use quantize::Fsq;
pub use train::CompressTrainConfig;
