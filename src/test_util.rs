//! Shared test helpers.

#![cfg(test)]

use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Tensor};

/// Tests run on the CPU ndarray backend so that CI needs no GPU.
pub type TestBackend = burn_ndarray::NdArray<f32>;

/// Used by the training-step tests; unused until the harness lands.
#[allow(dead_code)]
pub type TestAutodiffBackend = burn::backend::Autodiff<TestBackend>;

/// Assert two tensors agree elementwise, reporting the worst offender rather
/// than just "not equal".
pub fn assert_close<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, tol: f32) {
    assert_eq!(a.dims(), b.dims(), "shape mismatch");
    // `into_scalar` yields the backend's own float element type, which is only
    // `f32` by coincidence on `NdArray<f32>`; `elem` converts it explicitly so
    // the helper stays generic over backends.
    let diff: f32 = (a.clone() - b.clone())
        .abs()
        .max()
        .into_scalar()
        .elem::<f32>();
    assert!(
        diff <= tol,
        "tensors differ by {diff} (tolerance {tol})\n  left:  {a}\n  right: {b}"
    );
}
