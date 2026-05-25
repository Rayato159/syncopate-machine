//! # syncopate-machine
//!
//! Tiny Burn-backed transformer for action-sequence prediction.
//! It predicts integer action IDs, not tokenizer text.

// ---- Public modules (the only ones users should care about) ----
pub mod device;
pub mod prelude;

// ---- Internal modules ----
pub(crate) mod error;
pub(crate) mod model;
pub(crate) mod runtime;

// ---- WASM module (only compiled for wasm32 target with the wasm feature) ----
#[cfg(feature = "wasm")]
pub mod wasm;

// ---- High-level API re-exports ----
#[cfg(not(feature = "cuda"))]
pub use device::cpu;
pub use device::{auto_device, cuda};

// ---- Core types (available through prelude) ----
pub use error::{Error, Result};
pub use model::{
    AttentionKernel, DefaultSyncopateModel, EvaluationResult, ModelTrainingConfig,
    ModelTrainingReport, SyncopateModel, SyncopateModelConfig, SyncopateParameterBudget,
    cross_entropy_loss_with_mask,
};
pub use runtime::{DefaultAutodiffBackend, DefaultBackend, Device, device_label};

#[cfg(not(feature = "cuda"))]
pub use runtime::default_device;

#[cfg(feature = "cuda")]
pub use runtime::{CudaAutodiffBackend, CudaDevice, CudaSyncopateModel};

// ---- Burn re-exports ----
pub use burn::{
    tensor::backend::{AutodiffBackend, Backend},
    tensor::{Int, Tensor, TensorData},
};

#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;
