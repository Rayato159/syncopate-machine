//! # syncopate-machine
//!
//! A small decoder-only transformer for game NPC/action chat, powered by
//! [Burn](https://github.com/tracel-ai/burn).
//!
//! ## Quick Start
//!
//! ### Training
//!
//! ```rust,no_run
//! use syncopate_machine::prelude::*;
//!
//! fn main() -> syncopate_machine::Result<()> {
//!     let mut trainer = Trainer::builder()
//!         .vocab_size(1000)
//!         .budget(ParameterBudget::Params10M)
//!         .device(auto_device()?)
//!         .batch_size(16).seq_len(128).steps(50_000)
//!         .build()?;
//!
//!     let sequences = vec![vec![5, 17, 3, 9, 2], vec![7, 16, 3, 11, 2]];
//!     let report = trainer.train_on_token_sequences(&sequences)?;
//!     Ok(())
//! }
//! ```
//!
//! ### Inference
//!
//! ```rust,no_run
//! use syncopate_machine::prelude::*;
//!
//! fn main() -> syncopate_machine::Result<()> {
//!     let model = ChatModel::load("checkpoints/latest.mpk")?;
//!
//!     // One-shot
//!     let ids = model.generate(&[5, 17, 3], GenerationConfig::default())?;
//!
//!     // Streaming
//!     model.generate_stream(&[5, 17, 3], GenerationConfig::default(), |id, _| {
//!         print!("{id} "); true
//!     })?;
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! Enable the `cuda` feature for NVIDIA GPU acceleration.
//! Default uses Burn Flex for CPU training with auto SIMD detection.

// ---- Public modules (the only ones users should care about) ----
pub mod device;
pub mod inference;
pub mod prelude;
pub mod training;

// ---- Internal modules ----
pub(crate) mod config;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod layout;
pub(crate) mod lm;
pub(crate) mod model;
pub(crate) mod optim;
pub(crate) mod param_io;
pub(crate) mod runtime;
pub(crate) mod screen;
pub(crate) mod tile;

// ---- WASM module (only compiled for wasm32 target with the wasm feature) ----
#[cfg(feature = "wasm")]
pub mod wasm;

// ---- High-level API re-exports ----
#[cfg(not(feature = "cuda"))]
pub use device::cpu;
pub use device::{auto_device, cuda};
pub use inference::{ChatModel, GenerationConfig};
pub use training::{ParameterBudget, Trainer, TrainingReport};

// ---- Core types (available through prelude) ----
pub use error::{Error, Result};
pub use model::{
    AttentionKernel, DefaultMultiscreenModel, DefaultSyncopateModel, EvaluationResult,
    ModelInferenceConfig, ModelTrainingConfig, ModelTrainingReport, MultiscreenModel,
    MultiscreenModelConfig, MultiscreenModelOutput, MultiscreenParameterBudget, SyncopateModel,
    SyncopateModelConfig, SyncopateModelOutput, SyncopateParameterBudget,
    cross_entropy_loss_with_mask,
};
pub use runtime::{DefaultAutodiffBackend, DefaultBackend, Device, device_label};

#[cfg(not(feature = "cuda"))]
pub use runtime::default_device;

#[cfg(feature = "cuda")]
pub use runtime::{CudaAutodiffBackend, CudaDevice, CudaMultiscreenModel, CudaSyncopateModel};

// ---- Engine types (lightweight transition engine) ----
pub use config::{InferenceConfig, MultiscreenConfig, TrimConfig};
pub use engine::{InferenceOutput, MultiscreenEngine, TrainInput, TrainReport};
pub use layout::{
    ScreenLayout, TokenSpan, causal_softmask, causal_trim_relevance, trim_and_square,
};
pub use screen::{Screen, ScreenConfig};
pub use tile::{ScreeningGridConfig, Tile, TileConfig};

// ---- Burn re-exports ----
pub use burn::{
    tensor::backend::{AutodiffBackend, Backend},
    tensor::{Int, Tensor, TensorData},
};

#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;

#[deprecated(note = "use MultiscreenConfig; the paper styles this as one word")]
pub type MultiScreenConfig = MultiscreenConfig;

#[deprecated(note = "use MultiscreenEngine; the paper styles this as one word")]
pub type MultiScreenEngine = MultiscreenEngine;

#[deprecated(note = "use ScreeningGridConfig for N_L x N_H naming")]
pub type GridConfig = ScreeningGridConfig;
