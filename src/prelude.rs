// ---- High-level API (recommended for most users) ----
pub use crate::{
    ChatModel, GenerationConfig, ParameterBudget, Trainer, TrainingReport, auto_device, cuda,
};

#[cfg(not(feature = "cuda"))]
pub use crate::cpu;

// ---- Core types ----
pub use crate::{
    DefaultAutodiffBackend, DefaultBackend, DefaultMultiscreenModel, DefaultSyncopateModel, Device,
    Error, Result, device_label,
};

#[cfg(not(feature = "cuda"))]
pub use crate::default_device;

// ---- Model configuration ----
pub use crate::{
    AttentionKernel, EvaluationResult, ModelInferenceConfig, ModelTrainingConfig,
    ModelTrainingReport, MultiscreenModel, MultiscreenModelConfig, MultiscreenModelOutput,
    MultiscreenParameterBudget, SyncopateModel, SyncopateModelConfig, SyncopateModelOutput,
    SyncopateParameterBudget, cross_entropy_loss_with_mask,
};

// ---- Engine (lightweight transition engine) ----
pub use crate::{InferenceOutput, MultiscreenEngine, TrainInput, TrainReport};

// ---- Layout utilities ----
pub use crate::{
    InferenceConfig, Int, MultiscreenConfig, Screen, ScreenConfig, ScreenLayout,
    ScreeningGridConfig, Tensor, TensorData, TileConfig, TokenSpan, TrimConfig, causal_softmask,
    causal_trim_relevance, trim_and_square,
};

#[cfg(feature = "cuda")]
pub use crate::{CudaAutodiffBackend, CudaDevice, CudaMultiscreenModel, CudaSyncopateModel};

pub use crate::AutodiffBackend;
pub use crate::Backend;
