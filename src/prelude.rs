pub use crate::{auto_device, cuda};

#[cfg(not(feature = "cuda"))]
pub use crate::cpu;

pub use crate::{
    AttentionKernel, DefaultAutodiffBackend, DefaultBackend, DefaultSyncopateModel, Device, Error,
    EvaluationResult, ModelTrainingConfig, ModelTrainingReport, Result, SyncopateModel,
    SyncopateModelConfig, cross_entropy_loss_with_mask, device_label,
};

#[cfg(not(feature = "cuda"))]
pub use crate::default_device;

#[cfg(feature = "cuda")]
pub use crate::{CudaAutodiffBackend, CudaDevice, CudaSyncopateModel};

pub use crate::{AutodiffBackend, Backend, Int, Tensor, TensorData};
