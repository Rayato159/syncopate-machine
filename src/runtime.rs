use crate::error::{Error, Result};
use burn::{
    backend::Autodiff,
    tensor::backend::{Backend, BackendTypes},
};
use std::env;

// ---------------------------------------------------------------------------
// Inference-only backend (no autodiff)
// ---------------------------------------------------------------------------

/// Inference-only device type — same physical device, but without autodiff wrapper.
pub type InferenceDevice = <DefaultBackend as BackendTypes>::Device;

// ---------------------------------------------------------------------------
// Backend selection (feature-gated)
// ---------------------------------------------------------------------------

/// Default CPU backend (Flex) — used when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
pub type DefaultBackend = burn::backend::Flex;

/// Default CUDA backend — used when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
pub type DefaultBackend = burn::backend::Cuda;

/// Default training backend. Burn's `Autodiff` decorator adds backpropagation.
pub type DefaultAutodiffBackend = Autodiff<DefaultBackend>;

/// Device type for the active backend.
pub type Device = <DefaultAutodiffBackend as BackendTypes>::Device;

/// Chooses the default device based on the active backend.
///
/// With `cuda` feature: returns the default CUDA device (GPU 0).
/// Without `cuda` feature: returns the default Flex (CPU) device.
///
/// Also reads `SYNCOPATE_DEVICE` env var for manual override.
pub fn default_device() -> Result<Device> {
    let requested = env::var("SYNCOPATE_DEVICE")
        .or_else(|_| env::var("MULTISCREEN_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string());
    match requested.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(Device::default()),
        "cpu" | "flex" => {
            #[cfg(feature = "cuda")]
            {
                Err(Error::Config(
                    "SYNCOPATE_DEVICE=cpu is not supported when compiled with the cuda feature. \
                     Recompile without --features cuda for CPU training."
                        .to_string(),
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Device::default())
            }
        }
        other => Err(Error::Config(format!(
            "unsupported SYNCOPATE_DEVICE={other:?}; use auto"
        ))),
    }
}

/// Human-readable label for the active backend device.
pub fn device_label(device: &Device) -> String {
    <DefaultAutodiffBackend as Backend>::name(device)
}

// ---------------------------------------------------------------------------
// CUDA-specific types (always available for documentation)
// ---------------------------------------------------------------------------

/// Autodiff-wrapped CUDA backend for training on GPU.
#[cfg(feature = "cuda")]
pub type CudaAutodiffBackend = Autodiff<burn::backend::Cuda>;

/// Device type for the CUDA backend.
#[cfg(feature = "cuda")]
pub type CudaDevice = <CudaAutodiffBackend as BackendTypes>::Device;

/// Convenience alias for a Syncopate model backed by CUDA.
#[cfg(feature = "cuda")]
pub type CudaSyncopateModel = crate::model::SyncopateModel<CudaAutodiffBackend>;

#[cfg(feature = "cuda")]
pub type CudaMultiscreenModel = CudaSyncopateModel;
