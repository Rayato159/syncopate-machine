//! Device abstraction for CPU and CUDA.

use crate::error::Result;
use crate::runtime::Device;

/// Returns the default device for the active backend.
///
/// - Without `cuda` feature: returns Flex (CPU) device.
/// - With `cuda` feature: returns CUDA device (GPU 0).
///
/// ```rust,no_run
/// use syncopate_machine::prelude::*;
/// fn main() -> syncopate_machine::Result<()> {
///     let device = auto_device()?;
///     Ok(())
/// }
/// ```
pub fn auto_device() -> Result<Device> {
    Ok(Device::default())
}

/// Returns the default CPU device.
///
/// Only available without the `cuda` feature. When compiled with CUDA,
/// use [`auto_device`] instead.
///
/// ```rust,no_run
/// use syncopate_machine::prelude::*;
/// fn main() -> syncopate_machine::Result<()> {
///     let device = cpu()?;
///     Ok(())
/// }
/// ```
#[cfg(not(feature = "cuda"))]
pub fn cpu() -> Result<Device> {
    Ok(Device::default())
}

/// Returns a CUDA device for the given GPU index.
///
/// Only available with the `cuda` feature.
///
/// ```toml
/// [dependencies]
/// syncopate-machine = { version = "0.3", features = ["cuda"] }
/// ```
#[cfg(feature = "cuda")]
pub fn cuda(_index: usize) -> Result<Device> {
    // Burn's Cuda device doesn't support selecting GPU index via
    // the simple API yet. The index parameter is reserved for future use.
    Ok(Device::default())
}

/// Returns an error — CUDA is not compiled in.
#[cfg(not(feature = "cuda"))]
pub fn cuda(_index: usize) -> Result<Device> {
    Err(crate::error::Error::Config(
        "CUDA is not available. Enable the 'cuda' feature in Cargo.toml:\n\
         [dependencies]\n\
         syncopate-machine = { version = \"0.3\", features = [\"cuda\"] }"
            .to_string(),
    ))
}
