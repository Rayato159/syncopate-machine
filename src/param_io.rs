//! Burn recorder helpers for model parameter persistence.
//!
//! Neural model save/load now goes through Burn records instead of the old
//! direct raw tensor stream from the first neural port.

use crate::error::{Error, Result};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use std::path::Path;

#[allow(dead_code)]
pub type DefaultParameterRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

#[allow(dead_code)]
pub fn save_module<B, M>(module: M, path: impl AsRef<Path>) -> Result<()>
where
    B: Backend,
    M: Module<B>,
{
    let recorder = DefaultParameterRecorder::new();
    module
        .save_file(path.as_ref().to_path_buf(), &recorder)
        .map_err(|err| Error::Serialization(err.to_string()))
}

#[allow(dead_code)]
pub fn load_module<B, M>(module: M, path: impl AsRef<Path>, device: &B::Device) -> Result<M>
where
    B: Backend,
    M: Module<B>,
{
    let recorder = DefaultParameterRecorder::new();
    module
        .load_file(path.as_ref().to_path_buf(), &recorder, device)
        .map_err(|err| Error::Serialization(err.to_string()))
}
