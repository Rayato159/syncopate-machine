use std::error::Error as StdError;
use std::fmt;

/// Result alias used by the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level crate error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Config(String),
    Layout(String),
    Training(String),
    Inference(String),
    /// File I/O failure.
    Io(String),
    /// Serialization or deserialization failure.
    Serialization(String),
    /// Weights file config does not match the engine config.
    WeightsConfigMismatch(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(formatter, "invalid configuration: {message}"),
            Self::Layout(message) => write!(formatter, "layout error: {message}"),
            Self::Training(message) => write!(formatter, "training error: {message}"),
            Self::Inference(message) => write!(formatter, "inference error: {message}"),
            Self::Io(message) => write!(formatter, "I/O error: {message}"),
            Self::Serialization(message) => {
                write!(formatter, "serialization error: {message}")
            }
            Self::WeightsConfigMismatch(message) => {
                write!(formatter, "weights config mismatch: {message}")
            }
        }
    }
}

impl StdError for Error {}
