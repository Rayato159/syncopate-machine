//! High-level inference API.
//!
//! Provides [`GenerationConfig`] and [`ChatModel`] for easy token-level
//! text generation. Users encode/decode text with their own tokenizer.
//!
//! # Non-streaming (all tokens at once)
//!
//! ```rust,no_run
//! use syncopate_machine::prelude::*;
//!
//! fn main() -> syncopate_machine::Result<()> {
//!     let model = ChatModel::load("checkpoints/latest.mpk")?;
//!     let token_ids = model.generate(&[1, 2, 3], GenerationConfig::default())?;
//!     println!("generated tokens: {:?}", token_ids);
//!     Ok(())
//! }
//! ```
//!
//! # Streaming (token by token, like ChatGPT)
//!
//! ```rust,no_run
//! use syncopate_machine::prelude::*;
//!
//! fn main() -> syncopate_machine::Result<()> {
//!     let model = ChatModel::load("checkpoints/latest.mpk")?;
//!     let full = model.generate_stream(
//!         &[1, 2, 3],
//!         GenerationConfig::default(),
//!         |token_id, _index| {
//!             // Decode with YOUR tokenizer and print word-by-word
//!             print!("{} ", token_id);
//!             true // return false to stop early
//!         },
//!     )?;
//!     Ok(())
//! }
//! ```

use crate::error::{Error, Result};
use crate::model::{ModelInferenceConfig, SyncopateModel, SyncopateModelConfig};
use crate::runtime::{DefaultAutodiffBackend, DefaultBackend, InferenceDevice, default_device};
use burn::module::AutodiffModule;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// GenerationConfig
// ---------------------------------------------------------------------------

/// Configuration for text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate (default: 64).
    pub max_new_tokens: usize,
    /// Pad token ID (default: 0).
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            pad_token_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ChatModel
// ---------------------------------------------------------------------------

/// High-level model for token-level text generation.
///
/// Load a trained checkpoint and generate token IDs in a single call.
/// `ChatModel` automatically discovers the model config next to the
/// checkpoint file. Users bring their own tokenizer to encode/decode text.
///
/// # Example
///
/// ```rust,no_run
/// use syncopate_machine::prelude::*;
///
/// fn main() -> syncopate_machine::Result<()> {
///     let model = ChatModel::load("checkpoints/latest.mpk")?;
///     let token_ids = model.generate(&[1, 2, 3], GenerationConfig::default())?;
///     println!("generated tokens: {:?}", token_ids);
///     Ok(())
/// }
/// ```
pub struct ChatModel {
    model: SyncopateModel<DefaultBackend>,
    device: InferenceDevice,
    config: SyncopateModelConfig,
}

impl ChatModel {
    /// Load a `ChatModel` from a checkpoint path.
    ///
    /// `path` should point to a `.mpk` weights file (e.g.
    /// `"checkpoints/latest.mpk"` or `"runs/chat/checkpoints/latest.mpk"`).
    ///
    /// The method resolves `config.json` relative to the checkpoint's parent
    /// directory for model architecture. Falls back to Params10M defaults.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let checkpoint_path = path.as_ref();

        // Resolve the directory that contains the checkpoint file.
        let checkpoint_dir = checkpoint_path
            .parent()
            .ok_or_else(|| Error::Io(format!("cannot determine parent of {:?}", checkpoint_path)))?
            .to_path_buf();

        // ------ config.json ------
        let config = match find_file(&[checkpoint_dir.join("config.json")]) {
            Ok(config_path) => {
                let json = fs::read_to_string(&config_path).map_err(|e| {
                    Error::Io(format!("failed to read {}: {e}", config_path.display()))
                })?;
                serde_json::from_str::<SyncopateModelConfig>(&json).map_err(|e| {
                    Error::Serialization(format!("failed to parse {}: {e}", config_path.display()))
                })?
            }
            Err(_) => {
                // Fall back to 10M-parameter preset defaults.
                // User should provide config.json for correct architecture.
                SyncopateModelConfig::preset_10m(8192, 512)
            }
        };

        // ------ device + model ------
        // Load with Autodiff backend first (needed for parameter loading),
        // then convert to inference-only inner backend via .valid().
        // This prevents VRAM leak from autodiff computation graphs during
        // autoregressive generation.
        let device = default_device()?;
        let mut model = SyncopateModel::<DefaultAutodiffBackend>::new(config.clone(), &device)?;
        model.load_parameters(checkpoint_path)?;
        let inner_device = device;
        let model = model.valid(); // Strip Autodiff wrapper.

        Ok(Self {
            model,
            device: inner_device,
            config,
        })
    }

    /// Generate token IDs from a prompt token sequence.
    ///
    /// Returns all generated tokens (prompt + new) at once.
    /// For streaming / token-by-token output, use [`Self::generate_stream`].
    pub fn generate(&self, prompt: &[u32], config: GenerationConfig) -> Result<Vec<u32>> {
        let inference_config = ModelInferenceConfig {
            max_new_tokens: config.max_new_tokens,
            pad_token_id: config.pad_token_id,
        };
        let output = self
            .model
            .infer_tokens(prompt, &inference_config, &self.device)?;
        Ok(output.token_ids)
    }

    /// Generate token IDs one at a time, invoking a callback for each newly
    /// produced token.
    ///
    /// This enables streaming / word-by-word output similar to ChatGPT.
    /// The callback receives `(token_id, index)` where `index` is the
    /// zero-based position of the *new* token (0 = first generated token).
    /// Return `false` from the callback to stop generation early.
    ///
    /// Returns the full output sequence (prompt + generated tokens).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use syncopate_machine::prelude::*;
    ///
    /// fn main() -> syncopate_machine::Result<()> {
    ///     let model = ChatModel::load("checkpoints/latest.mpk")?;
    ///     let prompt: &[u32] = &[1, 2, 3];
    ///
    ///     let full_output = model.generate_stream(
    ///         prompt,
    ///         GenerationConfig::default(),
    ///         |token_id, _index| {
    ///             // Stream each token as it is produced.
    ///             // Decode with YOUR tokenizer and print word-by-word.
    ///             print!("{} ", token_id); // Replace with actual decoding
    ///             true // return false to stop early
    ///         },
    ///     )?;
    ///
    ///     println!("\nFull sequence: {:?}", full_output);
    ///     Ok(())
    /// }
    /// ```
    pub fn generate_stream(
        &self,
        prompt: &[u32],
        config: GenerationConfig,
        on_token: impl FnMut(u32, usize) -> bool,
    ) -> Result<Vec<u32>> {
        let inference_config = ModelInferenceConfig {
            max_new_tokens: config.max_new_tokens,
            pad_token_id: config.pad_token_id,
        };
        let output =
            self.model
                .infer_tokens_stream(prompt, &inference_config, &self.device, on_token)?;
        Ok(output.token_ids)
    }

    /// Run a forward pass on the padded context and return logits.
    ///
    /// Returns a tensor of shape `[1, seq_len, vocab_size]`.
    /// Use this for custom sampling strategies (top-k, temperature, etc.).
    pub fn predict_logits(&self, context: &[u32]) -> Result<burn::Tensor<DefaultBackend, 3>> {
        let pad_token_id = 0;
        self.model
            .forward_logits(context, pad_token_id, &self.device)
    }

    /// Access the underlying neural model.
    pub fn model(&self) -> &SyncopateModel<DefaultBackend> {
        &self.model
    }

    /// Access the model configuration.
    pub fn config(&self) -> &SyncopateModelConfig {
        &self.config
    }

    /// Access the device the model is running on.
    pub fn device(&self) -> &InferenceDevice {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the first path in `candidates` that exists on disk, or an error.
fn find_file(candidates: &[PathBuf]) -> Result<PathBuf> {
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }
    let descriptions = candidates
        .iter()
        .map(|p| format!("  {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(Error::Io(format!(
        "file not found; searched:\n{descriptions}"
    )))
}
