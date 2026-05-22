//! High-level training API with a builder pattern.
//!
//! The [`Trainer`] struct wraps model construction, token-sequence training,
//! and checkpoint management behind a single ergonomic interface with sensible
//! defaults.
//!
//! Users encode their own text into token IDs using any tokenizer they choose,
//! then pass the `Vec<Vec<u32>>` sequences to [`Trainer::train_on_token_sequences`].
//!
//! # Example
//!
//! ```rust,no_run
//! use syncopate_machine::prelude::*;
//!
//! fn main() -> syncopate_machine::Result<()> {
//!     let mut trainer = Trainer::builder()
//!         .vocab_size(1000)
//!         .budget(ParameterBudget::Params10M)
//!         .batch_size(4)
//!         .seq_len(64)
//!         .steps(100)
//!         .device(auto_device()?)
//!         .build()?;
//!
//!     // Token sequences from YOUR tokenizer
//!     let sequences = vec![
//!         vec![1, 2, 3, 4, 5],
//!         vec![1, 2, 6, 7, 5],
//!     ];
//!
//!     let report = trainer.train_on_token_sequences(&sequences)?;
//!     println!("trained {} steps, final loss {:.4}", report.steps, report.final_loss);
//!     Ok(())
//! }
//! ```

use crate::error::{Error, Result};
use crate::model::{
    AttentionKernel, DefaultSyncopateModel, ModelTrainingConfig, ModelTrainingReport,
    SyncopateModelConfig,
};
use crate::runtime::{Device, default_device};
use std::fs;
use std::path::Path;

/// Re-export of the parameter budget enum for convenience.
pub use crate::model::SyncopateParameterBudget as ParameterBudget;

// ---------------------------------------------------------------------------
// TrainingReport
// ---------------------------------------------------------------------------

/// Summary returned after training via the high-level [`Trainer`].
#[derive(Clone, Debug)]
pub struct TrainingReport {
    /// Number of training steps completed.
    pub steps: usize,
    /// Final training loss.
    pub final_loss: f32,
    /// The lowest loss observed across all training steps.
    pub best_loss: f32,
    /// The step at which `best_loss` was recorded.
    pub best_loss_step: usize,
    /// Total number of parameters in the model.
    pub parameter_count: usize,
    /// Path the checkpoint was saved to, if any.
    pub checkpoint_path: Option<String>,
}

impl TrainingReport {
    fn from_model_report(report: &ModelTrainingReport, checkpoint_path: Option<String>) -> Self {
        Self {
            steps: report.steps,
            final_loss: report.final_loss,
            best_loss: report.best_loss,
            best_loss_step: report.best_loss_step,
            parameter_count: report.parameter_count,
            checkpoint_path,
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// High-level trainer that bundles a model and training config.
///
/// Construct via [`Trainer::builder()`] and the [`TrainerBuilder`] struct.
/// Users provide token sequences (`Vec<Vec<u32>>`) from their own tokenizer.
pub struct Trainer {
    model: DefaultSyncopateModel,
    training_config: ModelTrainingConfig,
    checkpoint_dir: Option<String>,
    checkpoint_interval: usize,
    #[allow(dead_code)]
    run_dir: Option<String>,
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("training_config", &self.training_config)
            .field("checkpoint_dir", &self.checkpoint_dir)
            .field("checkpoint_interval", &self.checkpoint_interval)
            .field("run_dir", &self.run_dir)
            .finish_non_exhaustive()
    }
}

impl Trainer {
    /// Returns a new [`TrainerBuilder`] with sensible defaults.
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::new()
    }

    /// Trains the model on the provided token sequences.
    ///
    /// Each inner `Vec<u32>` is a tokenized text sample. Use your own
    /// tokenizer to produce these sequences before calling this method.
    ///
    /// The `on_step` callback is invoked after each optimizer step with
    /// `(step_index, loss_value)`.
    pub fn train_on_token_sequences_with_callback(
        &mut self,
        sequences: &[Vec<u32>],
        on_step: impl FnMut(usize, f32),
    ) -> Result<TrainingReport> {
        if sequences.is_empty() {
            return Err(Error::Training("no training sequences provided".into()));
        }

        let mut config = self.training_config.clone();
        config.checkpoint_dir = self.checkpoint_dir.clone();
        config.checkpoint_interval = self.checkpoint_interval;

        let device = self.model_device();
        let report = self
            .model
            .train_token_sequences(sequences, &config, &device, on_step)?;

        let checkpoint_path = match &self.checkpoint_dir {
            Some(dir) => {
                let dir_path = Path::new(dir);
                fs::create_dir_all(dir_path).map_err(|e| {
                    Error::Io(format!(
                        "failed to create checkpoint directory {:?}: {}",
                        dir, e
                    ))
                })?;
                let path = dir_path.join("checkpoint.mpk");
                self.model.save_parameters(&path)?;
                Some(path.to_string_lossy().into_owned())
            }
            None => None,
        };

        Ok(TrainingReport::from_model_report(&report, checkpoint_path))
    }

    /// Convenience wrapper that trains without a per-step callback.
    pub fn train_on_token_sequences(&mut self, sequences: &[Vec<u32>]) -> Result<TrainingReport> {
        self.train_on_token_sequences_with_callback(sequences, |_, _| {})
    }

    /// Trains the model on chat-style (prompt, response) token-ID pairs.
    ///
    /// Each element of `chat_pairs` is `(prompt_token_ids, response_token_ids)`.
    /// The model sees the full context but loss is computed only on response
    /// tokens, preventing the model from learning to generate role labels.
    ///
    /// The `on_step` callback is invoked after each optimizer step with
    /// `(step_index, loss_value)`.
    pub fn train_on_chat_sequences_with_callback(
        &mut self,
        chat_pairs: &[(Vec<u32>, Vec<u32>)],
        on_step: impl FnMut(usize, f32),
    ) -> Result<TrainingReport> {
        if chat_pairs.is_empty() {
            return Err(Error::Training("no training chat pairs provided".into()));
        }

        let mut config = self.training_config.clone();
        config.checkpoint_dir = self.checkpoint_dir.clone();
        config.checkpoint_interval = self.checkpoint_interval;

        let device = self.model_device();
        let report = self
            .model
            .train_chat_sequences(chat_pairs, &config, &device, on_step)?;

        let checkpoint_path = match &self.checkpoint_dir {
            Some(dir) => {
                let dir_path = Path::new(dir);
                fs::create_dir_all(dir_path).map_err(|e| {
                    Error::Io(format!(
                        "failed to create checkpoint directory {:?}: {}",
                        dir, e
                    ))
                })?;
                let path = dir_path.join("checkpoint.mpk");
                self.model.save_parameters(&path)?;
                Some(path.to_string_lossy().into_owned())
            }
            None => None,
        };

        Ok(TrainingReport::from_model_report(&report, checkpoint_path))
    }

    /// Convenience wrapper that trains on chat pairs without a per-step callback.
    pub fn train_on_chat_sequences(
        &mut self,
        chat_pairs: &[(Vec<u32>, Vec<u32>)],
    ) -> Result<TrainingReport> {
        self.train_on_chat_sequences_with_callback(chat_pairs, |_, _| {})
    }

    /// Trains on chat pairs with periodic validation evaluation.
    ///
    /// Every `eval_interval` steps the model is evaluated on `val_sequences`
    /// (flat token sequences) and `on_step` is called with
    /// `(step, train_loss, Some(val_loss))`.  Between evaluations the
    /// third argument is `None`.
    ///
    /// The best checkpoint is saved as `best_val.mpk` inside the
    /// checkpoint directory.  The final model is saved as `latest.mpk`.
    pub fn train_on_chat_sequences_with_validation(
        &mut self,
        chat_pairs: &[(Vec<u32>, Vec<u32>)],
        val_sequences: &[Vec<u32>],
        eval_interval: usize,
        on_step: impl FnMut(usize, f32, Option<f32>),
    ) -> Result<TrainingReport> {
        if chat_pairs.is_empty() {
            return Err(Error::Training("no training chat pairs provided".into()));
        }

        let mut config = self.training_config.clone();
        config.checkpoint_dir = self.checkpoint_dir.clone();
        config.checkpoint_interval = self.checkpoint_interval;

        let device = self.model_device();
        let report = self.model.train_chat_sequences_with_val(
            chat_pairs,
            val_sequences,
            eval_interval,
            &config,
            &device,
            on_step,
        )?;

        let checkpoint_path = self.checkpoint_dir.as_ref().map(|dir| {
            let dir_path = Path::new(dir);
            dir_path.join("best_val.mpk").to_string_lossy().into_owned()
        });

        Ok(TrainingReport::from_model_report(&report, checkpoint_path))
    }

    /// Saves a model checkpoint to the given path.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::Io(format!(
                    "failed to create checkpoint directory {:?}: {}",
                    parent, e
                ))
            })?;
        }
        self.model.save_parameters(path)
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &DefaultSyncopateModel {
        &self.model
    }

    /// Returns a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut DefaultSyncopateModel {
        &mut self.model
    }

    /// Returns a reference to the training configuration.
    pub fn training_config(&self) -> &ModelTrainingConfig {
        &self.training_config
    }

    /// Obtains the device the model is currently on.
    fn model_device(&self) -> Device {
        Device::default()
    }
}

// ---------------------------------------------------------------------------
// TrainerBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`Trainer`] with sensible defaults.
pub struct TrainerBuilder {
    vocab_size: Option<usize>,
    budget: ParameterBudget,
    device: Option<Device>,
    batch_size: usize,
    seq_len: usize,
    steps: usize,
    learning_rate: f64,
    min_learning_rate: f64,
    warmup_steps: usize,
    weight_decay: f64,
    grad_clip_norm: Option<f64>,
    attention_kernel: AttentionKernel,
    checkpoint_dir: Option<String>,
    checkpoint_interval: usize,
    run_dir: Option<String>,
}

impl TrainerBuilder {
    /// Creates a new builder with default values.
    fn new() -> Self {
        Self {
            vocab_size: None,
            budget: ParameterBudget::Params10M,
            device: None,
            batch_size: 4,
            seq_len: 128,
            steps: 1000,
            learning_rate: 3e-4,
            min_learning_rate: 3e-5,
            warmup_steps: 200,
            weight_decay: 0.1,
            grad_clip_norm: Some(1.0),
            attention_kernel: AttentionKernel::Softmax,
            checkpoint_dir: None,
            checkpoint_interval: 1000,
            run_dir: None,
        }
    }

    /// Sets the vocabulary size (required).
    ///
    /// This is the number of distinct token IDs your tokenizer produces.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Sets the parameter budget. Defaults to [`ParameterBudget::Params10M`].
    pub fn budget(mut self, budget: ParameterBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Sets the compute device. Defaults to [`default_device()`].
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Sets the batch size. Defaults to 4.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets the sequence length. Defaults to 128.
    pub fn seq_len(mut self, len: usize) -> Self {
        self.seq_len = len;
        self
    }

    /// Sets the number of training steps. Defaults to 1000.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// Sets the peak learning rate. Defaults to `3e-4`.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the cosine schedule floor. Defaults to `3e-5`.
    pub fn min_learning_rate(mut self, lr: f64) -> Self {
        self.min_learning_rate = lr;
        self
    }

    /// Sets warmup steps. Defaults to 200.
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Sets the weight decay. Defaults to `0.1`.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Sets gradient clip norm. Defaults to `Some(1.0)`.
    pub fn grad_clip_norm(mut self, norm: Option<f64>) -> Self {
        self.grad_clip_norm = norm;
        self
    }

    /// Sets the attention kernel. Defaults to [`AttentionKernel::Softmax`].
    ///
    /// Use [`AttentionKernel::HigherOrder`] for the experimental normalized
    /// second-order causal attention kernel.
    pub fn attention_kernel(mut self, kernel: AttentionKernel) -> Self {
        self.attention_kernel = kernel;
        self
    }

    /// Sets the directory where checkpoints are saved.
    pub fn checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Sets how often (in steps) to save checkpoints. Defaults to 1000.
    pub fn checkpoint_interval(mut self, steps: usize) -> Self {
        self.checkpoint_interval = steps;
        self
    }

    /// Overrides the run directory.
    pub fn run_dir(mut self, dir: impl Into<String>) -> Self {
        self.run_dir = Some(dir.into());
        self
    }

    /// Builds the [`Trainer`].
    ///
    /// This constructs the model config from the parameter budget + vocab size,
    /// and creates the model.
    pub fn build(self) -> Result<Trainer> {
        let vocab_size = self.vocab_size.ok_or_else(|| {
            Error::Config("vocab_size is required; call .vocab_size(n) before .build()".into())
        })?;

        let device = match self.device {
            Some(d) => d,
            None => default_device()?,
        };

        let config =
            SyncopateModelConfig::for_parameter_budget(self.budget, vocab_size, self.seq_len)
                .with_attention_kernel(self.attention_kernel);
        let model = DefaultSyncopateModel::new(config, &device)?;

        let training_config = ModelTrainingConfig {
            steps: self.steps,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            min_learning_rate: self.min_learning_rate,
            warmup_steps: self.warmup_steps,
            weight_decay: self.weight_decay,
            grad_clip_norm: self.grad_clip_norm,
            pad_token_id: 0,
            checkpoint_dir: None, // injected by Trainer during training
            checkpoint_interval: 0,
        };

        let run_dir = self.run_dir.or_else(|| Some("runs/latest".to_string()));

        Ok(Trainer {
            model,
            training_config,
            checkpoint_dir: self.checkpoint_dir,
            checkpoint_interval: self.checkpoint_interval,
            run_dir,
        })
    }
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_vocab_size() {
        let result = Trainer::builder().build();
        assert!(result.is_err(), "build should fail without vocab_size");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("vocab_size"),
            "error should mention vocab_size: {}",
            msg
        );
    }

    #[test]
    fn training_report_from_model_report() {
        let model_report = ModelTrainingReport {
            steps: 500,
            final_loss: 0.123,
            best_loss: 0.100,
            best_loss_step: 420,
            training_window_count: 100,
            parameter_count: 10_000_000,
        };
        let report =
            TrainingReport::from_model_report(&model_report, Some("runs/checkpoint.mpk".into()));
        assert_eq!(report.steps, 500);
        assert!((report.final_loss - 0.123).abs() < f32::EPSILON);
        assert!((report.best_loss - 0.100).abs() < f32::EPSILON);
        assert_eq!(report.best_loss_step, 420);
        assert_eq!(report.parameter_count, 10_000_000);
        assert_eq!(
            report.checkpoint_path.as_deref(),
            Some("runs/checkpoint.mpk")
        );
    }

    #[test]
    fn builder_defaults() {
        let builder = TrainerBuilder::new();
        assert!(builder.vocab_size.is_none());
        assert!(matches!(builder.budget, ParameterBudget::Params10M));
        assert!(builder.device.is_none());
        assert_eq!(builder.batch_size, 4);
        assert_eq!(builder.seq_len, 128);
        assert_eq!(builder.steps, 1000);
        assert!((builder.learning_rate - 3e-4).abs() < f64::EPSILON);
        assert!((builder.min_learning_rate - 3e-5).abs() < f64::EPSILON);
        assert_eq!(builder.warmup_steps, 200);
        assert!((builder.weight_decay - 0.1).abs() < f64::EPSILON);
        assert_eq!(builder.checkpoint_interval, 1000);
    }
}
