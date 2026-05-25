use crate::{
    error::{Error, Result},
    runtime::DefaultAutodiffBackend,
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Module, Param},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Int, Tensor, TensorData, activation,
        backend::{AutodiffBackend, Backend},
    },
};
use serde::{Deserialize, Serialize};
use std::{f32::consts::PI, f64::consts::PI as PI64, path::Path};

const EPS: f32 = 1e-6;
const NEG_INF: f32 = -1.0e9;

/// Supported Syncopate parameter budgets.
///
/// The final count is approximate because ID embeddings scale with the
/// caller's vocabulary size. Use [`SyncopateModelConfig::estimated_parameter_count`]
/// to inspect the count for a resolved config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncopateParameterBudget {
    Params1M,
    Params5M,
    Params10M,
    Params50M,
    Params100M,
}

impl SyncopateParameterBudget {
    pub const ALL: [Self; 5] = [
        Self::Params1M,
        Self::Params5M,
        Self::Params10M,
        Self::Params50M,
        Self::Params100M,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Params1M => "1M",
            Self::Params5M => "5M",
            Self::Params10M => "10M",
            Self::Params50M => "50M",
            Self::Params100M => "100M",
        }
    }

    pub fn target_parameter_count(self) -> usize {
        match self {
            Self::Params1M => 1_000_000,
            Self::Params5M => 5_000_000,
            Self::Params10M => 10_000_000,
            Self::Params50M => 50_000_000,
            Self::Params100M => 100_000_000,
        }
    }
}

/// Attention kernel used inside each decoder block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionKernel {
    /// GPT-1/Llama style causal softmax attention. This is the training-safe
    /// default for tiny NPC chat models.
    #[default]
    Softmax,
    /// Normalized second-order causal attention inspired by Higher-order Linear
    /// Attention. Implemented in the exact parallel form for small contexts.
    HigherOrder,
}

/// Decoder-only model configuration for syncopate-machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyncopateModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Context length.
    pub seq_len: usize,
    /// Number of decoder blocks.
    pub layers: usize,
    /// Action-ID embedding and residual width.
    pub d_model: usize,
    /// Query heads.
    pub attention_heads: usize,
    /// Shared key/value heads. Use fewer than `attention_heads` for GQA/MQA.
    pub kv_heads: usize,
    /// SwiGLU hidden width.
    pub intermediate_size: usize,
    /// Causal attention kernel.
    #[serde(default)]
    pub attention_kernel: AttentionKernel,
    /// RoPE base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl SyncopateModelConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            seq_len: 64,
            layers: 2,
            d_model: 96,
            attention_heads: 4,
            kv_heads: 1,
            intermediate_size: 256,
            attention_kernel: AttentionKernel::Softmax,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            vocab_size: 32,
            seq_len: 8,
            layers: 1,
            d_model: 16,
            attention_heads: 4,
            kv_heads: 1,
            intermediate_size: 32,
            attention_kernel: AttentionKernel::Softmax,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    pub fn for_parameter_budget(
        budget: SyncopateParameterBudget,
        vocab_size: usize,
        seq_len: usize,
    ) -> Self {
        match budget {
            SyncopateParameterBudget::Params1M => Self::preset_1m(vocab_size, seq_len),
            SyncopateParameterBudget::Params5M => Self::preset_5m(vocab_size, seq_len),
            SyncopateParameterBudget::Params10M => Self::preset_10m(vocab_size, seq_len),
            SyncopateParameterBudget::Params50M => Self::preset_50m(vocab_size, seq_len),
            SyncopateParameterBudget::Params100M => Self::preset_100m(vocab_size, seq_len),
        }
    }

    pub fn preset_1m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 2, 96, 4, 1, 256)
    }

    pub fn preset_action(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 2, 64, 4, 1, 128)
    }

    pub fn preset_5m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 8, 192, 6, 2, 512)
    }

    pub fn preset_10m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 10, 256, 8, 2, 704)
    }

    pub fn preset_50m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 16, 512, 8, 2, 1408)
    }

    pub fn preset_100m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 15, 768, 12, 4, 2048)
    }

    pub fn with_attention_kernel(mut self, kernel: AttentionKernel) -> Self {
        self.attention_kernel = kernel;
        self
    }

    pub fn estimated_parameter_count(&self) -> usize {
        let head_dim = self.head_dim();
        let kv_width = self.kv_heads.saturating_mul(head_dim);
        let embedding_params = self.vocab_size.saturating_mul(self.d_model);
        let attention_params = self
            .d_model
            .saturating_mul(self.d_model)
            .saturating_add(2usize.saturating_mul(self.d_model).saturating_mul(kv_width))
            .saturating_add(self.d_model.saturating_mul(self.d_model));
        let ffn_params = 3usize
            .saturating_mul(self.d_model)
            .saturating_mul(self.intermediate_size);
        let norm_params = 2usize.saturating_mul(self.d_model);
        let per_layer = attention_params
            .saturating_add(ffn_params)
            .saturating_add(norm_params);
        embedding_params
            .saturating_add(self.layers.saturating_mul(per_layer))
            .saturating_add(self.d_model)
    }

    pub(crate) fn from_dimensions(
        vocab_size: usize,
        seq_len: usize,
        layers: usize,
        d_model: usize,
        attention_heads: usize,
        kv_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            vocab_size,
            seq_len,
            layers,
            d_model,
            attention_heads,
            kv_heads,
            intermediate_size,
            attention_kernel: AttentionKernel::Softmax,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.attention_heads.max(1)
    }

    pub fn validate(&self) -> Result<()> {
        ensure(self.vocab_size > 0, "vocab_size must be greater than zero")?;
        ensure(self.seq_len > 0, "seq_len must be greater than zero")?;
        ensure(self.layers > 0, "layers must be greater than zero")?;
        ensure(self.d_model > 0, "d_model must be greater than zero")?;
        ensure(
            self.attention_heads > 0,
            "attention_heads must be greater than zero",
        )?;
        ensure(self.kv_heads > 0, "kv_heads must be greater than zero")?;
        ensure(
            self.d_model.is_multiple_of(self.attention_heads),
            "d_model must be divisible by attention_heads",
        )?;
        ensure(
            self.attention_heads.is_multiple_of(self.kv_heads),
            "attention_heads must be divisible by kv_heads",
        )?;
        ensure(
            self.intermediate_size > 0,
            "intermediate_size must be greater than zero",
        )?;
        ensure(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta must be positive and finite",
        )?;
        ensure(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps must be positive and finite",
        )?;
        Ok(())
    }
}

/// Training options for action-ID sequence training.
#[derive(Clone, Debug)]
pub struct ModelTrainingConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub min_learning_rate: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_clip_norm: Option<f64>,
    pub pad_action_id: u32,
    /// Directory to save checkpoints into. When `None`, no checkpoints are
    /// saved during training.
    pub checkpoint_dir: Option<String>,
    /// Save a checkpoint every N steps. Only used when `checkpoint_dir` is
    /// `Some`. A value of `0` disables periodic snapshots (only `best.mpk`
    /// is kept).
    pub checkpoint_interval: usize,
}

impl Default for ModelTrainingConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 4,
            learning_rate: 3e-4,
            min_learning_rate: 3e-5,
            warmup_steps: 200,
            weight_decay: 0.1,
            grad_clip_norm: Some(1.0),
            pad_action_id: 0,
            checkpoint_dir: None,
            checkpoint_interval: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelTrainingReport {
    pub steps: usize,
    pub final_loss: f32,
    /// The lowest loss observed across all training steps.
    pub best_loss: f32,
    /// The step at which `best_loss` was recorded.
    pub best_loss_step: usize,
    pub training_window_count: usize,
    pub parameter_count: usize,
}

/// Result of evaluating a model on held-out sequences.
#[derive(Clone, Debug)]
pub struct EvaluationResult {
    /// Average cross-entropy loss across all batches.
    pub loss: f32,
    /// Perplexity = exp(average_loss).
    pub perplexity: f32,
    /// Fraction of action IDs where argmax(logits) == target.
    pub accuracy: f64,
    /// Number of batches evaluated.
    pub num_batches: usize,
    /// Total number of unmasked action IDs evaluated.
    pub total_predictions: usize,
}

/// Burn-backed decoder-only transformer for action-ID prediction.
#[derive(Module, Debug)]
pub struct SyncopateModel<B: Backend = DefaultAutodiffBackend> {
    #[module(skip)]
    config: SyncopateModelConfig,
    token_embedding: Param<Tensor<B, 2>>,
    layers: Vec<SyncopateBlock<B>>,
    final_norm: RmsNorm<B>,
}

/// Convenience alias for the default Burn Flex autodiff model.
pub type DefaultSyncopateModel = SyncopateModel<DefaultAutodiffBackend>;

#[derive(Module, Debug)]
struct SyncopateBlock<B: Backend> {
    attention_norm: RmsNorm<B>,
    attention: CausalAttention<B>,
    ffn_norm: RmsNorm<B>,
    ffn: SwiGluFeedForward<B>,
}

#[derive(Module, Debug)]
struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    eps: f32,
}

#[derive(Module, Debug)]
struct CausalAttention<B: Backend> {
    w_q: Param<Tensor<B, 2>>,
    w_k: Param<Tensor<B, 2>>,
    w_v: Param<Tensor<B, 2>>,
    w_o: Param<Tensor<B, 2>>,
    #[module(skip)]
    attention_heads: usize,
    #[module(skip)]
    kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    kernel: AttentionKernel,
    #[module(skip)]
    rope_theta: f32,
}

#[derive(Module, Debug)]
struct SwiGluFeedForward<B: Backend> {
    w_gate: Param<Tensor<B, 2>>,
    w_up: Param<Tensor<B, 2>>,
    w_down: Param<Tensor<B, 2>>,
}

impl<B: Backend> SyncopateModel<B> {
    pub fn new(config: SyncopateModelConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;

        let mut seed = 0x5359_4e43_4f50_4154;
        let token_embedding =
            init_matrix(config.vocab_size, config.d_model, 0.02, &mut seed, device);
        let mut layers = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            layers.push(SyncopateBlock::new(&config, &mut seed, device));
        }
        let final_norm = RmsNorm::new(config.d_model, config.rms_norm_eps, device);

        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
        })
    }

    pub fn config(&self) -> &SyncopateModelConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        self.num_params()
    }

    pub fn forward(&self, action_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = action_ids.dims();
        // Use `select` instead of `one_hot` + matmul. Burn's `one_hot` calls
        // `into_scalar()` internally for bounds validation, which panics on WASM
        // because synchronous blocking futures are unsupported there.
        let indices = action_ids.reshape([batch * seq_len]);
        let mut x = self.token_embedding.val().select(0, indices).reshape([
            batch,
            seq_len,
            self.config().d_model,
        ]);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let x = self.final_norm.forward(x);
        linear3(x, self.token_embedding.val().swap_dims(0, 1))
    }

    pub fn save_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(path.as_ref().to_path_buf(), &recorder)
            .map_err(|err| Error::Serialization(err.to_string()))
    }

    pub fn load_parameters(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let device =
            self.devices().into_iter().next().ok_or_else(|| {
                Error::Serialization("model has no device for parameter load".into())
            })?;
        let loaded = self
            .clone()
            .load_file(path.as_ref().to_path_buf(), &recorder, &device)
            .map_err(|err| Error::Serialization(err.to_string()))?;
        *self = loaded;
        Ok(())
    }

    /// Run a forward pass and return the full logit tensor.
    pub fn forward_logits(
        &self,
        context: &[u32],
        pad_action_id: u32,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>> {
        if context.is_empty() {
            return Err(Error::Inference(
                "context must contain at least one action id".to_string(),
            ));
        }

        let input = context_window(context, self.config().seq_len, pad_action_id);
        let input = tensor_from_u32::<B, 2>(input, [1, self.config().seq_len], device)?;
        Ok(self.forward(input))
    }

    /// Evaluates the model on action-ID sequences.
    pub fn evaluate_on_sequences(
        &self,
        sequences: &[Vec<u32>],
        seq_len: usize,
        batch_size: usize,
        pad_action_id: u32,
        device: &B::Device,
    ) -> Result<EvaluationResult> {
        let windows = TrainingWindows::from_sequences(sequences, seq_len, pad_action_id)?;
        if windows.is_empty() {
            return Ok(EvaluationResult {
                loss: f32::NAN,
                perplexity: f32::NAN,
                accuracy: 0.0,
                num_batches: 0,
                total_predictions: 0,
            });
        }

        let num_batches = windows.len().div_ceil(batch_size);
        let mut total_loss = 0.0_f64;
        let mut total_correct = 0_usize;
        let mut total_predictions = 0_usize;

        for step in 0..num_batches {
            let batch = windows.batch::<B>(step, batch_size, device)?;
            let logits = self.forward(batch.inputs);
            let loss = cross_entropy_loss_with_mask(
                logits.clone(),
                batch.targets.clone(),
                batch.loss_mask.clone(),
            );
            total_loss += tensor_scalar(loss)? as f64;

            let [b, s, v] = logits.dims();
            let logits_vec = logits
                .reshape([b * s * v])
                .into_data()
                .into_vec::<f32>()
                .map_err(|err| Error::Training(err.to_string()))?;
            let targets_vec = batch
                .targets
                .reshape([b * s])
                .into_data()
                .into_vec::<i32>()
                .map_err(|err| Error::Training(err.to_string()))?;
            let mask_vec = batch
                .loss_mask
                .reshape([b * s])
                .into_data()
                .into_vec::<f32>()
                .map_err(|err| Error::Training(err.to_string()))?;

            for position_idx in 0..(b * s) {
                if mask_vec[position_idx] <= 0.0 {
                    continue;
                }
                let start = position_idx * v;
                let end = start + v;
                let predicted = argmax(&logits_vec[start..end])?;
                if predicted as i32 == targets_vec[position_idx] {
                    total_correct += 1;
                }
                total_predictions += 1;
            }
        }

        let loss = (total_loss / num_batches as f64) as f32;
        let perplexity = loss.exp();
        let accuracy = if total_predictions == 0 {
            0.0
        } else {
            total_correct as f64 / total_predictions as f64
        };

        Ok(EvaluationResult {
            loss,
            perplexity,
            accuracy,
            num_batches,
            total_predictions,
        })
    }
}

impl<B> SyncopateModel<B>
where
    B: AutodiffBackend,
{
    /// Trains this model directly on action-ID sequences.
    pub fn train_action_sequences(
        &mut self,
        sequences: &[Vec<u32>],
        training: &ModelTrainingConfig,
        device: &B::Device,
        mut on_step: impl FnMut(usize, f32),
    ) -> Result<ModelTrainingReport> {
        if training.batch_size == 0 {
            return Err(Error::Training(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        let windows = TrainingWindows::from_sequences(
            sequences,
            self.config().seq_len,
            training.pad_action_id,
        )?;
        if windows.is_empty() {
            return Err(Error::Training(
                "training requires at least one sequence with two or more IDs".to_string(),
            ));
        }

        let mut optimizer_config =
            AdamWConfig::new().with_weight_decay(training.weight_decay as f32);
        if let Some(max_norm) = training.grad_clip_norm.filter(|value| *value > 0.0) {
            optimizer_config = optimizer_config
                .with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm as f32)));
        }
        let mut optimizer = optimizer_config.init::<B, Self>();
        let mut model = self.clone();
        let mut final_loss = f32::NAN;
        let mut best_loss = f32::MAX;
        let mut best_loss_step: usize = 0;

        let ckpt_dir = training.checkpoint_dir.as_deref().map(Path::new);
        if let Some(dir) = &ckpt_dir {
            std::fs::create_dir_all(dir).map_err(|e| {
                Error::Io(format!(
                    "failed to create checkpoint directory {:?}: {}",
                    dir, e
                ))
            })?;
        }

        for step in 0..training.steps {
            let batch = windows.batch::<B>(step, training.batch_size, device)?;
            let logits = model.forward(batch.inputs);
            let loss = cross_entropy_loss_with_mask(logits, batch.targets, batch.loss_mask);
            final_loss = tensor_scalar(loss.clone())?;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            let lr = scheduled_learning_rate(training, step);
            model = optimizer.step(lr, model, grads);

            if final_loss < best_loss {
                best_loss = final_loss;
                best_loss_step = step;
                if let Some(dir) = &ckpt_dir {
                    let path = dir.join("best.mpk");
                    model.save_parameters(&path)?;
                }
            }
            if training.checkpoint_interval > 0
                && (step + 1) % training.checkpoint_interval == 0
                && let Some(dir) = &ckpt_dir
            {
                let path = dir.join(format!("step_{:06}.mpk", step + 1));
                model.save_parameters(&path)?;
            }

            on_step(step, final_loss);
        }

        if training.steps == 0 {
            let batch = windows.batch::<B>(0, training.batch_size, device)?;
            final_loss = tensor_scalar(cross_entropy_loss_with_mask(
                model.forward(batch.inputs),
                batch.targets,
                batch.loss_mask,
            ))?;
            best_loss = final_loss;
            best_loss_step = 0;
        }

        *self = model;

        Ok(ModelTrainingReport {
            steps: training.steps,
            final_loss,
            best_loss,
            best_loss_step,
            training_window_count: windows.len(),
            parameter_count: self.parameter_count(),
        })
    }
}
impl<B: Backend> SyncopateBlock<B> {
    fn new(config: &SyncopateModelConfig, seed: &mut u64, device: &B::Device) -> Self {
        Self {
            attention_norm: RmsNorm::new(config.d_model, config.rms_norm_eps, device),
            attention: CausalAttention::new(config, seed, device),
            ffn_norm: RmsNorm::new(config.d_model, config.rms_norm_eps, device),
            ffn: SwiGluFeedForward::new(config, seed, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attention_update = self
            .attention
            .forward(self.attention_norm.forward(x.clone()));
        let x = x + attention_update;
        let ffn_update = self.ffn.forward(self.ffn_norm.forward(x.clone()));
        x + ffn_update
    }
}

impl<B: Backend> RmsNorm<B> {
    fn new(width: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            weight: init_vector(width, 1.0, device),
            eps,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, width] = x.dims();
        let denom = x.clone().square().mean_dim(2).add_scalar(self.eps).sqrt();
        let weight = self
            .weight
            .val()
            .unsqueeze::<3>()
            .expand([batch, seq_len, width]);
        (x / denom) * weight
    }
}

impl<B: Backend> CausalAttention<B> {
    fn new(config: &SyncopateModelConfig, seed: &mut u64, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
        let kv_width = config.kv_heads * head_dim;
        let residual_std = 0.02 / (2.0 * config.layers as f32).sqrt();
        Self {
            w_q: init_matrix(config.d_model, config.d_model, 0.02, seed, device),
            w_k: init_matrix(config.d_model, kv_width, 0.02, seed, device),
            w_v: init_matrix(config.d_model, kv_width, 0.02, seed, device),
            w_o: init_matrix(config.d_model, config.d_model, residual_std, seed, device),
            attention_heads: config.attention_heads,
            kv_heads: config.kv_heads,
            head_dim,
            kernel: config.attention_kernel,
            rope_theta: config.rope_theta,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        let q = linear3(x.clone(), self.w_q.val())
            .reshape([batch, seq_len, self.attention_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = linear3(x.clone(), self.w_k.val())
            .reshape([batch, seq_len, self.kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = linear3(x, self.w_v.val())
            .reshape([batch, seq_len, self.kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let q = apply_rope(q, self.rope_theta);
        let k = repeat_kv_heads(apply_rope(k, self.rope_theta), self.attention_heads);
        let v = repeat_kv_heads(v, self.attention_heads);

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (self.head_dim as f32).sqrt());
        let mixed = match self.kernel {
            AttentionKernel::Softmax => softmax_attention(scores, v),
            AttentionKernel::HigherOrder => higher_order_attention(scores, v),
        };

        let merged = mixed.swap_dims(1, 2).reshape([batch, seq_len, d_model]);
        linear3(merged, self.w_o.val())
    }
}

impl<B: Backend> SwiGluFeedForward<B> {
    fn new(config: &SyncopateModelConfig, seed: &mut u64, device: &B::Device) -> Self {
        let residual_std = 0.02 / (2.0 * config.layers as f32).sqrt();
        Self {
            w_gate: init_matrix(config.d_model, config.intermediate_size, 0.02, seed, device),
            w_up: init_matrix(config.d_model, config.intermediate_size, 0.02, seed, device),
            w_down: init_matrix(
                config.intermediate_size,
                config.d_model,
                residual_std,
                seed,
                device,
            ),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(linear3(x.clone(), self.w_gate.val()));
        let up = linear3(x, self.w_up.val());
        linear3(gate * up, self.w_down.val())
    }
}

fn softmax_attention<B: Backend>(scores: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, seq_len, _] = scores.dims();
    let mask = causal_invalid_mask_tensor::<B>(seq_len, &scores.device())
        .unsqueeze::<4>()
        .expand([batch, heads, seq_len, seq_len]);
    let weights = activation::softmax(scores.mask_fill(mask, NEG_INF), 3);
    weights.matmul(v)
}

fn higher_order_attention<B: Backend>(scores: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, seq_len, _] = scores.dims();
    let keep = causal_keep_tensor::<B>(seq_len, &scores.device())
        .unsqueeze::<4>()
        .expand([batch, heads, seq_len, seq_len]);
    let w = scores * keep.clone();
    let second = w.clone().matmul(w.swap_dims(2, 3)) * keep;
    let norm = second.clone().abs().sum_dim(3).add_scalar(EPS);
    second.matmul(v) / norm
}

fn apply_rope<B: Backend>(x: Tensor<B, 4>, theta: f32) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = x.dims();
    let half = head_dim / 2;
    if half == 0 {
        return x;
    }

    let positions = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..seq_len).map(|idx| idx as f32).collect::<Vec<_>>(),
            [1, 1, seq_len, 1],
        ),
        &x.device(),
    )
    .expand([batch, heads, seq_len, 1]);

    let mut parts = Vec::with_capacity(head_dim);
    for pair_idx in 0..half {
        let inv_freq = theta.powf(-(2.0 * pair_idx as f32) / head_dim as f32);
        let angle = positions.clone().mul_scalar(inv_freq);
        let cos = angle.clone().cos();
        let sin = angle.sin();
        let even = x.clone().narrow(3, pair_idx * 2, 1);
        let odd = x.clone().narrow(3, pair_idx * 2 + 1, 1);
        parts.push(even.clone() * cos.clone() - odd.clone() * sin.clone());
        parts.push(even * sin + odd * cos);
    }
    if head_dim % 2 == 1 {
        parts.push(x.narrow(3, head_dim - 1, 1));
    }
    Tensor::cat(parts, 3)
}

fn repeat_kv_heads<B: Backend>(x: Tensor<B, 4>, attention_heads: usize) -> Tensor<B, 4> {
    let [_batch, kv_heads, _seq_len, _head_dim] = x.dims();
    if kv_heads == attention_heads {
        return x;
    }
    let repeats = attention_heads / kv_heads;
    let mut chunks = Vec::with_capacity(kv_heads);
    for head in 0..kv_heads {
        chunks.push(x.clone().narrow(1, head, 1).repeat_dim(1, repeats));
    }
    Tensor::cat(chunks, 1)
}

#[allow(dead_code)]
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let device = logits.device();
    let [batch, seq_len, _] = logits.dims();
    let loss_mask = Tensor::<B, 2>::ones([batch, seq_len], &device);
    cross_entropy_loss_with_mask(logits, targets, loss_mask)
}

pub fn cross_entropy_loss_with_mask<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab_size] = logits.dims();
    let position_count = batch * seq_len;
    let flat_logits = logits.reshape([position_count, vocab_size]);
    let flat_targets = targets.reshape([position_count]);
    let flat_mask = loss_mask.reshape([position_count]);
    let log_probs = activation::log_softmax(flat_logits, 1);
    let target_probs = flat_targets.one_hot::<2>(vocab_size).float();
    let picked = (log_probs * target_probs)
        .sum_dim(1)
        .reshape([position_count]);
    let masked_nll = (picked.neg() * flat_mask.clone()).sum();
    let mask_sum = flat_mask.sum();
    let denom = mask_sum.add_scalar(EPS);
    masked_nll / denom
}

fn causal_invalid_mask_tensor<B: Backend>(
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, burn::tensor::Bool> {
    let mut distances = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            distances.push(j as f32 - i as f32);
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(distances, [seq_len, seq_len]), device)
        .greater_elem(0.0)
}

fn causal_keep_tensor<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut distances = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            distances.push(j as f32 - i as f32);
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(distances, [seq_len, seq_len]), device)
        .lower_equal_elem(0.0)
        .float()
}

fn linear3<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, _seq_len, _in_dim] = x.dims();
    let [weight_in, out_dim] = weight.dims();
    x.matmul(weight.unsqueeze::<3>().expand([batch, weight_in, out_dim]))
}

fn init_vector<B: Backend>(len: usize, value: f32, device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::new(vec![value; len], [len]),
        device,
    ))
}

fn init_matrix<B: Backend>(
    rows: usize,
    cols: usize,
    std: f32,
    seed: &mut u64,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let values = gaussian_values(rows * cols, std, seed);
    Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(values, [rows, cols]),
        device,
    ))
}

fn gaussian_values(len: usize, std: f32, seed: &mut u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u1 = next_uniform(seed).max(1e-7);
        let u2 = next_uniform(seed);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        values.push(radius * theta.cos() * std);
        if values.len() < len {
            values.push(radius * theta.sin() * std);
        }
    }
    values
}

fn next_uniform(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f32 + 1.0) / ((1u32 << 24) as f32 + 2.0)
}

fn scheduled_learning_rate(training: &ModelTrainingConfig, step: usize) -> f64 {
    let max_lr = training.learning_rate.max(0.0);
    let min_lr = training.min_learning_rate.max(0.0).min(max_lr);
    if max_lr <= 0.0 {
        return 0.0;
    }
    if training.warmup_steps > 0 && step < training.warmup_steps {
        return max_lr * (step + 1) as f64 / training.warmup_steps as f64;
    }

    let decay_steps = training.steps.saturating_sub(training.warmup_steps).max(1);
    let elapsed = step.saturating_sub(training.warmup_steps).min(decay_steps);
    let progress = elapsed as f64 / decay_steps as f64;
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (PI64 * progress).cos())
}

fn context_window(context: &[u32], seq_len: usize, pad_action_id: u32) -> Vec<u32> {
    let mut input = vec![pad_action_id; seq_len];
    let suffix = if context.len() > seq_len {
        &context[context.len() - seq_len..]
    } else {
        context
    };
    input[..suffix.len()].copy_from_slice(suffix);
    input
}

fn argmax(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::Inference("cannot argmax empty logits".to_string()))
}

fn tensor_scalar<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|err| Error::Training(err.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Training("expected scalar tensor".to_string()))
}

fn tensor_from_u32<B: Backend, const D: usize>(
    values: Vec<u32>,
    shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D, Int>> {
    let values = values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| Error::Config(format!("action id {value} exceeds i32::MAX")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::<B, D, Int>::from_data(
        TensorData::new(values, shape),
        device,
    ))
}

fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Config(message.to_string()))
    }
}

struct TrainingWindow {
    inputs: Vec<u32>,
    targets: Vec<u32>,
    loss_mask: Vec<f32>,
}

struct TrainingWindows {
    windows: Vec<TrainingWindow>,
    seq_len: usize,
}

impl TrainingWindows {
    fn from_sequences(sequences: &[Vec<u32>], seq_len: usize, pad_action_id: u32) -> Result<Self> {
        let mut windows = Vec::new();
        for sequence in sequences {
            if sequence.len() < 2 {
                continue;
            }

            let mut start = 0;
            while start + 1 < sequence.len() {
                let end = (start + seq_len + 1).min(sequence.len());
                let chunk = &sequence[start..end];
                let prediction_count = chunk.len() - 1;

                let mut inputs = vec![pad_action_id; seq_len];
                let mut targets = vec![pad_action_id; seq_len];
                let mut loss_mask = vec![0.0; seq_len];
                inputs[..prediction_count].copy_from_slice(&chunk[..prediction_count]);
                targets[..prediction_count].copy_from_slice(&chunk[1..]);
                loss_mask[..prediction_count].fill(1.0);

                windows.push(TrainingWindow {
                    inputs,
                    targets,
                    loss_mask,
                });

                if end == sequence.len() {
                    break;
                }
                start += seq_len;
            }
        }

        Ok(Self { windows, seq_len })
    }

    fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    fn len(&self) -> usize {
        self.windows.len()
    }

    fn batch<B: Backend>(
        &self,
        step: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<ActionBatch<B>> {
        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);
        let mut loss_mask = Vec::with_capacity(batch_size * self.seq_len);

        for batch_idx in 0..batch_size {
            let index = (step * batch_size + batch_idx) % self.windows.len();
            let window = &self.windows[index];
            inputs.extend_from_slice(&window.inputs);
            targets.extend_from_slice(&window.targets);
            loss_mask.extend_from_slice(&window.loss_mask);
        }

        Ok(ActionBatch {
            inputs: tensor_from_u32(inputs, [batch_size, self.seq_len], device)?,
            targets: tensor_from_u32(targets, [batch_size, self.seq_len], device)?,
            loss_mask: Tensor::<B, 2>::from_data(
                TensorData::new(loss_mask, [batch_size, self.seq_len]),
                device,
            ),
        })
    }
}

struct ActionBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Tensor<B, 2>,
}

#[cfg(test)]
#[allow(dead_code)]
pub fn make_batch<B: Backend>(
    step: usize,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);

    for batch in 0..batch_size {
        let offset = (step * 7 + batch * 13) % vocab_size;
        for pos in 0..seq_len {
            let action = ((offset + pos) % vocab_size) as u32;
            let next = ((offset + pos + 1) % vocab_size) as u32;
            inputs.push(action);
            targets.push(next);
        }
    }

    Ok((
        tensor_from_u32(inputs, [batch_size, seq_len], device)?,
        tensor_from_u32(targets, [batch_size, seq_len], device)?,
    ))
}
