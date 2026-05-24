//! Browser-side inference runtime for the Syncopate model.
//!
//! This module provides [`BrowserRuntime`] — a plain Rust struct (no
//! `wasm_bindgen` annotations) that the Leptos frontend can call
//! directly because both compile into the same WASM binary.
//!
//! The runtime loads a `.mpk` checkpoint over HTTP, deserialises weights
//! via Burn's `NamedMpkBytesRecorder`, and exposes a [`step`](BrowserRuntime::step)
//! method for ID-by-ID generation.
//!

use crate::model::{AttentionKernel, SyncopateModel, SyncopateModelConfig};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::backend::BackendTypes,
};
use wasm_bindgen::JsValue;

/// Lazy WebGPU device initialisation.
///
/// The CubeCL/WGPU runtime must be bootstrapped before any tensor
/// operation. On WASM this is the *only* way to create the underlying
/// wgpu adapter/device/queue because the synchronous `block_on` path
/// is unavailable.
///
/// We create the wgpu adapter ourselves (with `force_fallback_adapter`
/// as a last resort) instead of delegating to
/// `burn::backend::wgpu::init_setup_async` which panics on adapter
/// failure without useful diagnostics.
static WEBGPU_DEVICE: std::sync::OnceLock<
    std::sync::Mutex<Option<burn::backend::wgpu::WgpuDevice>>,
> = std::sync::OnceLock::new();

async fn ensure_wgpu_device() -> Result<burn::backend::wgpu::WgpuDevice, String> {
    // Fast path: already initialised.
    {
        let guard = WEBGPU_DEVICE
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .map_err(|e| format!("device lock: {e}"))?;

        if let Some(dev) = guard.clone() {
            return Ok(dev);
        }
    }

    // Slow path: build the wgpu stack ourselves.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    // Try high-performance first, then fallback adapter.
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    {
        Ok(a) => a,
        Err(_) => instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: None,
            })
            .await
            .map_err(|e| {
                format!(
                    "WebGPU adapter not available ({e}). \
                     Please use Chrome 113+, Edge 113+, or another \
                     WebGPU-capable browser."
                )
            })?,
    };

    let info = adapter.get_info();

    // Start from downlevel defaults but bump compute workgroup limits
    // to match what the adapter actually supports. CubeCL kernels use
    // workgroup sizes up to 1024, but downlevel_webgl2_defaults caps
    // max_compute_invocations_per_workgroup at 256, which causes every
    // compute pipeline to fail on browsers that support higher limits.
    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
    limits.max_compute_invocations_per_workgroup =
        adapter_limits.max_compute_invocations_per_workgroup;
    limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
    limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
    limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_buffer_size;
    limits.max_compute_workgroups_per_dimension =
        adapter_limits.max_compute_workgroups_per_dimension;
    limits.max_storage_buffers_per_shader_stage =
        adapter_limits.max_storage_buffers_per_shader_stage;
    limits.max_bind_groups = adapter_limits.max_bind_groups;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("syncopate-inference"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| format!("wgpu request_device failed: {e}"))?;

    let backend = info.backend;
    let setup = burn::backend::wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    };

    let device = burn::backend::wgpu::init_device(setup, Default::default());

    let mut guard = WEBGPU_DEVICE
        .get()
        .expect("just initialised")
        .lock()
        .map_err(|e| format!("device lock: {e}"))?;
    *guard = Some(device.clone());

    Ok(device)
}

// ---------------------------------------------------------------------------
// Backend aliases
// ---------------------------------------------------------------------------

/// GPU inference backend: Burn Wgpu → WebGPU in the browser.
type WgpuBackend = burn::backend::Wgpu;

/// CPU fallback backend. This is slower, but it keeps Chrome profiles without a
/// usable WebGPU adapter from turning the whole chat box into a brick.
type CpuBackend = burn::backend::Flex;

enum BrowserModel {
    Wgpu {
        model: SyncopateModel<WgpuBackend>,
        device: <WgpuBackend as BackendTypes>::Device,
    },
    Cpu {
        model: SyncopateModel<CpuBackend>,
        device: <CpuBackend as BackendTypes>::Device,
    },
}

impl BrowserModel {
    fn label(&self) -> &'static str {
        match self {
            Self::Wgpu { .. } => "webgpu",
            Self::Cpu { .. } => "cpu",
        }
    }
}

// ---------------------------------------------------------------------------
// BrowserRuntime
// ---------------------------------------------------------------------------

/// Browser inference backend preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserBackendPreference {
    /// Try WebGPU first, then fall back to CPU when no usable adapter exists.
    Auto,
    /// Require WebGPU. This surfaces adapter/device errors to the caller.
    WebGpu,
    /// Force the CPU/Flex backend.
    Cpu,
}

impl BrowserBackendPreference {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::WebGpu => "webgpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Status of the browser-side model runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStatus {
    /// Not loaded yet.
    Idle,
    /// Currently fetching the checkpoint.
    Loading,
    /// Ready to generate tokens.
    Ready,
    /// A previous operation failed.
    Error,
}

/// Owns the neural model and runs inference on WebGPU inside the browser.
///
/// Usage from Leptos (all inside the same WASM binary):
///
/// ```rust,ignore
/// let mut rt = BrowserRuntime::new();
/// rt.load_from_url("./model.mpk", 8192, 128).await.unwrap();
/// let logits = rt.step(&prompt_ids).unwrap();
/// // sample from logits, map IDs to your app actions, repeat...
/// ```
pub struct BrowserRuntime {
    model: Option<BrowserModel>,
    config: Option<SyncopateModelConfig>,
    status: RuntimeStatus,
    last_error: Option<String>,
    backend_label: &'static str,
}

impl BrowserRuntime {
    /// Create an empty runtime. Call [`Self::load_from_url`] next.
    pub fn new() -> Self {
        Self {
            model: None,
            config: None,
            status: RuntimeStatus::Idle,
            last_error: None,
            backend_label: "none",
        }
    }

    /// Fetch a `.mpk` checkpoint from `url`, deserialise weights, and
    /// prepare the model for inference.
    ///
    /// `vocab_size` and `seq_len` must match the training config.
    /// For the Dancing With My Code 1M preset they are 8192 and 128.
    pub async fn load_from_url(
        &mut self,
        url: &str,
        vocab_size: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        self.load_from_url_with_preference(url, vocab_size, seq_len, BrowserBackendPreference::Auto)
            .await
    }

    /// Fetch a `.mpk` checkpoint and load it with a concrete backend
    /// preference. Use this when the browser has a WebGPU adapter but the
    /// driver is unstable and the UI needs an explicit CPU escape hatch.
    pub async fn load_from_url_with_preference(
        &mut self,
        url: &str,
        vocab_size: usize,
        seq_len: usize,
        preference: BrowserBackendPreference,
    ) -> Result<(), String> {
        self.load_from_url_with_config_json(url, None, vocab_size, seq_len, preference)
            .await
    }

    /// Like [`load_from_url_with_preference`] but also loads a `config.json`
    /// file that specifies the attention kernel and other training-time
    /// settings. When `config_url` is `None`, falls back to the 1M preset
    /// with softmax attention.
    pub async fn load_from_url_with_config_json(
        &mut self,
        url: &str,
        config_url: Option<&str>,
        vocab_size: usize,
        seq_len: usize,
        preference: BrowserBackendPreference,
    ) -> Result<(), String> {
        self.status = RuntimeStatus::Loading;
        self.last_error = None;

        // 1. Build model config. If a config.json URL is provided, parse it
        // to pick up the correct attention kernel. Otherwise use the 1M preset.
        let config = if let Some(cfg_url) = config_url {
            match fetch_config_json(cfg_url).await {
                Ok(cfg) => cfg,
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("config.json load failed ({e}), using preset").into(),
                    );
                    default_browser_config(vocab_size, seq_len)
                }
            }
        } else {
            default_browser_config(vocab_size, seq_len)
        };

        // 2. Fetch checkpoint bytes before backend selection so WebGPU failure
        // can fall back to CPU without a second network request.
        let bytes = fetch_bytes(url).await.map_err(|e| {
            self.status = RuntimeStatus::Error;
            self.last_error = Some(format!("{e:?}"));
            format!("{e:?}")
        })?;

        // 3. Prefer WebGPU by default. If Chrome has WebGPU disabled, blocked,
        // or no suitable adapter, fall back to CPU instead of failing the
        // component. Explicit WebGPU keeps the error visible; explicit CPU
        // avoids broken Chrome driver paths entirely.
        let loaded = match preference {
            BrowserBackendPreference::Cpu => Self::load_cpu_model(bytes, &config).map_err(|e| {
                let msg = format!("CPU model load failed ({e})");
                self.status = RuntimeStatus::Error;
                self.last_error = Some(msg.clone());
                msg
            })?,
            BrowserBackendPreference::WebGpu => {
                Self::load_wgpu_model(bytes, &config).await.map_err(|e| {
                    let msg = format!("WebGPU model load failed ({e})");
                    self.status = RuntimeStatus::Error;
                    self.last_error = Some(msg.clone());
                    msg
                })?
            }
            BrowserBackendPreference::Auto => {
                match Self::load_wgpu_model(bytes.clone(), &config).await {
                    Ok(model) => model,
                    Err(wgpu_error) => {
                        web_sys::console::warn_1(
                            &format!("webgpu unavailable, falling back to cpu: {wgpu_error}")
                                .into(),
                        );
                        Self::load_cpu_model(bytes, &config).map_err(|cpu_error| {
                            let msg = format!(
                                "WebGPU failed ({wgpu_error}); CPU fallback failed ({cpu_error})"
                            );
                            self.status = RuntimeStatus::Error;
                            self.last_error = Some(msg.clone());
                            msg
                        })?
                    }
                }
            }
        };

        self.backend_label = loaded.label();
        self.model = Some(loaded);
        self.config = Some(config);
        self.status = RuntimeStatus::Ready;

        Ok(())
    }

    async fn load_wgpu_model(
        bytes: Vec<u8>,
        config: &SyncopateModelConfig,
    ) -> Result<BrowserModel, String> {
        // Bootstrap the WebGPU runtime. On WASM the wgpu adapter/device must be
        // created asynchronously before any CubeCL operation.
        let device = ensure_wgpu_device().await?;

        // Use WgpuBackend directly — avoids compiling autodiff shaders which
        // doubles load time for no benefit at inference time.
        let mut model = SyncopateModel::<WgpuBackend>::new(config.clone(), &device)
            .map_err(|e| e.to_string())?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(bytes, &device)
            .map_err(|e| format!("checkpoint load error: {e:?}"))?;
        model = model.load_record(record);

        Ok(BrowserModel::Wgpu { model, device })
    }

    fn load_cpu_model(
        bytes: Vec<u8>,
        config: &SyncopateModelConfig,
    ) -> Result<BrowserModel, String> {
        let device = <CpuBackend as BackendTypes>::Device::default();
        let mut model = SyncopateModel::<CpuBackend>::new(config.clone(), &device)
            .map_err(|e| e.to_string())?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(bytes, &device)
            .map_err(|e| format!("checkpoint load error: {e:?}"))?;
        model = model.load_record(record);

        Ok(BrowserModel::Cpu { model, device })
    }

    /// Single async forward pass → logits for the last token position.
    ///
    /// `context_ids` is the full sequence so far (prompt + generated).
    /// Returns a `Vec<f32>` of length `vocab_size`.
    ///
    /// This method is async because `into_data_async` is required on WASM
    /// where synchronous tensor reads are unsupported.
    ///
    /// The caller handles sampling (temperature, top-k, etc.).
    pub async fn step(&self, context_ids: &[u32]) -> Result<Vec<f32>, String> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| "config missing".to_string())?;
        match self
            .model
            .as_ref()
            .ok_or_else(|| "model not loaded".to_string())?
        {
            BrowserModel::Wgpu { model, device } => {
                step_with_backend(model, device, config, context_ids).await
            }
            BrowserModel::Cpu { model, device } => {
                step_with_backend(model, device, config, context_ids).await
            }
        }
    }

    /// Current runtime status.
    pub fn status(&self) -> RuntimeStatus {
        self.status
    }

    /// Model vocab size (0 if not loaded).
    pub fn vocab_size(&self) -> usize {
        self.config.as_ref().map(|c| c.vocab_size).unwrap_or(0)
    }

    /// Model seq len (0 if not loaded).
    pub fn seq_len(&self) -> usize {
        self.config.as_ref().map(|c| c.seq_len).unwrap_or(0)
    }

    /// Active browser inference backend.
    pub fn backend_label(&self) -> &'static str {
        self.backend_label
    }

    /// Last error message.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

async fn step_with_backend<B>(
    model: &SyncopateModel<B>,
    device: &B::Device,
    config: &SyncopateModelConfig,
    context_ids: &[u32],
) -> Result<Vec<f32>, String>
where
    B: burn::tensor::backend::Backend,
{
    let logits_tensor = model
        .forward_logits(context_ids, 0, device)
        .map_err(|e| e.to_string())?;

    let seq_actual = context_ids.len().min(config.seq_len);
    let last_index = seq_actual.saturating_sub(1);
    let vocab_size = config.vocab_size;

    let last_logits = logits_tensor
        .slice([0..1, last_index..last_index + 1, 0..vocab_size])
        .flatten::<1>(0, 2);

    use burn::tensor::TensorData;
    let data: TensorData = last_logits
        .into_data_async()
        .await
        .map_err(|e| format!("tensor read error: {e:?}"))?;

    let vec: Vec<f32> = data
        .to_vec::<f32>()
        .map_err(|e| format!("tensor conversion error: {e}"))?;

    Ok(vec)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Fetch a `config.json` produced by the training script and build a
/// [`SyncopateModelConfig`] from it. Only the fields that differ from
/// the 1M preset are applied (currently just `attention_kernel`).
async fn fetch_config_json(url: &str) -> Result<SyncopateModelConfig, String> {
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, window};

    let opts = RequestInit::new();
    opts.set_method("GET");
    let request =
        Request::new_with_str_and_init(url, &opts).map_err(|e| format!("request build: {e:?}"))?;
    let window = window().ok_or_else(|| "no window".to_string())?;
    let resp_val = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("fetch config: {e:?}"))?;
    let response: web_sys::Response = resp_val.into();
    if !response.ok() {
        return Err(format!("HTTP {} fetching {}", response.status(), url));
    }
    let text_js = JsFuture::from(
        response
            .text()
            .map_err(|e| format!("response.text: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("read config text: {e:?}"))?;
    let text = text_js.as_string().unwrap_or_default();

    // Parse only the attention_kernel field first; dimensions come from the
    // full config below so tiny action models load with the exact train shape.
    let kernel = if text.contains("\"higher-order\"") {
        AttentionKernel::HigherOrder
    } else {
        AttentionKernel::Softmax
    };

    // Parse the full JSON and extract all model dimension fields.
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;

    let vocab = json["vocab_size"].as_u64().unwrap_or(3200) as usize;
    let seq = json["seq_len"].as_u64().unwrap_or(128) as usize;
    let layers = json["layers"].as_u64().unwrap_or(2) as usize;
    let d_model = json["d_model"].as_u64().unwrap_or(96) as usize;
    let attention_heads = json["attention_heads"].as_u64().unwrap_or(4) as usize;
    let kv_heads = json["kv_heads"].as_u64().unwrap_or(1) as usize;
    let intermediate_size = json["intermediate_size"].as_u64().unwrap_or(256) as usize;

    Ok(SyncopateModelConfig::from_dimensions(
        vocab,
        seq,
        layers,
        d_model,
        attention_heads,
        kv_heads,
        intermediate_size,
    )
    .with_attention_kernel(kernel))
}

fn default_browser_config(vocab_size: usize, seq_len: usize) -> SyncopateModelConfig {
    if vocab_size <= 64 {
        SyncopateModelConfig::preset_action(vocab_size, seq_len)
    } else {
        SyncopateModelConfig::preset_1m(vocab_size, seq_len)
    }
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    use js_sys::Uint8Array;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, Response, window};

    let opts = RequestInit::new();
    opts.set_method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;

    let window = window().ok_or_else(|| JsValue::from_str("no window"))?;
    let response_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let response: Response = response_value.into();

    if !response.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP {} fetching {}",
            response.status(),
            url
        )));
    }

    let ab = JsFuture::from(response.array_buffer()?).await?;
    let uint8 = Uint8Array::new(&ab);
    Ok(uint8.to_vec())
}
