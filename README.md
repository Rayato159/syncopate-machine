# syncopate-machine 🧠⚡

Tiny Burn transformer that predicts **action IDs** — no tokenizer, no text gen, just vibes.

```text
user text → classifier (action + lang) → [action_id, lang_id, SEP] → 🔮 model predicts next actions → response graph writes the msg → mood GIF ✨
```

## 🏗️ Model Architecture

> 📊 Full architecture slides: [`slides.tex`](slides.tex) (Beamer, compile with `pdflatex`)

| Setting            | Value                                |
| ------------------ | ------------------------------------ |
| Type               | Causal Transformer (GQA, kv_heads=1) |
| Attention kernel   | **HigherOrder** (polynomial) 🔥       |
| Layers             | 1                                    |
| d_model            | 64                                   |
| Attention heads    | 4                                    |
| FFN (intermediate) | 64                                   |
| Vocab              | 23 IDs                               |
| Seq len            | 64                                   |
| **Total params**   | **~24K** 🤏                          |

> We tried Bahdanau (additive) attention — turns out transformers with GQA are just built different for tiny action-ID tasks. Fewer params, parallel training, no contest 🤷‍♂️

## 🚀 Quick Start

### 1. Prepare data

```bash
python examples\build_action_data.py --offline
```

### 2. Train (CUDA only)

```bash
cargo run --release --features cuda --example train_action_model -- \
  --steps 3000 --batch-size 32 --lr 0.003 --kernel higher-order --checkpoint-dir runs/action-model-v2
```

Outputs in `runs/action-model-v2/`:

- `final.mpk` — model weights
- `model-config.json` — config for browser loading
- `loss.csv` — training loss per step
- `report.json` — full training report

### 3. Inference (native Rust)

```rust,no_run
use syncopate_machine::prelude::*;

let device = auto_device()?;
let config = SyncopateModelConfig::preset_action(23, 64)
    .with_attention_kernel(AttentionKernel::HigherOrder);
let mut model = DefaultSyncopateModel::new(config, &device)?;
model.load_parameters("runs/action-model-v2/final.mpk")?;

// Forward pass: context_ids -> logits for the last position
let logits = model.forward_logits(&[1, 5, 3, 9], 0, &device)?;  // [1, seq_len, 23]
```

### 4. Inference (browser / WASM)

Load these assets into your WASM build:

- `model-personal-v2.mpk`
- `model-config-personal-v2.json`

```rust,ignore
let mut runtime = BrowserRuntime::new();
runtime.load_from_url_with_config_json(
    "assets/model-personal-v2.mpk",
    Some("assets/model-config-personal-v2.json"),
    23, 64,
    BrowserBackendPreference::Auto,
).await?;

// Single step: returns Vec<f32> of length vocab_size
let logits = runtime.step(&[1, 5, 3, 9]).await?;
```

Works on WebGPU or CPU — auto-fallback if GPU is broken.

### Attention Kernels

| Kernel | Normalization | Notes |
|---|---|---|
| `softmax` (default) | `softmax(QKᵀ / √d)` | standard, well-tested |
| `higher-order` | `(W·Wᵀ · keep) / Σ|W·Wᵀ|` | polynomial, zero extra params |

Pass `--kernel higher-order` at train time to use it. The kernel is saved in `model-config.json` and loaded automatically by the browser runtime.

## 🪪 LICENSE

MIT. Break it, fix it, ship it 🚀
