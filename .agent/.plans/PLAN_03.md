# PLAN_03.md — Neural Multiscreen Model Port & Next Development Rules

## Read-First Rule

Before any future development in this repository, read the latest numbered plan
inside:

```text
.agent/.plans/
```

Use the highest `PLAN_XX.md` number as the current source of truth before
editing code. As of this file, `PLAN_03.md` is the latest plan.

Do not rely on older assumptions from `PLAN_01.md` or `PLAN_02.md` when they
conflict with this file. Those older plans describe the first layout/transition
API and JSON state persistence work; the current library now also contains a
real Candle-backed neural Multiscreen model.

---

## Current Goal

Port the Multiscreen neural model path from:

```text
./multiscreen-testing/src/multiscreen.rs
```

into the root library so users can train and run inference directly through
`multiscreen-rs`.

Keep the Transformer model from `./multiscreen-testing` out of this crate.

---

## Current Implementation Status

Implemented:

- `src/model.rs`
  - `MultiscreenModelConfig`
  - `MultiscreenModel`
  - `ModelTrainingConfig`
  - `ModelInferenceConfig`
  - `ModelTrainingReport`
  - `MultiscreenModelOutput`
  - Neural forward pass with:
    - token embedding
    - tied logits
    - gated screening tiles
    - `W_Q`, `W_K`, `W_V`, `W_G`, `W_O`
    - learned `s_w`, `s_r`, `s_o`
    - MiPE
    - trim-and-square screening
    - causal softmask
    - TanhNorm
    - residual layer updates
  - `train_token_sequences(...)`
  - `train_texts(...)`
  - `infer_tokens(...)`
  - `infer_text(...)`
  - `save_parameters(...)`
  - `load_parameters(...)`

- `src/optim.rs`
  - Minimal AdamW optimizer with optional global norm clipping.

- `src/param_io.rs`
  - Binary parameter save/load with `MSCRP001` magic.
  - This is for neural model tensors, not the old JSON transition state.

- `src/runtime.rs`
  - `default_device()`
  - `device_label()`
  - Optional CUDA feature support through `candle-core/cuda`.

- `src/lm.rs`
  - `LanguageModel`
  - `TrainableLanguageModel`

- `Cargo.toml`
  - `candle-core = "0.10"`
  - `sentencepiece-rs = "0.1"`
  - `serde`
  - `serde_json`
  - `tempfile` for tests
  - `cuda` feature mapping to `candle-core/cuda`

- `README.md`
  - Updated to describe the neural Multiscreen model path.
  - No longer claims the crate has no backprop/GPU tensor ops.

- `examples/`
  - `quick_start.rs` uses `MultiscreenModel`.
  - `train.rs` uses `MultiscreenModel`.
  - `inference.rs` uses `MultiscreenModel`.

- `tests/model_tests.rs`
  - Verifies forward shape.
  - Verifies train + infer token smoke path.

Still intentionally retained:

- Layout/scoring utility API:
  - `MultiscreenConfig`
  - `ScreenLayout`
  - `trim_and_square`
  - `causal_softmask`

- Lightweight transition engine:
  - `MultiscreenEngine`
  - JSON state save/load

These are now utility APIs. The real neural model API is `MultiscreenModel`.

---

## Public Neural Model API

Minimal token training and inference:

```rust
use multiscreen_rs::prelude::*;

fn main() -> candle_core::Result<()> {
    let device = default_device()?;
    let model = MultiscreenModel::new(MultiscreenModelConfig::tiny_for_tests(), &device)?;

    model.train_token_sequences(
        &[vec![1, 2, 3, 4], vec![1, 2, 5, 4]],
        &ModelTrainingConfig {
            steps: 8,
            batch_size: 2,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            grad_clip_norm: Some(1.0),
            pad_token_id: 0,
        },
        &device,
    )?;

    let output = model.infer_tokens(
        &[1, 2],
        &ModelInferenceConfig {
            max_new_tokens: 4,
            pad_token_id: 0,
        },
        &device,
    )?;

    println!("{:?}", output.token_ids);
    Ok(())
}
```

Text path:

```rust
let tokenizer = SentencePieceTokenizer::from_file("model.spm")
    .map_err(|err| candle_core::Error::msg(err.to_string()))?;

model.train_texts(
    &tokenizer,
    ["hello world", "hello multiscreen"],
    &ModelTrainingConfig::default(),
    &device,
)?;

let output = model.infer_text(
    &tokenizer,
    "hello",
    &ModelInferenceConfig::default(),
    &device,
)?;
```

---

## Hard Scope Rules Going Forward

Must keep:

- Multiscreen neural model support.
- SentencePiece tokenizer integration.
- CPU training/inference.
- Optional CUDA support via feature flag.
- Parameter save/load for Multiscreen model tensors.
- Tests and examples for train/infer.

Must not add:

- Transformer implementation from `multiscreen-testing`.
- Transformer checkpoints or Transformer training path.
- PyTorch checkpoint loading unless a future plan explicitly requests it.
- Large dataset/model artifacts into the crate package.

---

## Known Design Notes

1. `MultiscreenModel` methods return `candle_core::Result`, not the crate's
   layout/transition `Result`.
2. `MultiscreenModelConfig` is separate from the older `MultiscreenConfig`.
   - `MultiscreenModelConfig`: neural model dimensions.
   - `MultiscreenConfig`: layout/scoring utility config.
3. `save_parameters` / `load_parameters` save neural tensor parameters.
4. `save_weights` / `load_weights` on `MultiscreenEngine` save the old JSON
   transition engine state. Do not confuse the two.
5. `cargo clippy --all-targets --all-features -- -D warnings` on Windows with
   CUDA requires the Visual Studio build environment so `nvcc` can find
   `cl.exe`.

Example CUDA clippy command:

```powershell
cmd /S /C "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && cargo clippy --all-targets --all-features -- -D warnings"
```

---

## Verification Commands

CPU/default path:

```bash
cargo fmt
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

All features with CUDA on Windows:

```powershell
cmd /S /C "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && cargo clippy --all-targets --all-features -- -D warnings"
```

Package verification:

```bash
cargo package --allow-dirty
```

Latest known verification status after the neural model port:

- `cargo check --all-targets` passed
- `cargo test` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo doc --no-deps` passed
- CUDA/all-features clippy passed via `vcvars64.bat`
- `cargo package --allow-dirty` passed

---

## Next Recommended Work

1. Add a higher-level `MultiscreenTrainer` facade if the API feels too raw.
2. Add config + parameter metadata sidecar so checkpoints can validate model
   dimensions before load.
3. Add proper text dataset batching utilities instead of only direct
   `Vec<Vec<u32>>` / text arrays.
4. Add sampling options beyond greedy argmax.
5. Add README section showing how to load an existing tokenizer + checkpoint
   from `multiscreen-testing/models`.
6. Add benchmark examples for CPU vs CUDA.

