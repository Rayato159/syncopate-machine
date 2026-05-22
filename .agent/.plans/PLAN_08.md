# PLAN_08.md - Client Training Handoff Summary

> Superseded by PLAN_09 — the high-level public API refactoring below.

## Goal

Rewrite `SUMMARY.md` so it can be copied into a client project and used by
another coding agent to implement a training binary for `multiscreen-rs`.

## Changes

- Replaced the old broad summary with a client handoff document.
- Added exact dependency snippets for crates.io and local path usage.
- Added a trainer-oriented API walkthrough:
  - `default_device()`
  - `SentencePieceTokenizer::from_file(...)`
  - `tokenizer.vocab_size()`
  - `MultiscreenModelConfig::for_parameter_budget(...)`
  - `DefaultMultiscreenModel::new(...)`
  - `train_texts(...)`
  - `save_parameters(...)`
  - `load_parameters(...)`
  - `infer_text(...)`
- Added suggested CLI flags for a client `src/bin/train.rs`.
- Added a minimal client trainer skeleton using `clap` and `anyhow`.
- Added caveats about high-level training not being a streaming trainer.
- Added "do not do" rules:
  - do not copy Transformer model
  - do not train SentencePiece inside the client trainer
  - do not commit checkpoints/datasets
  - do not fake benchmarks
  - do not use CUDA by default in CI
- Added `SentencePieceTokenizer::vocab_size()` to support the client skeleton.
- Updated README to mention using `tokenizer.vocab_size()` for model config.

## Verification

Run after this plan is written:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
cargo package --allow-dirty --offline
```

## Verification Result

Latest known status after the summary handoff update:

- `cargo fmt --all --check` passed
- `cargo check --all-targets` passed
- `cargo test` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo doc --no-deps` passed
- `cargo package --allow-dirty --offline` passed

---

# PLAN_09 — High-Level Public API Refactoring

## Goal

Refactor the public API of multiscreen-rs to make the library dramatically
easier to use for beginners and production users. The current workflow exposes
too many low-level implementation details. The new API should feel like a
modern plug-and-play ML library.

## Changes

### New files created

- **`src/training.rs`** — `Trainer`, `TrainerBuilder`, `TrainingReport`,
  `ParameterBudget` re-export. Builder-pattern training with sensible defaults.
  Loads `.txt` and `.jsonl` data from a dataset directory.

- **`src/engine.rs`** — `MultiscreenEngine`, `TrainInput`, `TrainReport`,
  `InferenceOutput`, `ScreeningState`. Consolidated from the old `train.rs`
  and old `inference.rs` (lightweight transition engine). Now an internal
  `pub(crate)` module — types are re-exported through `lib.rs`.

- **`src/device.rs`** — `cpu()`, `cuda(index)`, `auto_device()`. Clean device
  abstraction with feature-gated CUDA support and helpful error messages.

### Files consolidated / renamed

- Old `src/train.rs` (TrainInput, TrainReport, ScreeningState) → merged into
  `src/engine.rs`
- Old `src/inference.rs` (MultiscreenEngine, InferenceOutput) → merged into
  `src/engine.rs`
- Old `src/inference_highlevel.rs` (ChatModel, GenerationConfig) → became the
  NEW `src/inference.rs`

### Deleted files

- `src/train.rs` — content merged into `src/engine.rs`
- `src/inference_highlevel.rs` — content moved to `src/inference.rs`

### Updated files

- **`src/lib.rs`** — Only 4 public modules: `device`, `inference`, `prelude`,
  `training`. All other modules are `pub(crate)`. High-level re-exports at
  crate root. Rewrote crate-level doc comments.

- **`src/inference.rs`** — Now contains only `ChatModel` and
  `GenerationConfig` (the high-level API).

- **`src/prelude.rs`** — Restructured into sections: high-level API, core
  types, model configuration, tokenizer, engine, layout utilities.

- **`README.md`** — Complete rewrite. Beginner-friendly, copy-paste-ready.

- **`examples/train.rs`** — Uses `Trainer::builder()`.
- **`examples/inference.rs`** — Uses `ChatModel::load()` + `.chat()`.
- **`examples/quick_start.rs`** — Uses `cpu()` + low-level in-memory API.

- **`src/optim.rs`**, **`src/lm.rs`**, **`src/param_io.rs`**, **`src/model.rs`**
  — Added `#[allow(dead_code)]` / `#[allow(unused_imports)]` where items are
  `pub` within `pub(crate)` modules.

### Final source structure

```
src/
├── device.rs         ← pub     — cpu(), cuda(), auto_device()
├── inference.rs      ← pub     — ChatModel, GenerationConfig
├── prelude.rs        ← pub     — re-exports everything users need
├── training.rs       ← pub     — Trainer, TrainerBuilder, ParameterBudget
├── config.rs         ← pub(crate)
├── engine.rs         ← pub(crate) — MultiscreenEngine, TrainInput, TrainReport
├── error.rs          ← pub(crate)
├── layout.rs         ← pub(crate)
├── lm.rs             ← pub(crate)
├── model.rs          ← pub(crate)
├── optim.rs          ← pub(crate)
├── param_io.rs       ← pub(crate)
├── runtime.rs        ← pub(crate)
├── screen.rs         ← pub(crate)
├── tile.rs           ← pub(crate)
└── tokenizer.rs      ← pub(crate)
```

Users only see 4 modules. All internal complexity is hidden.

## New Public API Summary

### Training

```rust
use multiscreen_rs::prelude::*;

let mut trainer = Trainer::builder()
    .dataset_dir("train")
    .tokenizer("train/tokenizer.model")
    .budget(ParameterBudget::Params10M)
    .device(cpu()?)
    .batch_size(16)
    .seq_len(128)
    .steps(50_000)
    .build()?;

let report = trainer.train()?;
```

### Inference / Chat

```rust
use multiscreen_rs::prelude::*;

let model = ChatModel::load("checkpoints/latest.mpk")?;
let response = model.chat("Why is Rust memory safe?", GenerationConfig::default())?;
println!("{response}");
```

### Device Selection

```rust
use multiscreen_rs::prelude::*;

let device = cpu()?;         // CPU
let device = cuda(0)?;       // CUDA (feature-gated)
let device = auto_device()?; // best available
```

## Verification

```bash
cargo fmt --all --check
 cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
cargo package --allow-dirty --offline
```

## Verification Result

All checks pass:

- `cargo fmt --all --check` ✅
- `cargo check --all-targets` ✅
- `cargo test` ✅ (29 unit tests + 8 doc tests)
- `cargo clippy --all-targets -- -D warnings` ✅
- `cargo doc --no-deps` ✅
- `cargo package --allow-dirty --offline` ✅ (32 files)
- `cargo run --release --example quick_start` ✅ (runs end-to-end)
