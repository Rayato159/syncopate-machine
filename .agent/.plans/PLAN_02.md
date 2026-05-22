# PLAN_02.md — Weight Loading, Config Validation & Docs Refresh

## Goal

Add model weight persistence with strict config validation, integrate `candle-core`
for tensor operations, and refresh all documentation.

---

## Tasks

### 1. Add Dependencies to `Cargo.toml`

```toml
candle-core = "0.10"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

### 2. Error Handling (`src/error.rs`)

Add variants to `Error` enum:

- `Io(String)` — file read/write failures
- `Serialization(String)` — serde encode/decode failures
- `WeightsConfigMismatch(String)` — weights file config ≠ engine config

Add `Display` + `From` impls for each.

### 3. Serialization Derives

Add `#[derive(Serialize, Deserialize)]` to:

- `MultiScreenConfig` (`src/config.rs`)
- `TrimConfig` (`src/config.rs`)
- `ScreenConfig` (`src/screen.rs`)
- `TileConfig` (`src/tile.rs`)
- `GridConfig` (`src/tile.rs`)
- `InferenceConfig` (`src/inference.rs`)
- `LearnedState` (`src/train.rs`)
- `TrainReport` (`src/train.rs`)

### 4. Weight File Format

JSON structure:

```json
{
  "config": { ... },
  "state": { ... },
  "report": { ... }
}
```

### 5. `MultiScreenEngine` Weight Methods (`src/inference.rs`)

```rust
/// Save config + learned state to JSON file.
pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()>;

/// Load weights into existing engine. Rejects if config mismatches.
pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<TrainReport>;

/// Create a new engine directly from a weights file.
pub fn from_weights_file(path: impl AsRef<Path>) -> Result<Self>;
```

### 6. Config Validation on Load

When `load_weights` is called:

1. Deserialize the weights file
2. Compare `file.config` against `self.config` using `PartialEq`
3. If mismatch → return `Error::WeightsConfigMismatch` with a descriptive message
4. If match → replace `self.state` with the loaded state

### 7. Crate-Level Documentation (`src/lib.rs`)

Add comprehensive `//!` doc comments with:

- Overview of what the crate does
- Quick start code example (copy-paste friendly)
- Training example
- Inference example
- Weight loading example
- Links to README

### 8. Rewrite `README.md` (Gen-Z Style)

- Casual, high-energy tone
- Heavy emoji usage
- Zero-friction copy-paste code blocks
- Table of contents
- Quick start first
- What it does / doesn't do
- Training, inference, config, tokenizer sections
- Weight saving/loading section
- License + crates.io notes

### 9. Tests

Add tests in `tests/inference_tests.rs`:

- `save_and_load_weights_roundtrip` — save then load, verify state
- `load_weights_rejects_config_mismatch` — save with config A, load with config B → error
- `from_weights_file_creates_engine` — load from file, verify engine works

### 10. Build Verification

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo doc --no-deps
```

---

## Acceptance Criteria

1. `cargo build` succeeds with `candle-core`, `serde`, `serde_json`
2. `save_weights` writes valid JSON
3. `load_weights` restores state when config matches
4. `load_weights` returns `WeightsConfigMismatch` when config differs
5. `from_weights_file` creates a working engine from a weights file
6. `src/lib.rs` has full crate-level docs visible on docs.rs
7. `README.md` is Gen-Z styled with copy-paste examples
8. All tests pass
9. No Transformer code was ported
