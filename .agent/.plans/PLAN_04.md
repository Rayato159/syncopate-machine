# PLAN_04.md — InferenceConfig Relocation

## Goal

Move `InferenceConfig` from `src/inference.rs` to `src/config.rs` where it
belongs alongside the other configuration types.

## Why

`InferenceConfig` is a field of `MultiscreenConfig` but was defined in
`inference.rs`. This created a circular dependency:

- `config.rs` imported `InferenceConfig` from `inference.rs`
- `inference.rs` imported `MultiscreenConfig` from `config.rs`

The naming was also confusing alongside `ModelInferenceConfig` in `model.rs`.

## Changes

### `src/config.rs`

- Added `InferenceConfig` struct definition, `Default` impl, and `validate()`.
- Removed `use crate::inference::InferenceConfig` import.
- Added doc comment distinguishing it from `ModelInferenceConfig`.

### `src/inference.rs`

- Removed `InferenceConfig` struct, `Default` impl, and `validate()`.
- Added `use crate::config::MultiscreenConfig` (no longer needs `InferenceConfig`).

### `src/lib.rs`

- Changed `pub use inference::{InferenceConfig, ...}` to `pub use config::{InferenceConfig, ...}`.

### `src/prelude.rs`

- Already re-exports `InferenceConfig` via the crate root — no change needed.

### README.md

- No changes needed. `InferenceConfig` is already used correctly in the
  "Utility Layout Config" section.

## Verification

```bash
cargo fmt --check          # ✅
cargo clippy -- -D warnings # ✅
cargo test                  # 22 tests pass
cargo doc --no-deps         # ✅
```

## Result

`InferenceConfig` now lives in the `config` module, breaking the circular
dependency. The public API is unchanged — `multiscreen_rs::InferenceConfig`
still works the same way.
