# PLAN_05.md - Burn Backend Migration

## Goal

Move the neural `MultiscreenModel` path from direct Candle tensors to Burn.

The crate should no longer expose a Candle-backed model API as the primary
neural path. The layout/scoring utility APIs and JSON transition engine remain
unchanged.

## Why

Burn gives this crate a real framework-level abstraction:

- Generic backend support through Burn's `Backend` trait.
- Autodiff through Burn's `Autodiff` backend decorator.
- Built-in module/parameter ownership for optimizer and checkpoint flows.
- A Candle-free CPU default through the Burn Flex backend.

Direct Candle code was enough for the first port, but it hard-wired the model to
one tensor runtime and forced this crate to maintain manual optimizer and tensor
checkpoint plumbing.

## Current Migration Scope

Implement now:

- Replace `candle-core` dependency with `burn`.
- Use Burn Flex as the default CPU backend.
- Keep the public model names:
  - `MultiscreenModelConfig`
  - `MultiscreenModel`
  - `ModelTrainingConfig`
  - `ModelInferenceConfig`
  - `ModelTrainingReport`
  - `MultiscreenModelOutput`
- Port the Multiscreen math:
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
- Keep token-sequence training and greedy inference.
- Save/load model parameters through Burn recorders.
- Update README, examples, and tests.

Defer unless explicitly requested:

- Recreating the Transformer model from `multiscreen-testing`.
- PyTorch checkpoint import.
- Long training runs.
- Benchmark numbers.

## API Direction

Burn modules are value-updated by optimizers. Therefore training may require a
mutable model or may return an updated model. Preserve ergonomics where possible,
but prefer Burn-correct ownership over hiding mutation behind unsafe tricks.

The default user path should be Candle-free:

```rust
type Backend = multiscreen_rs::DefaultAutodiffBackend;

let device = default_device();
let mut model = MultiscreenModel::<Backend>::new(config, &device)?;
model.train_token_sequences(...)?;
```

Inference can use the same autodiff backend for simplicity, while advanced users
can use Burn's inner backend through the generic type parameter later.

## Verification

Run at minimum:

```bash
cargo fmt
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
```

If Burn dependency download is needed, request network escalation through Codex
instead of pretending local cache is enough.

## Implemented Status

Implemented in this migration:

- `Cargo.toml`
  - Replaced `candle-core` with `burn = "0.21"`.
  - Default Burn features are `std`, `autodiff`, and `flex`.
  - `cuda` maps to Burn's CUDA feature, but the default device remains Flex.
- `src/runtime.rs`
  - Added `DefaultBackend = burn::backend::Flex`.
  - Added `DefaultAutodiffBackend = burn::backend::Autodiff<DefaultBackend>`.
  - `default_device()` now returns the default Burn Flex device.
- `src/model.rs`
  - Ported `MultiscreenModel` to `#[derive(Module)]`.
  - Parameters are now Burn `Param<Tensor<B, D>>`.
  - Added `DefaultMultiscreenModel` alias for the default Burn Flex autodiff
    path.
  - Training now mutates the model through Burn `AdamWConfig`,
    `GradientsParams`, and `Optimizer::step`.
  - Inference and loss now use Burn tensors.
  - Save/load now uses Burn `NamedMpkFileRecorder<FullPrecisionSettings>`.
- `src/optim.rs`
  - Replaced the local Candle AdamW implementation with Burn optimizer
    re-exports.
- `src/param_io.rs`
  - Replaced raw tensor checkpoint helpers with Burn module recorder helpers.
- README, examples, and tests now use `DefaultMultiscreenModel`.

## Verification Result

Latest known status after the Burn migration:

- `cargo fmt` passed
- `cargo check --all-targets` passed
- `cargo test` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo doc --no-deps` passed

CUDA/all-features verification was not run in this pass. Burn CUDA is a
backend-specific path and may require extra local GPU/toolchain setup.

`cargo package --allow-dirty` was attempted after the migration, but it timed
out while trying to update the crates.io index in the restricted network
sandbox. Escalated retry was not available in this session because the approval
request was rejected by the host usage limit.
