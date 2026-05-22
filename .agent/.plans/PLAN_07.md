# PLAN_07.md - Parameter Budget Presets

## Goal

Let users choose practical Multiscreen neural model sizes instead of only using
fixed tiny and 10M configs.

## Changes

- Added `MultiscreenParameterBudget`
  - `Params1M`
  - `Params5M`
  - `Params10M`
  - `Params50M`
  - `Params100M`
- Added `MultiscreenModelConfig::for_parameter_budget(...)`.
- Added direct preset constructors:
  - `preset_1m(...)`
  - `preset_5m(...)`
  - `preset_10m(...)`
  - `preset_50m(...)`
  - `preset_100m(...)`
- Kept `paper_10m(...)` as a compatibility alias for `preset_10m(...)`.
- Added `estimated_parameter_count()` using the implemented model formula:

```text
V*d_E + 2 + N_L*N_H*(d_E*(2*d_K + 3*d_V)+3)
```

The formula counts:

- token embedding: `V*d_E`
- scalar embedding/logit scales: `s_e`, `s_f`
- per tile matrices: `W_Q`, `W_K`, `W_V`, `W_G`, `W_O`
- per tile learned scalars: `s_w`, `s_r`, `s_o`

## Presets For `vocab_size = 8192`, `seq_len = 96`

| Budget | `N_L` | `N_H` | `d_E` | `d_K` | `d_V` | Estimated params |
|---|---:|---:|---:|---:|---:|---:|
| `Params1M` | 2 | 2 | 128 | 32 | 64 | 1,179,662 |
| `Params5M` | 2 | 4 | 384 | 96 | 192 | 5,505,050 |
| `Params10M` | 3 | 4 | 512 | 128 | 256 | 10,485,798 |
| `Params50M` | 6 | 4 | 960 | 240 | 480 | 52,101,194 |
| `Params100M` | 8 | 4 | 1216 | 304 | 608 | 104,595,554 |

## Notes

The final parameter count is approximate for each budget because vocabulary
size controls the token embedding term. Users should call
`estimated_parameter_count()` before training if they need the exact count for a
specific tokenizer.

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

Latest known status after adding parameter budget presets:

- `cargo fmt --all --check` passed
- `cargo check --all-targets` passed
- `cargo test` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo doc --no-deps` passed
- `cargo package --allow-dirty --offline` passed
