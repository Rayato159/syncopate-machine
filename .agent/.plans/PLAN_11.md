# PLAN_11: Training Reports, Evaluation Metrics & Self-Contained Examples

**Status:** ✅ Complete
**Date:** 2025-07-12

---

## Overview

Added comprehensive training reports, per-step loss tracking, train/val/test evaluation, inference latency benchmarking, and loss plot generation to the example workflow. Every training run now produces a full report — all saved under `runs/` (gitignored).

---

## User Requirements

| Requirement | Status |
|---|---|
| Every train run must work out of the box | ✅ |
| Save parameters somewhere not in git | ✅ (`runs/` gitignored) |
| Train report: train time | ✅ |
| Train report: loss plot | ✅ CSV + Python script |
| Train report: accuracy from test and val | ✅ loss, perplexity, accuracy |
| Train report: avg latency on inference | ✅ ms/token |
| Train report: parameters size | ✅ |
| Loss plot code in git, report not in git | ✅ |
| Update README.md | ✅ |
| Commands to run train and chat | ✅ |
| Train 10M params with 10k steps | ✅ (Flex CPU backend) |

---

## Library Changes

### `src/model.rs`

| Change | Detail |
|---|---|
| `train_token_sequences` signature | Added `on_step: impl FnMut(usize, f32)` callback — invoked after each optimizer step with `(step_index, loss_value)` |
| `evaluate_on_sequences` | New method — evaluates on held-out sequences, returns `EvaluationResult` |
| `EvaluationResult` | New struct: `loss`, `perplexity`, `accuracy`, `num_batches`, `total_tokens` |

### `src/training.rs`

| Change | Detail |
|---|---|
| `train_on_token_sequences_with_callback` | New method — passes `on_step` callback to the model |
| `train_on_token_sequences` | Kept as convenience wrapper (backward compatible) |

### `src/lib.rs`, `src/prelude.rs`

- Exported `EvaluationResult`

---

## Example Changes

### `examples/train_with_tokenizer.rs` — Rewritten

New features added:

- **Per-step loss CSV** — `runs/<name>/loss.csv` with `step,loss` columns
- **Wall-clock timing** — training duration, steps/sec throughput
- **Train/val/test split** — 80/10/10 default (configurable via `--val-split`)
- **Evaluation** — automatic val + test evaluation with loss, perplexity, next-token accuracy
- **Inference latency** — average ms/token after training
- **report.json** — machine-readable full report
- **report.md** — human-readable markdown report with tables
- **Progress logging** — configurable via `--log-interval` (default 100)

New CLI options:

| Option | Default | Description |
|---|---|---|
| `--val-split` | `0.1` | Fraction of data for validation |
| `--latency-tokens` | `20` | Tokens to generate for latency benchmark |
| `--log-interval` | `100` | Print loss every N steps |

Default `--steps` changed from 5000 → 10000.

### `examples/plot_loss.py` — New File

Python script to generate loss plots from CSV:

- Raw + smoothed loss curves (moving average)
- Statistics text box (final, best, worst loss)
- Customizable smoothing window, output path, title
- Outputs `loss_plot.png` in the CSV's directory

---

## Output Structure

```
runs/10m-10k/
├── checkpoints/
│   ├── config.json       # Model architecture config
│   ├── latest.mpk        # Trained weights (parameters)
│   └── latest.json       # Run metadata
├── tokenizer.model       # Copy of the tokenizer
├── loss.csv              # Per-step loss values (step,loss)
├── report.json           # Machine-readable full report
└── report.md             # Human-readable training report
```

All under `runs/` — gitignored, never committed.

---

## Report Contents

The `report.md` includes:

- **Configuration** — budget, parameter count, seq len, batch size, learning rate
- **Data** — train/val/test sequence counts, total tokens
- **Training** — duration, throughput (steps/s), final loss, best loss
- **Validation** — loss, perplexity, next-token accuracy
- **Test** — loss, perplexity, next-token accuracy
- **Inference** — avg latency (ms/token), tokens generated, total time

---

## Commands

### Train 10M params, 10k steps

```bash
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/10m-10k --budget 10m --steps 10000
```

### Chat with the trained model

```bash
cargo run --release --example chat_with_tokenizer -- \
    --run-dir runs/10m-10k
```

### Generate a loss plot

```bash
python examples/plot_loss.py runs/10m-10k/loss.csv
```

### Quick test (1M params, 500 steps)

```bash
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/test --budget 1m --steps 500
```

---

## Modified Files

| File | Change |
|---|---|
| `src/model.rs` | Added `on_step` callback to `train_token_sequences`, added `evaluate_on_sequences`, added `EvaluationResult` |
| `src/training.rs` | Added `train_on_token_sequences_with_callback`, kept `train_on_token_sequences` as wrapper |
| `src/lib.rs` | Exported `EvaluationResult`, fixed doc example |
| `src/prelude.rs` | Exported `EvaluationResult` |
| `examples/train_with_tokenizer.rs` | Full rewrite — reporting, evaluation, latency, CSV, split |
| `examples/plot_loss.py` | New — Python loss plot generator |
| `tests/model_tests.rs` | Updated `train_token_sequences` call with callback |
| `README.md` | Updated with train/chat/plot commands, CLI options, reports section |
| `src/device.rs` | Fixed `auto_device()` to always return `Device` (Flex); added `#[cfg]` guard on `Error` import |
| `.gitignore` | No changes needed (`/runs` already gitignored) |

---

## Verification

| Check | Status |
|---|---|
| `cargo fmt --all -- --check` | ✅ |
| `cargo check --examples` | ✅ (0 warnings) |
| `cargo check --features cuda --example train_with_tokenizer` | ✅ |
| `cargo test --lib --tests` | ✅ (25 tests) |
| `cargo clippy --all-targets -- -D warnings` | ✅ |
| `cargo clippy --all-targets --features cuda -- -D warnings` | ✅ |

---

## Notes

- The `Trainer` and `DefaultMultiscreenModel` always use the Flex (CPU) backend. The `auto_device()` function returns `FlexDevice` regardless of whether the `cuda` feature is enabled. To use CUDA, construct a `MultiscreenModel<CudaAutodiffBackend>` directly — this is a future enhancement.
- `evaluate_on_sequences` uses the autodiff backend (computes correct values but builds computation graphs). A future optimization could use the inner backend for inference-only evaluation.
- The loss plot script requires Python 3 with `matplotlib` and `numpy` — not a Rust dependency.
- Data is split deterministically: first 80% train, next 10% val, last 10% test (after dedup + sort).
- The `chat_with_tokenizer.rs` example was not modified — it still works with any trained model in `runs/`.
