# PLAN_10: Convert lookhin-but-ai Experiment to Examples

**Status:** ✅ Complete
**Date:** 2025-07-11 (revised 2025-07-12)

---

## Overview

Converted the `lookhin-but-ai/` experiment folder into two concise, high-level examples for the main `multiscreen-rs` crate. Both examples use the library's `Trainer` and `ChatModel` APIs instead of reimplementing low-level training loops and inference.

Also added streaming token-by-token generation support (`generate_stream`) and a bundled `examples/data/` directory with tokenizer + sample training data so the examples work out of the box.

---

## Example Files

| File | Lines | Description |
|---|---|---|
| `examples/train_with_tokenizer.rs` | ~308 | Train a Multiscreen LM with SentencePiece tokenization |
| `examples/chat_with_tokenizer.rs` | ~198 | Interactive streaming chat with a trained model |
| `examples/data/tokenizer.model` | — | Bundled SentencePiece model (~280KB, 2778 vocab) |
| `examples/data/sample_chat.txt` | 15 lines | Sample English chat training data for quick testing |

### Removed Examples

The following low-level examples were removed because their functionality is now covered by the two tokenizer examples and the library's high-level API:

| File | Reason |
|---|---|
| `examples/train.rs` | Superseded by `train_with_tokenizer.rs` |
| `examples/inference.rs` | Superseded by `chat_with_tokenizer.rs` |
| `examples/quick_start.rs` | Inline docs in `lib.rs` serve the same purpose |

---

## Design

### High-level API usage

Both examples delegate to the library's public API rather than reimplementing internals:

- **`train_with_tokenizer.rs`** — uses `Trainer::builder()` + `train_on_token_sequences()`
- **`chat_with_tokenizer.rs`** — uses `ChatModel::load()` + `generate_stream()`

### Bundled data

`examples/data/` contains `tokenizer.model` (copied from `experiment/train/`) and `sample_chat.txt` (15-line English chat). This means:

- `train_with_tokenizer` works with `--train-dir examples/data` out of the box
- `chat_with_tokenizer` finds the tokenizer via `examples/data/` fallback
- No need to reference `experiment/train/` at all

### Output structure

Training produces:

```
runs/my-model/
  checkpoints/
    config.json      — MultiscreenModelConfig (used by ChatModel::load)
    latest.mpk       — model weights
    latest.json      — run metadata (step, loss, params)
  tokenizer.model    — copy of the tokenizer for reproducibility
```

`ChatModel::load("runs/my-model/checkpoints/latest.mpk")` automatically discovers `config.json` in the checkpoint directory.

---

## Streaming API (added as part of this plan)

```rust
// Low-level
model.infer_tokens_stream(prompt, config, device, |token_id, index| {
    // decode & print token
    true  // return false to stop early
})?;

// High-level
model.generate_stream(prompt, config, |token_id, index| {
    // decode & print token
    true
})?;
```

---

## Dev Dependencies

```toml
[dev-dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
sentencepiece-rs = { path = "../sentencepiecers" }
```

`chrono` was removed — the simplified examples no longer need timestamped logging.

---

## Modified Files

| File | Change |
|---|---|
| `Cargo.toml` | Dev-deps: `anyhow`, `clap`, `sentencepiece-rs`; removed `chrono` |
| `src/model.rs` | Added `infer_tokens_stream()` with per-token callback |
| `src/inference.rs` | Added `generate_stream()` to `ChatModel` |
| `src/lib.rs` | Re-exported `cross_entropy_loss_with_mask`, updated doc examples |
| `src/prelude.rs` | Re-exported `cross_entropy_loss_with_mask` |
| `.gitignore` | Added `/runs` |

---

## Usage

```bash
# Train with bundled sample data
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/my-model --steps 5000

# Chat with the trained model (streaming output)
cargo run --release --example chat_with_tokenizer -- --run-dir runs/my-model

# One-shot prompt
cargo run --release --example chat_with_tokenizer -- \
    --run-dir runs/my-model --prompt "User: hello Assistant:"

# Train with your own data
cargo run --release --example train_with_tokenizer -- \
    --train-dir /path/to/my/data --run-dir runs/custom --budget 10m --steps 50000
```

---

## Verification

| Check | Status |
|---|---|
| `cargo fmt --all -- --check` | ✅ |
| `cargo check --examples` | ✅ (0 warnings) |
| `cargo test --lib --tests` | ✅ (25 tests) |
| `cargo clippy --all-targets -- -D warnings` | ✅ |
| End-to-end train → chat pipeline | ✅ |

---

## Notes

- The `lookhin-but-ai/` and `experiment/` folders are preserved as-is for reference
- `sentencepiece-rs` is a dev-dependency only — library users bring their own tokenizer
- The `examples/data/` directory is included in the crate package via `include = ["examples/**"]`
