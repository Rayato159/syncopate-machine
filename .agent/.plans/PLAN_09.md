# PLAN_09 — Remove Built-in Tokenizer & Simplify README

## Goal

Remove the built-in SentencePiece tokenizer from multiscreen-rs so that users bring
their own tokenizer. The library should accept raw token ID vectors (`Vec<u32>`)
directly for both training and inference. Also simplify the README to be concise
and focused on quick-start code examples.

## Rationale

- **Flexibility**: Users should be free to use any tokenizer (SentencePiece, BPE,
  WordPiece, Tiktoken, custom) without being locked into a specific implementation.
- **Simplicity**: Removing the `sentencepiece-rs` dependency reduces the dependency
  tree and removes a C++ native dependency that can cause build issues.
- **API clarity**: The library now has a clear boundary — it handles the neural
  model, users handle text ↔ token conversion.

## Changes

### Deleted files

- **`src/tokenizer.rs`** — Entire `SentencePieceTokenizer` wrapper removed
- **`tests/tokenizer_tests.rs`** — Associated tokenizer tests removed

### Dependency changes

- **`Cargo.toml`** — Removed `sentencepiece-rs = "0.1"` dependency, updated
  description and keywords

### Source changes

- **`src/error.rs`** — Removed `TokenizerError` enum, `Tokenizer(TokenizerError)`
  variant from `Error`, and `From<TokenizerError> for Error` impl

- **`src/lib.rs`** — Removed `pub(crate) mod tokenizer`, `SentencePieceTokenizer`
  re-export, `TokenizerError` re-export; updated all doc examples to token-level APIs

- **`src/prelude.rs`** — Removed `SentencePieceTokenizer` re-export

- **`src/model.rs`** — Removed `decoded_text` from `MultiscreenModelOutput`,
  removed `infer_text()` and `train_texts()` methods

- **`src/engine.rs`** — Removed `tokenizer` field from `MultiscreenEngine`,
  removed `with_tokenizer()`, `tokenizer()`, `set_tokenizer()`, `infer(&self, text)`
  methods, removed `Texts` variant from `TrainInput` along with `from_texts()`/
  `from_strings()`, removed `decoded_text` from `InferenceOutput`

- **`src/inference.rs`** — Refactored `ChatModel` to work with token IDs:
  removed tokenizer field, replaced `chat()` with `generate(&[u32]) -> Vec<u32>`,
  removed tokenizer auto-discovery from `load()`

- **`src/training.rs`** — Refactored `Trainer` to use `vocab_size` instead of
  tokenizer path: replaced `.tokenizer(path)`/`.dataset_dir()` with `.vocab_size(n)`,
  replaced `train()`/`train_on_texts()` with `train_on_token_sequences(&[Vec<u32>])`,
  removed `load_texts_from_dir()` helper

### Example changes

- **`examples/train.rs`** — Uses `vocab_size()` builder and `train_on_token_sequences()`
- **`examples/inference.rs`** — Uses token-level `generate()` API
- **`examples/quick_start.rs`** — Unchanged (already token-level)

### Test changes

- **`tests/tokenizer_tests.rs`** — Deleted
- **`tests/inference_tests.rs`** — Removed `raw_text_inference_requires_tokenizer` test
- **`tests/train_tests.rs`** — Removed `raw_text_training_requires_tokenizer` test

### Documentation

- **`README.md`** — Complete rewrite. Short, focused on quick-start code examples
  for training and inference. No longer documents SentencePiece tokenizer usage.

## New Public API

### Training (high-level)

```rust
use multiscreen_rs::prelude::*;

let mut trainer = Trainer::builder()
    .vocab_size(1000)                        // your tokenizer's vocab size
    .budget(ParameterBudget::Params10M)
    .device(cpu()?)
    .steps(50_000)
    .build()?;

let sequences = vec![vec![1, 2, 3, 4], vec![1, 2, 5, 4]]; // from YOUR tokenizer
let report = trainer.train_on_token_sequences(&sequences)?;
```

### Inference (high-level)

```rust
use multiscreen_rs::prelude::*;

let model = ChatModel::load("checkpoints/latest.mpk")?;
let token_ids = model.generate(&[1, 2, 3], GenerationConfig::default())?;
```

### Training (low-level)

```rust
use multiscreen_rs::prelude::*;

let device = cpu()?;
let mut model = DefaultMultiscreenModel::new(MultiscreenModelConfig::tiny_for_tests(), &device)?;
model.train_token_sequences(&[vec![1, 2, 3, 4]], &ModelTrainingConfig { .. }, &device)?;
let output = model.infer_tokens(&[1, 2], &ModelInferenceConfig { .. }, &device)?;
```

### Builder changes

| Old API | New API |
|---|---|
| `.tokenizer("path/to/model")` | `.vocab_size(n)` |
| `.dataset_dir("data/train")` | *(removed — pass sequences directly)* |
| `trainer.train()` | `trainer.train_on_token_sequences(&sequences)` |
| `trainer.train_on_texts(texts)` | `trainer.train_on_token_sequences(&sequences)` |
| `model.chat("prompt", config)` | `model.generate(&[1, 2, 3], config)` |
| `model.infer_text(tok, text, ...)` | `model.infer_tokens(&[1, 2], ...)` |
| `model.train_texts(tok, texts, ...)` | `model.train_token_sequences(&seqs, ...)` |
| `engine.infer("text")` | `engine.infer_tokens(&[1, 2])` |
| `TrainInput::Texts(vec)` | *(removed — use TokenSequences only)* |

## Verification

All checks pass:

- `cargo fmt --all --check` ✅
- `cargo check --all-targets` ✅
- `cargo test --lib --tests` ✅ (25 tests)
- `cargo clippy --all-targets -- -D warnings` ✅
- `cargo doc --no-deps` ✅
- `cargo run --release --example quick_start` ✅

## Migration Guide

If you were using the old tokenizer-based API:

1. **Training**: Load your tokenizer externally, encode your texts into `Vec<u32>`,
   then pass those sequences to `trainer.train_on_token_sequences()`. Set
   `.vocab_size(your_tokenizer.vocab_size())` on the builder.

2. **Inference**: Load your tokenizer externally, encode the prompt into `Vec<u32>`,
   call `model.generate(&token_ids, config)`, then decode the output with your
   tokenizer.

3. **Engine**: Use `TrainInput::from_token_sequences()` instead of
   `TrainInput::from_texts()`. Use `engine.infer_tokens()` instead of
   `engine.infer()`.

4. **Checkpoint loading**: `ChatModel::load()` no longer searches for
   `tokenizer.model`. It only needs the `.mpk` weights file and optionally
   `config.json` in the same directory.
