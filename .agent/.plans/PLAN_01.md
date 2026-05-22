# PLAN.md

## Goal

Port only the multi-screen / multi-tile logic from `./multiscreen-testing`
into the current Rust library project.

Do NOT port, copy, rewrite, or depend on any Transformer model code
from `./multiscreen-testing`.

This library must expose a clean Rust API that allows users to:

- configure multi-screen parameters
- configure tile/grid behavior
- tokenize text using `sentencepiece-rs = "0.1"`
- train using user-provided data
- run inference easily
- publish the crate to crates.io

---

## Hard Scope Rules

### Must include

- Multi-screen logic
- Multi-tile / screen layout logic
- Configurable parameters
- Train API
- Inference API
- SentencePiece tokenizer integration via:

```toml
sentencepiece-rs = "0.1"
```

- Documentation
- Examples
- Tests
- MIT license covering only this library code
- crates.io-ready package metadata

### Must NOT include

- Full Transformer implementation from `./multiscreen-testing`
- Attention layer code copied from experiments
- Any unrelated experiment code
- Any dead prototype code
- Any copied license-incompatible code

> **Note:** `candle-core` IS included as a dependency for tensor operations,
> safetensors weight loading, and config validation. What we do NOT do is
> copy Transformer/attention implementations from `./multiscreen-testing`.

If code in `./multiscreen-testing` mixes multi-screen logic with Transformer
logic, extract only the reusable multi-screen portion and rewrite the boundary
cleanly.

---

## Source Discovery

Before coding, inspect:

```text
./multiscreen-testing
./.agents
```

Use markdown files inside `./.agents` as implementation guidance when behavior
is unclear.

---

## Proposed Library Structure

```text
src/
  lib.rs
  config.rs
  error.rs
  tokenizer.rs
  screen.rs
  tile.rs
  layout.rs
  train.rs
  inference.rs
  prelude.rs

examples/
  quick_start.rs
  train.rs
  inference.rs

tests/
  config_tests.rs
  tokenizer_tests.rs
  layout_tests.rs
  train_tests.rs
  inference_tests.rs

README.md
LICENSE
Cargo.toml
```

---

## Tokenization

Use:

```toml
sentencepiece-rs = "0.1"
```

Expected behavior:

```rust
let tokenizer = SentencePieceTokenizer::from_file("model.spm")?;
let ids = tokenizer.encode("hello world")?;
let text = tokenizer.decode(&ids)?;
```

Do not invent a custom tokenizer.

---

## Training API

Training should be callable with a simple API.

```rust
engine.train(input)?;
```

Supported constructors:

```rust
TrainInput::from_texts(Vec<&str>)
TrainInput::from_strings(Vec<String>)
TrainInput::from_token_ids(Vec<Vec<u32>>)
```

Do not implement Transformer training.

---

## Inference API

```rust
let output = engine.infer("some input text")?;
```

Also support:

```rust
let output = engine.infer_tokens(&[1, 2, 3, 4])?;
```

---

## README.md Requirements

README must contain:

1. What this crate does
2. What it does not do
3. Installation
4. Quick start
5. Training example
6. Inference example
7. Configuration example
8. Tokenizer notes
9. License notes
10. crates.io publishing notes

Quick start must be copy-paste friendly.

---

## LICENSE Requirements

Add MIT LICENSE with wording clarifying:

- MIT applies only to this library code
- third-party dependencies are governed by their own licenses
- users are responsible for dependency license review

---

## Cargo.toml Requirements

```toml
[package]
name = "REPLACE_WITH_CRATE_NAME"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Configurable multi-screen and multi-tile processing library for Rust."
repository = "REPLACE_WITH_REPOSITORY_URL"
readme = "README.md"
keywords = ["multiscreen", "tiles", "tokenizer", "sentencepiece"]
categories = ["science", "text-processing"]

[dependencies]
sentencepiece-rs = "0.1"
candle-core = "0.10"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Only add dependencies that serve the library's core purpose.

---

## Tests

Run:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo doc --no-deps
```

---

## crates.io Preparation

Before publishing:

```bash
cargo package
cargo publish --dry-run
```

Checklist:

- README renders correctly
- LICENSE exists
- tests pass
- examples compile
- docs compile
- no Transformer code included
- no accidental large files included

---

## Final Acceptance Criteria

1. Rust lib builds successfully
2. Users can configure screens and tiles
3. Users can tokenize using sentencepiece-rs
4. Users can train easily
5. Users can run inference easily
6. README has copy-paste quick start
7. MIT LICENSE exists
8. crate is ready for crates.io
9. no Transformer code was ported
