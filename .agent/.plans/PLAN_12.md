# PLAN_12: Chat Inference Fixes — VRAM Leak, Prompt Format, Sampling & Loss Masking

**Status:** ✅ Complete
**Date:** 2025-07-13

---

## Overview

Fixed multiple cascading issues with the `chat_with_tokenizer` example and the training pipeline. After training a 50M-parameter model, chatting with it interactively revealed: the app exiting immediately, CUDA OOM crashes, and the model outputting garbage text (colons, role labels like `assistant:`, `user:`). The fixes spanned VRAM management, training data format, decoding strategy, and evaluation architecture.

---

## Problems & Fixes

### 1. VRAM Leak from Autodiff Backend During Inference

**Problem:** `ChatModel` used `MultiscreenModel<Autodiff<DefaultBackend>>`. Every forward pass in `predict_next_token()` created an autodiff computation graph that was never freed. After generating ~64 tokens across multiple turns, VRAM filled up and crashed with `can't allocate buffer of size: 12576256`.

**Fix:** Load model with Autodiff backend, then call `.valid()` (from `AutodiffModule` trait) to strip the autodiff wrapper → `MultiscreenModel<DefaultBackend>` (inference-only, no gradient tracking).

| File | Change |
|---|---|
| `src/inference.rs` | `ChatModel` now stores `MultiscreenModel<DefaultBackend>` via `.valid()` |
| `src/runtime.rs` | Added `InferenceDevice` type alias for inference-only device |

### 2. Chat Prompt Format Mismatch

**Problem:** Training data was formatted as `system: ...\nuser: ...\nassistant: ...` (lowercase roles, newline-separated). `chat_with_tokenizer` used `User: {input} Assistant:` (uppercase, inline) — completely different format. The model didn't recognize the prompt format → fell back to generating `:` tokens in a loop.

**Fix:** Changed prompt format in `chat_with_tokenizer` to match training data exactly, with the full system prompt prepended.

| File | Change |
|---|---|
| `examples/chat_with_tokenizer.rs` | Prompt format: `system: ...\nuser: ...\nassistant:` |

### 3. Greedy Decoding Loop

**Problem:** Original code used argmax (greedy decoding) which gets stuck generating the same token repeatedly. Model kept outputting `::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::`.

**Fix:** Added temperature scaling, top-k sampling (k=40), and repetition penalty (1.2) with a configurable CLI interface.

| File | Change |
|---|---|
| `examples/chat_with_tokenizer.rs` | Added `sample_token()` with temperature/top-k/repetition penalty; added `--temperature`, `--top-k`, `--repetition-penalty`, `--system-prompt` CLI flags |
| `Cargo.toml` | Added `rand = "0.8"` to dev-dependencies for sampling |

### 4. Training Loss on Role Labels

**Problem:** `TrainingWindows::from_sequences` set `loss_mask = 1.0` for ALL tokens in every sequence. Model learned to generate `system:`, `user:`, `assistant:` labels as if they were content. During inference, model output contained "istant", "ass", "user:", and other label fragments.

**Fix:** Added `TrainingWindows::from_chat_sequences` that creates loss masks — `0.0` for prompt tokens (system + user), `1.0` for response tokens (assistant content only). Added corresponding `train_chat_sequences` method and updated `load_samples` to return `(prompt, response)` pairs from JSONL data.

| File | Change |
|---|---|
| `src/model.rs` | Added `TrainingWindows::from_chat_sequences` with loss masking; added `train_chat_sequences` method |
| `src/training.rs` | Added `train_on_chat_sequences_with_callback` and `train_on_chat_sequences` |
| `examples/train_with_tokenizer.rs` | `load_samples` returns `Vec<(String, String)>` pairs; uses `train_chat_sequences` for chat data |

### 5. Evaluation OOM (Multiple Rounds)

**Problem:** Training completed but evaluation crashed — both training model and eval model on GPU simultaneously. Even after `drop(trainer)`, evaluation still OOM'd because `evaluate_on_sequences` was in `impl<B: AutodiffBackend>` block, causing VRAM growth on every forward pass.

**Fix:**
1. Moved `evaluate_on_sequences` from `impl<B: AutodiffBackend>` to `impl<B: Backend>` block (it doesn't need autodiff)
2. Used `.valid()` to convert eval model to inference-only backend
3. Used `inner_device` for evaluation calls
4. Set eval batch_size back to 4 (works fine with inference-only model)

| File | Change |
|---|---|
| `src/model.rs` | Moved `evaluate_on_sequences` to `Backend` impl; added `forward_logits()` method |
| `examples/train_with_tokenizer.rs` | `drop(trainer)` before eval; eval model uses `.valid()` |

---

## Modified Files

| File | Change |
|---|---|
| `src/inference.rs` | `ChatModel` uses `.valid()` for inference-only backend; added `predict_logits()` method; `InferenceDevice` type |
| `src/model.rs` | `evaluate_on_sequences` moved to `Backend` impl; `forward_logits()` method; `train_chat_sequences`; `TrainingWindows::from_chat_sequences` with loss masking |
| `src/runtime.rs` | Added `InferenceDevice` type alias |
| `src/training.rs` | Added `train_on_chat_sequences_with_callback` and `train_on_chat_sequences` |
| `examples/chat_with_tokenizer.rs` | Fixed prompt format; temperature/top-k/repetition penalty sampling; CLI flags; manual autoregressive loop |
| `examples/train_with_tokenizer.rs` | `load_samples` returns prompt/response pairs; chat-format detection; `train_chat_sequences`; inference-only eval |
| `Cargo.toml` | Added `rand = "0.8"` to dev-dependencies |
| `README.md` | Added chat CLI options table, loss masking docs, sampling docs, architecture section |

---

## Verification

| Check | Status |
|---|---|
| `cargo check --features cuda --examples` | ✅ |
| `cargo test --lib --tests` | ✅ (25 tests) |
| `cargo clippy --features cuda --examples -- -D warnings` | ✅ |

---

## Training Results (50m-1k)

- 1000 steps completed successfully
- Final loss: 2.113813, Best loss: 1.781314
- Chat-format data detected → loss masking applied
- Evaluation completed without OOM (inference-only backend)
- Model generates coherent Thai text fragments with sampling (temperature=0.8, top_k=40)

---

## Action Items & Next Steps

### Immediate

- **Re-train with 30k steps** using the new loss-masking pipeline:
  ```bash
  cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/50m-30k-v2 \
    --budget 50m --seq-len 256 --batch-size 4 --steps 30000
  ```
- **Chat with the new model:**
  ```bash
  cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/50m-30k-v2
  ```

### Future Improvements

- **Cache tokenized data** — save tokenized sequences as binary to avoid re-tokenizing 100k samples (~30-60s overhead)
- **Parallel tokenization** — use `rayon` for parallel SentencePiece encoding
- **More training data** — 100k samples with 20 unique response patterns is too repetitive; need more diverse assistant responses
- **Learning rate scheduling** — cosine/linear warmup not yet supported
- **Top-p (nucleus) sampling** — in addition to top-k for better quality
- **Godot + Rust dating sim** — scaffold a GDExtension project using `godot-rust` + `multiscreen-rs` for NPC dialogue generation
