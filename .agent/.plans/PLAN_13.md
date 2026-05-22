# PLAN_13: Miu NPC Tokenizer + Training Pipeline Setup

**Status:** ✅ Complete
**Date:** 2025-05-14

---

## Overview

Re-trained the SentencePiece tokenizer from `miu_corpus.txt` (100k Thai sentences for the หมิว NPC character), fixed encoding issues in `train.bat`, and set up the full training pipeline to use the 100k dialogue dataset (`miu_dialogues_100k.csv`). Verified the pipeline works end-to-end with a 50-step training run.

---

## Problems & Fixes

### 1. Thai Encoding in `train.bat` — `user_defined_symbols` Corrupted

**Problem:** `train.bat` used `set RESERVED_SYMBOLS=หมิว,ธันวา,ซอมบี้,...` directly in the batch file. Windows batch files don't reliably handle UTF-8 Thai text in `set` commands, causing SentencePiece to see `????` instead of Thai characters → "duplicated symbols are not allowed" error.

**Fix:** Changed to `--user_defined_symbols_file=user_symbols.txt`, generating the file at runtime via `echo` commands with proper encoding.

| File | Change |
|---|---|
| `experiment/train/train.bat` | `--user_defined_symbols` → `--user_defined_symbols_file=user_symbols.txt` |

### 2. Vocabulary Size Too High for Corpus

**Problem:** `--vocab_size=16000` with `--character_coverage=1.0` and `--byte_fallback=true` but the corpus only has 54 unique characters → SentencePiece could only generate 472 pieces max. Error: "Vocabulary size too high (16000). Please set it to a value <= 472."

**Fix:**
- Reduced `vocab_size` from 16000 → 400
- Disabled `byte_fallback` (not needed with full character coverage)
- Enabled `split_by_whitespace=true` (better for Thai subword segmentation)
- Added `--hard_vocab_limit=false` (allows SentencePiece to use fewer pieces if corpus doesn't need more)

| File | Change |
|---|---|
| `experiment/train/train.bat` | `vocab_size=400`, `byte_fallback=false`, `split_by_whitespace=true`, `hard_vocab_limit=false` |

---

## Tokenizer Details

| Parameter | Value |
|---|---|
| Input | `miu_corpus.txt` (100,000 sentences) |
| Model | `miu_tokenizer.model` |
| Vocab | `miu_tokenizer.vocab` (211 tokens) |
| Model type | Unigram |
| Character coverage | 1.0 |
| User-defined symbols | หมิว, ธันวา, ซอมบี้, แว่น, เสบียง, เลนส์ |
| Special tokens | `<unk>=0`, `<s>=1`, `</s>=2` |

---

## Training Data Setup

### Files in `examples/data/`

| File | Description | Size |
|---|---|---|
| `tokenizer.model` | Trained SentencePiece model (from `miu_corpus.txt`) | 5.7 KB |
| `miu_corpus.txt` | 100k Thai sentences (zombie survival + หมิว character lore) | ~9 MB |
| `miu_dialogues_100k.jsonl` | 100k dialogue pairs converted from CSV | ~50 MB |
| `.gitkeep` | Placeholder | — |

### Data Pipeline

```
miu_corpus.txt ──→ spm_train ──→ miu_tokenizer.model + .vocab
                                        │
miu_dialogues_100k.csv ──→ csv_to_jsonl.py ──→ miu_dialogues_100k.jsonl
                                        │
                                        ▼
                    train_with_tokenizer.rs reads examples/data/
                    - .txt files → plain text samples (pre-training)
                    - .jsonl files → chat-format samples (fine-tuning)
                    - tokenizer.model → tokenization
```

### CSV → JSONL Conversion

Used existing `csv_to_jsonl.py`:

```bash
python examples/game_data/csv_to_jsonl.py \
    examples/game_data/miu_dialogues_100k.csv \
    -o examples/data/miu_dialogues_100k.jsonl
```

Output format (OpenAI chat):
```json
{"messages": [
  {"role": "system", "content": "You are หมิว NPC. Always respond in JSON: {\"message\": \"...\", \"mood\": \"...\"}. moods: shy, happy, sad, angry, nervous, blush, calm"},
  {"role": "user", "content": "วางแผนยังไงดี"},
  {"role": "assistant", "content": "{\"message\": \"แผนที่บอกว่า...\", \"mood\": \"calm\"}"}
]}
```

### Dataset Stats (from CSV)

- **Total rows:** 100,000
- **Unique prompts:** 25
- **Mood distribution:**
  - blush: 24,047 (24.0%)
  - happy: 16,065 (16.1%)
  - calm: 15,901 (15.9%)
  - sad: 12,035 (12.0%)
  - nervous: 11,964 (12.0%)
  - angry: 8,029 (8.0%)
  - shy: 7,973 (8.0%)
  - panic: 3,986 (4.0%)

---

## Training Test Run (50 steps)

```bash
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/miu-model-test \
    --steps 50 --log-interval 10 \
    --batch-size 2 --seq-len 128
```

### Results

| Metric | Value |
|---|---|
| Vocab size | 211 |
| Samples loaded | 100,001 |
| Deduped sequences | 126 (2,700,809 tokens) |
| Split | 100 train / 13 val / 13 test |
| Model params | 6,399,526 (~6.4M, budget=10m) |
| Device | autodiff\<flex\> (CPU) |
| Training time | 16.5s (3.0 steps/s) |
| Final train loss | 4.985222 |
| Best train loss | 4.300650 |
| Val loss | 19.9270 (ppl=451M) |
| Val accuracy | 0.30% |
| Inference | 45ms/token |
| Checkpoint | `runs/miu-model-test/checkpoints/latest.mpk` |

Loss decreased consistently: 22.5 → 4.3 in 50 steps, confirming the pipeline works.

---

## Modified Files

| File | Change |
|---|---|
| `experiment/train/train.bat` | Fixed Thai encoding (user_symbols_file), reduced vocab_size=400, byte_fallback=false, split_by_whitespace=true, hard_vocab_limit=false |
| `experiment/train/miu_corpus.txt` | Input corpus (already existed) |
| `experiment/train/miu_tokenizer.model` | **Generated** — trained tokenizer model |
| `experiment/train/miu_tokenizer.vocab` | **Generated** — trained tokenizer vocab (211 tokens) |
| `examples/data/tokenizer.model` | **Copied** from experiment/train/ |
| `examples/data/miu_corpus.txt` | **Copied** from experiment/train/ |
| `examples/data/miu_dialogues_100k.jsonl` | **Generated** from CSV via csv_to_jsonl.py |

---

## How to Train for Real

### Full training (recommended):

```bash
# CPU (flex backend)
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/miu-model \
    --budget 10m \
    --steps 5000 \
    --batch-size 4 \
    --seq-len 256 \
    --lr 0.0002 \
    --val-split 0.1 \
    --log-interval 100

# GPU (cuda backend) — if available
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/miu-model-cuda \
    --budget 10m \
    --steps 10000 \
    --batch-size 8 \
    --seq-len 256 \
    --lr 0.0002 \
    --val-split 0.1 \
    --log-interval 100
```

### Chat with the trained model:

```bash
cargo run --release --example chat_with_tokenizer -- --run-dir runs/miu-model
```

### Retrain tokenizer (if corpus changes):

```bash
cd experiment/train && train.bat
```

Then copy the new model:
```bash
cp experiment/train/miu_tokenizer.model examples/data/tokenizer.model
```

---

## Action Items & Next Steps

### Immediate

- **Full training run** — 5,000–10,000 steps with `budget=10m` or `budget=50m` on GPU
- **Evaluate chat quality** — test with `chat_with_tokenizer` after training
- **Adjust hyperparams** — try higher `seq-len=256` or `512` for longer context

### Data Improvements

- **More unique prompts** — only 25 unique prompts across 100k rows is too repetitive; need more diverse user inputs
- **Balance mood distribution** — `panic` has only 4% while `blush` has 24%
- **Add conversation context** — current data is single-turn; multi-turn would improve coherence
- **Consistent system prompt** — corpus uses narrative style, JSONL uses JSON-output format; aligning these would help

### Technical Improvements

- **Cache tokenized data** — avoid re-tokenizing 100k samples on every run (~17s overhead)
- **Learning rate scheduling** — cosine warmup for better convergence
- **Mixed precision training** — f16/bf16 for faster GPU training
- **Top-p (nucleus) sampling** — in addition to top-k
- **GRU/RWKV attention** — for better long-context handling

---

## Verification

| Check | Status |
|---|---|
| `train.bat` runs successfully | ✅ (miu_tokenizer.model + .vocab generated) |
| Tokenizer: 211 vocab, Thai user symbols | ✅ |
| CSV → JSONL conversion: 100k samples | ✅ |
| `train_with_tokenizer` loads all data | ✅ (100,001 samples, 126 deduped sequences) |
| Training runs: 50 steps, loss 22.5→4.3 | ✅ |
| Checkpoint saved | ✅ (`runs/miu-model-test/checkpoints/latest.mpk`) |
| Report generated | ✅ (`runs/miu-model-test/report.md`) |
