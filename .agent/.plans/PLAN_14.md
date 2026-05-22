# PLAN_14: Train Model with Datasets_5000.csv (General Thai Instruction)

**Status:** ✅ Complete (data prepared, ready to train)
**Date:** 2025-05-14

---

## Overview

Used `Datasets_5000.csv` (6,000 rows of Thai instruction-following data) as the training dataset. Created a corpus for SentencePiece tokenizer training, trained a new tokenizer, and converted the CSV to JSONL format for model training.

---

## Dataset: Datasets_5000.csv

| Property | Value |
|---|---|
| **File** | `examples/data/Datasets_5000.csv` |
| **Columns** | `No`, `HumanEval`, `instruction`, `input`, `output` |
| **Total rows** | 6,000 |
| **Converted to JSONL** | 5,005 rows |
| **Skipped** | 995 rows (empty instruction or output) |
| **Content** | General Thai instruction-following (translations, explanations, stories, Q&A) |

---

## Pipeline

### 1. Corpus Creation

Script: `examples/data/build_corpus_from_csv.py`

```bash
python examples/data/build_corpus_from_csv.py examples/data/Datasets_5000.csv
```

- Extracts text from `instruction`, `input`, and `output` columns
- Each non-empty field → one line (multi-line fields collapsed to single line)
- Output: `experiment/train/datasets_corpus.txt`
- Result: **12,102 lines** from 6,000 CSV rows

### 2. Tokenizer Training

Script: `experiment/train/train_datasets.bat`

```bash
cd experiment/train && train_datasets.bat
# Or directly:
cd experiment/train && spm_train.exe --input=datasets_corpus.txt ...
```

| Parameter | Value |
|---|---|
| **Input** | `datasets_corpus.txt` (12,102 lines) |
| **Model prefix** | `datasets_tokenizer` |
| **Vocab size** | 800 |
| **Model type** | Unigram |
| **Character coverage** | 1.0 |
| **byte_fallback** | false |
| **split_by_whitespace** | true |
| **hard_vocab_limit** | false |
| **Result** | `datasets_tokenizer.model` + `datasets_tokenizer.vocab` (800 tokens) |

Tokenizer copied to `examples/data/tokenizer.model` (11,862 bytes).

### 3. CSV → JSONL Conversion

Script: `examples/data/csv_to_jsonl_general.py`

```bash
python examples/data/csv_to_jsonl_general.py examples/data/Datasets_5000.csv -o examples/data/Datasets_5000.jsonl
```

- Output: `examples/data/Datasets_5000.jsonl` (4.37 MB)
- Format: OpenAI chat format (`user` + `assistant` messages)
- If `input` is non-empty: `user = instruction + "\n" + input`
- If `input` is empty: `user = instruction`
- 5,005 samples converted, 995 skipped

---

## Files Created

| File | Description |
|---|---|
| `examples/data/build_corpus_from_csv.py` | Extract corpus from CSV for tokenizer training |
| `experiment/train/train_datasets.bat` | SentencePiece training batch file for this dataset |
| `experiment/train/datasets_corpus.txt` | Corpus (12,102 lines) |
| `experiment/train/datasets_tokenizer.model` | Trained tokenizer model (800 vocab) |
| `experiment/train/datasets_tokenizer.vocab` | Trained tokenizer vocab |
| `examples/data/csv_to_jsonl_general.py` | CSV → JSONL converter for instruction format |
| `examples/data/tokenizer.model` | Tokenizer model (copied from experiment/train/) |
| `examples/data/Datasets_5000.jsonl` | Training data in JSONL format (5,005 samples) |

---

## How to Train

### CPU (flex backend):
```bash
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/datasets-5000 \
    --budget 10m \
    --steps 10000 \
    --batch-size 4 \
    --seq-len 256 \
    --lr 0.0002 \
    --val-split 0.1 \
    --log-interval 100
```

### GPU (CUDA backend):
```bash
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/datasets-5000-cuda \
    --budget 10m \
    --steps 10000 \
    --batch-size 16 \
    --seq-len 256 \
    --lr 0.0001 \
    --val-split 0.1 \
    --log-interval 100
```

### Larger model (50M params) for better quality:
```bash
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/datasets-5000-50m \
    --budget 50m \
    --steps 20000 \
    --batch-size 16 \
    --seq-len 256 \
    --lr 0.0001 \
    --val-split 0.1 \
    --log-interval 100
```

## How to Chat

```bash
# With CPU model
cargo run --release --example chat_with_tokenizer -- --run-dir runs/datasets-5000

# With CUDA model
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/datasets-5000-cuda
```

---

## Differences from Previous Miu NPC Pipeline (PLAN_13)

| Aspect | Miu NPC (PLAN_13) | Datasets_5000 (PLAN_14) |
|---|---|---|
| **Data type** | NPC dialogue (prompt/message/mood) | General Thai instructions |
| **Vocab size** | 400 (211 actual) | 800 |
| **User-defined symbols** | หมิว, ธันวา, ซอมบี้, etc. | None |
| **JSONL format** | system + user + assistant (with mood JSON) | user + assistant only |
| **Unique samples** | 126 (very repetitive) | ~5,005 (diverse) |
| **System prompt** | Miu NPC character prompt | No system prompt |

---

## Verification

| Check | Status |
|---|---|
| Corpus created: 12,102 lines | ✅ |
| Tokenizer trained: 800 vocab | ✅ |
| Tokenizer copied to `examples/data/tokenizer.model` | ✅ |
| JSONL created: 5,005 samples | ✅ |
| Ready to train | ✅ |
