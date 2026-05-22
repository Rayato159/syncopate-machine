# syncopate-machine

`syncopate-machine` is a small Rust transformer for game NPC chat. It trains and
runs local token-level language models with [Burn](https://github.com/tracel-ai/burn),
with a simple SentencePiece example pipeline for quick experiments.

The neural model is no longer the old Multiscreen tile stack. That path was the
wrong weapon for this job: no real transformer block, no FFN, no RoPE, no
RMSNorm, and not enough boring stability machinery. The current model is a
compact decoder-only LM built from pieces that actually matter for small NPC
chat models:

- GPT-1 style causal next-token training with tied token embeddings.
- TinyLlama/Llama-style RoPE, pre-norm RMSNorm, SwiGLU, and grouped-query attention.
- Optional normalized second-order higher-order attention for experiments.
- AdamW with weight decay, gradient clipping, warmup, and cosine decay.

Current status: the code compiles and the test suite passes. No new model has
been trained from this architecture yet, so do not treat any old run directory
as proof of quality for `syncopate-machine`.

## Quickstart

```bash
# Train a small model on tokenizer.model + text/jsonl/csv files in examples/data
cargo run --release --example train_with_tokenizer -- \
    --budget 10m --steps 10000 --batch-size 16 --seq-len 256

# Chat with the trained checkpoint
cargo run --release --example chat_with_tokenizer -- --run-dir runs/my-model
```

Enable CUDA when the project is built with the `cuda` feature:

```bash
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/npc-chat --budget 10m
```

## CLI

### Train

```bash
# CPU smoke run
cargo run --release --example train_with_tokenizer -- \
    --budget 1m --steps 500 --batch-size 4 --seq-len 128

# NPC/chat dataset
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data \
    --run-dir runs/npc-chat \
    --budget 10m \
    --steps 50000 \
    --batch-size 16 \
    --seq-len 256 \
    --lr 0.0003 \
    --min-lr 0.00003 \
    --warmup-steps 200 \
    --weight-decay 0.1 \
    --grad-clip 1.0
```

### Chat

```bash
# Interactive streaming
cargo run --release --example chat_with_tokenizer -- --run-dir runs/npc-chat

# One-shot
cargo run --release --example chat_with_tokenizer -- \
    --run-dir runs/npc-chat --prompt "Who are you?"
```

## Data Formats

Put training data beside `tokenizer.model` in any directory.

| Format | Description |
|---|---|
| `.csv` | `prompt,response` columns for NPC dialogue or Q&A |
| `.txt` | Blank lines separate samples; each sample is split into prompt/response continuation |
| `.jsonl` text | `{"text": "..."}` |
| `.jsonl` chat | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |

For NPCs, prefer direct prompt/response data. Generic instruction data is useful
for language coverage, but character behavior comes from character dialogue.

Strong NPC data beats vague general instruction sludge. Include the player
input, the NPC response, and any compact state/mood tags you expect the runtime
to provide. If a character should never sound like a generic assistant, do not
train mostly generic assistant data and hope vibes fix it.

## Library Usage

```toml
[dependencies]
syncopate-machine = "0.1"
# syncopate-machine = { version = "0.3", features = ["cuda"] }
```

```rust
use syncopate_machine::prelude::*;

let mut trainer = Trainer::builder()
    .vocab_size(1000)
    .budget(ParameterBudget::Params10M)
    .device(auto_device()?)
    .batch_size(16)
    .seq_len(256)
    .steps(50_000)
    .learning_rate(3e-4)
    .build()?;

let report = trainer.train_on_chat_sequences(&chat_pairs)?;

let model = ChatModel::load("runs/npc-chat/checkpoints/latest.mpk")?;
let tokens = model.generate(&prompt_ids, GenerationConfig::default())?;
```

## Architecture

The default kernel is standard causal softmax attention because tiny chat models
need stable training before they need clever math. Yes, softmax is boring. Boring
is useful when the dataset is small and the model has to learn a character
instead of performing research theater.

```text
tokens
  -> tied token embedding
  -> repeated decoder blocks:
       RMSNorm -> RoPE GQA causal attention -> residual
       RMSNorm -> SwiGLU feed-forward       -> residual
  -> RMSNorm
  -> tied LM head
```

For higher-order attention experiments:

```rust
let config = SyncopateModelConfig::preset_10m(vocab_size, seq_len)
    .with_attention_kernel(AttentionKernel::HigherOrder);
```

The higher-order kernel is implemented as a normalized second-order causal
parallel form. Treat it as experimental until it has real training curves.

Parameter budgets: `1m`, `5m`, `10m`, `50m`, `100m`.

## Research Basis

The architecture choices are anchored in the local papers under `papers/`:

| Paper | What made it into the implementation |
|---|---|
| `papers/gpt-1.pdf` | Decoder-only causal LM objective, tied embedding projection, warmup plus cosine training recipe |
| `papers/tiny_llama.pdf` | RoPE, pre-norm RMSNorm, SwiGLU, grouped-query attention, AdamW defaults |
| `papers/higher-order.pdf` | Optional normalized second-order causal attention kernel |

The higher-order kernel is exposed for experiments, not as the default religion.
Until it has real loss curves on the NPC data, the stable transformer path is
the sane baseline.

## Reports

Training writes checkpoints, `loss.csv`, `report.json`, and `report.md` under
the selected run directory.

```bash
python examples/plot_loss.py runs/npc-chat/loss.csv
```

## License

[MIT](LICENSE)
