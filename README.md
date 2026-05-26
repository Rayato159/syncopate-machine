# syncopate-machine ⚡

Tiny Burn transformer for **Dancing With My Code** chat and his game. It predicts action IDs,
then the website response graph turns those actions into short Thai/English
messages and mood GIFs. Small brain, fast vibes.

```text
user text -> website classifier -> [action_id, lang_id, SEP]
          -> syncopate-machine -> next action IDs
          -> response graph -> message + mood GIF
```

## Quick Start 🚀

Build the action data:

```powershell
python examples\build_action_data.py --offline
```

Train on CUDA:

```powershell
cargo run --release --features cuda --example train_action_model -- `
  --steps 20000 --batch-size 64 --lr 0.003 `
  --checkpoint-dir runs/action-v4-d16-clean
```

Ship these into the website:

```text
dancing-with-my-code-v2/assets/model-personal-v2.mpk
dancing-with-my-code-v2/assets/model-config-personal-v2.json
```

## Action Vocabulary 📝

15 tokens — control codes, action intents, and language tags:

| ID | Token | Role |
| --- | --- | --- |
| 0 | `PAD` | padding |
| 1 | `SOS` | start of sequence |
| 2 | `EOS` | end of sequence |
| 3 | `SEP` | separator |
| 4 | `Unknown` | unclassified input |
| 5 | `Greeting` | สวัสดี / hello vibes |
| 6 | `Farewell` | บ๊ายบาย / goodbye |
| 7 | `Insult` | ด่า / roast mode |
| 8 | `Programming` | code talk |
| 9 | `Identity` | who am I? |
| 10 | `Resume` | แนะนำตัว / intro |
| 11 | `Links` | link drops |
| 12 | `Course` | course info |
| 13 | `TH` | Thai language tag |
| 14 | `EN` | English language tag |

## Model Vibe 🧠

| Setting | Value |
| --- | --- |
| Task | causal action-sequence prediction |
| Vocab | 15 action/lang/control IDs |
| Sequence length | 16 |
| Layers | 1 |
| d_model | 16 |
| Attention | causal scaled dot-product softmax |
| Attention heads | 4 |
| KV heads | 1 |
| FFN | SwiGLU, width 16 |
| Position encoding | RoPE |
| Norm | RMSNorm |
| Output projection | tied to embedding |
| Params | 1,696 |

## Current Checkpoint 💾

The website uses:

```text
runs/action-v4-d16-clean/final.mpk
```

Current validation metrics:

| Metric | Value |
| --- | --- |
| Validation loss | 0.6742 |
| Validation perplexity | 1.96 |
| Validation accuracy | 71.15% |

## Training Loss 📉

![Training loss curve](docs/loss-action-model.png)

Raw loss is noisy because the batches are tiny. The pink line is an 80-step
smooth, which shows the real shape: hard drop early, then slow grind.

## Browser Runtime 🌐

| Mode | Behavior |
| --- | --- |
| `auto` | try WebGPU, fall back to CPU |
| `gpu` | require WebGPU |
| `cpu` | force CPU/Flex backend |

MIT. Break it, fix it, ship it. ✨
