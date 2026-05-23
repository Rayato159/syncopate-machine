# syncopate-machine 🧟

Small Rust decoder-only Transformer for game NPC chat. It is built for school
zombie survival, awkward flirting, and not pretending a 5M model is a deity.

Stack: Burn + SentencePiece + RoPE + RMSNorm + GQA + SwiGLU + optional
higher-order attention.

## ⚡ Quick Start

Prepare the bundled NPC data and tokenizer first:

```powershell
& "C:\Users\HashTable\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" examples\build_school_zombie_data.py
.\experiment\train\spm_train.exe --input=examples\data\tokenizer_corpus.spm --model_prefix=examples\data\tokenizer --vocab_size=4096 --model_type=bpe --character_coverage=0.9995 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false --split_digits=true --normalization_rule_name=nmt_nfkc --max_sentence_length=2048 "--user_defined_symbols=<mew>,<jin>,<th>,<en>"
```

Train with CUDA and the experimental higher-order attention kernel:

```powershell
cargo run --release --features cuda --example train_with_tokenizer -- --train-dir examples/data --run-dir runs/school-zombie-npc-5m-lang-ho --budget 5m --steps 2500 --batch-size 4 --seq-len 128 --lr 0.0002 --min-lr 0.00002 --warmup-steps 200 --weight-decay 0.1 --grad-clip 1.0 --attention-kernel higher-order --val-split 0.15 --eval-interval 100 --checkpoint-interval 250 --log-interval 50 --latency-tokens 32
```

Chat with the checkpoint:

```powershell
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/school-zombie-npc-5m-lang-ho --checkpoint runs/school-zombie-npc-5m-lang-ho/checkpoints/best_val.mpk --npc mew
```

Evaluate an existing checkpoint without training again:

```powershell
cargo run --release --features cuda --example train_with_tokenizer -- --train-dir examples/data --run-dir runs/school-zombie-npc-5m-lang-ho --eval-only
```

Switch NPC or force language while chatting:

```text
/npc jin
/lang th
/lang en
/lang auto
```

## 🎛️ Knobs That Actually Matter

Train command parameters:

| Parameter | Use it for |
|---|---|
| `--train-dir` | Folder with `tokenizer.model` and train files. No tokenizer, no party. |
| `--run-dir` | Output folder for checkpoints, reports, cache, and loss CSV. |
| `--budget` | Model size preset: `1m`, `5m`, `10m`, `50m`, `100m`. Bigger is not a personality patch. |
| `--steps` | Optimizer steps. Watch validation loss; overfitting does not send an invitation. |
| `--batch-size` | Samples per step. Raise it if VRAM allows. |
| `--seq-len` | Context length. Current NPC recipe uses `128`. |
| `--lr` | Peak learning rate after warmup. |
| `--min-lr` | Final cosine-decay learning rate. |
| `--warmup-steps` | Linear warmup steps before cosine decay. |
| `--weight-decay` | AdamW regularization. Helps stop tiny models from memorizing like a cursed notebook. |
| `--grad-clip` | Gradient norm cap. `0` disables it. |
| `--attention-kernel` | `softmax` for boring stability, `higher-order` for the spicy experiment. |
| `--val-split` | Fraction held out for validation. |
| `--eval-interval` | Steps between validation runs; also saves `best_val.mpk`. |
| `--checkpoint-interval` | Steps between checkpoint saves. `0` means only best/latest behavior. |
| `--log-interval` | Steps between console loss logs. |
| `--latency-tokens` | Generated-token count for the latency benchmark after training. |
| `--max-samples` | Load only N samples. `0` means all data. |
| `--eval-only` | Skip training and evaluate/report from an existing checkpoint. |

Chat/test command parameters:

| Parameter | Use it for |
|---|---|
| `--run-dir` | Run folder containing `checkpoints/latest.mpk` and usually `tokenizer.model`. |
| `--checkpoint` | Manual checkpoint path if `latest.mpk` is not the one. |
| `--npc` | `mew` or `jin`; if omitted in interactive mode, the CLI asks. |
| `--prompt` | One-shot test message. Omit it for interactive chat. |
| `--lang` | `auto`, `th`, or `en`. Use this when the model gets cute and flips language. |
| `--prompt-contract` | `auto`, `plain`, or `npc-lang-v1`. New runs write this automatically; old checkpoints use `plain`. |
| `--max-new-tokens` | Hard cap for generated reply tokens. |
| `--temperature` | `0` is greedy, higher is more random. Default is conservative because valid JSON beats jazz. |
| `--top-k` | Sample only from the top K token candidates. Default is `10`. |
| `--repetition-penalty` | Penalizes recent repeated tokens. Default is `1.1`. |
| `--show-raw` | Prints raw generated JSON text after parsed fields. Useful when the model lies badly. |
| `--language-retries` | Extra attempts when parsed reply language does not match the input. |
| `--retry-temperature` | Temperature for those language-match retries. |

## 🛠️ Prepare Train Data

The lazy path:

```powershell
& "C:\Users\HashTable\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" examples\build_school_zombie_data.py
```

That rebuilds the source data:

```text
datasets.jsonl
tokenizer_corpus.spm
personas.json
dataset.schema.json
```

Then `spm_train.exe` creates:

```text
tokenizer.model
tokenizer.vocab
```

The not-lazy path:

1. Put your JSONL rows in `examples/data/datasets.jsonl`.
2. Keep `input` as natural player text.
3. Keep `output` as a JSON string.
4. Put every useful text fragment into `examples/data/tokenizer_corpus.spm`.
5. Train SentencePiece.

Tokenizer command:

```powershell
.\experiment\train\spm_train.exe --input=examples\data\tokenizer_corpus.spm --model_prefix=examples\data\tokenizer --vocab_size=4096 --model_type=bpe --character_coverage=0.9995 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false --split_digits=true --normalization_rule_name=nmt_nfkc --max_sentence_length=2048 "--user_defined_symbols=<mew>,<jin>,<th>,<en>"
```

Quick sanity check:

```powershell
(Get-Content examples\data\datasets.jsonl | Measure-Object -Line).Lines
(Select-String -Path examples\data\tokenizer.vocab -Pattern '^<0x' | Measure-Object).Count
```

Expected: around `10000` rows and `0` byte-fallback tokens. If this says
`PERSONA=...` or vomits `<0x00>` through `<0xFF>`, the data pipeline is haunted.
Delete the stale files and rebuild.

## 🧾 Dataset Contract

Active data lives in `examples/data`.

Rows are JSONL. Keep the player input natural. Do not shove a tax form into
`input`; the model is tiny, not a bureaucrat.

```json
{"npc":"mew","input":"Are you okay?","output":"{\"message\":\"I'm okay. Let's keep going.\",\"mood\":\"shy\",\"relation_point\":1}"}
```

Required fields:

| Field | Meaning |
|---|---|
| `npc` | `mew` or `jin` |
| `input` | Natural player message, Thai or English |
| `output` | JSON string with `message`, `mood`, `relation_point` |

Allowed moods:

```text
normal, happy, sad, shy, mad, scary
```

`relation_point` must be `-1`, `0`, or `1`.

The trainer prepends `<mew>/<jin>` and `<th>/<en>` internally, so runtime chat
and training share the same tiny brain without leaking metadata sludge into the
visible prompt.

Other loader formats still exist for experiments:

| Format | Expected shape |
|---|---|
| CSV | `prompt,response` |
| JSONL text | `{"text":"..."}` |
| JSONL chat | `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` |

For character SFT, use the NPC format. Generic assistant data will make generic
assistant soup. Delicious only if you hate personality.

## 💬 Chat Shape

Interactive output is intentionally boring and game-friendly:

```text
me>
Are you okay?

mew>
message: I'm okay. Let's keep going.
mood: shy
point: 1
```

One-shot mode:

```powershell
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/school-zombie-npc-5m-lang-ho --npc jin --prompt "Stay close to me." --lang en
```

## 🧠 Model Math

Full math slides live here: [syncopate_model_math_slides.tex](syncopate_model_math_slides.tex).

Input token IDs:

```text
x = [x_1, ..., x_T], T <= seq_len
h_0 = E[x] + RoPE position signal inside attention
```

Each decoder block:

```text
a_l = Attention(RMSNorm(h_l))
h'_l = h_l + a_l
f_l = W_down(SiLU(W_gate RMSNorm(h'_l)) * W_up RMSNorm(h'_l))
h_{l+1} = h'_l + f_l
```

Grouped-query attention:

```text
Q = XW_q
K = XW_k
V = XW_v
S = (RoPE(Q) RoPE(K)^T) / sqrt(d_head)
```

Default attention:

```text
Attention(X) = softmax(causal_mask(S)) V
```

Higher-order attention:

```text
W = causal_mask(S)
H = causal_mask(W W^T)
Attention(X) = (H V) / (sum(abs(H)) + eps)
```

Final logits with tied embeddings:

```text
z = RMSNorm(h_L)
logits = z E^T
loss = cross_entropy(next_token), masked so prompt tokens do not get graded
```

Presets:

| Budget | Layers | Width | Heads | KV Heads | FFN |
|---|---:|---:|---:|---:|---:|
| `1m` | 2 | 96 | 4 | 1 | 256 |
| `5m` | 8 | 192 | 6 | 2 | 512 |
| `10m` | 10 | 256 | 8 | 2 | 704 |
| `50m` | 16 | 512 | 8 | 2 | 1408 |
| `100m` | 15 | 768 | 12 | 4 | 2048 |

## 📦 Outputs

Training writes:

```text
runs/.../checkpoints/latest.mpk
runs/.../checkpoints/best_val.mpk
runs/.../loss.csv
runs/.../report.json
runs/.../report.md
```

Plot loss:

```powershell
python examples/plot_loss.py runs/school-zombie-npc-5m-ho-cuda/loss.csv
```

## 🪪 License

MIT. Use it, break it, fix it, do not blame the model for bad data.
