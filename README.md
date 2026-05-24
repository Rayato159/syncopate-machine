# 🤖 syncopate-machine

> tiny transformer energy for your NPCs and chatbots — no tokenizer, no GPU required at runtime, no 7B parameter cosplay 💀

a decoder-only transformer built on [Burn](https://github.com/tracel-ai/burn) that predicts **action sequences** (not raw text) for game NPCs, chatbots, or literally anything that needs a tiny brain in the browser. runs on WebGPU in production, trains on CUDA, weighs **~71K parameters** at the smallest preset. yes, it actually works.

---

## 🧠 how it works

```
user text
  → classify into Action + Language
  → [action_id, lang_id, SEP]
  → tiny transformer predicts next action IDs
  → response graph turns actions into replies
  → profit 📈
```

the key insight: **separate "what to say" from "how to say it."** the model only predicts discrete action categories — your response graph handles the actual words. this means a 71K-param model can run diverse, non-repetitive conversations because it never generates free text.

---

## ⚡ quick start

### add the crate

```toml
[dependencies]
syncopate-machine = "0.2"
```

### training

```rust,no_run
use syncopate_machine::prelude::*;

fn main() -> syncopate_machine::Result<()> {
    let mut trainer = Trainer::builder()
        .vocab_size(23)
        .budget(ParameterBudget::Params1M)
        .device(auto_device()?)
        .batch_size(16)
        .seq_len(64)
        .steps(50_000)
        .build()?;

    // sequences of integer IDs — no tokenizer needed
    let sequences = vec![
        vec![5, 21, 3, 9, 2],
        vec![7, 22, 3, 11, 2],
    ];

    let report = trainer.train_on_token_sequences(&sequences)?;
    println!("final loss: {:.4}", report.final_loss);
    Ok(())
}
```

### inference

```rust,no_run
use syncopate_machine::prelude::*;

fn main() -> syncopate_machine::Result<()> {
    let model = ChatModel::load("checkpoints/latest.mpk")?;

    // one-shot generation
    let ids = model.generate(&[5, 21, 3], GenerationConfig::default())?;

    // streaming — get tokens one by one
    model.generate_stream(&[5, 21, 3], GenerationConfig::default(), |id, _| {
        print!("{id} ");
        true // return false to stop
    })?;

    Ok(())
}
```

### 🌐 wasm (browser mode)

```toml
[dependencies]
syncopate-machine = { version = "0.2", features = ["wasm"] }
```

then build with `trunk build --release` and you're running a transformer in the browser. WebGPU first, CPU fallback for cursed environments.

---

## 🔢 action vocab

your vocabulary is just integer IDs. here's an example layout:

```
 0 PAD    1 SOS    2 EOS    3 SEP
 4 Unknown
 5 Greeting       6 Farewell       7 Frustrated
 8 Sad            9 Happy          10 Question
 11 Insult        12 Compliment    13 Agree
 14 Disagree      15 General
 16 Eating        17 DailyLife     18 RustGo
 19 Identity      20 ShitTalk
 21 TH            22 EN
```

vocab size: **23 IDs**. no SentencePiece. no BPE. just vibes (and integers).

---

## 📐 the math

this is a **LLaMA-style decoder-only transformer** with grouped-query attention (GQA), rotary position embeddings (RoPE), RMSNorm, and SwiGLU feed-forward.

for the full mathematical deep-dive with worked examples and calculations, see **[slides.tex](slides.tex)** — a 17-slide Beamer presentation covering every component from RMSNorm to the 71K parameter breakdown.

### 📊 parameter presets

| preset | layers | $d$ | heads | KV heads | FFN | params |
|--------|--------|-----|-------|----------|-----|--------|
| action | 2 | 64 | 4 | 1 | 128 | ~71K |

the smallest preset (action) weighs **71,424 params** — small enough to load in the browser in under 100ms. the `.mpk` checkpoint? **286 KB**. try that with your 7B model 😌

---

## 🏋️ training

### generate data

```bash
python examples/build_action_data.py --offline
```

### train the model

```bash
cargo run --release --features cuda --example train_action_model -- \
    --steps 3000 --batch-size 32 --lr 0.003
```

training is **CUDA-only** by policy. CPU/Flex is the runtime fallback for the browser, not the training lane.

outputs land in `runs/action-model/`:

```
final.mpk          ← the checkpoint
model-config.json  ← model dimensions
loss.csv           ← per-step loss
report.json        ← training summary
```

### ship to production

copy the checkpoint and config to wherever your app loads them:

```bash
cp runs/action-model/final.mpk your-app/assets/model.mpk
cp runs/action-model/model-config.json your-app/assets/model-config.json
```

---

## 🎛️ feature flags

| flag | what it does |
|------|-------------|
| (default) | CPU inference via Burn Flex with auto SIMD |
| `cuda` | NVIDIA GPU acceleration for training |
| `wasm` | WebGPU browser runtime (`wasm32-unknown-unknown`) |

---

## 🧪 crate philosophy

- the model predicts **integer IDs**, not text tokens — your vocabulary is whatever you want
- `Trainer` and `ChatModel` stay generic — give them any ID sequences
- big checkpoints, generated data, and random artifacts don't belong in the crate
- weight-tied output projection = fewer params, faster loading

---

## 📜 license

MIT. break it, fix it, ship it. 🚀
