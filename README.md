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

this is a **LLaMA-style decoder-only transformer** with grouped-query attention (GQA), rotary position embeddings (RoPE), RMSNorm, and SwiGLU feed-forward. here's exactly what happens inside:

### model architecture

```
Input tokens → Token Embedding
    ┌─────────────────────────────┐
    │  ×N SyncopateBlock          │
    │  ┌─────────────────────────┐│
    │  │ RMSNorm → Attention → + ││
    │  │ RMSNorm → SwiGLU FFN → +││
    │  └─────────────────────────┘│
    └─────────────────────────────┘
Final RMSNorm → Output Projection (weight-tied)
```

### RMSNorm

replaces LayerNorm. no centering, just scale:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot w$$

### rotary position embedding (RoPE)

encodes position by rotating pairs of dimensions at different frequencies. no learned position embeddings needed:

$$\theta_k = \theta^{-2k / d_{\text{head}}}$$

$$\text{RoPE}(x, m)_{2k:2k+1} = \begin{pmatrix} x_{2k} \cos(m\theta_k) - x_{2k+1} \sin(m\theta_k) \\ x_{2k} \sin(m\theta_k) + x_{2k+1} \cos(m\theta_k) \end{pmatrix}$$

where $m$ is the position index and $\theta = 10000$ (the base frequency).

### grouped-query attention (GQA)

standard multi-head attention but $n_{\text{kv}} < n_{\text{heads}}$ — K/V heads are shared and repeated. saves memory, runs faster:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

where $M$ is a causal mask ($-\infty$ for future positions). with GQA, $K$ and $V$ have `kv_heads` copies, each repeated $\lceil n_{\text{heads}} / n_{\text{kv}} \rceil$ times to match $Q$.

### SwiGLU feed-forward

the FFN uses a gated linear unit with SiLU activation:

$$\text{FFN}(x) = \bigl(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\bigr) W_{\text{down}}$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ and $\sigma$ is the sigmoid function.

### 🧮 worked example: 71K parameter budget

let's trace through the **action preset** — the smallest practical config:

| parameter | value |
|-----------|-------|
| vocab size $V$ | 23 |
| sequence length | 64 |
| layers $N$ | 2 |
| model dimension $d$ | 64 |
| attention heads | 4 |
| KV heads | 1 |
| head dimension $d_k$ | 64 / 4 = 16 |
| FFN intermediate | 128 |

**step 1: embedding layer**

$$P_{\text{embed}} = V \times d = 23 \times 64 = 1{,}472$$

**step 2: attention per layer** (weight-tied output uses same projection)

$$P_{\text{attn}} = \underbrace{d^2}_{W_Q} + \underbrace{2 \cdot d \cdot (n_{\text{kv}} \cdot d_k)}_{W_K + W_V} + \underbrace{d^2}_{W_O} = 64^2 + 2 \times 64 \times 16 + 64^2 = 10{,}240$$

**step 3: SwiGLU FFN per layer** (3 weight matrices)

$$P_{\text{ffn}} = 3 \times d \times d_{\text{ff}} = 3 \times 64 \times 128 = 24{,}576$$

**step 4: RMSNorm per layer** (2 norms × $d$ params each)

$$P_{\text{norm}} = 2 \times d = 2 \times 64 = 128$$

**step 5: total**

$$P_{\text{total}} = \underbrace{1{,}472}_{\text{embed}} + N \times (10{,}240 + 24{,}576 + 128) + \underbrace{64}_{\text{final norm}}$$

$$= 1{,}472 + 2 \times 34{,}944 + 64 = \boxed{71{,}424 \text{ parameters}}$$

that's **~71K params** — small enough to load in the browser in under 100ms. the `.mpk` checkpoint? **286 KB**. try that with your 7B model 😌

### 📊 parameter presets

| preset | layers | $d$ | heads | KV heads | FFN | params |
|--------|--------|-----|-------|----------|-----|--------|
| action | 2 | 64 | 4 | 1 | 128 | ~71K |
| 1M | 2 | 96 | 4 | 1 | 256 | ~1M |
| 5M | 8 | 192 | 6 | 2 | 512 | ~5M |
| 10M | 10 | 256 | 8 | 2 | 704 | ~10M |
| 50M | 16 | 512 | 8 | 2 | 1,408 | ~50M |
| 100M | 15 | 768 | 12 | 4 | 2,048 | ~100M |

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
