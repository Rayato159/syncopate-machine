# syncopate-machine

Tiny Burn transformer for action prediction. It does not tokenize text and it
does not generate words. The website classifies text into an action/lang pair,
then this model predicts the next action IDs.

## Flow

```text
user text
-> website classifier: Action + Lang
-> [action_id, lang_id, SEP]
-> syncopate-machine predicts action IDs
-> website response graph writes the message
-> mood GIF follows the first action
```

## Action Vocab

```text
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

## Train

```powershell
python examples\build_action_data.py --offline
cargo run --release --features cuda --example train_action_model -- --steps 3000 --batch-size 32 --lr 0.003 --checkpoint-dir runs/action-model-personal-v2
```

Training is CUDA-only. Browser runtime can use WebGPU or CPU.

## Browser Runtime

Use the `wasm` feature and load:

```text
assets/model-personal-v2.mpk
assets/model-config-personal-v2.json
```

The active personal model is `23` vocab IDs, `seq_len=64`, `2` layers,
`d_model=64`, about `71K` params.

## What Is Gone

- tokenizer examples
- text chat examples
- high-level `ChatModel` / `Trainer` wrappers
- legacy Multiscreen transition/layout engine
- SentencePiece and old NPC prompt-contract docs

MIT. Break it, fix it, ship it.
