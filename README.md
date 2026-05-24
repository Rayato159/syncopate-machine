# syncopate-machine

Tiny Rust transformer for browser NPC/chat action prediction. No tokenizer. No
SentencePiece. No "please wait while I become a 7B model" cosplay.

## What It Does

```text
user text
-> classify Action + Lang
-> [action_id, lang_id, SEP]
-> tiny transformer predicts action IDs
-> response graph turns actions into short TH/EN replies
-> first action drives the mood GIF
```

Current action model:

- vocab: `23` integer IDs
- context: `64`
- shape: `2 layers`, `d_model=64`, `4 heads`, `1 kv head`
- params: about `71K`
- browser assets: `model.mpk` + `model-config.json`
- runtime: WebGPU first, CPU fallback for cursed Chrome profiles

## Train

```powershell
python examples\build_action_data.py --offline
cargo run --release --features cuda --example train_action_model -- --steps 3000 --batch-size 32 --lr 0.003
```

Training is CUDA-only by policy for real runs. CPU/Flex is just the fallback
runtime path for the website, not the training lane.

Outputs land in `runs/action-model/`:

```text
final.mpk
model-config.json
loss.csv
report.json
```

## Ship To Website

```powershell
Copy-Item runs\action-model-personal-v2\final.mpk ..\dancing-with-my-code-v2\assets\model-personal-v2.mpk
Copy-Item runs\action-model-personal-v2\model-config.json ..\dancing-with-my-code-v2\assets\model-config-personal-v2.json
```

## Action Vocab

```text
0 PAD   1 SOS   2 EOS   3 SEP
4 Unknown
5 Greeting      6 Farewell      7 Frustrated
8 Sad           9 Happy         10 Question
11 Insult       12 Compliment   13 Agree
14 Disagree     15 General
16 Eating       17 DailyLife    18 RustGo
19 Identity     20 ShitTalk
21 TH           22 EN
```

## Crate Notes

- Core model is still a causal next-ID predictor.
- Website IDs are actions, not text tokens.
- `Trainer` and `ChatModel` stay generic for people who want raw ID sequences.
- Big checkpoints, generated data, and random model artifacts do not belong in
  the crate package.

## License

MIT. Break it, fix it, ship it.
