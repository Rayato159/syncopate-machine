# NPC data and tokenizer cleanup

Date: 2026-05-23

## Summary

Rebuilt the school-zombie NPC training data around natural player dialogue
instead of prompt-scaffold metadata. The previous `PERSONA=...`, `TRAITS=...`,
`TASK=...` style rows were too noisy for this small model and made the user
input distribution look nothing like runtime chat.

## Active Dataset

- Location: `examples/data`
- Rows: 10,000
- Row shape: `npc`, natural `input`, JSON-string `output`
- NPC values: `mew`, `jin`
- Mood values: `normal`, `happy`, `sad`, `shy`, `mad`, `scary`
- Output JSON keys: `message`, `mood`, `relation_point`

`train_with_tokenizer.rs` now prepends `<mew>` or `<jin>` internally when a row
has `npc`, so the file can keep natural input such as `ไหวไหม` while the model
still receives the speaker-control token.

## External Corpus Check

Hugging Face has Thai instruction/chat corpora such as
`Suraponn/thai_instruction_sft`,
`pythainlp/oasst2_thai_top1_chat_format`,
`ZombitX64/ThaiChatbotConversation`, and
`ping98k/lmsys-chat-1m-thai-filtered`, but those are not clean drop-in NPC SFT
rows. They use mixed schemas and generic assistant/chatbot tone, so they should
be treated as optional language pretraining corpus, not merged directly into the
character JSON-output dataset.

## Tokenizer

Rebuilt `examples/data/tokenizer.model` and `examples/data/tokenizer.vocab` from
`examples/data/tokenizer_corpus.spm` using `experiment/train/spm_train.exe`.
Byte fallback is disabled, so the old `<0x00>` through `<0xFF>` vocab block is
gone. The final vocab has 4059 pieces because `hard_vocab_limit=false` avoids
padding a tiny corpus with garbage pieces just to hit 4096 exactly.

## Verification

- Dataset validation: 10,000 rows, 0 bad rows.
- No prompt scaffold strings found in data.
- No old moods found in data.
- No `source` metadata field in rows.
- No tokenizer `<0x..>` byte fallback pieces.
- `cargo fmt --all --check` passed.
- `cargo check --example train_with_tokenizer` passed.
- `cargo check --example chat_with_tokenizer` passed.
- `cargo check --features cuda --example train_with_tokenizer` passed.
- `cargo check --features cuda --example chat_with_tokenizer` passed.
