# syncopate-machine architecture update

Date: 2026-05-22

## Summary

Renamed the project from `multiscreen-rs` to `syncopate-machine` and replaced
the neural model story with a small decoder-only transformer aimed at game NPC
chat.

The old Multiscreen tile stack was not a good base for this job. It lacked the
core pieces expected from a trainable small LM: proper transformer blocks,
feed-forward layers, RoPE, RMSNorm, grouped-query attention, and a modern
training schedule.

## Papers Read

- `papers/gpt-1.pdf`: causal decoder LM objective, tied output projection,
  warmup plus cosine learning-rate schedule, Adam-style training.
- `papers/tiny_llama.pdf`: Llama-style small model choices: RoPE, pre-norm
  RMSNorm, SwiGLU, grouped-query attention, AdamW, weight decay, grad clipping.
- `papers/higher-order.pdf`: second-order causal attention as an optional
  experimental kernel, not the default baseline.

## Implementation Notes

- Added `SyncopateModelConfig`, `SyncopateModel`, `SyncopateParameterBudget`,
  `AttentionKernel`, and `DefaultSyncopateModel`.
- Kept `Multiscreen*` aliases as compatibility bridges.
- Default attention is causal softmax for stable tiny-model training.
- Added optional `AttentionKernel::HigherOrder` using a normalized second-order
  causal parallel form.
- Added tied token embeddings and tied LM head.
- Added RoPE over attention query/key heads.
- Added pre-norm RMSNorm around attention and SwiGLU FFN sublayers.
- Added GQA by allowing fewer key/value heads than query heads.
- Added AdamW warmup/cosine scheduling fields through the trainer and CLI.
- Updated README, package metadata, examples, tests, license, and contribution
  notes for `syncopate-machine`.

## Verification

Commands run:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
```

All passed.

## Important Caveats

- No new training run was started.
- Existing Multiscreen checkpoints are not compatible with the new Syncopate
  model architecture.
- The higher-order attention kernel is experimental until it has real training
  curves on NPC dialogue data.
- For NPC behavior, direct character dialogue data is more important than broad
  generic instruction data.
