# Contributing

Hey, thanks for wanting to help with `syncopate-machine`.

This crate is meant to stay small, readable, and honest about what it does:
Burn-backed Syncopate model code, action/ID-sequence training examples, browser
inference, and checkpoint persistence. Good patches are very welcome.

## Before You Start

For small fixes, open a pull request directly.

For larger changes, please open an issue or discussion first so we can agree on
the shape before anyone spends a weekend wrestling Rust generics for nothing.

Good contributions include:

- Bug fixes with a focused test.
- Clear docs/examples that make the API easier to use.
- Safer error handling and validation.
- Burn backend improvements that keep the default CPU/Flex path working.
- Architecture improvements with clear tests or training evidence.
- Action-model fixes that keep the browser path small and obvious.

Please avoid:

- Adding large datasets, checkpoints, model artifacts, or generated junk.
- Claiming benchmark numbers unless you actually ran the benchmark and recorded
  the setup.
- Breaking the default CPU/Flex path just to make a custom GPU setup happy.

## Local Checks

Run these before opening a PR:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
```

Before release-related work, also run:

```bash
cargo package --allow-dirty --list
cargo publish --dry-run --allow-dirty
```

Do not run a real `cargo publish` from a contribution branch.

## CI

GitHub Actions runs the default CPU/Flex checks on pushes and pull requests.

The `cuda` feature is not part of the default CI gate because Burn CUDA depends
on local GPU/toolchain setup. If your PR changes CUDA behavior, say exactly how
you verified it in the PR notes.

## Code Style

Keep changes narrow. If a PR starts as "fix one action-runtime bug" and turns into a
full crate rewrite, split it.

Prefer:

- Plain names that match the Syncopate/action-chat API.
- Explicit validation over `unwrap()` in library code.
- Small tests that pin behavior.
- README updates when public behavior changes.

Tests may use `unwrap()` when it keeps the test readable. Library code should
return `Result` with a useful error.

## Commit Messages

Use simple Conventional Commit-style messages:

```text
fix: handle empty action windows
feat: add Burn recorder roundtrip test
docs: explain default Flex backend
ci: add Rust quality checks
```

No need to be fancy. Make the intent obvious.

## Pull Request Notes

In the PR description, include:

- What changed.
- Why it changed.
- Which commands you ran.
- Any known limits, especially around CUDA, packaging, or benchmarks.

If you did not run a check, say so. Honest uncertainty beats fake green lights.
