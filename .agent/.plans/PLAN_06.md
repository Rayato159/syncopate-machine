# PLAN_06.md - Contribution Notes and CI

## Goal

Add friendly contribution guidance and a GitHub Actions CI gate before the crate
is pushed to Git and prepared for crates.io.

## Changes

- Added `CONTRIBUTING.md`
  - Friendly contributor guidance.
  - Scope rules for what belongs in this crate.
  - Local check commands.
  - PR expectations.
  - Conventional Commit-style examples.
- Added `.github/workflows/ci.yml`
  - Runs on pushes and pull requests to `main` and `master`.
  - Installs stable Rust with `rustfmt` and `clippy`.
  - Runs:
    - `cargo fmt --all --check`
    - `cargo check --all-targets`
    - `cargo test`
    - `cargo clippy --all-targets -- -D warnings`
    - `cargo doc --no-deps`
    - `cargo package`
- Updated `README.md`
  - Added a `Contributing` section linking to `CONTRIBUTING.md`.
  - Aligned publishing checks with CI-style commands.
- Updated `Cargo.toml`
  - Included `CONTRIBUTING.md` in the crate package.

## Notes

CI intentionally does not run `--all-features` because the `cuda` feature maps
to Burn CUDA and may require local GPU/toolchain setup. Default CI protects the
normal CPU/Flex path.

The `.github/` workflow and `.agent/` plans are repository files, not crate
package files.

## Verification

Run after this plan is written:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps
cargo package --allow-dirty --offline
```

## Verification Result

Latest known status after adding contribution docs and CI:

- `cargo fmt --all --check` passed
- `cargo check --all-targets` passed
- `cargo test` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo doc --no-deps` passed
- `cargo package --allow-dirty --list --offline` passed and includes
  `CONTRIBUTING.md`
- `cargo package --allow-dirty --offline` passed
