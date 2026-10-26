# Changelog

All notable changes to this project are documented here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — release as `0.4.0`

Release the next version as **0.4.0**: `Ortex.load/4` changed from returning the model to
returning `{:ok, model} | {:error, %Ortex.Error{}}`, which is a breaking change for every
caller. `Ortex.load!/4` is the old behaviour.

Everything below came out of the ortex audit (issues MONO-144/145 and the stages under it).
The version numbers in `mix.exs` and `native/ortex/Cargo.toml` are bumped by hand; the
release workflow then builds and tests the precompiled NIFs and publishes them.

### Added

- `Ortex.Error`: structured failures with `:reason`, `:provider` and `:detail`. The reasons
  are `:onnxruntime`, `:invalid_option`, `:invalid_input`, `:execution_provider` and `:qnn`.
- `Ortex.load/4` returns `{:ok, model} | {:error, %Ortex.Error{}}`; `Ortex.load!/4` keeps the
  old raise-and-return behaviour.
- `Ortex.memory/0` and `Ortex.Memory`: the bytes of tensor data held outside the BEAM heap,
  plus the `[:ortex, :memory]` telemetry event (`Ortex.Memory.poll/0`).
- `Ortex.RunOptions`: cancel a run in flight from another process (`terminate/1`,
  `unterminate/1`), backed by ONNX Runtime's `RunOptions::terminate`.
- Named inputs: `Ortex.run(model, %{"x" => x, "y" => y})` (or a list of `{name, tensor}`
  pairs), in addition to the positional tuple.
- Grouped load options: `sessions:`, `threads: [intra:, inter:, intra_op_spinning:,
  inter_op_spinning:]`, `trace:`, `env: %{}`, `qnn: [...]`.
- `mix ortex.copy_ort_libs`: copies ONNX Runtime from `ORT_LIB_LOCATION` into `priv/native`
  for source builds against a runtime that is not installed system-wide.
- `sessions: N`: a pool of N ONNX Runtime sessions over the same graph, which is the only way
  to get concurrent inference from one model (`Session::run` needs `&mut self`).
- `trace` cargo feature: the diagnostics path (`trace_path`/`ORTEX_TRACE`, VERBOSE ORT log)
  compiled on demand instead of always.
- A test suite for the serving and error paths (`models/tinymodel.onnx`), replacing the
  resnet50-only tests that were implicitly skipped.
- CI: ONNX Runtime 1.24.4 (minimum) and 1.31.0 (latest), Elixir 1.17.3/OTP 27 through
  1.20.4/OTP 29, `cargo fmt --check`, `cargo clippy -D warnings`, `cargo audit`,
  `mix hex.audit`, on Linux and macOS.

### Changed

- Inference runs on the dirty CPU pool instead of the 10-thread dirty I/O pool.
- Inspecting a model no longer takes the session lock, so it cannot block behind a running
  inference.
- QNN loads default to `intra_threads: 1` and `intra_op_spinning: false`, which avoids
  measurable thermal throttling on an accelerator (24 -> 20.6 fps on a Dragon Q6A).
- `Ortex.Serving` forces `Nx.Defn.Evaluator`, so an application configured with EXLA does not
  round-trip every batch through the device to evaluate an identity function.
- Tensors share their buffers (`ArcArray`), so reshape/flatten/squeeze are refcount bumps;
  run outputs stay ORT-owned and are copied to the BEAM once instead of twice.
- `ort` 2.0.0-rc.12 -> 2.0.0-rc.15, keeping ONNX Runtime API level 24.
- Precompiled NIFs are CPU-only by contract, and requesting an EP that was not compiled in
  raises naming the cargo feature instead of falling back to the CPU.
- Rust edition 2021, `rust-version = 1.92`, and `rust-toolchain.toml` pinning the compiler the
  release is built with.
- The QNN plugin registration is retried after a failure instead of caching the first error.

### Fixed

- `-1` dynamic dimensions from ONNX Runtime are reported as `nil`, as the docs always said,
  and the element type is reported in its elementary form (`i32`, `f32`, ...).
- `get_bytes` no longer relies on `mem::zeroed` for empty tensors and asserts the standard
  layout invariant its raw pointer cast depends on.
- `concatenate` materializes with one copy instead of two.
- Input tensors are validated against the model's declared dtype, rank and shape before the
  run, so two same-typed inputs in the wrong order fail instead of returning a plausible
  wrong answer.

### Removed

- Copying ONNX Runtime into `priv/native` from the NIF module's body: it also ran while
  a consumer compiled the dependency, globbing build directories for stray libraries.

## [0.3.1]

Published before this changelog existed; see the git history and the releases page for the
changes up to and including the first round of audit fixes.
