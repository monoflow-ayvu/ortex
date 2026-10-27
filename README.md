# Ortex

`Ortex` is a wrapper around [ONNX Runtime](https://onnxruntime.ai/) (implemented as
bindings to [`ort`](https://github.com/pykeio/ort)). Ortex leverages
[`Nx.Serving`](https://hexdocs.pm/nx/Nx.Serving.html) to easily deploy ONNX models
that run concurrently and distributed in a cluster. Ortex also provides a storage-only
tensor implementation for ease of use.

ONNX models are a standard machine learning model format that can be exported from most ML
libraries like PyTorch and TensorFlow. Ortex allows for easy loading and fast inference of
ONNX models using different backends available to ONNX Runtime such as CUDA, TensorRT, Core
ML, and ARM Compute Library.

## Examples

TL;DR:

```elixir
iex> model = Ortex.load!("./models/resnet50.onnx")
#Ortex.Model<
  inputs: [{"input", "f32", [nil, 3, 224, 224]}]
  outputs: [{"output", "f32", [nil, 1000]}]>
iex> {output} = Ortex.run(model, Nx.broadcast(0.0, {1, 3, 224, 224}))
iex> output |> Nx.backend_transfer() |> Nx.argmax
#Nx.Tensor<
  s64
  499
>
```

Inspecting a model shows the expected inputs, outputs, data types, and shapes. Axes with
`nil` represent a dynamic size.

Inputs are positional by default, and can be named instead: a map (or a list of pairs) is
matched against the model's own input names, which takes ordering out of the contract. Both
forms are validated against the declared dtype and shape before the run, so a swapped pair
of same-typed inputs fails with the input name and axis instead of returning a plausible
wrong answer.

```elixir
Ortex.run(model, %{"x" => x, "y" => y})
```

To see more real world examples see the `examples` folder.

### Serving

`Ortex` also implements `Nx.Serving` behaviour. To use it in your application's
supervision tree consult the `Nx.Serving` docs.

```elixir
iex> serving = Nx.Serving.new(Ortex.Serving, model)
iex> batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
iex> {result} = Nx.Serving.run(serving, batch)
iex> result |> Nx.backend_transfer() |> Nx.argmax(axis: 1)
#Nx.Tensor<
  s64[1]
  [499]
>
```

An ONNX Runtime session is exclusive, so with the default `sessions: 1` concurrent
inferences on the same model queue behind each other - `Nx.Serving` partitions organise a
queue, they do not add parallelism. `sessions: N` loads N sessions of the *same* graph and
round-robins inferences over them:

```elixir
model = Ortex.load!("./models/resnet50.onnx", [:cpu], 3, sessions: 4)
```

Each pooled session loads its own copy of the graph - its thread pools are shared with the
rest of the process by default, see below - so N is bounded by memory.

### Cancellation

A run that never returns holds its session, and with the default `sessions: 1` every later
inference on that model queues behind it. A handle from `Ortex.RunOptions` lets another
process stop it:

```elixir
options = Ortex.RunOptions.new(tag: "request-42")
Ortex.run(model, inputs, run_options: options)

Ortex.RunOptions.terminate(options)  # from a timeout or a cancellation request
```

`terminate/1` returns immediately and ONNX Runtime stops the graph at the next node
boundary; the run raises `Ortex.Error` with reason `:onnxruntime`. `unterminate/1` resets
the flag, and one handle used by several runs terminates all of them.

### Thread pools and memory

ONNX Runtime's threads are not BEAM processes: they do not show up in
`:erlang.system_info(:process_count)` and the scheduler does not know about them, but they
compete for the same cores.

By default Ortex asks the runtime for its *global* thread pool: one pool per process,
created with the environment and shared by every session built from it. Its size is fixed
when it is created - `intra_op` defaults to one thread per core, `inter_op` to one - and
neither the number of models nor `sessions:` multiplies it:

| models x sessions | one pool per session | global pool |
|---|---|---|
| 1 x 1 | 3 threads | 6 |
| 3 x 2 | 18 | 6 |
| 6 x 4 | 72 | 6 |

(Measured on an 8-thread x86 board with `models/tinymodel.onnx`; the left column is
`ORTEX_GLOBAL_THREAD_POOL=0`, i.e. what this library did before. The pool has a floor: a
process with a single small model pays its fixed size too.)

That is the default because the per-session shape scales with the deployment instead of the
work: three models on a 6-core board used to mean roughly 18 ONNX Runtime threads before any
inference ran, and `sessions: N` multiplied them again because each pooled session built its
own pools. The pool is shared by runs, not owned by them, so concurrent inference over one
model or six does not need one pool each.

The same measurement, 6 models x 4 sessions with 16 callers: 10.7k-16.1k inferences/s and a
651-888us p50 with per-session pools, 77.3k-80.6k/s and 162-171us p50 with the shared pool.
With a single model on one session the two are within noise of each other (22.1k-28.8k/s,
p50 27-45us), so the shared pool is not a tax on the serial case, and 3 models x 2 sessions
gains about 6% throughput.

Sizing and opting out, all read once per process at the first `Ortex.load/4` (the
environment, and with it the pool, is built then and cannot change afterwards):

  * `ORTEX_INTRA_THREADS` / `ORTEX_INTER_THREADS` - the pool's size. Unset keeps ONNX
    Runtime's defaults (one intra-op thread per core, one inter-op).
  * `ORTEX_SPIN_CONTROL=0` - park idle pool threads instead of letting them spin.
  * `ORTEX_GLOBAL_THREAD_POOL=0` - back to one pool per session.
  * `threads: [intra: n, ...]` in `Ortex.load/4` - that model gets its own pool instead of
    the shared one, which is the only way its values can take effect: ONNX Runtime ignores
    a session's thread options once the session is using the global pool. A QNN load does
    the same unconditionally, because its defaults (`intra: 1` with spinning off) are a
    measurement rather than a preference - idle spinning workers heated a Dragon Q6A until
    the package throttled the NPU doing the work, 24 -> 20.6 fps - and the shared pool would
    override exactly those two values.

So:

  * leave the thread options alone to share the pool, and size it with `ORTEX_INTRA_THREADS`
    when all cores is not what you want on a board that also runs the rest of the VM;
  * `sessions: N` is for concurrency, not throughput per inference: size it to the graph
    memory you can afford, since each pooled session keeps a copy of the graph - not a set of
    thread pools;
  * `threads: [intra: n]` on one model is still there for a model that needs its own
    schedule; accept that the per-model arithmetic above comes back with it;
  * on an accelerator EP, `threads: [intra: 1, intra_op_spinning: false]` beats the default,
    and is already what a `:qnn` load gets.

Tensor data lives on the Rust heap, so the same is true of memory: `:erlang.memory/0` does
not see it and the collector feels no pressure from it. `Ortex.memory/0` reports it, and
`Ortex.Memory.poll/0` emits it as `[:ortex, :memory]` for whatever poller your app already
runs.

## Installation

`Ortex` can be installed by adding `ortex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ortex, "~> 0.3.1"}
  ]
end
```

You will need [Rust](https://www.rust-lang.org/tools/install) for compilation to succeed.

At runtime the NIF `dlopen()`s `libonnxruntime` (ort's `load-dynamic` feature), so the
shared library must be present on the target. Either set `ORT_DYLIB_PATH` to the library
path at runtime, or set `ORT_LIB_LOCATION` at compile time to have it copied into
`priv/native` (where the `$ORIGIN` rpath finds it). Nerves system images provide it
system-wide. Model loads without it fail; tests need it too.

The NIF asks the runtime for ONNX Runtime API version 24 (`api-24` in
`native/ortex/Cargo.toml`), which ORT 1.24.x and newer provide, so 1.24 is the minimum
supported runtime.

CI runs the suite against both ends of that range - 1.24.4 (minimum) and the newest
release (1.31.0 at the time of writing) - because `load-dynamic` makes the runtime a
deployment choice, not a build input: the same NIF dlopen()s whatever ORT the target
has. A regression that only shows on one of them is exactly what a single-version matrix
would hide.

Language support is tested the same way: Elixir 1.17.3 / OTP 27 (the floor declared in
`mix.exs`) and 1.20.4 / OTP 29 (current), plus the Elixir 1.18 / OTP 28 pair that
`devenv.nix` provides, so the development shell is a tested configuration rather than a
guess.

## Testing

`mix test` runs the suite against `models/tinymodel.onnx`, the small model committed to
the repo: inference, `Nx.Serving` (batching and partitioned), shape/dtype contracts and the
Rustler error paths. No download or export step is needed.

The `resnet50` tests are opt-in because their weights are not in the repo
(`python/export_resnet.py` exports them). With `models/resnet50.onnx` present:

```sh
ORTEX_RESNET50=1 mix test
```

## Execution Providers

Non-CPU execution providers are compiled in through `ort`'s cargo features. The default
depends on the OS: `coreml` on macOS, `directml` on Windows, nothing (CPU only) on Linux.
`ORTEX_FEATURES` overrides it with a comma-separated list of `cuda`, `tensorrt`, `acl`,
`onednn`, `rocm`, `coreml` and `directml`:

```sh
ORTEX_FEATURES=cuda,tensorrt mix compile
```

A feature only enables the bindings; the vendor runtime itself has to be installed already.
Since `ort` rc.13 each provider's binding is behind its feature, so asking `Ortex.load/4`
for a provider that was not compiled in raises with the feature name to rebuild with,
instead of quietly running the model on the CPU.

QNN is the exception. It is registered at session creation rather than compiled in, so it
needs no feature flag. See `Ortex.load/4` for its options.

## Diagnostics

`trace_path:` in `Ortex.load/4` (or the `ORTEX_TRACE` environment variable) writes an
ONNX Runtime trace to a file: EP registration, device selection and per-node placement.
That last one is how you check whether an accelerator actually claimed the graph; the
NIF's stdout goes to the Nerves console, not to the caller, so a file is the only way to
read it. `RUST_LOG` selects the `tracing` filter, and onnxruntime's own log level is
raised to VERBOSE at the same time.

It is compiled in through the `trace` cargo feature, off by default: a subscriber drags
in `tracing-subscriber` with `env-filter`, which pulls roughly ten extra crates
(`regex-automata`, `aho-corasick`, `sharded-slab`, ...) into the NIF - weight with no
purpose in a production embedded artifact. Enable it when you need it:

```sh
ORTEX_FEATURES=trace mix compile
```

Without the feature, `trace_path` and `ORTEX_TRACE` are a no-op and say so once on stderr
rather than writing an empty file.

`ORTEX_FEATURES` is read at compile time and belongs to the compile-time application
environment, so changing it for an already-built project needs a clean first - `mix clean`
here, or `mix deps.clean ortex --build` from a consumer - otherwise Mix refuses to run with
"a different value set for key Ortex.Native during runtime compared to compile time".


## Packaging and Offline Builds

On `aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu` (including Nerves targets),
ortex downloads a precompiled NIF from the GitHub release matching its version; no Rust
toolchain is needed. On any other platform the crate builds from source, which requires
Rust.

Those precompiled NIFs contain the CPU execution provider only, and nothing else. A
downloaded artifact never goes through `use Rustler`, so `config :ortex, Ortex.Native,
features: [...]` has no effect on it - a CUDA/TensorRT/etc. binding needs a source build
(see `force_build` below) with `ORTEX_FEATURES` set. Requesting such a provider from a
CPU-only artifact raises instead of silently running the model on the CPU.

To force a source build, add this to the consuming project's config:

```elixir
config :rustler_precompiled, :force_build, ortex: true
```

or set `RUSTLER_PRECOMPILED_FORCE_BUILD_ALL=1` in the environment.

To build against an ONNX Runtime that is already installed:

```sh
ORT_PREFER_DYNAMIC_LINK=1 \
ORT_LIB_LOCATION=/path/to/onnxruntime/lib \
mix compile
```

Releases: bump `version` in both `mix.exs` and `native/ortex/Cargo.toml` in a PR and merge
it. The `Release Precompiled NIF` workflow then builds both targets, runs `mix test`
against the precompiled artifacts on each of them, publishes `v<version>` with the
artifacts, and opens a PR with the generated `checksum-Elixir.Ortex.Native.exs` - that
file is what lets consumers verify the download, so it is merged like any other change
rather than pushed to `main` by the bot.

## QNN plugin library

The QNN execution provider is a *plugin* EP in current ONNX Runtime. It is not built into
`libonnxruntime.so` from the ONNX Runtime releases, and it is not built into this NIF either:
it is a separate shared library that Ortex registers with the runtime at session creation.

`libonnxruntime_providers_qnn.so` comes from the **`onnxruntime/onnxruntime-qnn`** project
(the Qualcomm-maintained plugin EP), not from `microsoft/onnxruntime`:

| plugin | ships `libonnxruntime_providers_qnn.so` for | QAIRT SDK | works with ORT |
|---|---|---|---|
| 2.0.0 | Windows ARM64 | 2.x | >= 1.24.1 |
| 2.1.0 | Windows ARM64, Linux ARM64 (preview) | 2.45.40 | >= 1.24.1 |
| 2.1.1 | Windows ARM64, **Linux ARM64 (tgz + wheels)** | 2.45.41 | >= 1.24.1 |
| 2.2.0 | + Linux ARM64 | 2.46.0 | >= 1.24.1 |
| 2.3.0 | + Linux x86_64 wheels (preview) | 2.47.0 | >= 1.24.1 |
| 2.4.0 | + **Linux x86_64 tgz + wheels** | 2.48.40 | >= 1.24.1 |
| 2.5.0 | same | 2.49.40 | >= 1.24.1 |
| 2.6.0 | same | 2.50.40 | >= 1.24.1 |
| 2.7.0 | same | 2.51.0 | >= 1.24.1 |

(From the release notes of each plugin version. `>= 1.24.1` is the EP ABI version the plugin
requires; the QAIRT column is the SDK the plugin was built and tested against.)

`libQnnHtp.so` - the backend the QNN EP dispatches to - comes from the **Qualcomm AI Runtime
SDK (QAIRT)**, installed on the device or taken from the same plugin package where it is
bundled. On a Nerves board it has to be part of the system image; on Debian both libraries
are typically under `/usr/lib`.

Ortex looks in `/usr/lib/libonnxruntime_providers_qnn.so` and `/usr/lib/libQnnHtp.so` by
default, overridable per load or through the environment:

```elixir
Ortex.load!("./models/model.onnx", [:qnn], 3,
  qnn: [
    provider_path: "/data/ortex/lib/libonnxruntime_providers_qnn.so",
    backend_path: "/data/ortex/lib/libQnnHtp.so",
    htp_arch: 68
  ],
  env: %{"DSP_LIBRARY_PATH" => "/usr/lib/rfsa/adsp"}
)
```

```sh
ORTEX_QNN_PROVIDER_PATH=... ORTEX_QNN_BACKEND_PATH=... mix test
```

`htp_arch` is effectively required on HTP targets - without it the EP logs "Failed to get HTP
arch" and the graph ends up on the CPU at roughly 1/35th the speed. `68` is V68, the arch of
the QCS6490 used on the Dragon Q6A. `env:` exists because QNN's own libraries call `getenv()`,
and `System.put_env/2` only writes Erlang's table.

Every failure here is an error, never a CPU fallback: a QNN session that quietly runs on the
CPU looks identical to a working one apart from being slower.

## About this fork

This is `monoflow-ayvu/ortex`, a fork of the Elixir ONNX Runtime bindings formerly maintained
as [`elixir-nx/ortex`](https://github.com/elixir-nx/ortex) (last upstream commit 2026-02-10,
last upstream Hex release 0.1.10 on Nx 0.6).

The contract:

  * **No automatic sync with upstream.** Upstream is not tracked, rebased or merged; the
    differences from it are the point of the fork (Nx 1.x, `ort` 2.0 rc.15, the session pool,
    the QNN plugin path, the precompiled NIF pipeline). Divergence is recorded here and in
    `CHANGELOG.md`, not implied to be temporary.
  * **Versioning is independent.** `mix.exs` and `native/ortex/Cargo.toml` carry the same
    version and are bumped by hand; tags look like `v0.4.0`. Breaking API changes get a
    minor bump while the major version is 0, and are listed under "Changed"/"Removed" in the
    changelog.
  * **Not published on Hex.** The `ortex` package name on Hex belongs to upstream, so
    consumers depend on this fork by git reference. Whether to publish under another name is
    an open product decision, not a property of this code.
  * **Packages on the upstream name are not this project.** `{:ortex, "~> 0.1"}` from Hex is
    upstream's; this fork's `source_url`, `homepage_url` and package links point here.
