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
iex> model = Ortex.load("./models/resnet50.onnx")
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
model = Ortex.load("./models/resnet50.onnx", [:cpu], 3, sessions: 4)
```

Each pooled session loads the graph and its own thread pools, so N is bounded by memory.

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
