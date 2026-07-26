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
  inputs: [{"input", "Float32", [nil, 3, 224, 224]}]
  outputs: [{"output", "Float32", [nil, 1000]}]>
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

## Installation

`Ortex` can be installed by adding `ortex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ortex, "~> 0.2.0-rc.1"}
  ]
end
```

You will need [Rust](https://www.rust-lang.org/tools/install) for compilation to succeed.

## Execution Providers

Non-CPU execution providers are compiled in through `ort`'s cargo features. The default
depends on the OS: `coreml` on macOS, `directml` on Windows, nothing (CPU only) on Linux.
`ORTEX_FEATURES` overrides it with a comma-separated list of `cuda`, `tensorrt`, `coreml`
and `directml`:

```sh
ORTEX_FEATURES=cuda,tensorrt mix compile
```

A feature only enables the bindings; the vendor runtime itself has to be installed already.

QNN is the exception. It is registered at session creation rather than compiled in, so it
needs no feature flag. See `Ortex.load/4` for its options.

## Packaging and Offline Builds

`ORTEX_SKIP_COMPILE=1` skips building the Rust crate, for packaging against a precompiled
NIF. That NIF and the `libonnxruntime` it needs must already be in `priv/native` for the
target platform.

To build against an ONNX Runtime that is already installed:

```sh
ORTEX_SKIP_DOWNLOAD=1 \
ORT_PREFER_DYNAMIC_LINK=1 \
ORT_LIB_LOCATION=/path/to/onnxruntime/lib \
mix compile
```
