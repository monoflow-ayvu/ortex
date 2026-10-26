defmodule Ortex do
  @moduledoc """
  Documentation for `Ortex`.

  `Ortex` is an Elixir wrapper around [ONNX Runtime](https://onnxruntime.ai/) using
  [Rustler](https://hexdocs.pm/rustler) and [ORT](https://github.com/pykeio/ort).

  Failures are `Ortex.Error` structs: `load/4` returns `{:ok, model} | {:error, error}`,
  `load!/4` and `run/2` raise them. Each carries a `:reason`, the `:provider` when the
  failure is about one, and the message in `:detail`.
  """

  @doc """
  Load an `Ortex.Model` from disk. Optionally pass the execution providers as a list
  of descending priority and graph optimization level 1-3. Any graph optimization level
  beyond the range of 1-3 will disable graph optimization.

  CPU is always available. The other execution providers (CUDA, TensorRT, ACL, oneDNN, CoreML,
  DirectML, ROCm) are compiled in through cargo features - see the README - and the library
  that ships them has to be present too. Requesting a provider that was not compiled in raises
  at load, naming the feature to rebuild with, rather than silently running on the CPU.

  `qnn_opts` configures the QNN execution provider. `backend_path` and `provider_path` pick
  the QNN libraries and any other key is forwarded as a QNN provider option (`htp_arch: 68`);
  those only apply when `:qnn` is among `eps`. `trace_path` and `"env.NAME"` keys take effect
  regardless: the first writes an ORT trace (it needs the `trace` cargo feature - see the
  README - and is a no-op that says so without it), the second exports `NAME` into the OS
  environment, the only way QNN's own `getenv` calls can see it.

  `sessions: N` loads N copies of the same graph into the model. An ONNX Runtime session is
  exclusive - `Session::run` takes `&mut self` - so with the default of 1 every inference on
  the model is serialised, and `Nx.Serving` partitions/batch_size only queue work. Set
  `sessions:` above 1 when concurrent inferences have to overlap. Each pooled session loads
  the graph and its thread pools again, so the extra memory is proportional to N.

  When `:qnn` is among `eps`, the thread pool defaults assume the graph is one offloaded NPU
  node: `intra_threads: 1` and `intra_op_spinning: false`. The default pool's spinning workers
  burn every core while the NPU runs, which measurably throttles the package (24 -> 20.6 fps
  on a Dragon Q6A). Pass `intra_threads:` or `intra_op_spinning:` in `qnn_opts` to override,
  and `inter_threads`/`inter_op_spinning` for CPU-side EPs, which keep onnxruntime's defaults.

  ## Examples

      iex> {:ok, _model} = Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load!("./models/tinymodel.onnx", [:cpu], 0)
      iex> Ortex.load!("./models/tinymodel.onnx", [:cpu], 3, sessions: 4)

  A provider list is ordered by priority, and only providers compiled into the NIF can be
  requested:

  ```elixir
  case Ortex.load("./models/resnet50.onnx", [:cuda, :cpu]) do
    {:ok, model} -> model
    {:error, %Ortex.Error{reason: :execution_provider, provider: :cuda} = error} -> raise error
  end
  ```

  """
  defdelegate load(path, eps \\ [:cpu], opt \\ 3, opts \\ []), to: Ortex.Model

  @doc """
  Same as `load/4` but returns the model directly, raising `Ortex.Error` on failure.

  This was `load/4`'s behaviour before it returned a tagged tuple.

  ## Examples

      iex> match?(%Ortex.Model{}, Ortex.load!("./models/tinymodel.onnx"))
      true

  """
  defdelegate load!(path, eps \\ [:cpu], opt \\ 3, opts \\ []), to: Ortex.Model

  @doc """
  Run a forward pass through a model.

  This takes a model and tuple of `Nx.Tensors`,
  optionally transfers them to the `Ortex.Backend` if they aren't there already,
  and runs a forward pass through the model. This will return a tuple of `Ortex.Backend`
  tensors, it's up to the user to transfer these back to another backend if additional
  ops are required.

  If there is only one input you can optionally pass a bare tensor rather than a tuple.

  Raises `Ortex.Error` when the tensors do not fit the model (`:invalid_input`) or ONNX
  Runtime refuses the run (`:onnxruntime`).

  ## Examples

      iex> model = Ortex.load!("./models/tinymodel.onnx")
      iex> {%Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}}} = Ortex.run(
      ...>    model, {
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32),
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:f32)
      ...>    })

  """
  defdelegate run(model, tensors), to: Ortex.Model

  @doc """
  Bytes of tensor data currently held by live Ortex tensors.

  Ortex tensors live on the Rust heap, invisible to `:erlang.memory/0` and to the BEAM's
  garbage collector: holding onto them grows the process RSS without creating GC
  pressure. This is that number, so a serving cache or a queue of batches can be watched
  before it becomes an OOM.

  See `Ortex.Memory` for the exact accounting and the `[:ortex, :memory]` telemetry event.

  ## Examples

      iex> Ortex.memory()
      0

  """
  defdelegate memory, to: Ortex.Memory, as: :bytes
end
