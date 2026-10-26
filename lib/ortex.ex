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

  ## Options

  The fourth argument groups options by what they configure:

    * `sessions: N` - loads N copies of the same graph into the model. An ONNX Runtime
      session is exclusive (`Session::run` takes `&mut self`), so with the default of 1
      every inference on the model is serialised and `Nx.Serving` partitions/batch_size
      only queue work. Each pooled session loads the graph and its thread pools again, so
      the extra memory is proportional to N.

    * `threads: [...]` - ONNX Runtime's thread pools: `intra:` (intra-op threads),
      `inter:` (inter-op threads), `intra_op_spinning:` and `inter_op_spinning:`
      (booleans). These apply to every provider, which is why they are not QNN options.

    * `trace: path` - writes an ORT trace (EP registration, device selection, per-node
      placement) to that file. Needs the `trace` cargo feature - see the README - and is
      a no-op that says so without it. `ORTEX_TRACE` does the same from the environment.

    * `env: %{"NAME" => value}` - exports variables into the OS environment. This is the
      only way QNN's own `getenv()` calls can see them: `System.put_env/2` writes Erlang's
      table, which the NIF's C libraries never read.

    * `qnn: [...]` - QNN execution provider options: `provider_path` and `backend_path`
      pick the QNN plugin and backend libraries, and everything else (`htp_arch: 68`) is
      forwarded as a provider option. Only used when `:qnn` is among `eps`.

  When `:qnn` is among `eps`, the thread pool defaults assume the graph is one offloaded NPU
  node: `threads: [intra: 1, intra_op_spinning: false]`. The default pool's spinning workers
  burn every core while the NPU runs, which measurably throttles the package (24 -> 20.6 fps
  on a Dragon Q6A). Override with `threads:`; CPU-side providers keep onnxruntime's defaults
  for `inter`.

  The flat form (`intra_threads: 1`, `trace_path: ...`, `htp_arch: 68`, `"env.NAME"`) is
  accepted for one release cycle, translated to the groups above and warned about.

  ## Examples

      iex> {:ok, _model} = Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load!("./models/tinymodel.onnx", [:cpu], 0)
      iex> Ortex.load!("./models/tinymodel.onnx", [:cpu], 3, sessions: 4)

  Grouped options, QNN included:

  ```elixir
  Ortex.load!("./models/model.onnx", [:qnn], 3,
    qnn: [htp_arch: 68],
    threads: [intra: 1, intra_op_spinning: false],
    trace: "/tmp/ortex.trace"
  )
  ```

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

  Inputs can be addressed positionally or by name, and both are validated against the
  model's declared dtypes and shapes before the run:

    * a tuple (or a bare tensor for a single-input model) is positional;
    * a map or a list of `{name, tensor}` pairs names them: `%{"x" => x, "y" => y}`.

  Positional inputs have to match the model's declared order - a model with two
  same-typed inputs accepts them swapped and returns a plausible wrong answer otherwise.
  Named inputs remove that ordering from the contract entirely.

  Raises `Ortex.Error` when the tensors do not fit the model (`:invalid_input`, naming the
  input, axis and both shapes) or ONNX Runtime refuses the run (`:onnxruntime`).

  ## Cancellation

  An inference that never returns (a wedged driver, an NPU in a bad state) holds its
  session, and with `sessions: 1` every later inference on that model queues behind it. Pass
  a handle from `Ortex.RunOptions.new/1` to be able to stop it from another process:

  ```elixir
  options = Ortex.RunOptions.new(tag: "request-42")
  Ortex.run(model, inputs, run_options: options)

  # from a timeout, a supervisor, a cancellation request:
  Ortex.RunOptions.terminate(options)
  ```

  The run stops at the next node boundary and raises `Ortex.Error` with reason
  `:onnxruntime`.

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
  defdelegate run(model, tensors, opts), to: Ortex.Model

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
