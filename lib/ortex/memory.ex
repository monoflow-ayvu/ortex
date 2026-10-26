defmodule Ortex.Memory do
  @moduledoc """
  Visibility for the tensor memory Ortex holds outside the BEAM heap.

  Ortex tensors live on the Rust heap and reach the BEAM only as opaque resource handles,
  so `:erlang.memory/0`, the garbage collector and the refc binary accounting cannot see
  them. A process that keeps tensors alive - a serving cache, a queue of pending batches,
  a model output held in a GenServer state - grows the OS process's RSS without creating
  any GC pressure, and the first symptom is the OOM killer. On a Nerves board with a
  fixed RAM budget that is a reboot, not a log line.

  `bytes/0` is that number: the sum of the buffers behind every live Ortex tensor. A
  tensor's buffer is charged when the handle is created and released when the BEAM's GC
  drops the last reference to it, so the value follows what is actually reachable from
  Elixir - not what was allocated since boot.

  A buffer shared by two live handles (Nx's `reshape` and `flatten` are refcount bumps,
  not copies) is counted once per handle, so the number is an upper bound on unique bytes.
  It is exact for fresh allocations - inputs from `Nx.backend_transfer/2`, `slice`,
  `concatenate` - and for ONNX Runtime output buffers.

  ## Telemetry

  `poll/0` emits `[:ortex, :memory]` for whatever poller you already run:

    * measurements: `%{tensor_bytes: non_neg_integer()}`
    * metadata: `%{node: node()}`

  Emitting on every allocation would mean a NIF-to-Elixir round trip per tensor, so the
  event is polled instead. A once-a-minute poll is enough to see a leak trend and cheap
  enough to leave on in production:

      :telemetry.attach("ortex-memory-log", [:ortex, :memory], fn _event, measurements, meta, _ ->
        Logger.info("ortex tensor memory: \#{measurements.tensor_bytes} bytes on \#{meta.node}")
      end, nil)

      # ... then, from your supervision tree or a :timer:
      Ortex.Memory.poll()
  """

  @event [:ortex, :memory]

  @doc """
  Bytes of tensor data currently held by live Ortex tensors.
  """
  @spec bytes() :: non_neg_integer()
  def bytes, do: Ortex.Native.memory_usage()

  @doc """
  The telemetry event `poll/0` emits.
  """
  @spec event() :: [atom()]
  def event, do: @event

  @doc """
  Reads the current value and emits it as `[:ortex, :memory]`.

  Returns the measurements that were emitted, so it doubles as a cheap one-off read.
  """
  @spec poll() :: %{tensor_bytes: non_neg_integer()}
  def poll do
    measurements = %{tensor_bytes: bytes()}
    :telemetry.execute(@event, measurements, %{node: node()})
    measurements
  end
end
