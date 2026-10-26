defmodule Ortex.RunOptions do
  @moduledoc """
  A handle for cancelling a run in flight.

  An inference that never returns - a driver wedged, an NPU in a bad state - holds its
  ONNX Runtime session for as long as it lives. With `sessions: 1` (the default) that means
  every later inference on the model is blocked behind it, from a dirty scheduler, with no
  timeout and no way out. This is that way out.

      options = Ortex.RunOptions.new(tag: "request-42")

      # ... in the process that runs the inference:
      Ortex.run(model, inputs, run_options: options)

      # ... in another process, when the request is cancelled or times out:
      Ortex.RunOptions.terminate(options)

  `terminate/1` returns immediately: it sets the flag ONNX Runtime checks between graph
  nodes, so the run stops as soon as it can and raises `Ortex.Error` with reason
  `:onnxruntime`. The run is not killed mid-node, which is exactly what makes this safe to
  call from anywhere - and why a run wedged inside a driver may still take a while to
  notice.

  The flag is per handle: pass one `%Ortex.RunOptions{}` to a single run and nothing else is
  affected. Sharing one across concurrent runs terminates all of them - that is a feature
  for "cancel this whole batch", and a footgun otherwise. After a run has been terminated,
  `unterminate/1` resets the flag so the handle can be reused.
  """

  @enforce_keys [:ref]
  defstruct [:ref]

  @type t :: %__MODULE__{ref: reference()}

  @doc """
  Creates a cancellation handle.

  The optional `:tag` shows up in ONNX Runtime's own logs, which is what makes a cancelled
  run identifiable in a trace.

  ## Examples

      iex> %Ortex.RunOptions{} = Ortex.RunOptions.new(tag: "request-42")

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    case Keyword.fetch(opts, :tag) do
      {:ok, tag} when is_binary(tag) -> %__MODULE__{ref: Ortex.Native.run_options_new(tag)}
      {:ok, other} -> raise ArgumentError, "tag: must be a string, got: #{inspect(other)}"
      :error -> %__MODULE__{ref: Ortex.Native.run_options_new(nil)}
    end
  end

  @doc """
  Asks every run using this handle to stop as soon as ONNX Runtime can.

  Returns immediately; the runs stop on their own and raise. Returns `:ok`, or
  `{:error, %Ortex.Error{}}` if ONNX Runtime refused the flag.
  """
  @spec terminate(t()) :: :ok | {:error, Ortex.Error.t()}
  def terminate(%__MODULE__{ref: ref}) do
    case Ortex.Native.run_options_terminate(ref) do
      :ok -> :ok
      {:error, reason} -> {:error, Ortex.Error.from_nif(reason)}
    end
  end

  @doc """
  Clears the termination flag set by `terminate/1`, so the handle can be used again.
  """
  @spec unterminate(t()) :: :ok | {:error, Ortex.Error.t()}
  def unterminate(%__MODULE__{ref: ref}) do
    case Ortex.Native.run_options_unterminate(ref) do
      :ok -> :ok
      {:error, reason} -> {:error, Ortex.Error.from_nif(reason)}
    end
  end
end
