defmodule Ortex.Error do
  @moduledoc """
  Every failure that comes out of Ortex, as data instead of a message.

  The NIF returns `{:error, {reason, provider, detail}}`; this exception is that tuple
  with the message built in, so a caller can react to a failure without matching on
  strings:

      case Ortex.load(path, [:cuda]) do
        {:error, %Ortex.Error{reason: :execution_provider, provider: :cuda}} ->
          Logger.warning("no CUDA in this NIF, rebuilding, falling back to CPU")

        {:ok, model} ->
          model
      end

  ## Reasons

    * `:execution_provider` - the requested provider is not in this build (the message
      names the cargo feature) or is not a provider at all. `provider` is set.
    * `:qnn` - the QNN *plugin* library, its backend, or the device it would select.
      `provider` is `:qnn`.
    * `:invalid_option` - a load option was present but unusable (`sessions: "abc"`,
      `intra_threads: -1`).
    * `:invalid_input` - the tensors handed to `Ortex.run/2` do not fit the model: wrong
      input count, a slice whose rank does not match, a dtype with no conversion.
    * `:onnxruntime` - ONNX Runtime itself refused the call.

  `detail` is the raw message (the same text as `message`), kept separate so it can be
  logged or reported without the exception wrapper.
  """

  defexception [:reason, :provider, :detail, message: nil]

  @type reason ::
          :onnxruntime
          | :invalid_option
          | :invalid_input
          | :execution_provider
          | :qnn

  @type t :: %__MODULE__{
          reason: reason(),
          provider: atom() | nil,
          detail: String.t(),
          message: String.t()
        }

  @impl true
  def exception(opts) do
    detail = Keyword.fetch!(opts, :detail) |> to_string()

    %__MODULE__{
      reason: Keyword.get(opts, :reason, :onnxruntime),
      provider: Keyword.get(opts, :provider),
      detail: detail,
      message: Keyword.get(opts, :message, detail)
    }
  end

  @doc false
  @spec from_nif(term()) :: t()
  def from_nif({reason, provider, detail}) when is_atom(reason) do
    exception(reason: reason, provider: provider, detail: detail)
  end

  # Defensive: a NIF error that is not the structured tuple (an older artifact, a
  # Rustler-internal failure) still has to become an Ortex.Error rather than a
  # RuntimeError with a term in it.
  def from_nif(other) do
    exception(reason: :onnxruntime, detail: inspect(other))
  end
end
