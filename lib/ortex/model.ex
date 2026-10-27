defmodule Ortex.Model do
  @moduledoc """
  A model for running Ortex inference with.

  Implements a human-readable representation of a model including the name, dimension, and
  type of each input and output

  ```
  #Ortex.Model<
  inputs: [{"x", "i32", [nil, 100]}, {"y", "f32", [nil, 100]}]
  outputs: [
    {"9", "f32", [nil, 10]},
    {"onnx::Add_7", "f32", [nil, 10]},
    {"onnx::Add_8", "f32", [nil, 10]}
  ]>
  ```

  Types are the elementary ONNX Runtime type names (`i32`, `f32`, `bf16`, ...) and `nil`
  values represent dynamic dimensions.
  """

  @enforce_keys [:reference]
  defstruct [:reference]

  @doc false
  def load(path, eps \\ [:cpu], opt \\ 3, opts \\ []) do
    with {:ok, nif_opts} <- Ortex.Options.to_nif(opts) do
      case Ortex.Native.init(path, eps, opt, nif_opts) do
        {:error, reason} -> {:error, Ortex.Error.from_nif(reason)}
        model -> {:ok, %Ortex.Model{reference: model}}
      end
    end
  end

  @doc false
  def load!(path, eps \\ [:cpu], opt \\ 3, opts \\ []) do
    case load(path, eps, opt, opts) do
      {:ok, model} -> model
      {:error, error} -> raise error
    end
  end

  @doc false
  def run(%Ortex.Model{} = model, %Nx.Tensor{} = tensor) do
    run(model, {tensor})
  end

  # Positional: the NIF checks each tensor against the model's declared dtype and shape,
  # so a swapped pair of same-typed inputs fails instead of producing a plausible result.
  def run(%Ortex.Model{reference: model}, tensors) when is_tuple(tensors) do
    model
    |> Ortex.Native.run(tensors |> Tuple.to_list() |> Enum.map(&to_ref/1))
    |> unwrap()
  end

  # Named: a map (or a list of pairs) keys the inputs by the graph's own input names -
  # `%{"x" => x, "y" => y}` - which removes positional ordering from the contract.
  def run(%Ortex.Model{reference: model}, inputs) when is_map(inputs) do
    model
    |> Ortex.Native.run_named(
      inputs
      |> Enum.map(fn {name, tensor} -> {to_string(name), to_ref(tensor)} end)
    )
    |> unwrap()
  end

  def run(%Ortex.Model{reference: model}, inputs) when is_list(inputs) do
    model
    |> Ortex.Native.run_named(Enum.map(inputs, &named_ref!/1))
    |> unwrap()
  end

  defp named_ref!({name, tensor}), do: {to_string(name), to_ref(tensor)}

  defp named_ref!(other) do
    raise ArgumentError,
          "expected {name, tensor} pairs when passing a list of inputs, got: #{inspect(other)}"
  end

  defp to_ref(tensor) do
    %Nx.Tensor{data: %Ortex.Backend{ref: ref}} = Nx.backend_transfer(tensor, Ortex.Backend)
    ref
  end

  defp unwrap({:error, reason}), do: raise(Ortex.Error.from_nif(reason))

  defp unwrap(output) do
    # Pack the output into new Ortex.Backend tensor(s)
    output
    |> Enum.map(fn {ref, shape, dtype_atom, dtype_bits} ->
      %Nx.Tensor{
        data: %Ortex.Backend{ref: ref},
        shape: shape |> List.to_tuple(),
        type: {dtype_atom, dtype_bits},
        names: List.duplicate(nil, length(shape))
      }
    end)
    |> List.to_tuple()
  end
end

defimpl Inspect, for: Ortex.Model do
  import Inspect.Algebra

  def inspect(%Ortex.Model{reference: model}, inspect_opts) do
    case Ortex.Native.show_session(model) do
      {:error, reason} ->
        raise Ortex.Error.from_nif(reason)

      {inputs, outputs} ->
        force_unfit(
          concat([
            color("#Ortex.Model<", :map, inspect_opts),
            line(),
            # to_doc/2, not Inspect.List.inspect/2: as of Elixir 1.19 the protocol
            # implementations return {doc, opts} rather than a bare doc, so
            # passing one to concat/2 raises FunctionClauseError - which made
            # inspecting an %Ortex.Model{} in IEx blow up, including as a field of
            # some other term.
            nest(concat(["  inputs: ", to_doc(inputs, inspect_opts)]), 2),
            line(),
            nest(concat(["  outputs: ", to_doc(outputs, inspect_opts)]), 2),
            color(">", :map, inspect_opts)
          ])
        )
    end
  end
end
