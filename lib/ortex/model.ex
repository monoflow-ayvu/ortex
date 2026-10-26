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
    case Ortex.Native.init(path, eps, opt, stringify_opts(opts)) do
      {:error, reason} -> {:error, Ortex.Error.from_nif(reason)}
      model -> {:ok, %Ortex.Model{reference: model}}
    end
  end

  @doc false
  def load!(path, eps \\ [:cpu], opt \\ 3, opts \\ []) do
    case load(path, eps, opt, opts) do
      {:ok, model} -> model
      {:error, error} -> raise error
    end
  end

  # The NIF takes both key and value as strings; keys prefixed "env." it exports to the
  # OS environment, the only place QNN's getenv() lookups will find them.
  defp stringify_opts(opts), do: Enum.map(opts, fn {k, v} -> {to_string(k), to_string(v)} end)

  @doc false
  def run(%Ortex.Model{} = model, tensor) when not is_tuple(tensor) do
    run(model, {tensor})
  end

  @doc false
  def run(%Ortex.Model{reference: model}, tensors) do
    # Move tensors into Ortex backend and pass the reference to the Ortex NIF
    output =
      case Ortex.Native.run(
             model,
             tensors
             |> Tuple.to_list()
             |> Enum.map(fn x -> x |> Nx.backend_transfer(Ortex.Backend) end)
             |> Enum.map(fn %Nx.Tensor{data: %Ortex.Backend{ref: x}} -> x end)
           ) do
        {:error, reason} -> raise Ortex.Error.from_nif(reason)
        output -> output
      end

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
