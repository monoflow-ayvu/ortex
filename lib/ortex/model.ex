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
  def run(%Ortex.Model{} = model, %Nx.Tensor{} = tensor), do: run(model, {tensor}, [])

  def run(%Ortex.Model{} = model, inputs), do: run(model, inputs, [])

  # Positional: the NIF checks each tensor against the model's declared dtype and shape,
  # so a swapped pair of same-typed inputs fails instead of producing a plausible result.
  def run(%Ortex.Model{reference: model}, tensors, opts) when is_tuple(tensors) do
    refs = tensors |> Tuple.to_list() |> Enum.map(&to_ref/1)

    model
    |> call(:run, :run_with_options, refs, opts)
    |> unwrap()
  end

  # Named: a map (or a list of pairs) keys the inputs by the graph's own input names -
  # `%{"x" => x, "y" => y}` - which removes positional ordering from the contract.
  def run(%Ortex.Model{reference: model}, inputs, opts) when is_map(inputs) do
    refs = Enum.map(inputs, fn {name, tensor} -> {to_string(name), to_ref(tensor)} end)

    model
    |> call(:run_named, :run_named_with_options, refs, opts)
    |> unwrap()
  end

  def run(%Ortex.Model{reference: model}, inputs, opts) when is_list(inputs) do
    refs = Enum.map(inputs, &named_ref!/1)

    model
    |> call(:run_named, :run_named_with_options, refs, opts)
    |> unwrap()
  end

  # The cancellation handle only exists when the caller made one: building a RunOptions per
  # inference for the common case would be a syscall per run for nothing.
  defp call(model, plain, _with_options, inputs, []),
    do: apply(Ortex.Native, plain, [model, inputs])

  defp call(model, _plain, with_options, inputs, opts),
    do: apply(Ortex.Native, with_options, [model, inputs, run_options_ref!(opts)])

  defp run_options_ref!(opts) do
    case Keyword.fetch(opts, :run_options) do
      {:ok, %Ortex.RunOptions{ref: ref}} ->
        ref

      {:ok, other} ->
        raise ArgumentError,
              "run_options: must be an Ortex.RunOptions, got: #{inspect(other)}"

      :error ->
        raise ArgumentError,
              "unknown run option(s): #{inspect(Keyword.keys(opts))}; the only one is :run_options"
    end
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
