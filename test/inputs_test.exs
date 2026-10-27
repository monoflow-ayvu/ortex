defmodule Ortex.InputsTest do
  use ExUnit.Case

  # tinymodel: inputs x: i32[batch][100] and y: f32[batch][100], three f32[batch][10]
  # outputs. Two inputs of *different* types are enough to catch a swap, and the batch
  # axis is dynamic so shape validation has to allow any value there.
  setup_all do
    %{model: Ortex.load!("./models/tinymodel.onnx")}
  end

  defp tensors(row) do
    x = Nx.broadcast(row, {1, 100}) |> Nx.as_type(:s32)
    y = Nx.broadcast(row / 10, {1, 100}) |> Nx.as_type(:f32)
    {x, y}
  end

  test "named inputs produce the same result as positional", %{model: model} do
    {x, y} = tensors(3)

    positional = model |> Ortex.run({x, y}) |> Tuple.to_list() |> Enum.map(&Nx.backend_transfer/1)

    by_map =
      model
      |> Ortex.run(%{"x" => x, "y" => y})
      |> Tuple.to_list()
      |> Enum.map(&Nx.backend_transfer/1)

    assert by_map == positional
  end

  test "named inputs do not depend on the order they are given in", %{model: model} do
    {x, y} = tensors(2)

    forwards = Ortex.run(model, [{"x", x}, {"y", y}])
    backwards = Ortex.run(model, [{"y", y}, {"x", x}])

    to_list = fn out -> out |> Tuple.to_list() |> Enum.map(&Nx.backend_transfer/1) end
    assert to_list.(forwards) == to_list.(backwards)
  end

  test "an unknown input name lists the names the model expects", %{model: model} do
    {x, y} = tensors(0)

    error =
      assert_raise Ortex.Error, ~r/No input named "z"/, fn ->
        Ortex.run(model, %{"x" => x, "z" => y})
      end

    assert error.reason == :invalid_input
    assert error.detail =~ ~s("x", "y")
  end

  test "a named input given twice is rejected", %{model: model} do
    {x, y} = tensors(0)

    error =
      assert_raise Ortex.Error, ~r/given twice/, fn ->
        Ortex.run(model, [{"x", x}, {"x", y}])
      end

    assert error.reason == :invalid_input
  end

  test "a positional input of the wrong dtype is rejected before the run", %{model: model} do
    {x, _y} = tensors(1)

    error =
      assert_raise Ortex.Error, ~r/Input "y" expects Float32, got Int32/, fn ->
        Ortex.run(model, {x, x})
      end

    assert error.reason == :invalid_input
  end

  test "a positional input of the wrong width names the axis and both shapes", %{model: model} do
    {x, _y} = tensors(1)
    narrow = Nx.broadcast(0.0, {1, 50}) |> Nx.as_type(:f32)

    error =
      assert_raise Ortex.Error, ~r/axis 1 expects 100, got 50/, fn ->
        Ortex.run(model, {x, narrow})
      end

    assert error.reason == :invalid_input
  end
end
