defmodule Ortex.TestModel do
  # End-to-end inference against the only ONNX model in the repo (models/tinymodel.onnx,
  # 8.7 KB): 2 inputs `x: i32[batch][100]`, `y: f32[batch][100]`, 3 f32[batch][10] outputs.
  use ExUnit.Case

  setup_all do
    %{model: Ortex.load("./models/tinymodel.onnx")}
  end

  defp inputs(row) do
    x = Nx.broadcast(row, {1, 100}) |> Nx.as_type(:s32)
    y = Nx.broadcast(row / 10, {1, 100}) |> Nx.as_type(:f32)
    {x, y}
  end

  defp run!(model, tensor) do
    model
    |> Ortex.run(tensor)
    |> Tuple.to_list()
    |> Enum.map(&Nx.backend_transfer/1)
  end

  test "run returns one tensor per output, with the model's shapes and dtypes", %{model: model} do
    assert {o1, o2, o3} = Ortex.run(model, inputs(1))

    for output <- [o1, o2, o3] do
      assert %Nx.Tensor{data: %Ortex.Backend{}} = output
      assert Nx.shape(output) == {1, 10}
      assert Nx.type(output) == {:f, 32}
    end

    assert Nx.to_flat_list(o1) != Nx.to_flat_list(o2)
  end

  test "inference is deterministic", %{model: model} do
    [a | _] = run!(model, inputs(3))
    [b | _] = run!(model, inputs(3))

    assert Nx.to_flat_list(a) == Nx.to_flat_list(b)
  end

  # The graph is row-wise, so batching must not mix rows: row i of a batched run equals
  # the result of running row i on its own. This is the value-level check that the
  # shape-only assertions above cannot make.
  test "each row of a batch matches the same row run alone", %{model: model} do
    {x1, y1} = inputs(1)
    {x2, y2} = inputs(2)

    [alone1, _, _] = run!(model, {x1, y1})
    [alone2, alone2_b, _] = run!(model, {x2, y2})
    [b1, b2, _] = run!(model, {Nx.concatenate([x1, x2]), Nx.concatenate([y1, y2])})

    assert Nx.shape(b1) == {2, 10}
    assert b1[0] == alone1[0]
    assert b1[1] == alone2[0]
    assert b2[1] == alone2_b[0]
  end

  test "an optimization level of 0 still produces a usable session", %{model: model} do
    unoptimized = Ortex.load("./models/tinymodel.onnx", [:cpu], 0)

    [o1 | _] = run!(unoptimized, inputs(1))
    [o2 | _] = run!(model, inputs(1))

    assert Nx.shape(o1) == {1, 10}
    assert_in_delta Nx.to_flat_list(o1) |> hd(), Nx.to_flat_list(o2) |> hd(), 1.0e-5
  end
end
