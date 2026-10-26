defmodule Ortex.TestModel do
  # End-to-end inference against the only ONNX model in the repo (models/tinymodel.onnx,
  # 8.7 KB): 2 inputs `x: i32[batch][100]`, `y: f32[batch][100]`, 3 f32[batch][10] outputs.
  use ExUnit.Case

  setup_all do
    %{model: Ortex.load!("./models/tinymodel.onnx")}
  end

  defp inputs(row) do
    x = Nx.broadcast(row, {1, 100}) |> Nx.as_type(:s32)
    y = Nx.broadcast(row / 10, {1, 100}) |> Nx.as_type(:f32)
    {x, y}
  end

  defp close?(left, right) do
    Nx.all_close(Nx.backend_transfer(left), Nx.backend_transfer(right), atol: 1.0e-5)
    |> Nx.to_number() == 1
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
  #
  # Compared with a tolerance on purpose: ONNX Runtime picks a different GEMM kernel and
  # tiling for a batch-2 matrix than for a batch-1 one, so the last bits differ on some
  # builds (macOS + ORT 1.31 shows ~6e-8 relative difference). Row identity is the
  # contract; bit-identical arithmetic across shapes is not.
  test "each row of a batch matches the same row run alone", %{model: model} do
    {x1, y1} = inputs(1)
    {x2, y2} = inputs(2)

    [alone1, _, _] = run!(model, {x1, y1})
    [alone2, alone2_b, _] = run!(model, {x2, y2})
    [b1, b2, _] = run!(model, {Nx.concatenate([x1, x2]), Nx.concatenate([y1, y2])})

    assert Nx.shape(b1) == {2, 10}
    assert close?(b1[0], alone1[0])
    assert close?(b1[1], alone2[0])
    assert close?(b2[1], alone2_b[0])
  end

  test "sessions: N loads a pool and concurrent runs stay correct" do
    model = Ortex.load!("./models/tinymodel.onnx", [:cpu], 3, sessions: 4)

    results =
      1..16
      |> Enum.map(fn row -> Task.async(fn -> {row, run!(model, inputs(row))} end) end)
      |> Task.await_many(30_000)

    for {row, [o1, o2, o3]} <- results do
      assert Nx.shape(o1) == {1, 10}
      assert Nx.shape(o2) == {1, 10}
      assert Nx.shape(o3) == {1, 10}
      # Each pooled session must produce the same numbers for the same input.
      assert Nx.to_flat_list(o1) == Nx.to_flat_list(run!(model, inputs(row)) |> hd())
    end
  end

  test "sessions: 0 is rejected instead of deadlocking on an empty pool" do
    assert {:error, %Ortex.Error{reason: :invalid_option, detail: detail}} =
             Ortex.load("./models/tinymodel.onnx", [:cpu], 3, sessions: 0)

    assert detail =~ "sessions must be at least 1, got 0"

    assert {:error, %Ortex.Error{reason: :invalid_option, detail: detail}} =
             Ortex.load("./models/tinymodel.onnx", [:cpu], 3, sessions: "four")

    assert detail =~ "sessions must be a non-negative integer"
  end

  # Outputs arrive as ORT-owned values (P6). Any array operation has to materialize them
  # first, and must still produce exactly the same numbers as a transferred tensor.
  test "a run output can be reshaped, sliced and concatenated", %{model: model} do
    {o1, _, _} = Ortex.run(model, inputs(1))
    reference = Nx.backend_transfer(o1)

    reshaped = Nx.reshape(o1, {2, 5})
    assert Nx.shape(reshaped) == {2, 5}
    assert Nx.to_flat_list(reshaped) == Nx.to_flat_list(reference)

    sliced = Nx.slice(o1, [0, 3], [1, 4])
    assert Nx.to_flat_list(sliced) == Nx.to_flat_list(Nx.slice(reference, [0, 3], [1, 4]))

    concatenated = Nx.concatenate([o1, o1], axis: 0)
    assert Nx.shape(concatenated) == {2, 10}

    assert Nx.to_flat_list(concatenated) ==
             Nx.to_flat_list(reference) ++ Nx.to_flat_list(reference)
  end

  test "an optimization level of 0 still produces a usable session", %{model: model} do
    unoptimized = Ortex.load!("./models/tinymodel.onnx", [:cpu], 0)

    [o1 | _] = run!(unoptimized, inputs(1))
    [o2 | _] = run!(model, inputs(1))

    assert Nx.shape(o1) == {1, 10}
    assert_in_delta Nx.to_flat_list(o1) |> hd(), Nx.to_flat_list(o2) |> hd(), 1.0e-5
  end
end
