defmodule Ortex.TestServing do
  use ExUnit.Case

  setup_all do
    %{model: Ortex.load("./models/tinymodel.onnx")}
  end

  defp batch(rows) do
    Nx.Batch.stack(
      for row <- rows do
        {Nx.broadcast(row, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(row / 10, {100}) |> Nx.as_type(:f32)}
      end
    )
  end

  defp values({o1, o2, o3}) do
    [o1, o2, o3] |> Enum.map(&(&1 |> Nx.backend_transfer() |> Nx.to_flat_list()))
  end

  test "a single-row batch returns one row per output", %{model: model} do
    serving = Nx.Serving.new(Ortex.Serving, model)
    assert {o1, o2, o3} = Nx.Serving.run(serving, batch([1]))

    for output <- [o1, o2, o3] do
      assert Nx.shape(output) == {1, 10}
      assert Nx.type(output) == {:f, 32}
    end
  end

  test "a three-row batch keeps row order", %{model: model} do
    serving = Nx.Serving.new(Ortex.Serving, model)
    {batched, _, _} = Nx.Serving.run(serving, batch([1, 2, 3]))

    assert Nx.shape(batched) == {3, 10}

    for {row, index} <- Enum.with_index([1, 2, 3]) do
      {single, _, _} = Nx.Serving.run(serving, batch([row]))

      assert batched[index]
             |> Nx.backend_transfer()
             |> Nx.to_flat_list() ==
               single |> Nx.backend_transfer() |> Nx.to_flat_list()
    end
  end

  test "partitions: true fans out across the compiler's partitions", %{model: model} do
    serving = Nx.Serving.new(Ortex.Serving, model, max_concurrency: 4)

    start_supervised!(
      {Nx.Serving,
       serving: serving,
       name: __MODULE__.Partitioned,
       partitions: true,
       batch_size: 8,
       batch_timeout: 10}
    )

    results =
      1..8
      |> Enum.map(fn row ->
        Task.async(fn -> {row, Nx.Serving.batched_run(__MODULE__.Partitioned, batch([row]))} end)
      end)
      |> Task.await_many(30_000)

    for {_row, outputs} <- results do
      {o1, o2, o3} = outputs
      assert Nx.shape(o1) == {1, 10}
      assert Nx.shape(o2) == {1, 10}
      assert Nx.shape(o3) == {1, 10}
    end

    # The same input through the partitioned serving is still deterministic.
    assert values(Nx.Serving.batched_run(__MODULE__.Partitioned, batch([5]))) ==
             values(Nx.Serving.batched_run(__MODULE__.Partitioned, batch([5])))
  end

  # A host app that configured `compiler: EXLA` globally is the Nx-recommended setup; the
  # batch must still run through the Ortex NIF and not be compiled by EXLA (P8).
  test "an inherited EXLA default does not intercept the batch", %{model: model} do
    previous = Nx.Defn.default_options()

    try do
      Nx.Defn.default_options(compiler: EXLA)

      serving = Nx.Serving.new(Ortex.Serving, model)
      {o1, _, _} = Nx.Serving.run(serving, batch([2]))

      assert %Nx.Tensor{data: %Ortex.Backend{}} = o1
      assert Nx.shape(o1) == {1, 10}
    after
      Nx.Defn.default_options(previous)
    end
  end
end
