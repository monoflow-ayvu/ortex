defmodule OrtexTest do
  use ExUnit.Case
  doctest Ortex

  @tag :resnet50
  test "resnet50" do
    model = Ortex.load("./models/resnet50.onnx")

    input = Nx.broadcast(0.0, {1, 3, 224, 224})
    {output} = Ortex.run(model, {input})
    argmax = output |> Nx.backend_transfer() |> Nx.argmax(axis: 1)

    assert argmax == Nx.tensor([499])
  end

  @tag :resnet50
  test "Nx.Serving with resnet50" do
    model = Ortex.load("./models/resnet50.onnx")

    serving = Nx.Serving.new(Ortex.Serving, model)
    batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
    {result} = Nx.Serving.run(serving, batch)
    assert result |> Nx.backend_transfer() |> Nx.argmax(axis: 1) == Nx.tensor([499])
  end

  test "Nx.Serving with tinymodel" do
    model = Ortex.load("./models/tinymodel.onnx")

    serving = Nx.Serving.new(Ortex.Serving, model)

    # Create a batch of size 3 with {int32, float32} inputs
    batch =
      Nx.Batch.stack([
        {Nx.broadcast(0, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(0.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(1, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(1.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(2, {100}) |> Nx.as_type(:s32), Nx.broadcast(2.0, {100}) |> Nx.as_type(:f32)}
      ])

    {%Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}} =
      Nx.Serving.run(serving, batch)
  end

  test "Nx.Serving ignores an app-default defn compiler" do
    model = Ortex.load("./models/tinymodel.onnx")
    original_defaults = Nx.Defn.default_options()

    try do
      Nx.Defn.default_options(Keyword.put(original_defaults, :compiler, EXLA))
      serving = Nx.Serving.new(Ortex.Serving, model)

      single =
        {Nx.broadcast(0, {100}) |> Nx.as_type(:s32), Nx.broadcast(0.0, {100}) |> Nx.as_type(:f32)}

      {%Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}} =
        Nx.Serving.run(serving, Nx.Batch.stack([single, single, single]))
    after
      Nx.Defn.default_options(original_defaults)
    end
  end

  test "inspect uses nil for dynamic dimensions and a readable element type" do
    model = Ortex.load("./models/tinymodel.onnx")

    format = inspect(model)

    assert format =~ ~S({"x", "Int32", [nil, 100]})
    assert format =~ ~S({"y", "Float32", [nil, 100]})
    assert format =~ ~S({"output1", "Float32", [nil, 10]})
    refute format =~ "-1"
    refute format =~ "Tensor {"
  end
end
