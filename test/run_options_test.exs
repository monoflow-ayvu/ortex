defmodule Ortex.RunOptionsTest do
  use ExUnit.Case

  # tinymodel: x: i32[batch][100], y: f32[batch][100]
  setup_all do
    %{model: Ortex.load!("./models/tinymodel.onnx")}
  end

  defp inputs(row) do
    {Nx.broadcast(row, {1, 100}) |> Nx.as_type(:s32),
     Nx.broadcast(row / 10, {1, 100}) |> Nx.as_type(:f32)}
  end

  test "a run with a terminated handle fails instead of executing", %{model: model} do
    options = Ortex.RunOptions.new(tag: "cancelled-request")
    assert :ok = Ortex.RunOptions.terminate(options)

    error =
      assert_raise Ortex.Error, ~r/terminat/i, fn ->
        Ortex.run(model, inputs(1), run_options: options)
      end

    assert error.reason == :onnxruntime
  end

  test "unterminate/1 makes the handle usable again", %{model: model} do
    options = Ortex.RunOptions.new()

    assert :ok = Ortex.RunOptions.terminate(options)
    assert :ok = Ortex.RunOptions.unterminate(options)

    assert {%Nx.Tensor{shape: {1, 10}}, _, _} = Ortex.run(model, inputs(1), run_options: options)
  end

  test "terminating one handle does not affect a run using another", %{model: model} do
    cancelled = Ortex.RunOptions.new()
    fresh = Ortex.RunOptions.new()

    assert :ok = Ortex.RunOptions.terminate(cancelled)
    assert {%Nx.Tensor{shape: {1, 10}}, _, _} = Ortex.run(model, inputs(2), run_options: fresh)
  end

  test "terminating from another process stops a run in flight", %{model: model} do
    options = Ortex.RunOptions.new(tag: "batch")
    parent = self()

    # A run of tinymodel finishes in microseconds, so a single run cannot be raced
    # reliably. A loop of them is the same thing the real caller does (a serving process
    # that keeps inferring), and the flag is sticky: whichever run is next, it stops.
    spawn_link(fn -> loop(model, options, parent) end)

    assert_receive {:running, _count}, 5_000
    assert :ok = Ortex.RunOptions.terminate(options)
    assert_receive {:stopped, %Ortex.Error{} = error}, 10_000

    assert error.reason == :onnxruntime
    assert error.detail =~ ~r/terminat/i
  end

  test "run_options: rejects anything that is not a handle", %{model: model} do
    assert_raise ArgumentError, ~r/run_options: must be an Ortex.RunOptions/, fn ->
      Ortex.run(model, inputs(0), run_options: :nope)
    end

    assert_raise ArgumentError, ~r/unknown run option/, fn ->
      Ortex.run(model, inputs(0), some_option: true)
    end
  end

  defp loop(model, options, parent) do
    Enum.reduce_while(1..100_000, 0, fn index, _acc ->
      if index == 1_000, do: send(parent, {:running, index})

      try do
        Ortex.run(model, inputs(index), run_options: options)
        {:cont, index}
      rescue
        error in Ortex.Error ->
          send(parent, {:stopped, error})
          {:halt, index}
      end
    end)
  end
end
