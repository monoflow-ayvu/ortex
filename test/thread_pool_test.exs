defmodule Ortex.ThreadPoolTest do
  # Thread counts are the observable effect of the pool shape, and they are process-wide:
  # this file must not run while another test loads models.
  use ExUnit.Case, async: false

  @model "./models/tinymodel.onnx"

  unless File.exists?("/proc/self/status") do
    @moduletag :skip
  end

  # These tests assert the default (one shared pool). A VM started with the documented
  # opt-out is a different, equally supported configuration: skip instead of failing.
  if System.get_env("ORTEX_GLOBAL_THREAD_POOL") in ["0", "false", "no", "off"] do
    @moduletag skip: "ORTEX_GLOBAL_THREAD_POOL turns the shared pool off"
  end

  defp threads do
    "/proc/self/status"
    |> File.read!()
    |> String.split("\n")
    |> Enum.find_value(fn line ->
      case Regex.run(~r/^Threads:\s+(\d+)/, line) do
        [_, count] -> String.to_integer(count)
        _ -> nil
      end
    end)
  end

  defp inputs do
    {Nx.broadcast(1, {1, 100}) |> Nx.as_type(:s32),
     Nx.broadcast(1.0, {1, 100}) |> Nx.as_type(:f32)}
  end

  # One run forces the graph and whatever the session created to be materialised.
  defp exercise(models) do
    Enum.each(models, fn model -> Ortex.run(model, inputs()) end)
  end

  # A model loaded with `sessions:` but no `threads:` joins the process-wide pool: the
  # sessions do not add threads, whatever N is. Before the global pool this was one intra-op
  # pool per session (3 threads each on this class of machine), so 4 models x 3 sessions
  # would have added 36.
  test "sessions without thread options share the process-wide pool" do
    seed = Ortex.load!(@model, [:cpu], 3)
    exercise([seed])

    base = threads()
    models = for _ <- 1..4, do: Ortex.load!(@model, [:cpu], 3, sessions: 3)

    added = threads() - base
    assert added <= 2, "4 models x 3 sessions added #{added} threads; the pool is not shared"

    # Last use, after the measurement: keeping the list alive keeps its threads alive, so
    # the count above is not reading a half-collected pool.
    exercise(models)
    exercise([seed])
  end

  # `threads:` still does what it says: this model builds its own pool, because ONNX Runtime
  # ignores per-session thread options while the session is on the global pool.
  test "a model with threads: keeps its own pool" do
    seed = Ortex.load!(@model, [:cpu], 3)
    exercise([seed])

    base = threads()
    models = for _ <- 1..2, do: Ortex.load!(@model, [:cpu], 3, threads: [intra: 4])

    added = threads() - base
    assert added >= 4, "two models with intra: 4 added #{added} threads; they shared the pool"

    {output, _, _} = Ortex.run(hd(models), inputs())
    assert Nx.shape(output) == {1, 10}

    exercise(models)
    exercise([seed])
  end
end
