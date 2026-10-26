defmodule Ortex.TestErrors do
  use ExUnit.Case

  # Which EP is absent depends on how this NIF was built (ORTEX_FEATURES), so pick one
  # that is definitely not compiled in rather than assuming the default feature set.
  @compiled Application.compile_env(:ortex, Ortex.Native, []) |> Keyword.get(:features, [])
  @absent_ep if :cuda in @compiled, do: :tensorrt, else: :cuda

  # The Rustler contract: a Rust-side failure crosses the NIF as
  # {:error, {reason, provider, detail}}, and Ortex.Error is that tuple on the Elixir
  # side. Every test here exercises a path that used to be unreachable in the suite (M4)
  # and used to arrive as a RuntimeError with a string (M6).

  test "run with the wrong number of inputs raises :invalid_input with the expected arity" do
    model = Ortex.load!("./models/tinymodel.onnx")
    x = Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32)

    error =
      assert_raise Ortex.Error, ~r/Expected 2 input\(s\), got 1/, fn ->
        Ortex.run(model, {x})
      end

    assert error.reason == :invalid_input
    assert error.provider == nil
  end

  test "load/4 returns a tagged error instead of raising for a missing model" do
    assert {:error, %Ortex.Error{} = error} = Ortex.load("./models/does-not-exist.onnx")
    assert error.reason == :onnxruntime
    assert error.detail =~ ~r/no such file|not found|does not exist/i
  end

  test "load!/4 raises the same struct load/4 returns" do
    assert {:error, %Ortex.Error{} = error} = Ortex.load("./models/does-not-exist.onnx")

    raised =
      assert_raise Ortex.Error, fn ->
        Ortex.load!("./models/does-not-exist.onnx")
      end

    assert raised == error
  end

  test "an invalid intra value is :invalid_option at load" do
    assert {:error, %Ortex.Error{reason: :invalid_option, detail: detail}} =
             Ortex.load("./models/tinymodel.onnx", [:cpu], 3, threads: [intra: "abc"])

    assert detail =~ "threads: intra must be a non-negative integer"
  end

  test "the grouped options are flattened to the NIF's key/value pairs" do
    assert {:ok, flat} =
             Ortex.Options.to_nif(
               sessions: 2,
               threads: [intra: 1, inter: 2, intra_op_spinning: false],
               trace: "/tmp/ortex.trace",
               env: %{"DSP_LIBRARY_PATH" => "/usr/lib/rfsa/adsp"},
               qnn: [htp_arch: 68]
             )

    assert Enum.sort(flat) ==
             Enum.sort([
               {"sessions", "2"},
               {"intra_threads", "1"},
               {"inter_threads", "2"},
               {"intra_op_spinning", "false"},
               {"trace_path", "/tmp/ortex.trace"},
               {"env.DSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp"},
               {"htp_arch", "68"}
             ])
  end

  test "the deprecated flat form is translated with a warning" do
    warning =
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        assert {:ok, flat} =
                 Ortex.Options.to_nif(
                   sessions: 2,
                   intra_threads: 1,
                   trace_path: "/tmp/legacy.trace",
                   "env.DSP_LIBRARY_PATH": "/usr/lib/rfsa/adsp",
                   htp_arch: 68
                 )

        assert Enum.sort(flat) ==
                 Enum.sort([
                   {"sessions", "2"},
                   {"intra_threads", "1"},
                   {"trace_path", "/tmp/legacy.trace"},
                   {"env.DSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp"},
                   {"htp_arch", "68"}
                 ])
      end)

    assert warning =~ "flat options are deprecated"
  end

  test "an unknown load option names the groups that are accepted" do
    assert {:error, %Ortex.Error{reason: :invalid_option, detail: detail}} =
             Ortex.load("./models/tinymodel.onnx", [:cpu], 3, threads: [intra: 1], sesions: 2)

    assert detail =~ ":sesions"
    assert detail =~ "expected groups"
  end

  test "an ortex-owned key inside qnn: points at the group that owns it" do
    assert {:error, %Ortex.Error{reason: :invalid_option, detail: detail}} =
             Ortex.load("./models/tinymodel.onnx", [:qnn], 3, qnn: [intra_threads: 1])

    assert detail =~ "qnn: intra_threads is an Ortex option"
    assert detail =~ "pass it to threads:"
  end

  test "an execution provider that was not compiled in names the provider and the feature" do
    assert {:error, %Ortex.Error{reason: :execution_provider, provider: provider} = error} =
             Ortex.load("./models/tinymodel.onnx", [@absent_ep])

    assert provider == @absent_ep
    assert error.message =~ "not compiled into this NIF"
    assert error.message =~ "`#{@absent_ep}`"
  end

  test "an unknown execution provider is an :execution_provider error naming it" do
    assert {:error, %Ortex.Error{reason: :execution_provider, provider: :not_a_provider}} =
             Ortex.load("./models/tinymodel.onnx", [:not_a_provider])
  end

  test "from_binary rejects a byte length that does not match the shape" do
    assert {:error, {:invalid_input, nil, detail}} =
             Ortex.Native.from_binary(<<1, 2, 3>>, {2}, {:f, 32})

    assert detail =~ "Binary length mismatch"
    assert detail =~ "expected 8 bytes, got 3"
  end

  test "slice rejects a zero stride instead of panicking inside ndarray" do
    ref =
      Ortex.Native.from_binary(
        <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>,
        {4},
        {:s, 32}
      )

    assert {:error, {:invalid_input, nil, "Slice stride must be non-zero"}} =
             Ortex.Native.slice(ref, [0], [2], [0])
  end

  test "slice rejects arguments that do not match the tensor rank" do
    ref =
      Ortex.Native.from_binary(
        <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>,
        {4},
        {:s, 32}
      )

    assert {:error, {:invalid_input, nil, detail}} =
             Ortex.Native.slice(ref, [0, 0], [2, 2], [1, 1])

    assert detail =~ "must match tensor rank of 1"
  end

  test "concatenate rejects an unknown dtype instead of picking an arm" do
    a = Nx.tensor([1, 2]) |> Nx.backend_transfer(Ortex.Backend)

    assert {:error, {:invalid_input, nil, detail}} =
             Ortex.Native.concatenate([a.data.ref, a.data.ref], {:f, 8}, 0)

    assert detail =~ "Can't concatenate f8 tensors"
  end
end
