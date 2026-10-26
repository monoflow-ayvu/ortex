defmodule Ortex.TestErrors do
  use ExUnit.Case

  # Which EP is absent depends on how this NIF was built (ORTEX_FEATURES), so pick one
  # that is definitely not compiled in rather than assuming the default feature set.
  @compiled Application.compile_env(:ortex, Ortex.Native, []) |> Keyword.get(:features, [])
  @absent_ep if :cuda in @compiled, do: :tensorrt, else: :cuda

  # The Rustler contract: a Rust-side failure crosses the NIF as {:error, term}, and this
  # backend raises it. Every test here exercises a path that used to be unreachable in
  # the suite (M4).

  test "run with the wrong number of inputs raises with the expected arity" do
    model = Ortex.load("./models/tinymodel.onnx")
    x = Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32)

    assert_raise RuntimeError, ~r/Expected 2 input\(s\), got 1/, fn ->
      Ortex.run(model, {x})
    end
  end

  test "load of a missing model raises instead of returning a broken session" do
    assert_raise RuntimeError, ~r/no such file|not found|does not exist/i, fn ->
      Ortex.load("./models/does-not-exist.onnx")
    end
  end

  test "an invalid intra_threads value is rejected at load" do
    assert_raise RuntimeError, ~r/intra_threads must be a non-negative integer/, fn ->
      Ortex.load("./models/tinymodel.onnx", [:cpu], 3, intra_threads: "abc")
    end
  end

  test "an execution provider that was not compiled in raises instead of running on CPU" do
    error =
      assert_raise RuntimeError, fn ->
        Ortex.load("./models/tinymodel.onnx", [@absent_ep])
      end

    assert error.message =~ "not compiled into this NIF"
    assert error.message =~ "`#{@absent_ep}`"
  end

  test "from_binary rejects a byte length that does not match the shape" do
    assert {:error, msg} = Ortex.Native.from_binary(<<1, 2, 3>>, {2}, {:f, 32})
    assert msg =~ "Binary length mismatch"
    assert msg =~ "expected 8 bytes, got 3"
  end

  test "slice rejects a zero stride instead of panicking inside ndarray" do
    ref =
      Ortex.Native.from_binary(
        <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>,
        {4},
        {:s, 32}
      )

    assert {:error, "Slice stride must be non-zero"} = Ortex.Native.slice(ref, [0], [2], [0])
  end

  test "slice rejects arguments that do not match the tensor rank" do
    ref =
      Ortex.Native.from_binary(
        <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>,
        {4},
        {:s, 32}
      )

    assert {:error, msg} = Ortex.Native.slice(ref, [0, 0], [2, 2], [1, 1])
    assert msg =~ "must match tensor rank of 1"
  end

  test "concatenate rejects an unknown dtype instead of picking an arm" do
    a = Nx.tensor([1, 2]) |> Nx.backend_transfer(Ortex.Backend)

    assert {:error, msg} =
             Ortex.Native.concatenate([a.data.ref, a.data.ref], {:f, 8}, 0)

    assert msg =~ "Can't concatenate f8 tensors"
  end
end
