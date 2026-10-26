defmodule Ortex.TestReshape do
  use ExUnit.Case

  test "1d reshape" do
    t = Nx.tensor([1, 2, 3, 4])
    bin = t |> Nx.reshape({2, 2})

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.reshape({2, 2}) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d reshape" do
    shape = Nx.tensor([[0], [0], [0], [0]])
    t = Nx.tensor([1, 2, 3, 4])
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape |> Nx.backend_copy(Ortex.Backend))
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "scalar reshape" do
    shape = {1, 1, 1}
    t = Nx.tensor(1)
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape)
      |> Nx.backend_transfer()

    assert bin == ort
  end

  # The shape alone proves nothing: {2,3} -> {3,2} and {2,3} -> {6} differ only in the
  # order the elements come out, so check the values on the Ortex side of the transfer.
  test "reshape preserves the element order" do
    t = Nx.tensor([[1, 2, 3], [4, 5, 6]])

    ortex =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape({3, 2})

    assert %Nx.Tensor{data: %Ortex.Backend{}, shape: {3, 2}} = ortex
    assert Nx.to_flat_list(ortex) == [1, 2, 3, 4, 5, 6]
    assert Nx.backend_transfer(ortex) == Nx.reshape(t, {3, 2})
  end

  test "reshape to a shape with a different number of elements raises" do
    t = Nx.tensor([[1, 2, 3], [4, 5, 6]]) |> Nx.backend_copy(Ortex.Backend)

    assert_raise ArgumentError, fn -> Nx.reshape(t, {4, 2}) end
  end

  test "auto reshape" do
    shape = {:auto, 2}
    t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape)
      |> Nx.backend_transfer()

    assert bin == ort
  end
end
