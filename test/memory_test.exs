defmodule Ortex.MemoryTest do
  use ExUnit.Case

  # 64 x 64 f32 = 16 384 bytes
  @tensor_bytes 64 * 64 * 4

  def handle_event(event, measurements, metadata, config) do
    send(config, {event, measurements, metadata})
  end

  defp allocate_outside_this_process do
    parent = self()

    pid =
      spawn(fn ->
        tensor = Nx.broadcast(1.0, {64, 64}) |> Nx.backend_transfer(Ortex.Backend)
        # Keep the tensor alive while the caller reads the counter, so the measurement is
        # not racing a collection of a term it just made.
        send(parent, {:allocated, Ortex.memory()})
        receive do: (:release -> :ok)
        _ = tensor
        :ok
      end)

    receive do
      {:allocated, allocated} -> {pid, allocated}
    after
      5_000 -> flunk("allocation process did not report back")
    end
  end

  test "memory/0 charges fresh tensor buffers and releases them with the resource" do
    baseline = Ortex.memory()
    {pid, allocated} = allocate_outside_this_process()

    assert allocated - baseline >= @tensor_bytes,
           "expected at least #{@tensor_bytes} bytes to be accounted, got #{allocated - baseline}"

    # The resource dies with the process that owns the last reference, which is the only
    # way to drop a term from a test: the Rust destructor is what decrements the counter.
    monitor = Process.monitor(pid)
    send(pid, :release)
    assert_receive {:DOWN, ^monitor, :process, ^pid, _reason}, 5_000

    assert Ortex.memory() <= allocated - @tensor_bytes
  end

  test "poll/0 emits [:ortex, :memory] with the same value it returns" do
    handler = "ortex-memory-test-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler, [:ortex, :memory], &__MODULE__.handle_event/4, self())
    on_exit(fn -> :telemetry.detach(handler) end)

    assert %{tensor_bytes: bytes} = measurements = Ortex.Memory.poll()
    assert measurements == %{tensor_bytes: bytes}
    assert Ortex.Memory.event() == [:ortex, :memory]

    assert_receive {[:ortex, :memory], %{tensor_bytes: ^bytes}, %{node: node}}
    assert node == node()
  end
end
