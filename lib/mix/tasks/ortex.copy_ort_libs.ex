defmodule Mix.Tasks.Ortex.CopyOrtLibs do
  @shortdoc "Copies libonnxruntime from ORT_LIB_LOCATION into priv/native"

  @moduledoc """
  Copies the ONNX Runtime shared library into `priv/native`, where the NIF's `$ORIGIN`
  rpath finds it.

  Only needed for a build from source against an ORT that is not installed system-wide:

      ORT_LIB_LOCATION=/path/to/onnxruntime/lib mix ortex.copy_ort_libs

  The `compile` alias in this project runs it automatically, so a normal `mix compile` in
  this repo picks the library up. Set `ORT_DYLIB_PATH` at runtime instead if you would
  rather keep the runtime where it is.
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Mix.Task.run("app.config")

    case Ortex.Util.copy_ort_libs() do
      [] ->
        :ok

      paths ->
        Mix.shell().info("Copied ONNX Runtime libraries: #{Enum.join(paths, ", ")}")
    end
  end
end
