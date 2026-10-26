defmodule Ortex.Util do
  @moduledoc false

  @doc """
  Copies the ONNX Runtime shared library from `ORT_LIB_LOCATION` into `priv/native`.

  Ortex loads ONNX Runtime with `dlopen()` (ort's `load-dynamic` feature) and the NIF is
  linked with an `$ORIGIN` rpath, so a library sitting in `priv/native` next to the NIF is
  found without any environment variable. That is the only reason this exists: a source
  build against a runtime that is not installed system-wide.

  It used to run from `Ortex.Native`'s module body, which meant it also ran - as a side
  effect - while a *consumer* compiled the dependency, globbing `native/ortex/target/**`
  for whatever `libonnxruntime*` a previous build left behind, before the precompiled NIF
  had even been downloaded. Scoped now to the one case it serves:

    * only with `ORT_LIB_LOCATION` set (nothing to copy otherwise),
    * only from that location (its `lib/` and `lib64/` subdirectories included),
    * and only when explicitly called - see `Mix.Tasks.Ortex.CopyOrtLibs`.

  Returns the list of copied paths.
  """
  @spec copy_ort_libs() :: [Path.t()]
  def copy_ort_libs do
    case System.get_env("ORT_LIB_LOCATION") do
      nil ->
        []

      location ->
        destination = Path.join(priv_dir(), "native")
        paths = location |> sources() |> Enum.flat_map(&Path.wildcard/1) |> Enum.uniq()

        case paths do
          [] ->
            raise """
            Unable to locate libonnxruntime binaries.
            ORT_LIB_LOCATION: #{location}
            Searched: #{Enum.join(sources(location), ", ")}
            """

          paths ->
            File.mkdir_p!(destination)

            Enum.map(paths, fn path ->
              target = Path.join(destination, Path.basename(path))
              File.cp!(path, target)
              target
            end)
        end
    end
  end

  defp priv_dir do
    :ortex |> :code.priv_dir() |> to_string()
  end

  # A file means "copy exactly this"; a directory means the usual lib/ + lib64/ layout.
  defp sources(location) do
    expanded = Path.expand(location)

    if File.dir?(expanded) do
      [expanded, Path.join(expanded, "lib"), Path.join(expanded, "lib64")]
      |> Enum.filter(&File.dir?/1)
      |> Enum.map(&lib_glob/1)
      |> Enum.uniq()
    else
      [expanded]
    end
  end

  defp lib_glob(base) do
    suffix =
      case :os.type() do
        {:win32, _} -> "libonnxruntime*.dll*"
        {:unix, :darwin} -> "libonnxruntime*.dylib*"
        {:unix, _} -> "libonnxruntime*.so*"
      end

    Path.join([base, suffix])
  end
end
