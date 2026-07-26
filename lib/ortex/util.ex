defmodule Ortex.Util do
  @moduledoc false

  @lib_dirs ~w(
    native/ortex/release
    native/ortex/debug
    native/ortex/target/release
    native/ortex/target/debug
    native/ortex/target/**
  )

  @doc """
    Copies the libraries downloaded during the ORT build into a path that
    Elixir can use
  """
  def copy_ort_libs() do
    build_root = Path.absname(:code.priv_dir(:ortex)) |> Path.dirname()
    ort_lib_location = System.get_env("ORT_LIB_LOCATION")
    destination_dir = Path.join([:code.priv_dir(:ortex), "native"])
    File.mkdir_p!(destination_dir)

    search_patterns =
      [build_root, find_project_root(build_root), find_project_root(File.cwd!())]
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()
      |> Enum.flat_map(&patterns_for_root/1)
      |> Enum.concat(ort_lib_location_patterns(ort_lib_location))
      |> Enum.uniq()

    case search_patterns |> Enum.flat_map(&Path.wildcard/1) |> Enum.uniq() do
      [] ->
        # Finding nothing is only fatal when the user pointed us at a location and
        # priv/native wasn't already populated by an earlier build.
        if ort_lib_location && Path.wildcard(lib_glob(destination_dir)) == [] do
          raise """
          Unable to locate libonnxruntime binaries.
          ORT_LIB_LOCATION: #{ort_lib_location}
          Searched: #{Enum.join(search_patterns, ", ")}
          Destination: #{destination_dir}
          """
        end

        :ok

      paths ->
        Enum.each(paths, &File.cp!(&1, Path.join(destination_dir, Path.basename(&1))))
    end
  end

  defp patterns_for_root(root) do
    Enum.map(@lib_dirs, &lib_glob(Path.join(root, &1)))
  end

  defp ort_lib_location_patterns(nil), do: []

  defp ort_lib_location_patterns(path) do
    expanded = Path.expand(path)

    if File.dir?(expanded) do
      [expanded, Path.join(expanded, "lib"), Path.join(expanded, "lib64")]
      |> Enum.filter(&File.dir?/1)
      |> Enum.map(&lib_glob/1)
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

  defp find_project_root(path) do
    expanded = Path.expand(path)
    parent = Path.dirname(expanded)

    cond do
      File.exists?(Path.join(expanded, "mix.exs")) -> expanded
      parent == expanded -> nil
      true -> find_project_root(parent)
    end
  end
end
