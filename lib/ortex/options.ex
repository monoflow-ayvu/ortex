defmodule Ortex.Options do
  @moduledoc false
  # `Ortex.load/4`'s options: grouped the way a caller thinks about them, flattened into
  # the `{key, value}` pairs the NIF takes.
  #
  #   sessions: 4
  #   threads: [intra: 1, inter: 2, intra_op_spinning: false]
  #   trace: "/tmp/ortex.trace"
  #   env: %{"DSP_LIBRARY_PATH" => "/usr/lib/rfsa/adsp"}
  #   qnn: [htp_arch: 68, provider_path: "/usr/lib/libonnxruntime_providers_qnn.so"]
  #
  # The flat form (`intra_threads: 1`, `trace_path: ...`, `htp_arch: 68`, `"env.FOO"`)
  # still works for one release cycle and is translated here, with a warning: `qnn_opts`
  # used to be the name for a bag that configured ortex itself (thread pools, trace,
  # environment) as much as the QNN provider, which is what R5 in the audit called out.

  @groups [:qnn, :threads, :trace, :env, :sessions]

  # Keys ortex itself owns. They are not QNN provider options and must not be forwarded.
  @reserved_qnn ~w(backend_path provider_path)

  @type t :: %{
          optional(:sessions) => non_neg_integer(),
          optional(:threads) => keyword(),
          optional(:trace) => Path.t(),
          optional(:env) => map() | keyword(),
          optional(:qnn) => keyword() | map()
        }

  @doc """
  Validates `Ortex.load/4` options and flattens them for the NIF.
  """
  @spec to_nif(keyword()) :: {:ok, [{String.t(), String.t()}]} | {:error, Ortex.Error.t()}
  def to_nif(opts) when is_list(opts) do
    with {:ok, grouped} <- normalize(opts),
         {:ok, grouped} <- validate(grouped) do
      {:ok, flatten(grouped)}
    end
  end

  def to_nif(other) do
    {:error, invalid("load options must be a keyword list, got: #{inspect(other)}")}
  end

  # Grouped form, or the deprecated flat one. `qnn_opts:` is also accepted as the
  # deprecated alias for the same flat form.
  defp normalize(opts) do
    {flat, opts} = Keyword.pop(opts, :qnn_opts)

    cond do
      flat != nil ->
        warn_deprecated("qnn_opts:")
        {:ok, merge_flat(opts, flat)}

      opts == [] ->
        {:ok, %{}}

      Enum.all?(opts, fn {key, _} -> key in @groups end) ->
        {:ok, Map.new(opts)}

      # A grouped option next to a flat one: the caller is halfway through the migration,
      # and guessing which keys were meant as QNN provider options is how typos become
      # silently forwarded provider options.
      Enum.any?(opts, fn {key, _} -> key in (@groups -- [:sessions]) end) ->
        unknown = Enum.reject(opts, fn {key, _} -> key in @groups end)
        {:error, invalid(unknown_message(unknown))}

      true ->
        warn_deprecated(hd(opts) |> elem(0))
        {:ok, from_flat(opts)}
    end
  end

  defp merge_flat(grouped, flat) do
    grouped
    |> Enum.reduce(%{}, fn
      {:qnn, qnn}, acc -> Map.update(acc, :qnn, qnn, &(&1 ++ qnn))
      {:threads, threads}, acc -> Map.update(acc, :threads, threads, &(&1 ++ threads))
      {:env, env}, acc -> Map.update(acc, :env, env, &Map.merge(&1, Map.new(env)))
      other, acc -> Map.put(acc, elem(other, 0), elem(other, 1))
    end)
    |> then(&from_flat(flat, &1))
  end

  defp from_flat(flat, base \\ %{}) do
    Enum.reduce(flat, base, fn {key, value}, acc ->
      key = to_string(key)

      cond do
        key == "sessions" ->
          Map.put(acc, :sessions, value)

        key == "trace_path" ->
          Map.put(acc, :trace, value)

        key in ~w(intra_threads inter_threads intra_op_spinning inter_op_spinning) ->
          Map.update(
            acc,
            :threads,
            [{thread_option(key), value}],
            &(&1 ++ [{thread_option(key), value}])
          )

        String.starts_with?(key, "env.") ->
          name = String.replace_prefix(key, "env.", "")
          Map.update(acc, :env, %{name => value}, &Map.put(&1, name, value))

        true ->
          Map.update(acc, :qnn, [{key, value}], &(&1 ++ [{key, value}]))
      end
    end)
  end

  defp validate(grouped) do
    with :ok <- validate_sessions(grouped),
         :ok <- validate_threads(grouped),
         :ok <- validate_trace(grouped),
         :ok <- validate_env(grouped),
         :ok <- validate_qnn(grouped) do
      {:ok, grouped}
    end
  end

  defp validate_sessions(grouped) do
    case Map.fetch(grouped, :sessions) do
      :error ->
        :ok

      {:ok, count} when is_integer(count) and count > 0 ->
        :ok

      {:ok, other} ->
        {:error, invalid("sessions: must be a positive integer, got: #{inspect(other)}")}
    end
  end

  defp validate_threads(grouped) do
    case Map.fetch(grouped, :threads) do
      :error ->
        :ok

      {:ok, threads} when is_list(threads) ->
        Enum.reduce_while(threads, :ok, fn {key, value}, _acc ->
          cond do
            key not in [:intra, :inter, :intra_op_spinning, :inter_op_spinning] ->
              {:halt,
               {:error,
                invalid(
                  "threads: #{inspect(key)} is not a thread option; expected one of " <>
                    ":intra, :inter, :intra_op_spinning, :inter_op_spinning"
                )}}

            key in [:intra, :inter] and not (is_integer(value) and value >= 0) ->
              {:halt,
               {:error,
                invalid("threads: #{key} must be a non-negative integer, got: #{inspect(value)}")}}

            key in [:intra_op_spinning, :inter_op_spinning] and not is_boolean(value) ->
              {:halt,
               {:error, invalid("threads: #{key} must be a boolean, got: #{inspect(value)}")}}

            true ->
              {:cont, :ok}
          end
        end)

      {:ok, other} ->
        {:error, invalid("threads: must be a keyword list, got: #{inspect(other)}")}
    end
  end

  defp validate_trace(grouped) do
    case Map.fetch(grouped, :trace) do
      :error -> :ok
      {:ok, path} when is_binary(path) -> :ok
      {:ok, other} -> {:error, invalid("trace: must be a path, got: #{inspect(other)}")}
    end
  end

  defp validate_env(grouped) do
    case Map.fetch(grouped, :env) do
      :error ->
        :ok

      {:ok, env} when is_map(env) or is_list(env) ->
        Enum.reduce_while(env, :ok, fn {name, value}, _acc ->
          if is_binary(to_string(name)) and (is_binary(value) or is_number(value)) do
            {:cont, :ok}
          else
            {:halt, {:error, invalid("env: #{inspect(name)} must map to a string or number")}}
          end
        end)

      {:ok, other} ->
        {:error, invalid("env: must be a map or keyword list, got: #{inspect(other)}")}
    end
  end

  defp validate_qnn(grouped) do
    case Map.fetch(grouped, :qnn) do
      :error ->
        :ok

      {:ok, qnn} when is_map(qnn) or is_list(qnn) ->
        Enum.reduce_while(qnn, :ok, fn {key, _value}, _acc ->
          key = to_string(key)

          if key in @reserved_qnn or key not in ortex_owned_keys() do
            {:cont, :ok}
          else
            {:halt,
             {:error,
              invalid(
                "qnn: #{key} is an Ortex option, not a QNN provider option; " <>
                  "pass it to #{destination(key)} instead"
              )}}
          end
        end)

      {:ok, other} ->
        {:error, invalid("qnn: must be a keyword list or map, got: #{inspect(other)}")}
    end
  end

  # Keys the NIF reads for itself (model.rs RESERVED). Passing them inside `qnn:` would
  # be silently dropped by the Rust side, so reject them with the group that owns them.
  defp ortex_owned_keys do
    ~w(sessions trace_path intra_threads inter_threads intra_op_spinning inter_op_spinning)
  end

  defp destination("sessions"), do: "sessions:"
  defp destination("trace_path"), do: "trace:"
  defp destination(_), do: "threads:"

  defp flatten(grouped) do
    Enum.flat_map(grouped, fn
      {:sessions, count} ->
        [{"sessions", to_string(count)}]

      {:trace, path} ->
        [{"trace_path", path}]

      {:threads, threads} ->
        Enum.map(threads, fn {key, value} -> {thread_key(key), to_string(value)} end)

      {:env, env} ->
        Enum.map(env, fn {name, value} -> {"env.#{name}", to_string(value)} end)

      {:qnn, qnn} ->
        Enum.map(qnn, fn {key, value} -> {to_string(key), to_string(value)} end)
    end)
  end

  defp thread_key(:intra), do: "intra_threads"
  defp thread_key(:inter), do: "inter_threads"
  defp thread_key(:intra_op_spinning), do: "intra_op_spinning"
  defp thread_key(:inter_op_spinning), do: "inter_op_spinning"

  defp thread_option("intra_threads"), do: :intra
  defp thread_option("inter_threads"), do: :inter
  defp thread_option("intra_op_spinning"), do: :intra_op_spinning
  defp thread_option("inter_op_spinning"), do: :inter_op_spinning

  defp invalid(detail), do: Ortex.Error.exception(reason: :invalid_option, detail: detail)

  defp unknown_message(unknown) do
    keys = Enum.map_join(unknown, ", ", fn {key, _} -> inspect(key) end)

    "unknown load option(s): #{keys}; expected groups are :qnn, :threads, :trace, :env " <>
      "and :sessions (the flat form is deprecated)"
  end

  defp warn_deprecated(key) do
    IO.warn(
      "Ortex.load/4: flat options are deprecated (#{inspect(key)} was passed). " <>
        "Use the grouped form instead: sessions:, threads: [intra:, inter:, " <>
        "intra_op_spinning:, inter_op_spinning:], trace:, env: %{}, qnn: [...]."
    )
  end
end
