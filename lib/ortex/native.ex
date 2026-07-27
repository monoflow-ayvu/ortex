defmodule Ortex.Native do
  @moduledoc false

  Ortex.Util.copy_ort_libs()

  version = Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :ortex,
    crate: :ortex,
    base_url: "https://github.com/monoflow-ayvu/ortex/releases/download/v#{version}",
    version: version,
    targets: ~w(aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu),
    nif_versions: ~w(2.15)

  # When loading a NIF module, dummy clauses for all NIF function are required.
  # NIF dummies usually just error out when called when the NIF is not loaded, as that should never normally happen.
  def init(_model_path, _execution_providers, _optimization_level, _qnn_opts),
    do: :erlang.nif_error(:nif_not_loaded)

  def run(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def from_binary(_bin, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def to_binary(_reference, _bits, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def show_session(_model), do: :erlang.nif_error(:nif_not_loaded)

  def slice(_tensor, _start_indicies, _lengths, _strides),
    do: :erlang.nif_error(:nif_not_loaded)

  def reshape(_tensor, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def concatenate(_tensors_refs, _type, _axis), do: :erlang.nif_error(:nif_not_loaded)
end
