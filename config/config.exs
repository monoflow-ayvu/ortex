import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

# Cargo feature flags for the execution provider to compile in. Defaults to what
# the host OS can use, overridable with e.g. ORTEX_FEATURES=cuda,tensorrt
default_features =
  case :os.type() do
    {:win32, _} -> ["directml"]
    {:unix, :darwin} -> ["coreml"]
    {:unix, _} -> []
  end

ortex_features =
  case System.get_env("ORTEX_FEATURES") do
    nil -> default_features
    features -> String.split(features, ",", trim: true)
  end

config :ortex, Ortex.Native, features: ortex_features

# Ortex itself always compiles the crate from source; precompiled artifacts are for
# projects that depend on ortex.
config :rustler_precompiled, :force_build, ortex: true
