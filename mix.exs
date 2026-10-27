defmodule Ortex.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex,
      version: "0.4.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: "https://github.com/monoflow-ayvu/ortex",
      homepage_url: "https://github.com/monoflow-ayvu/ortex",
      docs: [
        main: "readme",
        extras: ["README.md", "CHANGELOG.md"]
      ],
      # `mix docs` is a dev-only task; naming it here means it runs in the env that has
      # ex_doc, without passing MIX_ENV=dev by hand.
      preferred_cli_env: [docs: :dev],
      package: package()
    ]
  end

  # `mix ortex.copy_ort_libs` runs first: it copies libonnxruntime from ORT_LIB_LOCATION
  # into priv/native (a no-op when the variable is unset), which the NIF's $ORIGIN rpath
  # needs for a source build against a runtime that is not installed system-wide.
  defp aliases do
    [
      compile: ["ortex.copy_ort_libs", "compile"]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.38", optional: true},
      {:rustler_precompiled, "~> 0.10"},
      {:nx, "~> 1.0"},
      # Ortex.Memory emits [:ortex, :memory]; nx already depends on telemetry, so this is
      # a declaration, not new weight.
      {:telemetry, "~> 0.4.0 or ~> 1.0"},
      {:tokenizers, "~> 0.5", only: :dev},
      {:ex_doc, "~> 0.40.4", only: :dev, runtime: false},
      {:exla, "~> 1.0", only: :dev},
      {:torchx, "~> 1.0", only: :dev}
    ]
  end

  defp package do
    [
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* native/ortex/src/ config/config.exs
        native/ortex/Cargo.lock native/ortex/Cargo.toml native/ortex/.cargo/config.toml
        checksum-*.exs),
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/monoflow-ayvu/ortex"},
      description: "ONNX Runtime bindings for Elixir"
    ]
  end
end
