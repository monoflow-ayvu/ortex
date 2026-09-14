defmodule Ortex.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex,
      version: "0.2.0-rc.2",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: "https://github.com/elixir-nx/ortex",
      homepage_url: "http://github.com/elixir-nx/ortex",
      docs: [
        main: "readme",
        extras: ["README.md"]
      ],
      package: package()
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
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 1.0"},
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
      links: %{"GitHub" => "https://github.com/elixir-nx/ortex"},
      description: "ONNX Runtime bindings for Elixir"
    ]
  end
end
