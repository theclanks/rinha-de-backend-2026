defmodule RinhaFraud.MixProject do
  use Mix.Project

  def project do
    [
      app: :rinha_fraud,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      test_paths: ["test"]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto],
      mod: {RinhaFraud.Application, []}
    ]
  end

  defp deps do
    [
      {:bandit, "~> 1.5"},
      {:plug, "~> 1.16"},
      {:jason, "~> 1.4"}
    ]
  end
end
