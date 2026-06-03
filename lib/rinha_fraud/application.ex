defmodule RinhaFraud.Application do
  use Application

  @impl true
  def start(_type, _args) do
    case System.get_env("MODE", "api") do
      "proxy" ->
        port = String.to_integer(System.get_env("PORT", "9999"))
        backends = [
          {System.get_env("BACKEND_1_HOST", "api1"), 9999},
          {System.get_env("BACKEND_2_HOST", "api2"), 9999}
        ]

        Task.start(fn -> RinhaProxy.start(port, backends) end)
        Supervisor.start_link([], strategy: :one_for_one)

      _ ->
        children = [
          RinhaFraud.VectorStore,
          RinhaFraud.ReadyFlag,
          {Bandit, plug: RinhaFraud.Router, port: 9999, startup_log: false}
        ]

        opts = [strategy: :one_for_one, name: RinhaFraud.Supervisor]
        {:ok, pid} = Supervisor.start_link(children, opts)

        Task.start(fn ->
          RinhaFraud.ReferencesLoader.load!()
          RinhaFraud.ReadyFlag.set_ready()
        end)

        {:ok, pid}
    end
  end
end
