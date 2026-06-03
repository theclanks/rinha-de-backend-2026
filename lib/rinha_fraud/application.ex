defmodule RinhaFraud.Application do
  use Application

  @impl true
  def start(_type, _args) do
    is_proxy = System.get_env("IS_PROXY") in ["true", "1"]

    case is_proxy do
      true ->
        IO.puts("=== PROXY MODE STARTED ===")
        port = String.to_integer(System.get_env("PORT", "9999"))
        backend1 = System.get_env("BACKEND_1_HOST", "api1")
        backend2 = System.get_env("BACKEND_2_HOST", "api2")

        IO.puts("Starting proxy on port #{port}, connecting to #{backend1} and #{backend2}")

        :persistent_term.put({:proxy_rr_counter, :counter}, :atomics.new(1, []))
        IO.puts("Counter initialized: #{inspect(:persistent_term.get({:proxy_rr_counter, :counter}))}")

        children = [
          {Bandit,
           plug: RinhaFraud.ProxyRouter,
           port: port,
           startup_log: false,
           thousand_island_options: [num_acceptors: 64]}
        ]

        opts = [strategy: :one_for_one, name: RinhaFraud.Supervisor]
        {:ok, pid} = Supervisor.start_link(children, opts)

        Task.start(fn ->
          :timer.sleep(2000)
          connect_to_backend(backend1)
          connect_to_backend(backend2)
          IO.puts("Connected nodes: #{inspect(Node.list())}")
        end)

        {:ok, pid}

      false ->
        port = String.to_integer(System.get_env("PORT", "9999"))

        children = [
          RinhaFraud.VectorStore,
          RinhaFraud.ReadyFlag,
          {Bandit,
           plug: RinhaFraud.Router,
           port: port,
           startup_log: false,
           thousand_island_options: [num_acceptors: 64]}
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

  defp connect_to_backend(host) do
    node_name = :"rinha_fraud@#{host}"
    IO.puts("Attempting to connect to #{node_name}...")
    case Node.connect(node_name) do
      true -> IO.puts("Connected to #{node_name}")
      false -> 
        IO.puts("Failed to connect to #{node_name}")
        IO.puts("Current node: #{Node.self()}")
    end
  end
end
