defmodule RinhaProxy do
  @moduledoc """
  Proxy TCP round-robin simples. Escuta na porta configurada e distribui
  conexoes entre os backends em round-robin.
  """

  def start(port, backends) do
    {:ok, listen_socket} = :gen_tcp.listen(port, [
      :binary, {:active, false}, {:reuseaddr, true}, {:packet, :raw}
    ])

    counter = :atomics.new(1, [])
    accept_loop(listen_socket, backends, counter)
  end

  defp accept_loop(listen_socket, backends, counter) do
    case :gen_tcp.accept(listen_socket) do
      {:ok, client_socket} ->
        idx = :atomics.add(counter, 1, 1)
        backend = Enum.at(backends, rem(idx - 1, length(backends)))
        Task.start(fn -> handle_client(client_socket, backend) end)
        accept_loop(listen_socket, backends, counter)

      {:error, :closed} ->
        :ok

      {:error, _} ->
        :timer.sleep(100)
        accept_loop(listen_socket, backends, counter)
    end
  end

  defp handle_client(client_socket, {backend_host, backend_port}) do
    :inet.setopts(client_socket, [{:active, :once}])

    case :gen_tcp.connect(to_charlist(backend_host), backend_port, [:binary, {:active, :once}, {:packet, :raw}]) do
      {:ok, backend_socket} ->
        relay(client_socket, backend_socket)
        :gen_tcp.close(client_socket)
        :gen_tcp.close(backend_socket)

      {:error, _} ->
        :gen_tcp.send(client_socket, "HTTP/1.1 502 Bad Gateway\r\n\r\n")
        :gen_tcp.close(client_socket)
    end
  end

  defp relay(client, backend) do
    receive do
      {:tcp, ^client, data} ->
        :gen_tcp.send(backend, data)
        :inet.setopts(backend, [{:active, :once}])
        :inet.setopts(client, [{:active, :once}])
        relay(client, backend)

      {:tcp, ^backend, data} ->
        :gen_tcp.send(client, data)
        :inet.setopts(backend, [{:active, :once}])
        :inet.setopts(client, [{:active, :once}])
        relay(client, backend)

      {:tcp_closed, _} ->
        :ok

      {:tcp_error, _, _} ->
        :ok
    after
      30_000 -> :ok
    end
  end
end
