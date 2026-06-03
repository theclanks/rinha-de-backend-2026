defmodule RinhaFraud.ProxyRouter do
  use Plug.Router
  require Logger

  plug Plug.Parsers,
    parsers: [:json],
    json_decoder: Jason

  plug :match
  plug :dispatch

  get "/ready" do
    send_resp(conn, 200, "{\"status\":\"ok\"}")
  end

  post "/fraud-score" do
    payload = conn.body_params
    node = pick_node()

    case :rpc.call(node, RinhaFraud.Detector, :detect, [payload], 1000) do
      {:ok, score} ->
        respond(conn, score)
      {:badrpc, _} ->
        respond(conn, 0.5)
      :timeout ->
        respond(conn, 0.5)
    end
  end

  match _ do
    send_resp(conn, 404, "{\"error\":\"not found\"}")
  end

  defp pick_node do
    nodes = Node.list()
    if nodes == [] do
      Logger.warning("No backend nodes connected")
      raise "No backend nodes"
    end

    counter = :persistent_term.get({:proxy_rr_counter, :counter})
    idx = :atomics.add_get(counter, 1, 1)
    Enum.at(nodes, rem(idx - 1, length(nodes)))
  end

  defp respond(conn, score) do
    approved = score < 0.6
    response = Jason.encode!(%{approved: approved, fraud_score: score})
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, response)
  end
end
