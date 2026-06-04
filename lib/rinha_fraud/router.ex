defmodule RinhaFraud.Router do
  use Plug.Router

  plug :match
  plug :dispatch

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()

  @resp_approved "{\"approved\":true,\"fraud_score\":0.0}"
  @resp_rejected "{\"approved\":false,\"fraud_score\":1.0}"

  get "/ready" do
    if RinhaFraud.ReadyFlag.ready?() do
      send_resp(conn, 200, "{\"status\":\"ok\"}")
    else
      send_resp(conn, 503, "{\"status\":\"loading\"}")
    end
  end

  post "/fraud-score" do
    {:ok, body, conn} = Plug.Conn.read_body(conn)
    payload = Jason.decode!(body)
    vec = RinhaFraud.Vectorizer.vectorize(payload, @consts, @mcc_risk)
    score = lda_predict(vec)
    respond(conn, score)
  end

  match _ do
    send_resp(conn, 404, "{\"error\":\"not found\"}")
  end

  defp lda_predict(vec) do
    [{:lda_w, lda_w_bin}] = :ets.lookup(:vector_store, :lda_w)
    [{:lda_w0, lda_w0_bin}] = :ets.lookup(:vector_store, :lda_w0)
    [{:svd_matrix, svd_matrix}] = :ets.lookup(:vector_store, :svd_matrix)

    vec_bin = for f <- vec, into: <<>>, do: <<f::float-little-32>>
    vec_8d_bin = RinhaFraud.KnnNif.project_svd(vec_bin, svd_matrix)

    w = binary_to_floats(lda_w_bin, 8)
    [w0] = binary_to_floats(lda_w0_bin, 1)

    vec_8d = for <<f::float-little-32 <- vec_8d_bin>>, do: f

    lda_score = dot_product(w, vec_8d) + w0

    cond do
      lda_score > 10.0 -> 1.0
      lda_score < -10.0 -> 0.0
      true -> if lda_score > 0, do: 1.0, else: 0.0
    end
  end

  defp binary_to_floats(bin, _n) do
    for <<f::float-little-32 <- bin>>, do: f
  end

  defp dot_product(a, b) do
    Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end

  defp respond(conn, score) do
    body = if score < 0.6, do: @resp_approved, else: @resp_rejected
    conn
    |> put_resp_header("content-type", "application/json")
    |> send_resp(200, body)
  end
end
