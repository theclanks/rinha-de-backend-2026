defmodule RinhaFraud.Router do
  use Plug.Router

  plug(:match)
  plug(:dispatch)

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()
  @rf_threshold 0.5
  @fallback_band 0.10
  @fallback_nprobe 64

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
    score = predict(vec)
    respond(conn, score)
  end

  match _ do
    send_resp(conn, 404, "{\"error\":\"not found\"}")
  end

  defp predict(vec) do
    forest = RinhaFraud.VectorStore.rf_forest()
    fraud_prob = RinhaFraud.RandomForest.predict_proba(forest, vec)
    threshold = max(@rf_threshold, forest.threshold)

    if RinhaFraud.VectorStore.ivf14_ready?() and abs(fraud_prob - threshold) <= @fallback_band do
      vec
      |> RinhaFraud.VectorStore.knn14(5, @fallback_nprobe)
      |> Enum.count(fn {_dist, label} -> label == 1 end)
      |> then(fn fraud_count -> fraud_count / 5.0 end)
    else
      if fraud_prob >= threshold, do: 1.0, else: 0.0
    end
  end

  defp respond(conn, score) do
    body = if score < 0.6, do: @resp_approved, else: @resp_rejected

    conn
    |> put_resp_header("content-type", "application/json")
    |> send_resp(200, body)
  end
end
