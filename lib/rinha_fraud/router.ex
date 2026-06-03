defmodule RinhaFraud.Router do
  use Plug.Router

  plug Plug.Parsers,
    parsers: [:json],
    json_decoder: Jason

  plug :match
  plug :dispatch

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()

  get "/ready" do
    if RinhaFraud.ReadyFlag.ready?() do
      send_resp(conn, 200, "{\"status\":\"ok\"}")
    else
      send_resp(conn, 503, "{\"status\":\"loading\"}")
    end
  end

  post "/fraud-score" do
    t0 = System.monotonic_time(:microsecond)
    payload = conn.body_params

    vec = RinhaFraud.Vectorizer.vectorize(payload, @consts, @mcc_risk)
    t1 = System.monotonic_time(:microsecond)

    case lda_path(vec) do
      {:ok, score} ->
        t2 = System.monotonic_time(:microsecond)
        log_trace(t0, t1 - t0, 0, 0, "lda", score)
        respond(conn, score)

      :miss ->
        # Mahalanobis distance to centroids
        case mahalanobis_path(vec) do
          {:ok, score} ->
            t2 = System.monotonic_time(:microsecond)
            log_trace(t0, t1 - t0, t2 - t1, 0, "mahal", score)
            respond(conn, score)

          :miss ->
            # IVF KNN with SVD+AVX
            t2 = System.monotonic_time(:microsecond)
            nprobe = 10
            try do
              neighbors = RinhaFraud.VectorStore.knn(vec, 5, nprobe)
              t3 = System.monotonic_time(:microsecond)

              fraud_count = Enum.count(neighbors, fn {_, label} -> label == 1 end)
              score = fraud_count / 5.0
              t4 = System.monotonic_time(:microsecond)
              log_trace(t0, t1 - t0, t2 - t1, t3 - t2, "ivf", score)

              respond(conn, score)
            rescue
              e ->
                :logger.error("[ERROR] IVF failed: ~p", [e])
                respond(conn, 0.5)
            end
        end
    end
  end

  match _ do
    send_resp(conn, 404, "{\"error\":\"not found\"}")
  end

  defp lda_path(vec) do
    [{:lda_w, lda_w_bin}] = :ets.lookup(:vector_store, :lda_w)
    [{:lda_w0, lda_w0_bin}] = :ets.lookup(:vector_store, :lda_w0)

    # Project to 8D first
    query_14d = floats_to_binary(vec)
    [{:svd_matrix, svd_matrix}] = :ets.lookup(:vector_store, :svd_matrix)
    query_8d = RinhaFraud.KnnNif.project_svd(query_14d, svd_matrix)

    # LDA decision: score = w^T * x + w0
    w = binary_to_floats(lda_w_bin, 8)
    [w0] = binary_to_floats(lda_w0_bin, 1)
    x = binary_to_floats(query_8d, 8)

    lda_score = dot_product(w, x) + w0

    # Thresholds based on analysis (thresh=10.0 gives 97.8% hit rate with minimal errors)
    cond do
      lda_score > 10.0 -> {:ok, 1.0}
      lda_score < -10.0 -> {:ok, 0.0}
      true -> :miss
    end
  end

  defp mahalanobis_path(vec) do
    [{:svd_matrix, svd_matrix}] = :ets.lookup(:vector_store, :svd_matrix)
    [{:fraud_centroid, fraud_c}] = :ets.lookup(:vector_store, :fraud_centroid)
    [{:legit_centroid, legit_c}] = :ets.lookup(:vector_store, :legit_centroid)
    [{:cov_inv, cov_inv}] = :ets.lookup(:vector_store, :cov_inv)

    # Project to 8D first
    query_14d = floats_to_binary(vec)
    query_8d = RinhaFraud.KnnNif.project_svd(query_14d, svd_matrix)

    # Mahalanobis in 8D space
    d_fraud = mahalanobis_dist_8d(query_8d, fraud_c, cov_inv)
    d_legit = mahalanobis_dist_8d(query_8d, legit_c, cov_inv)
    ratio = d_legit / (d_fraud + d_legit + 1.0e-10)

    cond do
      ratio > 0.70 -> {:ok, 1.0}
      ratio < 0.30 -> {:ok, 0.0}
      true -> :miss
    end
  end

  defp mahalanobis_dist_8d(vec_bin, centroid_bin, cov_inv_bin) do
    vec = binary_to_floats(vec_bin, 8)
    centroid = binary_to_floats(centroid_bin, 8)
    cov_inv = binary_to_floats(cov_inv_bin, 64)

    diff = Enum.zip(vec, centroid) |> Enum.map(fn {v, c} -> v - c end)
    temp = multiply_matrix_vector(cov_inv, diff, 8)
    :math.sqrt(Enum.sum(Enum.zip(diff, temp) |> Enum.map(fn {a, b} -> a * b end)))
  end

  defp binary_to_floats(bin, n) do
    for <<f::float-little-32 <- bin>>, do: f
  end

  defp dot_product(a, b) do
    Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end

  defp multiply_matrix_vector(matrix, vector, n) do
    for i <- 0..(n - 1) do
      row = Enum.slice(matrix, i * n, n)
      Enum.sum(Enum.zip(row, vector) |> Enum.map(fn {a, b} -> a * b end))
    end
  end

  defp respond(conn, score) do
    approved = score < 0.6
    response = Jason.encode!(%{approved: approved, fraud_score: score})
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, response)
  end

  defp floats_to_binary(floats) do
    for f <- floats, into: <<>>, do: <<f::float-little-32>>
  end

  defp log_trace(t0, vec_us, mahal_us, ivf_us, method, score) do
    total = System.monotonic_time(:microsecond) - t0
    :logger.info("[TRACE] ~s vec=~pμs mahal=~pμs ivf=~pμs total=~pμs score=~p",
      [method, vec_us, mahal_us, ivf_us, total, score])
  end
end
