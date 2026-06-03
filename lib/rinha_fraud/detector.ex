defmodule RinhaFraud.Detector do
  @moduledoc """
  Core fraud detection logic, callable via RPC or locally.
  """

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()

  def detect(payload) do
    vec = RinhaFraud.Vectorizer.vectorize(payload, @consts, @mcc_risk)
    lda_path(vec)
  end

  defp lda_path(vec) do
    [{:lda_w, lda_w_bin}] = :ets.lookup(:vector_store, :lda_w)
    [{:lda_w0, lda_w0_bin}] = :ets.lookup(:vector_store, :lda_w0)

    query_14d = floats_to_binary(vec)
    [{:svd_matrix, svd_matrix}] = :ets.lookup(:vector_store, :svd_matrix)
    query_8d = RinhaFraud.KnnNif.project_svd(query_14d, svd_matrix)

    w = binary_to_floats(lda_w_bin, 8)
    [w0] = binary_to_floats(lda_w0_bin, 1)
    x = binary_to_floats(query_8d, 8)

    lda_score = dot_product(w, x) + w0

    cond do
      lda_score > 10.0 -> {:ok, 1.0}
      lda_score < -10.0 -> {:ok, 0.0}
      true -> {:ok, 0.5}
    end
  end

  defp binary_to_floats(bin, n) do
    for <<f::float-little-32 <- bin>>, do: f
  end

  defp dot_product(a, b) do
    Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end

  defp floats_to_binary(floats) do
    for f <- floats, into: <<>>, do: <<f::float-little-32>>
  end
end
