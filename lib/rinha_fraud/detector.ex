defmodule RinhaFraud.Detector do
  @moduledoc """
  Core fraud detection logic, callable via RPC or locally.
  """

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()
  @lda_threshold 10.0

  def detect(payload) do
    vec = RinhaFraud.Vectorizer.vectorize(payload, @consts, @mcc_risk)
    {lda_w_bin, lda_w0_bin, cart_tree_bin} = RinhaFraud.VectorStore.lda_cart14()
    w = binary_to_floats(lda_w_bin)
    [w0] = binary_to_floats(lda_w0_bin)
    lda_score = dot_product(w, vec) + w0

    score =
      cond do
        lda_score > @lda_threshold ->
          1.0

        lda_score < -@lda_threshold ->
          0.0

        true ->
          vec_bin = for f <- vec, into: <<>>, do: <<f::float-little-32>>
          fraud_prob = RinhaFraud.KnnNif.cart_predict_14(vec_bin, cart_tree_bin)
          if fraud_prob > 0.5, do: 1.0, else: 0.0
      end

    {:ok, score}
  end

  defp binary_to_floats(bin), do: for(<<f::float-little-32 <- bin>>, do: f)

  defp dot_product(a, b) do
    Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end
end
