defmodule RinhaFraud.Detector do
  @moduledoc """
  Core fraud detection logic, callable via RPC or locally.
  """

  @consts RinhaFraud.Vectorizer.load_consts()
  @mcc_risk RinhaFraud.Vectorizer.load_mcc_risk()

  def detect(payload) do
    vec = RinhaFraud.Vectorizer.vectorize(payload, @consts, @mcc_risk)
    forest = RinhaFraud.VectorStore.rf_forest()
    threshold = forest.threshold
    fraud_prob = RinhaFraud.RandomForest.predict_proba(forest, vec)

    score =
      if RinhaFraud.VectorStore.ivf14_ready?() and abs(fraud_prob - threshold) <= 0.10 do
        vec
        |> RinhaFraud.VectorStore.knn14(5, 64)
        |> Enum.count(fn {_dist, label} -> label == 1 end)
        |> then(fn fraud_count -> fraud_count / 5.0 end)
      else
        if fraud_prob >= threshold, do: 1.0, else: 0.0
      end

    {:ok, score}
  end
end
