{:ok, _pid} = RinhaFraud.VectorStore.start_link()
RinhaFraud.ReferencesLoader.load!("resources")

entries =
  "test/test-data.json"
  |> File.read!()
  |> Jason.decode!()
  |> Map.fetch!("entries")

initial = %{tp: 0, fp: 0, fn: 0, tn: 0, fallback: 0}

stats =
  Enum.reduce(entries, initial, fn entry, acc ->
    payload = entry["request"]
    expected_fraud? = not entry["expected_approved"]
    {:ok, score} = RinhaFraud.Detector.detect(payload)
    pred_fraud? = score >= 0.6

    cond do
      pred_fraud? and expected_fraud? -> %{acc | tp: acc.tp + 1}
      pred_fraud? -> %{acc | fp: acc.fp + 1}
      expected_fraud? -> %{acc | fn: acc.fn + 1}
      true -> %{acc | tn: acc.tn + 1}
    end
  end)

total = length(entries)
weighted = stats.fp + 3 * stats.fn
penalty = stats.fp * 25 + stats.fn * 90
acc = (stats.tp + stats.tn) / total

IO.puts(
  "total=#{total} acc=#{Float.round(acc, 5)} TP=#{stats.tp} FP=#{stats.fp} FN=#{stats.fn} TN=#{stats.tn} weighted=#{weighted} penalty=#{penalty}"
)
