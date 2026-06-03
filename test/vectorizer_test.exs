defmodule RinhaFraud.VectorStoreTest do
  use ExUnit.Case, async: false

  setup do
    case Process.whereis(RinhaFraud.VectorStore) do
      nil ->
        {:ok, pid} = RinhaFraud.VectorStore.start_link()
        {:ok, pid: pid}

      pid ->
        :ok
        {:ok, pid: pid}
    end
  end

  test "encontra vizinho exato quando indexado" do
    # Create 2 clusters with 50 vectors each
    entries = for i <- 1..100 do
      vec = if i <= 50 do
        for j <- 1..8, do: 0.2 + j / 100.0
      else
        for j <- 1..8, do: 0.7 + j / 100.0
      end
      label = if rem(i, 2) == 0, do: :fraud, else: :legit
      %{vector: vec, label: label}
    end

    {v_bin, l_bin, count} =
      Enum.reduce(entries, {<<>>, <<>>, 0}, fn e, {v_acc, l_acc, c} ->
        vb = for f <- e.vector, into: <<>>, do: <<f::float-little-32>>
        lb = if e.label == :fraud, do: <<1>>, else: <<0>>
        {v_acc <> vb, l_acc <> lb, c + 1}
      end)

    # 2 centroids
    c1 = for _ <- 1..8, do: 0.25
    c2 = for _ <- 1..8, do: 0.75
    centroids_bin = (for f <- c1 ++ c2, into: <<>>, do: <<f::float-little-32>>)

    # Bucket starts: cluster 0 at 0, cluster 1 at 50, end at 100
    bs_bin = <<0::little-32, 50::little-32, 100::little-32>>

    svd_matrix = <<0.0::float-little-32>>
    RinhaFraud.VectorStore.set_data(v_bin, l_bin, count, centroids_bin, bs_bin, svd_matrix, <<>>, <<>>, <<>>)

    # Query close to cluster 0
    query = for j <- 1..8, do: 0.2 + j / 100.0
    q_bin = for f <- query, into: <<>>, do: <<f::float-little-32>>

    result = RinhaFraud.KnnNif.knn_search_ivf(v_bin, l_bin, centroids_bin, bs_bin, q_bin, 5, 2, 2)
    |> Enum.sort()

    assert length(result) > 0
    {dist, _label} = List.first(result)
    assert dist < 0.01
  end

  test "fraud_score calculado corretamente" do
    base = List.duplicate(0.5, 8)
    entries = [
      %{vector: perturb(base, 0.001), label: :fraud},
      %{vector: perturb(base, 0.002), label: :fraud},
      %{vector: perturb(base, 0.003), label: :fraud},
      %{vector: perturb(base, 0.004), label: :legit},
      %{vector: perturb(base, 0.005), label: :legit}
    ]

    {v_bin, l_bin, count} =
      Enum.reduce(entries, {<<>>, <<>>, 0}, fn e, {v_acc, l_acc, c} ->
        vb = for f <- e.vector, into: <<>>, do: <<f::float-little-32>>
        lb = if e.label == :fraud, do: <<1>>, else: <<0>>
        {v_acc <> vb, l_acc <> lb, c + 1}
      end)

    centroids_bin = <<0.5::float-little-32>>
    bs_bin = <<0::little-32, 5::little-32>>
    svd_matrix = <<0.0::float-little-32>>
    RinhaFraud.VectorStore.set_data(v_bin, l_bin, count, centroids_bin, bs_bin, svd_matrix, <<>>, <<>>, <<>>)

    q_bin = for f <- base, into: <<>>, do: <<f::float-little-32>>

    neighbors = RinhaFraud.KnnNif.knn_search_ivf(v_bin, l_bin, centroids_bin, bs_bin, q_bin, 5, 1, 1)
    |> Enum.sort()

    fraud_count = Enum.count(neighbors, fn {_, label} -> label == 1 end)
    score = fraud_count / 5.0

    assert score >= 0.6
  end

  defp perturb(vec, delta) do
    Enum.map(vec, fn v -> v + delta * :rand.uniform() end)
  end
end
