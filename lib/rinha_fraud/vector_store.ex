defmodule RinhaFraud.VectorStore do
  @moduledoc """
  Vetores 8D (SVD 14->8) com IVF index. KNN via NIF AVX.
  """

  @dims 8

  def child_spec(_opts) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, []},
      type: :worker,
      restart: :permanent,
      shutdown: 500
    }
  end

  def start_link do
    :ets.new(:vector_store, [:named_table, :public, {:read_concurrency, true}])
    Agent.start_link(fn -> :ok end, name: __MODULE__)
  end

  def set_data(vectors_bin, labels_bin, count, centroids_bin, bucket_starts_bin, svd_matrix, fraud_centroid, legit_centroid, cov_inv) do
    :ets.insert(:vector_store, {:vectors, vectors_bin})
    :ets.insert(:vector_store, {:labels, labels_bin})
    :ets.insert(:vector_store, {:count, count})
    :ets.insert(:vector_store, {:centroids, centroids_bin})
    :ets.insert(:vector_store, {:bucket_starts, bucket_starts_bin})
    :ets.insert(:vector_store, {:svd_matrix, svd_matrix})
    :ets.insert(:vector_store, {:fraud_centroid, fraud_centroid})
    :ets.insert(:vector_store, {:legit_centroid, legit_centroid})
    :ets.insert(:vector_store, {:cov_inv, cov_inv})
  end

  def size do
    case :ets.lookup(:vector_store, :count) do
      [{:count, n}] -> n
      [] -> 0
    end
  end

  def knn(query_vec, k, nprobe \\ 10) do
    [{:vectors, vectors_bin}] = :ets.lookup(:vector_store, :vectors)
    [{:labels, labels_bin}] = :ets.lookup(:vector_store, :labels)
    [{:centroids, centroids_bin}] = :ets.lookup(:vector_store, :centroids)
    [{:bucket_starts, bucket_starts_bin}] = :ets.lookup(:vector_store, :bucket_starts)
    [{:svd_matrix, svd_matrix}] = :ets.lookup(:vector_store, :svd_matrix)
    [{:count, count}] = :ets.lookup(:vector_store, :count)

    n_clusters = byte_size(centroids_bin) |> div(@dims * 4)

    query_14d = floats_to_binary(query_vec)
    query_8d = RinhaFraud.KnnNif.project_svd(query_14d, svd_matrix)

    RinhaFraud.KnnNif.knn_search_ivf(vectors_bin, labels_bin, centroids_bin, bucket_starts_bin, query_8d, k, nprobe, n_clusters)
    |> Enum.sort()
  end

  defp floats_to_binary(floats) do
    for f <- floats, into: <<>>, do: <<f::float-little-32>>
  end
end
