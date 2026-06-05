defmodule RinhaFraud.VectorStore do
  @moduledoc false

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

  def set_data(rf_forest) when is_binary(rf_forest) do
    set_data(RinhaFraud.RandomForest.load(rf_forest))
  end

  def set_data(%RinhaFraud.RandomForest{} = rf_forest) do
    :ets.insert(:vector_store, {:rf_forest, rf_forest})
  end

  def set_lda_cart14(lda_w, lda_w0, cart_tree) do
    :ets.insert(:vector_store, [
      {:lda_w_14d, lda_w},
      {:lda_w0_14d, lda_w0},
      {:cart_tree_14d, cart_tree}
    ])
  end

  def lda_cart14 do
    [{:lda_w_14d, lda_w}] = :ets.lookup(:vector_store, :lda_w_14d)
    [{:lda_w0_14d, lda_w0}] = :ets.lookup(:vector_store, :lda_w0_14d)
    [{:cart_tree_14d, cart_tree}] = :ets.lookup(:vector_store, :cart_tree_14d)
    {lda_w, lda_w0, cart_tree}
  end

  def set_ivf14(vectors, labels, centroids, bucket_starts) do
    n_clusters = div(byte_size(centroids), 14 * 2)

    :ets.insert(:vector_store, [
      {:ivf14_vectors, vectors},
      {:ivf14_labels, labels},
      {:ivf14_centroids, centroids},
      {:ivf14_bucket_starts, bucket_starts},
      {:ivf14_n_clusters, n_clusters}
    ])
  end

  def ivf14_ready? do
    :ets.member(:vector_store, :ivf14_vectors)
  end

  def knn14(query_vec, k \\ 5, nprobe \\ 24) do
    [{:ivf14_vectors, vectors}] = :ets.lookup(:vector_store, :ivf14_vectors)
    [{:ivf14_labels, labels}] = :ets.lookup(:vector_store, :ivf14_labels)
    [{:ivf14_centroids, centroids}] = :ets.lookup(:vector_store, :ivf14_centroids)
    [{:ivf14_bucket_starts, bucket_starts}] = :ets.lookup(:vector_store, :ivf14_bucket_starts)
    [{:ivf14_n_clusters, n_clusters}] = :ets.lookup(:vector_store, :ivf14_n_clusters)

    query_bin = for f <- query_vec, into: <<>>, do: <<quantize_i16(f)::little-signed-16>>

    RinhaFraud.KnnNif.knn_search_ivf_14_i16(
      vectors,
      labels,
      centroids,
      bucket_starts,
      query_bin,
      k,
      nprobe,
      n_clusters
    )
    |> Enum.sort()
  end

  defp quantize_i16(value) do
    value
    |> Kernel.*(32767)
    |> round()
    |> max(-32767)
    |> min(32767)
  end

  def set_data(
        vectors,
        labels,
        count,
        centroids,
        bucket_starts,
        svd_matrix,
        lda_w,
        lda_w0,
        cov_inv
      ) do
    :ets.insert(:vector_store, [
      {:vectors, vectors},
      {:labels, labels},
      {:count, count},
      {:centroids, centroids},
      {:bucket_starts, bucket_starts},
      {:svd_matrix, svd_matrix},
      {:lda_w, lda_w},
      {:lda_w0, lda_w0},
      {:cov_inv, cov_inv}
    ])
  end

  def rf_forest do
    [{:rf_forest, forest}] = :ets.lookup(:vector_store, :rf_forest)
    forest
  end
end
