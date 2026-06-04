defmodule RinhaFraud.ReferencesLoader do
  @moduledoc """
  Carrega vetores 8D pre-processados (IVF + SVD 14->8) + parametros LDA.
  """

  require Logger

  def load!(path \\ "/app/resources") do
    Logger.info("[ReferencesLoader] Carregando #{path}")
    t0 = System.monotonic_time(:millisecond)

    vectors_bin = File.read!("#{path}/vectors_8d_sorted.bin")
    labels_bin = File.read!("#{path}/labels_sorted.bin")
    centroids_bin = File.read!("#{path}/centroids.bin")
    bucket_starts_bin = File.read!("#{path}/bucket_starts.bin")
    svd_matrix = File.read!("#{path}/svd_matrix.bin")
    lda_w = File.read!("#{path}/lda_w.bin")
    lda_w0 = File.read!("#{path}/lda_w0.bin")
    fraud_centroid = File.read!("#{path}/fraud_centroid.bin")
    legit_centroid = File.read!("#{path}/legit_centroid.bin")
    cov_inv = File.read!("#{path}/cov_inv.bin")
    cart_tree = File.read!("#{path}/cart_tree.bin")
    count = byte_size(labels_bin)

    RinhaFraud.VectorStore.set_data(vectors_bin, labels_bin, count, centroids_bin, bucket_starts_bin, svd_matrix, lda_w, lda_w0, fraud_centroid, legit_centroid, cov_inv, cart_tree)

    elapsed = System.monotonic_time(:millisecond) - t0
    Logger.info("[ReferencesLoader] #{count} vetores 8D (IVF) em #{elapsed}ms")
    :ok
  end
end
