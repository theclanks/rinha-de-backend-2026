defmodule RinhaFraud.ReferencesLoader do
  @moduledoc """
  Carrega vetores 8D pre-processados (IVF + SVD 14->8) + parametros LDA.
  """

  require Logger

  def load!(path \\ default_path()) do
    Logger.info("[ReferencesLoader] Carregando #{path}")
    t0 = System.monotonic_time(:millisecond)

    rf_forest = File.read!("#{path}/rf_forest.bin")

    RinhaFraud.VectorStore.set_data(rf_forest)

    with {:ok, vectors} <- File.read("#{path}/vectors_14d_i16_sorted.bin"),
         {:ok, labels} <- File.read("#{path}/labels_14d_sorted.bin"),
         {:ok, centroids} <- File.read("#{path}/centroids_14d_i16.bin"),
         {:ok, bucket_starts} <- File.read("#{path}/bucket_starts_14d.bin") do
      RinhaFraud.VectorStore.set_ivf14(vectors, labels, centroids, bucket_starts)
      Logger.info("[ReferencesLoader] IVF 14D carregado com #{byte_size(labels)} vetores")
    else
      _ -> Logger.info("[ReferencesLoader] IVF 14D ausente; usando RF-only")
    end

    elapsed = System.monotonic_time(:millisecond) - t0
    Logger.info("[ReferencesLoader] RF forest carregado em #{elapsed}ms")
    :ok
  end

  defp default_path do
    if File.dir?("/app/resources"), do: "/app/resources", else: "resources"
  end
end
