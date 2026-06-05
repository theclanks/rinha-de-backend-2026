defmodule RinhaFraud.ReferencesLoader do
  @moduledoc """
  Carrega vetores 8D pre-processados (IVF + SVD 14->8) + parametros LDA.
  """

  require Logger

  def load!(path \\ default_path()) do
    Logger.info("[ReferencesLoader] Carregando #{path}")
    t0 = System.monotonic_time(:millisecond)

    lda_w = File.read!("#{path}/lda_w_14d.bin")
    lda_w0 = File.read!("#{path}/lda_w0_14d.bin")
    cart_tree = File.read!("#{path}/cart_tree_14d.bin")

    RinhaFraud.VectorStore.set_lda_cart14(lda_w, lda_w0, cart_tree)

    elapsed = System.monotonic_time(:millisecond) - t0
    Logger.info("[ReferencesLoader] LDA/CART 14D carregado em #{elapsed}ms")
    :ok
  end

  defp default_path do
    if File.dir?("/app/resources"), do: "/app/resources", else: "resources"
  end
end
