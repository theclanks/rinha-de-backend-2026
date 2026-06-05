defmodule RinhaFraud.KnnNif do
  @moduledoc false

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:rinha_fraud), "knn")
    :erlang.load_nif(path, 0)
  end

  def knn_search_ivf(
        _vectors_bin,
        _labels_bin,
        _centroids_bin,
        _bucket_starts_bin,
        _query_bin,
        _k,
        _nprobe,
        _n_clusters
      ) do
    exit("NIF knn_search_ivf/8 not implemented")
  end

  def knn_search_ivf_14(
        _vectors_bin,
        _labels_bin,
        _centroids_bin,
        _bucket_starts_bin,
        _query_bin,
        _k,
        _nprobe,
        _n_clusters
      ) do
    exit("NIF knn_search_ivf_14/8 not implemented")
  end

  def knn_search_ivf_14_i16(
        _vectors_bin,
        _labels_bin,
        _centroids_bin,
        _bucket_starts_bin,
        _query_bin,
        _k,
        _nprobe,
        _n_clusters
      ) do
    exit("NIF knn_search_ivf_14_i16/8 not implemented")
  end

  def project_svd(_query_bin, _matrix_bin) do
    exit("NIF project_svd/2 not implemented")
  end

  def mahalanobis_score(_query_bin, _cov_inv_bin, _fraud_bin, _legit_bin) do
    exit("NIF mahalanobis_score/4 not implemented")
  end

  def knn_brute_force(_vectors_bin, _labels_bin, _query_bin, _k) do
    exit("NIF knn_brute_force/4 not implemented")
  end

  def cart_predict(_query_bin, _tree_bin) do
    exit("NIF cart_predict/2 not implemented")
  end

  def cart_predict_14(_query_bin, _tree_bin) do
    exit("NIF cart_predict_14/2 not implemented")
  end

  def rf_predict(_query_bin, _forest_bin) do
    exit("NIF rf_predict/2 not implemented")
  end
end
