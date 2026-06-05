defmodule RinhaFraud.RandomForest do
  @moduledoc false

  defstruct [:n_features, :threshold, :trees]

  @type rf_node :: {integer(), float(), integer(), integer(), float()}
  @type tree :: tuple()
  @type t :: %__MODULE__{
          n_features: pos_integer(),
          threshold: float(),
          trees: tuple()
        }

  def load(bin) when is_binary(bin) do
    case parse(bin) do
      {:ok, forest} -> forest
      {:error, reason} -> raise ArgumentError, "invalid random forest model: #{reason}"
    end
  end

  def parse(<<n_trees::little-signed-32, n_features::little-signed-32, rest::binary>>)
      when n_trees > 0 and n_features > 0 do
    with {:ok, trees, <<threshold::float-little-32>>} <- parse_trees(rest, n_trees, []) do
      {:ok,
       %__MODULE__{n_features: n_features, threshold: threshold, trees: List.to_tuple(trees)}}
    else
      {:ok, _trees, leftover} ->
        {:error, "expected 4 trailing threshold bytes, got #{byte_size(leftover)}"}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def parse(_), do: {:error, "missing header"}

  def predict_proba(%__MODULE__{n_features: n_features, trees: trees}, features)
      when is_list(features) and length(features) >= n_features do
    feature_tuple = features |> Enum.take(n_features) |> List.to_tuple()
    n_trees = tuple_size(trees)

    sum =
      for idx <- 0..(n_trees - 1), reduce: 0.0 do
        acc -> acc + predict_tree(elem(trees, idx), feature_tuple)
      end

    sum / n_trees
  end

  def classify(%__MODULE__{threshold: threshold} = forest, features) do
    prob = predict_proba(forest, features)
    if prob >= threshold, do: 1.0, else: 0.0
  end

  defp parse_trees(rest, 0, acc), do: {:ok, Enum.reverse(acc), rest}

  defp parse_trees(<<n_nodes::little-signed-32, rest::binary>>, remaining, acc)
       when n_nodes > 0 do
    case parse_nodes(rest, n_nodes, []) do
      {:ok, nodes, rest} -> parse_trees(rest, remaining - 1, [List.to_tuple(nodes) | acc])
      {:error, reason} -> {:error, reason}
    end
  end

  defp parse_trees(_, _remaining, _acc), do: {:error, "truncated tree header"}

  defp parse_nodes(rest, 0, acc), do: {:ok, Enum.reverse(acc), rest}

  defp parse_nodes(
         <<feature::little-signed-32, threshold::float-little-32, left::little-signed-32,
           right::little-signed-32, fraud_prob::float-little-32, rest::binary>>,
         remaining,
         acc
       ) do
    parse_nodes(rest, remaining - 1, [{feature, threshold, left, right, fraud_prob} | acc])
  end

  defp parse_nodes(_, _remaining, _acc), do: {:error, "truncated node"}

  defp predict_tree(tree, features), do: walk(tree, features, 0)

  defp walk(tree, features, idx) when idx >= 0 and idx < tuple_size(tree) do
    {feature, threshold, left, right, fraud_prob} = elem(tree, idx)

    if feature == -1 do
      fraud_prob
    else
      next = if elem(features, feature) <= threshold, do: left, else: right
      walk(tree, features, next)
    end
  end
end
