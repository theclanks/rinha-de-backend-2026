defmodule RinhaFraud.LSHIndex do
  @moduledoc """
  Indice LSH E2LSH com L=8 tabelas, K=4 funcoes por tabela, W=4.0
  para busca aproximada de vizinhos mais proximos (distancia euclidiana).

  ETS tables: :lsh_projections, :lsh_buckets (bag), :lsh_entries
  """

  use GenServer

  @dims 14
  @l 8
  @k 4
  @w 4.0

  def start_link(opts \\ []), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  def knn(query_vec, k \\ 5), do: GenServer.call(__MODULE__, {:knn, query_vec, k}, 10_000)

  def size, do: :ets.info(:lsh_entries, :size)

  @impl true
  def init(_opts) do
    :ets.new(:lsh_projections, [:named_table, :public, {:read_concurrency, true}])
    :ets.new(:lsh_buckets, [:named_table, :public, :bag, {:read_concurrency, true}])
    :ets.new(:lsh_entries, [:named_table, :public, {:read_concurrency, true}])

    projections = generate_projections()
    store_projections(projections)

    {:ok, %{projections: projections, indexed: 0}}
  end

  @impl true
  def handle_call({:knn, query_vec, k}, _from, state) do
    {:reply, do_knn(query_vec, k, state.projections), state}
  end

  def index_sync(entries) do
    projections = load_projections()
    index_entries(entries, projections)
    :ok
  end

  defp generate_projections do
    for _table <- 1..@l do
      for _func <- 1..@k do
        a = for _ <- 1..@dims, do: :rand.normal()
        b = :rand.uniform() * @w
        {a, b}
      end
    end
  end

  defp store_projections(projections) do
    projections
    |> Enum.with_index()
    |> Enum.each(fn {table_funcs, idx} ->
      :ets.insert(:lsh_projections, {idx, table_funcs})
    end)
  end

  defp load_projections do
    for idx <- 0..(@l - 1) do
      [{^idx, funcs}] = :ets.lookup(:lsh_projections, idx)
      funcs
    end
  end

  defp hash_vector(vec, table_funcs) do
    table_funcs
    |> Enum.map(fn {a, b} ->
      dot = dot_product(a, vec)
      floor((dot + b) / @w)
    end)
    |> List.to_tuple()
  end

  defp dot_product(a, b) do
    Enum.zip_reduce(a, b, 0.0, fn x, y, acc -> acc + x * y end)
  end

  defp index_entries(entries, projections) do
    Enum.each(entries, fn %{id: id, vector: vec, label: label} ->
      :ets.insert(:lsh_entries, {id, vec, label})
      Enum.with_index(projections, fn table_funcs, table_idx ->
        bucket_key = hash_vector(vec, table_funcs)
        :ets.insert(:lsh_buckets, {{table_idx, bucket_key}, id})
      end)
    end)
  end

  defp do_knn(query_vec, k, projections) do
    candidate_ids =
      projections
      |> Enum.with_index()
      |> Enum.flat_map(fn {table_funcs, table_idx} ->
        bucket_key = hash_vector(query_vec, table_funcs)
        :ets.lookup(:lsh_buckets, {table_idx, bucket_key})
        |> Enum.map(fn {_key, id} -> id end)
      end)
      |> MapSet.new()

    candidates =
      if MapSet.size(candidate_ids) < k * 3 do
        extra = sample_entries(500)
        MapSet.union(candidate_ids, MapSet.new(extra))
      else
        candidate_ids
      end

    candidates
    |> Enum.map(fn id ->
      case :ets.lookup(:lsh_entries, id) do
        [{^id, vec, label}] -> {euclidean_distance(query_vec, vec), label}
        [] -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.take(k)
  end

  defp euclidean_distance(v1, v2) do
    Enum.zip_reduce(v1, v2, 0.0, fn a, b, acc -> acc + (a - b) * (a - b) end)
    |> :math.sqrt()
  end

  defp sample_entries(n) do
    total = :ets.info(:lsh_entries, :size)
    if total <= n do
      :ets.select(:lsh_entries, [{{:"$1", :_, :_}, [], [:"$1"]}])
    else
      Stream.iterate(:ets.first(:lsh_entries), fn key ->
        case :ets.next(:lsh_entries, key) do
          :"$end_of_table" -> :ets.first(:lsh_entries)
          next -> next
        end
      end)
      |> Stream.take(n)
      |> Enum.to_list()
    end
  end
end
