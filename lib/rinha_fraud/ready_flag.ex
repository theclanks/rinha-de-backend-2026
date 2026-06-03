defmodule RinhaFraud.ReadyFlag do
  @moduledoc """
  Flag simples via ETS para indicar quando o indice esta pronto.
  """

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
    :ets.new(:ready_flag, [:named_table, :public, {:read_concurrency, true}])
    :ets.insert(:ready_flag, {:ready, false})
    Agent.start_link(fn -> :ok end, name: __MODULE__)
  end

  def ready?, do: :ets.lookup_element(:ready_flag, :ready, 2)
  def set_ready, do: :ets.insert(:ready_flag, {:ready, true})
end
