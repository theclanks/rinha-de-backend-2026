defmodule RinhaFraud.Vectorizer do
  @moduledoc """
  Transforma payload de transacao nos 14 vetores normalizados
  conforme DETECTION_RULES.md.
  """

  @defaults %{
    "max_amount" => 10_000,
    "max_installments" => 12,
    "amount_vs_avg_ratio" => 10,
    "max_minutes" => 1_440,
    "max_km" => 1_000,
    "max_tx_count_24h" => 20,
    "max_merchant_avg_amount" => 10_000
  }

  def load_consts(path \\ default_resource_path("normalization.json")) do
    case File.read(path) do
      {:ok, raw} -> Map.merge(@defaults, Jason.decode!(raw))
      _ -> @defaults
    end
  end

  def load_mcc_risk(path \\ default_resource_path("mcc_risk.json")) do
    case File.read(path) do
      {:ok, raw} -> Jason.decode!(raw)
      _ -> %{}
    end
  end

  @doc """
  Vetoriza payload decodificado (map com chaves string).
  Retorna lista de 14 floats.
  """
  def vectorize(payload, consts, mcc_risk) do
    tx = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last_tx = payload["last_transaction"]

    {:ok, requested_at, _} = DateTime.from_iso8601(tx["requested_at"])
    hour = requested_at.hour
    dow = Date.day_of_week(DateTime.to_date(requested_at)) - 1

    [
      clamp(tx["amount"] / consts["max_amount"]),
      clamp(tx["installments"] / consts["max_installments"]),
      clamp(tx["amount"] / safe_div(customer["avg_amount"]) / consts["amount_vs_avg_ratio"]),
      hour / 23.0,
      dow / 6.0,
      minutes_since_last(last_tx, requested_at, consts["max_minutes"]),
      km_from_last(last_tx, consts["max_km"]),
      clamp(terminal["km_from_home"] / consts["max_km"]),
      clamp(customer["tx_count_24h"] / consts["max_tx_count_24h"]),
      bool_int(terminal["is_online"]),
      bool_int(terminal["card_present"]),
      unknown_merchant(merchant["id"], customer["known_merchants"]),
      Map.get(mcc_risk, merchant["mcc"], 0.5),
      clamp(merchant["avg_amount"] / consts["max_merchant_avg_amount"])
    ]
  end

  defp clamp(x) when x < 0.0, do: 0.0
  defp clamp(x) when x > 1.0, do: 1.0
  defp clamp(x), do: x * 1.0

  defp bool_int(true), do: 1
  defp bool_int(false), do: 0
  defp bool_int(1), do: 1
  defp bool_int(_), do: 0

  defp safe_div(nil), do: 1.0
  defp safe_div(0), do: 1.0
  defp safe_div(0.0), do: 1.0
  defp safe_div(v), do: v

  defp unknown_merchant(merchant_id, known) when is_list(known) do
    if merchant_id in known, do: 0, else: 1
  end

  defp unknown_merchant(_, _), do: 1

  defp minutes_since_last(nil, _requested_at, _max), do: -1.0

  defp minutes_since_last(last_tx, requested_at, max_minutes) do
    {:ok, last_dt, _} = DateTime.from_iso8601(last_tx["timestamp"])
    diff_seconds = DateTime.diff(requested_at, last_dt)
    minutes = diff_seconds / 60.0
    clamp(minutes / max_minutes)
  end

  defp km_from_last(nil, _max), do: -1.0

  defp km_from_last(last_tx, max_km) do
    clamp(last_tx["km_from_current"] / max_km)
  end

  defp default_resource_path(file) do
    app_path = "/app/resources/#{file}"
    if File.exists?(app_path), do: app_path, else: "resources/#{file}"
  end
end
