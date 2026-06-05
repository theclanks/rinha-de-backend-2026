import argparse
import json
from datetime import datetime

import numpy as np

from evaluate_rf import load_forest, predict_proba
from evaluate_twostage_ivf import ivf_knn_score, load_ivf


DEFAULTS = {
    "max_amount": 10000,
    "max_installments": 12,
    "amount_vs_avg_ratio": 10,
    "max_minutes": 1440,
    "max_km": 1000,
    "max_tx_count_24h": 20,
    "max_merchant_avg_amount": 10000,
}


def clamp(x):
    return max(0.0, min(1.0, float(x)))


def parse_dt(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def vectorize(payload, consts, mcc_risk):
    tx = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last_tx = payload.get("last_transaction")

    requested_at = parse_dt(tx["requested_at"])
    dow = requested_at.weekday()

    avg = customer.get("avg_amount") or 1.0
    if avg == 0:
        avg = 1.0

    if last_tx is None:
        minutes_since_last = -1.0
        km_from_last = -1.0
    else:
        last_dt = parse_dt(last_tx["timestamp"])
        minutes_since_last = clamp(((requested_at - last_dt).total_seconds() / 60.0) / consts["max_minutes"])
        km_from_last = clamp(last_tx["km_from_current"] / consts["max_km"])

    return [
        clamp(tx["amount"] / consts["max_amount"]),
        clamp(tx["installments"] / consts["max_installments"]),
        clamp((tx["amount"] / avg) / consts["amount_vs_avg_ratio"]),
        requested_at.hour / 23.0,
        dow / 6.0,
        minutes_since_last,
        km_from_last,
        clamp(terminal["km_from_home"] / consts["max_km"]),
        clamp(customer["tx_count_24h"] / consts["max_tx_count_24h"]),
        1 if terminal["is_online"] else 0,
        1 if terminal["card_present"] else 0,
        0 if merchant["id"] in customer.get("known_merchants", []) else 1,
        mcc_risk.get(merchant["mcc"], 0.5),
        clamp(merchant["avg_amount"] / consts["max_merchant_avg_amount"]),
    ]


def add(stats, pred_fraud, expected_fraud):
    if pred_fraud and expected_fraud:
        stats["tp"] += 1
    elif pred_fraud:
        stats["fp"] += 1
    elif expected_fraud:
        stats["fn"] += 1
    else:
        stats["tn"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", default="test/test-data.json")
    parser.add_argument("--resources", default="resources")
    parser.add_argument("--forest", default="resources/rf_forest.bin")
    parser.add_argument("--threshold", type=float, default=0.513)
    parser.add_argument("--band", type=float, default=0.20)
    parser.add_argument("--nprobe", type=int, default=24)
    args = parser.parse_args()

    consts = DEFAULTS | json.load(open(f"{args.resources}/normalization.json"))
    mcc_risk = json.load(open(f"{args.resources}/mcc_risk.json"))
    payloads = json.load(open(args.payloads))["entries"]
    forest = load_forest(args.forest)
    ivf = load_ivf(args.resources)

    rf_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    two_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    fallback = 0

    for entry in payloads:
        vec = vectorize(entry["request"], consts, mcc_risk)
        expected_fraud = not entry["expected_approved"]
        prob = predict_proba(forest, vec)
        rf_fraud = prob >= args.threshold
        add(rf_stats, rf_fraud, expected_fraud)

        if abs(prob - args.threshold) <= args.band:
            fallback += 1
            score = ivf_knn_score(np.asarray(vec, dtype=np.float32), ivf, args.nprobe)
            two_fraud = score >= 0.6
        else:
            two_fraud = rf_fraud

        add(two_stats, two_fraud, expected_fraud)

    total = len(payloads)
    print(f"total={total:,} fallback={fallback:,} ({fallback / total * 100:.2f}%)")
    for name, stats in [("rf", rf_stats), ("two_stage", two_stats)]:
        weighted = stats["fp"] + 3 * stats["fn"]
        penalty = stats["fp"] * 25 + stats["fn"] * 90
        acc = (stats["tp"] + stats["tn"]) / total
        print(
            f"{name} acc={acc:.5f} TP={stats['tp']:,} FP={stats['fp']:,} "
            f"FN={stats['fn']:,} TN={stats['tn']:,} weighted={weighted:,} penalty={penalty:,}"
        )


if __name__ == "__main__":
    main()
