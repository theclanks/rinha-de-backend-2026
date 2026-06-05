import gzip
import json
from datetime import datetime

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


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
        requested_at.weekday() / 6.0,
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


def report(name, expected, pred):
    tn, fp, fn, tp = confusion_matrix(expected, pred, labels=[0, 1]).ravel()
    weighted = fp + 3 * fn
    penalty = fp * 25 + fn * 90
    acc = (tp + tn) / len(expected)
    print(f"{name}: acc={acc:.5f} TP={tp} FP={fp} FN={fn} TN={tn} weighted={weighted} penalty={penalty}")


print("loading references...")
refs = json.load(gzip.open("resources/references.json.gz", "rt"))
X = np.asarray([r["vector"] for r in refs], dtype=np.float32)
y = np.asarray([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.int8)
del refs

print("loading payloads...")
consts = DEFAULTS | json.load(open("resources/normalization.json"))
mcc_risk = json.load(open("resources/mcc_risk.json"))
entries = json.load(open("test/test-data.json"))["entries"]
Xp = np.asarray([vectorize(e["request"], consts, mcc_risk) for e in entries], dtype=np.float32)
yp = np.asarray([0 if e["expected_approved"] else 1 for e in entries], dtype=np.int8)

print("training LDA 14D...")
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
scores = lda.decision_function(Xp)
for threshold in [10, 5, 3, 2, 1, 0.5, 0]:
    report(f"lda14 threshold={threshold}", yp, (scores >= threshold).astype(np.int8))

print("training CART 14D...")
for depth in [8, 10, 12, 15, 20]:
    for class_weight in [None, "balanced"]:
        cart = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=42,
        )
        cart.fit(X, y)
        proba = cart.predict_proba(Xp)[:, 1]
        label = "balanced" if class_weight else "default"
        for threshold in [0.30, 0.40, 0.50, 0.60]:
            report(f"cart14 d={depth} {label} threshold={threshold}", yp, (proba >= threshold).astype(np.int8))

print("hybrid LDA14 + CART14...")
cart = DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42)
cart.fit(X, y)
cart_pred = (cart.predict_proba(Xp)[:, 1] >= 0.5).astype(np.int8)
for threshold in [10, 5, 3, 2]:
    pred = np.where(scores > threshold, 1, np.where(scores < -threshold, 0, cart_pred))
    fallback = np.mean(np.abs(scores) <= threshold) * 100
    report(f"hybrid14 lda={threshold} fallback={fallback:.2f}%", yp, pred.astype(np.int8))
