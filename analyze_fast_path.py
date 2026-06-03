import gzip
import json
import sys

def fast_path(vec):
    amount, installments, amount_vs_avg, hour, dow, \
    min_since_last, km_from_last, km_from_home, tx_count, \
    is_online, card_present, unknown_merchant, mcc_risk, merchant_avg = vec

    risk = 0.0
    risk += 0.15 if amount > 0.7 else 0.0
    risk += 0.10 if installments > 0.6 else 0.0
    risk += 0.15 if amount_vs_avg > 0.8 else 0.0
    risk += 0.15 if km_from_home > 0.7 else 0.0
    risk += 0.15 if unknown_merchant == 1 else 0.0
    risk += 0.10 if mcc_risk > 0.6 else 0.0
    risk += 0.10 if tx_count > 0.8 else 0.0
    risk += 0.10 if min_since_last == -1.0 and amount > 0.5 else 0.0

    if risk < 0.15 and card_present == 1 and unknown_merchant == 0 and amount < 0.3:
        return "legit_fast", 0.0
    elif risk > 0.70:
        return "fraud_fast", 1.0
    else:
        return "miss", None

path = sys.argv[1] if len(sys.argv) > 1 else "resources/references.json.gz"
print(f"Analyzing {path}...")

data = json.loads(gzip.open(path, "rt").read())
total = len(data)
print(f"Total entries: {total:,}")

stats = {
    "legit_fast": 0,
    "fraud_fast": 0,
    "miss": 0,
    "legit_miss": 0,
    "fraud_miss": 0,
}

risk_dist = []

for i, entry in enumerate(data):
    vec = entry["vector"]
    label = entry["label"]
    method, _ = fast_path(vec)

    stats[method] += 1
    if method == "miss":
        if label == "legit":
            stats["legit_miss"] += 1
        else:
            stats["fraud_miss"] += 1

    if (i + 1) % 500_000 == 0:
        print(f"  Processed {i + 1:,}/{total:,}")

fast_total = stats["legit_fast"] + stats["fraud_fast"]
hit_rate = fast_total / total * 100

print(f"\n=== FAST PATH ANALYSIS ===")
print(f"Total:           {total:,}")
print(f"Fast hit:        {fast_total:,} ({hit_rate:.1f}%)")
print(f"  Legit fast:    {stats['legit_fast']:,} ({stats['legit_fast']/total*100:.1f}%)")
print(f"  Fraud fast:    {stats['fraud_fast']:,} ({stats['fraud_fast']/total*100:.1f}%)")
print(f"KNN needed:      {stats['miss']:,} ({stats['miss']/total*100:.1f}%)")
print(f"  Legit miss:    {stats['legit_miss']:,}")
print(f"  Fraud miss:    {stats['fraud_miss']:,}")

# Analyze risk distribution for misses
print(f"\n=== RISK DISTRIBUTION (misses) ===")
print(f"  Borderline range (0.15-0.70): {stats['miss']:,} entries need KNN")

# Check accuracy of fast path decisions
print(f"\n=== FAST PATH CORRECTNESS ===")
# All legit_fast should be legit, all fraud_fast should be fraud
legit_fast_correct = sum(1 for e in data if fast_path(e["vector"])[0] == "legit_fast" and e["label"] == "legit")
fraud_fast_correct = sum(1 for e in data if fast_path(e["vector"])[0] == "fraud_fast" and e["label"] == "fraud")

print(f"  Legit fast correct: {legit_fast_correct:,}/{stats['legit_fast']:,} ({legit_fast_correct/max(1,stats['legit_fast'])*100:.1f}%)")
print(f"  Fraud fast correct: {fraud_fast_correct:,}/{stats['fraud_fast']:,} ({fraud_fast_correct/max(1,stats['fraud_fast'])*100:.1f}%)")
