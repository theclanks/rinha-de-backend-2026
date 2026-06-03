import gzip
import json
import sys

def compute_risk(vec):
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

    return risk

path = sys.argv[1] if len(sys.argv) > 1 else "resources/references.json.gz"
data = json.loads(gzip.open(path, "rt").read())

# Analyze false positives in legit_fast
print("=== FALSE POSITIVES (classified as legit_fast but actually fraud) ===")
fp_count = 0
for e in data:
    vec = e["vector"]
    label = e["label"]
    risk = compute_risk(vec)

    amount, installments, amount_vs_avg, hour, dow, \
    min_since_last, km_from_last, km_from_home, tx_count, \
    is_online, card_present, unknown_merchant, mcc_risk, merchant_avg = vec

    if risk < 0.15 and card_present == 1 and unknown_merchant == 0 and amount < 0.3:
        if label == "fraud":
            fp_count += 1
            if fp_count <= 20:
                print(f"  risk={risk:.2f} amt={amount:.3f} inst={installments:.3f} avg={amount_vs_avg:.3f} "
                      f"home={km_from_home:.3f} mcc={mcc_risk:.2f} tx={tx_count:.3f} "
                      f"min={min_since_last:.1f} card={card_present} unknown={unknown_merchant}")

print(f"Total false positives: {fp_count:,} ({fp_count/len(data)*100:.2f}%)")

# Analyze what features are most discriminative for borderline cases
print(f"\n=== BORDERLINE ANALYSIS (risk 0.15-0.70) ===")
borderline = []
for e in data:
    risk = compute_risk(e["vector"])
    if 0.15 <= risk <= 0.70:
        borderline.append((e["vector"], e["label"], risk))

print(f"Borderline entries: {len(borderline):,}")

# Check fraud rate in borderline
fraud_in_borderline = sum(1 for _, l, _ in borderline if l == "fraud")
print(f"  Fraud in borderline: {fraud_in_borderline:,} ({fraud_in_borderline/len(borderline)*100:.1f}%)")
print(f"  Legit in borderline: {len(borderline) - fraud_in_borderline:,}")

# Try different thresholds to optimize
print(f"\n=== THRESHOLD OPTIMIZATION ===")
for legit_thresh in [0.10, 0.12, 0.15, 0.18, 0.20]:
    for fraud_thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        fast = 0
        fp = 0
        for vec, label, risk in borderline:
            amount, installments, amount_vs_avg, hour, dow, \
            min_since_last, km_from_last, km_from_home, tx_count, \
            is_online, card_present, unknown_merchant, mcc_risk, merchant_avg = vec

            if risk < legit_thresh and card_present == 1 and unknown_merchant == 0 and amount < 0.3:
                fast += 1
                if label == "fraud":
                    fp += 1
            elif risk > fraud_thresh:
                fast += 1

        total_fast = fast + sum(1 for vec, label, risk in [
            (v, l, r) for v, l, r in borderline
            if not (r < legit_thresh and vec[10] == 1 and vec[11] == 0 and vec[0] < 0.3)
            and not r > fraud_thresh
        ])

        hit_rate = (fast + 1243983 + 244617) / len(data) * 100
        print(f"  legit<{legit_thresh} fraud>{fraud_thresh}: fast={fast+1243983+244617:,} ({hit_rate:.1f}%) fp={fp}")
