import gzip
import json
import numpy as np
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "resources/references.json.gz"
print(f"Loading {path}...")
data = json.loads(gzip.open(path, "rt").read())

vectors = np.array([r["vector"] for r in data], dtype=np.float32)
labels = np.array([1 if r["label"] == "fraud" else 0 for r in data], dtype=np.uint8)

# Compute centroids
fraud_mask = labels == 1
legit_mask = labels == 0

fraud_centroid = vectors[fraud_mask].mean(axis=0)
legit_centroid = vectors[legit_mask].mean(axis=0)

print(f"Fraud centroid: {fraud_centroid}")
print(f"Legit centroid: {legit_centroid}")

# Save centroids
with open("resources/fraud_centroid.bin", "wb") as f:
    f.write(fraud_centroid.astype(np.float32).tobytes())
with open("resources/legit_centroid.bin", "wb") as f:
    f.write(legit_centroid.astype(np.float32).tobytes())

# Test centroid-based classifier
def centroid_classify(vec, fraud_c, legit_c, threshold=0.5):
    dist_fraud = np.sqrt(np.sum((vec - fraud_c)**2))
    dist_legit = np.sqrt(np.sum((vec - legit_c)**2))
    ratio = dist_legit / (dist_fraud + dist_legit + 1e-10)
    return ratio

# Analyze centroid classifier on borderline cases
print(f"\n=== CENTROID CLASSIFIER ANALYSIS ===")

# Fast path stats
fast_correct = 0
fast_total = 0
centroid_needed = 0
centroid_correct = 0
knn_needed = 0

for i, (vec, label) in enumerate(zip(vectors, labels)):
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

    if risk < 0.10 and card_present == 1 and unknown_merchant == 0 and amount < 0.3:
        fast_total += 1
        if label == 0:
            fast_correct += 1
    elif risk > 0.60:
        fast_total += 1
        if label == 1:
            fast_correct += 1
    else:
        # Borderline - try centroid classifier
        centroid_needed += 1
        ratio = centroid_classify(vec, fraud_centroid, legit_centroid)

        if ratio > 0.65:
            # Predict fraud
            if label == 1:
                centroid_correct += 1
        elif ratio < 0.35:
            # Predict legit
            if label == 0:
                centroid_correct += 1
        else:
            # Still need KNN
            knn_needed += 1

    if (i + 1) % 500_000 == 0:
        print(f"  Processed {i + 1:,}/{len(data):,}")

total = len(data)
fast_rate = fast_total / total * 100
centroid_rate = centroid_needed / total * 100
knn_rate = knn_needed / total * 100
centroid_accuracy = centroid_correct / max(1, centroid_needed) * 100

print(f"\n=== THREE-TIER ANALYSIS ===")
print(f"Fast path:       {fast_total:,} ({fast_rate:.1f}%) - correct: {fast_correct}/{fast_total}")
print(f"Centroid:        {centroid_needed:,} ({centroid_rate:.1f}%) - correct: {centroid_correct}/{centroid_needed} ({centroid_accuracy:.1f}%)")
print(f"KNN needed:      {knn_needed:,} ({knn_rate:.1f}%)")
print(f"Total fast+cent: {fast_total + centroid_correct:,} ({(fast_total + centroid_correct)/total*100:.1f}%)")
