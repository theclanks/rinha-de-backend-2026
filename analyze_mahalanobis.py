import gzip
import json
import numpy as np
from scipy.spatial.distance import mahalanobis
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "resources/references.json.gz"
print(f"Loading {path}...")
data = json.loads(gzip.open(path, "rt").read())

vectors = np.array([r["vector"] for r in data], dtype=np.float32)
labels = np.array([1 if r["label"] == "fraud" else 0 for r in data], dtype=np.uint8)

# Use SVD to project to 8D first
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=8, random_state=42)
vectors_8d = svd.fit_transform(vectors)

# Compute centroids and covariance in 8D space
fraud_mask = labels == 1
legit_mask = labels == 0

fraud_centroid = vectors_8d[fraud_mask].mean(axis=0)
legit_centroid = vectors_8d[legit_mask].mean(axis=0)

fraud_cov = np.cov(vectors_8d[fraud_mask][:100000].T)
legit_cov = np.cov(vectors_8d[legit_mask][:100000].T)
pooled_cov = (fraud_cov + legit_cov) / 2
pooled_cov += np.eye(8) * 0.01
pooled_inv = np.linalg.inv(pooled_cov)

print(f"Fraud centroid: {fraud_centroid}")
print(f"Legit centroid: {legit_centroid}")

# Save
with open("resources/fraud_centroid.bin", "wb") as f:
    f.write(fraud_centroid.astype(np.float32).tobytes())
with open("resources/legit_centroid.bin", "wb") as f:
    f.write(legit_centroid.astype(np.float32).tobytes())
with open("resources/cov_inv.bin", "wb") as f:
    f.write(pooled_inv.astype(np.float32).tobytes())

# Test Mahalanobis classifier on borderline
print(f"\n=== MAHALANOBIS CLASSIFIER ANALYSIS ===")

fast_total = 0
fast_correct = 0
mahal_needed = 0
mahal_correct = 0
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
        # Borderline - try Mahalanobis in 8D
        mahal_needed += 1
        vec_8d = svd.transform(vec.reshape(1, -1))[0]
        d_fraud = mahalanobis(vec_8d, fraud_centroid, pooled_inv)
        d_legit = mahalanobis(vec_8d, legit_centroid, pooled_inv)
        ratio = d_legit / (d_fraud + d_legit + 1e-10)

        if ratio > 0.70:
            # Predict fraud
            if label == 1:
                mahal_correct += 1
        elif ratio < 0.30:
            # Predict legit
            if label == 0:
                mahal_correct += 1
        else:
            # Still need KNN
            knn_needed += 1

    if (i + 1) % 500_000 == 0:
        print(f"  Processed {i + 1:,}/{len(data):,}")

total = len(data)
fast_rate = fast_total / total * 100
mahal_rate = mahal_needed / total * 100
knn_rate = knn_needed / total * 100
mahal_accuracy = mahal_correct / max(1, mahal_needed) * 100

print(f"\n=== THREE-TIER ANALYSIS (Mahalanobis) ===")
print(f"Fast path:       {fast_total:,} ({fast_rate:.1f}%) - correct: {fast_correct}/{fast_total}")
print(f"Mahalanobis:     {mahal_needed:,} ({mahal_rate:.1f}%) - correct: {mahal_correct}/{mahal_needed} ({mahal_accuracy:.1f}%)")
print(f"KNN needed:      {knn_needed:,} ({knn_rate:.1f}%)")
print(f"Total fast+mahal:{fast_total + mahal_correct:,} ({(fast_total + mahal_correct)/total*100:.1f}%)")
