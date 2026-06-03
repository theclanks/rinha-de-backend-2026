import gzip
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import struct
import sys
import time

input_path = sys.argv[1] if len(sys.argv) > 1 else "resources/references.json.gz"
output_dir = "resources"
n_components = 8
n_clusters = 4096

t0 = time.time()
print(f"Loading {input_path}...")
data = json.loads(gzip.open(input_path, "rt").read())
n = len(data)
print(f"Parsed {n} entries in {time.time()-t0:.1f}s")

# Extract vectors
vectors = np.array([r["vector"] for r in data], dtype=np.float32)
labels = np.array([1 if r["label"] == "fraud" else 0 for r in data], dtype=np.uint8)

print(f"Computing SVD {vectors.shape[1]}D -> {n_components}D...")
svd = TruncatedSVD(n_components=n_components, random_state=42)
vectors_8d = svd.fit_transform(vectors)
print(f"SVD done in {time.time()-t0:.1f}s, explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# K-means clustering for IVF
print(f"Computing k-means with {n_clusters} clusters...")
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000, n_init=10)
assignments = kmeans.fit_predict(vectors_8d)
centroids = kmeans.cluster_centers_.astype(np.float32)
print(f"K-means done in {time.time()-t0:.1f}s")

# Sort vectors by cluster assignment for cache locality
sort_idx = np.argsort(assignments, kind='mergesort')
vectors_sorted = vectors_8d[sort_idx]
labels_sorted = labels[sort_idx]
assignments_sorted = assignments[sort_idx]

# Build bucket starts
bucket_starts = np.zeros(n_clusters + 1, dtype=np.int32)
unique, counts = np.unique(assignments_sorted, return_counts=True)
bucket_starts[unique + 1] = np.cumsum(counts)

print(f"Index built: {n_clusters} clusters, avg size {n/n_clusters:.0f}")

# LDA Parameters - trained on FULL 14D data for maximum accuracy
print("Computing LDA parameters on 14D data...")
fraud_mask = labels == 1
legit_mask = labels == 0

mu_fraud_14d = vectors[fraud_mask].mean(axis=0)
mu_legit_14d = vectors[legit_mask].mean(axis=0)

# Pooled covariance inverse on 14D (sample for memory)
sample_size = min(500000, fraud_mask.sum(), legit_mask.sum())
fraud_sample = vectors[fraud_mask][:sample_size]
legit_sample = vectors[legit_mask][:sample_size]

fraud_cov = np.cov(fraud_sample.T)
legit_cov = np.cov(legit_sample.T)
pooled_cov = (fraud_cov + legit_cov) / 2
pooled_cov += np.eye(vectors.shape[1]) * 0.01
cov_inv = np.linalg.inv(pooled_cov)

# LDA weights: w = Sigma^-1 * (mu_fraud - mu_legit)
w_14d = cov_inv @ (mu_fraud_14d - mu_legit_14d)
# LDA intercept: w0 = -0.5 * w^T * (mu_fraud + mu_legit)
w0_14d = -0.5 * w_14d.T @ (mu_fraud_14d + mu_legit_14d)

print(f"LDA 14D weights computed: w0={w0_14d:.4f}")

# Also compute SVD and K-means for IVF (keep for fallback)
print(f"Computing SVD {vectors.shape[1]}D -> {n_components}D...")

# Save files
print("Saving binary files...")
with open(f"{output_dir}/vectors_8d_sorted.bin", "wb") as f:
    f.write(vectors_sorted.tobytes())
with open(f"{output_dir}/labels_sorted.bin", "wb") as f:
    f.write(labels_sorted.tobytes())
with open(f"{output_dir}/centroids.bin", "wb") as f:
    f.write(centroids.tobytes())
with open(f"{output_dir}/bucket_starts.bin", "wb") as f:
    f.write(bucket_starts.tobytes())
# Save LDA 14D parameters
with open(f"{output_dir}/lda_w.bin", "wb") as f:
    f.write(w_14d.astype(np.float32).tobytes())
with open(f"{output_dir}/lda_w0.bin", "wb") as f:
    f.write(np.array([w0_14d], dtype=np.float32).tobytes())

# Save SVD matrix for IVF fallback
with open(f"{output_dir}/svd_matrix.bin", "wb") as f:
    f.write(svd.components_.astype(np.float32).tobytes())

# Save centroids in 8D space for Mahalanobis
with open(f"{output_dir}/fraud_centroid.bin", "wb") as f:
    f.write(vectors_8d[labels == 1].mean(axis=0).astype(np.float32).tobytes())
with open(f"{output_dir}/legit_centroid.bin", "wb") as f:
    f.write(vectors_8d[labels == 0].mean(axis=0).astype(np.float32).tobytes())

with open(f"{output_dir}/cov_inv.bin", "wb") as f:
    f.write(cov_inv.astype(np.float32).tobytes())

print(f"Saved all files in {time.time()-t0:.1f}s")
