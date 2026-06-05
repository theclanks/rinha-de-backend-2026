import argparse
import gzip
import json
import os
import struct
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--out-dir", default="resources")
    parser.add_argument("--clusters", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    print(f"loading {args.references}...")
    with gzip.open(args.references, "rt") as f:
        data = json.load(f)

    x = np.asarray([entry["vector"] for entry in data], dtype=np.float32)
    y = np.asarray([1 if entry["label"] == "fraud" else 0 for entry in data], dtype=np.uint8)
    del data
    print(f"loaded rows={len(y):,} dims={x.shape[1]} fraud={int(y.sum()):,}")

    print(f"training MiniBatchKMeans clusters={args.clusters} batch={args.batch_size}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        batch_size=args.batch_size,
        n_init=3,
        random_state=args.seed,
        reassignment_ratio=0.01,
        verbose=0,
    )
    assignments = kmeans.fit_predict(x)

    print("sorting vectors by cluster...")
    order = np.argsort(assignments, kind="stable")
    sorted_assignments = assignments[order]
    x_sorted = np.ascontiguousarray(x[order], dtype=np.float32)
    y_sorted = np.ascontiguousarray(y[order], dtype=np.uint8)

    counts = np.bincount(sorted_assignments, minlength=args.clusters).astype(np.int32)
    starts = np.zeros(args.clusters + 1, dtype=np.int32)
    starts[1:] = np.cumsum(counts, dtype=np.int32)
    centroids = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    vectors_path = os.path.join(args.out_dir, "vectors_14d_sorted.bin")
    labels_path = os.path.join(args.out_dir, "labels_14d_sorted.bin")
    centroids_path = os.path.join(args.out_dir, "centroids_14d.bin")
    starts_path = os.path.join(args.out_dir, "bucket_starts_14d.bin")

    x_sorted.tofile(vectors_path)
    y_sorted.tofile(labels_path)
    centroids.tofile(centroids_path)
    starts.tofile(starts_path)

    meta_path = os.path.join(args.out_dir, "ivf14_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "clusters": args.clusters,
                "dims": int(x.shape[1]),
                "rows": int(len(y)),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "seconds": round(time.time() - started, 2),
            },
            f,
            separators=(",", ":"),
        )

    for path in [vectors_path, labels_path, centroids_path, starts_path, meta_path]:
        print(f"wrote {path}: {os.path.getsize(path):,} bytes")

    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
