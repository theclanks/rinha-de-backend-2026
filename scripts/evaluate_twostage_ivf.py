import argparse
import gzip
import json

import numpy as np

from evaluate_rf import load_forest, predict_proba


def load_ivf(prefix):
    vectors = np.fromfile(f"{prefix}/vectors_14d_sorted.bin", dtype=np.float32).reshape(-1, 14)
    labels = np.fromfile(f"{prefix}/labels_14d_sorted.bin", dtype=np.uint8)
    centroids = np.fromfile(f"{prefix}/centroids_14d.bin", dtype=np.float32).reshape(-1, 14)
    starts = np.fromfile(f"{prefix}/bucket_starts_14d.bin", dtype=np.int32)
    return vectors, labels, centroids, starts


def ivf_knn_score(vec, ivf, nprobe, k=5):
    vectors, labels, centroids, starts = ivf
    centroid_dists = np.sum((centroids - vec) ** 2, axis=1)
    cluster_ids = np.argpartition(centroid_dists, nprobe - 1)[:nprobe]

    best_dists = np.full(k, np.inf, dtype=np.float32)
    best_labels = np.zeros(k, dtype=np.uint8)

    for cluster_id in cluster_ids:
        start = starts[cluster_id]
        end = starts[cluster_id + 1]
        if end <= start:
            continue

        bucket = vectors[start:end]
        dists = np.sum((bucket - vec) ** 2, axis=1)
        take = min(k, len(dists))
        local = np.argpartition(dists, take - 1)[:take]

        for idx in local:
            dist = dists[idx]
            worst = int(np.argmax(best_dists))
            if dist < best_dists[worst]:
                best_dists[worst] = dist
                best_labels[worst] = labels[start + idx]

    return float(np.sum(best_labels) / k)


def add_result(stats, pred, label):
    if pred == 1 and label == 1:
        stats["tp"] += 1
    elif pred == 1:
        stats["fp"] += 1
    elif label == 1:
        stats["fn"] += 1
    else:
        stats["tn"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", default="resources/rf_forest.bin")
    parser.add_argument("--resources", default="resources")
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--threshold", type=float, default=0.513)
    parser.add_argument("--band", type=float, default=0.20)
    parser.add_argument("--nprobe", type=int, default=24)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    forest = load_forest(args.forest)
    ivf = load_ivf(args.resources)
    entries = json.load(gzip.open(args.references, "rt"))
    if args.limit:
        entries = entries[: args.limit]

    rf_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    two_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    fallback = 0

    for idx, entry in enumerate(entries, 1):
        vec = np.asarray(entry["vector"], dtype=np.float32)
        label = 1 if entry["label"] == "fraud" else 0
        prob = predict_proba(forest, entry["vector"])
        rf_pred = 1 if prob >= args.threshold else 0
        add_result(rf_stats, rf_pred, label)

        if abs(prob - args.threshold) <= args.band:
            fallback += 1
            score = ivf_knn_score(vec, ivf, args.nprobe)
            two_pred = 1 if score >= 0.6 else 0
        else:
            two_pred = rf_pred

        add_result(two_stats, two_pred, label)

        if idx % 100000 == 0:
            print(f"processed={idx:,} fallback={fallback:,}")

    total = len(entries)
    print(f"total={total:,} fallback={fallback:,} ({fallback / total * 100:.2f}%)")
    for name, stats in [("rf", rf_stats), ("two_stage", two_stats)]:
        penalty = stats["fp"] * 25 + stats["fn"] * 90
        acc = (stats["tp"] + stats["tn"]) / total
        print(
            f"{name} acc={acc:.5f} TP={stats['tp']:,} FP={stats['fp']:,} "
            f"FN={stats['fn']:,} TN={stats['tn']:,} penalty={penalty:,}"
        )


if __name__ == "__main__":
    main()
