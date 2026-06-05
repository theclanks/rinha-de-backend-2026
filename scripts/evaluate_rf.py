import argparse
import gzip
import json
import struct


def iter_json_array(path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt") as f:
        buf = ""
        in_array = False

        while True:
            chunk = f.read(chunk_size)
            if not chunk and not buf:
                return
            buf += chunk

            while True:
                buf = buf.lstrip()
                if not in_array:
                    if not buf:
                        break
                    if buf[0] != "[":
                        raise ValueError("expected JSON array")
                    buf = buf[1:]
                    in_array = True
                    continue

                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == "]":
                    return
                if buf[0] == ",":
                    buf = buf[1:]
                    continue

                try:
                    item, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                yield item
                buf = buf[idx:]

            if not chunk:
                return


def load_forest(path):
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    n_trees, n_features = struct.unpack_from("<ii", data, offset)
    offset += 8
    trees = []

    for _ in range(n_trees):
        (n_nodes,) = struct.unpack_from("<i", data, offset)
        offset += 4
        nodes = []

        for _ in range(n_nodes):
            nodes.append(struct.unpack_from("<ifiif", data, offset))
            offset += 20

        trees.append(nodes)

    (threshold,) = struct.unpack_from("<f", data, offset)
    return n_features, threshold, trees


def predict_tree(nodes, vec):
    idx = 0
    while True:
        feature, threshold, left, right, fraud_prob = nodes[idx]
        if feature == -1:
            return fraud_prob
        idx = left if vec[feature] <= threshold else right


def predict_proba(forest, vec):
    n_features, _threshold, trees = forest
    if len(vec) < n_features:
        raise ValueError(f"expected {n_features} features, got {len(vec)}")
    return sum(predict_tree(tree, vec) for tree in trees) / len(trees)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", default="resources/rf_forest.bin")
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    forest = load_forest(args.forest)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    stats = {t: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for t in thresholds}

    entries = iter_json_array(args.references)
    if not args.stream:
        entries = json.load(gzip.open(args.references, "rt"))

    total = 0
    fraud = 0
    for entry in entries:
        vec = entry["vector"]
        label = 1 if entry["label"] == "fraud" else 0
        proba = predict_proba(forest, vec)

        fraud += label
        total += 1

        for threshold in thresholds:
          pred = 1 if proba >= threshold else 0
          if pred == 1 and label == 1:
              stats[threshold]["tp"] += 1
          elif pred == 1:
              stats[threshold]["fp"] += 1
          elif label == 1:
              stats[threshold]["fn"] += 1
          else:
              stats[threshold]["tn"] += 1

        if total % 500000 == 0:
            print(f"processed={total:,}")
        if args.limit and total >= args.limit:
            break

    print(f"model: trees={len(forest[2])} features={forest[0]} model_threshold={forest[1]:.4f}")
    print(f"data: total={total:,} fraud={fraud:,} legit={total - fraud:,}")

    best = None
    for threshold in thresholds:
        s = stats[threshold]
        penalty = s["fp"] * 25 + s["fn"] * 90
        acc = (s["tp"] + s["tn"]) / total
        line = (
            f"threshold={threshold:.3f} acc={acc:.5f} "
            f"TP={s['tp']:,} FP={s['fp']:,} FN={s['fn']:,} TN={s['tn']:,} "
            f"penalty={penalty:,}"
        )
        print(line)
        if best is None or penalty < best[0]:
            best = (penalty, threshold)

    print(f"best_threshold={best[1]:.3f} best_penalty={best[0]:,}")


if __name__ == "__main__":
    main()
