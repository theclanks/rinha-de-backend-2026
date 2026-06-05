import argparse
import gzip
import json

from evaluate_rf import load_forest, predict_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", default="resources/rf_forest.bin")
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--threshold", type=float, default=0.513)
    parser.add_argument("--bands", default="0.02,0.04,0.06,0.08,0.10,0.15,0.20")
    args = parser.parse_args()

    forest = load_forest(args.forest)
    entries = json.load(gzip.open(args.references, "rt"))
    bands = [float(b) for b in args.bands.split(",")]
    stats = {band: {"fallback": 0, "errors": 0, "fp": 0, "fn": 0} for band in bands}
    total = len(entries)
    total_errors = 0

    for idx, entry in enumerate(entries, 1):
        label = 1 if entry["label"] == "fraud" else 0
        proba = predict_proba(forest, entry["vector"])
        pred = 1 if proba >= args.threshold else 0
        error = pred != label
        total_errors += int(error)

        for band in bands:
            if abs(proba - args.threshold) <= band:
                stats[band]["fallback"] += 1
                if error:
                    stats[band]["errors"] += 1
                    if pred == 1:
                        stats[band]["fp"] += 1
                    else:
                        stats[band]["fn"] += 1

        if idx % 500000 == 0:
            print(f"processed={idx:,}")

    print(f"threshold={args.threshold:.3f} total={total:,} total_errors={total_errors:,}")
    for band in bands:
        s = stats[band]
        covered = s["errors"] / total_errors * 100 if total_errors else 0.0
        rate = s["fallback"] / total * 100
        print(
            f"band=+-{band:.3f} fallback={s['fallback']:,} ({rate:.2f}%) "
            f"errors_covered={s['errors']:,}/{total_errors:,} ({covered:.2f}%) "
            f"fp={s['fp']:,} fn={s['fn']:,}"
        )


if __name__ == "__main__":
    main()
