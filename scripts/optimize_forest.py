import argparse
import gzip
import json
import os
import struct
import time

import numpy as np
import optuna
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def penalty_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return fp * 25 + fn * 90, (tn, fp, fn, tp)


def best_threshold(y_true, proba):
    best = None
    for threshold in np.linspace(0.30, 0.70, 81):
        penalty, matrix = penalty_at_threshold(y_true, proba, float(threshold))
        if best is None or penalty < best[0]:
            best = (penalty, float(threshold), matrix)
    return best


def load_data(path):
    print(f"loading {path}...")
    with gzip.open(path, "rt") as f:
        data = json.load(f)

    x = np.asarray([entry["vector"] for entry in data], dtype=np.float32)
    y = np.asarray([1 if entry["label"] == "fraud" else 0 for entry in data], dtype=np.int8)
    print(f"loaded rows={len(y):,} fraud={int(y.sum()):,} legit={int((1 - y).sum()):,}")
    return x, y


def build_model(trial, seed):
    model_type = trial.suggest_categorical("model", ["rf", "extra_trees"])
    n_estimators = trial.suggest_int("n_estimators", 5, 80)
    max_depth = trial.suggest_int("max_depth", 5, 24)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 250, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 80, log=True)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.5, 0.75])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

    klass = ExtraTreesClassifier if model_type == "extra_trees" else RandomForestClassifier
    if model_type == "extra_trees" and class_weight == "balanced_subsample":
        class_weight = "balanced"

    return klass(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        criterion=criterion,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )


def serialize_forest(model, threshold, path):
    n_trees = len(model.estimators_)
    n_features = model.n_features_in_

    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n_trees, n_features))

        for est in model.estimators_:
            tree = est.tree_
            f.write(struct.pack("<i", tree.node_count))

            values = tree.value[:, 0, :]
            for idx in range(tree.node_count):
                feature = int(tree.feature[idx])
                if feature == -2:
                    feature = -1

                total = values[idx][0] + values[idx][1]
                fraud_prob = float(values[idx][1] / total) if total > 0 else 0.0

                f.write(struct.pack("<i", feature))
                f.write(struct.pack("<f", float(tree.threshold[idx])))
                f.write(struct.pack("<i", int(tree.children_left[idx])))
                f.write(struct.pack("<i", int(tree.children_right[idx])))
                f.write(struct.pack("<f", fraud_prob))

        f.write(struct.pack("<f", float(threshold)))

    size = os.path.getsize(path)
    print(f"wrote {path}: {size:,} bytes ({size / 1024:.1f} KiB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", default="resources/references.json.gz")
    parser.add_argument("--output", default="resources/rf_forest_candidate.bin")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--train-size", type=int, default=700_000)
    parser.add_argument("--valid-size", type=int, default=700_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-full", action="store_true")
    parser.add_argument("--params-json")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    x, y = load_data(args.references)

    if args.params_json:
        params = json.loads(args.params_json)
        fixed = optuna.trial.FixedTrial(params)
        model = build_model(fixed, args.seed)
        threshold = args.threshold if args.threshold is not None else 0.5
        print(f"training fixed model threshold={threshold:.3f} params={params}")
        model.fit(x, y)
        serialize_forest(model, threshold, args.output)
        return

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        train_size=args.train_size,
        test_size=args.valid_size,
        stratify=y,
        random_state=args.seed,
    )

    def objective(trial):
        model = build_model(trial, args.seed)
        started = time.time()
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_valid)[:, 1]
        penalty, threshold, (tn, fp, fn, tp) = best_threshold(y_valid, proba)

        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("tn", int(tn))
        trial.set_user_attr("fp", int(fp))
        trial.set_user_attr("fn", int(fn))
        trial.set_user_attr("tp", int(tp))
        trial.set_user_attr("seconds", round(time.time() - started, 2))

        print(
            f"trial={trial.number} penalty={penalty:,} threshold={threshold:.3f} "
            f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,} seconds={time.time() - started:.1f}"
        )
        return penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    threshold = best.user_attrs["threshold"]
    print(f"best penalty={best.value:,} threshold={threshold:.3f} params={best.params}")

    fixed = optuna.trial.FixedTrial(best.params)
    model = build_model(fixed, args.seed)
    if args.retrain_full:
        print("retraining best model on full dataset...")
        model.fit(x, y)
    else:
        print("training best model on optimization training split...")
        model.fit(x_train, y_train)

    serialize_forest(model, threshold, args.output)


if __name__ == "__main__":
    main()
