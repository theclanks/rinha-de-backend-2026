import json
import gzip
import numpy as np
import struct
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import time

print("Loading 3M reference vectors...")
with gzip.open("resources/references.json.gz", "rt") as f:
    data = json.load(f)

X = np.array([r["vector"] for r in data], dtype=np.float32)
y = np.array([1 if r["label"] == "fraud" else 0 for r in data], dtype=np.int32)
del data
print(f"Shape: {X.shape}, Fraud: {y.sum()}, Legit: {(1-y).sum()}")

# ============================================================
# Grid search best RF config
# ============================================================
print("\n" + "="*60)
print("RF grid search on 500K subsample")
print("="*60)

rng = np.random.RandomState(42)
idx = rng.choice(len(X), 500000, replace=False)
Xs, ys = X[idx], y[idx]

best_penalty = float('inf')
best_cfg = None

for n_trees in [5, 10, 15, 20]:
    for depth in [8, 10, 12, 15]:
        for min_leaf in [30, 50, 100]:
            rf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=depth, min_samples_leaf=min_leaf,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            rf.fit(Xs, ys)
            preds = rf.predict(Xs)
            tn, fp, fn, tp = confusion_matrix(ys, preds).ravel()
            penalty = fp * 25 + fn * 90
            acc = (tp+tn) / (tp+tn+fp+fn)
            if penalty < best_penalty:
                best_penalty = penalty
                best_cfg = (n_trees, depth, min_leaf)
                print(f"  NEW BEST: n={n_trees}, d={depth}, leaf={min_leaf}: acc={acc:.4f}, TP={tp}, FP={fp}, FN={fn}, penalty={penalty}")
            elif n_trees == 10 and depth == 10 and min_leaf == 100:
                print(f"  baseline: n={n_trees}, d={depth}, leaf={min_leaf}: acc={acc:.4f}, TP={tp}, FP={fp}, FN={fn}, penalty={penalty}")

print(f"\nBest config: n_trees={best_cfg[0]}, depth={best_cfg[1]}, min_leaf={best_cfg[2]}, penalty={best_penalty}")

# ============================================================
# Test probability thresholds with best config
# ============================================================
print("\n" + "="*60)
print("Probability thresholds with best config")
print("="*60)

n_trees, depth, min_leaf = best_cfg
rf = RandomForestClassifier(
    n_estimators=n_trees, max_depth=depth, min_samples_leaf=min_leaf,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(Xs, ys)
proba = rf.predict_proba(Xs)[:, 1]

for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    preds = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds).ravel()
    acc = (tp+tn) / (tp+tn+fp+fn)
    penalty = fp * 25 + fn * 90
    print(f"  thresh={thresh:.2f}: acc={acc:.4f}, TP={tp}, FP={fp}, FN={fn}, penalty={penalty}")

# ============================================================
# Retrain best RF on FULL 3M dataset
# ============================================================
print(f"\n{'='*60}")
print(f"Retraining on FULL 3M dataset: RF n={n_trees}, d={depth}, min_leaf={min_leaf}")
print("="*60)

rf_full = RandomForestClassifier(
    n_estimators=n_trees, max_depth=depth, min_samples_leaf=min_leaf,
    class_weight='balanced', random_state=42, n_jobs=-1
)
t0 = time.time()
rf_full.fit(X, y)
print(f"Training time: {time.time()-t0:.1f}s")

preds_full = rf_full.predict(X)
tn, fp, fn, tp = confusion_matrix(y, preds_full).ravel()
acc = (tp+tn) / (tp+tn+fp+fn)
penalty = fp * 25 + fn * 90
print(f"FULL 3M: acc={acc:.4f}, TP={tp}, FP={fp}, FN={fn}, penalty={penalty}")

# ============================================================
# Serialize RF to binary format
# ============================================================
print(f"\n{'='*60}")
print("Serializing RF to binary")
print("="*60)

# Format:
#   int32 n_trees
#   int32 n_features
#   Per tree:
#     int32 n_nodes
#     Per node (n_nodes times):
#       int32 feature (-1 = leaf)
#       float32 threshold
#       int32 left_child
#       int32 right_child
#       float32 fraud_prob (value if leaf)

n_features = X.shape[1]
total_nodes = sum(est.tree_.node_count for est in rf_full.estimators_)
print(f"Total nodes across {n_trees} trees: {total_nodes}")
print(f"File size estimate: {4 + 4 + n_trees * (4 + total_nodes//n_trees * 20)} bytes")

print("Writing binary...")
with open("resources/rf_forest.bin", "wb") as f:
    f.write(struct.pack("<ii", n_trees, n_features))
    
    for i, est in enumerate(rf_full.estimators_):
        tree = est.tree_
        n_nodes = tree.node_count
        f.write(struct.pack("<i", n_nodes))
        
        features = tree.feature
        thresholds = tree.threshold.astype(np.float32)
        children_left = tree.children_left
        children_right = tree.children_right
        values = tree.value[:, 0, :]
        
        for j in range(n_nodes):
            feat = int(features[j])
            if feat == -2:
                feat = -1
            thresh = float(thresholds[j])
            left = int(children_left[j])
            right = int(children_right[j])
            
            total = values[j][0] + values[j][1]
            fraud_prob = float(values[j][1] / total) if total > 0 else 0.0
            
            # 20 bytes per node: i32 feat, f32 thresh, i32 left, i32 right, f32 fraud_prob
            f.write(struct.pack("<i", feat))
            f.write(struct.pack("<f", thresh))
            f.write(struct.pack("<i", left))
            f.write(struct.pack("<i", right))
            f.write(struct.pack("<f", fraud_prob))
    
    # Write default threshold (0.5)
    f.write(struct.pack("<f", 0.5))

import os
fsize = os.path.getsize("resources/rf_forest.bin")
print(f"Written rf_forest.bin: {fsize} bytes ({fsize/1024:.1f} KB)")

# ============================================================
# Verify by reading back
# ============================================================
print("\nVerifying binary...")
with open("resources/rf_forest.bin", "rb") as f:
    n_trees_r, n_feat_r = struct.unpack("<ii", f.read(8))
    print(f"n_trees={n_trees_r}, n_features={n_feat_r}")
    
    for t in range(n_trees_r):
        n_nodes = struct.unpack("<i", f.read(4))[0]
        print(f"  Tree {t}: {n_nodes} nodes")
        for n in range(n_nodes):
            feat, thresh, left, right, fp = struct.unpack("<ifiif", f.read(20))
            pass  # skip printing all nodes
    threshold = struct.unpack("<f", f.read(4))[0]
    print(f"Default threshold: {threshold}")

print("\nDone! Binary ready at resources/rf_forest.bin")
