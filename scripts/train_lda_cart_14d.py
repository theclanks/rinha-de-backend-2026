import gzip
import json
import struct

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


print("loading references...")
refs = json.load(gzip.open("resources/references.json.gz", "rt"))
X = np.asarray([r["vector"] for r in refs], dtype=np.float32)
y = np.asarray([1 if r["label"] == "fraud" else 0 for r in refs], dtype=np.int8)
del refs

print("training LDA 14D...")
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
w = lda.coef_[0].astype(np.float32)
w0 = np.asarray([lda.intercept_[0]], dtype=np.float32)
w.tofile("resources/lda_w_14d.bin")
w0.tofile("resources/lda_w0_14d.bin")
print(f"wrote LDA: w={w.shape} w0={w0[0]:.6f}")

print("training CART 14D...")
cart = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
)
cart.fit(X, y)
tree = cart.tree_

with open("resources/cart_tree_14d.bin", "wb") as f:
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

print(f"wrote resources/cart_tree_14d.bin nodes={tree.node_count}")
