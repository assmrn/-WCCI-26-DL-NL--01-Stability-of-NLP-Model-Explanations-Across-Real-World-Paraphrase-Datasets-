import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

RESULTS_PATH = r"results/stability/stability_20260118_232158.json"
SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# LOAD RESULTS
# ===============================

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

scores = np.array(data["stability_scores"])

print("Loaded:", RESULTS_PATH)
print("Samples:", len(scores))
print("Mean:", scores.mean())
print("Std :", scores.std())
print("Min :", scores.min())
print("Max :", scores.max())

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(8, 5))
plt.hist(scores, bins=15)
plt.title("Explanation Stability Distribution (Baseline BERT + LIME)")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.grid(True)

save_path = os.path.join(SAVE_DIR, "baseline_stability_hist.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print("\n📊 Plot saved to:", save_path)
