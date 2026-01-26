import json, os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = r"results/distilbert_stability/distilbert_stability_20260120_203200.json"
SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

scores = np.array(data["stability_scores"])

print("Samples:", len(scores))
print("Mean:", scores.mean())
print("Std:", scores.std())
print("Min:", scores.min())
print("Max:", scores.max())

plt.hist(scores, bins=20)
plt.title("DistilBERT Explanation Stability")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.grid(True)

out = os.path.join(SAVE_DIR, "distilbert_stability_hist.png")
plt.savefig(out, dpi=200)
plt.show()

print("\n📊 Plot saved to:", out)

