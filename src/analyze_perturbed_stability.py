import json
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = r"results/stability_perturbed/stability_perturbed_20260118_235804.json"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(PATH, "r") as f:
    data = json.load(f)

scores = np.array(data["scores"])

print("Samples:", len(scores))
print("Mean:", scores.mean())
print("Std :", scores.std())
print("Min :", scores.min())
print("Max :", scores.max())

plt.hist(scores, bins=25)
plt.title("Perturbed explanation stability (LIME + BERT)")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.grid(True)

out_path = os.path.join(PLOT_DIR, "perturbed_stability_hist.png")
plt.savefig(out_path, dpi=200)
plt.close()

print("\n📊 Plot saved to:", out_path)
