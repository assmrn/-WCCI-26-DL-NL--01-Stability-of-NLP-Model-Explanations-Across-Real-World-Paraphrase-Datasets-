import json, os
import numpy as np
import matplotlib.pyplot as plt

PATH = r"results/distilbert_stability_perturbed\distilbert_stability_perturbed_20260121_184311.json"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

with open(PATH) as f:
    data = json.load(f)

scores = np.array(data["scores"])

print("Samples:", len(scores))
print("Mean:", scores.mean())
print("Std:", scores.std())
print("Min:", scores.min())
print("Max:", scores.max())

plt.hist(scores, bins=25)
plt.title("Perturbed Stability (DistilBERT)")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.grid(True)

out = os.path.join(PLOT_DIR, "distilbert_perturbed_stability_hist.png")
plt.savefig(out, dpi=200)
plt.close()

print("\n📊 Plot saved to:", out)
