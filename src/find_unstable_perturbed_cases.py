import json
import numpy as np

JSON_PATH = r"results/stability_perturbed/stability_perturbed_20260118_235804.json"
TOP_K = 10   # how many most-unstable cases you want

with open(JSON_PATH, "r") as f:
    data = json.load(f)

scores = np.array(data["scores"])

# sort indices by stability (ascending = most unstable first)
sorted_idx = np.argsort(scores)

print("\n🔥 Most unstable perturbed explanation cases (by index):\n")

for rank, idx in enumerate(sorted_idx[:TOP_K], start=1):
    print(f"Rank {rank} | Index {idx} | Stability = {scores[idx]:.4f}")

print("\n==============================")
print("Total samples:", len(scores))
print("Mean stability:", scores.mean())
print("Min stability :", scores.min())
print("Max stability :", scores.max())
print("==============================")
