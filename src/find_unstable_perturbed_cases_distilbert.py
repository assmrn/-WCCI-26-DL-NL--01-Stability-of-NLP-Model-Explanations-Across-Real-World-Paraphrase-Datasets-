import json, numpy as np

JSON_PATH = r"results\distilbert_stability_perturbed\distilbert_stability_perturbed_20260121_184311.json"
TOP_K = 10
#"D:\research_sprint\results\distilbert_stability_perturbed\distilbert_stability_perturbed_20260121_184311.json"
with open(JSON_PATH) as f:
    data = json.load(f)

scores = np.array(data["scores"])
idx = np.argsort(scores)

print("\n🔥 Most unstable perturbed DistilBERT cases:\n")

for r, i in enumerate(idx[:TOP_K], 1):
    print(f"Rank {r} | Index {i} | Stability = {scores[i]:.4f}")

print("\nTotal:", len(scores))
print("Mean:", scores.mean())
print("Min:", scores.min())
print("Max:", scores.max())

