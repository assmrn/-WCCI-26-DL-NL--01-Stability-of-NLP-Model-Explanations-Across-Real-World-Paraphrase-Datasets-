import json
import pandas as pd
import numpy as np

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
RESULTS_PATH = r"results/stability/stability_20260118_232158.json"

df = pd.read_csv(DATA_PATH)

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

scores = np.array(data["stability_scores"])

# Get indices of 5 lowest stability cases
worst_idx = np.argsort(scores)[:5]

print("\n🔥 Most unstable explanation cases:\n")

for rank, i in enumerate(worst_idx, 1):
    row = df.iloc[i]
    print("===================================")
    print(f"Rank {rank} | Stability = {scores[i]:.4f}\n")
    print("A:", row["text_a"])
    print("B:", row["text_b"])
