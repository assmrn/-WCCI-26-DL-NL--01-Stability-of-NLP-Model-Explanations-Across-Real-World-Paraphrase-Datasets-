import json
import pandas as pd
import numpy as np
import os

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
RESULTS_PATH = r"results/stability/stability_20260118_232158.json"
OUT_PATH = "results/unstable_cases.json"

df = pd.read_csv(DATA_PATH)

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

scores = np.array(data["stability_scores"])

worst_idx = np.argsort(scores)[:10]

unstable = []

for i in worst_idx:
    row = df.iloc[i]
    unstable.append({
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "stability": float(scores[i])
    })

os.makedirs("results", exist_ok=True)

with open(OUT_PATH, "w") as f:
    json.dump(unstable, f, indent=4)

print("🔥 Saved unstable cases to:", OUT_PATH)
