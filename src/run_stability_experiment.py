import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

from lime_explanation_utils import explain_pair
from explanation_vectorizer import vectorize_explanations
from stability_metrics import cosine_similarity

# ===============================
# CONFIG
# ===============================

RESULTS_DIR = "results/stability"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
N_SAMPLES = 100   # small test first

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(DATA_PATH).sample(N_SAMPLES, random_state=42)

scores = []

print("Running stability experiment on", len(df), "samples")

# ===============================
# EXPERIMENT LOOP
# ===============================

for idx, row in df.iterrows():
    text_a = row["text_a"]
    text_b = row["text_b"]

    exp1 = explain_pair(text_a, text_b)
    exp2 = explain_pair(text_b, text_a)  # perturbation

    features, v1, v2 = vectorize_explanations(exp1, exp2)
    sim = cosine_similarity(v1, v2)

    scores.append(sim)
    print(f"[{len(scores)}] Stability:", round(sim, 4))

scores = np.array(scores)

mean_stability = scores.mean()
std_stability = scores.std()
min_stability = scores.min()
max_stability = scores.max()

print("\n==============================")
print("Mean stability:", mean_stability)
print("Std deviation :", std_stability)
print("Min stability :", min_stability)
print("Max stability :", max_stability)
print("==============================")

# ===============================
# SAVE RESULTS
# ===============================

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "bert-baseline-1epoch",
    "explainer": "LIME",
    "metric": "cosine_similarity",
    "num_samples": int(len(scores)),
    "stability_scores": scores.tolist(),
    "mean": float(mean_stability),
    "std": float(std_stability),
    "min": float(min_stability),
    "max": float(max_stability)
}

filename = f"stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
path = os.path.join(RESULTS_DIR, filename)

with open(path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n📁 Results saved to: {path}")
