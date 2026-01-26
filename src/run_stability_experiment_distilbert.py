import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

from lime_explanation_utils_distilbert import explain_pair
from explanation_vectorizer import vectorize_explanations
from stability_metrics import cosine_similarity

# ===============================
# CONFIG
# ===============================

RESULTS_DIR = "results/distilbert_stability"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
N_SAMPLES = 100

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(DATA_PATH).sample(N_SAMPLES, random_state=42)
scores = []

print("Running DistilBERT stability experiment on", len(df), "samples")

# ===============================
# LOOP
# ===============================

for idx, row in df.iterrows():
    exp1 = explain_pair(row["text_a"], row["text_b"])
    exp2 = explain_pair(row["text_b"], row["text_a"])

    _, v1, v2 = vectorize_explanations(exp1, exp2)
    sim = cosine_similarity(v1, v2)

    scores.append(sim)
    print(f"[{len(scores)}] Stability:", round(sim, 4))

scores = np.array(scores)

# ===============================
# SAVE
# ===============================

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "distilbert-baseline-1epoch",
    "explainer": "LIME",
    "metric": "cosine_similarity",
    "num_samples": int(len(scores)),
    "stability_scores": scores.tolist(),
    "mean": float(scores.mean()),
    "std": float(scores.std()),
    "min": float(scores.min()),
    "max": float(scores.max())
}

fname = f"distilbert_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
path = os.path.join(RESULTS_DIR, fname)

with open(path, "w") as f:
    json.dump(results, f, indent=4)

print("\n📁 Results saved to:", path)
