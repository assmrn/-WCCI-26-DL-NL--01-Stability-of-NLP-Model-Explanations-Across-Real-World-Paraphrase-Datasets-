import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

from lime_explanation_utils_distilbert import explain_pair
from explanation_vectorizer import vectorize_explanations
from stability_metrics import cosine_similarity
from perturbations import simple_perturb

RESULTS_DIR = "results/distilbert_stability_perturbed"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
N_SAMPLES = 100

df = pd.read_csv(DATA_PATH).sample(N_SAMPLES, random_state=42)

scores = []
cases = []

print("Running DistilBERT PERTURBED stability experiment...")

for _, row in df.iterrows():

    exp1 = explain_pair(row["text_a"], row["text_b"])

    ta = simple_perturb(row["text_a"])
    tb = simple_perturb(row["text_b"])

    exp2 = explain_pair(ta, tb)

    _, v1, v2 = vectorize_explanations(exp1, exp2)
    sim = cosine_similarity(v1, v2)

    scores.append(sim)

    cases.append({
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "perturbed_a": ta,
        "perturbed_b": tb,
        "stability": float(sim)
    })

    print(f"[{len(scores)}] Stability:", round(sim, 4))

scores = np.array(scores)

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "distilbert-baseline-1epoch",
    "explainer": "LIME",
    "perturbation": "lowercase + punctuation removal",
    "num_samples": len(scores),
    "scores": scores.tolist(),
    "mean": float(scores.mean()),
    "std": float(scores.std()),
    "min": float(scores.min()),
    "max": float(scores.max()),
    "cases": cases
}

fname = f"distilbert_stability_perturbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
path = os.path.join(RESULTS_DIR, fname)

with open(path, "w") as f:
    json.dump(results, f, indent=4)

print("\n📁 Results saved to:", path)
