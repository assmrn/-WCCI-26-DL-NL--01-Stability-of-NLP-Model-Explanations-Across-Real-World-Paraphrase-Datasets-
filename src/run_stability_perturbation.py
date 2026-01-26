import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

from lime_explanation_utils import explain_pair
from explanation_vectorizer import vectorize_explanations
from stability_metrics import cosine_similarity
from perturbations import simple_perturb

# ===============================
# CONFIG
# ===============================

RESULTS_DIR = "results/stability_perturbed"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
N_SAMPLES = 100  # start small, can increase later

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(DATA_PATH).sample(N_SAMPLES, random_state=42)

scores = []
all_cases = []

print("Running PERTURBATION stability experiment on", len(df), "samples")

# ===============================
# LOOP
# ===============================

for idx, row in df.iterrows():
    text_a = row["text_a"]
    text_b = row["text_b"]

    # original explanation
    exp1 = explain_pair(text_a, text_b)

    # perturbed explanation
    text_a_p = simple_perturb(text_a)
    text_b_p = simple_perturb(text_b)

    exp2 = explain_pair(text_a_p, text_b_p)

    features, v1, v2 = vectorize_explanations(exp1, exp2)
    sim = cosine_similarity(v1, v2)

    scores.append(sim)

    all_cases.append({
        "text_a": text_a,
        "text_b": text_b,
        "perturbed_a": text_a_p,
        "perturbed_b": text_b_p,
        "stability": float(sim)
    })

    print(f"[{len(scores)}] Stability:", round(sim, 4))

# ===============================
# SUMMARY
# ===============================

scores = np.array(scores)

mean_s = scores.mean()
std_s = scores.std()
min_s = scores.min()
max_s = scores.max()

print("\n==============================")
print("Mean stability:", mean_s)
print("Std deviation :", std_s)
print("Min stability :", min_s)
print("Max stability :", max_s)
print("==============================")

# ===============================
# SAVE RESULTS
# ===============================

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "bert-baseline-1epoch",
    "explainer": "LIME",
    "perturbation": "lowercase + punctuation removal",
    "num_samples": len(scores),
    "scores": scores.tolist(),
    "mean": float(mean_s),
    "std": float(std_s),
    "min": float(min_s),
    "max": float(max_s),
    "cases": all_cases
}

filename = f"stability_perturbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
path = os.path.join(RESULTS_DIR, filename)

with open(path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n📁 Results saved to: {path}")
