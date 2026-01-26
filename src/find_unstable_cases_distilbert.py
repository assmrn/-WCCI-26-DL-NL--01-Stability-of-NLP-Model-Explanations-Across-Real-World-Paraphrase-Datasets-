import json, pandas as pd, numpy as np

DATA_PATH = r"D:\research_sprint\data\processed\validation.csv"
RESULTS_PATH = r"results\distilbert_stability\distilbert_stability_20260120_203200.json"
#"D:\research_sprint\results\distilbert_stability\distilbert_stability_20260120_203200.json"
df = pd.read_csv(DATA_PATH)

with open(RESULTS_PATH) as f:
    data = json.load(f)

scores = np.array(data["stability_scores"])
worst = np.argsort(scores)[:5]

print("\n🔥 Most unstable DistilBERT cases:\n")

for r, i in enumerate(worst, 1):
    row = df.iloc[i]
    print("="*40)
    print(f"Rank {r} | Stability = {scores[i]:.4f}")
    print("A:", row["text_a"])
    print("B:", row["text_b"])
