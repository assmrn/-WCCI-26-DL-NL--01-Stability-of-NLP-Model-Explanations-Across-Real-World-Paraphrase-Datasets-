# ============================================================
# LIME explanation for Baseline BERT
# ============================================================

import numpy as np
from lime.lime_text import LimeTextExplainer
from baseline_model_wrapper import predict_proba

# -----------------------------
# Class names (for LIME display)
# -----------------------------
class_names = ["not paraphrase", "paraphrase"]

explainer = LimeTextExplainer(class_names=class_names)

# -----------------------------
# Example pair
# -----------------------------

text_a = "How can I learn machine learning?"
text_b = "What is the best way to study machine learning?"

combined_text = text_a + " [SEP] " + text_b

# -----------------------------
# Prediction wrapper for LIME
# -----------------------------

def lime_predict(texts):
    pairs = []
    for t in texts:
        if "[SEP]" in t:
            a, b = t.split("[SEP]", 1)
        else:
            a, b = t, ""
        pairs.append((a.strip(), b.strip()))

    return predict_proba(pairs)

# -----------------------------
# Run LIME
# -----------------------------

print("Generating LIME explanation...")

exp = explainer.explain_instance(
    combined_text,
    lime_predict,
    num_features=12,
    num_samples=1500
)

# -----------------------------
# Show results
# -----------------------------

print("\nTop explanation features:")
for word, weight in exp.as_list():
    print(f"{word:>15s} : {weight:.4f}")

# Save html visualization
exp.save_to_file("lime_baseline_example.html")
print("\n✅ Explanation saved to lime_baseline_example.html")
