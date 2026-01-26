# ============================================================
# DistilBERT Baseline Model Wrapper
# Provides a unified predict_proba() interface for explainers
# ============================================================

import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ===============================
# CONFIG
# ===============================

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "models/distilbert_baseline_final"   # <-- FOLDER, not .pt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD TOKENIZER & MODEL
# ===============================

print("Using device:", DEVICE)
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

print("Loading trained DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print("✅ DistilBERT model loaded successfully")

# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_proba(text_pairs, max_len=128, batch_size=16):
    """
    text_pairs: list of (text_a, text_b)
    returns: numpy array of probabilities [N, 2]
    """

    all_probs = []

    with torch.no_grad():
        for i in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[i:i+batch_size]

            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]

            enc = tokenizer(
                texts_a,
                texts_b,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt"
            )

            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)

            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)

# ===============================
# QUICK SELF-TEST
# ===============================

if __name__ == "__main__":

    test_pairs = [
        ("How can I learn machine learning?",
         "What is the best way to study machine learning?")
    ]

    probs = predict_proba(test_pairs)
    print("\nTest probabilities:", probs)
