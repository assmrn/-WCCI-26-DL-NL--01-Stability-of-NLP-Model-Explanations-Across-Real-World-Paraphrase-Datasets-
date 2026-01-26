# # ============================================================
# # Baseline BERT Model Wrapper
# # Provides a unified predict_proba() interface for explainers
# # ============================================================

# import torch
# import numpy as np
# from transformers import BertTokenizerFast, BertForSequenceClassification

# # ===============================
# # CONFIG
# # ===============================

# MODEL_NAME = "bert-base-uncased"
# MODEL_PATH = "models/final_baseline_bert.pt"   # path to your saved baseline model
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ===============================
# # LOAD TOKENIZER & MODEL
# # ===============================

# print("Using device:", DEVICE)
# print("Loading tokenizer...")
# tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# print("Loading model...")
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)

# model.to(DEVICE)
# model.eval()

# print("✅ Model loaded successfully")

# # ===============================
# # PREDICTION FUNCTION
# # ===============================

# def predict_proba(text_pairs, max_len=128, batch_size=16):
#     """
#     text_pairs: list of (text_a, text_b)
#     returns: numpy array of probabilities [N, 2]
#     """

#     all_probs = []

#     with torch.no_grad():
#         for i in range(0, len(text_pairs), batch_size):
#             batch_pairs = text_pairs[i:i+batch_size]

#             texts_a = [p[0] for p in batch_pairs]
#             texts_b = [p[1] for p in batch_pairs]

#             enc = tokenizer(
#                 texts_a,
#                 texts_b,
#                 truncation=True,
#                 padding=True,
#                 max_length=max_len,
#                 return_tensors="pt"
#             )

#             enc = {k: v.to(DEVICE) for k, v in enc.items()}

#             outputs = model(**enc)
#             probs = torch.softmax(outputs.logits, dim=-1)

#             all_probs.append(probs.cpu().numpy())

#     return np.vstack(all_probs)

# # ===============================
# # QUICK SELF-TEST (IMPORTANT)
# # ===============================

# if __name__ == "__main__":

#     test_pairs = [
#         ("How can I learn machine learning?",
#          "What is the best way to study machine learning?")
#     ]

#     probs = predict_proba(test_pairs)
#     print("\nTest probabilities:", probs)





#---------------------------------------
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
MODEL_PATH = "models/distilbert_baseline_final"   # folder where final model was saved
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD TOKENIZER & MODEL
# ===============================

print("Using device:", DEVICE)

print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Loading trained weights...")
state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=DEVICE)
model.load_state_dict(state_dict)

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
