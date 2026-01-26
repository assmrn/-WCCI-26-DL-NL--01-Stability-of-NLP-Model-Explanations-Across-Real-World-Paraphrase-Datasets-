import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ===============================
# CONFIG
# ===============================

MODEL_PATH = "models/final_baseline_bert.pt"
MODEL_NAME = "bert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD TOKENIZER
# ===============================

print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# ===============================
# LOAD MODEL
# ===============================

print("Loading trained baseline model...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully.")

# ===============================
# TEST SENTENCE PAIR
# ===============================

text_a = "How can I learn machine learning?"
text_b = "What is the best way to study machine learning?"

print("\nText A:", text_a)
print("Text B:", text_b)

inputs = tokenizer(
    text_a,
    text_b,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# ===============================
# INFERENCE
# ===============================

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

pred = torch.argmax(probs, dim=-1).item()

print("\nLogits:", logits.cpu().numpy())
print("Probabilities:", probs.cpu().numpy())
print("Predicted label:", pred)

if pred == 1:
    print("🟢 Model thinks these ARE paraphrases.")
else:
    print("🔴 Model thinks these are NOT paraphrases.")
