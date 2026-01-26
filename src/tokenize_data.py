import pandas as pd
from transformers import AutoTokenizer

# ---------- Paths ----------
train_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\train.csv"

# ---------- Load data ----------
df = pd.read_csv(train_path)

print("Total training samples:", len(df))
print(df.head())

# ---------- Load tokenizer ----------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ---------- Test tokenization ----------
sample = df.iloc[0]

encoded = tokenizer(
    sample["text_a"],
    sample["text_b"],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print("\nTokenizer output keys:", encoded.keys())
print("input_ids shape:", encoded["input_ids"].shape)
print("attention_mask shape:", encoded["attention_mask"].shape)
print("token_type_ids shape:", encoded["token_type_ids"].shape)
