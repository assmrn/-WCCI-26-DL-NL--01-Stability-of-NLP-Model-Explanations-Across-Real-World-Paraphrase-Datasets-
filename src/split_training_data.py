import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Load full dataset ----------
data_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\training_full.csv"
df = pd.read_csv(data_path)

print("Total samples:", len(df))
print("Overall label distribution:")
print(df["label"].value_counts())

# ---------- First split: train (80%) and temp (20%) ----------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ---------- Second split: validation (10%) and test (10%) ----------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nTrain size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(val_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

# ---------- Save splits ----------
base_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed"

train_df.to_csv(base_path + r"\train.csv", index=False)
val_df.to_csv(base_path + r"\validation.csv", index=False)
test_df.to_csv(base_path + r"\test.csv", index=False)

print("\n✅ Train / validation / test splits saved.")
