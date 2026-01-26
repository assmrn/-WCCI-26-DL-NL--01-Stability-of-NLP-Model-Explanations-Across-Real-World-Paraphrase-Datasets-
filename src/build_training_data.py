import pandas as pd

# ---------- Paths ----------
qqp_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\questions.csv"
paws_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_clean.tsv"

# ---------- Load ----------
qqp = pd.read_csv(qqp_path)
paws = pd.read_csv(paws_path, sep="\t")

# ---------- Standardize QQP ----------
qqp_df = qqp[["question1", "question2", "is_duplicate"]].dropna()
qqp_df.columns = ["text_a", "text_b", "label"]
qqp_df["source"] = "QQP"

# ---------- Standardize PAWS ----------
paws_df = paws[["sentence1", "sentence2", "label"]].dropna()
paws_df.columns = ["text_a", "text_b", "label"]
paws_df["source"] = "PAWS"

# ---------- Merge ----------
full_df = pd.concat([qqp_df, paws_df], ignore_index=True)

print("Final merged dataset size:", len(full_df))
print(full_df.head())
print("\nOverall label distribution:")
print(full_df["label"].value_counts())

# ---------- Save ----------
out_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\training_full.csv"
full_df.to_csv(out_path, index=False)

print("✅ Full training dataset saved.")
