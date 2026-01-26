import pandas as pd

qqp_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\questions.csv"

# Load QQP
qqp = pd.read_csv(qqp_path)

print("QQP columns:")
print(qqp.columns)

# Keep only true paraphrases
qqp_para = qqp[qqp["is_duplicate"] == 1]

# Select and rename columns
qqp_para = qqp_para[["question1", "question2"]]
qqp_para.columns = ["text_a", "text_b"]

# Add dataset source
qqp_para["dataset"] = "QQP"

print("Total QQP paraphrases:", len(qqp_para))
print(qqp_para.head())

# Save clean QQP paraphrases
out_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\qqp_paraphrases.csv"
qqp_para.to_csv(out_path, index=False)

print("✅ QQP paraphrase file saved.")
