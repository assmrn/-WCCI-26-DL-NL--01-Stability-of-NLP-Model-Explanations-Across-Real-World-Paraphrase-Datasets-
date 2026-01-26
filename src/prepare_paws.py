import pandas as pd

paws_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_clean.tsv"

# Load PAWS
paws = pd.read_csv(paws_path, sep="\t")

# Keep only true paraphrases
paws_para = paws[paws["label"] == 1]

# Select and rename columns
paws_para = paws_para[["sentence1", "sentence2"]]
paws_para.columns = ["text_a", "text_b"]

# Add dataset source
paws_para["dataset"] = "PAWS"

print("Total PAWS paraphrases:", len(paws_para))
print(paws_para.head())

# Save clean PAWS paraphrases
out_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\paws_paraphrases.csv"
paws_para.to_csv(out_path, index=False)

print("✅ PAWS paraphrase file saved.")
