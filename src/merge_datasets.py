import pandas as pd

qqp_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\qqp_paraphrases.csv"
paws_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\paws_paraphrases.csv"

qqp = pd.read_csv(qqp_path)
paws = pd.read_csv(paws_path)

print("QQP size:", len(qqp))
print("PAWS size:", len(paws))

combined = pd.concat([qqp, paws], ignore_index=True)

print("Combined size:", len(combined))
print(combined.head())

out_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\final_paraphrases.csv"
combined.to_csv(out_path, index=False)

print("✅ Final merged dataset saved.")
