import pandas as pd

paws_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_clean.tsv"

paws = pd.read_csv(paws_path, sep="\t")

print("PAWS columns:")
print(paws.columns)

print("\nFirst 5 rows:")
print(paws.head())

print("\nLabel counts:")
print(paws["label"].value_counts())
