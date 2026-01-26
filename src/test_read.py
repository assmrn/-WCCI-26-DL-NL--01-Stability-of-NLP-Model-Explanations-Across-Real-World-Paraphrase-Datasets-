import pandas as pd

paws = pd.read_csv(
    r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_train.tsv",
    sep="\t"
)

print(paws.columns)
print("\nFirst 5 rows:")
print(paws.head())


