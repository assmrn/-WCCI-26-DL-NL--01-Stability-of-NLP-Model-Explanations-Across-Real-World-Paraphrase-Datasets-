import pandas as pd

sent_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_sentences.tsv"

sent_df = pd.read_csv(
    sent_path,
    sep="\t",
    engine="python",     # more flexible parser
    on_bad_lines="skip"  # skip problematic lines
)

print("Columns in PAWS sentence file:")
print(sent_df.columns)

print("\nFirst 5 rows:")
print(sent_df.head())

print("\nTotal rows:", len(sent_df))
