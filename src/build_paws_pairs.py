import pandas as pd

# ---------- Paths ----------
pairs_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_train.tsv"
sent_path  = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\raw\paws_sentences.tsv"

# ---------- Load data ----------
pairs_df = pd.read_csv(pairs_path, sep="\t")
sent_df = pd.read_csv(
    sent_path,
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)

print("Pairs file columns:", pairs_df.columns)
print("Sentence file columns:", sent_df.columns)

# ---------- Keep only what we need ----------
pairs_df = pairs_df[['mapping1', 'mapping2']]

sent_df = sent_df[['id', 'sentence1', 'sentence2']]

# ---------- Create a lookup table ----------
sent_lookup = {}

for _, row in sent_df.iterrows():
    sent_lookup[row['id']] = (row['sentence1'], row['sentence2'])

# ---------- Build real sentence pairs ----------
text_a = []
text_b = []

missing = 0

for _, row in pairs_df.iterrows():
    id1 = row['mapping1']
    id2 = row['mapping2']

    if id1 in sent_lookup and id2 in sent_lookup:
        s1 = sent_lookup[id1][0]
        s2 = sent_lookup[id2][1]
        text_a.append(s1)
        text_b.append(s2)
    else:
        missing += 1

paws_clean = pd.DataFrame({
    "text_a": text_a,
    "text_b": text_b
})

print("Final PAWS pairs:", len(paws_clean))
print("Missing mappings:", missing)
print(paws_clean.head())

# ---------- Save result ----------
out_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\paws_clean.csv"
paws_clean.to_csv(out_path, index=False)

print("✅ Clean PAWS dataset saved.")
