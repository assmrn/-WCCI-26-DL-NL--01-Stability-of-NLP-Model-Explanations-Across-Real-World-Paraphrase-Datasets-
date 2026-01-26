import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ParaphraseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["text_a"],
            row["text_b"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }

        return item


# ---------- Quick test ----------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = ParaphraseDataset(
        csv_path=r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\train.csv",
        tokenizer=tokenizer
    )

    print("Dataset size:", len(dataset))
    sample = dataset[0]

    print("Keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["labels"])
