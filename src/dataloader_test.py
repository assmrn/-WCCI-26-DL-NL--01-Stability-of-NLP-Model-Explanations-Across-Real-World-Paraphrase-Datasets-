import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset_loader import ParaphraseDataset


def main():
    train_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\train.csv"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ParaphraseDataset(train_path, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,      # small batch for GPU safety
        shuffle=True,
        num_workers=0
    )

    print("Total batches:", len(train_loader))

    batch = next(iter(train_loader))

    print("\nBatch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("token_type_ids shape:", batch["token_type_ids"].shape)
    print("labels shape:", batch["labels"].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    batch = {k: v.to(device) for k, v in batch.items()}
    print("Batch successfully moved to GPU ✅")


if __name__ == "__main__":
    main()
