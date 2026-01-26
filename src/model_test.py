import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from dataset_loader import ParaphraseDataset

def main():
    train_path = r"C:\Users\ANUJA\Desktop\VS codes\research_sprint\data\processed\train.csv"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = ParaphraseDataset(train_path, tokenizer)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    model.to(device)
    model.train()

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        labels=batch["labels"]
    )

    print("\nForward pass successful ✅")
    print("Loss:", outputs.loss.item())
    print("Logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
