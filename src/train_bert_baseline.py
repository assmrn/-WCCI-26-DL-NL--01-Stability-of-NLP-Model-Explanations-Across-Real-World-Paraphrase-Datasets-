# ============================================================
# Phase 2 – Step 3.4.2
# Baseline BERT Fine-Tuning with Research-Grade Checkpointing
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from datetime import datetime


# ===============================
# CONFIGURATION
# ===============================

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8

EPOCHS = 1                     # ✅ ONLY 1 EPOCH (BASELINE)
LR = 2e-5
SAVE_EVERY_STEPS = 5000        # ✅ CHECKPOINT EVERY 5000 STEPS

TRAIN_PATH = r"D:\research_sprint\data\processed\train.csv"
VAL_PATH   = r"D:\research_sprint\data\processed\validation.csv"

CHECKPOINT_DIR = "models/bert_baseline_checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# DATASET
# ===============================

class ParaphraseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(
            row["text_a"],
            row["text_b"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)

        return item


# ===============================
# VALIDATION
# ===============================

def run_validation(model, val_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())

    model.train()
    return sum(losses) / len(losses)


# ===============================
# CHECKPOINTING
# ===============================

def save_checkpoint(state, epoch, batch_idx, global_step, val_loss=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tag = f"epoch{epoch}_batch{batch_idx}_step{global_step}"
    if val_loss is not None:
        tag += f"_valloss{val_loss:.4f}"

    filename = f"bert_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)

    torch.save(state, path)
    print(f"\nCheckpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    if "torch_rng" in ckpt and isinstance(ckpt["torch_rng"], torch.ByteTensor):
        torch.set_rng_state(ckpt["torch_rng"])
    else:
        print("⚠️ Torch RNG state not restored.")

    if "cuda_rng" in ckpt and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        except Exception as e:
            print("⚠️ CUDA RNG restore failed:", e)

    if "numpy_rng" in ckpt:
        try:
            np.random.set_state(ckpt["numpy_rng"])
        except:
            print("⚠️ NumPy RNG restore failed.")

    if "python_rng" in ckpt:
        try:
            random.setstate(ckpt["python_rng"])
        except:
            print("⚠️ Python RNG restore failed.")

    print(f"\nResumed from checkpoint: {path}")
    return ckpt["epoch"], ckpt["batch_idx"], ckpt["global_step"]


# ===============================
# TRAINING
# ===============================

def main(resume_checkpoint=None):

    print("Using device:", DEVICE)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = ParaphraseDataset(TRAIN_PATH, tokenizer, MAX_LEN)
    val_dataset   = ParaphraseDataset(VAL_PATH, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    start_epoch = 0
    resume_batch = 0
    global_step = 0

    if resume_checkpoint:
        start_epoch, resume_batch, global_step = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, DEVICE
        )

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        model.train()

        epoch_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, batch in epoch_bar:

            if epoch == start_epoch and batch_idx < resume_batch:
                continue

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_bar.set_postfix(loss=loss.item())

            if global_step % SAVE_EVERY_STEPS == 0:
                val_loss = run_validation(model, val_loader, DEVICE)

                save_checkpoint({
                    "epoch": epoch,
                    "batch_idx": batch_idx + 1,
                    "global_step": global_step,

                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),

                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate()

                }, epoch+1, batch_idx+1, global_step, val_loss)

        resume_batch = 0

    print("\nTraining complete.")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":

    main(resume_checkpoint=
         "models/bert_baseline_checkpoints/bert_epoch1_batch40000_step40000_valloss0.2415_20260117_160245.pt"
    )

#bert_epoch1_batch14000_step14000_valloss0.3377_20260116_224331.pt
#bert_epoch1_batch16000_step16000_valloss0.3259_20260117_124506.pt
#bert_epoch1_batch40000_step40000_valloss0.2415_20260117_160245.pt