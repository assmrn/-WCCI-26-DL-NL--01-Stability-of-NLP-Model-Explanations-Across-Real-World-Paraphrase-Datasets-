import torch
from transformers import BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = r"D:\research_sprint\models\bert_baseline_checkpoints\bert_epoch1_batch45000_step45000_valloss0.2471_20260118_113013.pt"
FINAL_MODEL_PATH = "models/final_baseline_bert.pt"

print("Loading checkpoint...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

torch.save(model.state_dict(), FINAL_MODEL_PATH)

print("✅ Final baseline model saved to:", FINAL_MODEL_PATH)
