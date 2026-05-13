import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\DS301Project\data\kagan_all_years.csv"
OUTPUT_DIR = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\DS301Project\models\kagan_bert"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
EARLY_STOP_N = 3
WARMUP_RATIO = 0.1
TRAIN_YEARS = list(range(2010, 2017))
VAL_YEARS = [2017, 2018]
TEST_YEARS = [2019]
CLASS_WEIGHTS = torch.tensor([1.58, 1.0])
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class KaganDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = df.reset_index(drop=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row = self.examples.iloc[idx]

        context = str(row["preceding_context"]).strip()
        utterance = str(row["justice_utterance"]).strip()
        label = int(row["label"])

        encoding = self.tokenizer(
            context,
            utterance,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_splits(data_path: str):
    df = pd.read_csv(data_path)
    df = df[df["label"] != -1].copy()
    df["label"] = df["label"].astype(int)

    df["year"] = df["case_id"].str.split("_").str[0].astype(int)

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df = df[df["year"].isin(VAL_YEARS)].copy()
    test_df = df[df["year"].isin(TEST_YEARS)].copy()

    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")
    print(f"Train label dist:\n{train_df['label'].value_counts(normalize=True).round(3)}")

    return train_df, val_df, test_df


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, val_df, test_df = load_splits(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    train_dataset = KaganDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = KaganDataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = KaganDataset(test_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model")

    print("\nStarting training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  → New best val acc: {best_val_acc:.4f} — model saved")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement ({epochs_no_improve}/{EARLY_STOP_N})")
            if epochs_no_improve >= EARLY_STOP_N:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nLoading best model for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["respondent", "petitioner"]))


if __name__ == "__main__":
    main()