"""
train_option1.py
Fine-tune InLegalBERT (law-ai/InLegalBERT) for binary vote prediction.

Input  : data/extracted.csv  (one row = one justice × case pair)
Split  : temporal — train < 2010, val 2010–2015, test >= 2015
Output : checkpoints/option1/best_model/   (HuggingFace format)
         checkpoints/option1/test_predictions.csv

Usage:
    python train_option1.py                        # defaults
    python train_option1.py --batch_size 8 --lr 3e-5
"""

import os
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_fscore_support,
)

# ── reproducibility ────────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── dataset ────────────────────────────────────────────────────────────────────

class SCOTUSDataset(Dataset):
    """
    Pre-tokenizes the all_utterances text column.
    Truncation to max_length=512 handles overlong inputs gracefully.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        texts = df["all_utterances"].fillna("").tolist()
        self.labels = df["label"].tolist()
        # tokenize entire split up-front; truncation silently handles >512 tokens
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": acc,
        "f1": macro_f1,
        "f1_class0": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "f1_class1": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
    }


# ── split summary ──────────────────────────────────────────────────────────────

def print_split_summary(splits: dict) -> None:
    print("\n" + "=" * 65)
    print("TEMPORAL SPLIT SUMMARY")
    print("=" * 65)
    for name, df in splits.items():
        vc = df["label"].value_counts().sort_index()
        n = len(df)
        print(
            f"  {name:<6}: {n:6,} rows  |  "
            f"label=0 (respondent): {vc.get(0, 0):,} ({vc.get(0, 0)/n*100:.1f}%)  |  "
            f"label=1 (petitioner): {vc.get(1, 0):,} ({vc.get(1, 0)/n*100:.1f}%)"
        )
    print("=" * 65 + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)

    # ── load & clean data ──────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path} ...")
    df = pd.read_csv(args.data_path)
    df = df[df["label"].isin([0, 1])].copy()
    df["all_utterances"] = df["all_utterances"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "label"]).copy()
    df["year"] = df["year"].astype(int)
    print(f"Loaded {len(df):,} valid rows. Year range: {df['year'].min()}–{df['year'].max()}")

    # ── temporal split ─────────────────────────────────────────────────────────
    train_df = df[df["year"] < 2010].reset_index(drop=True)
    val_df   = df[(df["year"] >= 2010) & (df["year"] < 2015)].reset_index(drop=True)
    test_df  = df[df["year"] >= 2015].reset_index(drop=True)

    print_split_summary({"Train (<2010)": train_df,
                          "Val  (2010–14)": val_df,
                          "Test (>=2015)": test_df})

    # ── tokenizer & model ──────────────────────────────────────────────────────
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # ── datasets ───────────────────────────────────────────────────────────────
    print("Tokenizing splits (this may take a moment) ...")
    train_dataset = SCOTUSDataset(train_df, tokenizer, args.max_length)
    val_dataset   = SCOTUSDataset(val_df,   tokenizer, args.max_length)
    test_dataset  = SCOTUSDataset(test_df,  tokenizer, args.max_length)

    # ── training arguments ─────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=4,
        save_total_limit=2,
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Starting training ...")
    trainer.train()

    # ── test evaluation ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST SET EVALUATION")
    print("=" * 65)

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_preds  = np.argmax(test_logits, axis=-1)
    test_labels = test_df["label"].tolist()

    probs = torch.softmax(torch.tensor(test_logits, dtype=torch.float32), dim=-1).numpy()

    print(classification_report(
        test_labels, test_preds,
        target_names=["Respondent (0)", "Petitioner (1)"],
        digits=4,
    ))

    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    acc       = accuracy_score(test_labels, test_preds)
    print(f"Overall  —  accuracy: {acc:.4f}  |  macro-F1: {macro_f1:.4f}")

    # ── save test predictions ──────────────────────────────────────────────────
    pred_df = test_df[["case_id", "justice_id", "year", "label"]].copy()
    pred_df["pred_label"]  = test_preds
    pred_df["prob_class0"] = probs[:, 0]
    pred_df["prob_class1"] = probs[:, 1]
    pred_df["correct"]     = (pred_df["pred_label"] == pred_df["label"]).astype(int)

    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nTest predictions saved → {pred_path}")

    # ── save best model ────────────────────────────────────────────────────────
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved → {best_model_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune InLegalBERT for SCOTUS vote prediction")
    parser.add_argument("--data_path",  default="data/extracted.csv",
                        help="Path to extracted.csv")
    parser.add_argument("--model_name", default="law-ai/InLegalBERT",
                        help="HuggingFace model identifier")
    parser.add_argument("--output_dir", default="checkpoints/option1",
                        help="Directory for checkpoints and outputs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length (utterances truncated to this)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device training batch size")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="Maximum training epochs")
    parser.add_argument("--lr",         type=float, default=2e-5,
                        help="Peak learning rate")
    parser.add_argument("--patience",   type=int, default=3,
                        help="Early stopping patience (epochs without val-F1 improvement)")
    args = parser.parse_args()
    main(args)
