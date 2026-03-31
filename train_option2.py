"""
train_option2.py
Hierarchical Transformer for SCOTUS vote prediction.

Architecture:
  Level 1 — InLegalBERT encodes each individual utterance → [CLS] embedding
  Level 2 — Small Transformer encoder (4 heads, 2 layers) over utterance sequence
  Level 3 — Linear classifier → binary prediction

Training strategy:
  Epochs 1–2 : InLegalBERT frozen  (only transformer encoder + classifier trained)
  Epoch  3+  : All weights unfrozen (joint fine-tuning with lower BERT lr)

Input  : data/extracted.csv
Split  : temporal — train < 2010, val 2010–2015, test >= 2015
Output : checkpoints/option2/best_model.pt
         checkpoints/option2/test_predictions.csv

Usage:
    python train_option2.py
    python train_option2.py --batch_size 4 --max_utt_len 128 --max_utterances 60
"""

import os
import math
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ── reproducibility ────────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── positional encoding ────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE for the utterance-level transformer."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── hierarchical model ─────────────────────────────────────────────────────────

class HierarchicalClassifier(nn.Module):
    """
    Two-stage classifier:
      1. InLegalBERT encodes each utterance → [CLS] token (768-d).
      2. Positional encoding + Transformer encoder over utterance sequence.
      3. Mean-pool over real utterances → linear classifier.
    """

    def __init__(
        self,
        bert_model_name: str,
        hidden_size: int = 768,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    # ── freeze / unfreeze BERT ─────────────────────────────────────────────────

    def freeze_bert(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = True

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, N, seq_len)
        attention_mask: torch.Tensor,  # (B, N, seq_len)
        utt_mask: torch.Tensor,        # (B, N)  bool — True = real utterance
        labels: torch.Tensor | None = None,
    ) -> dict:
        B, N, seq_len = input_ids.shape
        hidden_size = self.bert.config.hidden_size

        # flatten batch × utterance for BERT
        flat_ids  = input_ids.view(B * N, seq_len)
        flat_attn = attention_mask.view(B * N, seq_len)
        valid     = utt_mask.view(B * N)              # which rows are real utterances

        # encode only real (non-padding) utterances to save compute
        utt_embs = torch.zeros(B * N, hidden_size, device=input_ids.device)
        if valid.any():
            bert_out = self.bert(
                input_ids=flat_ids[valid],
                attention_mask=flat_attn[valid],
            )
            utt_embs[valid] = bert_out.last_hidden_state[:, 0, :]   # [CLS]

        utt_embs = utt_embs.view(B, N, hidden_size)  # (B, N, H)

        # positional encoding
        utt_embs = self.pos_enc(utt_embs)

        # transformer encoder — src_key_padding_mask: True = IGNORE
        pad_mask = ~utt_mask                          # (B, N)
        encoded  = self.transformer_encoder(utt_embs, src_key_padding_mask=pad_mask)

        # masked mean pool over real utterances
        mask_f  = utt_mask.float().unsqueeze(-1)     # (B, N, 1)
        pooled  = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(pooled)              # (B, 2)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# ── dataset ────────────────────────────────────────────────────────────────────

class HierarchicalDataset(Dataset):
    """
    Each sample = a list of utterances (split on |||).
    Each utterance is tokenized individually and truncated to max_utt_len.
    Utterances beyond max_utterances are silently dropped.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_utt_len: int = 128,
        max_utterances: int = 50,
    ):
        self.labels: list[int] = df["label"].tolist()
        self.samples: list[dict] = []

        for text in df["all_utterances"].fillna("").tolist():
            utts = [u.strip() for u in text.split("|||") if u.strip()]
            utts = utts[:max_utterances] if utts else [""]   # always ≥1 utterance

            enc = tokenizer(
                utts,
                truncation=True,
                max_length=max_utt_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.samples.append({
                "input_ids":      enc["input_ids"],      # (N, max_utt_len)
                "attention_mask": enc["attention_mask"],
            })

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "input_ids":      s["input_ids"],
            "attention_mask": s["attention_mask"],
            "n_utterances":   s["input_ids"].shape[0],
            "label":          self.labels[idx],
        }


# ── collate ────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Pad utterance count to max in this batch."""
    max_n   = max(item["n_utterances"] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]

    ids_list, attn_list, utt_mask_list, labels = [], [], [], []

    for item in batch:
        n     = item["n_utterances"]
        pad_n = max_n - n

        ids  = item["input_ids"]
        attn = item["attention_mask"]

        if pad_n > 0:
            ids  = torch.cat([ids,  torch.zeros(pad_n, seq_len, dtype=torch.long)], dim=0)
            attn = torch.cat([attn, torch.zeros(pad_n, seq_len, dtype=torch.long)], dim=0)

        utt_mask = torch.cat([
            torch.ones(n,     dtype=torch.bool),
            torch.zeros(pad_n, dtype=torch.bool),
        ])

        ids_list.append(ids)
        attn_list.append(attn)
        utt_mask_list.append(utt_mask)
        labels.append(item["label"])

    return {
        "input_ids":      torch.stack(ids_list),       # (B, max_n, seq_len)
        "attention_mask": torch.stack(attn_list),       # (B, max_n, seq_len)
        "utt_mask":       torch.stack(utt_mask_list),  # (B, max_n)
        "labels":         torch.tensor(labels, dtype=torch.long),
    }


# ── evaluation helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        ids   = batch["input_ids"].to(device)
        attn  = batch["attention_mask"].to(device)
        umask = batch["utt_mask"].to(device)
        lbls  = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(ids, attn, umask, lbls)

        total_loss += out["loss"].item()
        n_batches  += 1
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


# ── split summary ──────────────────────────────────────────────────────────────

def print_split_summary(splits: dict) -> None:
    print("\n" + "=" * 65)
    print("TEMPORAL SPLIT SUMMARY")
    print("=" * 65)
    for name, df in splits.items():
        vc = df["label"].value_counts().sort_index()
        n  = len(df)
        print(
            f"  {name:<6}: {n:6,} rows  |  "
            f"label=0 (respondent): {vc.get(0, 0):,} ({vc.get(0, 0)/n*100:.1f}%)  |  "
            f"label=1 (petitioner): {vc.get(1, 0):,} ({vc.get(1, 0)/n*100:.1f}%)"
        )
    print("=" * 65 + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── data ───────────────────────────────────────────────────────────────────
    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path)
    df = df[df["label"].isin([0, 1])].copy()
    df["all_utterances"] = df["all_utterances"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "label"]).copy()
    df["year"] = df["year"].astype(int)
    print(f"Loaded {len(df):,} rows. Year range: {df['year'].min()}–{df['year'].max()}")

    # ── temporal split ─────────────────────────────────────────────────────────
    train_df = df[df["year"] < 2010].reset_index(drop=True)
    val_df   = df[(df["year"] >= 2010) & (df["year"] < 2015)].reset_index(drop=True)
    test_df  = df[df["year"] >= 2015].reset_index(drop=True)

    print_split_summary({"Train (<2010)": train_df,
                          "Val  (2010–14)": val_df,
                          "Test (>=2015)": test_df})

    # ── tokenizer ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── datasets ───────────────────────────────────────────────────────────────
    print(f"Tokenizing (max_utt_len={args.max_utt_len}, max_utterances={args.max_utterances}) ...")
    train_dataset = HierarchicalDataset(train_df, tokenizer, args.max_utt_len, args.max_utterances)
    val_dataset   = HierarchicalDataset(val_df,   tokenizer, args.max_utt_len, args.max_utterances)
    test_dataset  = HierarchicalDataset(test_df,  tokenizer, args.max_utt_len, args.max_utterances)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # ── model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    model = HierarchicalClassifier(args.model_name).to(device)
    model.freeze_bert()
    print("InLegalBERT FROZEN for first 2 epochs.\n")

    # optimizer/scheduler — non-BERT params only while BERT is frozen
    non_bert_params = list(model.transformer_encoder.parameters()) + \
                      list(model.pos_enc.parameters()) + \
                      list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(non_bert_params, lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * len(train_loader) * 2)    # warmup over frozen phase
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1      = -1.0
    best_epoch       = 0
    patience_counter = 0
    bert_unfrozen    = False

    # ── training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # unfreeze BERT after epoch 2
        if epoch == 3 and not bert_unfrozen:
            model.unfreeze_bert()
            bert_unfrozen = True
            print("\n--- InLegalBERT UNFROZEN — joint fine-tuning begins ---\n")
            optimizer = torch.optim.AdamW([
                {"params": model.bert.parameters(),                "lr": args.lr * 0.1},
                {"params": model.pos_enc.parameters(),             "lr": args.lr},
                {"params": model.transformer_encoder.parameters(), "lr": args.lr},
                {"params": model.classifier.parameters(),          "lr": args.lr},
            ], weight_decay=0.01)
            remaining_steps = len(train_loader) * (args.epochs - 2)
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, remaining_steps)

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(device)
            attn  = batch["attention_mask"].to(device)
            umask = batch["utt_mask"].to(device)
            lbls  = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(ids, attn, umask, lbls)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += out["loss"].item()

            if (step + 1) % 100 == 0:
                avg = running_loss / (step + 1)
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | loss={avg:.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ── validate ──────────────────────────────────────────────────────────
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, use_amp)

        print(
            f"\nEpoch {epoch:2d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_counter = 0
            ckpt = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  --> Best model saved (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val F1={best_val_f1:.4f} at epoch {best_epoch}.")

    # ── test evaluation ────────────────────────────────────────────────────────
    print("Loading best checkpoint for test evaluation ...")
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    )
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, use_amp
    )

    print("\n" + "=" * 65)
    print("TEST SET EVALUATION")
    print("=" * 65)
    print(f"test_loss={test_loss:.4f}  |  test_acc={test_acc:.4f}  |  test_f1={test_f1:.4f}\n")
    print(classification_report(
        test_labels, test_preds,
        target_names=["Respondent (0)", "Petitioner (1)"],
        digits=4,
    ))

    # ── save test predictions ──────────────────────────────────────────────────
    pred_df = test_df[["case_id", "justice_id", "year", "label"]].copy()
    pred_df["pred_label"] = test_preds
    pred_df["correct"]    = (pred_df["pred_label"] == pred_df["label"]).astype(int)
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Test predictions saved → {pred_path}")

    # also save tokenizer config alongside checkpoint
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    print(f"Tokenizer saved → {args.output_dir}/tokenizer")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical Transformer for SCOTUS vote prediction"
    )
    parser.add_argument("--data_path",      default="data/extracted.csv")
    parser.add_argument("--model_name",     default="law-ai/InLegalBERT")
    parser.add_argument("--output_dir",     default="checkpoints/option2")
    parser.add_argument("--max_utt_len",    type=int,   default=128,
                        help="Max tokens per individual utterance")
    parser.add_argument("--max_utterances", type=int,   default=50,
                        help="Max utterances per sample (extras dropped)")
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--patience",       type=int,   default=3)
    args = parser.parse_args()
    main(args)
