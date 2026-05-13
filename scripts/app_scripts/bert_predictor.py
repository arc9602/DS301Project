import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

BASE_MODEL = "lexlms/legal-longformer-base"
LORA_R = 4
LORA_ALPHA = 8
MAX_LENGTH = 2048

class DualEmbeddingClassifier(nn.Module):
    def __init__(self, encoder, hidden_size: int = 768, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def mean_pool(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(self, pet_input_ids, pet_attention_mask, res_input_ids, res_attention_mask):
        v_pet = self.mean_pool(pet_input_ids, pet_attention_mask)
        v_res = self.mean_pool(res_input_ids, res_attention_mask)
        combined = torch.cat([v_pet, v_res], dim=-1)
        combined = self.dropout(combined)
        return self.classifier(combined)


def load_bert_model(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_encoder = AutoModel.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.1, target_modules=["query", "value"], bias="none")
    encoder = get_peft_model(base_encoder, lora_config)
    encoder.load_adapter(model_path, adapter_name="default")

    model = DualEmbeddingClassifier(encoder)

    classifier_path = os.path.join(model_path, "classifier.pt")
    state = torch.load(classifier_path, map_location="cpu")
    classifier_keys = {k: v for k, v in state.items() if k in ("weight", "bias")}
    head_keys = {k.removeprefix("classifier."): v for k, v in state.items() if k.startswith("classifier.")}
    if classifier_keys:
        model.classifier.load_state_dict(classifier_keys)
    elif head_keys:
        model.classifier.load_state_dict(head_keys)
    else:
        model.classifier.load_state_dict(state, strict=False)

    model.eval()
    return tokenizer, model


def bert_predict_court(tokenizer, model, df: pd.DataFrame) -> dict | None:
    pet_rows = df[df["side_addressed"] == 1]   
    res_rows = df[df["side_addressed"] == 0]   

    if len(pet_rows) == 0 and len(res_rows) == 0:
        return None

    def build_text(rows):
        if len(rows) == 0:
            return "[NO EXCHANGES]"
        return " [SEP] ".join(
            str(r["preceding_context"]) + " </s> " + str(r["justice_utterance"])
            for _, r in rows.iterrows()
        )

    pet_enc = tokenizer(build_text(pet_rows), max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    res_enc = tokenizer(build_text(res_rows), max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(pet_enc["input_ids"], pet_enc["attention_mask"], res_enc["input_ids"], res_enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
        prediction = logits.argmax(dim=-1).item()
        confidence = probs[prediction].item()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "prob_pet": probs[1].item(),
        "prob_res": probs[0].item(),
    }


def bert_predict_justice(tokenizer, model, df: pd.DataFrame, justice_id: str) -> dict | None:
    group = df[df["justice_id"] == justice_id]
    pet_rows = group[group["side_addressed"] == 1]   # petitioner
    res_rows = group[group["side_addressed"] == 0]   # respondent

    if len(pet_rows) == 0 and len(res_rows) == 0:
        return None

    def build_text(rows):
        if len(rows) == 0:
            return "[NO EXCHANGES]"
        return " [SEP] ".join(str(r["preceding_context"]) + " </s> " + str(r["justice_utterance"]) for _, r in rows.iterrows())

    pet_text = build_text(pet_rows)
    res_text = build_text(res_rows)

    pet_enc = tokenizer(pet_text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    res_enc = tokenizer(res_text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(pet_enc["input_ids"], pet_enc["attention_mask"], res_enc["input_ids"], res_enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
        prediction = logits.argmax(dim=-1).item()
        confidence = probs[prediction].item()

    return {
        "justice_id": justice_id,
        "prediction": prediction,
        "confidence": confidence,
        "prob_pet": probs[1].item(),
        "prob_res": probs[0].item(),
    }