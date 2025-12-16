from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""Smoke-test ProsusAI/finbert on zeroshot/twitter-financial-news-sentiment.

Dataset label convention (per dataset card):
  0 = Bearish
  1 = Bullish
  2 = Neutral

FinBERT convention (per model card):
  3-way sentiment: positive / negative / neutral

We DO NOT assume the logits index order; we read model.config.id2label.
"""

MODEL_NAME = "ProsusAI/finbert"
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
TEXT_COL_CANDIDATES = ("text", "sentence", "tweet", "content")
LABEL_COL_CANDIDATES = ("label", "labels", "sentiment")

# Dataset ids -> human labels
DS_ID2LABEL = {0: "bearish", 1: "bullish", 2: "neutral"}

# FinBERT labels -> dataset labels
# (Bearish~negative, Bullish~positive, Neutral~neutral)
FINBERT2DS = {"negative": "bearish", "positive": "bullish", "neutral": "neutral"}


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"Could not find any of {candidates} in dataset columns={columns}")


def main(sample_n: int = 2000, batch_size: int = 32, max_length: int = 128, device: str | None = None) -> None:
    ds = load_dataset(DATASET_NAME)
    split = "test" if "test" in ds else "validation" if "validation" in ds else "train"
    data = ds[split]

    text_col = _pick_column(list(data.column_names), TEXT_COL_CANDIDATES)
    label_col = _pick_column(list(data.column_names), LABEL_COL_CANDIDATES)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Inspect model output convention from config
    id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(model.config.id2label, dict) else dict(model.config.id2label)
    label2id = {v: int(k) for k, v in id2label.items()}
    print("FinBERT id2label:", id2label)
    print("FinBERT label2id:", label2id)

    # Sanity: ensure required keys exist (case-insensitive)
    normalized = {k: v.lower() for k, v in id2label.items()}
    inv_norm = {v: k for k, v in normalized.items()}
    for need in ("positive", "negative", "neutral"):
        if need not in inv_norm:
            raise ValueError(f"FinBERT config missing '{need}'. Found labels={set(normalized.values())}")

    n = min(sample_n, len(data))
    idx = np.random.RandomState(42).choice(len(data), size=n, replace=False)

    y_true = []
    y_pred = []

    def batches(arr, bs):
        for i in range(0, len(arr), bs):
            yield arr[i : i + bs]

    for b in batches(idx, batch_size):
        texts = [data[int(i)][text_col] for i in b]
        labels = [int(data[int(i)][label_col]) for i in b]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy().tolist()

        # Map FinBERT predicted class id -> finbert label -> dataset label
        for ds_lab, fin_id in zip(labels, pred_ids):
            fin_label = id2label[int(fin_id)].lower()
            ds_label_pred = FINBERT2DS.get(fin_label)
            if ds_label_pred is None:
                raise ValueError(f"Unexpected FinBERT label '{fin_label}' from id2label={id2label}")

            y_true.append(DS_ID2LABEL[int(ds_lab)])
            y_pred.append(ds_label_pred)

    # Simple metrics
    classes = ["bearish", "bullish", "neutral"]
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\nSplit={split}  n={len(y_true)}  accuracy={acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred\t" + "\t".join(classes)
    print(header)
    for t in classes:
        row = [str(cm[t][p]) for p in classes]
        print(t + "\t" + "\t".join(row))

    # Show a few examples
    print("\nExamples:")
    for i in range(5):
        print("-" * 80)
        # pick from the sampled set
        j = int(idx[i])
        text = data[j][text_col]
        true_lab = DS_ID2LABEL[int(data[j][label_col])]
        pred_lab = y_pred[i]
        print(f"text: {text}")
        print(f"true: {true_lab}  pred: {pred_lab}")


if __name__ == "__main__":
    # Adjust sample_n upward for a more stable estimate.
    main(sample_n=2000, batch_size=32, max_length=128)
