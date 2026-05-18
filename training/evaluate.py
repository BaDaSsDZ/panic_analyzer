"""
Evaluate the trained model on the held-out test set.
Prints per-tag precision/recall/F1 so you know which tags need more data.

Run: python -m training.evaluate
"""

import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.metrics import classification_report, f1_score

from data.dataset import PanicTagDataset, load_split
from model.classifier import PanicTagClassifier

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR       = Path(os.getenv("DATA_DIR", "./data/output"))
MODEL_SAVE_DIR = Path(os.getenv("MODEL_SAVE_DIR", "./model/saved"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", 16))
MAX_SEQ_LEN    = int(os.getenv("MAX_SEQ_LENGTH", 512))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()

    meta_path = MODEL_SAVE_DIR / "training_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("No trained model found. Run training/train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    threshold  = meta["threshold"]
    num_labels = meta["num_labels"]

    encoder_path = DATA_DIR / "label_encoder.json"
    with open(encoder_path, encoding="utf-8") as f:
        encoder = json.load(f)

    tag_names = [encoder["tag_metadata"][tid]["name"] for tid in encoder["classes"]]

    log.info("Loading model from %s", MODEL_SAVE_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_SAVE_DIR))
    model = PanicTagClassifier.from_pretrained(str(MODEL_SAVE_DIR), num_labels=num_labels)
    model.to(device)
    model.eval()

    test_texts, test_labels = load_split(DATA_DIR / "test.csv", num_labels)
    dataset = PanicTagDataset(test_texts, test_labels, tokenizer, MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    log.info("\n=== Test Set Results ===")
    log.info("Macro F1: %.4f  Micro F1: %.4f", macro_f1, micro_f1)
    log.info("\n--- Per-tag breakdown ---")

    report = classification_report(
        all_labels, all_preds,
        target_names=tag_names,
        zero_division=0,
        output_dict=True
    )

    # Print sorted by F1 descending
    tag_results = [(name, report[name]) for name in tag_names if name in report]
    tag_results.sort(key=lambda x: x[1]["f1-score"], reverse=True)

    print(f"\n{'Tag':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)
    for name, scores in tag_results:
        support = int(scores["support"])
        if support == 0:
            continue
        print(
            f"{name:<40} {scores['precision']:>10.3f} {scores['recall']:>10.3f} "
            f"{scores['f1-score']:>10.3f} {support:>10}"
        )

    # Flag tags needing more data
    print("\n--- Tags needing more training data (F1 < 0.5) ---")
    for name, scores in tag_results:
        if scores["f1-score"] < 0.5 and int(scores["support"]) > 0:
            print(f"  {name}: F1={scores['f1-score']:.3f} support={int(scores['support'])}")


if __name__ == "__main__":
    main()
