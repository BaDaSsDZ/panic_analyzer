"""
Fine-tune DistilBERT on the CASI panic tagging dataset.

Run: python -m training.train

Saves the best checkpoint (by val macro-F1) to MODEL_SAVE_DIR.
Works on CPU, CUDA, and Apple MPS (Mac). Windows CUDA supported.
"""

import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertTokenizerFast
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from data.dataset import PanicTagDataset, load_split
from model.classifier import PanicTagClassifier

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR       = Path(os.getenv("DATA_DIR", "./data/output"))
MODEL_SAVE_DIR = Path(os.getenv("MODEL_SAVE_DIR", "./model/saved"))
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL     = "distilbert-base-uncased"
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", 16))
EPOCHS         = int(os.getenv("EPOCHS", 5))
LR             = float(os.getenv("LEARNING_RATE", 2e-5))
MAX_SEQ_LEN    = int(os.getenv("MAX_SEQ_LENGTH", 512))
THRESHOLD      = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Using CPU (training will be slow — consider a GPU)")
    return device


def load_label_encoder():
    path = DATA_DIR / "label_encoder.json"
    if not path.exists():
        raise FileNotFoundError("Run data/preprocess.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, loader, device, threshold=THRESHOLD):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].numpy()

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1   = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    precision  = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall     = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "macro_f1":  macro_f1,
        "micro_f1":  micro_f1,
        "precision": precision,
        "recall":    recall,
    }


def main():
    device = get_device()

    encoder = load_label_encoder()
    num_labels = len(encoder["classes"])
    log.info("Loaded label encoder: %d tags", num_labels)

    log.info("Loading tokenizer and datasets...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    train_texts, train_labels = load_split(DATA_DIR / "train.csv", num_labels)
    val_texts,   val_labels   = load_split(DATA_DIR / "val.csv",   num_labels)

    log.info("Train: %d  Val: %d", len(train_texts), len(val_texts))

    train_dataset = PanicTagDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LEN)
    val_dataset   = PanicTagDataset(val_texts,   val_labels,   tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log.info("Loading DistilBERT...")
    config = DistilBertConfig.from_pretrained(BASE_MODEL)
    config.num_labels = num_labels
    model = PanicTagClassifier.from_pretrained(BASE_MODEL, config=config, num_labels=num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
    )

    best_f1 = 0.0
    best_epoch = 0

    log.info("Starting training: %d epochs, batch=%d, lr=%s", EPOCHS, BATCH_SIZE, LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device)

        log.info(
            "Epoch %d/%d  loss=%.4f  macro_f1=%.4f  micro_f1=%.4f  prec=%.4f  rec=%.4f",
            epoch, EPOCHS, avg_loss,
            metrics["macro_f1"], metrics["micro_f1"],
            metrics["precision"], metrics["recall"]
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            # Save model + tokenizer
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            # Save training metadata
            meta = {
                "best_epoch":   best_epoch,
                "best_macro_f1": best_f1,
                "num_labels":   num_labels,
                "threshold":    THRESHOLD,
                "base_model":   BASE_MODEL,
                "epochs_trained": EPOCHS,
                "batch_size":   BATCH_SIZE,
                "learning_rate": LR,
                "max_seq_length": MAX_SEQ_LEN,
            }
            with open(MODEL_SAVE_DIR / "training_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            log.info("  *** New best — saved checkpoint (macro_f1=%.4f) ***", best_f1)

    log.info("Training complete. Best epoch: %d  Best macro-F1: %.4f", best_epoch, best_f1)
    log.info("Model saved to: %s", MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
