"""
Preprocess extracted panic data into model-ready format.

Reads labeled_panics.csv and tags.csv, assembles the input text string
per panic, builds multi-hot label vectors, splits into train/val/test,
and saves processed_dataset.csv + label_encoder.json.

Run: python -m data.preprocess
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/output"))


def assemble_input_text(row):
    """
    Build a single input string from all panic signals.
    Structure matches the analysis: META → PROCEDURES → LOGS → COMMENTS → FORM
    """
    parts = []

    if row.get("meta_text"):
        parts.append(f"[META] {row['meta_text']}")

    if row.get("procedure_text"):
        parts.append(f"[PROCEDURES] {row['procedure_text']}")

    if row.get("log_text"):
        parts.append(f"[LOGS] {row['log_text']}")

    if row.get("comment_text"):
        parts.append(f"[COMMENTS] {row['comment_text']}")

    if row.get("form_text"):
        parts.append(f"[FORM] {row['form_text']}")

    return " ".join(parts)


def load_data():
    panics_path = DATA_DIR / "labeled_panics.csv"
    tags_path = DATA_DIR / "tags.csv"

    if not panics_path.exists():
        raise FileNotFoundError(f"Run data/extract.py first. Missing: {panics_path}")
    if not tags_path.exists():
        raise FileNotFoundError(f"Run data/extract.py first. Missing: {tags_path}")

    panics_df = pd.read_csv(panics_path)
    tags_df = pd.read_csv(tags_path)
    log.info("Loaded %d panics, %d tags", len(panics_df), len(tags_df))
    return panics_df, tags_df


def build_label_matrix(panics_df, tags_df):
    """Build multi-hot label matrix. Returns mlb and labels array."""
    # Parse tag_ids column (comma-separated UUIDs) into lists
    tag_id_lists = panics_df["tag_ids"].fillna("").apply(
        lambda x: [t.strip() for t in x.split(",") if t.strip()]
    )

    # Fit MultiLabelBinarizer on all active tag IDs so the matrix is stable
    all_tag_ids = tags_df["tag_id"].tolist()
    mlb = MultiLabelBinarizer(classes=all_tag_ids)
    mlb.fit([all_tag_ids])  # fit with all possible classes to fix ordering

    labels = mlb.transform(tag_id_lists)
    log.info("Label matrix shape: %s (panics x tags)", labels.shape)

    # Drop tags that appear in zero training samples
    tag_counts = labels.sum(axis=0)
    present_mask = tag_counts > 0
    if not present_mask.all():
        missing = [all_tag_ids[i] for i, v in enumerate(present_mask) if not v]
        log.warning("Tags with 0 training examples (will be excluded): %d", len(missing))

    return mlb, labels


def split_data(texts, labels, val_split=0.1, test_split=0.1, random_state=42):
    """
    Stratified-ish split. Falls back to using all data for train when the
    dataset is too small to produce non-empty val/test sets.
    """
    n = len(texts)
    test_size = int(n * test_split)
    val_size = int(n * val_split)

    # Not enough data to split — use everything for training
    if test_size == 0 or val_size == 0:
        log.warning(
            "Dataset too small to split (%d rows). Using all data for train; "
            "val and test will be copies of train. Add more panics for real evaluation.",
            n
        )
        return texts, labels, texts, labels, texts, labels

    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    test_idx  = indices[:test_size]
    val_idx   = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    return (
        [texts[i] for i in train_idx], labels[train_idx],
        [texts[i] for i in val_idx],   labels[val_idx],
        [texts[i] for i in test_idx],  labels[test_idx],
    )


def save_label_encoder(mlb, tags_df, output_dir):
    """Save tag ID → index mapping and tag metadata for inference."""
    tag_id_to_name = dict(zip(tags_df["tag_id"], tags_df["name"]))
    tag_id_to_advice = dict(zip(tags_df["tag_id"], tags_df["controller_advice"].fillna("")))

    encoder_data = {
        "classes": list(mlb.classes_),
        "tag_metadata": {
            tag_id: {
                "name": tag_id_to_name.get(tag_id, tag_id),
                "controller_advice": tag_id_to_advice.get(tag_id, ""),
            }
            for tag_id in mlb.classes_
        }
    }

    path = output_dir / "label_encoder.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encoder_data, f, indent=2, ensure_ascii=False)
    log.info("Saved label encoder → %s (%d classes)", path, len(mlb.classes_))


def save_split(texts, labels, split_name, output_dir):
    df = pd.DataFrame({
        "text": texts,
        "labels": [",".join(str(v) for v in row) for row in labels.tolist()]
    })
    path = output_dir / f"{split_name}.csv"
    df.to_csv(path, index=False)
    log.info("Saved %s → %s (%d rows)", split_name, path, len(df))


def main():
    val_split  = float(os.getenv("VALIDATION_SPLIT", 0.1))
    test_split = float(os.getenv("TEST_SPLIT", 0.1))

    panics_df, tags_df = load_data()

    log.info("Assembling input texts...")
    panics_df["input_text"] = panics_df.apply(assemble_input_text, axis=1)

    # Filter out rows with empty text (shouldn't happen, but be safe)
    panics_df = panics_df[panics_df["input_text"].str.strip().str.len() > 10]

    mlb, labels = build_label_matrix(panics_df, tags_df)

    texts = panics_df["input_text"].tolist()

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(
        texts, labels, val_split=val_split, test_split=test_split
    )

    log.info("Split: train=%d val=%d test=%d", len(train_texts), len(val_texts), len(test_texts))

    save_label_encoder(mlb, tags_df, DATA_DIR)
    save_split(train_texts, train_labels, "train", DATA_DIR)
    save_split(val_texts,   val_labels,   "val",   DATA_DIR)
    save_split(test_texts,  test_labels,  "test",  DATA_DIR)

    # Print a sample to verify the format looks right
    log.info("\n--- Sample input text (first panic) ---")
    log.info(texts[0][:600])


if __name__ == "__main__":
    main()
