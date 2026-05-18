"""
PyTorch Dataset for multi-label panic tagging.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PanicTagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  # numpy array shape (N, num_tags)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def load_split(csv_path, num_labels):
    """Load a train/val/test CSV into texts and labels array."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = np.array(
        [list(map(int, row.split(","))) for row in df["labels"]],
        dtype=np.float32
    )
    assert labels.shape[1] == num_labels, (
        f"Label width mismatch: got {labels.shape[1]}, expected {num_labels}"
    )
    return texts, labels
