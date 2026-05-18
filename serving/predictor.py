"""
Loads the trained model and runs inference.
Singleton — model is loaded once at startup.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from transformers import DistilBertTokenizerFast

from model.classifier import PanicTagClassifier

log = logging.getLogger(__name__)


class PanicTagPredictor:
    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        meta_path  = model_path / "training_meta.json"
        enc_path   = model_path.parent.parent / "data" / "output" / "label_encoder.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}. Run training/train.py first.")

        with open(meta_path) as f:
            self.meta = json.load(f)

        with open(enc_path, encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.num_labels = self.meta["num_labels"]
        self.threshold  = self.meta["threshold"]
        self.classes    = self.encoder["classes"]
        self.tag_meta   = self.encoder["tag_metadata"]
        self.model_version = f"v1-epoch{self.meta['best_epoch']}-f1{self.meta['best_macro_f1']:.3f}"

        # Device selection — works on Windows CUDA, Mac MPS, CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        log.info("Loading model on %s...", self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        self.model = PanicTagClassifier.from_pretrained(str(model_path), num_labels=self.num_labels)
        self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded: %s (%d tags)", self.model_version, self.num_labels)

    def predict(self, text: str, threshold: float = None) -> List[Dict]:
        """
        Run inference on assembled panic text.
        Returns list of {tag_id, tag_name, controller_advice, confidence} sorted by confidence desc.
        """
        if threshold is None:
            threshold = self.threshold

        encoding = self.tokenizer(
            text,
            max_length=self.meta["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            probs = torch.sigmoid(out["logits"]).cpu().numpy()[0]

        results = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                tag_id = self.classes[i]
                meta   = self.tag_meta.get(tag_id, {})
                results.append({
                    "tag_id":            tag_id,
                    "tag_name":          meta.get("name", tag_id),
                    "controller_advice": meta.get("controller_advice", ""),
                    "confidence":        round(float(prob), 4),
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results


# Global singleton — loaded once when the server starts
_predictor: PanicTagPredictor = None


def get_predictor(model_dir: str) -> PanicTagPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PanicTagPredictor(model_dir)
    return _predictor
