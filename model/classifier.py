"""
DistilBERT-based multi-label classifier for panic tagging.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel


class PanicTagClassifier(DistilBertPreTrainedModel):
    """
    DistilBERT with a sigmoid classification head.
    Outputs one probability per tag (multi-label, not mutually exclusive).
    """

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
