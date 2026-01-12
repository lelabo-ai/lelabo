# lab/models/bert_glue.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class HFSequenceClassifier(nn.Module):
    """
    Wrapper HF:
      - forward(**batch) -> outputs HF (comme d'habitude)
      - forward(**batch, return_cache=True) -> (outputs, {"hidden_states": ...}) si dispo
    """
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.hf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, **batch):
        # on ne peut pas ajouter un arg non présent dans la signature HF facilement ici,
        # donc on lit return_cache à la main:
        return_cache = bool(batch.pop("return_cache", False))

        if return_cache:
            outputs = self.hf(**batch, output_hidden_states=True, return_dict=True)
            cache: Dict[str, Any] = {}
            if getattr(outputs, "hidden_states", None) is not None:
                cache["hidden_states"] = outputs.hidden_states
            return outputs, cache

        return self.hf(**batch)


def build_bert_for_glue(model_name: str, num_labels: int):
    # si tu veux garder exactement l'ancien comportement, retourne AutoModel...
    # mais là le wrapper est plus pratique pour généraliser.
    return HFSequenceClassifier(model_name, num_labels)
