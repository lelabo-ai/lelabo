# lab/models/bert.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec

from .registry import register_model, ModelContext


def _require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Le module 'transformers' n'est pas disponible. "
            "Installe-le (pip install transformers) pour utiliser bert.py."
        ) from e


def _get_base_model(hf_model: nn.Module) -> nn.Module:
    # HuggingFace convention: base_model_prefix (e.g. "bert", "roberta", "deberta", ...)
    prefix = getattr(hf_model, "base_model_prefix", None)
    if prefix and hasattr(hf_model, prefix):
        return getattr(hf_model, prefix)
    if hasattr(hf_model, "get_base_model"):
        try:
            return hf_model.get_base_model()
        except Exception:
            pass
    return hf_model


def _find_embeddings(base: nn.Module) -> Optional[nn.Module]:
    for name in ("embeddings", "embed_tokens"):
        if hasattr(base, name):
            return getattr(base, name)
    return None


def _find_encoder_layers(base: nn.Module) -> List[nn.Module]:
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        return list(base.encoder.layer)

    if hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        return list(base.transformer.layer)

    if hasattr(base, "layers"):
        try:
            return list(base.layers)
        except Exception:
            pass

    return []


def _find_classifier_head(hf_model: nn.Module) -> nn.Module:
    for name in ("classifier", "score", "lm_head"):
        if hasattr(hf_model, name):
            return getattr(hf_model, name)
    if hasattr(hf_model, "pre_classifier"):
        return getattr(hf_model, "pre_classifier")
    raise ValueError("Impossible de trouver un head de classification dans le modèle HF.")


class HFSequenceClassifier(BlockModel):
    """
    Wrapper générique HF (AutoModelForSequenceClassification) avec exposition des blocks.

    - forward(**batch) : comportement HF standard (outputs HF)
    - forward(**batch, return_cache=True) :
        -> (outputs, cache) où cache contient:
           cache["hidden_states"] (tuple) si dispo
           cache["block_inputs"][<block_name>] (tensor) (best-effort)
           cache["head_input"] (tensor) (best-effort)

    Remarques:
      - On appelle HF avec output_hidden_states=True quand return_cache=True.
      - Les "block_inputs" pour encoder.layer{i} viennent de hidden_states[i].
      - Pour "embeddings", on met input_ids (ou ids-like) car l'entrée n'est pas un tensor continu.
    """

    def __init__(self, model_name: str, num_labels: int, *, trust_remote_code: bool = False):
        super().__init__()
        _require_transformers()
        from transformers import AutoModelForSequenceClassification

        self.model_name = str(model_name)
        self.num_labels = int(num_labels)

        self.hf = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            trust_remote_code=bool(trust_remote_code),
        )

        self._base = _get_base_model(self.hf)
        self._emb = _find_embeddings(self._base)
        self._layers = _find_encoder_layers(self._base)
        self._head = _find_classifier_head(self.hf)
        # compat helpers for existing update rules
        self.config = getattr(self.hf, "config", None)
        if not hasattr(self, "bert"):
            self.bert = self._base

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []

        if self._emb is not None:
            blocks.append(BlockSpec(name="embeddings", module=self._emb, rep="cls", is_output=False))

        for i, layer in enumerate(self._layers):
            blocks.append(BlockSpec(name=f"encoder.layer{i}", module=layer, rep="cls", is_output=False))

        blocks.append(BlockSpec(name="head", module=self._head, rep="cls", is_output=True))
        return blocks

    def forward(self, **batch):
        return_cache = bool(batch.pop("return_cache", False))

        if not return_cache:
            return self.hf(**batch)

        outputs = self.hf(**batch, output_hidden_states=True, return_dict=True)

        cache: Dict[str, Any] = {"block_inputs": {}}

        hs = getattr(outputs, "hidden_states", None)
        if hs is not None:
            cache["hidden_states"] = hs

        # embeddings input: store ids-like tensor(s) if present
        if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
            cache["block_inputs"]["embeddings"] = batch["input_ids"]
        elif "inputs_embeds" in batch and torch.is_tensor(batch["inputs_embeds"]):
            cache["block_inputs"]["embeddings"] = batch["inputs_embeds"]

        if hs is not None:
            # hs[0] = output embeddings = input to encoder.layer0
            for i in range(min(len(self._layers), len(hs) - 1)):
                cache["block_inputs"][f"encoder.layer{i}"] = hs[i]

            # head input: best-effort CLS from last hidden state
            try:
                last = hs[-1]
                if torch.is_tensor(last) and last.dim() >= 3:
                    cache["head_input"] = last[:, 0]
                    cache["block_inputs"]["head"] = cache["head_input"]
            except Exception:
                pass
        else:
            last = getattr(outputs, "last_hidden_state", None)
            if torch.is_tensor(last) and last.dim() >= 3:
                cache["head_input"] = last[:, 0]
                cache["block_inputs"]["head"] = cache["head_input"]

        return outputs, cache


def build_bert_for_glue(model_name: str, num_labels: int, *, trust_remote_code: bool = False):
    """Factory compatible GLUE/NLP: retourne un BlockModel HFSequenceClassifier."""
    return HFSequenceClassifier(model_name=model_name, num_labels=num_labels, trust_remote_code=trust_remote_code)


@register_model("hf")
def build_hf_model(ctx: ModelContext, args):
    model_name = getattr(args, "hf_model", None) or "bert-base-uncased"
    trust_remote_code = bool(getattr(args, "hf_trust_remote_code", False))
    return HFSequenceClassifier(model_name=model_name, num_labels=ctx.num_classes, trust_remote_code=trust_remote_code)


@register_model("bert")
def build_bert_model(ctx: ModelContext, args):
    return build_hf_model(ctx, args)
