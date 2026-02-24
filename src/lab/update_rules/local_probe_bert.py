# lab/update_rules/local_probe_bert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import UpdateRule
from ..core.steps import metric_payload_from_outputs

class LocalProbeBERT(UpdateRule):
    """
    Local probe learning for BERT-like models (e.g., BertForSequenceClassification).

    Updates:
      - Each encoder layer (block) using its own probe on CLS
      - The classifier head using final CLS (detached)
    Does NOT update embeddings unless you add an embedding probe (can be added easily).
    """
    def __init__(self, base_optimizer, probe_lr=1e-3, probe_optim_cls=torch.optim.AdamW):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.probe_lr = probe_lr
        self.probe_optim_cls = probe_optim_cls

        self.probes = None
        self.probe_optim = None
        self.is_regression = False
        self.num_labels = None

    def on_train_start(self, model, task, device, state=None):
        # task is your GLUETask (below). We'll read regression flag + label count.
        self.is_regression = getattr(task, "is_regression", False)
        self.num_labels = getattr(task, "num_labels", None)

        if self.num_labels is None:
            # fallback from HF config
            self.num_labels = int(getattr(model.config, "num_labels", 2))

        hidden = int(model.config.hidden_size)
        n_layers = int(model.config.num_hidden_layers)

        # One probe per transformer block
        self.probes = nn.ModuleList([
            nn.Linear(hidden, 1 if self.is_regression else self.num_labels)
            for _ in range(n_layers)
        ]).to(device)

        self.probe_optim = self.probe_optim_cls(self.probes.parameters(), lr=self.probe_lr, weight_decay=0.0)

    def _loss(self, logits, labels):
        if self.is_regression:
            # labels in STS-B are floats
            labels = labels.to(dtype=logits.dtype)
            return F.mse_loss(logits.squeeze(-1), labels)
        return F.cross_entropy(logits, labels)

    def train_step(self, model, task, batch, device, state=None):
        model.train()

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        self.base_optimizer.zero_grad(set_to_none=True)
        self.probe_optim.zero_grad(set_to_none=True)

        # ---- 1) Get hidden states WITHOUT building a giant graph ----
        # hidden_states: tuple length L+1: [emb_out, layer1_out, ..., layerL_out]
        with torch.no_grad():
            bert_out = model.bert(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = bert_out.hidden_states

        # Build extended attention mask (same as HF internals)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            # rare, but keep safe
            input_shape = inputs["input_ids"].shape
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=device)

        input_shape = inputs["input_ids"].shape
        try:
            extended_mask = model.get_extended_attention_mask(attention_mask, input_shape)
        except TypeError:
            extended_mask = model.get_extended_attention_mask(attention_mask, input_shape, device=device)

        # ---- 2) Local block updates ----
        encoder = model.bert.encoder
        n_layers = len(encoder.layer)

        total_local_loss = 0.0
        for l in range(n_layers):
            # input to block l is hidden_states[l]
            h_in = hidden_states[l].detach()

            # Recompute ONLY block l with grads enabled
            # BertLayer returns tuple (layer_output, ...)
            h_out = encoder.layer[l](h_in, attention_mask=extended_mask)[0]

            cls = h_out[:, 0, :]                  # [B, H]
            probe_logits = self.probes[l](cls)    # [B, C] or [B,1]
            loss_l = self._loss(probe_logits, labels)

            loss_l.backward()
            total_local_loss += float(loss_l.item())

        # ---- 3) Update classifier head locally from final CLS (detached) ----
        h_final = hidden_states[-1].detach()      # output of last block
        cls_final = h_final[:, 0, :]

        x = cls_final
        if hasattr(model, "dropout") and model.dropout is not None:
            x = model.dropout(x)
        head_logits = model.classifier(x)

        head_loss = self._loss(head_logits, labels)
        head_loss.backward()

        # ---- 4) Step params ----
        self.base_optimizer.step()
        self.probe_optim.step()

        # ---- stats (for tqdm postfix + epoch summary) ----
        stats = {"loss": float(head_loss.item())}

        if (not self.is_regression) and head_logits is not None:
            with torch.no_grad():
                pred = head_logits.argmax(dim=-1)
                acc = (pred == labels).float().mean().item()
            stats["acc"] = acc
        stats.update(metric_payload_from_outputs(head_logits, labels))

        # optional: log average local probe loss too
        stats["local_probe_loss"] = total_local_loss / max(1, n_layers)

        self.global_step += 1
        return stats
