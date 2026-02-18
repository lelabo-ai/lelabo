from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .bootstrap import ensure_lab_namespace

ensure_lab_namespace()

import torch
import torch.nn.functional as F

from lab.algorithms.update_rules.base import UpdateRule


_DEFAULT_EXPR: dict[str, Any] = {
    "t": "binary",
    "op": "outer",
    "a": {"t": "term", "name": "u"},
    "b": {"t": "term", "name": "x"},
}


@dataclass(frozen=True)
class EvolvedRuleParams:
    local_lr: float = 1e-3
    local_weight_decay: float = 0.0
    head_lr: float = 1e-3
    head_weight_decay: float = 0.0
    normalize_update: bool = False
    update_bias: bool = True
    update_expr: dict[str, Any] | None = None
    bias_expr: dict[str, Any] | None = None
    local_phase_epochs: int = 10
    head_phase_epochs: int = 40

    @staticmethod
    def from_genome(genome: dict[str, Any]) -> "EvolvedRuleParams":
        update_expr = genome.get("update_expr", _DEFAULT_EXPR)
        bias_expr = genome.get("bias_expr")
        if not isinstance(update_expr, dict):
            update_expr = copy.deepcopy(_DEFAULT_EXPR)
        if bias_expr is not None and not isinstance(bias_expr, dict):
            bias_expr = None
        return EvolvedRuleParams(
            local_lr=float(genome.get("local_lr", 1e-3)),
            local_weight_decay=float(genome.get("local_weight_decay", 0.0)),
            head_lr=float(genome.get("head_lr", 1e-3)),
            head_weight_decay=float(genome.get("head_weight_decay", 0.0)),
            normalize_update=bool(genome.get("normalize_update", False)),
            update_bias=bool(genome.get("update_bias", True)),
            update_expr=copy.deepcopy(update_expr),
            bias_expr=copy.deepcopy(bias_expr) if isinstance(bias_expr, dict) else None,
            local_phase_epochs=int(genome.get("local_phase_epochs", 10)),
            head_phase_epochs=int(genome.get("head_phase_epochs", 40)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_lr": self.local_lr,
            "local_weight_decay": self.local_weight_decay,
            "head_lr": self.head_lr,
            "head_weight_decay": self.head_weight_decay,
            "normalize_update": self.normalize_update,
            "update_bias": self.update_bias,
            "update_expr": copy.deepcopy(self.update_expr if isinstance(self.update_expr, dict) else _DEFAULT_EXPR),
            "bias_expr": copy.deepcopy(self.bias_expr) if isinstance(self.bias_expr, dict) else None,
            "local_phase_epochs": self.local_phase_epochs,
            "head_phase_epochs": self.head_phase_epochs,
        }


class EvolvedMLPUpdateRule(UpdateRule):
    """Expression-tree local updates for hidden layers + supervised BP on head."""

    def __init__(
        self,
        params: EvolvedRuleParams,
        *,
        track_bp_alignment: bool = False,
        bp_alignment_eps: float = 1e-12,
        effective_local_backprop: bool = False,
    ):
        super().__init__()
        self.params = params
        self.track_bp_alignment = bool(track_bp_alignment)
        self.bp_alignment_eps = float(bp_alignment_eps)
        self.effective_local_backprop = bool(effective_local_backprop)
        self._hidden_layers: list[torch.nn.Linear] = []
        self._head_layer: torch.nn.Linear | None = None
        self._head_optimizer: torch.optim.Optimizer | None = None
        self._bp_cosine_by_epoch: dict[int, list[float]] = {}
        self._bp_sign_match_by_epoch: dict[int, list[float]] = {}
        self._bp_update_gap_by_epoch: dict[int, list[float]] = {}

    def on_train_start(self, model, task, device, state=None):
        if not hasattr(model, "linears"):
            raise ValueError("EvolvedMLPUpdateRule requires a model exposing `linears` (MLPClassifier).")

        linears = list(model.linears)
        if len(linears) < 2:
            raise ValueError("EvolvedMLPUpdateRule requires at least one hidden layer and one head layer.")

        self._hidden_layers = [layer for layer in linears[:-1]]
        self._head_layer = linears[-1]

        for layer in self._hidden_layers:
            for p in layer.parameters():
                p.requires_grad_(False)

        for p in self._head_layer.parameters():
            p.requires_grad_(True)

        self._head_optimizer = torch.optim.AdamW(
            self._head_layer.parameters(),
            lr=float(self.params.head_lr),
            weight_decay=float(self.params.head_weight_decay),
        )
        self._bp_cosine_by_epoch = {}
        self._bp_sign_match_by_epoch = {}
        self._bp_update_gap_by_epoch = {}

    def train_step(self, model, task, batch, device, state=None) -> dict:
        if self._head_optimizer is None or self._head_layer is None:
            self.on_train_start(model, task, device, state)

        model.train()
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)

        epoch = int(getattr(state, "epoch", 1)) if state is not None else 1
        local_until = max(0, int(self.params.local_phase_epochs))
        phase = "local" if epoch <= local_until else "head"

        if phase == "local":
            if self.track_bp_alignment:
                logits, loss, mistake_scalar = self._local_step_with_bp_alignment(
                    model=model,
                    task=task,
                    x=x,
                    labels=labels,
                    epoch=epoch,
                )
            elif self.effective_local_backprop:
                logits, loss, mistake_scalar = self._local_step_with_effective_backprop(
                    model=model,
                    task=task,
                    x=x,
                    labels=labels,
                )
            else:
                with torch.no_grad():
                    logits, cache = model(x, return_cache=True)
                    loss = task.loss(logits, labels)
                    preds = logits.argmax(dim=1)
                    mistake_scalar = (preds != labels).float().unsqueeze(1)
                    updates = self._build_local_hidden_updates(
                        cache=cache,
                        logits=logits,
                        labels=labels,
                        mistake_scalar=mistake_scalar,
                    )
                    self._apply_local_hidden_updates(updates)
        else:
            assert self._head_optimizer is not None
            self._head_optimizer.zero_grad(set_to_none=True)
            logits = model(x, return_cache=False)
            loss = task.loss(logits, labels)
            loss.backward()
            self._head_optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                mistake_scalar = (preds != labels).float().unsqueeze(1)

        metrics = task.metrics(logits.detach(), labels)
        metrics["loss"] = float(loss.item())
        metrics["mistake_rate"] = float(mistake_scalar.mean().item())
        metrics["phase_local"] = 1.0 if phase == "local" else 0.0
        self.global_step += 1
        return metrics

    def alignment_summary(self) -> dict[str, Any]:
        if (
            not self._bp_cosine_by_epoch
            and not self._bp_sign_match_by_epoch
            and not self._bp_update_gap_by_epoch
        ):
            return {}
        out: dict[str, Any] = {}
        self._merge_epoch_stats(
            out=out,
            values_by_epoch=self._bp_cosine_by_epoch,
            prefix="bp_cosine",
        )
        self._merge_epoch_stats(
            out=out,
            values_by_epoch=self._bp_sign_match_by_epoch,
            prefix="bp_sign_match",
        )
        self._merge_epoch_stats(
            out=out,
            values_by_epoch=self._bp_update_gap_by_epoch,
            prefix="bp_update_gap",
        )
        return out

    @staticmethod
    def _merge_epoch_stats(
        *,
        out: dict[str, Any],
        values_by_epoch: dict[int, list[float]],
        prefix: str,
    ) -> None:
        if not values_by_epoch:
            return
        epoch_means: dict[str, float] = {}
        batch_values: list[float] = []
        for epoch in sorted(values_by_epoch):
            values = [float(v) for v in values_by_epoch[epoch]]
            if not values:
                continue
            epoch_means[str(epoch)] = float(sum(values) / len(values))
            batch_values.extend(values)
        if not epoch_means or not batch_values:
            return
        last_epoch = max(int(k) for k in epoch_means.keys())
        out[f"{prefix}_epoch_means"] = epoch_means
        out[f"{prefix}_epoch_mean"] = float(sum(epoch_means.values()) / len(epoch_means))
        out[f"{prefix}_last_local_epoch"] = float(epoch_means[str(last_epoch)])
        out[f"{prefix}_batch_mean"] = float(sum(batch_values) / len(batch_values))
        out[f"{prefix}_num_batches"] = int(len(batch_values))

    def _set_hidden_requires_grad(self, flag: bool) -> None:
        for layer in self._hidden_layers:
            for p in layer.parameters():
                p.requires_grad_(bool(flag))

    def _local_step_with_bp_alignment(self, *, model, task, x, labels, epoch: int):
        self._set_hidden_requires_grad(True)
        try:
            logits, cache = model(x, return_cache=True)
            loss = task.loss(logits, labels)
            bp_grads = torch.autograd.grad(
                loss,
                [layer.weight for layer in self._hidden_layers],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            bp_updates: list[torch.Tensor] = []
            for layer, grad in zip(self._hidden_layers, bp_grads):
                if grad is None:
                    bp_updates.append(torch.zeros_like(layer.weight))
                else:
                    bp_updates.append((-grad).detach())

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                mistake_scalar = (preds != labels).float().unsqueeze(1)
                updates = self._build_local_hidden_updates(
                    cache=cache,
                    logits=logits,
                    labels=labels,
                    mistake_scalar=mistake_scalar,
                )
                ga_updates = [dw for (dw, _) in updates]
                if self.effective_local_backprop:
                    bp_pack = [(dw.detach(), None) for dw in bp_updates]
                    self._apply_local_hidden_updates(bp_pack)
                else:
                    self._apply_local_hidden_updates(updates)
                cosine = _cosine_from_update_lists(
                    ga_updates,
                    bp_updates,
                    eps=self.bp_alignment_eps,
                )
                sign_match = _sign_match_ratio_from_update_lists(
                    ga_updates,
                    bp_updates,
                    eps=self.bp_alignment_eps,
                )
                update_gap = _relative_l2_gap_from_update_lists(
                    ga_updates,
                    bp_updates,
                    eps=self.bp_alignment_eps,
                )
                self._bp_cosine_by_epoch.setdefault(int(epoch), []).append(float(cosine))
                self._bp_sign_match_by_epoch.setdefault(int(epoch), []).append(float(sign_match))
                self._bp_update_gap_by_epoch.setdefault(int(epoch), []).append(float(update_gap))

            return logits.detach(), loss.detach(), mistake_scalar
        finally:
            self._set_hidden_requires_grad(False)

    def _local_step_with_effective_backprop(self, *, model, task, x, labels):
        self._set_hidden_requires_grad(True)
        try:
            logits = model(x, return_cache=False)
            loss = task.loss(logits, labels)
            bp_grads = torch.autograd.grad(
                loss,
                [layer.weight for layer in self._hidden_layers],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                mistake_scalar = (preds != labels).float().unsqueeze(1)
                bp_updates: list[tuple[torch.Tensor, torch.Tensor | None]] = []
                for layer, grad in zip(self._hidden_layers, bp_grads):
                    if grad is None:
                        bp_updates.append((torch.zeros_like(layer.weight), None))
                    else:
                        bp_updates.append(((-grad).detach(), None))
                self._apply_local_hidden_updates(bp_updates)
            return logits.detach(), loss.detach(), mistake_scalar
        finally:
            self._set_hidden_requires_grad(False)

    def _build_local_hidden_updates(
        self,
        *,
        cache: dict[str, Any],
        logits: torch.Tensor,
        labels: torch.Tensor,
        mistake_scalar: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        n_hidden = len(self._hidden_layers)
        expr = self.params.update_expr if isinstance(self.params.update_expr, dict) else _DEFAULT_EXPR
        updates: list[tuple[torch.Tensor, torch.Tensor | None]] = []

        for idx, layer in enumerate(self._hidden_layers):
            layer_name = f"layer{idx}"
            xin = cache["block_inputs"][layer_name].detach()
            u = cache["block_outputs"][layer_name].detach()
            if idx < (len(self._hidden_layers) - 1):
                y = cache["block_inputs"][f"layer{idx + 1}"].detach()
            else:
                y = cache["block_inputs"]["head"].detach()

            d_out, d_in = int(layer.weight.size(0)), int(layer.weight.size(1))
            signals = _build_signals(
                x=xin,
                u=u,
                y=y,
                layer_weight=layer.weight,
                labels=labels,
                mistake_scalar=mistake_scalar,
                num_classes=int(logits.size(1)),
                d_out=d_out,
                d_in=d_in,
                layer_idx=idx,
                num_hidden=n_hidden,
            )

            value = _eval_expr(expr, signals, d_out=d_out, d_in=d_in)
            dw = _to_weight_update(value, x=signals["x"], u=signals["u"], d_out=d_out, d_in=d_in).detach()
            if self.params.normalize_update:
                dw = _normalize_matrix(dw)

            db: torch.Tensor | None = None
            if layer.bias is not None and self.params.update_bias:
                if isinstance(self.params.bias_expr, dict):
                    bval = _eval_expr(self.params.bias_expr, signals, d_out=d_out, d_in=d_in)
                    db = _to_bias_update(bval, d_out=d_out).detach()
                else:
                    db = dw.mean(dim=1).detach()

            updates.append((dw, db))

        return updates

    def _apply_local_hidden_updates(self, updates: list[tuple[torch.Tensor, torch.Tensor | None]]) -> None:
        for layer, (dw, db) in zip(self._hidden_layers, updates):
            if self.params.local_weight_decay > 0.0:
                layer.weight.mul_(1.0 - self.params.local_lr * self.params.local_weight_decay)
            layer.weight.add_(dw, alpha=self.params.local_lr)
            if layer.bias is not None and db is not None:
                layer.bias.add_(db, alpha=self.params.local_lr)


def _build_signals(
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    layer_weight: torch.Tensor,
    labels: torch.Tensor,
    mistake_scalar: torch.Tensor,
    num_classes: int,
    d_out: int,
    d_in: int,
    layer_idx: int,
    num_hidden: int,
) -> dict[str, torch.Tensor]:
    batch = int(x.size(0))
    device = x.device
    dtype = x.dtype

    onehot = F.one_hot(labels.long(), num_classes=max(1, int(num_classes))).to(device=device, dtype=dtype)
    label = _to_dim(onehot, d_out)
    mistake = mistake_scalar.to(dtype=dtype).expand(-1, d_out)
    layer_val = torch.full((batch, 1), float(layer_idx), device=device, dtype=dtype)
    denom = float(max(1, num_hidden - 1))
    layer_ratio = torch.full((batch, 1), float(layer_idx) / denom, device=device, dtype=dtype)
    w = layer_weight.detach().to(device=device, dtype=dtype).view(1, d_out, d_in).expand(batch, -1, -1)

    return {
        "x": x,
        "u": u,
        "y": y,
        "w": w,
        "weight": w,
        "label": label,
        "mistake": mistake,
        "layer": layer_val.expand(-1, d_out),
        "layer_ratio": layer_ratio.expand(-1, d_out),
        "ones": torch.ones(batch, 1, device=device, dtype=dtype),
        "zeros": torch.zeros(batch, 1, device=device, dtype=dtype),
    }


def _eval_expr(node: Any, signals: dict[str, torch.Tensor], *, d_out: int, d_in: int) -> torch.Tensor:
    batch = int(signals["x"].size(0))
    device = signals["x"].device
    dtype = signals["x"].dtype

    if not isinstance(node, dict):
        return torch.zeros(batch, 1, device=device, dtype=dtype)

    t = str(node.get("t", "term")).lower()
    if t == "term":
        name = str(node.get("name", "zeros")).lower()
        if name in signals:
            return signals[name]
        return signals["zeros"]
    if t == "const":
        value = float(node.get("v", 0.0))
        return torch.full((batch, 1), value, device=device, dtype=dtype)
    if t == "unary":
        a = _eval_expr(node.get("a"), signals, d_out=d_out, d_in=d_in)
        return _apply_unary(node, a, d_out=d_out, d_in=d_in)
    if t == "binary":
        op = str(node.get("op", "add")).lower()
        a = _eval_expr(node.get("a"), signals, d_out=d_out, d_in=d_in)
        b = _eval_expr(node.get("b"), signals, d_out=d_out, d_in=d_in)
        return _apply_binary(op, a, b, d_out=d_out, d_in=d_in)
    return signals["zeros"]


def _apply_unary(node: dict[str, Any], a: torch.Tensor, *, d_out: int, d_in: int) -> torch.Tensor:
    op = str(node.get("op", "neg")).lower()
    coeff = float(node.get("c", 1.0))
    if op == "scale":
        return a * coeff
    if op == "winner_flip":
        temperature = float(node.get("c", 12.0))
        return _winner_sign_flip(a, temperature=temperature)
    if op == "neg":
        return -a
    if op == "abs":
        return a.abs()
    if op == "sign":
        return a.sign()
    if op == "square":
        return a * a
    if op == "sqrt_abs":
        return (a.abs() + 1e-8).sqrt()
    if op == "log1p_abs":
        return torch.log1p(a.abs())
    if op == "relu":
        return F.relu(a)
    if op == "tanh":
        return torch.tanh(a)
    if op == "sigmoid":
        return torch.sigmoid(a)
    if op == "softmax":
        dim = -1 if a.dim() >= 2 else 0
        return F.softmax(a, dim=dim)
    if op == "l2norm":
        return _l2_normalize(a)
    if op == "standardize":
        return _standardize(a)
    if op == "normalize":
        return _normalize_any(a)
    if op == "outer_normalize":
        return _normalize_any(_to_matrix(a, d_out=d_out, d_in=d_in))
    if op == "row_normalize":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        n = m.norm(dim=2, keepdim=True) + 1e-8
        return m / n
    if op == "col_normalize":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        n = m.norm(dim=1, keepdim=True) + 1e-8
        return m / n
    if op == "exp":
        return torch.exp(a.clamp(min=-12.0, max=12.0))
    if op == "sin":
        return torch.sin(a)
    if op == "cos":
        return torch.cos(a)
    if op == "clip":
        return torch.clamp(a, min=-1.0, max=1.0)
    if op in {"sum", "reduce_sum"}:
        return _reduce_nonbatch(a, reduce="sum")
    if op == "batch_sum":
        return _reduce_batch_broadcast(a, reduce="sum")
    if op == "batch_mean":
        return _reduce_batch_broadcast(a, reduce="mean")
    if op == "feature_sum":
        return _reduce_feature(a, reduce="sum")
    if op == "feature_mean":
        return _reduce_feature(a, reduce="mean")
    if op == "row_sum":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        return m.sum(dim=2, keepdim=True)
    if op == "col_sum":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        return m.sum(dim=1, keepdim=True)
    if op == "row_mean":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        return m.mean(dim=2, keepdim=True)
    if op == "col_mean":
        m = _to_matrix(a, d_out=d_out, d_in=d_in)
        return m.mean(dim=1, keepdim=True)
    if op == "mean":
        if a.dim() == 3:
            return a.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        if a.dim() >= 2:
            return a.mean(dim=1, keepdim=True)
        return a
    return a


def _apply_binary(op: str, a: torch.Tensor, b: torch.Tensor, *, d_out: int, d_in: int) -> torch.Tensor:
    if op == "outer":
        va = _to_dim(_to_2d(a), d_out)
        vb = _to_dim(_to_2d(b), d_in)
        return torch.einsum("bi,bj->bij", va, vb)

    if op == "dot":
        va, vb = _align_2d(_to_2d(a), _to_2d(b), d_out=d_out, d_in=d_in)
        return (va * vb).sum(dim=1, keepdim=True)
    if op == "matmul":
        return _matmul_any(a, b, d_out=d_out, d_in=d_in)

    aa, bb = _align_any(a, b, d_out=d_out, d_in=d_in)
    if op == "add":
        return aa + bb
    if op == "sub":
        return aa - bb
    if op == "mul":
        return aa * bb
    if op == "div":
        return aa / (bb.abs() + 1e-6)
    if op == "max":
        return torch.maximum(aa, bb)
    if op == "min":
        return torch.minimum(aa, bb)
    return aa + bb


def _align_any(a: torch.Tensor, b: torch.Tensor, *, d_out: int, d_in: int) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() == 3 or b.dim() == 3:
        ma = _to_matrix(a, d_out=d_out, d_in=d_in)
        mb = _to_matrix(b, d_out=d_out, d_in=d_in)
        return ma, mb
    va, vb = _align_2d(_to_2d(a), _to_2d(b), d_out=d_out, d_in=d_in)
    return va, vb


def _align_2d(a: torch.Tensor, b: torch.Tensor, *, d_out: int, d_in: int) -> tuple[torch.Tensor, torch.Tensor]:
    da, db = int(a.size(1)), int(b.size(1))
    if da == db:
        return a, b
    if da == 1:
        return a.expand(-1, db), b
    if db == 1:
        return a, b.expand(-1, da)

    target = d_out if (da == d_out or db == d_out) else d_in if (da == d_in or db == d_in) else max(da, db)
    return _to_dim(a, target), _to_dim(b, target)


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.mean(dim=2)
    if x.dim() == 1:
        return x.unsqueeze(1)
    if x.dim() == 0:
        return x.view(1, 1)
    return x.view(x.size(0), -1)


def _to_dim(x: torch.Tensor, target: int) -> torch.Tensor:
    if x.dim() != 2:
        x = _to_2d(x)
    feat = int(x.size(1))
    if feat == target:
        return x
    if feat > target:
        return x[:, :target]
    reps = (target + feat - 1) // feat
    return x.repeat(1, reps)[:, :target]


def _to_matrix(x: torch.Tensor, *, d_out: int, d_in: int) -> torch.Tensor:
    if x.dim() == 3:
        out = _resize_dim(x, dim=1, target=d_out)
        out = _resize_dim(out, dim=2, target=d_in)
        return out

    v = _to_2d(x)
    feat = int(v.size(1))
    if feat == d_out:
        return v.unsqueeze(2).expand(-1, d_out, d_in)
    if feat == d_in:
        return v.unsqueeze(1).expand(-1, d_out, d_in)
    if feat == 1:
        return v.unsqueeze(1).expand(-1, d_out, d_in)
    vv = _to_dim(v, d_out)
    return vv.unsqueeze(2).expand(-1, d_out, d_in)


def _resize_dim(x: torch.Tensor, *, dim: int, target: int) -> torch.Tensor:
    size = int(x.size(dim))
    if size == target:
        return x
    if size > target:
        sl = [slice(None)] * x.dim()
        sl[dim] = slice(0, target)
        return x[tuple(sl)]
    reps = [1] * x.dim()
    reps[dim] = (target + size - 1) // size
    out = x.repeat(*reps)
    sl = [slice(None)] * out.dim()
    sl[dim] = slice(0, target)
    return out[tuple(sl)]


def _to_weight_update(value: torch.Tensor, *, x: torch.Tensor, u: torch.Tensor, d_out: int, d_in: int) -> torch.Tensor:
    b = float(max(1, x.size(0)))
    if value.dim() == 3:
        return value.mean(dim=0)

    v = _to_2d(value)
    feat = int(v.size(1))
    if feat == d_out:
        return v.transpose(0, 1).matmul(_to_dim(_to_2d(x), d_in)) / b
    if feat == d_in:
        return _to_dim(_to_2d(u), d_out).transpose(0, 1).matmul(v) / b

    m = _to_matrix(v, d_out=d_out, d_in=d_in)
    return m.mean(dim=0)


def _to_bias_update(value: torch.Tensor, *, d_out: int) -> torch.Tensor:
    if value.dim() == 3:
        return value.mean(dim=(0, 2))
    v = _to_dim(_to_2d(value), d_out)
    return v.mean(dim=0)


def _normalize_matrix(dw: torch.Tensor) -> torch.Tensor:
    return dw / (dw.norm() + 1e-8)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        n = x.norm(dim=(1, 2), keepdim=True) + 1e-8
        return x / n
    if x.dim() >= 2:
        return F.normalize(x, dim=1)
    return x / (x.norm() + 1e-8)


def _standardize(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True, unbiased=False)
        return (x - mu) / (sigma + 1e-6)
    if x.dim() >= 2:
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / (sigma + 1e-6)
    mu = x.mean()
    sigma = x.std(unbiased=False)
    return (x - mu) / (sigma + 1e-6)


def _normalize_any(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        return x / (x.abs().max() + 1e-8)
    dims = tuple(range(1, x.dim()))
    n = x.norm(dim=dims, keepdim=True) + 1e-8
    return x / n


def _winner_sign_flip(x: torch.Tensor, *, temperature: float) -> torch.Tensor:
    v = _to_2d(x)
    if v.numel() == 0:
        return v
    flat = v.transpose(0, 1).contiguous()  # [features, batch]
    soft = torch.softmax(float(temperature) * flat, dim=0)
    soft = -soft
    win = torch.argmax(flat, dim=0)
    idx = torch.arange(flat.size(1), device=flat.device)
    soft[win, idx] = -soft[win, idx]
    return soft.transpose(0, 1).contiguous()


def _reduce_nonbatch(x: torch.Tensor, *, reduce: str) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    dims = tuple(range(1, x.dim()))
    if reduce == "sum":
        return x.sum(dim=dims, keepdim=True)
    return x.mean(dim=dims, keepdim=True)


def _reduce_batch_broadcast(x: torch.Tensor, *, reduce: str) -> torch.Tensor:
    if x.dim() == 0:
        return x.view(1, 1)
    if x.dim() == 1:
        val = x.sum() if reduce == "sum" else x.mean()
        return torch.full_like(x, val)
    red = x.sum(dim=0, keepdim=True) if reduce == "sum" else x.mean(dim=0, keepdim=True)
    return red.expand_as(x)


def _reduce_feature(x: torch.Tensor, *, reduce: str) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    dim = x.dim() - 1
    if reduce == "sum":
        return x.sum(dim=dim, keepdim=True)
    return x.mean(dim=dim, keepdim=True)


def _matmul_any(a: torch.Tensor, b: torch.Tensor, *, d_out: int, d_in: int) -> torch.Tensor:
    if a.dim() == 2 and b.dim() == 2:
        va = _to_2d(a)
        vb = _to_2d(b)
        batch = int(min(va.size(0), vb.size(0)))
        if batch <= 0:
            return torch.zeros(1, d_out, d_in, device=va.device, dtype=va.dtype)
        va = va[:batch]
        vb = vb[:batch]
        m = va.transpose(0, 1).matmul(vb)  # [fa, fb], equivalent to sum_b a[b]^T b[b]
        return m.unsqueeze(0).expand(batch, -1, -1)

    ma = _to_matrix(a, d_out=d_out, d_in=d_in)
    mb = _to_matrix(b, d_out=d_out, d_in=d_in)
    k = int(min(ma.size(2), mb.size(1)))
    if k <= 0:
        return torch.zeros_like(ma)
    return torch.bmm(ma[:, :, :k], mb[:, :k, :])


def _cosine_from_update_lists(
    ga_updates: list[torch.Tensor],
    bp_updates: list[torch.Tensor],
    *,
    eps: float,
) -> float:
    ga_flat = _flatten_update_list(ga_updates)
    bp_flat = _flatten_update_list(bp_updates)
    if ga_flat.numel() == 0 or bp_flat.numel() == 0:
        return 0.0

    if ga_flat.numel() != bp_flat.numel():
        n = int(min(ga_flat.numel(), bp_flat.numel()))
        if n <= 0:
            return 0.0
        ga_flat = ga_flat[:n]
        bp_flat = bp_flat[:n]

    den = float(ga_flat.norm().item() * bp_flat.norm().item())
    if den <= float(eps):
        return 0.0
    num = float(torch.dot(ga_flat, bp_flat).item())
    return float(num / (den + float(eps)))


def _sign_match_ratio_from_update_lists(
    ga_updates: list[torch.Tensor],
    bp_updates: list[torch.Tensor],
    *,
    eps: float,
) -> float:
    if len(ga_updates) != len(bp_updates):
        return 0.0

    same_total = 0.0
    count_total = 0
    for ga_raw, bp_raw in zip(ga_updates, bp_updates):
        if (not torch.is_tensor(ga_raw)) or (not torch.is_tensor(bp_raw)):
            return 0.0
        bp = bp_raw.detach()
        ga = ga_raw.detach()
        if bp.numel() == 0:
            continue
        try:
            ga = _broadcast_to_shape(ga, target_shape=tuple(int(s) for s in bp.shape))
        except RuntimeError:
            return 0.0

        ga_sign = _sign_with_eps(ga, eps=float(eps))
        bp_sign = _sign_with_eps(bp, eps=float(eps))
        same_total += float((ga_sign == bp_sign).float().sum().item())
        count_total += int(bp_sign.numel())

    if count_total <= 0:
        return 0.0
    return float(same_total / float(count_total))


def _relative_l2_gap_from_update_lists(
    ga_updates: list[torch.Tensor],
    bp_updates: list[torch.Tensor],
    *,
    eps: float,
) -> float:
    ga_flat = _flatten_update_list(ga_updates)
    bp_flat = _flatten_update_list(bp_updates)
    if ga_flat.numel() == 0 or bp_flat.numel() == 0:
        return 0.0

    if ga_flat.numel() != bp_flat.numel():
        n = int(min(ga_flat.numel(), bp_flat.numel()))
        if n <= 0:
            return 0.0
        ga_flat = ga_flat[:n]
        bp_flat = bp_flat[:n]

    diff_norm = float((ga_flat - bp_flat).norm().item())
    bp_norm = float(bp_flat.norm().item())
    if bp_norm <= float(eps):
        return 0.0 if diff_norm <= float(eps) else float(diff_norm)
    return float(diff_norm / (bp_norm + float(eps)))


def _broadcast_to_shape(x: torch.Tensor, *, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(int(s) for s in x.shape) == target_shape:
        return x
    y = x
    while y.dim() < len(target_shape):
        y = y.unsqueeze(0)
    return y.expand(*target_shape)


def _sign_with_eps(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    out = torch.zeros_like(x)
    out = torch.where(x > float(eps), torch.ones_like(out), out)
    out = torch.where(x < -float(eps), -torch.ones_like(out), out)
    return out


def _flatten_update_list(updates: list[torch.Tensor]) -> torch.Tensor:
    parts = [u.detach().reshape(-1) for u in updates if torch.is_tensor(u)]
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)


def render_generated_rule_source(
    *,
    rule_name: str,
    params: EvolvedRuleParams,
    score: float,
    objective_value: float | None,
) -> str:
    params_literal = _py_repr_dict(params.as_dict())
    objective_literal = "None" if objective_value is None else repr(float(objective_value))
    return f'''from __future__ import annotations

from .registry import UpdateRuleContext, register_update_rule
from genetic_algorithm.evolved_rule import EvolvedMLPUpdateRule, EvolvedRuleParams


RULE_NAME = "{rule_name}"
GA_SCORE = {repr(float(score))}
GA_OBJECTIVE_VALUE = {objective_literal}
PARAMS = {params_literal}


@register_update_rule(RULE_NAME)
def build_generated_rule(ctx: UpdateRuleContext):
    params = EvolvedRuleParams.from_genome(PARAMS)
    return EvolvedMLPUpdateRule(params)
'''


def _py_repr_dict(dct: dict[str, Any]) -> str:
    items = []
    for key in sorted(dct.keys()):
        value = dct[key]
        items.append(f"{key!r}: {value!r}")
    return "{" + ", ".join(items) + "}"
