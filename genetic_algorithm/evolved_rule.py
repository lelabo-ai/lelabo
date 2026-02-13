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

    def __init__(self, params: EvolvedRuleParams):
        super().__init__()
        self.params = params
        self._hidden_layers: list[torch.nn.Linear] = []
        self._head_layer: torch.nn.Linear | None = None
        self._head_optimizer: torch.optim.Optimizer | None = None

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
            with torch.no_grad():
                logits, cache = model(x, return_cache=True)
                loss = task.loss(logits, labels)
                preds = logits.argmax(dim=1)
                mistake_scalar = (preds != labels).float().unsqueeze(1)
                n_hidden = len(self._hidden_layers)

                for idx, layer in enumerate(self._hidden_layers):
                    layer_name = f"layer{idx}"
                    xin = cache["block_inputs"][layer_name]
                    u = cache["block_outputs"][layer_name]
                    if idx < (len(self._hidden_layers) - 1):
                        y = cache["block_inputs"][f"layer{idx + 1}"]
                    else:
                        y = cache["block_inputs"]["head"]

                    d_out, d_in = int(layer.weight.size(0)), int(layer.weight.size(1))
                    signals = _build_signals(
                        x=xin,
                        u=u,
                        y=y,
                        labels=labels,
                        mistake_scalar=mistake_scalar,
                        num_classes=int(logits.size(1)),
                        d_out=d_out,
                        layer_idx=idx,
                        num_hidden=n_hidden,
                    )

                    expr = self.params.update_expr if isinstance(self.params.update_expr, dict) else _DEFAULT_EXPR
                    value = _eval_expr(expr, signals, d_out=d_out, d_in=d_in)
                    dw = _to_weight_update(value, x=signals["x"], u=signals["u"], d_out=d_out, d_in=d_in)
                    if self.params.normalize_update:
                        dw = _normalize_matrix(dw)

                    if self.params.local_weight_decay > 0.0:
                        layer.weight.mul_(1.0 - self.params.local_lr * self.params.local_weight_decay)
                    layer.weight.add_(dw, alpha=self.params.local_lr)

                    if layer.bias is not None and self.params.update_bias:
                        if isinstance(self.params.bias_expr, dict):
                            bval = _eval_expr(self.params.bias_expr, signals, d_out=d_out, d_in=d_in)
                            db = _to_bias_update(bval, d_out=d_out)
                        else:
                            db = dw.mean(dim=1)
                        layer.bias.add_(db, alpha=self.params.local_lr)
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


def _build_signals(
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    mistake_scalar: torch.Tensor,
    num_classes: int,
    d_out: int,
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

    return {
        "x": x,
        "u": u,
        "y": y,
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
