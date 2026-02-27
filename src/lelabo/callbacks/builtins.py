from __future__ import annotations

from typing import Any

from ..core.callbacks import EarlyStopping, EarlyStoppingConfig
from .registry import CallbackContext, register_callback


def _parse_mode(raw_mode: Any, monitor: str) -> str:
    mode = str(raw_mode if raw_mode is not None else "auto").strip().lower()
    if mode == "auto":
        key = str(monitor).strip().lower()
        if any(token in key for token in ("acc", "f1", "precision", "recall", "r2", "auc")):
            return "max"
        return "min"
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported earlystopping mode '{raw_mode}'. Expected: auto|min|max.")
    return mode


@register_callback("earlystopping")
def build_early_stopping(ctx: CallbackContext):
    params = ctx.callback_params()
    monitor = str(params.get("monitor", "val.acc")).strip()
    if not monitor:
        raise ValueError("earlystopping callback requires a non-empty monitor.")

    mode = _parse_mode(params.get("mode", "auto"), monitor)
    min_delta_mode = str(params.get("min_delta_mode", "abs")).strip().lower()
    if min_delta_mode not in {"abs", "rel"}:
        raise ValueError("earlystopping.min_delta_mode must be one of: abs, rel.")

    patience = int(params.get("patience", 5))
    warmup = int(params.get("warmup", params.get("warmup_epochs", 5)))
    check_every = int(params.get("check_every_n_epochs", 1))
    if patience < 0:
        raise ValueError("earlystopping.patience must be >= 0.")
    if warmup < 0:
        raise ValueError("earlystopping.warmup must be >= 0.")
    if check_every <= 0:
        raise ValueError("earlystopping.check_every_n_epochs must be > 0.")

    cfg = EarlyStoppingConfig(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=float(params.get("min_delta", 0.0)),
        min_delta_mode=min_delta_mode,
        warmup_epochs=warmup,
        check_every_n_epochs=check_every,
        restore_best=bool(params.get("restore_best", True)),
        restore_optimizer=bool(params.get("restore_optimizer", False)),
        restore_schedulers=bool(params.get("restore_schedulers", False)),
        restore_train_state=bool(params.get("restore_train_state", False)),
    )
    return EarlyStopping(cfg)
