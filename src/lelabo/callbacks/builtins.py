from __future__ import annotations

from ..core.callbacks import EarlyStopping, EarlyStoppingConfig
from .registry import CallbackContext, register_callback


@register_callback("earlystopping")
def build_early_stopping(ctx: CallbackContext):
    params = ctx.callback_params()
    defaults = EarlyStoppingConfig()
    monitor = str(params.get("monitor", defaults.monitor)).strip()
    if not monitor:
        raise ValueError("earlystopping callback requires a non-empty monitor.")

    raw_mode = params.get("mode", defaults.mode)
    min_delta_mode = str(params.get("min_delta_mode", defaults.min_delta_mode)).strip().lower()
    if min_delta_mode not in {"abs", "rel"}:
        raise ValueError("earlystopping.min_delta_mode must be one of: abs, rel.")

    patience = int(params.get("patience", defaults.patience))
    warmup = int(params.get("warmup", params.get("warmup_epochs", defaults.warmup_epochs)))
    check_every = int(params.get("check_every_n_epochs", defaults.check_every_n_epochs))
    if patience < 0:
        raise ValueError("earlystopping.patience must be >= 0.")
    if warmup < 0:
        raise ValueError("earlystopping.warmup must be >= 0.")
    if check_every <= 0:
        raise ValueError("earlystopping.check_every_n_epochs must be > 0.")

    cfg = EarlyStoppingConfig(
        monitor=monitor,
        mode=str(raw_mode),
        patience=patience,
        min_delta=float(params.get("min_delta", defaults.min_delta)),
        min_delta_mode=min_delta_mode,
        warmup_epochs=warmup,
        check_every_n_epochs=check_every,
        restore_best=bool(params.get("restore_best", defaults.restore_best)),
        restore_optimizer=bool(params.get("restore_optimizer", defaults.restore_optimizer)),
        restore_learner_state=bool(params.get("restore_learner_state", defaults.restore_learner_state)),
        restore_schedulers=bool(params.get("restore_schedulers", defaults.restore_schedulers)),
        restore_train_state=bool(params.get("restore_train_state", defaults.restore_train_state)),
    )
    return EarlyStopping(cfg)
