"""
example.py

Purpose
-------
Add a trainer callback to this capsule.

Contract
--------
1. Register a builder with `@register_callback("epoch_echo")`
2. The builder receives a `CallbackContext`
3. The builder returns a callback object, usually a `Callback`

Where params come from
----------------------
For the extension recipe, read `resources/EXTENSION_RECIPES.md`.
Read params with `ctx.callback_params()`.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`EpochEchoCallback` prints one monitored scalar every `every_n_epochs`.

Config snippet
--------------
[[callbacks]]
name = "epoch_echo"
enabled = true

[callbacks.params]
every_n_epochs = 2
key = "val.loss"

How to activate
---------------
Uncomment `@register_callback("epoch_echo")`.

How to test
-----------
lelabo list callbacks
lelabo train supervised --config configs/train.supervised.detailed.toml --set callbacks.0.name=epoch_echo

Common errors
-------------
- Keep the callback side effects simple and explicit.
- Return a real callback object; subclassing `Callback` is the safest path.
"""

from __future__ import annotations

from typing import Any

from lelabo.callbacks import Callback, CallbackContext, register_callback


class EpochEchoCallback(Callback):
    def __init__(self, *, every_n_epochs: int = 1, key: str = "val.loss") -> None:
        super().__init__()
        self.every_n_epochs = int(max(1, every_n_epochs))
        self.key = str(key)

    def on_epoch_end(self, trainer, epoch_record, state: Any | None = None) -> None:
        _ = (trainer, state)
        if int(epoch_record.epoch) % self.every_n_epochs != 0:
            return
        logs = epoch_record.to_log_values()
        value = logs.get(self.key, None)
        if isinstance(value, (int, float)):
            print(f"[epoch_echo] epoch={epoch_record.epoch} {self.key}={float(value):.6f}")


# @register_callback("epoch_echo")
def build_epoch_echo(ctx: CallbackContext):
    params = ctx.callback_params()
    return EpochEchoCallback(
        every_n_epochs=int(params.get("every_n_epochs", 1)),
        key=str(params.get("key", "val.loss")),
    )
