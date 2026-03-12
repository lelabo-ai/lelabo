"""Custom callback templates for this capsule.

LeLabo callbacks are hook-based and plug into Trainer.
You can either subclass `lelabo.core.callbacks.Callback`
or return any object implementing the hooks you need.
"""

from typing import Any

from lelabo.callbacks import CallbackContext, register_callback
from lelabo.core.callbacks import Callback


# @register_callback("my_callback_minimal")
# def build_my_callback_minimal(ctx: CallbackContext):
#     # Minimal object can implement only the hooks it needs.
#     class _Minimal:
#         def on_epoch_end(self, trainer, epoch_record, state: Any | None = None) -> None:
#             print(
#                 f"[my_callback_minimal] epoch={epoch_record.epoch} "
#                 f"train.loss={epoch_record.train.loss:.6f}"
#             )
#
#     return _Minimal()
#
#
# class DetailedCallback(Callback):
#     """Advanced callback with state + multiple hooks."""
#
#     def __init__(self, every_n_epochs: int = 5, key: str = "val.loss") -> None:
#         super().__init__()
#         self.every_n_epochs = int(max(1, every_n_epochs))
#         self.key = str(key)
#         self.best = float("inf")
#
#     def on_train_start(self, trainer, state: Any | None = None) -> None:
#         self.best = float("inf")
#
#     def on_epoch_end(self, trainer, epoch_record, state: Any | None = None) -> None:
#         if (int(epoch_record.epoch) % self.every_n_epochs) != 0:
#             return
#         logs = epoch_record.to_log_values()
#         if self.key in logs and isinstance(logs[self.key], (int, float)):
#             value = float(logs[self.key])
#             if value < self.best:
#                 self.best = value
#             print(
#                 f"[detailed_callback] epoch={epoch_record.epoch} "
#                 f"{self.key}={value:.6f} best={self.best:.6f}"
#             )
#
#
# @register_callback("my_callback_detailed")
# def build_my_callback_detailed(ctx: CallbackContext):
#     params = ctx.callback_params()
#     return DetailedCallback(
#         every_n_epochs=int(params.get("every_n_epochs", 5)),
#         key=str(params.get("key", "val.loss")),
#     )
