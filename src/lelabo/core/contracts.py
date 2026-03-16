"""Runtime protocols and validation helpers for ``Trainer`` dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ..metrics.base import TrainerMetric
from ..schedulers import SchedulerController
from ..update_rules.base import UpdateRule
from .callbacks import Callback, EarlyStopping


LossCallable: TypeAlias = Callable[[Any, Any], Any]


@runtime_checkable
class ModelLike(Protocol):
    """Torch-like model interface required by the trainer runtime."""

    def to(self, device: str) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def parameters(self, recurse: bool = True) -> Iterable[Any]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> Any: ...


@runtime_checkable
class LoggerLike(Protocol):
    """Minimal logger interface expected by ``Trainer``."""

    def log(self, record: Mapping[str, Any]) -> None: ...


def validate_model(model: Any) -> None:
    """Validate that ``model`` satisfies the trainer's torch-like contract."""
    required = ("to", "train", "eval", "parameters", "state_dict", "load_state_dict")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "Trainer.model must implement a torch-like model contract. "
            f"Missing callable(s): {missing}."
        )


def validate_loss(loss: Any) -> None:
    """Validate that ``loss`` is directly callable."""
    if not callable(loss):
        raise TypeError(
            "Trainer.loss must be a callable accepting (predictions, targets). "
            f"Got {type(loss).__name__}."
        )


def validate_logger(logger: Any) -> None:
    """Validate that ``logger`` exposes ``log(record)``."""
    if not callable(getattr(logger, "log", None)):
        raise TypeError(
            "Trainer.logger must expose log(record: Mapping[str, Any]) -> None."
        )


def validate_learner(learner: Any) -> None:
    """Validate that ``learner`` is a LeLabo update rule instance."""
    if not isinstance(learner, UpdateRule):
        raise TypeError(
            "Trainer.learner must inherit from lelabo.update_rules.base.UpdateRule. "
            f"Got {type(learner).__name__}."
        )


def validate_callbacks(callbacks: list[Any]) -> None:
    """Validate callback instances and global callback invariants."""
    for idx, callback in enumerate(callbacks):
        if not isinstance(callback, Callback):
            raise TypeError(
                "Trainer.callbacks must contain Callback instances. "
                f"Item {idx} has type {type(callback).__name__}."
            )
    early_count = sum(1 for callback in callbacks if isinstance(callback, EarlyStopping))
    if early_count > 1:
        raise ValueError("Trainer supports at most one EarlyStopping callback.")


def validate_metrics(metrics: list[Any]) -> None:
    """Validate metric objects passed to ``Trainer``."""
    for idx, metric in enumerate(metrics):
        if not isinstance(metric, TrainerMetric):
            raise TypeError(
                "Trainer.metrics must contain TrainerMetric instances. "
                f"Item {idx} has type {type(metric).__name__}."
            )


def validate_schedulers(schedulers: list[Any]) -> None:
    """Validate scheduler controller objects passed to ``Trainer``."""
    for idx, scheduler in enumerate(schedulers):
        if not isinstance(scheduler, SchedulerController):
            raise TypeError(
                "Trainer.schedulers expects SchedulerController instances. "
                f"Item {idx} has type {type(scheduler).__name__}."
            )
