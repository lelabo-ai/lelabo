from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .versioning import TRAIN_CONFIG_SCHEMA_VERSION


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerSpec:
    name: str = "none"
    interval: str = "epoch"
    monitor: str = "val.loss"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSpec:
    device: str = "auto"
    seed: int = 2
    determinism: str = "relaxed"
    verbose: int = 1
    run_dir: str | None = None


@dataclass(frozen=True)
class SupervisedTrainSpec:
    epochs: int = 50
    batch: int = 64
    val_frac: float = 0.1
    input_noise_training: float = 0.0
    input_noise_dataset: float = 0.0
    noise_on_test: bool = False
    bp_alignment_every: int = 1
    bp_alignment_eps: float = 1e-12


@dataclass(frozen=True)
class EarlyStoppingSpec:
    name: str = "default"
    enabled: bool = True
    monitor: str = "val.acc"
    mode: str = "auto"
    patience: int = 5
    min_delta: float = 0.0
    warmup: int = 5
    restore_best: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustnessSpec:
    mode: str = "none"
    trials: int = 30
    max_samples: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HFSpec:
    glue_task: str = "sst2"
    model: str = "bert-base-uncased"
    trust_remote_code: bool = False
    max_length: int = 128


@dataclass(frozen=True)
class SupervisedConfig:
    config_version: str
    lelabo_version: str
    task: str
    dataset: ComponentSpec
    model: ComponentSpec
    update_rule: ComponentSpec
    optimizer: ComponentSpec
    scheduler: SchedulerSpec
    runtime: RuntimeSpec
    train: SupervisedTrainSpec
    early_stopping: EarlyStoppingSpec
    robustness: RobustnessSpec
    hf: HFSpec
    metrics: tuple[MetricSpec, ...] = ()


@dataclass(frozen=True)
class RLSpec:
    algo: str = "ppo"
    steps: int = 500_000
    eval_episodes: int = 10
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RLConfig:
    config_version: str
    lelabo_version: str
    task: str
    env: str
    model: ComponentSpec
    update_rule: ComponentSpec
    optimizer: ComponentSpec
    runtime: RuntimeSpec
    rl: RLSpec
