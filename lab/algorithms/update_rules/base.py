# lab/algorithms/update_rules/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class UpdateRule(ABC):
    """
    Plug-in interface: implement train_step().
    The trainer calls train_step for each batch.
    """
    def __init__(self):
        self.global_step = 0

    def on_train_start(self, model, task, device):
        pass

    @abstractmethod
    def train_step(self, model, task, batch, device) -> dict:
        """
        Returns dict with at least: {"loss": float}
        You can add anything: {"acc":..., "grad_norm":...}
        """
        raise NotImplementedError

    @torch.no_grad()
    def on_eval_start(self, model, task, device):
        model.eval()
