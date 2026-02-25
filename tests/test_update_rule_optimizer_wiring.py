from __future__ import annotations

import importlib
import sys
from argparse import Namespace

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
builders = importlib.import_module("lelabo.update_rules.builders")
registry_api = importlib.import_module("lelabo.update_rules.registry")
optimizers_api = importlib.import_module("lelabo.optimizers")


def _make_ctx(*, optimizer_name: str = "adamw", mode: str = "supervised"):
    param = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
    optimizer = optimizers_api.make_optimizer(
        optimizer_name,
        [param],
        lr=1e-3,
        weight_decay=1e-2,
        momentum=0.9,
    )
    args = Namespace(
        lr=1e-3,
        weight_decay=1e-2,
        optimizer=optimizer_name,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-2, "momentum": 0.9},
        update_rule_params={},
    )
    return registry_api.UpdateRuleContext(
        args=args,
        model=None,
        task=None,
        optimizer=optimizer,
        mode=mode,
        dataset="iris",
        extra={"update_rule_params": {}},
    ), param


def test_softhebb_reuses_runner_optimizer_instance() -> None:
    ctx, _ = _make_ctx(optimizer_name="sgd", mode="supervised")
    learner = builders.build_softhebb(ctx)
    head_params = [torch.nn.Parameter(torch.ones(2), requires_grad=True)]
    learner._ensure_head_optim(head_params)
    assert learner._head_optimizer is ctx.optimizer


def test_softhebb_rejects_head_lr_and_head_weight_decay_params() -> None:
    ctx, _ = _make_ctx(optimizer_name="sgd", mode="supervised")
    ctx.extra = {"update_rule_params": {"head_lr": 1e-3}}
    try:
        builders.build_softhebb(ctx)
        assert False, "Expected ValueError for deprecated softhebb head params"
    except ValueError as exc:
        message = str(exc)
        assert "Unsupported softhebb update_rule.params keys" in message
        assert "head_lr" in message
        assert "Allowed keys" in message


def test_targetprop_reuses_same_optimizer_for_forward_and_inverse() -> None:
    ctx, _ = _make_ctx(optimizer_name="adamw", mode="supervised")
    learner = builders.build_targetprop(ctx)
    assert learner.fwd_optimizer is ctx.optimizer
    assert learner.inv_optimizer is ctx.optimizer


def test_targetprop_shared_optimizer_keeps_existing_params_when_adding_decoders() -> None:
    ctx, base_param = _make_ctx(optimizer_name="adamw", mode="supervised")
    learner = builders.build_targetprop(ctx)
    assert learner.inv_optimizer is ctx.optimizer

    existing_ids_before = {
        id(p)
        for group in ctx.optimizer.param_groups
        for p in list(group.get("params", []))
        if isinstance(p, torch.nn.Parameter)
    }
    assert id(base_param) in existing_ids_before

    learner._ensure_decoders([(4, 6)], device=torch.device("cpu"), dtype=torch.float32)

    existing_ids_after = {
        id(p)
        for group in ctx.optimizer.param_groups
        for p in list(group.get("params", []))
        if isinstance(p, torch.nn.Parameter)
    }
    assert id(base_param) in existing_ids_after
    assert learner.decoders is not None
    decoder_ids = {id(p) for p in learner.decoders.parameters()}
    assert decoder_ids.issubset(existing_ids_after)
