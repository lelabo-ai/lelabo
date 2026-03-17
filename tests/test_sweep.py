"""Tests for the sweep infrastructure (grid utilities, plan builder, CLI)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lelabo.sweep.grid import (
    build_run_dirname,
    cartesian_grid,
    flagify,
    sanitize,
    stable_short_id,
    to_cli_args,
)
from lelabo.sweep.runner import SweepPlan, build_sweep_plan, load_sweep_config


# --- grid utilities ---


def test_flagify():
    assert flagify("weight_decay") == "--weight-decay"
    assert flagify("lr") == "--lr"


def test_to_cli_args_basic():
    result = to_cli_args({"lr": 0.001, "epochs": 10, "dataset": "iris"})
    assert "--lr" in result
    assert "0.001" in result
    assert "--epochs" in result
    assert "--dataset" in result


def test_to_cli_args_bool():
    result = to_cli_args({"save_checkpoints": True, "verbose": False})
    assert "--save-checkpoints" in result
    assert "--verbose" not in result


def test_to_cli_args_none_skipped():
    result = to_cli_args({"lr": 0.001, "device": None})
    assert "--device" not in result


def test_cartesian_grid_basic():
    grid = {"algo": ["bp", "dfa"], "seed": [0, 1]}
    combos = cartesian_grid(grid)
    assert len(combos) == 4
    assert {"algo": "bp", "seed": 0} in combos
    assert {"algo": "dfa", "seed": 1} in combos


def test_cartesian_grid_single_param():
    combos = cartesian_grid({"seed": [0, 1, 2]})
    assert len(combos) == 3


def test_stable_short_id_deterministic():
    d = {"algo": "bp", "seed": 42, "lr": 0.001}
    id1 = stable_short_id(d)
    id2 = stable_short_id(d)
    assert id1 == id2


def test_stable_short_id_order_independent():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert stable_short_id(d1) == stable_short_id(d2)


def test_sanitize():
    assert sanitize("path/to/thing") == "path-to-thing"
    assert sanitize("value with spaces") == "valuewithspaces"


def test_build_run_dirname():
    args = {"algo": "bp", "seed": 42, "lr": 0.001, "hidden": 256}
    name = build_run_dirname(args, ["algo", "seed"])
    assert name.startswith("algo=bp__seed=42__id=")


# --- sweep plan builder ---


def test_load_sweep_config(tmp_path: Path):
    cfg = {"name": "test", "base": {"dataset": "iris"}, "grid": {"seed": [0, 1]}}
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    loaded = load_sweep_config(config_path)
    assert loaded["name"] == "test"
    assert "grid" in loaded


def test_load_sweep_config_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_sweep_config(tmp_path / "missing.yaml")


def test_load_sweep_config_no_grid(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump({"base": {"dataset": "iris"}}))
    with pytest.raises(ValueError, match="grid"):
        load_sweep_config(config_path)


def test_build_sweep_plan(tmp_path: Path):
    cfg = {
        "name": "test_sweep",
        "display_keys": ["algo", "seed"],
        "base": {"dataset": "iris", "model": "mlp", "epochs": 5},
        "grid": {"algo": ["bp", "dfa"], "seed": [0, 1, 2]},
    }
    plan = build_sweep_plan(config=cfg, outdir=tmp_path)
    assert plan.name == "test_sweep"
    assert len(plan.jobs) == 6  # 2 algos * 3 seeds
    assert all("--run-dir" in " ".join(j.cmd) for j in plan.jobs)


def test_build_sweep_plan_injects_wandb_group(tmp_path: Path):
    cfg = {
        "name": "my_sweep",
        "base": {"dataset": "iris"},
        "grid": {"seed": [0]},
    }
    plan = build_sweep_plan(config=cfg, outdir=tmp_path, wandb_group="custom_group")
    cmd_str = " ".join(plan.jobs[0].cmd)
    assert "wandb.group=custom_group" in cmd_str


def test_sweep_plan_save(tmp_path: Path):
    cfg = {
        "name": "save_test",
        "base": {"dataset": "iris"},
        "grid": {"seed": [0]},
    }
    plan = build_sweep_plan(config=cfg, outdir=tmp_path)
    plan_path = plan.save()
    assert plan_path.exists()
    assert plan_path.name == "plan.json"


# --- capsule sweep template ---


def test_capsule_scaffold_includes_sweeps(tmp_path: Path):
    from lelabo.capsule.create import create_capsule_scaffold

    capsule_dir = create_capsule_scaffold(capsule_name="test_cap", base_dir=tmp_path)
    assert (capsule_dir / "sweeps").is_dir()
    assert (capsule_dir / "sweeps" / "example.yaml").is_file()

    cfg = yaml.safe_load((capsule_dir / "sweeps" / "example.yaml").read_text())
    assert "grid" in cfg
    assert "base" in cfg
