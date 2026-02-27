from __future__ import annotations

from pathlib import Path

import yaml

from conftest import REPO_ROOT


def test_all_experiment_configs_have_required_keys() -> None:
    cfg_dir = REPO_ROOT / "experiments" / "sweeps"
    cfg_files = sorted(cfg_dir.glob("*.yaml"))
    assert cfg_files, "No YAML configs found in experiments/sweeps."

    for path in cfg_files:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict), f"{path.name} must load to a mapping."
        assert "name" in cfg and str(cfg["name"]).strip(), f"{path.name} must define a non-empty 'name'."
        assert "base" in cfg and isinstance(cfg["base"], dict), f"{path.name} must define a mapping 'base'."
        assert "grid" in cfg and isinstance(cfg["grid"], dict), f"{path.name} must define a mapping 'grid'."


def test_grid_values_are_lists() -> None:
    cfg_dir = REPO_ROOT / "experiments" / "sweeps"
    for path in sorted(cfg_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in cfg.get("grid", {}).items():
            assert isinstance(value, list), f"{path.name}: grid.{key} must be a list."
