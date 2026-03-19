from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
sweep_cli = importlib.import_module("lelabo.cli.commands.sweep")
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")
capsule_create = importlib.import_module("lelabo.capsule.create")


def test_sweep_lists_workspace_capsules_and_sweeps(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cap_a = capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    cap_b = capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: False)

    rc = sweep_cli.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cap_a" in out
    assert "cap_b" in out
    assert "example | path: sweeps/example.yaml" in out
    assert "Run parameter sweep workflows" not in out
    assert str(cap_a) not in out
    assert str(cap_b) not in out


def test_sweep_lists_current_capsule_when_run_inside_it(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_single"
    workspace.mkdir()
    cap = capsule_create.create_capsule_scaffold(capsule_name="cap_single", base_dir=workspace, register=False)
    monkeypatch.chdir(cap)
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: False)

    rc = sweep_cli.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cap_single" in out
    assert "example | path: sweeps/example.yaml" in out


def test_sweep_run_resolves_named_sweep_from_single_capsule(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_run"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_single", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)

    calls: list[dict[str, object]] = []

    def _fake_run_from_config(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(sweep_cli, "_run_sweep_from_config", _fake_run_from_config)

    rc = sweep_cli.main(["run", "--sweep", "example", "--dry-run"])
    assert rc == 0
    assert calls
    assert str(calls[0]["config_path"]).endswith("cap_single/sweeps/example.yaml")
    assert calls[0]["dry_run"] is True


def test_sweep_run_requires_explicit_target_in_non_tty_multi_capsule(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_multi"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: False)

    with pytest.raises(SystemExit) as exc:
        sweep_cli.main(["run"])
    msg = str(exc.value)
    assert "--config <path>" in msg
    assert "--capsule <capsule_id> --sweep <name>" in msg


def test_sweep_dashboard_uses_interactive_selection_even_for_single_sweep(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_picker"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_single", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: True)

    selected = sweep_cli.DiscoveredSweep(
        capsule_id="cap_single",
        capsule_root=(workspace / "cap_single").resolve(),
        sweep_name="example",
        config_path=(workspace / "cap_single" / "sweeps" / "example.yaml").resolve(),
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sweep_cli, "_preview_and_confirm_run", lambda *args, **kwargs: (False, None, []))
    monkeypatch.setattr(sweep_cli, "_run_sweep_from_config", lambda **kwargs: calls.append(kwargs) or 0)

    rc = sweep_cli.main([])
    assert rc == 0
    assert calls == []


def test_sweep_run_uses_interactive_selection_when_missing_target_in_tty(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_interactive_run"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_single", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: True)

    selected = sweep_cli.DiscoveredSweep(
        capsule_id="cap_single",
        capsule_root=(workspace / "cap_single").resolve(),
        sweep_name="example",
        config_path=(workspace / "cap_single" / "sweeps" / "example.yaml").resolve(),
    )
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sweep_cli, "_preview_and_confirm_run", lambda *args, **kwargs: (True, selected, ["--dry-run"]))
    monkeypatch.setattr(sweep_cli, "_run_sweep_from_config", lambda **kwargs: calls.append(kwargs) or 0)

    rc = sweep_cli.main(["run"])
    assert rc == 0
    assert calls
    assert calls[0]["dry_run"] is True


def test_sweep_picker_run_and_dry_run_keys_return_selected_sweep(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_picker_keys"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_single", base_dir=workspace, register=False)
    capsules = sweep_cli.discover_workspace_capsules(workspace)
    selected = capsules[0].sweeps[0]

    monkeypatch.setattr(
        sweep_cli,
        "_pick_sweep_interactive",
        lambda *args, **kwargs: ("run", selected, ["--dry-run"]),
    )
    monkeypatch.setattr(sweep_cli, "_is_interactive_tty", lambda: True)

    should_run, resolved, flags = sweep_cli._preview_and_confirm_run(
        capsules,
        outdir="outputs/runs",
        name=None,
        max_parallel=1,
        gpus=None,
        dry_run=False,
        initial_sweep=selected,
    )
    assert should_run is True
    assert resolved == selected
    assert flags == ["--dry-run"]
