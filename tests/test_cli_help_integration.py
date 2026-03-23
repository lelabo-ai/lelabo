from __future__ import annotations

import subprocess
import sys


def _run_cli_help(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "lelabo.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_train_help_displays_train_root_usage() -> None:
    proc = _run_cli_help("train", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train" in proc.stdout
    assert "supervised" in proc.stdout
    assert "rl" in proc.stdout


def test_train_supervised_help_displays_parser_usage() -> None:
    proc = _run_cli_help("train", "supervised", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train supervised" in proc.stdout
    assert "--dataset" in proc.stdout
    assert "Override dataset.name." in proc.stdout
    assert "Override model.name." in proc.stdout
    assert "--rule" in proc.stdout
    assert "Override update_rule.name." in proc.stdout
    assert "Override optimizer.name." in proc.stdout
    assert "Override train.epochs." in proc.stdout
    assert "Override train.batch." in proc.stdout
    assert "--task" not in proc.stdout


def test_train_rl_help_displays_parser_usage() -> None:
    proc = _run_cli_help("train", "rl", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train rl" in proc.stdout
    assert "--env ENV_ID" in proc.stdout
    assert "--algo" in proc.stdout
    assert "--rule" in proc.stdout
    assert "--task" not in proc.stdout

def test_audit_help_displays_audit_parser_usage() -> None:
    proc = _run_cli_help("audit", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo audit" in proc.stdout
    assert "Run warn-only local update-rule audit" in proc.stdout

def test_list_help_displays_list_usage() -> None:
    proc = _run_cli_help("list", "-h")
    assert proc.returncode == 0
    assert "List available LeLabo registries" in proc.stdout
    assert "lelabo list [target] [--json]" in proc.stdout
    assert "update-rules" in proc.stdout
    assert "datasets" in proc.stdout
    assert "initializers" in proc.stdout
    assert "optimizers" in proc.stdout
    assert "losses" in proc.stdout
    assert "schedulers" in proc.stdout


def test_capsule_help_displays_lifecycle_subcommands() -> None:
    proc = _run_cli_help("capsule", "-h")
    assert proc.returncode == 0
    assert "Manage LeLabo experiment capsules" in proc.stdout
    assert "init" in proc.stdout
    assert "attach" in proc.stdout
    assert "stash" in proc.stdout
    assert "checkout" in proc.stdout
    assert "install" in proc.stdout
    assert "share" in proc.stdout
    assert "pack" not in proc.stdout
    assert "lelabo capsule sweep" not in proc.stdout


def test_push_help_displays_publish_usage() -> None:
    proc = _run_cli_help("push", "-h")
    assert proc.returncode == 0
    assert "Publish one capsule" in proc.stdout
    assert "--target NAME" in proc.stdout
    assert "--all-targets" in proc.stdout


def test_repo_help_displays_repo_subcommands() -> None:
    proc = _run_cli_help("repo", "-h")
    assert proc.returncode == 0
    assert "Manage publish repos and targets for capsules." in proc.stdout
    assert "list" in proc.stdout
    assert "create" in proc.stdout
    assert "add capsule" in proc.stdout
    assert "edit" in proc.stdout
    assert "detach" in proc.stdout


def test_config_help_displays_config_subcommands() -> None:
    proc = _run_cli_help("config", "-h")
    assert proc.returncode == 0
    assert "Manage LeLabo user settings." in proc.stdout
    assert "path" in proc.stdout
    assert "show" in proc.stdout
    assert "get" in proc.stdout
    assert "set" in proc.stdout
    assert "edit" in proc.stdout


def test_capsule_install_help_mentions_github_and_checkout() -> None:
    proc = _run_cli_help("capsule", "install", "-h")
    assert proc.returncode == 0
    assert "GitHub LeLabo repo URL" in proc.stdout
    assert "--ref" in proc.stdout
    assert "--checkout" in proc.stdout
    assert "--capsule" in proc.stdout
    assert "repeatable" in proc.stdout
    assert "--all" in proc.stdout
    assert "--rename-to" in proc.stdout
    assert "--force-replace" in proc.stdout


def test_capsule_share_help_mentions_github_publish_only() -> None:
    proc = _run_cli_help("capsule", "share", "-h")
    assert proc.returncode == 0
    assert "Publish a capsule with `lelabo push`" in proc.stdout
    assert "--owner" in proc.stdout
    assert "--repo" in proc.stdout
    assert "--mode" not in proc.stdout
    assert "lelabo export" in proc.stdout


def test_export_help_displays_local_bundle_usage() -> None:
    proc = _run_cli_help("export", "-h")
    assert proc.returncode == 0
    assert "Export one capsule as a local `.tar.gz` bundle." in proc.stdout
    assert "Usage:" in proc.stdout
    assert "lelabo export [capsule_ref]" in proc.stdout
    assert "--out" in proc.stdout


def test_gitspace_command_is_unknown() -> None:
    proc = _run_cli_help("gitspace", "-h")
    assert proc.returncode != 0
    assert "Unknown command: gitspace" in proc.stderr


def test_capsule_show_help_documents_positionals_and_examples() -> None:
    proc = _run_cli_help("capsule", "show", "-h")
    assert proc.returncode == 0
    assert "Show one stored capsule entry" in proc.stdout
    assert "Stored capsule id or alias to inspect" in proc.stdout
    assert "Examples:" in proc.stdout


def test_capsule_remove_help_documents_positionals_and_examples() -> None:
    proc = _run_cli_help("capsule", "remove", "-h")
    assert proc.returncode == 0
    assert "Remove one stored capsule entry" in proc.stdout
    assert "Stored capsule id or alias to remove" in proc.stdout
    assert "Examples:" in proc.stdout


def test_capsule_stash_help_includes_workbench_examples() -> None:
    proc = _run_cli_help("capsule", "stash", "-h")
    assert proc.returncode == 0
    assert "Move a local capsule into the local capsule store." in proc.stdout
    assert "lelabo capsule stash --all ./workspace_capsules" in proc.stdout


def test_capsule_attach_help_includes_linking_description() -> None:
    proc = _run_cli_help("capsule", "attach", "-h")
    assert proc.returncode == 0
    assert "Link an external capsule into the LeLabo capsule registry" in proc.stdout
    assert "--rename-to" in proc.stdout
    assert "--force-replace" in proc.stdout


def test_sweep_help_displays_workspace_first_usage() -> None:
    proc = _run_cli_help("sweep", "-h")
    assert proc.returncode == 0
    assert "Run parameter sweeps from the current workspace." in proc.stdout
    assert "lelabo sweep" in proc.stdout
    assert "lelabo sweep run" in proc.stdout
    assert "--capsule CAPSULE_ID" in proc.stdout
    assert "--sweep SWEEP_NAME" in proc.stdout
    assert "--config PATH" in proc.stdout


def test_train_without_subcommand_prints_help() -> None:
    proc = _run_cli_help("train")
    assert proc.returncode == 0
    assert "usage: lelabo train" in proc.stdout


def test_train_supervised_requires_dataset_flag() -> None:
    proc = _run_cli_help("train", "supervised")
    assert proc.returncode != 0
    assert "--dataset" in proc.stderr


def test_train_rl_requires_env_flag() -> None:
    proc = _run_cli_help("train", "rl")
    assert proc.returncode != 0
    assert "--env" in proc.stderr


def test_root_cli_rejects_removed_create_command() -> None:
    proc = _run_cli_help("create", "-h")
    assert proc.returncode != 0
    assert "Unknown command: create" in proc.stderr


def test_train_supervised_rejects_legacy_source_flag() -> None:
    proc = _run_cli_help("train", "supervised", "--dataset", "iris", "--source", "iris")
    assert proc.returncode != 0
    assert "unrecognized arguments: --source iris" in proc.stderr


def test_train_rl_rejects_legacy_rl_specific_flag() -> None:
    proc = _run_cli_help("train", "rl", "--env", "CartPole-v1", "--gamma", "0.95")
    assert proc.returncode != 0
    assert "unrecognized arguments: --gamma 0.95" in proc.stderr
