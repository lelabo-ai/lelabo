"""Capsule publication targets and push backends."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gitspace import add_capsule_to_gitspace, find_gitspace_root, init_gitspace, load_gitspace
from .github import parse_owner_repo_from_url
from .install import inspect_capsule_directory


PUBLISH_STATE_SCHEMA_VERSION = "1.0"


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err}")
    return proc.stdout.strip()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _user_cache_root() -> Path:
    home = Path.home()
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(home / "AppData" / "Local")
        return (Path(base).expanduser().resolve() / "LeLabo").resolve()
    if os.sys.platform == "darwin":
        return (home / "Library" / "Caches" / "lelabo").resolve()
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / "lelabo").resolve()
    return (home / ".cache" / "lelabo").resolve()


def publish_state_path() -> Path:
    raw = os.getenv("LELABO_PUBLISH_STATE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_user_cache_root() / "publish_targets.json").resolve()


def repo_checkout_cache_dir(owner: str, repo: str) -> Path:
    token = f"{owner}__{repo}".replace("/", "_")
    return (_user_cache_root() / "repo_checkouts" / token).resolve()


def _empty_publish_state() -> dict[str, Any]:
    return {
        "schema_version": PUBLISH_STATE_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capsules": {},
    }


def load_publish_state() -> dict[str, Any]:
    target = publish_state_path()
    if not target.exists():
        return _empty_publish_state()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_publish_state()
    if not isinstance(data, dict):
        return _empty_publish_state()
    data.setdefault("schema_version", PUBLISH_STATE_SCHEMA_VERSION)
    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    data.setdefault("capsules", {})
    if not isinstance(data["capsules"], dict):
        data["capsules"] = {}
    return data


def save_publish_state(state: dict[str, Any]) -> Path:
    out = dict(state)
    out["schema_version"] = PUBLISH_STATE_SCHEMA_VERSION
    out["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target = publish_state_path()
    _atomic_write_text(target, json.dumps(out, indent=2, ensure_ascii=False))
    return target


def capsule_state_key(capsule_root: Path) -> str:
    return str(capsule_root.expanduser().resolve())


def _capsule_state_entry(state: dict[str, Any], capsule_root: Path) -> dict[str, Any]:
    key = capsule_state_key(capsule_root)
    capsules = state.setdefault("capsules", {})
    if not isinstance(capsules, dict):
        state["capsules"] = {}
        capsules = state["capsules"]
    entry = capsules.setdefault(key, {"targets": {}, "default_target": None, "last_used_target": None})
    if not isinstance(entry, dict):
        entry = {"targets": {}, "default_target": None, "last_used_target": None}
        capsules[key] = entry
    entry.setdefault("targets", {})
    if not isinstance(entry["targets"], dict):
        entry["targets"] = {}
    return entry


def list_configured_targets(capsule_root: Path) -> list[dict[str, Any]]:
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    targets: list[dict[str, Any]] = []
    for name, target in sorted(entry["targets"].items()):
        if not isinstance(target, dict):
            continue
        row = dict(target)
        row["name"] = str(name)
        row["default"] = str(entry.get("default_target") or "") == str(name)
        row["last_used"] = str(entry.get("last_used_target") or "") == str(name)
        targets.append(row)
    return targets


def save_configured_target(capsule_root: Path, target: dict[str, Any], *, make_default: bool = False) -> dict[str, Any]:
    name = str(target.get("name", "")).strip()
    if not name:
        raise ValueError("Target name cannot be empty.")
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    payload = dict(target)
    payload["name"] = name
    entry["targets"][name] = payload
    if make_default or len(entry["targets"]) == 1:
        entry["default_target"] = name
    save_publish_state(state)
    out = dict(payload)
    out["default"] = str(entry.get("default_target") or "") == name
    out["last_used"] = str(entry.get("last_used_target") or "") == name
    return out


def remove_configured_target(capsule_root: Path, name: str) -> dict[str, Any]:
    token = str(name).strip()
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    payload = entry["targets"].pop(token, None)
    if payload is None:
        raise ValueError(f"Unknown target '{token}'.")
    if entry.get("default_target") == token:
        entry["default_target"] = next(iter(sorted(entry["targets"].keys())), None)
    if entry.get("last_used_target") == token:
        entry["last_used_target"] = None
    save_publish_state(state)
    out = dict(payload)
    out["name"] = token
    return out


def set_default_target(capsule_root: Path, name: str) -> dict[str, Any]:
    token = str(name).strip()
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    payload = entry["targets"].get(token)
    if payload is None:
        raise ValueError(f"Unknown target '{token}'.")
    entry["default_target"] = token
    save_publish_state(state)
    out = dict(payload)
    out["name"] = token
    out["default"] = True
    out["last_used"] = str(entry.get("last_used_target") or "") == token
    return out


def mark_last_used_target(capsule_root: Path, name: str) -> None:
    token = str(name).strip()
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    entry["last_used_target"] = token or None
    save_publish_state(state)


def get_target_preferences(capsule_root: Path) -> tuple[str | None, str | None]:
    state = load_publish_state()
    entry = _capsule_state_entry(state, capsule_root)
    default_target = str(entry.get("default_target") or "").strip() or None
    last_used_target = str(entry.get("last_used_target") or "").strip() or None
    return default_target, last_used_target


def git_repo_root(path: Path) -> Path:
    return Path(_run(["git", "-C", str(path), "rev-parse", "--show-toplevel"])).resolve()


def git_current_branch(path: Path) -> str:
    return str(_run(["git", "-C", str(path), "branch", "--show-current"])).strip()


def git_origin_url(path: Path) -> str | None:
    try:
        token = _run(["git", "-C", str(path), "remote", "get-url", "origin"]).strip()
    except RuntimeError:
        return None
    return token or None


def parse_owner_repo_from_origin(path: Path) -> tuple[str, str] | None:
    url = git_origin_url(path)
    if not url:
        return None
    try:
        return parse_owner_repo_from_url(url)
    except Exception:
        return None


def current_github_login() -> str | None:
    proc = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip()
    return token or None


def ensure_gh_auth() -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("`gh` is required for GitHub publish. Install GitHub CLI and run `gh auth login`.")
    proc = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "Run `gh auth login` before `lelabo push`."
        raise RuntimeError(err)


def repo_exists(owner: str, repo: str) -> bool:
    proc = subprocess.run(
        ["gh", "repo", "view", f"{owner}/{repo}", "--json", "name"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def create_repo(owner: str, repo: str, visibility: str) -> None:
    vis = str(visibility).strip().lower()
    flag = "--public" if vis == "public" else "--private"
    try:
        _run(["gh", "repo", "create", f"{owner}/{repo}", flag])
    except RuntimeError as exc:
        login = current_github_login()
        who = f"Authenticated GitHub user: {login}. " if login else ""
        hint = (
            f"{who}If you do not have permission on this owner, pass `--owner <your-github-login>` "
            "or update `lelabo config set github.owner <your-github-login>`."
        )
        raise RuntimeError(f"{exc}\n{hint}") from exc


def ensure_repo_manifest_for_capsule(capsule_root: Path) -> dict[str, Any]:
    root = capsule_root.expanduser().resolve()
    try:
        repo_root = git_repo_root(root)
    except RuntimeError as exc:
        raise RuntimeError(
            "This capsule is not inside a git repository. Create a publish target instead of using the workspace target."
        ) from exc
    init_gitspace(repo_root, name=repo_root.name)
    add_capsule_to_gitspace(root, gitspace_root=repo_root)
    return load_gitspace(repo_root)


def infer_workspace_target(capsule_root: Path) -> dict[str, Any] | None:
    root = capsule_root.expanduser().resolve()
    try:
        repo_root = git_repo_root(root)
    except RuntimeError:
        return None
    origin = parse_owner_repo_from_origin(repo_root)
    if origin is None:
        return None
    owner, repo = origin
    try:
        branch = git_current_branch(repo_root) or "main"
    except RuntimeError:
        branch = "main"
    rel = root.relative_to(repo_root).as_posix()
    return {
        "name": "workspace",
        "kind": "workspace",
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "path": rel,
        "repo_root": str(repo_root),
        "remote_url": git_origin_url(repo_root),
    }


def available_targets(capsule_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    workspace = infer_workspace_target(capsule_root)
    if workspace is not None:
        workspace["implicit"] = True
        workspace["default"] = False
        workspace["last_used"] = False
        out.append(workspace)
    configured = list_configured_targets(capsule_root)
    names = {str(item["name"]) for item in out}
    for item in configured:
        if str(item["name"]) in names:
            if str(item["name"]) == "workspace":
                continue
        out.append(item)
    return out


def resolve_target(capsule_root: Path, target_name: str) -> dict[str, Any]:
    token = str(target_name).strip()
    for item in available_targets(capsule_root):
        if str(item.get("name", "")).strip() == token:
            return item
    raise ValueError(f"Unknown publish target '{token}'.")


def preferred_target(capsule_root: Path) -> dict[str, Any] | None:
    targets = available_targets(capsule_root)
    if not targets:
        return None
    by_name = {str(item.get("name")): item for item in targets}
    default_target, last_used_target = get_target_preferences(capsule_root)
    if default_target and default_target in by_name:
        return by_name[default_target]
    if last_used_target and last_used_target in by_name:
        return by_name[last_used_target]
    if len(targets) == 1:
        return targets[0]
    return None


def auto_commit_message(capsule_root: Path) -> str:
    capsule_id = str(inspect_capsule_directory(capsule_root).get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    return f"Update {capsule_id}"


def _git_has_staged_changes(path: Path) -> bool:
    out = _run(["git", "-C", str(path), "diff", "--cached", "--name-only"])
    return bool(out.strip())


def _git_status_for_paths(path: Path, rel_paths: list[str]) -> bool:
    cmd = ["git", "-C", str(path), "status", "--porcelain", "--", *rel_paths]
    out = _run(cmd)
    return bool(out.strip())


def _git_stage_paths(path: Path, rel_paths: list[str]) -> None:
    _run(["git", "-C", str(path), "add", "-A", "--", *rel_paths])


def _git_commit(path: Path, message: str) -> bool:
    if not _git_has_staged_changes(path):
        return False
    _run(["git", "-C", str(path), "commit", "-m", message])
    return True


def push_workspace_target(
    *,
    capsule_root: Path,
    target: dict[str, Any],
    message: str | None,
    preview: bool,
) -> dict[str, Any]:
    root = capsule_root.expanduser().resolve()
    repo_root = Path(str(target.get("repo_root") or "")).expanduser().resolve() if str(target.get("repo_root") or "").strip() else git_repo_root(root)
    rel_paths = [root.relative_to(repo_root).as_posix(), ".lelabo/gitspace.toml"]
    target_branch = str(target.get("branch", "")).strip() or git_current_branch(repo_root) or "main"
    result = {
        "target_name": str(target.get("name", "workspace")),
        "target_kind": "workspace",
        "capsule_path": str(root),
        "repo_root": str(repo_root),
        "owner": target.get("owner"),
        "repo": target.get("repo"),
        "branch": target_branch,
        "path": rel_paths[0],
        "remote_url": target.get("remote_url") or git_origin_url(repo_root),
        "commit_message": str(message or "").strip() or auto_commit_message(root),
        "scoped_changes": False,
        "committed": False,
        "pushed": False,
        "preview": bool(preview),
    }
    if preview:
        result["scoped_changes"] = _git_status_for_paths(repo_root, [rel_paths[0]])
        return result
    manifest = ensure_repo_manifest_for_capsule(root)
    repo_root = Path(str(manifest["root"])).resolve()
    rel_paths = [root.relative_to(repo_root).as_posix(), ".lelabo/gitspace.toml"]
    if _git_has_staged_changes(repo_root):
        raise RuntimeError("The git index already has staged changes. Commit or unstage them before `lelabo push`.")
    scoped_dirty = _git_status_for_paths(repo_root, rel_paths)
    result["repo_root"] = str(repo_root)
    result["path"] = rel_paths[0]
    result["scoped_changes"] = bool(scoped_dirty)
    if scoped_dirty:
        _git_stage_paths(repo_root, rel_paths)
        try:
            result["committed"] = _git_commit(repo_root, result["commit_message"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Scoped changes were staged but commit failed. Configure git identity (`git config user.name/user.email`) "
                "and retry `lelabo push`."
            ) from exc
    remote_name = "origin"
    _run(["git", "-C", str(repo_root), "push", "-u", remote_name, f"HEAD:{target_branch}"])
    result["pushed"] = True
    mark_last_used_target(root, str(target.get("name", "workspace")))
    return result


def _clone_or_prepare_target_repo(
    *,
    owner: str,
    repo: str,
    branch: str,
    visibility: str,
    create_repo_if_missing: bool,
) -> tuple[Path, bool]:
    ensure_gh_auth()
    remote_url = f"https://github.com/{owner}/{repo}.git"
    created_repo = False
    if not repo_exists(owner, repo):
        if not create_repo_if_missing:
            raise RuntimeError(f"GitHub repo '{owner}/{repo}' does not exist.")
        create_repo(owner, repo, visibility)
        created_repo = True
    checkout = repo_checkout_cache_dir(owner, repo)
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", remote_url, str(checkout)])
    else:
        _run(["git", "-C", str(checkout), "fetch", "origin"])
    proc = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--verify", f"origin/{branch}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        _run(["git", "-C", str(checkout), "checkout", branch])
        _run(["git", "-C", str(checkout), "reset", "--hard", f"origin/{branch}"])
    else:
        proc_head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_head.returncode == 0:
            _run(["git", "-C", str(checkout), "checkout", "-B", branch])
        else:
            _run(["git", "-C", str(checkout), "checkout", "--orphan", branch])
            for child in checkout.iterdir():
                if child.name == ".git":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    return checkout, created_repo


def _copy_capsule_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(source, dest, ignore=ignore)


def push_github_target(
    *,
    capsule_root: Path,
    target: dict[str, Any],
    message: str | None,
    preview: bool,
    create_repo_if_missing: bool,
) -> dict[str, Any]:
    root = capsule_root.expanduser().resolve()
    capsule_info = inspect_capsule_directory(root)
    owner = str(target.get("owner", "")).strip()
    repo = str(target.get("repo", "")).strip()
    if not owner or not repo:
        raise ValueError("GitHub target requires owner and repo.")
    branch = str(target.get("branch", "")).strip() or "main"
    rel_path = str(target.get("path", "")).strip() or str(capsule_info.get("capsule_id", root.name))
    if rel_path in {"", "."}:
        raise ValueError("GitHub target path cannot be empty or '.'.")
    visibility = str(target.get("visibility", "private")).strip().lower() or "private"
    result = {
        "target_name": str(target.get("name", "")),
        "target_kind": "github",
        "capsule_path": str(root),
        "repo_root": str(repo_checkout_cache_dir(owner, repo)),
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "path": rel_path,
        "remote_url": f"https://github.com/{owner}/{repo}.git",
        "commit_message": str(message or "").strip() or auto_commit_message(root),
        "created_repo": False,
        "committed": False,
        "pushed": False,
        "preview": bool(preview),
    }
    if preview:
        return result
    checkout, created_repo = _clone_or_prepare_target_repo(
        owner=owner,
        repo=repo,
        branch=branch,
        visibility=visibility,
        create_repo_if_missing=create_repo_if_missing,
    )
    target_capsule_root = (checkout / rel_path).resolve()
    init_gitspace(checkout, name=repo)
    result["repo_root"] = str(checkout)
    result["created_repo"] = bool(created_repo)
    _copy_capsule_tree(root, target_capsule_root)
    add_capsule_to_gitspace(
        target_capsule_root,
        gitspace_root=checkout,
        capsule_id=str(capsule_info.get("capsule_id", root.name)),
    )
    rel_paths = [rel_path, ".lelabo/gitspace.toml"]
    _git_stage_paths(checkout, rel_paths)
    try:
        result["committed"] = _git_commit(checkout, result["commit_message"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Target repo update was staged but commit failed. Configure git identity (`git config user.name/user.email`) "
            "and retry `lelabo push`."
        ) from exc
    _run(["git", "-C", str(checkout), "push", "-u", "origin", f"HEAD:{branch}"])
    result["pushed"] = True
    mark_last_used_target(root, str(target.get("name", "")))
    return result


__all__ = [
    "auto_commit_message",
    "available_targets",
    "capsule_state_key",
    "current_github_login",
    "ensure_gh_auth",
    "ensure_repo_manifest_for_capsule",
    "get_target_preferences",
    "git_origin_url",
    "git_repo_root",
    "infer_workspace_target",
    "list_configured_targets",
    "load_publish_state",
    "mark_last_used_target",
    "parse_owner_repo_from_origin",
    "publish_state_path",
    "push_github_target",
    "push_workspace_target",
    "remove_configured_target",
    "repo_checkout_cache_dir",
    "resolve_target",
    "save_configured_target",
    "save_publish_state",
    "set_default_target",
]
