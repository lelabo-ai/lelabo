"""Share capsule workspaces to GitHub repositories."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .github import normalize_github_repo_url, parse_owner_repo_from_url


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


def _git_repo_root(path: Path) -> Path:
    root = _run(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    return Path(root).resolve()


def _git_head_exists(path: Path) -> None:
    _run(["git", "-C", str(path), "rev-parse", "--verify", "HEAD"])


def _git_is_clean(path: Path) -> bool:
    out = _run(["git", "-C", str(path), "status", "--porcelain"])
    return not bool(out.strip())


def _git_current_branch(path: Path) -> str:
    out = _run(["git", "-C", str(path), "branch", "--show-current"])
    return str(out).strip()


def _git_origin_url(path: Path) -> str | None:
    try:
        out = _run(["git", "-C", str(path), "remote", "get-url", "origin"])
    except RuntimeError:
        return None
    token = str(out).strip()
    return token or None


def _ensure_gh_auth() -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("`gh` is required for GitHub share. Install GitHub CLI and run `gh auth login`.")
    proc = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "Run `gh auth login` before `lelabo capsule share github`."
        raise RuntimeError(err)


def _repo_exists(owner: str, repo: str) -> bool:
    proc = subprocess.run(
        ["gh", "repo", "view", f"{owner}/{repo}", "--json", "name"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _create_repo(owner: str, repo: str, visibility: str) -> None:
    vis = str(visibility).strip().lower()
    if vis not in {"public", "private"}:
        raise ValueError("visibility must be 'public' or 'private'.")
    flag = "--public" if vis == "public" else "--private"
    _run(["gh", "repo", "create", f"{owner}/{repo}", flag, "--confirm"])


def _ensure_origin_remote(path: Path, remote_url: str) -> bool:
    existing = _git_origin_url(path)
    if existing is None:
        _run(["git", "-C", str(path), "remote", "add", "origin", remote_url])
        return True
    if normalize_github_repo_url(existing) != normalize_github_repo_url(remote_url):
        raise RuntimeError(
            "Git remote 'origin' points to a different repository. "
            "Refusing to overwrite remote configuration."
        )
    return False


def share_capsule_github(
    *,
    capsule_root: Path,
    owner: str,
    repo: str,
    branch: str,
    visibility: str,
    create_repo_if_missing: bool,
) -> dict[str, Any]:
    """Push a clean local capsule git repository to GitHub."""
    root = capsule_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Capsule root not found: {root}")
    if shutil.which("git") is None:
        raise RuntimeError("`git` is required for `lelabo capsule share github`.")

    repo_root = _git_repo_root(root)
    if repo_root != root:
        raise RuntimeError(
            f"Capsule root '{root}' is inside git repo '{repo_root}'. "
            "Run share from the git repository root to avoid partial pushes."
        )

    _git_head_exists(root)
    if not _git_is_clean(root):
        raise RuntimeError("Git worktree is dirty. Commit or stash changes before sharing.")

    _ensure_gh_auth()
    owner = str(owner).strip()
    repo = str(repo).strip()
    if not owner or not repo:
        raise ValueError("Both GitHub owner and repo are required.")

    remote_url = f"https://github.com/{owner}/{repo}.git"
    created_repo = False
    if not _repo_exists(owner, repo):
        if not bool(create_repo_if_missing):
            raise RuntimeError(f"GitHub repo '{owner}/{repo}' does not exist.")
        _create_repo(owner, repo, visibility)
        created_repo = True

    added_origin = _ensure_origin_remote(root, remote_url)

    target_branch = str(branch).strip() or _git_current_branch(root) or "main"
    _run(["git", "-C", str(root), "push", "-u", "origin", f"HEAD:{target_branch}"])

    return {
        "capsule_path": str(root),
        "owner": owner,
        "repo": repo,
        "remote_url": remote_url,
        "branch": target_branch,
        "created_repo": bool(created_repo),
        "created_origin_remote": bool(added_origin),
        "pushed": True,
    }


def parse_owner_repo_from_origin(path: Path) -> tuple[str, str] | None:
    """Parse `owner/repo` from `origin` when it points to GitHub."""
    url = _git_origin_url(path)
    if not url:
        return None
    try:
        return parse_owner_repo_from_url(url)
    except Exception:
        return None

