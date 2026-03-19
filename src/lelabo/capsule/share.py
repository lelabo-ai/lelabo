"""Share gitspaces and capsules to GitHub repositories."""

from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any

from .github import parse_owner_repo_from_url


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


def _init_git_repo(path: Path, *, initial_branch: str) -> None:
    branch = str(initial_branch).strip() or "main"
    _run(["git", "-C", str(path), "init", "-b", branch])
    _run(["git", "-C", str(path), "add", "-A"])
    try:
        _run(["git", "-C", str(path), "commit", "-m", "Initialize LeLabo gitspace for GitHub share"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Git repository initialized but initial commit failed. "
            "Configure git identity (`git config user.name/user.email`) and commit once, then retry share."
        ) from exc


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
        err = (proc.stderr or "").strip() or "Run `gh auth login` before `lelabo capsule share`."
        raise RuntimeError(err)


def _repo_exists(owner: str, repo: str) -> bool:
    proc = subprocess.run(
        ["gh", "repo", "view", f"{owner}/{repo}", "--json", "name"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _gh_current_login() -> str | None:
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


def _create_repo(owner: str, repo: str, visibility: str) -> None:
    vis = str(visibility).strip().lower()
    if vis not in {"public", "private"}:
        raise ValueError("visibility must be 'public' or 'private'.")
    flag = "--public" if vis == "public" else "--private"
    try:
        _run(["gh", "repo", "create", f"{owner}/{repo}", flag])
    except RuntimeError as exc:
        login = _gh_current_login()
        who = f"Authenticated GitHub user: {login}. " if login else ""
        hint = (
            f"{who}If you do not have permission on this owner, pass `--owner <your-github-login>` "
            "or update `lelabo config set github.owner <your-github-login>`."
        )
        raise RuntimeError(f"{exc}\n{hint}") from exc


def _push_refspec(path: Path, *, remote_target: str, refspec: str, branch: str) -> None:
    _run(["git", "-C", str(path), "push", "-u", remote_target, refspec])
    _run(["git", "-C", str(path), "fetch", remote_target, branch])


def current_github_login() -> str | None:
    """Return the currently authenticated GitHub login when available."""
    return _gh_current_login()


def git_repo_root(path: Path) -> Path:
    """Return the git repository root for one path."""
    return _git_repo_root(path)


def share_gitspace_github(
    *,
    gitspace_root: Path,
    owner: str,
    repo: str,
    branch: str | None,
    visibility: str,
    create_repo_if_missing: bool,
    default_branch: str = "main",
    auto_init_git: bool = False,
    assume_yes: bool = False,
) -> dict[str, Any]:
    """Push a clean gitspace repository to GitHub."""
    root = gitspace_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Gitspace root not found: {root}")
    if shutil.which("git") is None:
        raise RuntimeError("`git` is required for `lelabo capsule share`.")

    initialized_repo = False
    try:
        repo_root = _git_repo_root(root)
    except RuntimeError as exc:
        if not auto_init_git:
            raise RuntimeError(
                "GitHub share requires a git repository. "
                "Initialize git in your gitspace or pass `--yes` to let LeLabo bootstrap it."
            ) from exc
        if not assume_yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "No git repository found for this gitspace. "
                    "Run in interactive mode to confirm auto-init, or pass `--yes`."
                ) from exc
            answer = input(
                f"No git repository found for '{root}'. Initialize it and create an initial commit now? [y/N]: "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                raise RuntimeError("Git initialization canceled by user.") from exc
        _init_git_repo(root, initial_branch=default_branch)
        repo_root = root
        initialized_repo = True
    if repo_root != root:
        raise RuntimeError(
            f"Gitspace root '{root}' must be the git repository root, but git is initialized at '{repo_root}'."
        )

    _git_head_exists(repo_root)
    if not _git_is_clean(repo_root):
        raise RuntimeError("Git worktree is dirty. Commit or stash changes before sharing this gitspace.")

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

    origin_url = _git_origin_url(root)
    created_origin_remote = False
    if not origin_url:
        _run(["git", "-C", str(root), "remote", "add", "origin", remote_url])
        origin_url = remote_url
        created_origin_remote = True

    target_branch = str(branch or "").strip() or _git_current_branch(root) or str(default_branch).strip() or "main"
    remote_target = "origin" if origin_url == remote_url else remote_url
    _push_refspec(repo_root, remote_target=remote_target, refspec=f"HEAD:{target_branch}", branch=target_branch)

    return {
        "gitspace_root": str(root),
        "owner": owner,
        "repo": repo,
        "remote_url": remote_url,
        "branch": target_branch,
        "created_repo": bool(created_repo),
        "created_origin_remote": bool(created_origin_remote),
        "initialized_git_repo": bool(initialized_repo),
        "pushed": True,
    }


def share_capsule_github(**kwargs) -> dict[str, Any]:
    """Backward-compatible wrapper around gitspace GitHub share."""
    capsule_root = Path(kwargs.pop("capsule_root"))
    return share_gitspace_github(gitspace_root=capsule_root, **kwargs)


def parse_owner_repo_from_origin(path: Path) -> tuple[str, str] | None:
    """Parse `owner/repo` from `origin` when it points to GitHub."""
    url = _git_origin_url(path)
    if not url:
        return None
    try:
        return parse_owner_repo_from_url(url)
    except Exception:
        return None


__all__ = [
    "current_github_login",
    "git_repo_root",
    "share_gitspace_github",
    "share_capsule_github",
    "parse_owner_repo_from_origin",
]
