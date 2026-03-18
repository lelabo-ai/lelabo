"""Share capsule workspaces to GitHub repositories."""

from __future__ import annotations

import shutil
import subprocess
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


def _git_subtree_split(path: Path, *, prefix: str) -> str:
    out = _run(["git", "-C", str(path), "subtree", "split", "--prefix", prefix, "HEAD"])
    token = str(out).strip()
    if not token:
        raise RuntimeError(f"Could not compute subtree split for prefix '{prefix}'.")
    return token


def _init_git_repo(path: Path) -> None:
    _run(["git", "-C", str(path), "init", "-b", "main"])
    _run(["git", "-C", str(path), "add", "-A"])
    try:
        _run(["git", "-C", str(path), "commit", "-m", "Initialize capsule for GitHub share"])
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


def _push_refspec(path: Path, *, remote_url: str, refspec: str, branch: str) -> None:
    _run(["git", "-C", str(path), "push", "-u", remote_url, refspec])
    _run(["git", "-C", str(path), "fetch", remote_url, branch])


def share_capsule_github(
    *,
    capsule_root: Path,
    owner: str,
    repo: str,
    branch: str,
    visibility: str,
    create_repo_if_missing: bool,
    auto_init_git: bool = False,
    assume_yes: bool = False,
) -> dict[str, Any]:
    """Push a clean local capsule git repository to GitHub."""
    root = capsule_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Capsule root not found: {root}")
    if shutil.which("git") is None:
        raise RuntimeError("`git` is required for `lelabo capsule share github`.")

    initialized_repo = False
    try:
        repo_root = _git_repo_root(root)
    except RuntimeError as exc:
        if not auto_init_git:
            raise RuntimeError(
                "GitHub share requires a git repository. "
                "Initialize git in your workspace (e.g. `git init`) or use `lelabo capsule share --mode local`."
            ) from exc
        if not assume_yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "No git repository found for this capsule. "
                    "Run in interactive mode to confirm auto-init, or pass `--yes`."
                ) from exc
            answer = input(
                f"No git repository found for '{root}'. Initialize it and create an initial commit now? [y/N]: "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                raise RuntimeError("Git initialization canceled by user.") from exc
        _init_git_repo(root)
        repo_root = root
        initialized_repo = True
    if repo_root != root and repo_root not in root.parents:
        raise RuntimeError(
            f"Capsule root '{root}' is not inside git repo root '{repo_root}'."
        )

    _git_head_exists(repo_root)
    if not _git_is_clean(repo_root):
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

    target_branch = str(branch).strip() or _git_current_branch(root) or "main"
    if repo_root == root:
        refspec = f"HEAD:{target_branch}"
    else:
        prefix = root.relative_to(repo_root).as_posix()
        split_sha = _git_subtree_split(repo_root, prefix=prefix)
        refspec = f"{split_sha}:{target_branch}"
    _push_refspec(repo_root, remote_url=remote_url, refspec=refspec, branch=target_branch)

    return {
        "capsule_path": str(root),
        "workspace_path": str(repo_root),
        "owner": owner,
        "repo": repo,
        "remote_url": remote_url,
        "branch": target_branch,
        "created_repo": bool(created_repo),
        "created_origin_remote": False,
        "initialized_git_repo": bool(initialized_repo),
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
