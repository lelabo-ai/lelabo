"""GitHub helpers for capsule install/share workflows."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_HTTPS_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_SSH_RE = re.compile(r"^git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?$")
_OWNER_REPO_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


def is_github_repo_url(raw: str) -> bool:
    token = str(raw).strip()
    return bool(_HTTPS_RE.fullmatch(token) or _SSH_RE.fullmatch(token))


def normalize_github_repo_url(raw: str) -> str:
    token = str(raw).strip()
    match = _HTTPS_RE.fullmatch(token)
    if match:
        owner, repo = match.group(1), match.group(2)
        return f"https://github.com/{owner}/{repo}.git"
    match = _SSH_RE.fullmatch(token)
    if match:
        owner, repo = match.group(1), match.group(2)
        return f"https://github.com/{owner}/{repo}.git"
    raise ValueError(
        f"Unsupported GitHub repository URL '{raw}'. "
        "Expected 'https://github.com/<owner>/<repo>' or '.git' variant."
    )


def parse_owner_repo_spec(raw: str) -> tuple[str, str]:
    token = str(raw).strip()
    match = _OWNER_REPO_RE.fullmatch(token)
    if not match:
        raise ValueError(f"Invalid GitHub repo spec '{raw}'. Expected '<owner>/<repo>'.")
    return match.group(1), match.group(2)


def parse_owner_repo_from_url(raw: str) -> tuple[str, str]:
    token = str(raw).strip()
    match = _HTTPS_RE.fullmatch(token)
    if match:
        return match.group(1), match.group(2)
    match = _SSH_RE.fullmatch(token)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Could not parse GitHub owner/repo from URL '{raw}'.")


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


def clone_github_repo(
    *,
    repo_url: str,
    destination: Path,
    ref: str | None = None,
) -> dict[str, Any]:
    """Clone a GitHub repo into `destination` and optionally pin to `ref`."""
    if shutil.which("git") is None:
        raise RuntimeError("`git` is required for GitHub capsule install.")
    normalized = normalize_github_repo_url(repo_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", normalized, str(destination)])

    requested_ref = str(ref).strip() if ref else ""
    if requested_ref:
        try:
            _run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", requested_ref])
            _run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
        except RuntimeError as exc:
            raise ValueError(f"Failed to resolve git ref '{requested_ref}' for {repo_url}.") from exc

    resolved_ref = _run(["git", "-C", str(destination), "rev-parse", "HEAD"])
    owner, repo = parse_owner_repo_from_url(normalized)
    return {
        "repo_url": normalized,
        "owner": owner,
        "repo": repo,
        "requested_ref": requested_ref or None,
        "resolved_ref": resolved_ref,
    }

