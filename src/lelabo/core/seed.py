"""Reproducibility helpers for RNG seeding and determinism policy."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal

import numpy as np


DeterminismMode = Literal["off", "relaxed", "strict"]
_VALID_MODES: set[str] = {"off", "relaxed", "strict"}


def _require_torch():
    """Import torch lazily so non-torch helpers stay lightweight."""
    import torch  # local import: keeps non-torch utilities usable in lightweight environments.

    return torch


@dataclass(frozen=True)
class SeedState:
    """Summary of the effective seeding and deterministic-backend configuration."""

    seed: int
    mode: DeterminismMode
    deterministic_algorithms: bool
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None
    cublas_workspace_config: str | None


def normalize_determinism(mode: str | None, *, deterministic: bool | None = None) -> DeterminismMode:
    """Normalize legacy and modern determinism flags to one supported mode."""
    if mode is None:
        if deterministic is None:
            return "relaxed"
        return "strict" if bool(deterministic) else "relaxed"

    raw = str(mode).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "none": "off",
        "1": "strict",
        "true": "strict",
        "on": "strict",
    }
    raw = aliases.get(raw, raw)
    if raw not in _VALID_MODES:
        raise ValueError(f"Unsupported determinism mode '{mode}'. Expected one of: {sorted(_VALID_MODES)}")
    return raw  # type: ignore[return-value]


def derive_seed(seed: int, *parts: Any) -> int:
    """
    Deterministically derive a child seed from a root seed + namespace parts.
    """
    root = int(seed)
    if not parts:
        return root

    payload = "::".join([str(root), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Keep a positive 31-bit range to stay friendly with downstream RNG consumers.
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a ``torch.Generator`` seeded with ``seed``."""
    torch = _require_torch()
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Build a DataLoader worker-init function derived from ``seed``."""
    return partial(_seed_worker_init_impl, base_seed=int(seed))


def _seed_worker_init_impl(worker_id: int, *, base_seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs inside one DataLoader worker."""
    torch = _require_torch()
    worker_seed = derive_seed(base_seed, "worker", int(worker_id))
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def make_dataloader_seeding(seed: int, *, scope: str) -> tuple[torch.Generator, Callable[[int], None], int]:
    """Derive a namespaced DataLoader seed triple for one loader scope."""
    loader_seed = derive_seed(seed, "dataloader", scope)
    generator = make_torch_generator(loader_seed)
    worker_init_fn = make_worker_init_fn(loader_seed)
    return generator, worker_init_fn, loader_seed


def _set_deterministic_algorithms(enabled: bool) -> bool:
    """Enable or disable torch deterministic algorithms and report the effective state."""
    torch = _require_torch()
    try:
        torch.use_deterministic_algorithms(enabled)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to set torch deterministic algorithms to {bool(enabled)}."
        ) from exc

    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        return bool(torch.are_deterministic_algorithms_enabled())
    return bool(enabled)


def _configure_torch_backend(mode: DeterminismMode) -> SeedState:
    """Apply backend-level deterministic settings for one determinism mode."""
    torch = _require_torch()
    if mode == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        deterministic_algorithms = _set_deterministic_algorithms(True)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
    else:
        deterministic_algorithms = _set_deterministic_algorithms(False)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = (mode == "off")
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    if hasattr(torch.backends, "cudnn"):
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)

    return SeedState(
        seed=0,  # overwritten by caller
        mode=mode,
        deterministic_algorithms=deterministic_algorithms,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def seed_everything(
    seed: int,
    deterministic: bool | None = None,
    *,
    mode: str | None = None,
) -> SeedState:
    """
    Seed Python/NumPy/PyTorch and configure deterministic behavior.

    The returned ``SeedState`` captures the effective backend configuration.
    """
    root = int(seed)
    resolved_mode = normalize_determinism(mode, deterministic=deterministic)
    torch = _require_torch()

    os.environ["PYTHONHASHSEED"] = str(root)

    random.seed(root)
    np.random.seed(root % (2**32))
    torch.manual_seed(root)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(root)

    state = _configure_torch_backend(resolved_mode)
    return SeedState(
        seed=root,
        mode=resolved_mode,
        deterministic_algorithms=state.deterministic_algorithms,
        cudnn_deterministic=state.cudnn_deterministic,
        cudnn_benchmark=state.cudnn_benchmark,
        cublas_workspace_config=state.cublas_workspace_config,
    )
