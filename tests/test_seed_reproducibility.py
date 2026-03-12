from __future__ import annotations

import os
import random
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from conftest import REPO_ROOT, load_module_from_path


def _torch_import_healthy() -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


TORCH_OK = _torch_import_healthy()


def _load_seed_module() -> ModuleType:
    return load_module_from_path(
        "seed_module_for_tests",
        REPO_ROOT / "src" / "lelabo" / "core" / "seed.py",
    )


def _load_dataset_base_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    return load_module_from_path(
        "dataset_base_module_for_tests",
        REPO_ROOT / "src" / "lelabo" / "supervised" / "datasets" / "base.py",
    )


def test_derive_seed_is_stable_and_namespaced() -> None:
    seed_mod = _load_seed_module()
    a = seed_mod.derive_seed(123, "unit", "train")
    b = seed_mod.derive_seed(123, "unit", "train")
    c = seed_mod.derive_seed(123, "unit", "eval")
    assert a == b
    assert a != c


def test_seed_everything_strict_enables_deterministic_flags() -> None:
    if not TORCH_OK:
        pytest.skip("torch is unavailable in this environment")
    import torch

    seed_mod = _load_seed_module()

    state = seed_mod.seed_everything(2026, mode="strict")

    assert state.seed == 2026
    assert state.mode == "strict"
    assert os.environ.get("PYTHONHASHSEED") == "2026"
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        assert torch.are_deterministic_algorithms_enabled() is True
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    # Reset to a less strict mode to avoid leaking global torch flags to other tests.
    seed_mod.seed_everything(2026, mode="relaxed")


def test_make_loader_is_reproducible_with_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    if not TORCH_OK:
        pytest.skip("torch is unavailable in this environment")
    import torch

    base_mod = _load_dataset_base_module(monkeypatch)

    class _RandomDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, idx: int):
            # Worker seeding must stabilize all three RNG streams.
            return (
                int(idx),
                float(random.random()),
                float(np.random.random()),
                float(torch.rand(()).item()),
            )

    def collect(scope: str) -> list[tuple[int, float, float, float]]:
        loader = base_mod.make_loader(
            _RandomDataset(),
            batch_size=8,
            shuffle=True,
            seed=77,
            seed_scope=scope,
            num_workers=2,
            pin_memory=False,
        )
        out: list[tuple[int, float, float, float]] = []
        for idx, py_r, np_r, torch_r in loader:
            for i, a, b, c in zip(idx.tolist(), py_r.tolist(), np_r.tolist(), torch_r.tolist()):
                out.append((int(i), float(a), float(b), float(c)))
        return out

    rows_a = collect("unit.train")
    rows_b = collect("unit.train")
    rows_c = collect("unit.eval")

    assert rows_a == rows_b
    assert rows_a != rows_c


def test_training_non_regression_repro(monkeypatch: pytest.MonkeyPatch) -> None:
    if not TORCH_OK:
        pytest.skip("torch is unavailable in this environment")
    import torch

    seed_mod = _load_seed_module()
    base_mod = _load_dataset_base_module(monkeypatch)
    f = torch.nn.functional

    def run_once(seed: int):
        seed_mod.seed_everything(seed, mode="strict")
        x = torch.randn(96, 4)
        y = torch.randint(0, 3, (96,))
        ds = torch.utils.data.TensorDataset(x, y)
        loader = base_mod.make_loader(
            ds,
            batch_size=16,
            shuffle=True,
            seed=seed,
            seed_scope="unit.training",
            num_workers=0,
            pin_memory=False,
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.05)

        for _ in range(3):
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = f.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()

        return [p.detach().clone() for p in model.parameters()]

    params_a = run_once(2026)
    params_b = run_once(2026)
    params_c = run_once(2027)

    assert all(torch.equal(a, b) for a, b in zip(params_a, params_b))
    assert any(not torch.equal(a, c) for a, c in zip(params_a, params_c))
