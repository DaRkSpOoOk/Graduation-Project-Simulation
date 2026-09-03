"""Explicit, reproducible seeding for TASK-009B.

Every source of randomness the training loop touches is seeded from one integer,
and the settings that make cuDNN reproducible are set explicitly rather than
left to whatever the environment defaults to.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, PyTorch and CUDA, and report what was set."""

    if not 0 <= seed < 2**32:
        raise ValueError("seed must fit in 32 bits")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN's autotuner picks different algorithms run to run; benchmark=False
        # plus deterministic=True trades a little speed for a reproducible result.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS needs this to make some reductions reproducible on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed."""

    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def random_state() -> dict[str, Any]:
    """Capture RNG state so an interrupted run resumes the same stream."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any] | None) -> bool:
    """Restore a captured RNG state; returns False when nothing was restored."""

    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except (RuntimeError, ValueError):
            # A different GPU count than the interrupted run: model and optimizer
            # state still resume correctly, only the CUDA stream differs.
            return False
    return True


__all__ = ["seed_everything", "worker_init_fn", "random_state", "restore_random_state"]
