"""TASK-009B checkpoint format.

A checkpoint records the *whole* contract it was trained under, not just weights.
Loading re-checks that contract and refuses a mismatch: a checkpoint trained on
``bend_only`` silently loaded into a ``full`` pipeline would produce confident
nonsense rather than an error, which is the failure mode this guards against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..data.contract import CONTRACT_VERSION
from ..models.lstm_baseline import LSTMBaseline, LSTMBaselineConfig

CHECKPOINT_SCHEMA_VERSION = "task009b_checkpoint_v1"


class CheckpointError(ValueError):
    """A checkpoint is unreadable or incompatible with the requested contract."""


@dataclass(frozen=True)
class ExperimentSpec:
    """The identity of one experiment: what was trained, on what, how."""

    feature_set: str
    quaternion_policy: str
    pooling: str
    fold: str
    seed: int
    input_policy: str = "values_and_feature_valid"
    contract_version: str = CONTRACT_VERSION

    def slug(self) -> str:
        """Deterministic directory name; same spec always maps to the same path."""

        quaternion = self.quaternion_policy if self.feature_set == "full" else "na"
        return f"{self.feature_set}__q-{quaternion}__{self.pooling}__fold{self.fold}__seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "quaternion_policy": self.quaternion_policy,
            "pooling": self.pooling,
            "fold": self.fold,
            "seed": self.seed,
            "input_policy": self.input_policy,
            "contract_version": self.contract_version,
        }


def save_checkpoint(
    path: str | Path,
    *,
    model: LSTMBaseline,
    spec: ExperimentSpec,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    epoch: int = 0,
    best_epoch: int = 0,
    best_metric: float = 0.0,
    best_metric_name: str = "validation_macro_f1",
    early_stopping_counter: int = 0,
    history: list[dict[str, Any]] | None = None,
    training_config: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically, so an interrupt cannot truncate it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract_version": spec.contract_version,
        "experiment": spec.to_dict(),
        "model_config": model.config.to_dict(),
        "input_dim": model.config.input_dim,
        "num_classes": model.config.num_classes,
        "model_state": model.state_dict(),
        "optimizer_state": dict(optimizer_state) if optimizer_state is not None else None,
        "scheduler_state": dict(scheduler_state) if scheduler_state is not None else None,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "early_stopping_counter": early_stopping_counter,
        "history": list(history or []),
        "training_config": dict(training_config or {}),
        "rng_state": dict(rng_state) if rng_state is not None else None,
        "extra": dict(extra or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expect: ExperimentSpec | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Read a checkpoint and verify its contract before anything trusts it."""

    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"no checkpoint at {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as error:  # noqa: BLE001 - a corrupt file raises many types
        raise CheckpointError(f"{source}: unreadable ({type(error).__name__}: {error})") from error
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointError(f"{source}: not a TASK-009B checkpoint")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"{source}: checkpoint schema {payload.get('schema_version')!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise CheckpointError(
            f"{source}: trained under input contract {payload.get('contract_version')!r}, "
            f"this code implements {CONTRACT_VERSION!r}"
        )
    if expect is not None:
        stored = payload.get("experiment", {})
        for field in ("feature_set", "quaternion_policy", "pooling", "input_policy"):
            # fold and seed identify the run; they are not correctness constraints
            # on loading, so a checkpoint may be evaluated against another fold
            # deliberately. The tensor-semantics fields may not differ.
            if field == "quaternion_policy" and stored.get("feature_set") != "full":
                continue
            if stored.get(field) != getattr(expect, field):
                raise CheckpointError(
                    f"{source}: {field} is {stored.get(field)!r}, expected {getattr(expect, field)!r}"
                )
    return payload


def rebuild_model(payload: Mapping[str, Any]) -> LSTMBaseline:
    """Reconstruct the exact model a checkpoint describes and load its weights."""

    config_fields = LSTMBaselineConfig.__dataclass_fields__
    stored = {k: v for k, v in dict(payload["model_config"]).items() if k in config_fields}
    config = LSTMBaselineConfig(**stored)
    if config.input_dim != int(payload["input_dim"]):
        raise CheckpointError(
            f"checkpoint input_dim {payload['input_dim']} != rebuilt {config.input_dim}"
        )
    model = LSTMBaseline(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "CheckpointError", "ExperimentSpec",
    "save_checkpoint", "load_checkpoint", "rebuild_model",
]
