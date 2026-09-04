"""TASK-009C: verify the persisted deployment artifact.

Reads only what the deployment run wrote. Trains nothing, loads no optimizer, and
never re-scores the model on the data it was fitted to and calls that a
performance estimate.

The verification answers one question -- is this a valid, reproducible,
correctly-labelled deployment artifact -- and deliberately not "how good is it",
because a model fitted on every available sequence has no held-out data from which
that question could be answered.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..data.contract import CONTRACT_VERSION, NUM_CLASSES
from .plan import (
    DEPLOYMENT_FOLD_TAG,
    DEPLOYMENT_TRAINING_ROLE,
    EPOCH_POLICY,
    EXPECTED_MODEL_CONFIG,
    EXPECTED_SAMPLES,
    EXPECTED_TRAINING_CONFIG,
    PRIMARY_QUATERNION_POLICY,
    load_plan,
)
from .train import DEPLOYMENT_CHECKPOINT, DEPLOYMENT_METRIC_NAME, RESUME_CHECKPOINT

REQUIRED_FILES: tuple[str, ...] = (
    "deployment_plan.json", "status.json", "history.json", "training_summary.json",
    DEPLOYMENT_CHECKPOINT, RESUME_CHECKPOINT,
)
# Files worth hashing for reproducibility. The checkpoint is the artifact; the
# other three are the record of how it was produced.
HASHED_FILES: tuple[str, ...] = (
    DEPLOYMENT_CHECKPOINT, "deployment_plan.json", "training_summary.json", "history.json",
)


class DeploymentVerificationError(ValueError):
    """The persisted deployment artifact is missing, incomplete or inconsistent."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(run_root: str | Path) -> dict[str, Any]:
    """Every expected artifact exists and is non-empty."""

    root = Path(run_root)
    present: dict[str, Any] = {}
    problems: list[str] = []
    for name in REQUIRED_FILES:
        path = root / name
        exists = path.is_file() and path.stat().st_size > 0
        present[name] = {"present": exists,
                         "bytes": path.stat().st_size if path.is_file() else 0}
        if not exists:
            problems.append(f"{name} is missing or empty")
    return {"run_root": str(root), "files": present, "problems": problems,
            "passed": not problems}


def verify_status(run_root: str | Path, expected_epochs: int) -> dict[str, Any]:
    """The run finished, ran the frozen budget exactly, and was not a smoke run."""

    path = Path(run_root) / "status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("status") != "COMPLETE":
        problems.append(f"status is {payload.get('status')!r}, not COMPLETE")
    if payload.get("training_role") != DEPLOYMENT_TRAINING_ROLE:
        problems.append(f"training_role is {payload.get('training_role')!r}")
    if payload.get("epochs_planned") != expected_epochs:
        problems.append(f"epochs_planned {payload.get('epochs_planned')} != {expected_epochs}")
    if payload.get("epochs_completed") != expected_epochs:
        problems.append(f"epochs_completed {payload.get('epochs_completed')} != {expected_epochs}")
    # A smoke run overrides the frozen budget or subsets the data; either would
    # make the artifact something other than the deployment model.
    if payload.get("smoke_mode") is not False:
        problems.append(f"smoke_mode is {payload.get('smoke_mode')!r}, expected False")
    return {**payload, "problems": problems, "passed": not problems}


def verify_history(run_root: str | Path, expected_epochs: int) -> dict[str, Any]:
    """Exactly the frozen number of epochs, contiguous, finite and non-duplicated."""

    records = json.loads((Path(run_root) / "history.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    epochs = [record.get("epoch") for record in records]
    if len(records) != expected_epochs:
        problems.append(f"{len(records)} history entries != {expected_epochs}")
    if epochs != list(range(1, expected_epochs + 1)):
        problems.append("epochs are not exactly 1..N contiguous")
    if len(epochs) != len(set(epochs)):
        problems.append("duplicate epoch numbers (a resume wrote an epoch twice)")

    losses: list[float] = []
    accuracies: list[float] = []
    for record in records:
        for field in ("train_loss", "train_accuracy", "epoch_seconds"):
            if field not in record:
                problems.append(f"epoch {record.get('epoch')}: missing {field}")
        loss = record.get("train_loss")
        accuracy = record.get("train_accuracy")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            problems.append(f"epoch {record.get('epoch')}: train_loss {loss!r} is not finite")
        else:
            losses.append(float(loss))
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
            problems.append(f"epoch {record.get('epoch')}: train_accuracy {accuracy!r} is not finite")
        elif not 0.0 <= accuracy <= 1.0:
            problems.append(f"epoch {record.get('epoch')}: train_accuracy {accuracy} outside [0, 1]")
        else:
            accuracies.append(float(accuracy))
        if loss is not None and isinstance(loss, (int, float)) and loss < 0:
            problems.append(f"epoch {record.get('epoch')}: negative loss")

    summary: dict[str, Any] = {"entries": len(records), "problems": problems,
                               "passed": not problems}
    if losses and accuracies:
        # Reported for description only. The deployment checkpoint is the FINAL
        # epoch because the budget was frozen before training; no epoch is
        # selected from these numbers.
        increases = sum(1 for i in range(len(losses) - 1) if losses[i + 1] > losses[i])
        summary.update({
            "initial_train_loss": losses[0], "final_train_loss": losses[-1],
            "minimum_train_loss": min(losses),
            "minimum_train_loss_epoch": int(losses.index(min(losses)) + 1),
            "initial_train_accuracy": accuracies[0], "final_train_accuracy": accuracies[-1],
            "maximum_train_accuracy": max(accuracies),
            "maximum_train_accuracy_epoch": int(accuracies.index(max(accuracies)) + 1),
            "epochs_with_loss_increase": increases,
            "total_epoch_seconds": sum(float(r.get("epoch_seconds", 0.0)) for r in records),
            "selection_note": (
                "these values describe the run; they did not select anything. The "
                "deployment checkpoint is epoch "
                f"{expected_epochs} because the budget was frozen beforehand."
            ),
        })
    return summary


def verify_plan(run_root: str | Path, committed_plan: str | Path | None = None) -> dict[str, Any]:
    """The persisted plan is the frozen recipe, and matches the committed copy."""

    plan = load_plan(Path(run_root) / "deployment_plan.json")
    configuration = plan["configuration"]
    training = plan["training_config"]
    audit = plan["dataset_audit"]
    problems: list[str] = []

    if plan.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version {plan.get('contract_version')!r}")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if configuration.get(key) != expected:
            problems.append(f"configuration.{key}={configuration.get(key)!r} != {expected!r}")
    if configuration.get("quaternion_policy") != PRIMARY_QUATERNION_POLICY:
        problems.append(f"quaternion_policy={configuration.get('quaternion_policy')!r}")
    for key, expected in EXPECTED_TRAINING_CONFIG.items():
        if training.get(key) != expected:
            problems.append(f"training_config.{key}={training.get(key)!r} != {expected!r}")
    if plan["epoch_budget"].get("policy") != EPOCH_POLICY:
        problems.append(f"epoch policy {plan['epoch_budget'].get('policy')!r}")
    if audit.get("indexed_samples") != EXPECTED_SAMPLES:
        problems.append(f"training samples {audit.get('indexed_samples')} != {EXPECTED_SAMPLES}")
    if audit.get("rejected_samples") != 0:
        problems.append(f"{audit.get('rejected_samples')} rejected samples")
    if sorted(audit.get("signers", {})) != ["01", "02", "03"]:
        problems.append(f"signers {sorted(audit.get('signers', {}))}")
    if audit.get("classes") != NUM_CLASSES:
        problems.append(f"{audit.get('classes')} classes != {NUM_CLASSES}")

    matches_committed = None
    if committed_plan is not None:
        committed = json.loads(Path(committed_plan).read_text(encoding="utf-8"))
        matches_committed = committed == plan
        if not matches_committed:
            problems.append("persisted plan differs from the committed frozen copy")

    return {
        "deployment_epochs": plan["epoch_budget"]["deployment_epochs"],
        "epoch_policy": plan["epoch_budget"]["policy"],
        "source_best_epochs": plan["epoch_budget"]["source_best_epochs"],
        "configuration": configuration,
        "training_config": training,
        "training_samples": audit.get("indexed_samples"),
        "signers": audit.get("signers"),
        "classes": audit.get("classes"),
        "matches_committed_plan": matches_committed,
        "problems": problems,
        "passed": not problems,
    }


def verify_checkpoint_metadata(payload: Mapping[str, Any], expected_epochs: int) -> dict[str, Any]:
    """The checkpoint describes itself honestly as an all-signers deployment fit."""

    problems: list[str] = []
    experiment = payload.get("experiment", {})
    extra = payload.get("extra", {})

    if payload.get("schema_version") != "task009b_checkpoint_v1":
        problems.append(f"schema_version {payload.get('schema_version')!r}")
    if payload.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version {payload.get('contract_version')!r}")
    if experiment.get("fold") != DEPLOYMENT_FOLD_TAG:
        problems.append(f"fold {experiment.get('fold')!r} != {DEPLOYMENT_FOLD_TAG!r}")
    if experiment.get("fold") in ("01", "02", "03"):
        problems.append("checkpoint is falsely labelled as a LOSO fold")
    for key, expected in (("feature_set", EXPECTED_MODEL_CONFIG["feature_set"]),
                          ("pooling", EXPECTED_MODEL_CONFIG["pooling"]),
                          ("input_policy", EXPECTED_MODEL_CONFIG["input_policy"]),
                          ("quaternion_policy", PRIMARY_QUATERNION_POLICY)):
        if experiment.get(key) != expected:
            problems.append(f"experiment.{key}={experiment.get(key)!r} != {expected!r}")
    if payload.get("input_dim") != 92:
        problems.append(f"input_dim {payload.get('input_dim')} != 92")
    if payload.get("num_classes") != NUM_CLASSES:
        problems.append(f"num_classes {payload.get('num_classes')} != {NUM_CLASSES}")
    if payload.get("epoch") != expected_epochs:
        problems.append(f"epoch {payload.get('epoch')} != {expected_epochs}")

    # The single most important honesty check: a fixed-budget model selected
    # nothing, so it must not claim a selection metric it never had.
    if payload.get("best_metric_name") != DEPLOYMENT_METRIC_NAME:
        problems.append(
            f"best_metric_name {payload.get('best_metric_name')!r} != {DEPLOYMENT_METRIC_NAME!r}")
    if payload.get("best_metric_name") == "validation_macro_f1":
        problems.append("checkpoint falsely claims a validation-selected metric")

    for key, expected in (("training_role", DEPLOYMENT_TRAINING_ROLE),
                          ("training_scope", "all_core28_sequences"),
                          ("training_samples", EXPECTED_SAMPLES),
                          ("epoch_policy", EPOCH_POLICY),
                          ("deployment_epochs", expected_epochs),
                          ("epochs_completed", expected_epochs),
                          ("classes", NUM_CLASSES),
                          ("selection_metric", DEPLOYMENT_METRIC_NAME)):
        if extra.get(key) != expected:
            problems.append(f"extra.{key}={extra.get(key)!r} != {expected!r}")
    if sorted(extra.get("signers", [])) != ["01", "02", "03"]:
        problems.append(f"extra.signers {extra.get('signers')!r}")
    if extra.get("source_primary_best_epochs") != {"01": 28, "02": 20, "03": 27}:
        problems.append(f"extra.source_primary_best_epochs {extra.get('source_primary_best_epochs')!r}")
    if "none" not in str(extra.get("held_out_data", "")).lower():
        problems.append(f"extra.held_out_data does not state that none was held out")
    if "POST-EVALUATION DEPLOYMENT TRAINING" not in str(extra.get("scientific_status", "")):
        problems.append("extra.scientific_status does not carry the deployment disclaimer")
    reference = extra.get("loso_reference", {})
    if not isinstance(reference, Mapping) or "mean_test_accuracy" not in reference:
        problems.append("extra.loso_reference is missing the TASK-009B generalization evidence")

    return {
        "schema_version": payload.get("schema_version"),
        "contract_version": payload.get("contract_version"),
        "experiment": dict(experiment),
        "input_dim": payload.get("input_dim"),
        "num_classes": payload.get("num_classes"),
        "epoch": payload.get("epoch"),
        "best_metric_name": payload.get("best_metric_name"),
        "extra": {k: v for k, v in extra.items() if k != "loso_reference"},
        "loso_reference": dict(reference) if isinstance(reference, Mapping) else None,
        "problems": problems,
        "passed": not problems,
    }


__all__ = [
    "REQUIRED_FILES", "HASHED_FILES", "DeploymentVerificationError", "sha256_file",
    "verify_files", "verify_status", "verify_history", "verify_plan",
    "verify_checkpoint_metadata",
]
