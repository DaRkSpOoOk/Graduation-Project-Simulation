"""TASK-009C deployment plan: freeze the all-signers training recipe.

TASK-009B answered a scientific question -- how well does the representation
generalize to an unseen signer -- and that answer is frozen. TASK-009C answers a
different, engineering question: given that the configuration is already chosen,
how do we fit one final model on every available sequence for the demo?

The only genuinely new decision here is *how long to train*, because an all-data
model has no held-out validation split to early-stop on. Inventing a fresh 80/20
split would waste scarce data and would re-derive a stopping point from a
pseudo-test set. Instead the budget is taken from the three TASK-009B primary
LOSO runs, whose best epochs were selected on legitimate validation data before
any test set was touched.

The plan is written once and then treated as immutable: the epoch budget must not
be revisited after watching deployment training loss.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data.contract import CONTRACT_VERSION, NUM_CLASSES, feature_dimension
from ..models.lstm_baseline import model_input_dimension

PLAN_SCHEMA_VERSION = "task009c_deployment_plan_v1"

# The configuration selected by TASK-009B evidence. Frozen: TASK-009C performs no
# model selection of any kind.
PRIMARY_FEATURE_SET = "full"
PRIMARY_QUATERNION_POLICY = "absolute"
PRIMARY_POOLING = "masked_mean"
PRIMARY_INPUT_POLICY = "values_and_feature_valid"
PRIMARY_SEED = 1337
PRIMARY_FOLDS: tuple[str, ...] = ("01", "02", "03")

# Marks the checkpoint as an all-signers deployment fit rather than a LOSO fold.
DEPLOYMENT_FOLD_TAG = "all"
DEPLOYMENT_TRAINING_ROLE = "deployment_all_signers"
EPOCH_POLICY = "median_primary_loso_best_epoch"

EXPECTED_SAMPLES = 4222
EXPECTED_SIGNERS: tuple[str, ...] = ("01", "02", "03")

# Frozen TASK-009B training hyperparameters, re-verified against the persisted
# result metadata rather than retyped from memory.
EXPECTED_TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip_norm": 5.0,
    "batch_size": 32,
    "loss": "cross_entropy",
}
EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "feature_set": PRIMARY_FEATURE_SET,
    "input_policy": PRIMARY_INPUT_POLICY,
    "pooling": PRIMARY_POOLING,
    "hidden_size": 192,
    "num_layers": 2,
    "dropout": 0.3,
    "input_projection": None,
    "bidirectional": False,
    "num_classes": NUM_CLASSES,
}


class DeploymentPlanError(ValueError):
    """The deployment plan cannot be derived, or an existing plan is invalid."""


def primary_result_path(loso_run_root: str | Path, fold: str, seed: int = PRIMARY_SEED) -> Path:
    """Where TASK-009B persisted one primary-configuration fold result."""

    return (Path(loso_run_root) / PRIMARY_FEATURE_SET / f"q-{PRIMARY_QUATERNION_POLICY}" /
            PRIMARY_POOLING / f"fold{fold}" / f"seed{seed}" / "result.json")


def read_primary_loso_evidence(
    loso_run_root: str | Path, seed: int = PRIMARY_SEED
) -> dict[str, Any]:
    """Read the three primary LOSO results and verify they are what we think.

    Every configuration field is checked against the frozen expectation, so a
    plan can never be derived from a fold that was trained under a different
    recipe.
    """

    folds: dict[str, Any] = {}
    for fold in PRIMARY_FOLDS:
        path = primary_result_path(loso_run_root, fold, seed)
        if not path.is_file():
            raise DeploymentPlanError(f"missing TASK-009B primary result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

        experiment = payload.get("experiment", {})
        mismatches = []
        if payload.get("contract_version") != CONTRACT_VERSION:
            mismatches.append(f"contract_version={payload.get('contract_version')!r}")
        # The experiment block carries its own copy; both must agree, or the
        # result was written by a build whose tensor semantics we cannot vouch for.
        if experiment.get("contract_version") != CONTRACT_VERSION:
            mismatches.append(
                f"experiment.contract_version={experiment.get('contract_version')!r}")
        for field, expected in (
            ("feature_set", PRIMARY_FEATURE_SET),
            ("quaternion_policy", PRIMARY_QUATERNION_POLICY),
            ("pooling", PRIMARY_POOLING),
            ("input_policy", PRIMARY_INPUT_POLICY),
            ("fold", fold),
            ("seed", seed),
        ):
            if experiment.get(field) != expected:
                mismatches.append(f"{field}={experiment.get(field)!r} != {expected!r}")
        if mismatches:
            raise DeploymentPlanError(
                f"{path} is not the frozen primary configuration: {'; '.join(mismatches)}")

        best_epoch = payload.get("best_epoch")
        if not isinstance(best_epoch, int) or best_epoch < 1:
            raise DeploymentPlanError(f"{path}: best_epoch {best_epoch!r} is not a positive integer")

        folds[fold] = {
            "result_path": str(path),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": payload.get("best_validation_metric"),
            "epochs_completed": payload.get("training", {}).get("epochs_completed"),
            "stopped_early": payload.get("training", {}).get("stopped_early"),
            "train_samples": payload.get("split_sizes", {}).get("train"),
            "test_accuracy": payload.get("test_metrics", {}).get("accuracy"),
            "test_macro_f1": payload.get("test_metrics", {}).get("macro_f1"),
            "training_config": payload.get("training", {}).get("training_config", {}),
            "model_config": payload.get("model_config", {}),
        }
    return folds


def _consistent(folds: Mapping[str, Any], key: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Check that all three folds agree with each other and with expectation."""

    problems: list[str] = []
    for name, value in expected.items():
        observed = {fold: folds[fold][key].get(name) for fold in folds}
        distinct = {json.dumps(v, sort_keys=True) for v in observed.values()}
        if len(distinct) != 1:
            problems.append(f"{name} differs between folds: {observed}")
        elif next(iter(observed.values())) != value:
            problems.append(f"{name}={next(iter(observed.values()))!r} != expected {value!r}")
    return {"agrees": not problems, "problems": problems}


def derive_epoch_budget(best_epochs: Sequence[int]) -> dict[str, Any]:
    """The frozen epoch rule: median of the primary LOSO best epochs.

    Deterministic rounding: an odd count yields the exact middle value; an even
    count yields the mean of the two middle values rounded half-up
    (``floor(x + 0.5)``). Stated explicitly so the rule is reproducible rather
    than dependent on a library's tie-breaking.
    """

    values = [int(v) for v in best_epochs]
    if not values:
        raise DeploymentPlanError("cannot derive an epoch budget from no folds")
    if any(v < 1 for v in values):
        raise DeploymentPlanError(f"best epochs must be positive: {values}")
    raw = float(statistics.median(values))
    budget = int(math.floor(raw + 0.5))
    return {
        "policy": EPOCH_POLICY,
        "source_best_epochs": {fold: value for fold, value in zip(PRIMARY_FOLDS, values)}
        if len(values) == len(PRIMARY_FOLDS) else {str(i): v for i, v in enumerate(values)},
        "sorted_best_epochs": sorted(values),
        "raw_median": raw,
        "rounding_rule": "odd count: exact middle; even count: mean of middles, floor(x + 0.5)",
        "deployment_epochs": budget,
    }


def audit_deployment_dataset(
    index_path: str | Path, data_run_root: str | Path, *, load_every_sequence: bool = True
) -> dict[str, Any]:
    """Verify all 4,222 sequences are present, unique and loadable.

    ``load_every_sequence`` actually tensorizes each sequence through the frozen
    TASK-009A path. It costs seconds and is the only way to assert 0 rejected
    samples rather than assume it.
    """

    from ..data import (
        SequenceContractError,
        SequenceInputConfig,
        VirtualGloveSequenceDataset,
        load_index,
    )

    records = load_index(index_path)
    sample_ids = [record.sample_id for record in records]
    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1}) \
        if len(set(sample_ids)) != len(sample_ids) else []
    signers = Counter(record.signer_id for record in records)
    classes = Counter(record.label_index for record in records)

    config = SequenceInputConfig(
        feature_set=PRIMARY_FEATURE_SET,
        quaternion_policy=PRIMARY_QUATERNION_POLICY,
        verify_layout="first",
    )
    rejected: list[dict[str, str]] = []
    loaded = 0
    observed_dims: set[int] = set()
    if load_every_sequence:
        dataset = VirtualGloveSequenceDataset(records, data_run_root, config)
        for position in range(len(dataset)):
            try:
                item = dataset[position]
            except SequenceContractError as error:
                rejected.append({"sample_id": records[position].sample_id, "error": str(error)})
                continue
            observed_dims.add(int(item["values"].shape[1]))
            loaded += 1

    expected_feature_dim = feature_dimension(PRIMARY_FEATURE_SET)
    expected_input_dim = model_input_dimension(PRIMARY_FEATURE_SET, PRIMARY_INPUT_POLICY)
    problems: list[str] = []
    if len(records) != EXPECTED_SAMPLES:
        problems.append(f"{len(records)} indexed samples != {EXPECTED_SAMPLES}")
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate sample_id values")
    if sorted(signers) != list(EXPECTED_SIGNERS):
        problems.append(f"signers {sorted(signers)} != {list(EXPECTED_SIGNERS)}")
    if len(classes) != NUM_CLASSES or sorted(classes) != list(range(NUM_CLASSES)):
        problems.append(f"{len(classes)} distinct label_index values != {NUM_CLASSES} contiguous")
    if rejected:
        problems.append(f"{len(rejected)} sequences were rejected")
    if load_every_sequence and observed_dims != {expected_feature_dim}:
        problems.append(f"observed feature dimensions {sorted(observed_dims)} != {expected_feature_dim}")

    return {
        "index": str(index_path),
        "data_run_root": str(data_run_root),
        "indexed_samples": len(records),
        "expected_samples": EXPECTED_SAMPLES,
        "loaded_samples": loaded if load_every_sequence else None,
        "rejected_samples": len(rejected),
        "rejections": rejected[:20],
        "duplicate_sample_ids": duplicates[:20],
        "signers": dict(sorted(signers.items())),
        "classes": len(classes),
        "samples_per_class": {str(k): v for k, v in sorted(classes.items())},
        "contract_version": CONTRACT_VERSION,
        "feature_dimension": expected_feature_dim,
        "model_input_dimension": expected_input_dim,
        "sequences_verified_by_loading": load_every_sequence,
        "problems": problems,
        "passed": not problems,
    }


def build_deployment_plan(
    *,
    loso_run_root: str | Path,
    index_path: str | Path,
    data_run_root: str | Path,
    seed: int = PRIMARY_SEED,
    batch_size: int = 32,
    task009b_analysis_commit: str = "",
    load_every_sequence: bool = True,
) -> dict[str, Any]:
    """Derive the complete frozen deployment recipe."""

    folds = read_primary_loso_evidence(loso_run_root, seed)
    training_agreement = _consistent(folds, "training_config", EXPECTED_TRAINING_CONFIG)
    model_agreement = _consistent(folds, "model_config", EXPECTED_MODEL_CONFIG)
    if not training_agreement["agrees"]:
        raise DeploymentPlanError(
            "TASK-009B primary training config does not match the frozen expectation: "
            + "; ".join(training_agreement["problems"]))
    if not model_agreement["agrees"]:
        raise DeploymentPlanError(
            "TASK-009B primary model config does not match the frozen expectation: "
            + "; ".join(model_agreement["problems"]))

    epochs = derive_epoch_budget([folds[fold]["best_epoch"] for fold in PRIMARY_FOLDS])
    audit = audit_deployment_dataset(index_path, data_run_root,
                                     load_every_sequence=load_every_sequence)
    if not audit["passed"]:
        raise DeploymentPlanError(
            "deployment dataset audit failed: " + "; ".join(audit["problems"]))

    loso_reference = {
        "mean_test_accuracy": statistics.fmean(
            folds[fold]["test_accuracy"] for fold in PRIMARY_FOLDS),
        "mean_test_macro_f1": statistics.fmean(
            folds[fold]["test_macro_f1"] for fold in PRIMARY_FOLDS),
        "per_fold": {fold: {"accuracy": folds[fold]["test_accuracy"],
                            "macro_f1": folds[fold]["test_macro_f1"]}
                     for fold in PRIMARY_FOLDS},
        "note": (
            "This is the TASK-009B held-out-signer result for this configuration. It is "
            "the ONLY generalization evidence for the deployment model, which by "
            "construction has no held-out signer of its own."
        ),
    }
    mean_fold_train = statistics.fmean(
        folds[fold]["train_samples"] for fold in PRIMARY_FOLDS)

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "task": "TASK-009C",
        "purpose": "all-signers deployment training of the TASK-009B primary configuration",
        "contract_version": CONTRACT_VERSION,
        "training_role": DEPLOYMENT_TRAINING_ROLE,
        "training_scope": "all_core28_sequences",
        "configuration": {
            "feature_set": PRIMARY_FEATURE_SET,
            "quaternion_policy": PRIMARY_QUATERNION_POLICY,
            "pooling": PRIMARY_POOLING,
            "input_policy": PRIMARY_INPUT_POLICY,
            "selected_by": "TASK-009B LOSO evidence; TASK-009C performs no model selection",
            **EXPECTED_MODEL_CONFIG,
            "model_input_dimension": audit["model_input_dimension"],
        },
        "training_config": {**EXPECTED_TRAINING_CONFIG, "batch_size": batch_size,
                            "epochs": epochs["deployment_epochs"], "seed": seed},
        "epoch_budget": epochs,
        "epoch_budget_rationale": (
            "An all-data model has no held-out validation split, so early stopping is "
            "impossible without either withholding scarce data or peeking at a pseudo-test "
            "set. The three TASK-009B primary folds each chose a best epoch on legitimate "
            "validation data before any test evaluation, so their median is a stopping "
            "point derived entirely from already-spent evidence."
        ),
        "epoch_budget_caveat": (
            f"The LOSO best epochs were measured on ~{mean_fold_train:.0f} training "
            f"sequences per fold; deployment trains on {audit['indexed_samples']} "
            f"({audit['indexed_samples'] / mean_fold_train:.2f}x more), so an epoch here is "
            "that much larger in gradient steps. The frozen rule is a median of EPOCHS, not "
            "of steps, and is applied as specified; a step-matched budget would be smaller. "
            "This is disclosed rather than silently adjusted."
        ),
        "loso_reference": loso_reference,
        "primary_loso_folds": folds,
        "config_agreement": {"training": training_agreement, "model": model_agreement},
        "dataset_audit": audit,
        "source_task009b_analysis_commit": task009b_analysis_commit,
        "deployment_fold_tag": DEPLOYMENT_FOLD_TAG,
        "scientific_status": (
            "POST-EVALUATION DEPLOYMENT TRAINING. This model is fitted on every available "
            "sequence and therefore has NO held-out data. Its training accuracy is not a "
            "performance estimate, and it must never be described as independently tested. "
            "The recognition result for this project remains the TASK-009B LOSO evidence."
        ),
    }


def write_plan(path: str | Path, plan: Mapping[str, Any], *, force: bool = False) -> Path:
    """Write the plan once. An existing plan is immutable unless forced."""

    destination = Path(path)
    if destination.is_file() and not force:
        raise DeploymentPlanError(
            f"{destination} already exists; the frozen plan is immutable. Pass --force-plan "
            "only if you intend to discard the existing recipe."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(plan), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    return destination


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a frozen plan and check it is one, and one this code understands."""

    source = Path(path)
    if not source.is_file():
        raise DeploymentPlanError(
            f"no deployment plan at {source}; run the CLI with --prepare first")
    try:
        plan = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DeploymentPlanError(f"{source} is malformed: {error}") from error
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise DeploymentPlanError(
            f"{source}: plan schema {plan.get('schema_version')!r} != {PLAN_SCHEMA_VERSION!r}")
    if plan.get("contract_version") != CONTRACT_VERSION:
        raise DeploymentPlanError(
            f"{source}: plan contract {plan.get('contract_version')!r} != {CONTRACT_VERSION!r}")
    epochs = plan.get("epoch_budget", {}).get("deployment_epochs")
    if not isinstance(epochs, int) or epochs < 1:
        raise DeploymentPlanError(f"{source}: deployment_epochs {epochs!r} is not a positive integer")
    return plan


__all__ = [
    "PLAN_SCHEMA_VERSION", "PRIMARY_FEATURE_SET", "PRIMARY_QUATERNION_POLICY",
    "PRIMARY_POOLING", "PRIMARY_INPUT_POLICY", "PRIMARY_SEED", "PRIMARY_FOLDS",
    "DEPLOYMENT_FOLD_TAG", "DEPLOYMENT_TRAINING_ROLE", "EPOCH_POLICY", "EXPECTED_SAMPLES",
    "EXPECTED_TRAINING_CONFIG", "EXPECTED_MODEL_CONFIG", "DeploymentPlanError",
    "primary_result_path", "read_primary_loso_evidence", "derive_epoch_budget",
    "audit_deployment_dataset", "build_deployment_plan", "write_plan", "load_plan",
]
