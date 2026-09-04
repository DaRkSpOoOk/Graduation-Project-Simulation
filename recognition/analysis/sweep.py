"""TASK-009B sweep analysis: audit and aggregate the persisted experiment outputs.

Reads only what training already wrote. Nothing here trains, retrains, mutates a
checkpoint or touches the frozen TASK-008 production tree.

Aggregation uses pandas; the project-specific parts -- knowing which 15
experiments must exist, what a valid contract looks like, and how to compare
configurations that differ in exactly one factor -- are ours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.contract import CONTRACT_VERSION, NUM_CLASSES

FOLDS: tuple[str, ...] = ("01", "02", "03")
SEED = 1337

# The pre-registered matrix. `config` is the short name used in every table.
# `quaternion` is the directory tag, which is "na" where the feature set has no
# quaternion channels at all.
EXPECTED_MATRIX: tuple[dict[str, str], ...] = (
    {"config": "bend_only", "feature_set": "bend_only", "quaternion": "na",
     "pooling": "masked_mean"},
    {"config": "bend_spread", "feature_set": "bend_spread", "quaternion": "na",
     "pooling": "masked_mean"},
    {"config": "full_absolute_masked_mean", "feature_set": "full", "quaternion": "absolute",
     "pooling": "masked_mean"},
    {"config": "full_relative_masked_mean", "feature_set": "full",
     "quaternion": "relative_first_valid", "pooling": "masked_mean"},
    {"config": "full_absolute_final_hidden", "feature_set": "full", "quaternion": "absolute",
     "pooling": "final_hidden"},
)
# Pre-registered before any test score was seen; not re-chosen afterwards.
PRIMARY_CONFIG = "full_absolute_masked_mean"

REQUIRED_FILES: tuple[str, ...] = ("status.json", "history.json", "result.json",
                                   "predictions.json")
EXPECTED_CHECKPOINTS: tuple[str, ...] = ("best.pt", "last.pt")


class SweepAuditError(ValueError):
    """A persisted experiment is missing, incomplete or self-inconsistent."""


def experiment_dir(run_root: str | Path, entry: Mapping[str, str], fold: str,
                   seed: int = SEED) -> Path:
    return (Path(run_root) / entry["feature_set"] / f"q-{entry['quaternion']}" /
            entry["pooling"] / f"fold{fold}" / f"seed{seed}")


def audit_run_root(run_root: str | Path, seed: int = SEED) -> dict[str, Any]:
    """Check every expected experiment exists, is COMPLETE and is self-consistent.

    Nothing is repaired. A problem is recorded and surfaced, because silently
    patching a broken artifact would make the report describe a run that never
    happened.
    """

    root = Path(run_root)
    experiments: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []

    for entry in EXPECTED_MATRIX:
        for fold in FOLDS:
            directory = experiment_dir(root, entry, fold, seed)
            record: dict[str, Any] = {
                "config": entry["config"], "fold": fold, "seed": seed,
                "directory": str(directory),
                "files_present": {}, "checkpoints_present": {}, "status": None,
            }
            if not directory.is_dir():
                problems.append({"config": entry["config"], "fold": fold,
                                 "problem": "experiment directory is missing"})
                record["complete"] = False
                experiments.append(record)
                continue

            for name in REQUIRED_FILES:
                path = directory / name
                record["files_present"][name] = path.is_file() and path.stat().st_size > 0
                if not record["files_present"][name]:
                    problems.append({"config": entry["config"], "fold": fold,
                                     "problem": f"{name} is missing or empty"})
            for name in EXPECTED_CHECKPOINTS:
                path = directory / name
                record["checkpoints_present"][name] = path.is_file()
                record[f"{name}_bytes"] = path.stat().st_size if path.is_file() else 0
                if not path.is_file():
                    problems.append({"config": entry["config"], "fold": fold,
                                     "problem": f"{name} is missing"})

            if record["files_present"].get("result.json"):
                try:
                    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    problems.append({"config": entry["config"], "fold": fold,
                                     "problem": f"result.json is malformed: {error}"})
                    record["complete"] = False
                    experiments.append(record)
                    continue
                experiment = payload.get("experiment", {})
                checks = {
                    "contract_version": payload.get("contract_version") == CONTRACT_VERSION,
                    "experiment_contract_version": (
                        experiment.get("contract_version") == CONTRACT_VERSION),
                    "feature_set": experiment.get("feature_set") == entry["feature_set"],
                    "pooling": experiment.get("pooling") == entry["pooling"],
                    "fold": experiment.get("fold") == fold,
                    "seed": experiment.get("seed") == seed,
                    "input_policy": (
                        experiment.get("input_policy") == "values_and_feature_valid"),
                    "num_classes": payload.get("model_config", {}).get("num_classes") == NUM_CLASSES,
                }
                # A quaternion-free feature set has no quaternion channels, so the
                # directory tag is "na" while the config object may still carry the
                # CLI default. That is a non-applicable metadata field, not a
                # scientific quaternion experiment -- flagged, never silently
                # treated as a defect.
                if entry["quaternion"] == "na":
                    record["quaternion_policy_recorded"] = experiment.get("quaternion_policy")
                    record["quaternion_policy_applicable"] = False
                else:
                    checks["quaternion_policy"] = (
                        experiment.get("quaternion_policy") == entry["quaternion"])
                    record["quaternion_policy_applicable"] = True
                record["metadata_checks"] = checks
                for name, ok in checks.items():
                    if not ok:
                        problems.append({"config": entry["config"], "fold": fold,
                                         "problem": f"metadata mismatch: {name}"})

            if record["files_present"].get("status.json"):
                status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                record["status"] = status.get("status")
                if status.get("status") != "COMPLETE":
                    problems.append({"config": entry["config"], "fold": fold,
                                     "problem": f"status is {status.get('status')!r}"})

            record["complete"] = (
                record["status"] == "COMPLETE"
                and all(record["files_present"].values())
                and all(record["checkpoints_present"].values())
                and all(record.get("metadata_checks", {}).values())
            )
            experiments.append(record)

    complete = sum(1 for record in experiments if record.get("complete"))
    control_path = root / "duration_controls" / "control_a_length_only.json"
    return {
        "run_root": str(root),
        "expected_experiments": len(EXPECTED_MATRIX) * len(FOLDS),
        "complete_experiments": complete,
        "all_complete": complete == len(EXPECTED_MATRIX) * len(FOLDS),
        "contract_version": CONTRACT_VERSION,
        "seed": seed,
        "problems": problems,
        "problem_count": len(problems),
        "duration_control_a_present": control_path.is_file(),
        "experiments": experiments,
    }


def load_result(run_root: str | Path, entry: Mapping[str, str], fold: str,
                seed: int = SEED) -> dict[str, Any]:
    path = experiment_dir(run_root, entry, fold, seed) / "result.json"
    if not path.is_file():
        raise SweepAuditError(f"missing result.json for {entry['config']} fold {fold}")
    return json.loads(path.read_text(encoding="utf-8"))


def fold_table(run_root: str | Path, seed: int = SEED) -> pd.DataFrame:
    """One row per (config, fold) with the headline metrics and run facts."""

    rows: list[dict[str, Any]] = []
    for entry in EXPECTED_MATRIX:
        for fold in FOLDS:
            payload = load_result(run_root, entry, fold, seed)
            test = payload["test_metrics"]
            validation = payload["validation_metrics"]
            training = payload.get("training", {})
            lengths = payload.get("sequence_lengths", {})
            rows.append({
                "config": entry["config"],
                "feature_set": entry["feature_set"],
                "quaternion_policy_applied": (entry["quaternion"]
                                              if entry["quaternion"] != "na" else None),
                "pooling": entry["pooling"],
                "fold": fold,
                "seed": seed,
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_macro_precision": test["macro_precision"],
                "test_macro_recall": test["macro_recall"],
                "test_weighted_f1": test["weighted_f1"],
                "test_cross_entropy": test["cross_entropy"],
                "test_samples": test["samples"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "best_validation_macro_f1": payload["best_validation_metric"],
                "domain_gap_macro_f1": payload["best_validation_metric"] - test["macro_f1"],
                "best_epoch": payload["best_epoch"],
                "epochs_completed": training.get("epochs_completed"),
                "stopped_early": training.get("stopped_early"),
                "wall_seconds": training.get("wall_seconds"),
                "seconds_per_epoch": training.get("seconds_per_epoch"),
                "peak_gpu_mib": training.get("peak_gpu_mib"),
                "parameter_count": training.get("parameter_count"),
                "train_mean_length": float(np.mean(lengths["train"])) if lengths else np.nan,
                "test_mean_length": float(np.mean(lengths["test"])) if lengths else np.nan,
            })
    frame = pd.DataFrame(rows)
    order = [entry["config"] for entry in EXPECTED_MATRIX]
    frame["config"] = pd.Categorical(frame["config"], categories=order, ordered=True)
    return frame.sort_values(["config", "fold"]).reset_index(drop=True)


def summary_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across the three held-out signers.

    The SD is across LOSO folds with one fixed seed, so it measures held-out
    SIGNER variation, not initialization variance.
    """

    metrics = ["test_accuracy", "test_macro_f1", "validation_macro_f1",
               "domain_gap_macro_f1", "best_epoch", "wall_seconds"]
    grouped = folds.groupby("config", observed=True)[metrics]
    summary = grouped.agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()


def per_class_table(run_root: str | Path, labels_ar: Mapping[int, str] | None = None,
                    seed: int = SEED) -> pd.DataFrame:
    """Long-form per-class precision/recall/F1/support for every experiment."""

    rows: list[dict[str, Any]] = []
    for entry in EXPECTED_MATRIX:
        for fold in FOLDS:
            per_class = load_result(run_root, entry, fold, seed)["test_metrics"]["per_class"]
            for index in range(NUM_CLASSES):
                rows.append({
                    "config": entry["config"], "fold": fold, "label_index": index,
                    "label_ar": (labels_ar or {}).get(index, ""),
                    "precision": per_class["precision"][index],
                    "recall": per_class["recall"][index],
                    "f1": per_class["f1"][index],
                    "support": per_class["support"][index],
                })
    return pd.DataFrame(rows)


def paired_delta(folds: pd.DataFrame, treatment: str, baseline: str) -> pd.DataFrame:
    """Fold-wise treatment-minus-baseline differences for a one-factor contrast."""

    columns = ["fold", "test_accuracy", "test_macro_f1"]
    left = folds[folds["config"] == treatment][columns].set_index("fold")
    right = folds[folds["config"] == baseline][columns].set_index("fold")
    delta = (left - right).reset_index()
    delta.insert(0, "baseline", baseline)
    delta.insert(0, "treatment", treatment)
    delta["accuracy_delta_pp"] = delta.pop("test_accuracy") * 100.0
    delta["macro_f1_delta_points"] = delta.pop("test_macro_f1") * 100.0
    return delta


def paired_class_delta(per_class: pd.DataFrame, treatment: str, baseline: str) -> pd.DataFrame:
    """Per-class F1 differences, averaged over folds, for a one-factor contrast."""

    pivot = per_class.pivot_table(index=["label_index", "label_ar"], columns="config",
                                  values="f1", aggfunc="mean", observed=True)
    if treatment not in pivot.columns or baseline not in pivot.columns:
        raise SweepAuditError(f"cannot compare {treatment!r} against {baseline!r}")
    out = pd.DataFrame({
        "treatment": treatment, "baseline": baseline,
        "baseline_mean_f1": pivot[baseline], "treatment_mean_f1": pivot[treatment],
        "mean_f1_delta": pivot[treatment] - pivot[baseline],
    }).reset_index()
    return out.sort_values("mean_f1_delta", ascending=False).reset_index(drop=True)


def confusion_pairs(run_root: str | Path, config: str, seed: int = SEED,
                    limit: int = 10, labels_ar: Mapping[int, str] | None = None) -> pd.DataFrame:
    """Top off-diagonal confusion pairs per fold, from the saved matrices."""

    entry = next(e for e in EXPECTED_MATRIX if e["config"] == config)
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        matrix = np.asarray(
            load_result(run_root, entry, fold, seed)["test_metrics"]["confusion_matrix"],
            dtype=np.int64)
        off = matrix.copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(off, axis=None)[::-1][:limit]
        for position in flat:
            true_index, predicted_index = np.unravel_index(int(position), matrix.shape)
            count = int(off[true_index, predicted_index])
            if count <= 0:
                break
            rows.append({
                "config": config, "fold": fold,
                "true_label_index": int(true_index),
                "true_label_ar": (labels_ar or {}).get(int(true_index), ""),
                "predicted_label_index": int(predicted_index),
                "predicted_label_ar": (labels_ar or {}).get(int(predicted_index), ""),
                "count": count,
                "share_of_true_class": count / max(1, int(matrix[true_index].sum())),
            })
    return pd.DataFrame(rows)


def confusion_universality(pairs: pd.DataFrame) -> dict[str, Any]:
    """How many top confusion pairs recur across held-out signers?

    A pair appearing in one fold only is signer-specific; one appearing in all
    three is a candidate for a genuinely confusable letter pair.
    """

    if pairs.empty:
        return {"pairs": 0, "folds_per_pair": {}, "shared_by_all_folds": []}
    counts = (pairs.groupby(["true_label_index", "predicted_label_index"], observed=True)["fold"]
              .nunique().reset_index(name="folds"))
    distribution = counts["folds"].value_counts().sort_index()
    shared = counts[counts["folds"] == len(FOLDS)]
    shared_pairs = pairs.merge(shared[["true_label_index", "predicted_label_index"]],
                               on=["true_label_index", "predicted_label_index"])
    return {
        "distinct_pairs": int(len(counts)),
        "folds_per_pair": {int(k): int(v) for k, v in distribution.items()},
        "signer_specific_fraction": float((counts["folds"] == 1).mean()),
        "shared_by_all_folds": sorted({
            f"{row.true_label_ar}->{row.predicted_label_ar}"
            for row in shared_pairs.itertuples()
        }),
    }


def history_table(run_root: str | Path, seed: int = SEED) -> pd.DataFrame:
    """Every recorded epoch of every experiment, long form."""

    rows: list[dict[str, Any]] = []
    for entry in EXPECTED_MATRIX:
        for fold in FOLDS:
            path = experiment_dir(run_root, entry, fold, seed) / "history.json"
            for record in json.loads(path.read_text(encoding="utf-8")):
                rows.append({"config": entry["config"], "fold": fold, **record})
    return pd.DataFrame(rows)


__all__ = [
    "FOLDS", "SEED", "EXPECTED_MATRIX", "PRIMARY_CONFIG", "SweepAuditError",
    "experiment_dir", "audit_run_root", "load_result", "fold_table", "summary_table",
    "per_class_table", "paired_delta", "paired_class_delta", "confusion_pairs",
    "confusion_universality", "history_table",
]
