"""Tests for the TASK-009B sweep analysis.

Aggregation logic is tested on a synthetic run root built in a temporary
directory, so the tests do not depend on the real 186 MB training tree. Two
tests do read the persisted sweep, and skip cleanly when it is absent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recognition.analysis import (
    EXPECTED_MATRIX,
    FOLDS,
    PRIMARY_CONFIG,
    SEED,
    SweepAuditError,
    audit_run_root,
    confusion_pairs,
    confusion_universality,
    experiment_dir,
    fold_table,
    history_table,
    paired_class_delta,
    paired_delta,
    per_class_table,
    summary_table,
)
from recognition.data.contract import CONTRACT_VERSION, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")


def _confusion(diagonal: int = 5, extra: dict[tuple[int, int], int] | None = None) -> list[list[int]]:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.fill_diagonal(matrix, diagonal)
    for (true_index, predicted_index), count in (extra or {}).items():
        matrix[true_index, predicted_index] = count
    return matrix.tolist()


def _write_experiment(root: Path, entry: dict, fold: str, *, accuracy: float,
                      macro_f1: float, class_f1: float = 0.5,
                      confusion: list[list[int]] | None = None,
                      status: str = "COMPLETE", quaternion_policy: str | None = None,
                      contract_version: str = CONTRACT_VERSION,
                      omit: tuple[str, ...] = ()) -> Path:
    directory = experiment_dir(root, entry, fold)
    directory.mkdir(parents=True, exist_ok=True)
    experiment = {
        "feature_set": entry["feature_set"],
        # A quaternion-free feature set still records the CLI default; the audit
        # must treat that as non-applicable metadata, not a mismatch.
        "quaternion_policy": quaternion_policy or (
            entry["quaternion"] if entry["quaternion"] != "na" else "absolute"),
        "pooling": entry["pooling"], "fold": fold, "seed": SEED,
        "input_policy": "values_and_feature_valid", "contract_version": contract_version,
    }
    per_class = {"precision": [class_f1] * NUM_CLASSES, "recall": [class_f1] * NUM_CLASSES,
                 "f1": [class_f1] * NUM_CLASSES, "support": [50] * NUM_CLASSES}
    result = {
        "schema_version": "task009b_result_v1", "contract_version": contract_version,
        "experiment": experiment,
        "model_config": {"num_classes": NUM_CLASSES, "hidden_size": 192},
        "training": {"epochs_completed": 30, "stopped_early": True, "best_epoch": 18,
                     "wall_seconds": 100.0, "seconds_per_epoch": 3.3, "peak_gpu_mib": 85.0,
                     "parameter_count": 521500},
        "best_epoch": 18, "best_validation_metric": 0.95,
        "validation_metrics": {"accuracy": 0.95, "macro_f1": 0.95, "macro_precision": 0.95,
                               "macro_recall": 0.95, "weighted_f1": 0.95, "samples": 448,
                               "cross_entropy": 0.2, "per_class": per_class,
                               "confusion_matrix": _confusion()},
        "test_metrics": {"accuracy": accuracy, "macro_f1": macro_f1, "macro_precision": macro_f1,
                         "macro_recall": macro_f1, "weighted_f1": macro_f1, "samples": 1402,
                         "cross_entropy": 1.5, "per_class": per_class,
                         "confusion_matrix": confusion or _confusion()},
        "split_sizes": {"train": 2372, "validation": 448, "test": 1402},
        "sequence_lengths": {"train": [20] * 10, "validation": [20] * 5, "test": [15] * 5},
    }
    files = {
        "result.json": result,
        "status.json": {"status": status, "experiment": experiment,
                        "training": {"epochs_completed": 30, "best_epoch": 18,
                                     "best_metric": 0.95}},
        "history.json": [{"epoch": e, "train_loss": 3.0 - e * 0.1, "validation_loss": 2.9 - e * 0.09,
                          "validation_accuracy": 0.1 * e, "validation_macro_f1": 0.09 * e,
                          "best_validation_macro_f1": 0.09 * e, "epoch_seconds": 3.3}
                         for e in range(1, 6)],
        "predictions.json": {"validation": [], "test": []},
    }
    for name, payload in files.items():
        if name in omit:
            continue
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    for name in ("best.pt", "last.pt"):
        if name not in omit:
            (directory / name).write_bytes(b"synthetic checkpoint placeholder")
    return directory


def _complete_run_root(root: Path, accuracy_by_config: dict[str, float] | None = None) -> Path:
    accuracy_by_config = accuracy_by_config or {}
    for entry in EXPECTED_MATRIX:
        for position, fold in enumerate(FOLDS):
            base = accuracy_by_config.get(entry["config"], 0.6)
            _write_experiment(root, entry, fold, accuracy=base + position * 0.05,
                              macro_f1=base + position * 0.04)
    return root


class TestCompletenessAudit(unittest.TestCase):
    def test_a_complete_synthetic_sweep_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_run_root(_complete_run_root(Path(tmp)))
        self.assertTrue(audit["all_complete"])
        self.assertEqual(audit["complete_experiments"], 15)
        self.assertEqual(audit["expected_experiments"], 15)
        self.assertEqual(audit["problem_count"], 0)

    def test_a_missing_experiment_is_reported_not_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _complete_run_root(Path(tmp))
            import shutil
            shutil.rmtree(experiment_dir(root, EXPECTED_MATRIX[2], "02"))
            audit = audit_run_root(root)
        self.assertFalse(audit["all_complete"])
        self.assertEqual(audit["complete_experiments"], 14)
        self.assertTrue(any("directory is missing" in p["problem"] for p in audit["problems"]))

    def test_a_missing_checkpoint_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            (experiment_dir(root, EXPECTED_MATRIX[0], "01") / "best.pt").unlink()
            audit = audit_run_root(root)
        self.assertFalse(audit["all_complete"])
        self.assertTrue(any("best.pt is missing" in p["problem"] for p in audit["problems"]))

    def test_a_non_complete_status_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            _write_experiment(root, EXPECTED_MATRIX[1], "03", accuracy=0.5, macro_f1=0.5,
                              status="RUNNING")
            audit = audit_run_root(root)
        self.assertFalse(audit["all_complete"])
        self.assertTrue(any("status is 'RUNNING'" in p["problem"] for p in audit["problems"]))

    def test_a_foreign_contract_version_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            _write_experiment(root, EXPECTED_MATRIX[0], "01", accuracy=0.5, macro_f1=0.5,
                              contract_version="task009a_sequence_input_v0")
            audit = audit_run_root(root)
        self.assertFalse(audit["all_complete"])
        self.assertTrue(any("contract_version" in p["problem"] for p in audit["problems"]))

    def test_a_wrong_quaternion_policy_is_reported_only_where_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            # bend_only carries the CLI default "absolute" but has no quaternion
            # channels: non-applicable metadata, not a defect.
            bend_only = next(e for e in EXPECTED_MATRIX if e["config"] == "bend_only")
            audit = audit_run_root(root)
            self.assertTrue(audit["all_complete"])
            record = next(r for r in audit["experiments"]
                          if r["config"] == "bend_only" and r["fold"] == "01")
            self.assertFalse(record["quaternion_policy_applicable"])
            self.assertEqual(record["quaternion_policy_recorded"], "absolute")
            # On a feature set that DOES carry quaternions, a mismatch is a defect.
            full_absolute = next(e for e in EXPECTED_MATRIX
                                 if e["config"] == PRIMARY_CONFIG)
            _write_experiment(root, full_absolute, "01", accuracy=0.5, macro_f1=0.5,
                              quaternion_policy="relative_first_valid")
            broken = audit_run_root(root)
        self.assertFalse(broken["all_complete"])
        self.assertTrue(any("quaternion_policy" in p["problem"] for p in broken["problems"]))

    def test_a_malformed_result_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            (experiment_dir(root, EXPECTED_MATRIX[0], "02") / "result.json").write_text(
                "{not json", encoding="utf-8")
            audit = audit_run_root(root)
        self.assertFalse(audit["all_complete"])
        self.assertTrue(any("malformed" in p["problem"] for p in audit["problems"]))


class TestAggregation(unittest.TestCase):
    def test_fold_table_has_one_row_per_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folds = fold_table(_complete_run_root(Path(tmp)))
        self.assertEqual(len(folds), 15)
        self.assertEqual(set(folds.fold), set(FOLDS))
        self.assertEqual(len(set(folds.config)), 5)

    def test_domain_gap_is_validation_minus_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folds = fold_table(_complete_run_root(Path(tmp)))
        expected = folds.best_validation_macro_f1 - folds.test_macro_f1
        np.testing.assert_allclose(folds.domain_gap_macro_f1, expected)

    def test_summary_mean_matches_a_hand_computed_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _complete_run_root(Path(tmp), {"bend_only": 0.5})
            folds = fold_table(root)
            summary = summary_table(folds)
        row = summary[summary.config == "bend_only"].iloc[0]
        # accuracies were 0.50, 0.55, 0.60
        self.assertAlmostEqual(float(row.test_accuracy_mean), 0.55, places=9)
        self.assertAlmostEqual(float(row.test_accuracy_std), float(np.std([0.5, 0.55, 0.6], ddof=1)))

    def test_paired_delta_is_treatment_minus_baseline_in_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _complete_run_root(Path(tmp), {"bend_only": 0.5, "bend_spread": 0.6})
            delta = paired_delta(fold_table(root), "bend_spread", "bend_only")
        self.assertEqual(len(delta), 3)
        np.testing.assert_allclose(delta.accuracy_delta_pp, [10.0, 10.0, 10.0], atol=1e-9)

    def test_per_class_table_covers_every_class_and_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            per_class = per_class_table(_complete_run_root(Path(tmp)), {0: "ا"})
        self.assertEqual(len(per_class), 15 * NUM_CLASSES)
        self.assertEqual(set(per_class.label_index), set(range(NUM_CLASSES)))
        self.assertEqual(per_class[per_class.label_index == 0].label_ar.unique().tolist(), ["ا"])

    def test_paired_class_delta_averages_over_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            for fold in FOLDS:
                _write_experiment(root, EXPECTED_MATRIX[1], fold, accuracy=0.6,
                                  macro_f1=0.6, class_f1=0.7)
            delta = paired_class_delta(per_class_table(root), "bend_spread", "bend_only")
        self.assertEqual(len(delta), NUM_CLASSES)
        np.testing.assert_allclose(delta.mean_f1_delta, 0.2, atol=1e-9)

    def test_paired_class_delta_rejects_an_unknown_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            per_class = per_class_table(_complete_run_root(Path(tmp)))
            with self.assertRaises(SweepAuditError):
                paired_class_delta(per_class, "nonexistent", "bend_only")

    def test_history_table_is_long_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = history_table(_complete_run_root(Path(tmp)))
        self.assertEqual(len(history), 15 * 5)
        self.assertIn("validation_macro_f1", history.columns)


class TestConfusionAnalysis(unittest.TestCase):
    def test_top_pairs_exclude_the_diagonal_and_rank_by_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            entry = next(e for e in EXPECTED_MATRIX if e["config"] == PRIMARY_CONFIG)
            for fold in FOLDS:
                _write_experiment(root, entry, fold, accuracy=0.6, macro_f1=0.6,
                                  confusion=_confusion(50, {(2, 5): 9, (3, 4): 4}))
            pairs = confusion_pairs(root, PRIMARY_CONFIG, limit=5, labels_ar={2: "ت", 5: "ز"})
        first = pairs[pairs.fold == "01"].iloc[0]
        self.assertEqual(int(first.true_label_index), 2)
        self.assertEqual(int(first.predicted_label_index), 5)
        self.assertEqual(int(first["count"]), 9)
        self.assertEqual(first.true_label_ar, "ت")

    def test_universality_detects_a_pair_shared_by_every_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            entry = next(e for e in EXPECTED_MATRIX if e["config"] == PRIMARY_CONFIG)
            for fold in FOLDS:
                _write_experiment(root, entry, fold, accuracy=0.6, macro_f1=0.6,
                                  confusion=_confusion(50, {(2, 5): 9}))
            universality = confusion_universality(
                confusion_pairs(root, PRIMARY_CONFIG, limit=5, labels_ar={2: "ت", 5: "ز"}))
        self.assertEqual(universality["distinct_pairs"], 1)
        self.assertEqual(universality["folds_per_pair"], {3: 1})
        self.assertEqual(universality["signer_specific_fraction"], 0.0)
        self.assertEqual(universality["shared_by_all_folds"], ["ت->ز"])

    def test_universality_detects_signer_specific_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_run_root(root)
            entry = next(e for e in EXPECTED_MATRIX if e["config"] == PRIMARY_CONFIG)
            for position, fold in enumerate(FOLDS):
                _write_experiment(root, entry, fold, accuracy=0.6, macro_f1=0.6,
                                  confusion=_confusion(50, {(position, position + 10): 9}))
            universality = confusion_universality(
                confusion_pairs(root, PRIMARY_CONFIG, limit=5))
        self.assertEqual(universality["distinct_pairs"], 3)
        self.assertEqual(universality["signer_specific_fraction"], 1.0)
        self.assertEqual(universality["shared_by_all_folds"], [])

    def test_empty_pairs_do_not_crash(self) -> None:
        self.assertEqual(confusion_universality(pd.DataFrame())["pairs"], 0)


class TestPersistedSweep(unittest.TestCase):
    """Guards on the real sweep; skipped when the external run root is absent."""

    def setUp(self) -> None:
        if not REAL_RUN_ROOT.is_dir():
            self.skipTest(f"external run root {REAL_RUN_ROOT} is not present")

    def test_persisted_sweep_is_complete_under_the_frozen_contract(self) -> None:
        audit = audit_run_root(REAL_RUN_ROOT)
        self.assertTrue(audit["all_complete"], audit["problems"][:5])
        self.assertEqual(audit["complete_experiments"], 15)
        self.assertEqual(audit["contract_version"], CONTRACT_VERSION)

    def test_committed_summary_matches_the_persisted_artifacts(self) -> None:
        path = ROOT / "reports/recognition/TASK-009B-results-summary.json"
        if not path.is_file():
            self.skipTest("summary has not been generated yet")
        committed = json.loads(path.read_text(encoding="utf-8"))
        folds = fold_table(REAL_RUN_ROOT)
        for config, block in committed["results"].items():
            rows = folds[folds.config == config]
            self.assertAlmostEqual(block["test_accuracy_mean"],
                                   float(rows.test_accuracy.mean()), places=9, msg=config)
            self.assertAlmostEqual(block["test_macro_f1_mean"],
                                   float(rows.test_macro_f1.mean()), places=9, msg=config)


if __name__ == "__main__":
    unittest.main()
