"""Tests for the TASK-009C deployment artifact verification.

Each check is exercised on a conforming synthetic run root and on a deliberately
corrupted one. Two tests read the real persisted deployment run and skip when it
is absent.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recognition.data.contract import CONTRACT_VERSION, NUM_CLASSES
from recognition.deployment import (
    DEPLOYMENT_CHECKPOINT,
    RESUME_CHECKPOINT,
    deployment_metadata,
    sha256_file,
    verify_checkpoint_metadata,
    verify_files,
    verify_history,
    verify_plan,
    verify_status,
)
from recognition.deployment.plan import (
    DEPLOYMENT_TRAINING_ROLE,
    EPOCH_POLICY,
    EXPECTED_MODEL_CONFIG,
    EXPECTED_SAMPLES,
    EXPECTED_TRAINING_CONFIG,
    PLAN_SCHEMA_VERSION,
    PRIMARY_QUATERNION_POLICY,
)
from recognition.deployment.train import DEPLOYMENT_METRIC_NAME

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009c-core28-deployment")
EPOCHS = 27


def _plan(epochs: int = EPOCHS) -> dict:
    return {
        "schema_version": PLAN_SCHEMA_VERSION, "contract_version": CONTRACT_VERSION,
        "training_scope": "all_core28_sequences",
        "configuration": {**EXPECTED_MODEL_CONFIG,
                          "quaternion_policy": PRIMARY_QUATERNION_POLICY,
                          "model_input_dimension": 92},
        "training_config": {**EXPECTED_TRAINING_CONFIG, "epochs": epochs, "seed": 1337},
        "epoch_budget": {"policy": EPOCH_POLICY, "deployment_epochs": epochs,
                         "source_best_epochs": {"01": 28, "02": 20, "03": 27}},
        "dataset_audit": {"indexed_samples": EXPECTED_SAMPLES, "rejected_samples": 0,
                          "signers": {"01": 1402, "02": 1401, "03": 1419},
                          "classes": NUM_CLASSES},
        "loso_reference": {"mean_test_accuracy": 0.676298, "mean_test_macro_f1": 0.660679},
        "scientific_status": "POST-EVALUATION DEPLOYMENT TRAINING",
    }


def _history(epochs: int = EPOCHS) -> list[dict]:
    return [{"epoch": e, "train_loss": 3.0 / e, "train_accuracy": min(0.99, 0.05 * e),
             "epoch_seconds": 7.0} for e in range(1, epochs + 1)]


def _checkpoint_payload(epochs: int = EPOCHS, **overrides) -> dict:
    payload = {
        "schema_version": "task009b_checkpoint_v1",
        "contract_version": CONTRACT_VERSION,
        "experiment": {"feature_set": "full", "quaternion_policy": "absolute",
                       "pooling": "masked_mean", "fold": "all", "seed": 1337,
                       "input_policy": "values_and_feature_valid",
                       "contract_version": CONTRACT_VERSION},
        "input_dim": 92, "num_classes": NUM_CLASSES, "epoch": epochs,
        "best_metric_name": DEPLOYMENT_METRIC_NAME,
        "extra": deployment_metadata(_plan(epochs), epochs_completed=epochs),
    }
    for dotted, value in overrides.items():
        block, _, key = dotted.partition(".")
        if key:
            payload[block] = {**payload[block], key: value}
        else:
            payload[block] = value
    return payload


def _run_root(directory: Path, *, epochs: int = EPOCHS, status_overrides: dict | None = None,
              history: list[dict] | None = None, plan: dict | None = None) -> Path:
    (directory / "deployment_plan.json").write_text(
        json.dumps(plan or _plan(epochs)), encoding="utf-8")
    status = {"status": "COMPLETE", "training_role": DEPLOYMENT_TRAINING_ROLE,
              "epochs_planned": epochs, "epochs_completed": epochs,
              "wall_seconds": 188.5, "smoke_mode": False,
              "deployment_checkpoint": str(directory / DEPLOYMENT_CHECKPOINT)}
    status.update(status_overrides or {})
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "history.json").write_text(
        json.dumps(history if history is not None else _history(epochs)), encoding="utf-8")
    (directory / "training_summary.json").write_text(json.dumps({"epochs_completed": epochs}),
                                                     encoding="utf-8")
    for name in (DEPLOYMENT_CHECKPOINT, RESUME_CHECKPOINT):
        (directory / name).write_bytes(b"synthetic checkpoint bytes")
    return directory


class TestFileVerification(unittest.TestCase):
    def test_a_complete_run_root_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_files(_run_root(Path(tmp)))
        self.assertTrue(result["passed"])
        self.assertEqual(result["problems"], [])
        for name in ("deployment.pt", "last.pt", "history.json", "status.json",
                     "training_summary.json", "deployment_plan.json"):
            self.assertTrue(result["files"][name]["present"], name)

    def test_a_missing_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp))
            (root / DEPLOYMENT_CHECKPOINT).unlink()
            result = verify_files(root)
        self.assertFalse(result["passed"])
        self.assertTrue(any("deployment.pt is missing" in p for p in result["problems"]))

    def test_an_empty_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp))
            (root / "history.json").write_text("", encoding="utf-8")
            result = verify_files(root)
        self.assertFalse(result["passed"])


class TestStatusVerification(unittest.TestCase):
    def test_a_complete_status_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_status(_run_root(Path(tmp)), EPOCHS)
        self.assertTrue(result["passed"])
        self.assertEqual(result["epochs_completed"], EPOCHS)

    def test_an_incomplete_status_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp), status_overrides={"status": "IN_PROGRESS"})
            result = verify_status(root, EPOCHS)
        self.assertFalse(result["passed"])

    def test_a_short_run_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp), status_overrides={"epochs_completed": 26})
            result = verify_status(root, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("epochs_completed 26" in p for p in result["problems"]))

    def test_a_smoke_run_is_never_accepted_as_the_deployment_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp), status_overrides={"smoke_mode": True})
            result = verify_status(root, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("smoke_mode" in p for p in result["problems"]))


class TestHistoryVerification(unittest.TestCase):
    def test_a_clean_history_passes_and_reports_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_history(_run_root(Path(tmp)), EPOCHS)
        self.assertTrue(result["passed"])
        self.assertEqual(result["entries"], EPOCHS)
        self.assertEqual(result["initial_train_loss"], 3.0)
        self.assertAlmostEqual(result["final_train_loss"], 3.0 / EPOCHS)
        self.assertIn("selection_note", result)

    def test_a_wrong_epoch_count_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp), history=_history(26))
            result = verify_history(root, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("26 history entries" in p for p in result["problems"]))

    def test_a_duplicated_epoch_from_a_bad_resume_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = _history(EPOCHS)
            history[5] = dict(history[4])  # epoch 5 recorded twice
            root = _run_root(Path(tmp), history=history)
            result = verify_history(root, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("duplicate epoch" in p for p in result["problems"]))

    def test_a_missing_epoch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = _history(EPOCHS)
            history.pop(10)
            history.append({"epoch": 99, "train_loss": 0.1, "train_accuracy": 0.9,
                            "epoch_seconds": 7.0})
            root = _run_root(Path(tmp), history=history)
            result = verify_history(root, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("contiguous" in p for p in result["problems"]))

    def test_nan_and_inf_are_reported(self) -> None:
        for bad in ("NaN", "Infinity"):
            with tempfile.TemporaryDirectory() as tmp:
                root = _run_root(Path(tmp))
                raw = json.dumps(_history(EPOCHS))
                raw = raw.replace('"train_loss": 3.0,', f'"train_loss": {bad},', 1)
                (root / "history.json").write_text(raw, encoding="utf-8")
                result = verify_history(root, EPOCHS)
            self.assertFalse(result["passed"], bad)
            self.assertTrue(any("not finite" in p for p in result["problems"]), bad)

    def test_an_out_of_range_accuracy_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = _history(EPOCHS)
            history[3]["train_accuracy"] = 1.4
            root = _run_root(Path(tmp), history=history)
            result = verify_history(root, EPOCHS)
        self.assertFalse(result["passed"])

    def test_a_malformed_record_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = _history(EPOCHS)
            history[2].pop("train_accuracy")
            root = _run_root(Path(tmp), history=history)
            result = verify_history(root, EPOCHS)
        self.assertFalse(result["passed"])


class TestPlanVerification(unittest.TestCase):
    def test_a_frozen_plan_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_plan(_run_root(Path(tmp)))
        self.assertTrue(result["passed"])
        self.assertEqual(result["deployment_epochs"], EPOCHS)
        self.assertEqual(result["source_best_epochs"], {"01": 28, "02": 20, "03": 27})
        self.assertEqual(result["training_samples"], EXPECTED_SAMPLES)

    def test_a_changed_configuration_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan()
            plan["configuration"]["pooling"] = "final_hidden"
            result = verify_plan(_run_root(Path(tmp), plan=plan))
        self.assertFalse(result["passed"])
        self.assertTrue(any("pooling" in p for p in result["problems"]))

    def test_a_changed_hyperparameter_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan()
            plan["training_config"]["learning_rate"] = 3e-4
            result = verify_plan(_run_root(Path(tmp), plan=plan))
        self.assertFalse(result["passed"])
        self.assertTrue(any("learning_rate" in p for p in result["problems"]))

    def test_a_reduced_training_scope_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan()
            plan["dataset_audit"]["indexed_samples"] = 3000
            result = verify_plan(_run_root(Path(tmp), plan=plan))
        self.assertFalse(result["passed"])
        self.assertTrue(any("training samples 3000" in p for p in result["problems"]))

    def test_a_drift_from_the_committed_plan_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(Path(tmp))
            committed = Path(tmp) / "committed.json"
            other = _plan()
            other["epoch_budget"]["deployment_epochs"] = 20
            committed.write_text(json.dumps(other), encoding="utf-8")
            result = verify_plan(root, committed)
        self.assertFalse(result["passed"])
        self.assertFalse(result["matches_committed_plan"])
        self.assertTrue(any("differs from the committed" in p for p in result["problems"]))


class TestCheckpointMetadataVerification(unittest.TestCase):
    def test_an_honest_deployment_checkpoint_passes(self) -> None:
        result = verify_checkpoint_metadata(_checkpoint_payload(), EPOCHS)
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["experiment"]["fold"], "all")
        self.assertEqual(result["best_metric_name"], DEPLOYMENT_METRIC_NAME)
        self.assertEqual(result["input_dim"], 92)
        self.assertEqual(result["num_classes"], NUM_CLASSES)

    def test_a_checkpoint_claiming_a_validation_metric_is_rejected(self) -> None:
        payload = _checkpoint_payload()
        payload["best_metric_name"] = "validation_macro_f1"
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("falsely claims a validation-selected metric" in p
                            for p in result["problems"]))

    def test_a_checkpoint_labelled_as_a_loso_fold_is_rejected(self) -> None:
        payload = _checkpoint_payload(**{"experiment.fold": "01"})
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("falsely labelled as a LOSO fold" in p
                            for p in result["problems"]))

    def test_a_wrong_epoch_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["epoch"] = 20
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])

    def test_a_wrong_input_dimension_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["input_dim"] = 46
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])

    def test_missing_deployment_provenance_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["extra"] = {k: v for k, v in payload["extra"].items()
                            if k != "source_primary_best_epochs"}
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("source_primary_best_epochs" in p for p in result["problems"]))

    def test_a_missing_loso_reference_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["extra"] = {k: v for k, v in payload["extra"].items() if k != "loso_reference"}
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("loso_reference" in p for p in result["problems"]))

    def test_a_missing_scientific_status_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["extra"] = {**payload["extra"], "scientific_status": "a great model"}
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])
        self.assertTrue(any("deployment disclaimer" in p for p in result["problems"]))

    def test_a_foreign_contract_version_is_reported(self) -> None:
        payload = _checkpoint_payload()
        payload["contract_version"] = "task009a_sequence_input_v0"
        result = verify_checkpoint_metadata(payload, EPOCHS)
        self.assertFalse(result["passed"])


class TestHashing(unittest.TestCase):
    def test_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            payload = b"deployment checkpoint bytes" * 1000
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_hash_changes_when_one_byte_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.bin"
            path.write_bytes(b"aaaa")
            first = sha256_file(path)
            path.write_bytes(b"aaab")
            self.assertNotEqual(first, sha256_file(path))


class TestPersistedDeployment(unittest.TestCase):
    """Guards on the real deployment run; skipped when it is not present."""

    def setUp(self) -> None:
        if not REAL_RUN_ROOT.is_dir():
            self.skipTest(f"{REAL_RUN_ROOT} is not present")

    def test_the_real_run_passes_every_check(self) -> None:
        from recognition.training import load_checkpoint

        self.assertTrue(verify_files(REAL_RUN_ROOT)["passed"])
        plan = verify_plan(REAL_RUN_ROOT,
                           ROOT / "reports/recognition/TASK-009C-deployment-plan.json")
        self.assertTrue(plan["passed"], plan["problems"])
        self.assertTrue(plan["matches_committed_plan"])
        self.assertEqual(plan["deployment_epochs"], EPOCHS)
        self.assertTrue(verify_status(REAL_RUN_ROOT, EPOCHS)["passed"])
        history = verify_history(REAL_RUN_ROOT, EPOCHS)
        self.assertTrue(history["passed"], history["problems"])
        self.assertEqual(history["entries"], EPOCHS)
        metadata = verify_checkpoint_metadata(
            load_checkpoint(REAL_RUN_ROOT / DEPLOYMENT_CHECKPOINT), EPOCHS)
        self.assertTrue(metadata["passed"], metadata["problems"])

    def test_committed_manifest_matches_the_persisted_checkpoint(self) -> None:
        path = ROOT / "reports/recognition/TASK-009C-deployment-artifact.json"
        if not path.is_file():
            self.skipTest("artifact manifest has not been generated yet")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = REAL_RUN_ROOT / DEPLOYMENT_CHECKPOINT
        self.assertEqual(manifest["checkpoint_sha256"], sha256_file(checkpoint))
        self.assertEqual(manifest["checkpoint_size_bytes"], checkpoint.stat().st_size)
        self.assertEqual(manifest["epoch"], EPOCHS)
        self.assertEqual(manifest["samples"], EXPECTED_SAMPLES)
        self.assertEqual(manifest["signers"], ["01", "02", "03"])
        self.assertEqual(manifest["classes"], NUM_CLASSES)
        self.assertEqual(manifest["contract_version"], CONTRACT_VERSION)
        self.assertTrue(manifest["verification"]["all_passed"])
        # The manifest must carry the LOSO evidence and must NOT present the
        # in-sample training accuracy as a performance estimate.
        self.assertIn("mean_test_accuracy", manifest["scientific_loso_reference"])
        self.assertIn("NOT a performance estimate",
                      manifest["in_sample_training"]["warning"])
        self.assertNotIn("deployment_accuracy", manifest)


if __name__ == "__main__":
    unittest.main()
