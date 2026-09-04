"""Tests for the TASK-009C all-signers deployment training.

Plan derivation, dataset scope and checkpoint semantics are tested on synthetic
fixtures in temporary directories. A few tests read the real persisted TASK-009B
sweep and the frozen production tree, and skip cleanly when those are absent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recognition.data.contract import CONTRACT_VERSION, NUM_CLASSES
from recognition.deployment import (
    DEPLOYMENT_CHECKPOINT,
    DEPLOYMENT_FOLD_TAG,
    DEPLOYMENT_TRAINING_ROLE,
    EPOCH_POLICY,
    EXPECTED_SAMPLES,
    PLAN_SCHEMA_VERSION,
    PRIMARY_FEATURE_SET,
    PRIMARY_INPUT_POLICY,
    PRIMARY_POOLING,
    PRIMARY_QUATERNION_POLICY,
    DeploymentPlanError,
    build_deployment_plan,
    derive_epoch_budget,
    deployment_metadata,
    deployment_spec,
    load_plan,
    read_primary_loso_evidence,
    train_deployment_model,
    write_plan,
)
from recognition.deployment.plan import (
    EXPECTED_MODEL_CONFIG,
    EXPECTED_TRAINING_CONFIG,
    primary_result_path,
)
from recognition.deployment.train import DEPLOYMENT_METRIC_NAME
from recognition.models import LSTMBaseline, LSTMBaselineConfig, SequenceRecognizer
from recognition.training import ExperimentSpec, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
LOSO_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")
DATA_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
INDEX = ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv"
LABELS = ROOT / "datasets/manifests/karsl_core28_labels.csv"
FINGERS, CHAIN, SPREAD, HANDS = 5, 3, 4, 2


def _write_primary_result(root: Path, fold: str, best_epoch: int, *, seed: int = 1337,
                          overrides: dict | None = None) -> Path:
    path = primary_result_path(root, fold, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "experiment": {
            "feature_set": PRIMARY_FEATURE_SET,
            "quaternion_policy": PRIMARY_QUATERNION_POLICY,
            "pooling": PRIMARY_POOLING, "fold": fold, "seed": seed,
            "input_policy": PRIMARY_INPUT_POLICY, "contract_version": CONTRACT_VERSION,
        },
        "model_config": dict(EXPECTED_MODEL_CONFIG),
        "best_epoch": best_epoch,
        "best_validation_metric": 0.98,
        "training": {"epochs_completed": best_epoch + 12, "stopped_early": True,
                     "training_config": {**EXPECTED_TRAINING_CONFIG, "epochs": 60,
                                         "early_stopping_patience": 12,
                                         "selection_metric": "validation_macro_f1",
                                         "num_workers": 0}},
        "test_metrics": {"accuracy": 0.68, "macro_f1": 0.66},
        "split_sizes": {"train": 2372, "validation": 448, "test": 1402},
    }
    for dotted, value in (overrides or {}).items():
        block, key = dotted.split(".", 1)
        payload[block][key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_loso_root(root: Path, epochs=(28, 20, 27)) -> Path:
    for fold, best in zip(("01", "02", "03"), epochs):
        _write_primary_result(root, fold, best)
    return root


def _arrays(frames: int):
    present = np.ones((frames, HANDS), dtype=bool)
    bend_valid = np.repeat(present[:, :, None], FINGERS * CHAIN, axis=2).reshape(
        frames, HANDS, FINGERS, CHAIN)
    spread_valid = np.repeat(present[:, :, None], SPREAD, axis=2)
    quaternion = np.zeros((frames, HANDS, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    return {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": (np.arange(frames) / 30.0).astype(np.float64),
        "bend_normalized": np.where(bend_valid, 0.25, np.nan).astype(np.float32),
        "bend_valid": bend_valid,
        "spread_normalized": np.where(spread_valid, 0.5, np.nan).astype(np.float32),
        "spread_valid": spread_valid,
        "imu_quaternion_wxyz": quaternion, "palm_imu_valid": present.copy(),
        "tracking_state_code": np.ones((frames, HANDS), dtype=np.int32),
    }


def _minimal_plan(epochs: int = 27, samples: int = EXPECTED_SAMPLES) -> dict:
    return {
        "schema_version": PLAN_SCHEMA_VERSION, "contract_version": CONTRACT_VERSION,
        "training_scope": "all_core28_sequences",
        "configuration": {**EXPECTED_MODEL_CONFIG,
                          "quaternion_policy": PRIMARY_QUATERNION_POLICY,
                          "model_input_dimension": 92},
        "training_config": {**EXPECTED_TRAINING_CONFIG, "epochs": epochs, "seed": 1337},
        "epoch_budget": {"policy": EPOCH_POLICY, "deployment_epochs": epochs,
                         "source_best_epochs": {"01": 28, "02": 20, "03": 27}},
        "dataset_audit": {"indexed_samples": samples, "signers": {"01": 1, "02": 1, "03": 1},
                          "classes": NUM_CLASSES},
        "loso_reference": {"mean_test_accuracy": 0.676298, "mean_test_macro_f1": 0.660679},
        "scientific_status": "POST-EVALUATION DEPLOYMENT TRAINING",
        "source_task009b_analysis_commit": "0ddf0e6",
    }


class TestEpochDerivation(unittest.TestCase):
    def test_median_of_the_three_primary_best_epochs(self) -> None:
        budget = derive_epoch_budget([28, 20, 27])
        self.assertEqual(budget["deployment_epochs"], 27)
        self.assertEqual(budget["raw_median"], 27.0)
        self.assertEqual(budget["sorted_best_epochs"], [20, 27, 28])
        self.assertEqual(budget["policy"], EPOCH_POLICY)

    def test_median_is_order_independent(self) -> None:
        for order in ([28, 20, 27], [20, 27, 28], [27, 28, 20]):
            self.assertEqual(derive_epoch_budget(order)["deployment_epochs"], 27)

    def test_even_count_uses_documented_half_up_rounding(self) -> None:
        # middles 27 and 28 -> mean 27.5 -> floor(27.5 + 0.5) = 28
        self.assertEqual(derive_epoch_budget([20, 27, 28, 29])["deployment_epochs"], 28)
        # middles 20 and 27 -> mean 23.5 -> 24
        self.assertEqual(derive_epoch_budget([19, 20, 27, 28])["deployment_epochs"], 24)

    def test_the_median_is_not_the_mean(self) -> None:
        # mean of (28, 20, 27) is 25 -- the rule must not silently average.
        self.assertNotEqual(derive_epoch_budget([28, 20, 27])["deployment_epochs"], 25)

    def test_empty_or_invalid_epochs_are_rejected(self) -> None:
        with self.assertRaises(DeploymentPlanError):
            derive_epoch_budget([])
        with self.assertRaises(DeploymentPlanError):
            derive_epoch_budget([28, 0, 27])


class TestPrimaryEvidence(unittest.TestCase):
    def test_three_primary_folds_are_found_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folds = read_primary_loso_evidence(_synthetic_loso_root(Path(tmp)))
        self.assertEqual(sorted(folds), ["01", "02", "03"])
        self.assertEqual([folds[f]["best_epoch"] for f in ("01", "02", "03")], [28, 20, 27])

    def test_a_missing_fold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_primary_result(root, "01", 28)
            _write_primary_result(root, "02", 20)
            with self.assertRaises(DeploymentPlanError) as caught:
                read_primary_loso_evidence(root)
            self.assertIn("missing TASK-009B primary result", str(caught.exception))

    def test_a_fold_from_another_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _synthetic_loso_root(Path(tmp))
            _write_primary_result(root, "02", 20,
                                  overrides={"experiment.pooling": "final_hidden"})
            with self.assertRaises(DeploymentPlanError) as caught:
                read_primary_loso_evidence(root)
            self.assertIn("not the frozen primary configuration", str(caught.exception))

    def test_a_foreign_contract_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _synthetic_loso_root(Path(tmp))
            _write_primary_result(root, "03", 27,
                                  overrides={"experiment.contract_version": "v0"})
            with self.assertRaises(DeploymentPlanError):
                read_primary_loso_evidence(root)

    def test_a_missing_best_epoch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _synthetic_loso_root(Path(tmp))
            path = primary_result_path(root, "01")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["best_epoch"] = None
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DeploymentPlanError):
                read_primary_loso_evidence(root)


class TestPlanImmutability(unittest.TestCase):
    def test_an_existing_plan_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deployment_plan.json"
            write_plan(path, _minimal_plan())
            with self.assertRaises(DeploymentPlanError) as caught:
                write_plan(path, _minimal_plan(epochs=99))
            self.assertIn("immutable", str(caught.exception))
            self.assertEqual(load_plan(path)["epoch_budget"]["deployment_epochs"], 27)

    def test_force_overwrites_deliberately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deployment_plan.json"
            write_plan(path, _minimal_plan())
            write_plan(path, _minimal_plan(epochs=99), force=True)
            self.assertEqual(load_plan(path)["epoch_budget"]["deployment_epochs"], 99)

    def test_a_missing_plan_names_the_prepare_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DeploymentPlanError) as caught:
                load_plan(Path(tmp) / "absent.json")
            self.assertIn("--prepare", str(caught.exception))

    def test_a_malformed_or_foreign_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(DeploymentPlanError):
                load_plan(broken)
            foreign = Path(tmp) / "foreign.json"
            plan = _minimal_plan()
            plan["schema_version"] = "task009c_deployment_plan_v0"
            foreign.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(DeploymentPlanError):
                load_plan(foreign)

    def test_a_plan_without_a_positive_epoch_budget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan = _minimal_plan()
            plan["epoch_budget"]["deployment_epochs"] = 0
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(DeploymentPlanError):
                load_plan(path)


class TestDeploymentCheckpoint(unittest.TestCase):
    def _model(self) -> LSTMBaseline:
        return LSTMBaseline(LSTMBaselineConfig(
            feature_set=PRIMARY_FEATURE_SET, input_policy=PRIMARY_INPUT_POLICY,
            pooling=PRIMARY_POOLING))

    def test_spec_marks_the_run_as_all_signers_not_a_fold(self) -> None:
        spec = deployment_spec(_minimal_plan())
        self.assertEqual(spec.fold, DEPLOYMENT_FOLD_TAG)
        self.assertNotIn(spec.fold, ("01", "02", "03"))
        self.assertEqual(spec.feature_set, PRIMARY_FEATURE_SET)
        self.assertEqual(spec.pooling, PRIMARY_POOLING)
        self.assertEqual(spec.input_policy, PRIMARY_INPUT_POLICY)
        self.assertEqual(spec.contract_version, CONTRACT_VERSION)

    def test_metadata_records_the_deployment_provenance(self) -> None:
        metadata = deployment_metadata(_minimal_plan(), epochs_completed=27)
        self.assertEqual(metadata["training_role"], DEPLOYMENT_TRAINING_ROLE)
        self.assertEqual(metadata["training_scope"], "all_core28_sequences")
        self.assertEqual(metadata["training_samples"], EXPECTED_SAMPLES)
        self.assertEqual(metadata["signers"], ["01", "02", "03"])
        self.assertEqual(metadata["epoch_policy"], EPOCH_POLICY)
        self.assertEqual(metadata["deployment_epochs"], 27)
        self.assertEqual(metadata["source_primary_best_epochs"],
                         {"01": 28, "02": 20, "03": 27})
        self.assertIn("none", metadata["held_out_data"])
        self.assertEqual(metadata["selection_metric"], DEPLOYMENT_METRIC_NAME)

    def test_metadata_carries_the_loso_reference_not_a_deployment_score(self) -> None:
        metadata = deployment_metadata(_minimal_plan(), epochs_completed=27)
        self.assertAlmostEqual(metadata["loso_reference"]["mean_test_accuracy"], 0.676298)
        self.assertNotIn("deployment_accuracy", metadata)
        self.assertNotIn("validation_accuracy", metadata)

    def test_deployment_checkpoint_round_trips_through_the_research_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _minimal_plan()
            spec = deployment_spec(plan)
            model = self._model()
            path = save_checkpoint(
                Path(tmp) / DEPLOYMENT_CHECKPOINT, model=model, spec=spec, epoch=27,
                best_epoch=27, best_metric=0.0, best_metric_name=DEPLOYMENT_METRIC_NAME,
                extra=deployment_metadata(plan, epochs_completed=27))
            payload = load_checkpoint(path, expect=spec)
        self.assertEqual(payload["schema_version"], "task009b_checkpoint_v1")
        self.assertEqual(payload["contract_version"], CONTRACT_VERSION)
        self.assertEqual(payload["input_dim"], 92)
        self.assertEqual(payload["num_classes"], NUM_CLASSES)
        self.assertEqual(payload["best_metric_name"], DEPLOYMENT_METRIC_NAME)
        self.assertEqual(payload["extra"]["training_role"], DEPLOYMENT_TRAINING_ROLE)

    def test_a_deployment_checkpoint_is_not_mistaken_for_another_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _minimal_plan()
            path = save_checkpoint(Path(tmp) / "d.pt", model=self._model(),
                                   spec=deployment_spec(plan))
            wrong = ExperimentSpec("bend_only", "absolute", PRIMARY_POOLING, "all", 1337)
            from recognition.training import CheckpointError
            with self.assertRaises(CheckpointError):
                load_checkpoint(path, expect=wrong)

    def test_a_research_fold_checkpoint_still_loads_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            research = ExperimentSpec(PRIMARY_FEATURE_SET, PRIMARY_QUATERNION_POLICY,
                                      PRIMARY_POOLING, "01", 1337)
            path = save_checkpoint(Path(tmp) / "best.pt", model=self._model(), spec=research,
                                   epoch=40, best_epoch=28, best_metric=0.984)
            payload = load_checkpoint(path, expect=research)
        self.assertEqual(payload["experiment"]["fold"], "01")
        self.assertEqual(payload["best_metric_name"], "validation_macro_f1")
        self.assertEqual(payload["extra"], {})

    def test_both_checkpoint_kinds_serve_inference_through_one_recognizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _minimal_plan()
            deployment = save_checkpoint(Path(tmp) / DEPLOYMENT_CHECKPOINT,
                                         model=self._model(), spec=deployment_spec(plan),
                                         extra=deployment_metadata(plan, epochs_completed=27))
            research = save_checkpoint(
                Path(tmp) / "best.pt", model=self._model(),
                spec=ExperimentSpec(PRIMARY_FEATURE_SET, PRIMARY_QUATERNION_POLICY,
                                    PRIMARY_POOLING, "01", 1337))
            for path in (deployment, research):
                recognizer = SequenceRecognizer.from_checkpoint(path, label_table=LABELS)
                self.assertEqual(recognizer.config.feature_set, PRIMARY_FEATURE_SET)
                self.assertEqual(recognizer.config.quaternion_policy, PRIMARY_QUATERNION_POLICY)
                prediction = recognizer.predict_sequence(_arrays(23))
                self.assertIn(prediction.label_index, range(NUM_CLASSES))
                self.assertEqual(prediction.sequence_length, 23)
                self.assertEqual(len(prediction.logits), NUM_CLASSES)
                self.assertTrue(prediction.label_ar)
                self.assertTrue(prediction.sign_id)

    def test_deployment_prediction_resolves_the_authoritative_arabic_label(self) -> None:
        from recognition.data.labels import load_label_table

        table = load_label_table(LABELS)
        with tempfile.TemporaryDirectory() as tmp:
            plan = _minimal_plan()
            path = save_checkpoint(Path(tmp) / DEPLOYMENT_CHECKPOINT, model=self._model(),
                                   spec=deployment_spec(plan),
                                   extra=deployment_metadata(plan, epochs_completed=27))
            recognizer = SequenceRecognizer.from_checkpoint(path, label_table=LABELS)
            prediction = recognizer.predict_sequence(_arrays(9))
        reference = table[prediction.label_index]
        self.assertEqual(prediction.label_ar, reference.label_ar)
        self.assertEqual(prediction.sign_id, reference.sign_id)


class TestDeploymentTrainingLoop(unittest.TestCase):
    def _loader(self, batches: int = 2, size: int = 4):
        from recognition.data import SequenceInputConfig, build_feature_tensor, collate_sequences

        config = SequenceInputConfig(feature_set=PRIMARY_FEATURE_SET)
        loader = []
        for batch_index in range(batches):
            items = []
            for row in range(size):
                item = build_feature_tensor(_arrays(9 + row), config)
                item.update(sample_id=f"s{batch_index}_{row}", sign_id="0032", label_ar="ا",
                            label_index=(batch_index * size + row) % NUM_CLASSES,
                            signer_id="01", official_partition="train")
                items.append(item)
            loader.append(collate_sequences(items, config))
        return loader

    def test_training_runs_the_frozen_number_of_epochs_and_writes_deployment_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = train_deployment_model(
                model=LSTMBaseline(LSTMBaselineConfig()), plan=_minimal_plan(epochs=3),
                loader=self._loader(), device=torch.device("cpu"), output_dir=Path(tmp),
                log=lambda message: None)
            self.assertEqual(summary["epochs_planned"], 3)
            self.assertEqual(summary["epochs_completed"], 3)
            self.assertEqual(len(summary["history"]), 3)
            self.assertTrue((Path(tmp) / DEPLOYMENT_CHECKPOINT).is_file())
            self.assertTrue((Path(tmp) / "last.pt").is_file())
            self.assertTrue((Path(tmp) / "history.json").is_file())
            self.assertFalse((Path(tmp) / "best.pt").exists())

    def test_summary_reports_in_sample_values_with_an_explicit_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = train_deployment_model(
                model=LSTMBaseline(LSTMBaselineConfig()), plan=_minimal_plan(epochs=2),
                loader=self._loader(), device=torch.device("cpu"), output_dir=Path(tmp),
                log=lambda message: None)
        self.assertIn("IN-SAMPLE", summary["in_sample_warning"])
        self.assertTrue(np.isfinite(summary["final_train_loss"]))
        self.assertNotIn("validation_accuracy", summary)
        self.assertNotIn("test_accuracy", summary)

    def test_resume_continues_a_partial_run_of_the_same_budget(self) -> None:
        plan = _minimal_plan(epochs=4)
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader()
            # Simulate an interruption after epoch 2 of a 4-epoch budget: write the
            # resume checkpoint exactly as the loop would have left it.
            model = LSTMBaseline(LSTMBaselineConfig())
            from recognition.training import build_optimizer
            from recognition.training.trainer import TrainingConfig
            optimizer = build_optimizer(model, TrainingConfig())
            save_checkpoint(
                Path(tmp) / "last.pt", model=model, spec=deployment_spec(plan),
                optimizer_state=optimizer.state_dict(), epoch=2, best_epoch=2,
                best_metric=0.0, best_metric_name=DEPLOYMENT_METRIC_NAME,
                history=[{"epoch": 1, "train_loss": 3.2, "train_accuracy": 0.05,
                          "epoch_seconds": 1.0},
                         {"epoch": 2, "train_loss": 3.0, "train_accuracy": 0.08,
                          "epoch_seconds": 1.0}],
                extra=deployment_metadata(plan, epochs_completed=2))
            summary = train_deployment_model(
                model=LSTMBaseline(LSTMBaselineConfig()), plan=plan, loader=loader,
                device=torch.device("cpu"), output_dir=Path(tmp), resume=True,
                log=lambda message: None)
        self.assertEqual(summary["epochs_completed"], 4)
        # Prior history is carried forward and only epochs 3-4 were newly run.
        self.assertEqual([entry["epoch"] for entry in summary["history"]], [1, 2, 3, 4])
        self.assertAlmostEqual(summary["history"][0]["train_loss"], 3.2)

    def test_resume_refuses_a_changed_epoch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader()
            train_deployment_model(
                model=LSTMBaseline(LSTMBaselineConfig()), plan=_minimal_plan(epochs=4),
                loader=loader, device=torch.device("cpu"), output_dir=Path(tmp),
                log=lambda message: None)
            # The stored run targeted 4 epochs; a plan claiming 9 must not resume,
            # because the frozen budget is the whole basis of the stopping point.
            with self.assertRaises(ValueError) as caught:
                train_deployment_model(
                    model=LSTMBaseline(LSTMBaselineConfig()), plan=_minimal_plan(epochs=9),
                    loader=loader, device=torch.device("cpu"), output_dir=Path(tmp),
                    resume=True, log=lambda message: None)
            self.assertIn("frozen plan must not change", str(caught.exception))


class TestPersistedEvidence(unittest.TestCase):
    """Guards on the real TASK-009B sweep and production data; skipped if absent."""

    def setUp(self) -> None:
        if not LOSO_RUN_ROOT.is_dir():
            self.skipTest(f"{LOSO_RUN_ROOT} is not present")

    def test_real_primary_best_epochs_are_28_20_27_with_median_27(self) -> None:
        folds = read_primary_loso_evidence(LOSO_RUN_ROOT)
        best = [folds[fold]["best_epoch"] for fold in ("01", "02", "03")]
        self.assertEqual(best, [28, 20, 27])
        self.assertEqual(derive_epoch_budget(best)["deployment_epochs"], 27)

    def test_committed_plan_matches_the_persisted_evidence(self) -> None:
        path = ROOT / "reports/recognition/TASK-009C-deployment-plan.json"
        if not path.is_file():
            self.skipTest("committed plan copy is not present")
        plan = load_plan(path)
        folds = read_primary_loso_evidence(LOSO_RUN_ROOT)
        self.assertEqual(
            plan["epoch_budget"]["source_best_epochs"],
            {fold: folds[fold]["best_epoch"] for fold in ("01", "02", "03")})
        self.assertEqual(plan["epoch_budget"]["deployment_epochs"], 27)
        self.assertEqual(plan["dataset_audit"]["indexed_samples"], EXPECTED_SAMPLES)
        self.assertEqual(plan["dataset_audit"]["rejected_samples"], 0)
        self.assertEqual(plan["configuration"]["feature_set"], PRIMARY_FEATURE_SET)
        self.assertEqual(plan["configuration"]["pooling"], PRIMARY_POOLING)
        self.assertEqual(plan["configuration"]["model_input_dimension"], 92)

    def test_deployment_scope_is_every_signer_and_class(self) -> None:
        if not DATA_RUN_ROOT.is_dir():
            self.skipTest(f"{DATA_RUN_ROOT} is not present")
        from recognition.deployment import audit_deployment_dataset

        audit = audit_deployment_dataset(INDEX, DATA_RUN_ROOT, load_every_sequence=False)
        self.assertTrue(audit["passed"], audit["problems"])
        self.assertEqual(audit["indexed_samples"], EXPECTED_SAMPLES)
        self.assertEqual(sorted(audit["signers"]), ["01", "02", "03"])
        self.assertEqual(audit["classes"], NUM_CLASSES)
        self.assertEqual(audit["duplicate_sample_ids"], [])
        self.assertEqual(audit["model_input_dimension"], 92)
        # No LOSO filtering: every signer's full count is present.
        self.assertEqual(sum(audit["signers"].values()), EXPECTED_SAMPLES)

    def test_a_real_research_checkpoint_still_loads(self) -> None:
        path = (LOSO_RUN_ROOT / PRIMARY_FEATURE_SET / f"q-{PRIMARY_QUATERNION_POLICY}" /
                PRIMARY_POOLING / "fold01/seed1337/best.pt")
        if not path.is_file():
            self.skipTest("research checkpoint is not present")
        payload = load_checkpoint(path)
        self.assertEqual(payload["contract_version"], CONTRACT_VERSION)
        self.assertEqual(payload["experiment"]["fold"], "01")
        recognizer = SequenceRecognizer.from_checkpoint(path, label_table=LABELS)
        self.assertEqual(recognizer.config.feature_set, PRIMARY_FEATURE_SET)


if __name__ == "__main__":
    unittest.main()
