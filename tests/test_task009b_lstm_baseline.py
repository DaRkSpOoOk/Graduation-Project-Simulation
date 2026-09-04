"""Tests for the TASK-009B LSTM baseline, training plumbing and inference API.

Everything is synthetic and CPU-only. No production sequence is read, no real
training is performed, and no test needs a GPU.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recognition.data import (
    SequenceInputConfig,
    build_feature_tensor,
    collate_sequences,
    feature_dimension,
)
from recognition.data.contract import CONTRACT_VERSION, NUM_CLASSES
from recognition.data.labels import load_label_table
from recognition.models import (
    LSTMBaseline,
    LSTMBaselineConfig,
    SequenceRecognizer,
    build_model_input,
    classification_metrics,
    confusion_matrix,
    masked_mean,
    model_input_dimension,
    top_confusions,
)
from recognition.training import (
    CheckpointError,
    ExperimentSpec,
    LengthOnlyClassifier,
    TrainingConfig,
    build_optimizer,
    evaluate,
    load_checkpoint,
    oracle_accuracy_from_length,
    rebuild_model,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
FINGERS, CHAIN, SPREAD, HANDS = 5, 3, 4, 2


def _arrays(frames: int, *, hand_present: np.ndarray | None = None, value: float = 0.25):
    if hand_present is None:
        hand_present = np.ones((frames, HANDS), dtype=bool)
    bend_valid = np.repeat(hand_present[:, :, None], FINGERS * CHAIN, axis=2).reshape(
        frames, HANDS, FINGERS, CHAIN)
    spread_valid = np.repeat(hand_present[:, :, None], SPREAD, axis=2)
    quaternion = np.zeros((frames, HANDS, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    return {
        "frame_index": np.arange(frames, dtype=np.int32),
        "timestamp_seconds": (np.arange(frames) / 30.0).astype(np.float64),
        "bend_normalized": np.where(bend_valid, value, np.nan).astype(np.float32),
        "bend_valid": bend_valid,
        "spread_normalized": np.where(spread_valid, 0.5, np.nan).astype(np.float32),
        "spread_valid": spread_valid,
        "imu_quaternion_wxyz": quaternion,
        "palm_imu_valid": hand_present.copy(),
        "tracking_state_code": np.where(hand_present, 1, 0).astype(np.int32),
    }


def _item(frames: int, config: SequenceInputConfig, *, label: int = 0,
          sample_id: str = "a", hand_present: np.ndarray | None = None,
          value: float = 0.25) -> dict:
    item = build_feature_tensor(_arrays(frames, hand_present=hand_present, value=value), config)
    item.update(sample_id=sample_id, sign_id="0032", label_ar="ا", label_index=label,
                signer_id="01", official_partition="train")
    return item


def _batch(lengths, config: SequenceInputConfig, labels=None):
    labels = labels or [i % NUM_CLASSES for i in range(len(lengths))]
    items = [_item(n, config, label=labels[i], sample_id=f"s{i}") for i, n in enumerate(lengths)]
    return collate_sequences(items, config)


class TestModelInputDimensions(unittest.TestCase):
    def test_declared_dimensions_for_every_ablation(self) -> None:
        expected = {"bend_only": (30, 60), "bend_spread": (38, 76), "full": (46, 92)}
        for feature_set, (values_only, with_masks) in expected.items():
            self.assertEqual(feature_dimension(feature_set), values_only)
            self.assertEqual(model_input_dimension(feature_set, "values_only"), values_only)
            self.assertEqual(
                model_input_dimension(feature_set, "values_and_feature_valid"), with_masks)

    def test_mask_concatenation_puts_values_first_then_validity(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        batch = _batch([5, 9], config)
        joined = build_model_input(batch, "values_and_feature_valid")
        self.assertEqual(tuple(joined.shape), (2, 9, 92))
        self.assertTrue(torch.equal(joined[..., :46], batch["values"]))
        self.assertTrue(torch.equal(joined[..., 46:].bool(), batch["feature_valid"]))

    def test_values_only_policy_leaves_the_dimension_alone(self) -> None:
        config = SequenceInputConfig(feature_set="bend_spread")
        batch = _batch([4], config)
        self.assertEqual(tuple(build_model_input(batch, "values_only").shape), (1, 4, 38))

    def test_unknown_input_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            model_input_dimension("full", "telepathy")
        with self.assertRaises(ValueError):
            LSTMBaselineConfig(input_policy="telepathy")


class TestPooling(unittest.TestCase):
    def test_masked_mean_ignores_padded_steps(self) -> None:
        outputs = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        frame_valid = torch.tensor([[True, True, False]])
        pooled = masked_mean(outputs, frame_valid)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 2.0]])))

    def test_masked_mean_of_a_single_real_step(self) -> None:
        outputs = torch.tensor([[[5.0], [0.0]]])
        pooled = masked_mean(outputs, torch.tensor([[True, False]]))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[5.0]])))

    def test_masked_mean_output_is_invariant_to_batch_padding(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        model = LSTMBaseline(LSTMBaselineConfig(pooling="masked_mean")).eval()
        short = _item(6, config, sample_id="short")
        long = _item(30, config, sample_id="long")
        with torch.no_grad():
            alone = model(collate_sequences([short], config))
            together = model(collate_sequences([short, long], config))
        self.assertTrue(torch.allclose(alone[0], together[0], atol=1e-5))

    def test_final_hidden_pooling_also_ignores_padding(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        model = LSTMBaseline(LSTMBaselineConfig(pooling="final_hidden")).eval()
        short = _item(6, config, sample_id="short")
        long = _item(30, config, sample_id="long")
        with torch.no_grad():
            alone = model(collate_sequences([short], config))
            together = model(collate_sequences([short, long], config))
        self.assertTrue(torch.allclose(alone[0], together[0], atol=1e-5))

    def test_the_two_pooling_policies_are_actually_different(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        batch = _batch([7, 12], config)
        torch.manual_seed(0)
        mean_model = LSTMBaseline(LSTMBaselineConfig(pooling="masked_mean", dropout=0.0)).eval()
        hidden_model = LSTMBaseline(LSTMBaselineConfig(pooling="final_hidden", dropout=0.0)).eval()
        hidden_model.load_state_dict(mean_model.state_dict())
        with torch.no_grad():
            self.assertFalse(torch.allclose(mean_model(batch), hidden_model(batch), atol=1e-4))

    def test_unknown_pooling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LSTMBaselineConfig(pooling="attention")


class TestForwardAndTraining(unittest.TestCase):
    def test_forward_returns_batch_by_28_logits(self) -> None:
        for feature_set in ("bend_only", "bend_spread", "full"):
            config = SequenceInputConfig(feature_set=feature_set)
            model = LSTMBaseline(LSTMBaselineConfig(feature_set=feature_set))
            logits = model(_batch([9, 70, 25], config))
            self.assertEqual(tuple(logits.shape), (3, NUM_CLASSES), feature_set)
            self.assertTrue(torch.isfinite(logits).all())

    def test_extreme_lengths_nine_and_seventy(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        model = LSTMBaseline(LSTMBaselineConfig())
        for length in (9, 70):
            logits = model(_batch([length], config))
            self.assertEqual(tuple(logits.shape), (1, NUM_CLASSES))

    def test_loss_is_finite_and_one_optimizer_step_changes_weights(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        seed_everything(7)
        model = LSTMBaseline(LSTMBaselineConfig())
        batch = _batch([5, 11, 8, 20], config)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, TrainingConfig())
        before = model.classifier.weight.detach().clone()
        loss = criterion(model(batch), batch["labels"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight.detach()))

    def test_train_one_epoch_reports_finite_statistics(self) -> None:
        config = SequenceInputConfig(feature_set="bend_only")
        model = LSTMBaseline(LSTMBaselineConfig(feature_set="bend_only"))
        loader = [_batch([6, 9], config), _batch([12, 15], config)]
        stats = train_one_epoch(model, loader, build_optimizer(model, TrainingConfig()),
                                torch.device("cpu"), TrainingConfig())
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreaterEqual(stats["accuracy"], 0.0)

    def test_batch_with_the_wrong_dimension_is_rejected_not_silently_reshaped(self) -> None:
        model = LSTMBaseline(LSTMBaselineConfig(feature_set="full"))
        wrong = _batch([5], SequenceInputConfig(feature_set="bend_only"))
        with self.assertRaises(ValueError):
            model(wrong)

    def test_evaluation_is_deterministic(self) -> None:
        config = SequenceInputConfig(feature_set="full")
        model = LSTMBaseline(LSTMBaselineConfig())
        loader = [_batch([7, 13], config, labels=[3, 9])]
        first = evaluate(model, loader, torch.device("cpu"))
        second = evaluate(model, loader, torch.device("cpu"))
        self.assertEqual(first["accuracy"], second["accuracy"])
        self.assertEqual(first["confusion_matrix"], second["confusion_matrix"])

    def test_seeding_reports_the_settings_it_applied(self) -> None:
        report = seed_everything(99)
        self.assertEqual(report["seed"], 99)
        self.assertTrue(report["cudnn_deterministic"])
        self.assertFalse(report["cudnn_benchmark"])


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix_is_always_28_by_28(self) -> None:
        matrix = confusion_matrix([0, 1, 1], [0, 1, 2])
        self.assertEqual(matrix.shape, (NUM_CLASSES, NUM_CLASSES))
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[1, 1], 1)
        self.assertEqual(matrix[1, 2], 1)

    def test_perfect_predictions_score_one(self) -> None:
        labels = list(range(NUM_CLASSES))
        metrics = classification_metrics(labels, labels)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["classes_present"], NUM_CLASSES)

    def test_macro_f1_matches_a_hand_computed_value(self) -> None:
        # class 0: TP=1 FP=1 FN=0 -> P=.5 R=1 F1=2/3 ; class 1: TP=1 FP=0 FN=1 -> F1=2/3
        metrics = classification_metrics([0, 1, 1], [0, 1, 0])
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0, places=6)

    def test_a_class_that_is_never_predicted_is_penalised(self) -> None:
        metrics = classification_metrics([0, 1], [0, 0])
        self.assertEqual(metrics["classes_present"], 2)
        self.assertAlmostEqual(metrics["per_class"]["recall"][1], 0.0)
        self.assertLess(metrics["macro_f1"], 1.0)

    def test_mismatched_lengths_and_out_of_range_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0])
        with self.assertRaises(ValueError):
            confusion_matrix([0], [NUM_CLASSES])

    def test_top_confusions_ranks_off_diagonal_pairs(self) -> None:
        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        matrix[3, 3] = 50   # diagonal must be ignored
        matrix[5, 7] = 9
        matrix[2, 1] = 4
        pairs = top_confusions(matrix, limit=5, labels_ar={5: "ز", 7: "س"})
        self.assertEqual(pairs[0]["true_label_index"], 5)
        self.assertEqual(pairs[0]["predicted_label_index"], 7)
        self.assertEqual(pairs[0]["count"], 9)
        self.assertEqual(pairs[0]["true_label_ar"], "ز")
        self.assertEqual(pairs[1]["count"], 4)

    def test_cross_entropy_is_computed_when_probabilities_are_supplied(self) -> None:
        probabilities = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
        metrics = classification_metrics([0, 1], [0, 1], probabilities=probabilities)
        self.assertAlmostEqual(metrics["cross_entropy"], float(np.log(NUM_CLASSES)), places=6)


class TestCheckpoints(unittest.TestCase):
    def _spec(self, **overrides) -> ExperimentSpec:
        base = dict(feature_set="full", quaternion_policy="absolute",
                    pooling="masked_mean", fold="01", seed=1337)
        base.update(overrides)
        return ExperimentSpec(**base)

    def test_save_and_load_round_trip_restores_identical_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig())
            path = save_checkpoint(Path(tmp) / "best.pt", model=model, spec=self._spec(),
                                   epoch=4, best_epoch=3, best_metric=0.42)
            payload = load_checkpoint(path, expect=self._spec())
            restored = rebuild_model(payload)
            self.assertEqual(payload["best_epoch"], 3)
            self.assertAlmostEqual(payload["best_metric"], 0.42)
            for original, loaded in zip(model.state_dict().values(),
                                        restored.state_dict().values()):
                self.assertTrue(torch.equal(original, loaded))

    def test_checkpoint_records_the_whole_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig())
            path = save_checkpoint(Path(tmp) / "c.pt", model=model, spec=self._spec())
            payload = load_checkpoint(path)
            self.assertEqual(payload["contract_version"], CONTRACT_VERSION)
            for field in ("feature_set", "quaternion_policy", "pooling", "fold", "seed",
                          "input_policy"):
                self.assertIn(field, payload["experiment"])
            for field in ("hidden_size", "num_layers", "dropout", "pooling"):
                self.assertIn(field, payload["model_config"])
            self.assertEqual(payload["input_dim"], 92)
            self.assertEqual(payload["num_classes"], NUM_CLASSES)

    def test_feature_set_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig(feature_set="bend_only"))
            path = save_checkpoint(Path(tmp) / "c.pt", model=model,
                                   spec=self._spec(feature_set="bend_only"))
            with self.assertRaises(CheckpointError):
                load_checkpoint(path, expect=self._spec(feature_set="full"))

    def test_pooling_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig(pooling="final_hidden"))
            path = save_checkpoint(Path(tmp) / "c.pt", model=model,
                                   spec=self._spec(pooling="final_hidden"))
            with self.assertRaises(CheckpointError):
                load_checkpoint(path, expect=self._spec(pooling="masked_mean"))

    def test_foreign_contract_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            model = LSTMBaseline(LSTMBaselineConfig())
            save_checkpoint(path, model=model, spec=self._spec())
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["contract_version"] = "task009a_sequence_input_v0"
            torch.save(payload, path)
            with self.assertRaises(CheckpointError):
                load_checkpoint(path)

    def test_missing_and_corrupt_checkpoints_are_rejected_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CheckpointError):
                load_checkpoint(Path(tmp) / "absent.pt")
            corrupt = Path(tmp) / "corrupt.pt"
            corrupt.write_bytes(b"not a torch archive")
            with self.assertRaises(CheckpointError):
                load_checkpoint(corrupt)

    def test_resume_state_survives_a_save_load_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig())
            optimizer = build_optimizer(model, TrainingConfig())
            history = [{"epoch": 1, "validation_macro_f1": 0.1}]
            path = save_checkpoint(
                Path(tmp) / "last.pt", model=model, spec=self._spec(),
                optimizer_state=optimizer.state_dict(), epoch=7, best_epoch=5,
                best_metric=0.31, early_stopping_counter=2, history=history)
            payload = load_checkpoint(path, expect=self._spec())
            self.assertEqual(payload["epoch"], 7)
            self.assertEqual(payload["early_stopping_counter"], 2)
            self.assertEqual(payload["history"], history)
            self.assertIsNotNone(payload["optimizer_state"])
            build_optimizer(rebuild_model(payload), TrainingConfig()).load_state_dict(
                payload["optimizer_state"])

    def test_experiment_slug_is_deterministic_and_fold_specific(self) -> None:
        self.assertEqual(self._spec().slug(),
                         "full__q-absolute__masked_mean__fold01__seed1337")
        self.assertEqual(self._spec(fold="03").slug(),
                         "full__q-absolute__masked_mean__fold03__seed1337")
        # quaternion policy is meaningless without quaternion channels
        self.assertIn("q-na", self._spec(feature_set="bend_only").slug())


class TestFoldIsolation(unittest.TestCase):
    def test_frozen_folds_have_the_expected_sizes_and_no_leakage(self) -> None:
        from recognition.data import load_all_folds, load_index

        records = load_index(ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
        folds = load_all_folds(ROOT / "datasets/splits", records)
        expected = {"01": (2372, 448, 1402), "02": (2373, 448, 1401),
                    "03": (2355, 448, 1419)}
        for signer, (train, validation, test) in expected.items():
            fold = folds[signer]
            self.assertEqual(fold.counts(),
                             {"train": train, "validation": validation, "test": test})
            self.assertEqual(fold.signers("test"), {signer})
            self.assertNotIn(signer, fold.signers("train"))
            self.assertNotIn(signer, fold.signers("validation"))
            ids = [r.sample_id for role in ("train", "validation", "test")
                   for r in fold.roles[role]]
            self.assertEqual(len(ids), len(set(ids)))


class TestDurationControl(unittest.TestCase):
    def test_length_only_classifier_learns_a_separable_length_signal(self) -> None:
        lengths = [10] * 40 + [50] * 40
        labels = [0] * 40 + [1] * 40
        model = LengthOnlyClassifier().fit(lengths, labels)
        metrics = model.evaluate(lengths, labels)
        self.assertGreater(metrics["accuracy"], 0.95)

    def test_length_only_classifier_cannot_separate_identical_lengths(self) -> None:
        lengths = [20] * 60
        labels = [i % 3 for i in range(60)]
        model = LengthOnlyClassifier().fit(lengths, labels)
        self.assertLess(model.evaluate(lengths, labels)["accuracy"], 0.5)

    def test_probabilities_span_the_full_28_class_space(self) -> None:
        model = LengthOnlyClassifier().fit([5, 6, 30, 31], [0, 0, 1, 1])
        probabilities = model.predict_proba([5, 30])
        self.assertEqual(probabilities.shape, (2, NUM_CLASSES))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        self.assertEqual(float(probabilities[:, 5:].sum()), 0.0)

    def test_predicting_before_fitting_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            LengthOnlyClassifier().predict([10])

    def test_oracle_accuracy_is_an_upper_bound(self) -> None:
        self.assertAlmostEqual(oracle_accuracy_from_length([1, 1, 2], [0, 0, 1]), 1.0)
        self.assertAlmostEqual(oracle_accuracy_from_length([1, 1], [0, 1]), 0.5)
        self.assertEqual(oracle_accuracy_from_length([], []), 0.0)


class TestInferenceApi(unittest.TestCase):
    def test_one_variable_length_sequence_predicts_an_authoritative_arabic_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = SequenceInputConfig(feature_set="full")
            model = LSTMBaseline(LSTMBaselineConfig())
            path = save_checkpoint(
                Path(tmp) / "best.pt", model=model,
                spec=ExperimentSpec("full", "absolute", "masked_mean", "01", 1337))
            recognizer = SequenceRecognizer.from_checkpoint(
                path, label_table=ROOT / "datasets/manifests/karsl_core28_labels.csv")
            table = load_label_table(ROOT / "datasets/manifests/karsl_core28_labels.csv")

            for frames in (9, 23, 70):
                prediction = recognizer.predict_sequence(_arrays(frames))
                self.assertEqual(prediction.sequence_length, frames)
                self.assertIn(prediction.label_index, range(NUM_CLASSES))
                reference = table[prediction.label_index]
                self.assertEqual(prediction.label_ar, reference.label_ar)
                self.assertEqual(prediction.sign_id, reference.sign_id)
                self.assertEqual(len(prediction.logits), NUM_CLASSES)
                self.assertAlmostEqual(sum(prediction.probabilities), 1.0, places=5)
                self.assertAlmostEqual(
                    prediction.confidence, max(prediction.probabilities), places=6)

    def test_inference_needs_no_training_machinery(self) -> None:
        # Building a recognizer must not require an optimizer, loader or loop.
        with tempfile.TemporaryDirectory() as tmp:
            model = LSTMBaseline(LSTMBaselineConfig(feature_set="bend_spread"))
            path = save_checkpoint(
                Path(tmp) / "c.pt", model=model,
                spec=ExperimentSpec("bend_spread", "absolute", "masked_mean", "02", 7))
            recognizer = SequenceRecognizer.from_checkpoint(path)
            # The config comes from the checkpoint, not from the caller.
            self.assertEqual(recognizer.config.feature_set, "bend_spread")
            prediction = recognizer.predict_sequence(_arrays(12))
            self.assertEqual(prediction.sequence_length, 12)

    def test_top_k_is_ordered_by_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(
                Path(tmp) / "c.pt", model=LSTMBaseline(LSTMBaselineConfig()),
                spec=ExperimentSpec("full", "absolute", "masked_mean", "01", 1))
            recognizer = SequenceRecognizer.from_checkpoint(path)
            top = recognizer.predict_sequence(_arrays(15)).top_k(5)
            probabilities = [entry["probability"] for entry in top]
            self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_a_sequence_with_one_missing_hand_still_predicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(
                Path(tmp) / "c.pt", model=LSTMBaseline(LSTMBaselineConfig()),
                spec=ExperimentSpec("full", "absolute", "masked_mean", "01", 1))
            recognizer = SequenceRecognizer.from_checkpoint(path)
            present = np.ones((14, HANDS), dtype=bool)
            present[:, 0] = False  # LEFT absent for the whole sequence
            prediction = recognizer.predict_sequence(_arrays(14, hand_present=present))
            self.assertEqual(prediction.sequence_length, 14)
            self.assertIn(prediction.label_index, range(NUM_CLASSES))


class TestLabelTable(unittest.TestCase):
    def test_frozen_table_is_28_contiguous_classes(self) -> None:
        table = load_label_table(ROOT / "datasets/manifests/karsl_core28_labels.csv")
        self.assertEqual(sorted(table), list(range(NUM_CLASSES)))
        self.assertEqual(table[0].label_ar, "ا")
        self.assertEqual(table[0].sign_id, "0032")
        self.assertEqual(table[27].label_ar, "ي")
        self.assertEqual(table[27].sign_id, "0059")

    def test_a_short_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("sign_id,label_ar,label_en_if_available,label_index\n"
                            "0032,ا,alif,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_label_table(path)


if __name__ == "__main__":
    unittest.main()
