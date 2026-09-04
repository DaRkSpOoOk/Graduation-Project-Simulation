"""TASK-007D recognizer/visualizer integration tests.

The tests exercise the adapter contract with small doubles and, when the
already-produced local artifacts are available, one real engineering smoke
path.  They never train a model and never score a scientific benchmark.
"""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from visualizer.app.integration import run_headless_queue, run_headless_recognizer_queue
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueueItem
from visualizer.recognition import (
    CheckpointCompatibilityError,
    RecognitionController,
    RecognitionResult,
    RecognizerAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
CHECKPOINT = Path(
    "/home/hatim/graduation-project-runs/task009b-core28-lstm/"
    "full/q-absolute/masked_mean/fold01/seed1337/best.pt"
)
LABELS = ROOT / "datasets/manifests/karsl_core28_labels.csv"
MANIFEST = ROOT / "datasets/manifests/karsl_core28.csv"
CATALOG = ROOT / "visualizer/catalog/core28_exemplars.json"


class _CountingAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.metadata = None
        self.calls: list[str] = []
        self.fail = fail

    def clear_cache(self) -> None:
        pass

    def predict_queue_item(self, item):
        self.calls.append(str(item.sample_id))
        if self.fail:
            raise RuntimeError("synthetic model failure")
        return RecognitionResult(
            sample_id=str(item.sample_id),
            expected_label_index=item.label_index,
            expected_character=item.character,
            expected_sign_id=item.sign_id,
            predicted_label_index=0,
            predicted_character="ا",
            predicted_sign_id="0032",
            confidence=0.5,
            probabilities=(0.5,) + (0.5 / 27,) * 27,
            top_k=({"label_index": 0, "character": "ا", "probability": 0.5},),
            sequence_length=2,
            checkpoint_metadata=None,
            available=True,
        )


class Task007DUnitTests(unittest.TestCase):
    def _item(self, sample_id: str = "sample") -> PlaybackQueueItem:
        return PlaybackQueueItem(
            item_type="sign",
            character="م",
            sign_id="0055",
            label_index=23,
            sample_id=sample_id,
        )

    def test_controller_caches_sign_and_skips_gap(self) -> None:
        adapter = _CountingAdapter()
        controller = RecognitionController(adapter)
        first = controller.result_for(self._item())
        second = controller.result_for(self._item())
        self.assertIs(first, second)
        self.assertEqual(adapter.calls, ["sample"])
        self.assertIsNone(controller.result_for(PlaybackQueueItem.neutral_gap(" ")))
        self.assertIsNone(controller.current.result)

    def test_controller_marks_missing_checkpoint_without_prediction(self) -> None:
        controller = RecognitionController(disabled_reason="checkpoint was not selected")
        result = controller.result_for(self._item())
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.available)
        self.assertIsNone(result.predicted_character)
        self.assertEqual(result.expected_character, "م")

    def test_model_error_becomes_unavailable_and_does_not_change_queue_state(self) -> None:
        adapter = _CountingAdapter(fail=True)
        controller = RecognitionController(adapter)
        item = self._item()
        result = controller.result_for(item)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.available)
        self.assertEqual(item.state.value, "PENDING")
        self.assertEqual(adapter.calls, ["sample"])

    def test_result_serialization_never_adds_frames_or_expected_prediction(self) -> None:
        result = RecognitionResult.unavailable(
            sample_id="sample",
            expected_label_index=23,
            expected_character="م",
            expected_sign_id="0055",
            checkpoint_metadata=None,
            error="disabled",
        )
        payload = result.to_dict()
        self.assertIsNone(payload["predicted_character"])
        self.assertNotIn("frames", payload)
        self.assertNotIn("sensor_values", payload)

    def test_invalid_checkpoint_path_is_rejected(self) -> None:
        with self.assertRaises(CheckpointCompatibilityError):
            RecognizerAdapter.from_checkpoint(
                ROOT / "does-not-exist.pt",
                run_root=RUN_ROOT,
                labels_path=LABELS,
            )

    def test_incompatible_checkpoint_contract_is_rejected(self) -> None:
        from recognition.models.lstm_baseline import LSTMBaseline, LSTMBaselineConfig
        from recognition.training.checkpoint import ExperimentSpec, save_checkpoint

        model = LSTMBaseline(
            LSTMBaselineConfig(
                feature_set="bend_only",
                input_policy="values_and_feature_valid",
                pooling="masked_mean",
            )
        )
        spec = ExperimentSpec(
            feature_set="bend_only",
            quaternion_policy="absolute",
            pooling="masked_mean",
            fold="01",
            seed=1337,
        )
        with TemporaryDirectory() as temporary:
            checkpoint = save_checkpoint(Path(temporary) / "bend-only.pt", model=model, spec=spec)
            with self.assertRaises(CheckpointCompatibilityError):
                RecognizerAdapter.from_checkpoint(
                    checkpoint,
                    run_root=RUN_ROOT,
                    labels_path=LABELS,
                )

    def test_visualizer_only_headless_path_remains_available_without_checkpoint(self) -> None:
        if not RUN_ROOT.is_dir():
            self.skipTest("TASK-008 production run is not available")
        result = run_headless_queue(
            "م",
            run_root=RUN_ROOT,
            resolver=Core28Resolver(catalog_path=CATALOG),
            manifest_path=MANIFEST,
        )
        self.assertTrue(result.queue_complete)
        self.assertNotIn("recognition", result.played[0])


@unittest.skipUnless(
    RUN_ROOT.is_dir() and CHECKPOINT.is_file(),
    "real TASK-008 run and TASK-009B engineering checkpoint are not available",
)
class Task007DRealSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = Core28Resolver(catalog_path=CATALOG)
        cls.adapter = RecognizerAdapter.from_checkpoint(
            CHECKPOINT,
            run_root=RUN_ROOT,
            labels_path=LABELS,
        )

    def test_checkpoint_metadata_matches_preregistered_primary(self) -> None:
        metadata = self.adapter.metadata
        self.assertEqual(metadata.feature_set, "full")
        self.assertEqual(metadata.quaternion_policy, "absolute")
        self.assertEqual(metadata.pooling, "masked_mean")
        self.assertEqual(metadata.input_policy, "values_and_feature_valid")
        self.assertEqual(metadata.input_dim, 92)
        self.assertEqual(metadata.num_classes, 28)
        self.assertEqual(metadata.fold, "01")
        self.assertEqual(metadata.seed, 1337)
        self.assertTrue(metadata.demo_only)
        self.assertEqual(metadata.held_out_signer, "S01")

    def test_real_m_sample_uses_exact_stored_sequence(self) -> None:
        resolution = self.resolver.resolve_character("م")
        item = PlaybackQueueItem.from_resolution(resolution)
        result = self.adapter.predict_queue_item(item)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.available)
        self.assertEqual(result.sample_id, resolution.sample_id)
        self.assertEqual(result.expected_character, "م")
        self.assertEqual(result.expected_sign_id, "0055")
        self.assertEqual(len(result.probabilities), 28)
        self.assertEqual(result.sequence_length, resolution.sequence_descriptor.sequence_length)
        self.assertEqual(result.predicted_label_index, self.adapter.labels[result.predicted_label_index].label_index)

    def test_muhammad_gets_one_prediction_per_sign_and_advances(self) -> None:
        result = run_headless_recognizer_queue(
            "محمد",
            run_root=RUN_ROOT,
            recognition_adapter=self.adapter,
            resolver=self.resolver,
            manifest_path=MANIFEST,
        )
        self.assertTrue(result.queue_complete)
        self.assertEqual(result.completed_indices, (0, 1, 2, 3))
        self.assertEqual([entry["character"] for entry in result.played], list("محمد"))
        self.assertEqual([entry["sign_id"] for entry in result.played], ["0055", "0037", "0055", "0039"])
        self.assertTrue(all(entry["recognition"]["available"] for entry in result.played))
        self.assertEqual(
            [entry["recognition"]["expected_character"] for entry in result.played],
            list("محمد"),
        )
        self.assertEqual(len(self.adapter._cache), 3, "repeated م may reuse the same cached sequence result")
        self.assertTrue(all("frames" not in entry["recognition"] for entry in result.played))

    def test_gap_is_not_sent_to_recognizer(self) -> None:
        before = set(self.adapter._cache)
        result = run_headless_recognizer_queue(
            "م ",
            run_root=RUN_ROOT,
            recognition_adapter=self.adapter,
            resolver=self.resolver,
            manifest_path=MANIFEST,
        )
        self.assertIsNone(result.played[1]["recognition"])
        self.assertEqual(set(self.adapter._cache), before | {result.played[0]["sample_id"]})


if __name__ == "__main__":
    unittest.main()
