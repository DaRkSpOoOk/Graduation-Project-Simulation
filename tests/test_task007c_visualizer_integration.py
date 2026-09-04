"""End-to-end tests for the independent TASK-007A/TASK-007B integration."""

from __future__ import annotations

import unittest
from pathlib import Path

from visualizer import load_sequence
from visualizer.app.integration import (
    QueuePlaybackSession,
    VisualizerIntegrationError,
    load_sequence_for_item,
    run_headless_queue,
)
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueueItem, QueueState, UnsupportedTextError


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
MANIFEST = ROOT / "datasets" / "manifests" / "karsl_core28.csv"
CATALOG = ROOT / "visualizer" / "catalog" / "core28_exemplars.json"
MISSING_HAND_SAMPLE = "karsl_core28_s01_sign0037_test_rep007"


@unittest.skipUnless(RUN_ROOT.is_dir(), "TASK-008 production run is not available")
class Task007CProductionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = Core28Resolver(catalog_path=CATALOG)

    def test_all_28_canonical_exemplars_load_through_adapter(self) -> None:
        self.assertEqual(len(self.resolver.mapping.labels), 28)
        for label in self.resolver.mapping.labels:
            with self.subTest(character=label.character, sign_id=label.sign_id):
                resolution = self.resolver.resolve_character(label.character)
                item = PlaybackQueueItem.from_resolution(resolution)
                sequence = load_sequence_for_item(item, run_root=RUN_ROOT, manifest_path=MANIFEST)
                self.assertIsNotNone(sequence)
                assert sequence is not None
                self.assertEqual(sequence.sample_id, resolution.sample_id)
                self.assertEqual(len(sequence), resolution.sequence_descriptor.sequence_length)
                self.assertEqual(sequence.label_ar, label.character)
                self.assertEqual(sequence.label_index, label.label_index)
                self.assertEqual(len(sequence.sensor_layout), 20)
                self.assertTrue(any(hand.present for frame in sequence.frames for hand in frame.hands))
                mesh = sequence.metadata["mesh"]
                self.assertTrue(mesh["embedded_mano_vertices_available"])
                self.assertTrue(mesh["tracked_landmarks_3d_available"])
                self.assertFalse(mesh["surface_triangle_topology_available"])

    def test_muhammad_has_exact_logical_order_and_repeated_m(self) -> None:
        result = run_headless_queue(
            "محمد",
            run_root=RUN_ROOT,
            resolver=self.resolver,
            manifest_path=MANIFEST,
        )
        self.assertEqual([item["character"] for item in result.played], list("محمد"))
        self.assertEqual([item["sign_id"] for item in result.played], ["0055", "0037", "0055", "0039"])
        self.assertEqual(
            [item["sample_id"] for item in result.played],
            [
                "karsl_core28_s01_sign0055_train_rep010",
                "karsl_core28_s01_sign0037_train_rep026",
                "karsl_core28_s01_sign0055_train_rep010",
                "karsl_core28_s03_sign0039_test_rep001",
            ],
        )
        self.assertEqual(result.completed_indices, (0, 1, 2, 3))
        self.assertTrue(result.queue_complete)
        self.assertTrue(all(item["has_geometry"] for item in result.played))

    def test_completion_advances_session_without_manual_reload(self) -> None:
        session = QueuePlaybackSession(run_root=RUN_ROOT, resolver=self.resolver, manifest_path=MANIFEST)
        items = session.enqueue_text("محمد")
        self.assertEqual(session.start(), items[0])
        for index, item in enumerate(items):
            self.assertIs(session.queue.current, item)
            self.assertEqual(item.state, QueueState.PLAYING)
            next_item = session.complete_current()
            self.assertEqual(item.state, QueueState.COMPLETED)
            if index < len(items) - 1:
                self.assertIs(next_item, items[index + 1])
                self.assertIs(session.current_item, items[index + 1])
                self.assertIsNotNone(session.current_sequence)
            else:
                self.assertIsNone(next_item)
        self.assertTrue(session.queue.is_complete)

    def test_neutral_gap_is_not_loaded_as_a_sequence_or_frame(self) -> None:
        result = run_headless_queue(
            "م ح",
            run_root=RUN_ROOT,
            resolver=self.resolver,
            manifest_path=MANIFEST,
        )
        self.assertEqual([item["item_type"] for item in result.played], ["sign", "gap", "sign"])
        gap = result.played[1]
        self.assertEqual(gap["character"], " ")
        self.assertIsNone(gap["sample_id"])
        self.assertIsNone(gap["sequence_length"])
        self.assertFalse(gap["has_geometry"])
        self.assertEqual(result.completed_indices, (0, 1, 2))

    def test_unsupported_text_is_atomic_and_explicit(self) -> None:
        session = QueuePlaybackSession(run_root=RUN_ROOT, resolver=self.resolver, manifest_path=MANIFEST)
        with self.assertRaises(UnsupportedTextError):
            session.enqueue_text("مأ")
        self.assertEqual(session.queue.items, ())

    def test_exemplar_modes_are_explicit_and_deterministic(self) -> None:
        for mode in ("signer01", "signer02", "signer03"):
            with self.subTest(mode=mode):
                result = self.resolver.resolve_character("م", mode=mode)
                self.assertEqual(result.signer_id, mode[-2:])
        first = self.resolver.resolve_character("م", mode="random", rng_seed=17)
        second = self.resolver.resolve_character("م", mode="random", rng_seed=17)
        self.assertEqual(first.sample_id, second.sample_id)
        with self.assertRaises(ValueError):
            self.resolver.resolve_character("م", mode="random")

    def test_embedded_geometry_terminology_is_precise(self) -> None:
        entries = self.resolver.catalog.entries
        self.assertEqual(len(entries), 28)
        for entry in entries:
            flags = entry.metrics["artifact_flags"]
            self.assertTrue(flags["embedded_mano_vertices_available"])
            self.assertTrue(flags["tracked_landmarks_3d_available"])
            self.assertFalse(flags["surface_triangle_topology_available"])
        self.assertEqual(self.resolver.catalog.source.get("run_root"), str(RUN_ROOT))

    def test_missing_hand_preserves_track_identity_and_masks(self) -> None:
        sequence = load_sequence(RUN_ROOT, MISSING_HAND_SAMPLE, manifest_path=MANIFEST)
        left = sequence.frame_at(0).hand("LEFT")
        right = sequence.frame_at(0).hand("RIGHT")
        self.assertEqual(left.state, "MISSING")
        self.assertFalse(left.present)
        self.assertTrue(right.present)
        self.assertTrue(all(not reading.valid for reading in sequence.sensor_readings(0, "LEFT")))
        self.assertTrue(any(reading.valid for reading in sequence.sensor_readings(0, "RIGHT")))

    def test_descriptor_loader_uses_configured_run_root_and_exact_frames(self) -> None:
        resolution = self.resolver.resolve_character("م")
        descriptor = resolution.sequence_descriptor
        item = PlaybackQueueItem.from_resolution(resolution)
        sequence = load_sequence_for_item(item, run_root=RUN_ROOT, manifest_path=MANIFEST)
        assert sequence is not None
        self.assertEqual(len(sequence.frame_indices), descriptor.sequence_length)
        self.assertTrue(all(right > left for left, right in zip(sequence.frame_indices, sequence.frame_indices[1:])))
        self.assertTrue(all(right > left for left, right in zip(sequence.timestamps, sequence.timestamps[1:])))
        self.assertEqual(descriptor.sample_id, sequence.sample_id)

    def test_headless_queue_returns_metadata_without_generating_ml_frames(self) -> None:
        result = run_headless_queue(
            "م",
            run_root=RUN_ROOT,
            resolver=self.resolver,
            manifest_path=MANIFEST,
        )
        self.assertEqual(len(result.played), 1)
        self.assertEqual(
            set(result.played[0]),
            {
                "index",
                "item_type",
                "character",
                "sign_id",
                "sample_id",
                "sequence_length",
                "has_geometry",
            },
        )
        self.assertNotIn("frames", result.played[0])
        self.assertNotIn("sensor_values", result.played[0])

    def test_sign_item_without_descriptor_is_rejected(self) -> None:
        item = PlaybackQueueItem(item_type="sign", character="م", sign_id="0055")
        with self.assertRaises(VisualizerIntegrationError):
            load_sequence_for_item(item, run_root=RUN_ROOT, manifest_path=MANIFEST)


if __name__ == "__main__":
    unittest.main()
