"""TASK-007F persistent Qt Quick 3D application contract tests."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from virtual_glove.layout import layout_document
from visualizer.contract import FrameData, HandGeometry, PlaybackSequence, validate_sensor_layout
from visualizer.keyboard import Core28Keyboard
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueue

from smart_glove_app.app.recognition_bridge import RecognitionBridge
from smart_glove_app.rendering.hand_mesh_state import PersistentRenderScene, compute_vertex_normals
from smart_glove_app.rendering.mano_topology import (
    MANO_FACE_COUNT,
    MANO_VERTEX_COUNT,
    ManoTopologyError,
    load_mano_topology,
    topology_from_faces,
    validate_mano_faces,
)
from smart_glove_app.rendering.qt_geometry import QT_QUICK3D_AVAILABLE, QtHandGeometry
from smart_glove_app.rendering.sensor_markers import SensorMarkerModel


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORE28 = "\u0627\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u064a"


def _complete_faces() -> np.ndarray:
    """Return a deterministic full-coverage 1538-triangle test topology."""

    faces = np.empty((MANO_FACE_COUNT, 3), dtype=np.int64)
    for index in range(MANO_FACE_COUNT):
        first = index % MANO_VERTEX_COUNT
        faces[index] = (first, (first + 1) % MANO_VERTEX_COUNT, (first + 2) % MANO_VERTEX_COUNT)
    return faces


def _mesh(seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 0.25, size=(MANO_VERTEX_COUNT, 3)).astype(np.float32)
    points[:, 1] += np.float32(offset)
    return points


def _landmarks(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(-0.4, 0.4, 21, dtype=np.float32)
    points[:, 1] = np.linspace(-0.5, 0.8, 21, dtype=np.float32) + np.float32(offset)
    points[:, 2] = np.linspace(0.1, -0.1, 21, dtype=np.float32)
    return points


def _frame(position: int, *, missing_left: bool = False) -> FrameData:
    layout = validate_sensor_layout(layout_document())
    left = HandGeometry(
        "LEFT",
        "MISSING" if missing_left else "PRESENT",
        -1 if missing_left else 0,
        None if missing_left else _landmarks(0.0),
        None if missing_left else _mesh(11, 0.0),
    )
    right = HandGeometry("RIGHT", "PRESENT", 1, _landmarks(0.1), _mesh(12, 0.08 + position * 0.01))
    return FrameData(
        position=position,
        frame_index=100 + position,
        timestamp_seconds=position / 60.0,
        hands=(left, right),
        bend_normalized=np.zeros((2, 5, 3), dtype=np.float32),
        spread_normalized=np.zeros((2, 4), dtype=np.float32),
        bend_valid=np.ones((2, 5, 3), dtype=bool),
        spread_valid=np.ones((2, 4), dtype=bool),
        palm_quaternion_wxyz=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (2, 1)),
        palm_imu_valid=np.ones(2, dtype=bool),
    )


def _sequence() -> PlaybackSequence:
    return PlaybackSequence(
        sample_id="task007f-test-sample",
        label_ar="\u0645",
        label_index=18,
        signer_id="01",
        frames=(_frame(0), _frame(1, missing_left=True)),
        sensor_layout=validate_sensor_layout(layout_document()),
        geometry_source="synthetic-test-only",
        metadata={},
    )


class Task007FTopologyTests(unittest.TestCase):
    def test_topology_loader_extracts_authoritative_faces_and_hashes_source(self) -> None:
        faces = _complete_faces()
        neutral = _mesh(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MANO_RIGHT.pkl"
            path.write_bytes(pickle.dumps({"f": faces, "v_template": neutral}, protocol=4))
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            topology = load_mano_topology(path)

        self.assertIsNotNone(topology)
        assert topology is not None
        self.assertEqual(topology.vertex_count, MANO_VERTEX_COUNT)
        self.assertEqual(topology.face_count, MANO_FACE_COUNT)
        self.assertEqual(topology.source_hand, "RIGHT")
        self.assertEqual(topology.source_sha256, source_hash)
        np.testing.assert_array_equal(topology.faces_for_track("LEFT")[0], faces[0][[0, 2, 1]])
        np.testing.assert_allclose(topology.neutral_for_track("LEFT")[:, 0], -neutral[:, 0])

    def test_left_asset_is_canonicalized_then_reflected_with_reversed_winding(self) -> None:
        faces = _complete_faces()
        neutral = _mesh(43)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MANO_LEFT.pkl"
            path.write_bytes(pickle.dumps({"f": faces, "v_template": neutral}, protocol=4))
            topology = load_mano_topology(path)

        assert topology is not None
        np.testing.assert_array_equal(topology.faces[0], faces[0][[0, 2, 1]])
        np.testing.assert_array_equal(topology.faces_for_track("LEFT")[0], faces[0])
        np.testing.assert_allclose(topology.neutral_vertices[:, 0], -neutral[:, 0])

    def test_mirrored_left_triangle_uses_reversed_winding_for_the_same_front_normal(self) -> None:
        right = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
        left = right.copy()
        left[:, 0] *= -1.0
        right_normal = compute_vertex_normals(right, np.asarray(((0, 1, 2),), dtype=np.int64))
        left_normal = compute_vertex_normals(left, np.asarray(((0, 2, 1),), dtype=np.int64))
        np.testing.assert_allclose(right_normal, left_normal, atol=1e-6)

    def test_topology_validation_rejects_wrong_count_range_and_coverage(self) -> None:
        faces = _complete_faces()
        with self.assertRaises(ManoTopologyError):
            validate_mano_faces(faces[:10])
        invalid_range = faces[:1].copy()
        invalid_range[0, 2] = MANO_VERTEX_COUNT
        with self.assertRaises(ManoTopologyError):
            validate_mano_faces(invalid_range, expected_face_count=None, require_full_vertex_coverage=False)
        with self.assertRaises(ManoTopologyError):
            validate_mano_faces(faces[:100], expected_face_count=None, require_full_vertex_coverage=True)

    def test_topology_from_faces_supports_deterministic_in_memory_smoke_mesh(self) -> None:
        topology = topology_from_faces(_complete_faces(), source_path="<test>")
        self.assertEqual(topology.face_count, MANO_FACE_COUNT)
        self.assertEqual(topology.faces.dtype, np.uint32)
        self.assertFalse(topology.faces.flags.writeable)


class Task007FPersistentSceneTests(unittest.TestCase):
    def test_both_hands_are_created_once_and_frame_updates_keep_their_identity(self) -> None:
        topology = topology_from_faces(_complete_faces(), source_path="<test>")
        scene = PersistentRenderScene(topology)
        tokens = scene.geometry_tokens
        self.assertEqual(scene.scene_creation_count, 1)
        self.assertEqual(scene.view3d_creation_count, 1)
        self.assertEqual(scene.left.geometry_creation_count, 1)
        self.assertEqual(scene.right.geometry_creation_count, 1)
        self.assertTrue(scene.left.payload.visible)
        self.assertTrue(scene.right.payload.visible)
        self.assertEqual(scene.left.payload.state, "IDLE")
        self.assertEqual(scene.right.payload.state, "IDLE")

        sequence = _sequence()
        source_before = sequence.frame_at(0).hand("RIGHT").mesh_vertices.copy()
        scene.attach_sequence(sequence)
        interpolated = scene.update_sequence_frame(0, interpolation_alpha=0.5, smooth=True)
        self.assertTrue(interpolated.interpolated)
        self.assertEqual(interpolated.source_positions, (0, 1))
        self.assertIsNotNone(interpolated.left.normals)
        self.assertIsNotNone(interpolated.right.normals)
        self.assertEqual(scene.geometry_tokens, tokens)
        self.assertEqual(scene.left.geometry_creation_count, 1)
        self.assertEqual(scene.right.geometry_creation_count, 1)
        np.testing.assert_array_equal(sequence.frame_at(0).hand("RIGHT").mesh_vertices, source_before)

    def test_missing_observation_keeps_a_dimmed_visible_hand(self) -> None:
        scene = PersistentRenderScene(topology_from_faces(_complete_faces(), source_path="<test>"))
        scene.attach_sequence(_sequence())
        scene.update_sequence_frame(0)
        frame = scene.update_sequence_frame(1)
        self.assertTrue(frame.left.visible)
        self.assertEqual(frame.left.state, "MISSING")
        self.assertTrue(frame.left.dimmed)
        self.assertEqual(frame.left.source, "last-visible-pose")
        self.assertTrue(any(value is not None for value in frame.left.marker_positions.values()))
        self.assertTrue(frame.left.marker_valid)
        self.assertTrue(all(not value for value in frame.left.marker_valid.values()))
        self.assertTrue(frame.right.visible)
        self.assertFalse(frame.right.dimmed)
        self.assertEqual(scene.topology_status, "MANO SURFACE · 778 VERTICES")

    def test_no_topology_has_explicit_point_cloud_fallback_and_idle_scene(self) -> None:
        scene = PersistentRenderScene()
        self.assertFalse(scene.topology_available)
        self.assertEqual(scene.topology_status, "SURFACE TOPOLOGY UNAVAILABLE — POINT-CLOUD FALLBACK")
        self.assertEqual(scene.left.payload.vertices.shape, (MANO_VERTEX_COUNT, 3))
        self.assertEqual(scene.right.payload.vertices.shape, (MANO_VERTEX_COUNT, 3))
        self.assertIsNone(scene.left.payload.normals)
        self.assertIsNone(scene.right.payload.normals)

    def test_interpolation_is_presentation_only_and_bridge_accepts_queue_items(self) -> None:
        sequence = _sequence()
        before = sequence.frame_at(0).hand("RIGHT").mesh_vertices.copy()
        scene = PersistentRenderScene(topology_from_faces(_complete_faces(), source_path="<test>"))
        scene.attach_sequence(sequence)
        frame = scene.update_sequence_frame(0, interpolation_alpha=0.25, smooth=True)
        self.assertTrue(frame.interpolated)
        np.testing.assert_array_equal(sequence.frame_at(0).hand("RIGHT").mesh_vertices, before)
        bridge = RecognitionBridge()
        self.assertFalse(bridge.enabled)
        self.assertIsNone(bridge.predict(object()))
        self.assertNotIn("PresentationFrame", inspect.signature(bridge.predict).parameters)


class Task007FContractTests(unittest.TestCase):
    def test_core28_keyboard_uses_authoritative_logical_order_and_repeated_queue_events(self) -> None:
        resolver = Core28Resolver()
        keyboard = Core28Keyboard(resolver.mapping)
        self.assertEqual("".join(key.character for key in keyboard.keys), EXPECTED_CORE28)
        self.assertEqual(sum(len(row) for row in keyboard.rtl_rows), 28)

        text = "\u0645\u062d\u0645\u062f"
        queue = PlaybackQueue(resolver)
        items = queue.enqueue_text(text)
        self.assertEqual("".join(item.character for item in items), text)
        self.assertEqual([item.sign_id for item in items], ["0055", "0037", "0055", "0039"])
        self.assertIsNot(items[0], items[2])
        self.assertEqual([item.sample_id for item in items[:1]], ["karsl_core28_s01_sign0055_train_rep010"])

    def test_qml_owns_one_persistent_view3d_and_no_legacy_gui_dependency(self) -> None:
        main_qml = (ROOT / "smart_glove_app" / "qml" / "Main.qml").read_text(encoding="utf-8")
        viewport_qml = (ROOT / "smart_glove_app" / "qml" / "components" / "HandViewport.qml").read_text(
            encoding="utf-8"
        )
        python_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "smart_glove_app").rglob("*.py")
        )
        self.assertEqual(viewport_qml.count("View3D {"), 1)
        self.assertIn("FrameAnimation", main_qml)
        self.assertIn("leftGeometry", viewport_qml)
        self.assertIn("rightGeometry", viewport_qml)
        self.assertNotIn("Tkinter", python_sources)
        self.assertNotIn("matplotlib", python_sources.lower())
        self.assertNotIn("C:\\Users\\hatem", python_sources)
        self.assertNotIn("/home/hatim", python_sources)

    def test_qt_geometry_buffers_are_updated_without_recreating_objects(self) -> None:
        if not QT_QUICK3D_AVAILABLE:
            self.skipTest("PySide6 Qt Quick 3D is not installed")
        from PySide6.QtGui import QGuiApplication

        app = QGuiApplication.instance() or QGuiApplication([])
        del app
        topology = topology_from_faces(_complete_faces(), source_path="<test>")
        scene = PersistentRenderScene(topology)
        geometry = QtHandGeometry("LEFT", topology)
        geometry.set_payload(scene.left.payload)
        identity = geometry.scene_object_token
        vertex_size = len(geometry.vertexData())
        index_data = bytes(geometry.indexData())
        scene.attach_sequence(_sequence())
        geometry.set_payload(scene.update_sequence_frame(0).left)
        self.assertEqual(geometry.geometry_creation_count, 1)
        self.assertEqual(geometry.scene_object_token, identity)
        self.assertEqual(geometry.update_count, 2)
        self.assertEqual(len(geometry.vertexData()), vertex_size)
        self.assertEqual(bytes(geometry.indexData()), index_data)

    def test_sensor_marker_rows_update_in_place_without_model_reset(self) -> None:
        if not QT_QUICK3D_AVAILABLE:
            self.skipTest("PySide6 Qt Quick 3D is not installed")
        from PySide6.QtGui import QGuiApplication

        app = QGuiApplication.instance() or QGuiApplication([])
        del app
        markers = SensorMarkerModel(validate_sensor_layout(layout_document()))
        rows_identity = id(markers._rows)
        row_identities = tuple(id(row) for row in markers._rows)
        markers.update_markers({"IMU_PALM": np.asarray((1.0, 2.0, 3.0))}, {"IMU_PALM": True})
        self.assertEqual(markers.model_creation_count, 1)
        self.assertEqual(markers.update_count, 1)
        self.assertEqual(id(markers._rows), rows_identity)
        self.assertEqual(tuple(id(row) for row in markers._rows), row_identities)
        self.assertEqual(markers.rowCount(), 20)


if __name__ == "__main__":
    unittest.main()
