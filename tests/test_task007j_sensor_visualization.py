"""TASK-007J virtual-glove sensor overlay and panel contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from virtual_glove.layout import layout_document
from visualizer.contract import SensorReading, validate_sensor_layout

from smart_glove_app.rendering.sensor_layout import (
    SensorPresentationLayoutError,
    load_sensor_presentation_layout,
)
from smart_glove_app.rendering.sensor_markers import (
    QT_MARKERS_AVAILABLE,
    SensorValueModel,
)


ROOT = Path(__file__).resolve().parents[1]
SENSOR_LAYOUT_PATH = (
    ROOT / "smart_glove_app" / "assets" / "sensor_layouts" / "core28_virtual_glove_v1.json"
)


class SensorPresentationLayoutTests(unittest.TestCase):
    def test_map_is_exactly_the_frozen_twenty_package_order(self) -> None:
        scientific = validate_sensor_layout(layout_document())
        presentation = load_sensor_presentation_layout()

        self.assertEqual(len(presentation), 20)
        self.assertEqual(
            [spec.sensor_id for spec in presentation],
            [spec.sensor_id for spec in scientific],
        )
        self.assertEqual(sum(spec.sensor_type.startswith("hall_") for spec in presentation), 19)
        self.assertEqual(sum(spec.sensor_type == "imu_package" for spec in presentation), 1)
        self.assertTrue(all(spec.marker == "H" for spec in presentation[:19]))
        self.assertEqual(presentation[-1].marker, "IMU")
        self.assertEqual(presentation[0].description, scientific[0].description)

    def test_anatomical_anchor_map_is_explicit_and_dorsal(self) -> None:
        presentation = {spec.sensor_id: spec for spec in load_sensor_presentation_layout()}

        self.assertEqual(presentation["H_THUMB_PROXIMAL"].anchor_bone, "thumb_1")
        self.assertEqual(presentation["H_THUMB_MIDDLE"].anchor_bone, "thumb_2")
        self.assertEqual(presentation["H_THUMB_DISTAL"].anchor_bone, "thumb_3")
        self.assertEqual(presentation["H_INDEX_PROXIMAL"].anchor_bone, "index_1")
        self.assertEqual(presentation["H_INDEX_MIDDLE"].anchor_bone, "index_2")
        self.assertEqual(presentation["H_INDEX_DISTAL"].anchor_bone, "index_3")
        self.assertEqual(
            presentation["H_SPREAD_INDEX_MIDDLE"].anchor_bones,
            ("index_meta", "middle_meta"),
        )
        self.assertEqual(presentation["IMU_PALM"].anchor_bone, "palm")
        # Blender MCP calibration showed local +Z points toward the dorsal
        # surface in the exported TASK-007G canonical frame.
        self.assertTrue(all(spec.local_offset[2] > 0 for spec in presentation.values()))

    def test_wrong_value_channel_is_rejected(self) -> None:
        payload = json.loads(SENSOR_LAYOUT_PATH.read_text(encoding="utf-8"))
        payload["sensors"][0]["array_index"] = [1, 0]
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "invalid.json"
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SensorPresentationLayoutError):
                load_sensor_presentation_layout(candidate)


@unittest.skipUnless(QT_MARKERS_AVAILABLE, "PySide6 is not installed")
class SensorValueModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtGui import QGuiApplication

        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def _index_for(self, model: SensorValueModel, sensor_id: str):
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if model.data(index, model.SensorIdRole) == sensor_id:
                return index
        self.fail(f"sensor row not found: {sensor_id}")

    def test_values_are_exact_source_readings_and_rows_are_persistent(self) -> None:
        presentation = load_sensor_presentation_layout()
        scientific = {spec.sensor_id: spec for spec in validate_sensor_layout(layout_document())}
        model = SensorValueModel(presentation)
        rows_identity = id(model._rows)
        row_identities = tuple(id(row) for row in model._rows)

        readings = [
            SensorReading("LEFT", scientific["H_THUMB_PROXIMAL"], 0.422, True),
            SensorReading("LEFT", scientific["H_SPREAD_INDEX_MIDDLE"], 0.125, True),
            SensorReading("LEFT", scientific["IMU_PALM"], (0.9821, 0.0322, -0.1731, 0.0618), True),
            SensorReading("LEFT", scientific["H_INDEX_MIDDLE"], None, False),
        ]
        model.update_readings(readings)

        self.assertEqual(model.rowCount(), 20)
        self.assertEqual(model.model_creation_count, 1)
        self.assertEqual(model.update_count, 1)
        self.assertEqual(id(model._rows), rows_identity)
        self.assertEqual(tuple(id(row) for row in model._rows), row_identities)

        thumb = self._index_for(model, "H_THUMB_PROXIMAL")
        self.assertEqual(model.data(thumb, model.NormalizedTextRole), "0.422")
        self.assertEqual(model.data(thumb, model.AngleTextRole), "75.96°")
        self.assertEqual(model.data(thumb, model.ValidRole), True)
        self.assertEqual(model.data(thumb, model.ValidTextRole), "YES")

        spread = self._index_for(model, "H_SPREAD_INDEX_MIDDLE")
        self.assertEqual(model.data(spread, model.NormalizedTextRole), "0.125")
        self.assertEqual(model.data(spread, model.AngleTextRole), "22.50°")

        invalid = self._index_for(model, "H_INDEX_MIDDLE")
        self.assertEqual(model.data(invalid, model.NormalizedTextRole), "—")
        self.assertEqual(model.data(invalid, model.ValidTextRole), "NO")

        imu = self._index_for(model, "IMU_PALM")
        quaternion = model.data(imu, model.QuaternionTextRole)
        self.assertIn("W +0.9821", quaternion)
        self.assertIn("Z +0.0618", quaternion)
        self.assertEqual(model.data(imu, model.ValidTextRole), "YES")

        model.set_selected("H_INDEX_MIDDLE")
        self.assertTrue(model.data(invalid, model.SelectedRole))
        self.assertNotIn("adc", {name.decode().lower() for name in model.roleNames().values()})

        # The controller owns a second persistent model for RIGHT; it uses the
        # same frozen channel order and formatting, with only the track label
        # changing in the source reading.
        right_model = SensorValueModel(presentation)
        right_model.update_readings(
            [
                SensorReading("RIGHT", scientific["H_INDEX_MIDDLE"], 0.314, True),
                SensorReading("RIGHT", scientific["IMU_PALM"], (1.0, 0.0, 0.0, 0.0), True),
            ]
        )
        right_index = self._index_for(right_model, "H_INDEX_MIDDLE")
        self.assertEqual(right_model.data(right_index, right_model.NormalizedTextRole), "0.314")
        self.assertEqual(right_model.data(right_index, right_model.AngleTextRole), "56.52°")
        right_imu = self._index_for(right_model, "IMU_PALM")
        self.assertIn("W +1.0000", right_model.data(right_imu, right_model.QuaternionTextRole))


class SensorQmlStructureTests(unittest.TestCase):
    def test_markers_are_persistent_and_follow_the_loaded_rig(self) -> None:
        stage = (ROOT / "smart_glove_app" / "qml" / "components" / "HandStage.qml").read_text(
            encoding="utf-8"
        )
        badge = (ROOT / "smart_glove_app" / "qml" / "components" / "SensorBadge3D.qml").read_text(
            encoding="utf-8"
        )
        overlay = (ROOT / "smart_glove_app" / "qml" / "components" / "SensorBadgeOverlay.qml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(stage.count("View3D {"), 1)
        self.assertEqual(stage.count("RuntimeLoader {"), 1)
        self.assertEqual(stage.count("RiggedHand {"), 2)
        # The inline RiggedHand component declares 20 persistent instances;
        # the two RiggedHand instances produce 40 runtime marker nodes.
        self.assertEqual(stage.count("SensorBadge3D {"), 20)
        self.assertIn("camera: camera", stage)
        self.assertIn("mapFrom3DScene", stage)
        self.assertIn("registerSensorMarker", stage)
        self.assertIn("mapPositionToScene", badge)
        self.assertIn("mapPositionFromScene", badge)
        self.assertIn('source: "#Cylinder"', badge)
        self.assertNotIn("landmarks_3d", badge.lower())
        self.assertIn("markerNode.handNode", overlay)
        self.assertIn("setSensorHand(root.handSide)", overlay)
        self.assertNotIn("console.log", stage + badge + overlay)

    def test_panel_has_source_values_and_no_fabricated_adc_channels(self) -> None:
        panel = (ROOT / "smart_glove_app" / "qml" / "components" / "SensorPanel.qml").read_text(
            encoding="utf-8"
        )
        main = (ROOT / "smart_glove_app" / "qml" / "Main.qml").read_text(encoding="utf-8")

        self.assertIn("SensorPanel", main)
        self.assertIn("sensorValidity", main)
        self.assertIn("frozen TASK-008 source frame", panel)
        self.assertIn("Derived angle", panel)
        self.assertIn("quaternionText", panel)
        self.assertIn("no ADC channels", panel)
        self.assertNotIn("fabricated ADC", panel)
        self.assertNotIn("accelerometer", panel.lower())
        self.assertNotIn("gyroscope", panel.lower())
        self.assertNotIn("magnetometer", panel.lower())


if __name__ == "__main__":
    unittest.main()
