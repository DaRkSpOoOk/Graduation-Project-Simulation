"""Persistent Qt models for the sensor overlay and sensor-value panel.

The 3D marker nodes themselves live in ``HandStage.qml`` so they can follow
the final RuntimeLoader skeleton.  These models carry only panel state and
are updated in place; they never own or recreate a rendered marker object.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


try:  # Optional so pure rendering tests do not require a GUI runtime.
    from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
    from PySide6.QtGui import QVector3D

    QT_MARKERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal CI environments
    QAbstractListModel = object  # type: ignore[assignment,misc]
    QModelIndex = object  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment]
    QVector3D = None  # type: ignore[assignment,misc]
    QT_MARKERS_AVAILABLE = False


if QT_MARKERS_AVAILABLE:

    class SensorMarkerModel(QAbstractListModel):
        """Twenty stable marker rows whose positions update in place."""

        PositionRole = Qt.UserRole + 1
        ValidRole = Qt.UserRole + 2
        ActiveRole = Qt.UserRole + 3
        LabelRole = Qt.UserRole + 4

        def __init__(self, sensor_layout: Iterable[Any], parent: Any | None = None) -> None:
            super().__init__(parent)
            self._layout = tuple(sensor_layout)
            self._rows: list[dict[str, Any]] = []
            self.model_creation_count = 1
            self.update_count = 0
            self._replace_rows({}, {})

        def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API name
            return {
                self.PositionRole: b"position",
                self.ValidRole: b"valid",
                self.ActiveRole: b"active",
                self.LabelRole: b"label",
            }

        def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API name
            if parent.isValid():
                return 0
            return len(self._rows)

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
            if not index.isValid() or not 0 <= index.row() < len(self._rows):
                return None
            row = self._rows[index.row()]
            if role == self.PositionRole:
                return row["position"]
            if role == self.ValidRole:
                return row["valid"]
            if role == self.ActiveRole:
                return row["active"]
            if role in {self.LabelRole, Qt.DisplayRole}:
                return row["label"]
            return None

        def _build_rows(
            self,
            positions: Mapping[str, Any],
            valid: Mapping[str, bool],
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for sensor in self._layout:
                sensor_id = str(sensor.sensor_id)
                value = positions.get(sensor_id)
                point = None
                if value is not None:
                    candidate = tuple(float(item) for item in value)
                    if len(candidate) == 3:
                        point = QVector3D(*candidate)
                rows.append(
                    {
                        "position": point or QVector3D(0.0, 0.0, 0.0),
                        "valid": bool(valid.get(sensor_id, False)),
                        "active": point is not None,
                        "label": str(sensor.display_marker),
                    }
                )
            return rows

        def _replace_rows(self, positions: Mapping[str, Any], valid: Mapping[str, bool]) -> None:
            rows = self._build_rows(positions, valid)
            if not self._rows:
                self._rows.extend(rows)
                return
            for current, replacement in zip(self._rows, rows):
                current.update(replacement)

        def update_markers(
            self,
            positions: Mapping[str, Any],
            valid: Mapping[str, bool],
        ) -> None:
            """Update marker roles without resetting rows or QML delegates."""

            self._replace_rows(positions, valid)
            self.update_count += 1
            if self._rows:
                self.dataChanged.emit(
                    self.index(0, 0),
                    self.index(len(self._rows) - 1, 0),
                    [self.PositionRole, self.ValidRole, self.ActiveRole, self.LabelRole],
                )


    class SensorValueModel(QAbstractListModel):
        """Twenty stable source-reading rows for one physical hand."""

        SensorIdRole = Qt.UserRole + 20
        DisplayNameRole = Qt.UserRole + 21
        ShortIdRole = Qt.UserRole + 22
        GroupRole = Qt.UserRole + 23
        SensorTypeRole = Qt.UserRole + 24
        MarkerRole = Qt.UserRole + 25
        NormalizedTextRole = Qt.UserRole + 26
        AngleTextRole = Qt.UserRole + 27
        QuaternionTextRole = Qt.UserRole + 28
        ValidRole = Qt.UserRole + 29
        ValidTextRole = Qt.UserRole + 30
        HasAngleRole = Qt.UserRole + 31
        SelectedRole = Qt.UserRole + 32
        DescriptionRole = Qt.UserRole + 33

        def __init__(self, sensor_layout: Iterable[Any], parent: Any | None = None) -> None:
            super().__init__(parent)
            self._layout = tuple(sensor_layout)
            self._rows: list[dict[str, Any]] = [self._empty_row(spec) for spec in self._layout]
            self.model_creation_count = 1
            self.update_count = 0

        @staticmethod
        def _empty_row(spec: Any) -> dict[str, Any]:
            return {
                "sensorId": str(spec.sensor_id),
                "displayName": str(spec.display_name),
                "shortId": str(spec.short_id),
                "group": str(spec.group),
                "sensorType": str(spec.sensor_type),
                "marker": str(spec.marker),
                "normalizedText": "—",
                "angleText": "—",
                "quaternionText": "—",
                "valid": False,
                "validText": "NO",
                "hasAngle": bool(spec.has_angle),
                "selected": False,
                "description": str(spec.description),
            }

        def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API name
            return {
                self.SensorIdRole: b"sensorId",
                self.DisplayNameRole: b"displayName",
                self.ShortIdRole: b"shortId",
                self.GroupRole: b"group",
                self.SensorTypeRole: b"sensorType",
                self.MarkerRole: b"marker",
                self.NormalizedTextRole: b"normalizedText",
                self.AngleTextRole: b"angleText",
                self.QuaternionTextRole: b"quaternionText",
                self.ValidRole: b"valid",
                self.ValidTextRole: b"validText",
                self.HasAngleRole: b"hasAngle",
                self.SelectedRole: b"selected",
                self.DescriptionRole: b"description",
            }

        def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
            if parent.isValid():
                return 0
            return len(self._rows)

        def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
            if not index.isValid() or not 0 <= index.row() < len(self._rows):
                return None
            row = self._rows[index.row()]
            if role == Qt.DisplayRole:
                return row["displayName"]
            role_to_key = {
                self.SensorIdRole: "sensorId",
                self.DisplayNameRole: "displayName",
                self.ShortIdRole: "shortId",
                self.GroupRole: "group",
                self.SensorTypeRole: "sensorType",
                self.MarkerRole: "marker",
                self.NormalizedTextRole: "normalizedText",
                self.AngleTextRole: "angleText",
                self.QuaternionTextRole: "quaternionText",
                self.ValidRole: "valid",
                self.ValidTextRole: "validText",
                self.HasAngleRole: "hasAngle",
                self.SelectedRole: "selected",
                self.DescriptionRole: "description",
            }
            key = role_to_key.get(role)
            return row.get(key) if key is not None else None

        def _emit_all(self) -> None:
            if self._rows:
                self.dataChanged.emit(
                    self.index(0, 0),
                    self.index(len(self._rows) - 1, 0),
                    list(self.roleNames()),
                )

        def update_readings(self, readings: Iterable[Any]) -> None:
            """Publish one exact source-frame reading set without resetRows."""

            by_id = {str(reading.sensor.sensor_id): reading for reading in readings}
            for spec, row in zip(self._layout, self._rows):
                reading = by_id.get(spec.sensor_id)
                valid = bool(reading is not None and reading.valid)
                row["valid"] = valid
                row["validText"] = "YES" if valid else "NO"
                row["normalizedText"] = "—"
                row["angleText"] = "—"
                row["quaternionText"] = "—"
                if not valid or reading is None or reading.value is None:
                    continue
                if spec.has_angle:
                    normalized = float(reading.value)
                    row["normalizedText"] = f"{normalized:.3f}"
                    row["angleText"] = f"{normalized * 180.0:.2f}°"
                else:
                    values = tuple(float(item) for item in reading.value)
                    if len(values) == 4:
                        row["quaternionText"] = (
                            f"W {values[0]:+.4f}   X {values[1]:+.4f}\n"
                            f"Y {values[2]:+.4f}   Z {values[3]:+.4f}"
                        )
            self.update_count += 1
            self._emit_all()

        def clear_values(self) -> None:
            """Show no source frame while retaining the persistent rows."""

            for spec, row in zip(self._layout, self._rows):
                replacement = self._empty_row(spec)
                replacement["selected"] = row["selected"]
                row.update(replacement)
            self._emit_all()

        def set_selected(self, sensor_id: str) -> None:
            selected = str(sensor_id)
            changed = False
            for row in self._rows:
                value = row["sensorId"] == selected
                if row["selected"] != value:
                    row["selected"] = value
                    changed = True
            if changed:
                self._emit_all()


else:

    class SensorMarkerModel:  # type: ignore[no-redef]
        """Import-safe placeholder for non-Qt environments."""

        def __init__(self, sensor_layout: Iterable[Any], parent: Any | None = None) -> None:
            self.sensor_layout = tuple(sensor_layout)

        def update_markers(self, positions: Mapping[str, Any], valid: Mapping[str, bool]) -> None:
            return None


    class SensorValueModel:  # type: ignore[no-redef]
        """Import-safe placeholder for non-Qt environments."""

        def __init__(self, sensor_layout: Iterable[Any], parent: Any | None = None) -> None:
            self.sensor_layout = tuple(sensor_layout)
            self._rows = []
            self.model_creation_count = 1
            self.update_count = 0

        def update_readings(self, readings: Iterable[Any]) -> None:
            self.update_count += 1

        def clear_values(self) -> None:
            return None

        def set_selected(self, sensor_id: str) -> None:
            return None


__all__ = ["QT_MARKERS_AVAILABLE", "SensorMarkerModel", "SensorValueModel"]
