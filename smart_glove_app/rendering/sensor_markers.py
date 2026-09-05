"""Small Qt Quick 3D sensor-marker model for the clean hero scene."""

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


else:

    class SensorMarkerModel:  # type: ignore[no-redef]
        """Import-safe placeholder for non-Qt environments."""

        def __init__(self, sensor_layout: Iterable[Any], parent: Any | None = None) -> None:
            self.sensor_layout = tuple(sensor_layout)

        def update_markers(self, positions: Mapping[str, Any], valid: Mapping[str, bool]) -> None:
            return None


__all__ = ["QT_MARKERS_AVAILABLE", "SensorMarkerModel"]
