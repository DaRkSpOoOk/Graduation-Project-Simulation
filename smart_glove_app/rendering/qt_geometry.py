"""PySide6 Qt Quick 3D geometry bridge.

Each :class:`QtHandGeometry` instance is created once for its logical track.
Frame playback replaces the interleaved position/normal byte buffer and calls
``update()``; it never creates a new ``Model``, ``View3D``, or geometry object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hand_mesh_state import HandRenderPayload
from .mano_topology import MANO_VERTEX_COUNT, ManoTopology


try:  # Keep all non-GUI rendering contracts importable without PySide6.
    from PySide6.QtGui import QVector3D
    from PySide6.QtQuick3D import QQuick3DGeometry

    QT_QUICK3D_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal CI environments
    QVector3D = None  # type: ignore[assignment,misc]
    QQuick3DGeometry = object  # type: ignore[assignment,misc]
    QT_QUICK3D_AVAILABLE = False


class QtGeometryUnavailableError(RuntimeError):
    """Raised when the optional PySide6 Qt Quick 3D runtime is unavailable."""


if QT_QUICK3D_AVAILABLE:

    class QtHandGeometry(QQuick3DGeometry):
        """A persistent custom geometry provider for one hand."""

        def __init__(self, track: str, topology: ManoTopology | None = None) -> None:
            super().__init__()
            normalized = str(track).upper()
            if normalized not in {"LEFT", "RIGHT"}:
                raise KeyError(f"unknown hand track: {track!r}")
            self.track = normalized
            self.topology = topology
            self.mode = "surface" if topology is not None else "point_cloud"
            self.geometry_creation_count = 1
            self.update_count = 0
            self._vertex_buffer_size = 0
            self._configure_static_geometry()

        def _configure_static_geometry(self) -> None:
            self.clear()
            attribute = QQuick3DGeometry.Attribute
            if self.topology is None:
                self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Points)
                self.setStride(3 * np.dtype(np.float32).itemsize)
                self.addAttribute(attribute.PositionSemantic, 0, attribute.F32Type)
                return

            faces = self.topology.faces_for_track(self.track)
            self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Triangles)
            self.setStride(6 * np.dtype(np.float32).itemsize)
            self.setIndexData(np.ascontiguousarray(faces, dtype=np.uint32).tobytes())
            self.addAttribute(attribute.IndexSemantic, 0, attribute.U32Type)
            self.addAttribute(attribute.PositionSemantic, 0, attribute.F32Type)
            self.addAttribute(attribute.NormalSemantic, 3 * np.dtype(np.float32).itemsize, attribute.F32Type)

        @staticmethod
        def _bounds(vertices: np.ndarray) -> tuple[QVector3D, QVector3D]:
            lower = np.min(vertices, axis=0)
            upper = np.max(vertices, axis=0)
            pad = max(float(np.max(upper - lower)) * 0.015, 1e-3)
            lower = lower - pad
            upper = upper + pad
            return QVector3D(float(lower[0]), float(lower[1]), float(lower[2])), QVector3D(
                float(upper[0]), float(upper[1]), float(upper[2])
            )

        def _upload(self, data: bytes) -> None:
            # Qt 6.6+ exposes a partial update overload.  Keeping the buffer
            # size fixed lets the graphics backend update the existing GPU
            # resource; the fallback is for older PySide6 bindings only.
            if self._vertex_buffer_size == len(data) and self._vertex_buffer_size:
                try:
                    self.setVertexData(0, data)
                except TypeError:  # pragma: no cover - compatibility fallback
                    self.setVertexData(data)
            else:
                self.setVertexData(data)
            self._vertex_buffer_size = len(data)

        def set_payload(self, payload: HandRenderPayload) -> None:
            """Upload one presentation payload on the Qt GUI thread."""

            vertices = np.asarray(payload.vertices, dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"{self.track} payload vertices are invalid: {vertices.shape}")
            if self.mode == "surface":
                if vertices.shape[0] != MANO_VERTEX_COUNT or payload.normals is None:
                    raise ValueError("surface geometry requires 778 vertices and computed normals")
                normals = np.asarray(payload.normals, dtype=np.float32)
                if normals.shape != vertices.shape or not np.isfinite(normals).all():
                    raise ValueError("surface geometry normals are invalid")
                interleaved = np.ascontiguousarray(np.concatenate((vertices, normals), axis=1), dtype=np.float32)
            else:
                interleaved = np.ascontiguousarray(vertices, dtype=np.float32)
            self._upload(interleaved.tobytes())
            lower, upper = self._bounds(vertices)
            self.setBounds(lower, upper)
            self.update()
            self.update_count += 1

        @property
        def scene_object_token(self) -> int:
            """Stable identity useful for diagnostics/tests."""

            return id(self)


else:

    class QtHandGeometry:  # type: ignore[no-redef]
        """Import-safe placeholder when PySide6 is not installed."""

        def __init__(self, track: str, topology: ManoTopology | None = None) -> None:
            raise QtGeometryUnavailableError(
                "PySide6 Qt Quick 3D is required for the desktop application; "
                "install the optional 'gui' dependencies"
            )


def require_qt_quick3d() -> None:
    if not QT_QUICK3D_AVAILABLE:
        raise QtGeometryUnavailableError(
            "PySide6 Qt Quick 3D is not installed; install with "
            "python -m pip install -e .[gui]"
        )


__all__ = [
    "QT_QUICK3D_AVAILABLE",
    "QtGeometryUnavailableError",
    "QtHandGeometry",
    "require_qt_quick3d",
]
