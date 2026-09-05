"""Presentation-only hand state and interpolation for TASK-007F.

This module is deliberately independent of Qt and of the recognition input
contract.  It turns one frozen :class:`visualizer.contract.PlaybackSequence`
frame into display vertices, normals, marker positions, and an explicit
missing/idle state.  The returned arrays are never written back to the
scientific sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from visualizer.geometry import sensor_marker_positions

from .mano_topology import MANO_VERTEX_COUNT, ManoTopology


TRACKS = ("LEFT", "RIGHT")
DISPLAY_HAND_HEIGHT = 3.45


class PresentationGeometryError(ValueError):
    """Raised when stored geometry cannot be represented by the renderer."""


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    value.setflags(write=False)
    return value


def _finite_points(value: Any, *, expected_vertices: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        return None
    if expected_vertices is not None and array.shape[0] != expected_vertices:
        raise PresentationGeometryError(
            f"stored MANO vertices must have shape ({expected_vertices}, 3), got {array.shape}"
        )
    if not np.isfinite(array).all():
        return None
    return np.ascontiguousarray(array)


def _fallback_hand_points(vertex_count: int = MANO_VERTEX_COUNT, *, mirror: bool = False) -> np.ndarray:
    """Create a deterministic presentation-only point hand when no asset exists.

    This is used only for the startup placeholder.  It is never presented to
    the recognizer and is never serialized as TASK-008 data.
    """

    if vertex_count <= 0:
        raise ValueError("vertex_count must be positive")
    skeleton = np.asarray(
        [
            (0.00, -1.28, 0.00),
            (-0.42, -0.93, 0.02),
            (-0.58, -0.52, 0.00),
            (-0.65, -0.08, -0.02),
            (-0.65, 0.35, -0.01),
            (-0.23, -0.96, 0.02),
            (-0.24, -0.42, 0.00),
            (-0.24, 0.16, -0.01),
            (-0.24, 0.73, -0.01),
            (0.00, -0.98, 0.01),
            (0.00, -0.37, 0.00),
            (0.00, 0.27, -0.01),
            (0.00, 0.88, -0.01),
            (0.24, -0.98, 0.01),
            (0.24, -0.41, 0.00),
            (0.24, 0.17, -0.01),
            (0.24, 0.72, -0.01),
            (0.45, -0.93, 0.02),
            (0.49, -0.47, 0.00),
            (0.49, -0.02, -0.01),
            (0.47, 0.48, -0.01),
        ],
        dtype=np.float32,
    )
    edges = (
        (skeleton[0], skeleton[1]),
        (skeleton[1], skeleton[2]),
        (skeleton[2], skeleton[3]),
        (skeleton[3], skeleton[4]),
        (skeleton[0], skeleton[5]),
        (skeleton[5], skeleton[6]),
        (skeleton[6], skeleton[7]),
        (skeleton[7], skeleton[8]),
        (skeleton[0], skeleton[9]),
        (skeleton[9], skeleton[10]),
        (skeleton[10], skeleton[11]),
        (skeleton[11], skeleton[12]),
        (skeleton[0], skeleton[13]),
        (skeleton[13], skeleton[14]),
        (skeleton[14], skeleton[15]),
        (skeleton[15], skeleton[16]),
        (skeleton[0], skeleton[17]),
        (skeleton[17], skeleton[18]),
        (skeleton[18], skeleton[19]),
        (skeleton[19], skeleton[20]),
    )
    points = np.empty((vertex_count, 3), dtype=np.float32)
    for index in range(vertex_count):
        first, second = edges[index % len(edges)]
        cycle = index // len(edges)
        t = (cycle % 37) / 36.0
        center = (1.0 - t) * first + t * second
        spread = 0.012 * ((index % 11) - 5)
        points[index] = center + np.asarray((spread, -spread * 0.25, spread * 0.35), dtype=np.float32)
    if mirror:
        points[:, 0] *= -1.0
    return _readonly(points)


def compute_vertex_normals(vertices: Any, faces: Any) -> np.ndarray:
    """Compute area-weighted vertex normals for one finite triangle mesh."""

    points = _finite_points(vertices)
    if points is None:
        raise PresentationGeometryError("cannot compute normals for non-finite vertices")
    triangle_indices = np.asarray(faces, dtype=np.int64)
    if triangle_indices.ndim != 2 or triangle_indices.shape[1] != 3:
        raise PresentationGeometryError(f"faces must have shape (F, 3), got {triangle_indices.shape}")
    if triangle_indices.size and (
        int(triangle_indices.min()) < 0 or int(triangle_indices.max()) >= len(points)
    ):
        raise PresentationGeometryError("face index is outside the supplied vertex array")
    first = points[triangle_indices[:, 0]]
    second = points[triangle_indices[:, 1]]
    third = points[triangle_indices[:, 2]]
    face_normals = np.cross(second - first, third - first)
    normals = np.zeros_like(points, dtype=np.float32)
    for column in range(3):
        np.add.at(normals, triangle_indices[:, column], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    normals[valid] /= lengths[valid]
    normals[~valid] = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    return _readonly(normals)


@dataclass(frozen=True, slots=True)
class HandRenderPayload:
    """One frame of display state for one persistent hand geometry object."""

    track: str
    vertices: np.ndarray
    normals: np.ndarray | None
    state: str
    source: str
    dimmed: bool
    visible: bool
    marker_positions: Mapping[str, np.ndarray | None]
    marker_valid: Mapping[str, bool]
    frame_position: int | None
    interpolated: bool

    @property
    def surface(self) -> bool:
        return self.normals is not None


@dataclass(frozen=True, slots=True)
class PresentationFrame:
    """Both hand payloads plus exact source-frame provenance for the display."""

    position: int | None
    frame_index: int | None
    timestamp_seconds: float | None
    left: HandRenderPayload
    right: HandRenderPayload
    interpolated: bool
    source_positions: tuple[int, ...]

    def hand(self, track: str) -> HandRenderPayload:
        normalized = str(track).upper()
        if normalized == "LEFT":
            return self.left
        if normalized == "RIGHT":
            return self.right
        raise KeyError(f"unknown hand track: {track!r}")


class HandMeshState:
    """Persistent presentation state for one logical hand track."""

    def __init__(
        self,
        track: str,
        *,
        topology: ManoTopology | None = None,
        neutral_vertices: Any | None = None,
    ) -> None:
        normalized = str(track).upper()
        if normalized not in TRACKS:
            raise KeyError(f"unknown hand track: {track!r}")
        self.track = normalized
        self.topology = topology
        self.mode = "surface" if topology is not None else "point_cloud"
        supplied_neutral = _finite_points(neutral_vertices, expected_vertices=MANO_VERTEX_COUNT)
        if supplied_neutral is None and topology is not None:
            supplied_neutral = topology.neutral_for_track(normalized)
        self._neutral_vertices = (
            _readonly(supplied_neutral)
            if supplied_neutral is not None
            else _fallback_hand_points(MANO_VERTEX_COUNT, mirror=normalized == "LEFT")
        )
        self.scene_object_token = object()
        self.geometry_creation_count = 1
        self.update_count = 0
        self._last_visible_vertices: np.ndarray | None = None
        self._last_marker_positions: dict[str, np.ndarray] = {}
        self._reference_span: float | None = None
        self._last_anchor = np.zeros(3, dtype=np.float32)
        self._last_scale = 1.0
        self._last_y_center = 0.0
        self._payload = self._make_payload(
            self._neutral_vertices,
            state="IDLE",
            source="neutral-template",
            dimmed=False,
            marker_positions={},
            marker_valid={},
            frame_position=None,
            interpolated=False,
        )

    @property
    def payload(self) -> HandRenderPayload:
        return self._payload

    @property
    def neutral_vertices(self) -> np.ndarray:
        return self._neutral_vertices

    def _reference_for(self, points: np.ndarray) -> float:
        span = float(np.max(np.ptp(points, axis=0)))
        if not np.isfinite(span) or span <= 1e-6:
            span = 1.0
        if self._reference_span is None:
            self._reference_span = span
        return self._reference_span

    def _transform_points(self, points: np.ndarray, *, anchor: np.ndarray | None = None) -> np.ndarray:
        reference = self._reference_for(points)
        selected_anchor = np.asarray(anchor if anchor is not None else points.mean(axis=0), dtype=np.float32)
        if selected_anchor.shape != (3,) or not np.isfinite(selected_anchor).all():
            selected_anchor = points.mean(axis=0).astype(np.float32)
        scale = DISPLAY_HAND_HEIGHT / max(reference, 1e-6)
        transformed = (points - selected_anchor) * scale
        y_center = float((np.min(transformed[:, 1]) + np.max(transformed[:, 1])) / 2.0)
        transformed[:, 1] -= y_center
        self._last_anchor = selected_anchor.copy()
        self._last_scale = float(scale)
        self._last_y_center = y_center
        return _readonly(transformed)

    def _transform_markers(
        self,
        marker_positions: Mapping[str, np.ndarray | None],
    ) -> dict[str, np.ndarray | None]:
        result: dict[str, np.ndarray | None] = {}
        for sensor_id, value in marker_positions.items():
            if value is None:
                result[sensor_id] = None
                continue
            point = np.asarray(value, dtype=np.float32)
            if point.shape != (3,) or not np.isfinite(point).all():
                result[sensor_id] = None
                continue
            transformed = (point - self._last_anchor) * self._last_scale
            transformed[1] -= self._last_y_center
            result[sensor_id] = _readonly(transformed)
        return result

    def _make_payload(
        self,
        raw_vertices: np.ndarray,
        *,
        state: str,
        source: str,
        dimmed: bool,
        marker_positions: Mapping[str, np.ndarray | None],
        marker_valid: Mapping[str, bool],
        frame_position: int | None,
        interpolated: bool,
    ) -> HandRenderPayload:
        points = _finite_points(raw_vertices)
        if points is None:
            raise PresentationGeometryError("presentation vertices must be finite")
        anchor = points.mean(axis=0)
        display_vertices = self._transform_points(points, anchor=anchor)
        normals = None
        if self.mode == "surface":
            if len(display_vertices) != MANO_VERTEX_COUNT or self.topology is None:
                raise PresentationGeometryError("surface mode requires exactly 778 vertices and MANO topology")
            normals = compute_vertex_normals(display_vertices, self.topology.faces_for_track(self.track))
        return HandRenderPayload(
            track=self.track,
            vertices=display_vertices,
            normals=normals,
            state=str(state).upper(),
            source=source,
            dimmed=bool(dimmed),
            visible=True,
            marker_positions=self._transform_markers(marker_positions),
            marker_valid={str(key): bool(value) for key, value in marker_valid.items()},
            frame_position=frame_position,
            interpolated=bool(interpolated),
        )

    def reset_to_idle(self) -> HandRenderPayload:
        self._last_visible_vertices = None
        self._last_marker_positions.clear()
        self._reference_span = None
        self._payload = self._make_payload(
            self._neutral_vertices,
            state="IDLE",
            source="neutral-template",
            dimmed=False,
            marker_positions={},
            marker_valid={},
            frame_position=None,
            interpolated=False,
        )
        return self._payload

    def _usable_mesh(self, hand: Any) -> np.ndarray | None:
        value = getattr(hand, "mesh_vertices", None)
        return _finite_points(value, expected_vertices=MANO_VERTEX_COUNT)

    def update(
        self,
        hand: Any,
        *,
        marker_positions: Mapping[str, np.ndarray | None] | None = None,
        marker_valid: Mapping[str, bool] | None = None,
        next_hand: Any | None = None,
        interpolation_alpha: float = 0.0,
        smooth: bool = False,
        frame_position: int | None = None,
    ) -> HandRenderPayload:
        """Update this persistent state from real data for presentation only."""

        current_mesh = self._usable_mesh(hand)
        next_mesh = self._usable_mesh(next_hand) if next_hand is not None else None
        alpha = min(1.0, max(0.0, float(interpolation_alpha)))
        interpolated = bool(smooth and alpha > 0.0 and alpha < 1.0 and current_mesh is not None and next_mesh is not None)
        if interpolated:
            selected_mesh = ((1.0 - alpha) * current_mesh + alpha * next_mesh).astype(np.float32)
        else:
            selected_mesh = current_mesh

        current_state = str(getattr(hand, "state", "MISSING")).upper()
        valid_observation = current_mesh is not None and current_state not in {
            "MISSING",
            "LIKELY_OCCLUDED",
            "REJECTED_QUALITY",
        }
        if valid_observation and selected_mesh is not None:
            self._last_visible_vertices = _readonly(selected_mesh)
            source = "interpolated" if interpolated else "stored-mano"
            state = current_state
            dimmed = False
            render_vertices = selected_mesh
        else:
            if self._last_visible_vertices is not None:
                render_vertices = self._last_visible_vertices
                source = "last-visible-pose"
            else:
                render_vertices = self._neutral_vertices
                source = "neutral-template"
            state = "MISSING" if current_state != "IDLE" else "IDLE"
            dimmed = state != "IDLE"

        raw_markers = dict(marker_positions or {})
        for sensor_id, value in raw_markers.items():
            if value is not None:
                point = np.asarray(value, dtype=np.float32)
                if point.shape == (3,) and np.isfinite(point).all():
                    self._last_marker_positions[sensor_id] = point.copy()
        # Keep marker locations stable through an unobserved frame, but let
        # the validity map turn them into dim presentation markers.  This is
        # visual persistence only; no values are written to the source frame.
        for sensor_id, value in list(raw_markers.items()):
            if value is None and sensor_id in self._last_marker_positions:
                raw_markers[sensor_id] = self._last_marker_positions[sensor_id]
        if not raw_markers and self._last_marker_positions:
            raw_markers = dict(self._last_marker_positions)
        self._payload = self._make_payload(
            np.asarray(render_vertices, dtype=np.float32),
            state=state,
            source=source,
            dimmed=dimmed,
            marker_positions=raw_markers,
            marker_valid=marker_valid or {},
            frame_position=frame_position,
            interpolated=interpolated,
        )
        self.update_count += 1
        return self._payload


class PersistentRenderScene:
    """Own two hand states once and update their payloads without recreation."""

    def __init__(self, topology: ManoTopology | None = None) -> None:
        self.topology = topology
        self.left = HandMeshState("LEFT", topology=topology)
        self.right = HandMeshState("RIGHT", topology=topology)
        self.scene_creation_count = 1
        self.view3d_creation_count = 1
        self._sequence: Any | None = None

    @property
    def hand_tracks(self) -> tuple[str, str]:
        return TRACKS

    @property
    def geometry_tokens(self) -> tuple[object, object]:
        return (self.left.scene_object_token, self.right.scene_object_token)

    @property
    def topology_available(self) -> bool:
        return self.topology is not None

    @property
    def topology_status(self) -> str:
        if self.topology is None:
            return "SURFACE TOPOLOGY UNAVAILABLE — POINT-CLOUD FALLBACK"
        return "MANO SURFACE · 778 VERTICES"

    def attach_sequence(self, sequence: Any) -> None:
        self._sequence = sequence
        self.left.reset_to_idle()
        self.right.reset_to_idle()

    def clear_sequence(self) -> PresentationFrame:
        self._sequence = None
        left = self.left.reset_to_idle()
        right = self.right.reset_to_idle()
        return PresentationFrame(None, None, None, left, right, False, ())

    def _update_hand(self, state: HandMeshState, frame: Any, next_frame: Any | None, alpha: float, smooth: bool) -> HandRenderPayload:
        hand = frame.hand(state.track)
        next_hand = next_frame.hand(state.track) if next_frame is not None else None
        readings = self._sequence.sensor_readings(frame.position, state.track) if self._sequence is not None else ()
        valid_by_sensor = {reading.sensor.sensor_id: reading.valid for reading in readings}
        if str(getattr(hand, "state", "MISSING")).upper() in {
            "MISSING",
            "LIKELY_OCCLUDED",
            "REJECTED_QUALITY",
        }:
            valid_by_sensor = {sensor_id: False for sensor_id in valid_by_sensor}
        positions = sensor_marker_positions(hand.landmarks_3d, self._sequence.sensor_layout) if self._sequence is not None else {}
        return state.update(
            hand,
            marker_positions=positions,
            marker_valid=valid_by_sensor,
            next_hand=next_hand,
            interpolation_alpha=alpha,
            smooth=smooth,
            frame_position=frame.position,
        )

    def update_sequence_frame(
        self,
        position: int,
        *,
        interpolation_alpha: float = 0.0,
        smooth: bool = False,
    ) -> PresentationFrame:
        if self._sequence is None:
            raise PresentationGeometryError("no sequence is attached to the persistent scene")
        frame = self._sequence.frame_at(int(position))
        next_frame = self._sequence.frames[position + 1] if position + 1 < len(self._sequence.frames) else None
        alpha = float(interpolation_alpha) if smooth else 0.0
        left = self._update_hand(self.left, frame, next_frame, alpha, smooth)
        right = self._update_hand(self.right, frame, next_frame, alpha, smooth)
        return PresentationFrame(
            position=frame.position,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            left=left,
            right=right,
            interpolated=left.interpolated or right.interpolated,
            source_positions=(frame.position, frame.position + 1) if left.interpolated or right.interpolated else (frame.position,),
        )


__all__ = [
    "DISPLAY_HAND_HEIGHT",
    "HandMeshState",
    "HandRenderPayload",
    "PersistentRenderScene",
    "PresentationFrame",
    "PresentationGeometryError",
    "TRACKS",
    "compute_vertex_normals",
]
