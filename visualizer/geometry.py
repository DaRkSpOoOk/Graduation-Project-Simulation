"""Geometry and sensor-marker placement helpers for the renderer."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .contract import SensorSpec

# OpenPose/MANO 21-joint order inherited from TASK-005.  The renderer uses
# tracked landmarks for identity and does not infer identity from screen x.
FINGER_CHAINS: dict[str, tuple[int, int, int, int, int]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

SKELETON_EDGES: tuple[tuple[int, int], ...] = tuple(
    edge
    for chain in FINGER_CHAINS.values()
    for edge in zip(chain[:-1], chain[1:])
)


def _finite_point(point: np.ndarray | None) -> bool:
    return point is not None and np.asarray(point).shape == (3,) and bool(np.isfinite(point).all())


def sensor_marker_positions(
    landmarks_3d: np.ndarray | None,
    sensor_layout: Iterable[SensorSpec],
) -> dict[str, np.ndarray | None]:
    """Place every logical sensor near its frozen landmark-defined location.

    Bend markers sit on the corresponding chain joint.  Spread markers are
    between adjacent finger bases and slightly toward the palm centre to
    approximate the interdigital web.  The IMU sits at the centre of the four
    palm landmarks.  These are 3D landmark-derived locations, not screen
    coordinates and not new anatomical claims.
    """

    positions: dict[str, np.ndarray | None] = {}
    if landmarks_3d is None:
        return {sensor.sensor_id: None for sensor in sensor_layout}
    points = np.asarray(landmarks_3d, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {points.shape}")

    palm_indices = (0, 5, 9, 17)
    palm_points = points[list(palm_indices)]
    palm_center = palm_points.mean(axis=0) if np.isfinite(palm_points).all() else None
    for sensor in sensor_layout:
        if sensor.role == "bend":
            assert sensor.finger is not None and sensor.joint is not None
            joint_index = FINGER_CHAINS[sensor.finger][("proximal", "middle", "distal").index(sensor.joint) + 1]
            point = points[joint_index]
            positions[sensor.sensor_id] = point.copy() if _finite_point(point) else None
        elif sensor.role == "spread":
            assert sensor.pair is not None
            first_base = FINGER_CHAINS[sensor.pair[0]][1]
            second_base = FINGER_CHAINS[sensor.pair[1]][1]
            first = points[first_base]
            second = points[second_base]
            if _finite_point(first) and _finite_point(second) and palm_center is not None:
                positions[sensor.sensor_id] = (0.70 * ((first + second) / 2.0) + 0.30 * palm_center).astype(float)
            else:
                positions[sensor.sensor_id] = None
        else:
            positions[sensor.sensor_id] = palm_center.copy() if palm_center is not None else None
    return positions


def sequence_bounds(sequence: object) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed finite 3D bounds for one loaded sequence."""

    all_points: list[np.ndarray] = []
    for frame in sequence.frames:  # type: ignore[attr-defined]
        for hand in frame.hands:
            for points in (hand.mesh_vertices, hand.landmarks_3d):
                if points is None:
                    continue
                array = np.asarray(points, dtype=float)
                finite = array[np.isfinite(array).all(axis=-1)]
                if finite.size:
                    all_points.append(finite.reshape(-1, 3))
    if not all_points:
        raise ValueError("sequence has no finite 3D geometry")
    points = np.concatenate(all_points, axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = float(np.max(upper - lower))
    pad = max(span * 0.08, 1e-3)
    return lower - pad, upper + pad
