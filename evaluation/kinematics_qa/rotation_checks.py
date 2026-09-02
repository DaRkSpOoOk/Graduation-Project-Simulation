"""Rotation and quaternion consistency checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_orthogonality_error(matrix: np.ndarray) -> float:
    identity = np.eye(3, dtype=np.float64)
    delta = matrix.T @ matrix - identity
    return float(np.max(np.abs(delta)))


def rotation_delta_degrees(left: np.ndarray, right: np.ndarray) -> float:
    rel = left.T @ right
    trace = float(np.trace(rel))
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(math.degrees(math.acos(cosine)))


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quaternion]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def percentile_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p95": None, "p99": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }
