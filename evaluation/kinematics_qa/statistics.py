"""Distribution and temporal diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

HAND_NAMES = ("LEFT", "RIGHT")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
JOINT_NAMES = ("MCP", "PIP", "DIP")
SPREAD_NAMES = (
    "thumb_index",
    "index_middle",
    "middle_ring",
    "ring_pinky",
)


def distribution(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(vector.size),
        "min": float(np.min(vector)),
        "p1": float(np.percentile(vector, 1)),
        "p50": float(np.percentile(vector, 50)),
        "p95": float(np.percentile(vector, 95)),
        "p99": float(np.percentile(vector, 99)),
        "max": float(np.max(vector)),
    }
