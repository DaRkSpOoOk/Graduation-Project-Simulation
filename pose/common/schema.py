"""Minimal common pose representation for future extractor outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Landmark2D:
    x: float
    y: float
    visibility: float | None = None


@dataclass(slots=True)
class Landmark3D:
    x: float
    y: float
    z: float
    visibility: float | None = None


@dataclass(slots=True)
class HandPoseFrame:
    frame_index: int
    timestamp_seconds: float
    hand_present: bool
    handedness_label: str | None = None
    handedness_confidence: float | None = None
    detection_confidence: float | None = None
    landmarks_2d: list[Landmark2D] = field(default_factory=list)
    landmarks_3d: list[Landmark3D] = field(default_factory=list)
    wrist_position: Landmark3D | None = None
    mano_params: dict[str, Any] | None = None
    mano_references: dict[str, Any] | None = None
    extractor_metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
