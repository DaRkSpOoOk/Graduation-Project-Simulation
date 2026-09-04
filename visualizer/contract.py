"""Renderer-facing data contract for TASK-007A.

This module contains no model code and no Arabic-label resolution logic.  It
only describes one already-produced TASK-008 sequence in a form that a
renderer can safely play back frame by frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

TRACK_ORDER: tuple[str, str] = ("LEFT", "RIGHT")
FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
CHAIN_ORDER: tuple[str, str, str] = ("proximal", "middle", "distal")
SPREAD_PAIRS: tuple[tuple[str, str], ...] = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)

EXPECTED_BEND_SENSORS = 15
EXPECTED_SPREAD_SENSORS = 4
EXPECTED_HALL_SENSORS = 19
EXPECTED_IMU_SENSORS = 1
EXPECTED_SENSOR_PACKAGES = 20


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """One physical sensor package from the frozen TASK-006 layout."""

    sensor_id: str
    sensor_type: str
    finger: str | None
    pair: tuple[str, str] | None
    joint: str | None
    role: str
    logical_location: str
    display_marker: str
    description: str
    array: str
    array_index: tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorSpec":
        pair = payload.get("pair")
        normalized_pair = tuple(str(item) for item in pair) if pair is not None else None
        return cls(
            sensor_id=str(payload.get("sensor_id", "")),
            sensor_type=str(payload.get("sensor_type", "")),
            finger=(str(payload["finger"]) if payload.get("finger") is not None else None),
            pair=normalized_pair,  # type: ignore[arg-type]
            joint=(str(payload["joint"]) if payload.get("joint") is not None else None),
            role=str(payload.get("role", "")),
            logical_location=str(payload.get("logical_location", "")),
            display_marker=str(payload.get("display_marker", "")),
            description=str(payload.get("description", "")),
            array=str(payload.get("array", "")),
            array_index=tuple(int(item) for item in _as_tuple(payload.get("array_index"))),
        )


def validate_sensor_layout(payload: Mapping[str, Any]) -> tuple[SensorSpec, ...]:
    """Validate and return the exact 20-entry TASK-006 layout.

    The validation is intentionally strict: a visualization should fail early
    rather than silently drawing a marker against the wrong array slot.
    """

    raw_sensors = payload.get("sensors")
    if not isinstance(raw_sensors, list):
        raise ValueError("sensor layout must contain a sensors list")
    if len(raw_sensors) != EXPECTED_SENSOR_PACKAGES:
        raise ValueError(
            f"sensor layout must contain {EXPECTED_SENSOR_PACKAGES} entries, got {len(raw_sensors)}"
        )
    if payload.get("layout_version") != "ideal_virtual_glove_v1":
        raise ValueError("sensor layout version must be ideal_virtual_glove_v1")

    track_order = tuple(str(value).upper() for value in _as_tuple(payload.get("track_order")))
    if track_order != TRACK_ORDER:
        raise ValueError(f"sensor layout track order must be {TRACK_ORDER}, got {track_order}")
    if tuple(_as_tuple(payload.get("finger_order"))) != FINGER_ORDER:
        raise ValueError("sensor layout finger order does not match TASK-006")
    if tuple(_as_tuple(payload.get("chain_joint_order"))) != CHAIN_ORDER:
        raise ValueError("sensor layout chain order does not match TASK-006")
    spread_pairs = tuple(tuple(str(item) for item in pair) for pair in _as_tuple(payload.get("spread_pairs")))
    if spread_pairs != SPREAD_PAIRS:
        raise ValueError("sensor layout spread-pair order does not match TASK-006")

    markers = payload.get("display_markers", {})
    if not isinstance(markers, Mapping) or markers.get("hall") != "H" or markers.get("imu") != "IMU":
        raise ValueError("sensor layout display markers must be hall=H and imu=IMU")

    sensors = tuple(SensorSpec.from_dict(value) for value in raw_sensors)
    ids = [sensor.sensor_id for sensor in sensors]
    if any(not sensor_id for sensor_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("sensor layout sensor IDs must be non-empty and unique")
    expected_ids = tuple(
        f"H_{finger.upper()}_{joint.upper()}"
        for finger in FINGER_ORDER
        for joint in CHAIN_ORDER
    ) + tuple(f"H_SPREAD_{first.upper()}_{second.upper()}" for first, second in SPREAD_PAIRS) + ("IMU_PALM",)
    if tuple(ids) != expected_ids:
        raise ValueError("sensor layout sensor IDs/order do not match TASK-006")

    bend = tuple(sensor for sensor in sensors if sensor.role == "bend")
    spread = tuple(sensor for sensor in sensors if sensor.role == "spread")
    imu = tuple(sensor for sensor in sensors if sensor.role == "palm_orientation")
    if (len(bend), len(spread), len(imu)) != (
        EXPECTED_BEND_SENSORS,
        EXPECTED_SPREAD_SENSORS,
        EXPECTED_IMU_SENSORS,
    ):
        raise ValueError("sensor layout role counts are not 15 bend, 4 spread, 1 IMU")

    bend_slots = set()
    for sensor in bend:
        if sensor.display_marker != "H" or sensor.array != "bend_angle_deg":
            raise ValueError(f"invalid bend sensor contract: {sensor.sensor_id}")
        if sensor.finger not in FINGER_ORDER or sensor.joint not in CHAIN_ORDER:
            raise ValueError(f"invalid bend sensor location: {sensor.sensor_id}")
        if len(sensor.array_index) != 2:
            raise ValueError(f"invalid bend array index: {sensor.sensor_id}")
        slot = tuple(sensor.array_index)
        if slot != (FINGER_ORDER.index(sensor.finger), CHAIN_ORDER.index(sensor.joint)):
            raise ValueError(f"bend sensor array mapping is not canonical: {sensor.sensor_id}")
        bend_slots.add(slot)
    if len(bend_slots) != EXPECTED_BEND_SENSORS:
        raise ValueError("bend sensor array slots are not one-to-one")

    spread_slots = set()
    expected_pairs = set(SPREAD_PAIRS)
    for sensor in spread:
        if sensor.display_marker != "H" or sensor.array != "spread_angle_deg":
            raise ValueError(f"invalid spread sensor contract: {sensor.sensor_id}")
        if sensor.pair is None or sensor.pair not in expected_pairs:
            raise ValueError(f"invalid spread pair: {sensor.sensor_id}")
        if len(sensor.array_index) != 1 or sensor.array_index[0] != SPREAD_PAIRS.index(sensor.pair):
            raise ValueError(f"spread sensor array mapping is not canonical: {sensor.sensor_id}")
        spread_slots.add(sensor.array_index[0])
    if spread_slots != set(range(EXPECTED_SPREAD_SENSORS)):
        raise ValueError("spread sensor array slots are not one-to-one")

    if len(imu) != 1:
        raise ValueError("sensor layout must contain exactly one palm IMU")
    if imu[0].display_marker != "IMU" or imu[0].array != "imu_quaternion_wxyz" or imu[0].array_index:
        raise ValueError("palm IMU mapping or display marker is invalid")

    return sensors


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One synchronized sensor reading with an explicit validity mask."""

    track: str
    sensor: SensorSpec
    value: float | tuple[float, ...] | None
    valid: bool


@dataclass(frozen=True, slots=True)
class HandGeometry:
    """Geometry for one physical track at one source frame."""

    track: str
    state: str
    raw_detection_index: int
    landmarks_3d: np.ndarray | None
    mesh_vertices: np.ndarray | None

    @property
    def present(self) -> bool:
        return self.landmarks_3d is not None or self.mesh_vertices is not None


@dataclass(frozen=True, slots=True)
class FrameData:
    """All renderer inputs for one exact stored frame."""

    position: int
    frame_index: int
    timestamp_seconds: float
    hands: tuple[HandGeometry, HandGeometry]
    bend_normalized: np.ndarray
    spread_normalized: np.ndarray
    bend_valid: np.ndarray
    spread_valid: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    palm_imu_valid: np.ndarray

    def hand(self, track: str) -> HandGeometry:
        normalized = track.upper()
        if normalized == "LEFT":
            return self.hands[0]
        if normalized == "RIGHT":
            return self.hands[1]
        raise KeyError(f"unknown track: {track}")


@dataclass(frozen=True, slots=True)
class PlaybackSequence:
    """A single variable-length sequence ready for visualization."""

    sample_id: str
    label_ar: str | None
    label_index: int | None
    signer_id: str | None
    frames: tuple[FrameData, ...]
    sensor_layout: tuple[SensorSpec, ...]
    geometry_source: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("playback sequence must contain at least one frame")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_index for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_seconds for frame in self.frames)

    def frame_at(self, position: int) -> FrameData:
        if not 0 <= int(position) < len(self.frames):
            raise IndexError(f"frame position out of range: {position}")
        return self.frames[int(position)]

    def position_for_frame(self, frame_index: int) -> int:
        for position, frame in enumerate(self.frames):
            if frame.frame_index == int(frame_index):
                return position
        raise KeyError(f"stored frame index not found: {frame_index}")

    def sensor_readings(self, position: int, track: str) -> tuple[SensorReading, ...]:
        """Return layout-ordered readings and explicit validity for a frame."""

        frame = self.frame_at(position)
        track_name = track.upper()
        track_index = TRACK_ORDER.index(track_name)
        result: list[SensorReading] = []
        for sensor in self.sensor_layout:
            if sensor.role == "bend":
                finger, joint = sensor.array_index
                raw = float(frame.bend_normalized[track_index, finger, joint])
                valid = bool(frame.bend_valid[track_index, finger, joint]) and np.isfinite(raw)
                value: float | tuple[float, ...] | None = raw if valid else None
            elif sensor.role == "spread":
                pair_index = sensor.array_index[0]
                raw = float(frame.spread_normalized[track_index, pair_index])
                valid = bool(frame.spread_valid[track_index, pair_index]) and np.isfinite(raw)
                value = raw if valid else None
            else:
                raw_quaternion = np.asarray(frame.palm_quaternion_wxyz[track_index], dtype=float)
                valid = bool(frame.palm_imu_valid[track_index]) and bool(np.isfinite(raw_quaternion).all())
                value = tuple(float(item) for item in raw_quaternion) if valid else None
            result.append(SensorReading(track_name, sensor, value, valid))
        return tuple(result)
