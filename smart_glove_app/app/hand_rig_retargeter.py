"""Presentation-only TASK-005-to-Blender direct-bone retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from smart_glove_app.rendering.rig_profile import (
    FINGERS,
    SIDES,
    SPREADS,
    ChannelCalibration,
    RigProfile,
)


_TRACK_INDEX = {"LEFT": 0, "RIGHT": 1}
_FINGER_INDEX = {finger: index for index, finger in enumerate(FINGERS)}
_SPREAD_INDEX = {pair: index for index, pair in enumerate(SPREADS)}
_AXIS_VECTOR = {
    "X": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    "Y": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    "Z": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
}


def _identity() -> np.ndarray:
    return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)


def _normalise(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        return _identity()
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else _identity()


def quaternion_multiply(first: Any, second: Any) -> np.ndarray:
    """Multiply WXYZ quaternions, applying ``second`` after ``first``."""

    w1, x1, y1, z1 = _normalise(first)
    w2, x2, y2, z2 = _normalise(second)
    return _normalise(
        np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dtype=np.float64,
        )
    )


def quaternion_inverse(quaternion: Any) -> np.ndarray:
    value = _normalise(quaternion)
    return np.asarray((value[0], -value[1], -value[2], -value[3]), dtype=np.float64)


def axis_angle_quaternion(axis: str, degrees: float) -> np.ndarray:
    vector = _AXIS_VECTOR[str(axis).upper()]
    half = math.radians(float(degrees)) * 0.5
    sine = math.sin(half)
    return np.asarray((math.cos(half), *(vector * sine)), dtype=np.float64)


def quaternion_slerp(first: Any, second: Any, alpha: float) -> np.ndarray:
    """Shortest-path SLERP for presentation interpolation."""

    one = _normalise(first)
    two = _normalise(second)
    t = min(1.0, max(0.0, float(alpha)))
    dot = float(np.dot(one, two))
    if dot < 0.0:
        two = -two
        dot = -dot
    if dot > 0.9995:
        return _normalise(one + t * (two - one))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sine = math.sin(theta)
    return _normalise(
        (math.sin((1.0 - t) * theta) / sine) * one
        + (math.sin(t * theta) / sine) * two
    )


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class HandRigPose:
    side: str
    bone_deltas_wxyz: Mapping[str, tuple[float, float, float, float]]
    state: str
    dimmed: bool
    bend_valid: Mapping[str, bool]
    spread_valid: Mapping[str, bool]
    palm_valid: bool

    def as_qml(self) -> dict[str, Any]:
        return {
            "bones": {name: list(values) for name, values in self.bone_deltas_wxyz.items()},
            "state": self.state,
            "dimmed": self.dimmed,
            "bend_valid": dict(self.bend_valid),
            "spread_valid": dict(self.spread_valid),
            "palm_valid": self.palm_valid,
        }


class HandRigRetargeter:
    """Map frozen TASK-005 channels to an already-loaded persistent rig.

    The values returned by this class are render-only local quaternion deltas.
    They are never written to a ``PlaybackSequence`` and are never supplied to
    the recognizer. Stored TASK-008 normalized values are converted back to the
    frozen 0..180-degree presentation input using the contract divisor only.
    """

    def __init__(self, profile: RigProfile) -> None:
        self.profile = profile
        self._last_bends: dict[str, dict[str, np.ndarray]] = {side: {} for side in SIDES}
        self._last_spreads: dict[str, dict[str, np.ndarray]] = {side: {} for side in SIDES}
        self._palm_reference: dict[str, np.ndarray | None] = {side: None for side in SIDES}
        self._last_palm_delta: dict[str, np.ndarray] = {side: _identity() for side in SIDES}

    def reset(self) -> None:
        for side in SIDES:
            self._last_bends[side].clear()
            self._last_spreads[side].clear()
            self._palm_reference[side] = None
            self._last_palm_delta[side] = _identity()

    def neutral_pose(self, side: str, *, state: str = "IDLE", dimmed: bool = False) -> HandRigPose:
        normalized = str(side).upper()
        if normalized not in SIDES:
            raise KeyError(f"unknown hand side: {side!r}")
        bones = {bone: tuple(float(v) for v in _identity()) for bone in self.profile.required_deform_bones}
        return HandRigPose(normalized, bones, state, dimmed, {}, {}, False)

    def neutral_qml_pose(self) -> dict[str, Any]:
        return {side: self.neutral_pose(side).as_qml() for side in SIDES}

    def _channel_value(
        self,
        value: Any,
        valid: Any,
        calibration: ChannelCalibration,
        *,
        previous: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        if bool(valid) and _finite_scalar(value):
            # TASK-008's normalized value is exactly the frozen TASK-005
            # degrees divided by 180. No learned or presentation normalization
            # is introduced here.
            degrees = float(value) * 180.0
            degrees = min(calibration.safe_max_deg, max(calibration.safe_min_deg, degrees))
            degrees += calibration.neutral_offset_deg
            return axis_angle_quaternion(calibration.axis, calibration.sign * degrees), True
        return (previous.copy() if previous is not None else _identity()), False

    def _palm_delta(self, frame: Any, side: str, *, update_state: bool, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
        track = _TRACK_INDEX[side]
        valid = bool(np.asarray(frame.palm_imu_valid)[track])
        raw = np.asarray(frame.palm_quaternion_wxyz)[track]
        if not valid or raw.shape != (4,) or not np.isfinite(raw).all():
            return fallback.copy(), False
        current = _normalise(raw)
        reference = self._palm_reference[side]
        if reference is None:
            if update_state:
                self._palm_reference[side] = current.copy()
            return _identity(), True
        if float(np.dot(reference, current)) < 0.0:
            current = -current
        delta = quaternion_multiply(quaternion_inverse(reference), current)
        if update_state:
            self._last_palm_delta[side] = delta.copy()
        return delta, True

    def pose_for_frame(
        self,
        frame: Any,
        side: str,
        *,
        update_state: bool = True,
        fallback: HandRigPose | None = None,
    ) -> HandRigPose:
        normalized = str(side).upper()
        if normalized not in SIDES:
            raise KeyError(f"unknown hand side: {side!r}")
        track = _TRACK_INDEX[normalized]
        hand = frame.hand(normalized)
        state = str(getattr(hand, "state", "MISSING")).upper()
        dimmed = state in {"MISSING", "LIKELY_OCCLUDED", "REJECTED_QUALITY"}
        fallback_bones = dict(fallback.bone_deltas_wxyz) if fallback is not None else {}
        deltas: dict[str, np.ndarray] = {
            bone: np.asarray(fallback_bones.get(bone, _identity()), dtype=np.float64)
            for bone in self.profile.required_deform_bones
        }
        bend_valid: dict[str, bool] = {}
        spread_valid: dict[str, bool] = {}

        bend_values = np.asarray(frame.bend_normalized)
        bend_mask = np.asarray(frame.bend_valid)
        for channel, sides in self.profile.bends.items():
            finger, index_text = channel.split("[", 1)
            joint_index = int(index_text.rstrip("]"))
            calibration = sides[normalized]
            previous = self._last_bends[normalized].get(channel)
            value, valid = self._channel_value(
                bend_values[track, _FINGER_INDEX[finger], joint_index],
                bend_mask[track, _FINGER_INDEX[finger], joint_index],
                calibration,
                previous=previous if previous is not None else deltas[calibration.bone],
            )
            deltas[calibration.bone] = value
            bend_valid[channel] = valid
            if valid and update_state:
                self._last_bends[normalized][channel] = value.copy()

        spread_values = np.asarray(frame.spread_normalized)
        spread_mask = np.asarray(frame.spread_valid)
        for channel, item in self.profile.spreads.items():
            calibration = item[normalized]
            previous = self._last_spreads[normalized].get(channel)
            value, valid = self._channel_value(
                spread_values[track, _SPREAD_INDEX[channel]],
                spread_mask[track, _SPREAD_INDEX[channel]],
                calibration,
                previous=previous if previous is not None else deltas[calibration.bone],
            )
            deltas[calibration.bone] = value
            spread_valid[channel] = valid
            if valid and update_state:
                self._last_spreads[normalized][channel] = value.copy()

        palm_fallback = np.asarray(
            fallback_bones.get(self.profile.palm_bone, self._last_palm_delta[normalized]),
            dtype=np.float64,
        )
        palm_delta, palm_valid = self._palm_delta(frame, normalized, update_state=update_state, fallback=palm_fallback)
        deltas[self.profile.palm_bone] = palm_delta
        if palm_valid and update_state:
            self._last_palm_delta[normalized] = palm_delta.copy()

        return HandRigPose(
            normalized,
            {bone: tuple(float(v) for v in _normalise(value)) for bone, value in deltas.items()},
            state,
            dimmed,
            bend_valid,
            spread_valid,
            palm_valid,
        )

    def frame_pose(
        self,
        frame: Any,
        *,
        next_frame: Any | None = None,
        interpolation_alpha: float = 0.0,
        smooth: bool = False,
    ) -> dict[str, HandRigPose]:
        current = {side: self.pose_for_frame(frame, side) for side in SIDES}
        alpha = min(1.0, max(0.0, float(interpolation_alpha)))
        if not smooth or not 0.0 < alpha < 1.0 or next_frame is None:
            return current
        result: dict[str, HandRigPose] = {}
        for side in SIDES:
            upcoming = self.pose_for_frame(next_frame, side, update_state=False, fallback=current[side])
            deltas = {
                bone: tuple(
                    float(v)
                    for v in quaternion_slerp(
                        current[side].bone_deltas_wxyz[bone],
                        upcoming.bone_deltas_wxyz[bone],
                        alpha,
                    )
                )
                for bone in self.profile.required_deform_bones
            }
            result[side] = HandRigPose(
                side,
                deltas,
                current[side].state,
                current[side].dimmed,
                current[side].bend_valid,
                current[side].spread_valid,
                current[side].palm_valid,
            )
        return result


__all__ = [
    "HandRigPose",
    "HandRigRetargeter",
    "axis_angle_quaternion",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_slerp",
]
