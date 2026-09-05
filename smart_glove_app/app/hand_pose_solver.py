"""TASK-007G presentation pose solver.

This turns one frozen TASK-008 frame into local bone rotations for the
canonical presentation rig.  It is strictly one-way: nothing computed here is
written back to a ``PlaybackSequence`` and nothing is shown to the recognizer.

Two things separate it from the TASK-007F retargeter it replaces:

* One uniform articulation convention.  Because the left rig is an exact
  matrix mirror of the right, a negative rotation about a bone's local X flexes
  it toward the palm on *both* hands, so there is no per-bone axis/sign table
  to get wrong.
* Presentation and recorded motion are separated.  Recorded wrist rotation is
  applied at the wrist joint and hard-clamped; screen placement, base
  orientation and the palm/back view live on the presentation root and are
  never touched by recorded data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
    SIDES,
    SPREAD_PAIRS,
    PresentationRig,
)

_TRACK_INDEX = {"LEFT": 0, "RIGHT": 1}
_FINGER_INDEX = {finger: index for index, finger in enumerate(FINGERS)}
_PAIR_INDEX = {pair: index for index, pair in enumerate(SPREAD_PAIRS)}
_AXIS_VECTOR = {
    "X": np.asarray((1.0, 0.0, 0.0)),
    "Y": np.asarray((0.0, 1.0, 0.0)),
    "Z": np.asarray((0.0, 0.0, 1.0)),
}
# TASK-008 stores the frozen TASK-005 degrees divided by 180. That divisor is
# the contract; no learned or presentation-specific rescaling is introduced.
CONTRACT_DEGREE_SCALE = 180.0

IDLE_STATES = frozenset({"MISSING", "LIKELY_OCCLUDED", "REJECTED_QUALITY"})


def identity_quaternion() -> np.ndarray:
    return np.asarray((1.0, 0.0, 0.0, 0.0))


def normalize_quaternion(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        return identity_quaternion()
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-12 else identity_quaternion()


def quaternion_multiply(first: Any, second: Any) -> np.ndarray:
    """Multiply WXYZ quaternions, applying ``second`` in the frame of ``first``."""

    w1, x1, y1, z1 = normalize_quaternion(first)
    w2, x2, y2, z2 = normalize_quaternion(second)
    return normalize_quaternion(
        np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
    )


def quaternion_inverse(value: Any) -> np.ndarray:
    quaternion = normalize_quaternion(value)
    return np.asarray((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]))


def axis_angle_quaternion(axis: str, degrees: float) -> np.ndarray:
    vector = _AXIS_VECTOR[str(axis).upper()]
    half = math.radians(float(degrees)) * 0.5
    return normalize_quaternion(np.asarray((math.cos(half), *(vector * math.sin(half)))))


def quaternion_angle_deg(value: Any) -> float:
    """Rotation magnitude of a WXYZ quaternion, in degrees."""

    quaternion = normalize_quaternion(value)
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(quaternion[0])))))


def slerp(first: Any, second: Any, alpha: float) -> np.ndarray:
    one = normalize_quaternion(first)
    two = normalize_quaternion(second)
    t = min(1.0, max(0.0, float(alpha)))
    dot = float(np.dot(one, two))
    if dot < 0.0:
        two, dot = -two, -dot
    if dot > 0.9995:
        return normalize_quaternion(one + t * (two - one))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sine = math.sin(theta)
    return normalize_quaternion(
        (math.sin((1.0 - t) * theta) / sine) * one + (math.sin(t * theta) / sine) * two
    )


def clamp_quaternion_angle(value: Any, max_degrees: float) -> np.ndarray:
    """Scale a rotation down so it never exceeds ``max_degrees``.

    This is what keeps recorded wrist motion from destroying the framing: the
    hand can still turn at the wrist, but only inside a bounded cone.
    """

    quaternion = normalize_quaternion(value)
    if quaternion[0] < 0.0:  # shortest arc
        quaternion = -quaternion
    angle = quaternion_angle_deg(quaternion)
    if angle <= max_degrees or angle < 1e-9:
        return quaternion
    return slerp(identity_quaternion(), quaternion, max_degrees / angle)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class HandPose:
    """Local bone rotations for one hand, plus presentation-only status."""

    side: str
    bones_wxyz: Mapping[str, tuple[float, float, float, float]]
    state: str
    dimmed: bool
    bend_valid_count: int
    spread_valid_count: int
    wrist_valid: bool
    wrist_angle_deg: float

    def as_qml(self) -> dict[str, Any]:
        return {
            "bones": {name: list(values) for name, values in self.bones_wxyz.items()},
            "state": self.state,
            "dimmed": self.dimmed,
            "bendValid": self.bend_valid_count,
            "spreadValid": self.spread_valid_count,
            "wristValid": self.wrist_valid,
            "wristAngle": self.wrist_angle_deg,
        }


class HandPoseSolver:
    """Solve presentation bone rotations for both hands, frame by frame."""

    def __init__(self, rig: PresentationRig) -> None:
        self.rig = rig
        self._wrist_reference: dict[str, np.ndarray | None] = {side: None for side in SIDES}
        self._last_wrist: dict[str, np.ndarray] = {side: identity_quaternion() for side in SIDES}
        self._last_bones: dict[str, dict[str, np.ndarray]] = {side: {} for side in SIDES}

    def reset(self) -> None:
        """Forget per-sequence state.

        This must run between queue items.  Carrying a wrist reference from a
        previous sign is what makes a new letter start from the previous
        letter's orientation instead of from its own first frame.
        """

        for side in SIDES:
            self._wrist_reference[side] = None
            self._last_wrist[side] = identity_quaternion()
            self._last_bones[side].clear()

    # ---- neutral ------------------------------------------------------------

    def neutral_pose(self, side: str) -> HandPose:
        normalized = str(side).upper()
        if normalized not in SIDES:
            raise KeyError(f"unknown hand side: {side!r}")
        bones = {bone: (1.0, 0.0, 0.0, 0.0) for bone in self.rig.required_bones}
        return HandPose(normalized, bones, "IDLE", False, 0, 0, False, 0.0)

    def neutral_qml_pose(self) -> dict[str, Any]:
        return {side: self.neutral_pose(side).as_qml() for side in SIDES}

    # ---- articulation -------------------------------------------------------

    def _bend_quaternion(self, degrees: float, limit_deg: float) -> np.ndarray:
        bounded = min(limit_deg, max(0.0, float(degrees)))
        return axis_angle_quaternion(self.rig.bend_axis, self.rig.bend_sign * bounded)

    def _spread_degrees(self, measured_deg: Mapping[str, float]) -> dict[str, float]:
        low, high = self.rig.spread_clamp_deg
        neutral = self.rig.spread_neutral_deg
        result: dict[str, float] = {}
        for target in self.rig.spread_targets:
            total = 0.0
            for pair in target.sum_of:
                total += measured_deg.get(pair, neutral[pair]) - neutral[pair]
            result[target.bone] = target.sign * min(high, max(low, total))
        return result

    def _wrist_quaternion(self, frame: Any, side: str, *, update_state: bool) -> tuple[np.ndarray, bool]:
        track = _TRACK_INDEX[side]
        valid = bool(np.asarray(frame.palm_imu_valid)[track])
        raw = np.asarray(frame.palm_quaternion_wxyz)[track]
        if not valid or raw.shape != (4,) or not np.isfinite(raw).all():
            return self._last_wrist[side].copy(), False
        current = normalize_quaternion(raw)
        reference = self._wrist_reference[side]
        if reference is None:
            if update_state:
                self._wrist_reference[side] = current.copy()
                self._last_wrist[side] = identity_quaternion()
            return identity_quaternion(), True
        if float(np.dot(reference, current)) < 0.0:
            current = -current
        delta = quaternion_multiply(quaternion_inverse(reference), current)
        clamped = clamp_quaternion_angle(delta, self.rig.wrist_max_angle_deg)
        if update_state:
            self._last_wrist[side] = clamped.copy()
        return clamped, True

    def pose_for_frame(self, frame: Any, side: str, *, update_state: bool = True) -> HandPose:
        normalized = str(side).upper()
        if normalized not in SIDES:
            raise KeyError(f"unknown hand side: {side!r}")
        track = _TRACK_INDEX[normalized]
        hand = frame.hand(normalized)
        state = str(getattr(hand, "state", "MISSING")).upper()

        bones: dict[str, np.ndarray] = {bone: identity_quaternion() for bone in self.rig.required_bones}
        held = self._last_bones[normalized]

        bend_values = np.asarray(frame.bend_normalized)
        bend_mask = np.asarray(frame.bend_valid)
        bend_valid_count = 0
        for finger, chain in self.rig.chains.items():
            finger_index = _FINGER_INDEX[finger]
            for joint_index, bone in enumerate(chain.joints):
                raw = bend_values[track, finger_index, joint_index]
                valid = bool(bend_mask[track, finger_index, joint_index]) and _finite(raw)
                if valid:
                    quaternion = self._bend_quaternion(
                        float(raw) * CONTRACT_DEGREE_SCALE, chain.joint_limits_deg[joint_index]
                    )
                    bend_valid_count += 1
                    if update_state:
                        held[bone] = quaternion.copy()
                else:
                    quaternion = held.get(bone, identity_quaternion())
                bones[bone] = quaternion

        spread_values = np.asarray(frame.spread_normalized)
        spread_mask = np.asarray(frame.spread_valid)
        measured: dict[str, float] = {}
        spread_valid_count = 0
        for pair in SPREAD_PAIRS:
            pair_index = _PAIR_INDEX[pair]
            raw = spread_values[track, pair_index]
            if bool(spread_mask[track, pair_index]) and _finite(raw):
                measured[pair] = float(raw) * CONTRACT_DEGREE_SCALE
                spread_valid_count += 1
        for bone, degrees in self._spread_degrees(measured).items():
            quaternion = axis_angle_quaternion(self.rig.spread_axis, degrees)
            bones[bone] = quaternion
            if update_state:
                held[bone] = quaternion.copy()

        wrist, wrist_valid = self._wrist_quaternion(frame, normalized, update_state=update_state)
        bones[self.rig.wrist_bone] = wrist

        return HandPose(
            side=normalized,
            bones_wxyz={
                bone: tuple(float(v) for v in normalize_quaternion(value))
                for bone, value in bones.items()
            },
            state=state,
            dimmed=state in IDLE_STATES,
            bend_valid_count=bend_valid_count,
            spread_valid_count=spread_valid_count,
            wrist_valid=wrist_valid,
            wrist_angle_deg=quaternion_angle_deg(wrist),
        )

    def frame_pose(
        self,
        frame: Any,
        *,
        next_frame: Any | None = None,
        interpolation_alpha: float = 0.0,
        smooth: bool = False,
    ) -> dict[str, HandPose]:
        current = {side: self.pose_for_frame(frame, side) for side in SIDES}
        alpha = min(1.0, max(0.0, float(interpolation_alpha)))
        if not smooth or next_frame is None or not 0.0 < alpha < 1.0:
            return current
        blended: dict[str, HandPose] = {}
        for side in SIDES:
            upcoming = self.pose_for_frame(next_frame, side, update_state=False)
            bones = {
                bone: tuple(
                    float(v)
                    for v in slerp(current[side].bones_wxyz[bone], upcoming.bones_wxyz[bone], alpha)
                )
                for bone in self.rig.required_bones
            }
            blended[side] = HandPose(
                side=side,
                bones_wxyz=bones,
                state=current[side].state,
                dimmed=current[side].dimmed,
                bend_valid_count=current[side].bend_valid_count,
                spread_valid_count=current[side].spread_valid_count,
                wrist_valid=current[side].wrist_valid,
                wrist_angle_deg=current[side].wrist_angle_deg,
            )
        return blended

    @staticmethod
    def qml_pose(poses: Mapping[str, HandPose]) -> dict[str, Any]:
        return {side: pose.as_qml() for side, pose in poses.items()}


__all__ = [
    "CONTRACT_DEGREE_SCALE",
    "HandPose",
    "HandPoseSolver",
    "axis_angle_quaternion",
    "clamp_quaternion_angle",
    "identity_quaternion",
    "normalize_quaternion",
    "quaternion_angle_deg",
    "quaternion_inverse",
    "quaternion_multiply",
    "slerp",
]
