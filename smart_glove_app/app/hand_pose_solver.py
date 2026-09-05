"""TASK-007G/007I presentation pose solver.

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
* For the shipped skinned GLB, stored TASK-008 landmark directions provide the
  signed segment orientation that unsigned TASK-005 spread cannot encode.  A
  shortest-arc swing is solved from immutable rest pose to each direction,
  preserving the authored axial roll and avoiding accumulated transforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kinematics.hand_frame import build_palm_frame
from kinematics.layout import FINGER_CHAINS

from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
    SIDES,
    SPREAD_PAIRS,
    PresentationRig,
)
from smart_glove_app.rendering.rig_pose_calibration import (
    GlbPoseCalibration,
    matrix_to_quaternion_wxyz,
    quaternion_to_matrix_wxyz,
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

# TASK-005's palm frame is [lateral | palmar normal | distal].  The shipped
# Qt/GLB presentation frame is [screen-left/right | distal | palmar normal].
# The lateral axis is negated because the canonical GLB is mirrored so the
# viewer sees both palms as hands rather than two copies of one side.  This is
# a proper rotation (determinant +1), not a reflection of the hand geometry.
_DEFAULT_SOURCE_TO_PRESENTATION = np.asarray(
    (
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
_SOURCE_CHAIN_NAMES = {
    finger: tuple(int(index) for index in chain)
    for finger, chain in FINGER_CHAINS.items()
}
_BASE_SPREAD_VALIDITY: Mapping[str, tuple[str, ...]] = {
    "thumb": ("thumb-index",),
    "index": ("thumb-index", "index-middle"),
    "middle": ("index-middle", "middle-ring"),
    "ring": ("middle-ring", "ring-pinky"),
    "pinky": ("ring-pinky",),
}


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
    """Solve presentation bone rotations for both hands, frame by frame.

    The TASK-005 channels remain the primary input and remain untouched.  When
    the normal TASK-007G GLB is available, the solver additionally uses the
    already stored TASK-008 3D landmarks to recover the *direction* of every
    source bone.  This is necessary for sign fidelity because TASK-005 spread
    is intentionally an unsigned pairwise measurement: it cannot say whether
    a particular finger is on the thumb side or the pinky side of the palm.

    Landmark guidance is presentation-only.  It is never written to a source
    sequence and is never passed to TASK-009A or the recognizer.  If an asset,
    landmark frame, or required validity channel is unavailable, the original
    TASK-007G channel solver remains the safe fallback.
    """

    def __init__(
        self,
        rig: PresentationRig,
        *,
        rig_asset_paths: Mapping[str, str | Path] | None = None,
    ) -> None:
        self.rig = rig
        self._wrist_reference: dict[str, np.ndarray | None] = {side: None for side in SIDES}
        self._last_wrist: dict[str, np.ndarray] = {side: identity_quaternion() for side in SIDES}
        self._last_bones: dict[str, dict[str, np.ndarray]] = {side: {} for side in SIDES}
        self._landmark_calibration: dict[str, GlbPoseCalibration | None] = {
            side: None for side in SIDES
        }
        self._landmark_calibration_errors: dict[str, str] = {}
        self._source_to_presentation = self._load_source_frame_mapping()
        for side in SIDES:
            path = (rig_asset_paths or {}).get(side)
            if path is None:
                continue
            try:
                self._landmark_calibration[side] = GlbPoseCalibration.from_glb(
                    path, self.rig.required_bones
                )
            except (OSError, ValueError) as exc:
                # A broken optional calibration must not prevent visualization
                # in a valid channel-only/diagnostic setup.
                self._landmark_calibration_errors[side] = f"{type(exc).__name__}: {exc}"

    @property
    def landmark_guidance_available(self) -> bool:
        """Whether both persistent hand GLBs have immutable rest calibration."""

        return all(self._landmark_calibration[side] is not None for side in SIDES)

    @property
    def landmark_calibration_errors(self) -> Mapping[str, str]:
        """Read-only diagnostics for optional landmark-guidance calibration."""

        return dict(self._landmark_calibration_errors)

    def _load_source_frame_mapping(self) -> np.ndarray:
        raw = self.rig.raw.get("landmark_retargeting", {})
        candidate = raw.get("source_to_presentation_matrix") if isinstance(raw, Mapping) else None
        matrix = (
            np.asarray(candidate, dtype=np.float64)
            if candidate is not None
            else _DEFAULT_SOURCE_TO_PRESENTATION.copy()
        )
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return _DEFAULT_SOURCE_TO_PRESENTATION.copy()
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            return _DEFAULT_SOURCE_TO_PRESENTATION.copy()
        if float(np.linalg.det(matrix)) < 0.0:
            return _DEFAULT_SOURCE_TO_PRESENTATION.copy()
        return matrix

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

    # ---- landmark-guided presentation retargeting -------------------------

    def _landmark_targets(
        self,
        landmarks: Any,
        side: str,
        calibration: GlbPoseCalibration,
    ) -> dict[str, np.ndarray] | None:
        """Build absolute source-bone frames in the GLB presentation frame.

        TASK-005 deliberately stores bend magnitudes and unsigned spread
        magnitudes.  The stored TASK-008 landmarks retain the missing signed
        direction and the thumb-opposition information.  Each source chain has
        four segments and each presentation chain has one metacarpal plus
        three phalanges, so the mapping is one-to-one.
        """

        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        palm_frame, _ = build_palm_frame(points, side.lower())
        if palm_frame is None:
            return None

        local_points = (points - points[0]) @ palm_frame.rotation
        source_to_presentation = self._source_to_presentation
        palm_normal = source_to_presentation @ np.asarray((0.0, 1.0, 0.0))
        targets: dict[str, np.ndarray] = {}

        for finger in FINGERS:
            source_chain = _SOURCE_CHAIN_NAMES[finger]
            chain = self.rig.chains[finger]
            bone_names = (chain.metacarpal, *chain.joints)
            for segment_index, bone in enumerate(bone_names):
                vector = source_to_presentation @ (
                    local_points[source_chain[segment_index + 1]]
                    - local_points[source_chain[segment_index]]
                )
                length = float(np.linalg.norm(vector))
                if not math.isfinite(length) or length <= 1e-8:
                    return None
                direction = vector / length

                # Use the source palm normal to choose a stable roll while the
                # segment direction determines the actual bone articulation.
                # At the rare pole where a segment is parallel to the normal,
                # the immutable GLB rest Z axis supplies the least surprising
                # roll; no synthetic position or scientific channel is made.
                z_axis = palm_normal - float(np.dot(palm_normal, direction)) * direction
                if float(np.linalg.norm(z_axis)) <= 1e-7:
                    rest_z = calibration.world_rotations[bone][:, 2]
                    z_axis = rest_z - float(np.dot(rest_z, direction)) * direction
                if float(np.linalg.norm(z_axis)) <= 1e-7:
                    lateral = source_to_presentation @ np.asarray((1.0, 0.0, 0.0))
                    z_axis = lateral - float(np.dot(lateral, direction)) * direction
                z_norm = float(np.linalg.norm(z_axis))
                if z_norm <= 1e-8:
                    return None
                z_axis = z_axis / z_norm
                x_axis = np.cross(direction, z_axis)
                x_norm = float(np.linalg.norm(x_axis))
                if x_norm <= 1e-8:
                    return None
                x_axis = x_axis / x_norm
                z_axis = np.cross(x_axis, direction)
                z_axis = z_axis / float(np.linalg.norm(z_axis))
                targets[bone] = np.column_stack((x_axis, direction, z_axis))

        return targets

    @staticmethod
    def _spread_valid_for_base(frame: Any, track: int, finger: str) -> bool:
        spread_mask = np.asarray(frame.spread_valid)
        if spread_mask.ndim != 2 or spread_mask.shape[0] <= track:
            return False
        for pair in _BASE_SPREAD_VALIDITY[finger]:
            pair_index = _PAIR_INDEX[pair]
            if not bool(spread_mask[track, pair_index]):
                return False
        return True

    @staticmethod
    def _swing_to_direction(
        current_direction: np.ndarray,
        target_direction: np.ndarray,
        reference_axis: np.ndarray,
    ) -> np.ndarray:
        """Return the shortest world-space swing between two bone directions.

        The stored landmarks determine where a segment points, but they do not
        provide a reliable surface normal for the segment's axial roll.  A
        full frame fit therefore lets tracker noise twist the skinned mesh
        around a finger.  Keeping the shortest swing preserves the authored
        rest roll while still reproducing the measured signed flexion,
        abduction and thumb opposition.
        """

        source = np.asarray(current_direction, dtype=np.float64)
        target = np.asarray(target_direction, dtype=np.float64)
        source_norm = float(np.linalg.norm(source))
        target_norm = float(np.linalg.norm(target))
        if source_norm <= 1e-8 or target_norm <= 1e-8:
            raise ValueError("bone direction cannot be zero")
        source = source / source_norm
        target = target / target_norm
        cross = np.cross(source, target)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if cross_norm <= 1e-8:
            if dot >= 0.0:
                return np.eye(3, dtype=np.float64)
            axis = np.asarray(reference_axis, dtype=np.float64)
            axis = axis - float(np.dot(axis, source)) * source
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-8:
                # The calibrated rest frame is orthonormal, so this is only a
                # defensive fallback for malformed caller input.
                axis = np.cross(source, np.asarray((1.0, 0.0, 0.0)))
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= 1e-8:
                    axis = np.cross(source, np.asarray((0.0, 1.0, 0.0)))
                    axis_norm = float(np.linalg.norm(axis))
            axis = axis / axis_norm
            angle = math.pi
        else:
            axis = cross / cross_norm
            angle = math.atan2(cross_norm, dot)

        x, y, z = axis
        skew = np.asarray(
            ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (
            1.0 - math.cos(angle)
        ) * (skew @ skew)

    def _landmark_bones(
        self,
        frame: Any,
        side: str,
        wrist: np.ndarray,
        fallback_bones: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """Solve absolute rest-relative deltas from one stored landmark frame.

        Only the source segment direction is used from each target frame.  A
        landmark-derived full frame was tested and rejected because tracker
        noise in its axial roll produced visible skinned twisting.  The
        shortest swing below keeps the neutral GLB roll authored by TASK-007G.
        """

        calibration = self._landmark_calibration.get(side)
        if calibration is None:
            return None
        hand = frame.hand(side)
        landmarks = getattr(hand, "landmarks_3d", None)
        result = self._landmark_targets(landmarks, side, calibration)
        if result is None:
            return None
        targets = result

        track = _TRACK_INDEX[side]
        bend_mask = np.asarray(frame.bend_valid)
        actual_world: dict[str, np.ndarray] = {
            self.rig.wrist_bone: calibration.world_rotations[self.rig.wrist_bone]
            @ quaternion_to_matrix_wxyz(wrist),
        }
        output = {bone: normalize_quaternion(value) for bone, value in fallback_bones.items()}
        output[self.rig.wrist_bone] = normalize_quaternion(wrist)

        # The target shape is expressed before the recorded wrist delta; the
        # same local wrist transform is then applied to every target frame.
        palm_world = calibration.world_rotations[self.rig.wrist_bone]
        shape_world = palm_world @ quaternion_to_matrix_wxyz(wrist) @ palm_world.T

        for finger in FINGERS:
            chain = self.rig.chains[finger]
            parent = self.rig.wrist_bone
            finger_index = _FINGER_INDEX[finger]
            bone_names = (chain.metacarpal, *chain.joints)
            for joint_index, bone in enumerate(bone_names):
                geometry_valid = (
                    self._spread_valid_for_base(frame, track, finger)
                    if joint_index == 0
                    else bool(
                    bend_mask.ndim == 3
                        and bend_mask.shape[0] > track
                        and bool(bend_mask[track, finger_index, joint_index - 1])
                    )
                )
                if geometry_valid:
                    target_world = shape_world @ targets[bone]
                    current_world = (
                        actual_world[parent] @ calibration.local_rotations[bone]
                    )
                    swing = self._swing_to_direction(
                        current_world[:, 1],
                        target_world[:, 1],
                        current_world[:, 0],
                    )
                    desired_world = swing @ current_world
                    local_delta = (
                        calibration.local_rotations[bone].T
                        @ actual_world[parent].T
                        @ desired_world
                    )
                    try:
                        delta = matrix_to_quaternion_wxyz(local_delta)
                    except (OSError, ValueError):
                        geometry_valid = False
                    else:
                        output[bone] = delta
                        actual_world[bone] = desired_world

                if not geometry_valid:
                    delta = normalize_quaternion(fallback_bones[bone])
                    output[bone] = delta
                    actual_world[bone] = (
                        actual_world[parent]
                        @ calibration.local_rotations[bone]
                        @ quaternion_to_matrix_wxyz(delta)
                    )
                parent = bone
        return output

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

        # The channel solver above is still the complete fallback path.  With
        # the shipped one-hand GLB and a valid stored TASK-008 landmark pose,
        # replace only the presentation deltas with absolute source-bone
        # orientations.  This removes the ambiguity of unsigned spread and
        # avoids clipping valid source flexion at the conservative fallback
        # limits.  Scientific arrays and masks are read-only inputs here.
        guided = self._landmark_bones(frame, normalized, wrist, bones)
        if guided is not None:
            bones = guided

        # Missing/invalid presentation channels hold the last *displayed*
        # transform.  Update this cache after landmark guidance so a later
        # missing frame cannot jump back to the channel-only fallback pose.
        if update_state:
            for bone, value in bones.items():
                held[bone] = normalize_quaternion(value).copy()

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
