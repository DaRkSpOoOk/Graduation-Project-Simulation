"""Per-frame hand kinematics from the TASK-004 tracked representation.

Strictly per-frame. No temporal filtering, smoothing, interpolation, forward
fill, or velocity estimation of any kind is performed here, so raw kinematic
behaviour stays observable for whatever validation comes next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    MIN_PROJECTED_NORM,
    angle_between_unit,
    project_onto_plane,
    rotation_matrix_to_quaternion_wxyz,
    safe_normalize,
    turn_angle,
)
from .hand_frame import build_palm_frame
from .layout import (
    CHAIN_ORDER,
    FINGER_CHAINS,
    FINGER_ORDER,
    N_CHAIN,
    N_FINGERS,
    N_JOINTS,
    N_SPREAD,
    N_TRACKS,
    SPREAD_PAIRS,
    SPREAD_SEGMENTS,
    TRACK_ORDER,
)

# Tracker states that carry a real reconstructed pose. Mirrors
# tracking.wilor.schema.POSE_STATES, restated so kinematics does not import
# the tracker at runtime (pipeline order is extraction -> tracking ->
# kinematics). tests/test_kinematics.py asserts the two cannot diverge.
POSE_BEARING_STATES = frozenset({"OBSERVED", "AMBIGUOUS"})

FLAG_NO_POSE = "NO_POSE_STATE_{state}"
FLAG_TRACK_AMBIGUOUS = "TRACK_STATE_AMBIGUOUS"
FLAG_NON_FINITE_JOINTS = "JOINTS_NON_FINITE"
FLAG_WRONG_JOINT_SHAPE = "JOINTS_WRONG_SHAPE"
FLAG_ZERO_LENGTH_BONE = "ZERO_LENGTH_BONE_{finger}_{chain}"
FLAG_SPREAD_DEGENERATE = "SPREAD_DIRECTION_DEGENERATE_{finger}"
FLAG_QUATERNION_UNNORMALIZABLE = "QUATERNION_NOT_NORMALIZABLE"


@dataclass(slots=True)
class HandKinematics:
    """Kinematics for one hand in one frame."""

    valid: bool = False
    palm_frame_valid: bool = False
    flexion_deg: np.ndarray = field(default_factory=lambda: np.full((N_FINGERS, N_CHAIN), np.nan))
    spread_deg: np.ndarray = field(default_factory=lambda: np.full((N_SPREAD,), np.nan))
    palm_rotation: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))
    palm_quaternion: np.ndarray = field(default_factory=lambda: np.full((4,), np.nan))
    flags: list[str] = field(default_factory=list)


def compute_flexion(joints: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Three turn angles per finger, in degrees, shape (5, 3).

    Angle k of a finger is the bend of the polyline at its k-th chain joint,
    i.e. the angle between the bone arriving at that joint and the bone
    leaving it. Straight is 0 degrees and bend increases positively; see
    ``geometry.turn_angle``.
    """

    flexion = np.full((N_FINGERS, N_CHAIN), np.nan, dtype=np.float64)
    flags: list[str] = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        chain = FINGER_CHAINS[finger]
        for chain_index in range(N_CHAIN):
            angle = turn_angle(
                joints[chain[chain_index]],
                joints[chain[chain_index + 1]],
                joints[chain[chain_index + 2]],
            )
            if angle is None:
                flags.append(
                    FLAG_ZERO_LENGTH_BONE.format(
                        finger=finger, chain=CHAIN_ORDER[chain_index]
                    )
                )
                continue
            flexion[finger_index, chain_index] = angle
    return flexion, flags


def compute_spread(joints: np.ndarray, palm_normal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Unsigned palm-plane angle between adjacent finger directions, shape (4,).

    Each finger's proximal-phalanx direction is projected onto the palm plane
    (the plane through the origin perpendicular to the palmar normal) and
    re-normalized; the output is the angle between adjacent projected
    directions. Projecting first is what makes the value palm-relative: a rigid
    rotation of the whole hand rotates the directions and the normal together,
    leaving every in-plane angle unchanged.

    The output is an UNSIGNED magnitude in [0, 180] degrees. No signed
    adduction/abduction convention is claimed: a robust signed convention needs
    a per-finger neutral axis that this representation does not establish.
    """

    directions: dict[str, np.ndarray | None] = {}
    flags: list[str] = []
    for finger in FINGER_ORDER:
        start, end = SPREAD_SEGMENTS[finger]
        raw = safe_normalize(joints[end] - joints[start])
        if raw is None:
            directions[finger] = None
            flags.append(FLAG_SPREAD_DEGENERATE.format(finger=finger))
            continue
        projected = safe_normalize(
            project_onto_plane(raw, palm_normal), minimum=MIN_PROJECTED_NORM
        )
        if projected is None:
            # The finger points too close to the palm normal for its in-plane
            # direction to mean anything: the projection is dominated by
            # reconstruction noise rather than anatomy. Emitting NaN here is
            # what keeps a noise-driven direction flip from being reported as
            # a 170-degree spread.
            directions[finger] = None
            flags.append(FLAG_SPREAD_DEGENERATE.format(finger=finger))
            continue
        directions[finger] = projected

    spread = np.full((N_SPREAD,), np.nan, dtype=np.float64)
    for pair_index, (first, second) in enumerate(SPREAD_PAIRS):
        one, two = directions[first], directions[second]
        if one is None or two is None:
            continue
        spread[pair_index] = angle_between_unit(one, two)
    return spread, flags


def compute_hand_kinematics(
    joints: np.ndarray | None, track: str, state: str
) -> HandKinematics:
    """Full kinematics for one hand in one frame.

    Two validity flags are returned, because they answer different questions:

    ``valid`` is the strict one -- True only when EVERY output channel for this
    hand-instance is finite. Any degeneracy clears it.

    ``palm_frame_valid`` is True when the hand had a usable pose, finite joints,
    and a well-conditioned palm frame, i.e. the orientation channels are
    trustworthy. It stays True when only a per-channel quantity is undefined,
    such as one spread pair whose finger points along the palm normal.

    Partial degeneracy does not discard good data: channels that are
    geometrically sound are still written, so a consumer wanting only flexion
    should mask with ``np.isfinite(flexion_deg)`` rather than with ``valid``.
    """

    result = HandKinematics()

    if state not in POSE_BEARING_STATES:
        # TASK-004 says there is no usable pose here. Nothing is invented:
        # every float channel stays NaN.
        result.flags.append(FLAG_NO_POSE.format(state=state))
        return result
    if joints is None:
        result.flags.append(FLAG_NO_POSE.format(state=state))
        return result

    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (N_JOINTS, 3):
        result.flags.append(FLAG_WRONG_JOINT_SHAPE)
        return result
    if not np.isfinite(joints).all():
        result.flags.append(FLAG_NON_FINITE_JOINTS)
        return result

    if state == "AMBIGUOUS":
        # A real pose exists and TASK-004 deliberately exposes it; the identity
        # assignment is what was uncertain, not the geometry. Kinematics are
        # computed and the caveat travels with the frame.
        result.flags.append(FLAG_TRACK_AMBIGUOUS)

    frame, frame_flags = build_palm_frame(joints, track)
    result.flags.extend(frame_flags)
    if frame is None:
        return result

    quaternion = rotation_matrix_to_quaternion_wxyz(frame.rotation)
    if quaternion is None:
        result.flags.append(FLAG_QUATERNION_UNNORMALIZABLE)
        return result
    result.palm_frame_valid = True

    flexion, flexion_flags = compute_flexion(joints)
    spread, spread_flags = compute_spread(joints, frame.normal)
    result.flags.extend(flexion_flags)
    result.flags.extend(spread_flags)

    result.flexion_deg = flexion
    result.spread_deg = spread
    result.palm_rotation = frame.rotation
    result.palm_quaternion = quaternion
    result.valid = bool(
        np.isfinite(flexion).all()
        and np.isfinite(spread).all()
        and np.isfinite(frame.rotation).all()
        and np.isfinite(quaternion).all()
    )
    return result


@dataclass(slots=True)
class SequenceKinematics:
    """Kinematics for one whole video, in the fixed output contract shapes."""

    sample_id: str
    frame_index: np.ndarray
    timestamp_seconds: np.ndarray
    tracking_state_code: np.ndarray
    source_raw_detection_index: np.ndarray
    valid_kinematics: np.ndarray
    valid_palm_frame: np.ndarray
    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    kinematic_flags_json: np.ndarray


def extract_sequence(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    sample_id: str,
) -> SequenceKinematics:
    """Derive kinematics for every frame and both tracks of one tracked video.

    ``arrays``/``metadata`` are exactly what ``tracking.wilor.load_tracked_sequence``
    returns. The tracked input is never written to.
    """

    import json

    code_to_state = {int(code): name for name, code in metadata["state_codes"].items()}
    track_order = tuple(metadata.get("track_order", TRACK_ORDER))
    if track_order != TRACK_ORDER:
        raise ValueError(
            f"{sample_id}: unexpected track order {track_order}; "
            f"TASK-005A preserves the TASK-004 order {TRACK_ORDER}"
        )

    frame_index = np.asarray(arrays["frame_index"], dtype=np.int32)
    frames = int(frame_index.shape[0])

    valid = np.zeros((frames, N_TRACKS), dtype=bool)
    palm_valid = np.zeros((frames, N_TRACKS), dtype=bool)
    flexion = np.full((frames, N_TRACKS, N_FINGERS, N_CHAIN), np.nan, dtype=np.float32)
    spread = np.full((frames, N_TRACKS, N_SPREAD), np.nan, dtype=np.float32)
    rotation = np.full((frames, N_TRACKS, 3, 3), np.nan, dtype=np.float32)
    quaternion = np.full((frames, N_TRACKS, 4), np.nan, dtype=np.float32)
    flags_json = np.empty((frames, N_TRACKS), dtype=object)

    landmarks = arrays["landmarks_3d"]
    state_code = np.asarray(arrays["state_code"], dtype=np.int32)
    raw_index = np.asarray(arrays["raw_detection_index"], dtype=np.int32)

    for row in range(frames):
        for column, track in enumerate(track_order):
            state = code_to_state[int(state_code[row, column])]
            hand = compute_hand_kinematics(landmarks[row, column], track, state)
            valid[row, column] = hand.valid
            palm_valid[row, column] = hand.palm_frame_valid
            flexion[row, column] = hand.flexion_deg.astype(np.float32)
            spread[row, column] = hand.spread_deg.astype(np.float32)
            rotation[row, column] = hand.palm_rotation.astype(np.float32)
            quaternion[row, column] = hand.palm_quaternion.astype(np.float32)
            flags_json[row, column] = json.dumps(hand.flags)

    return SequenceKinematics(
        sample_id=sample_id,
        frame_index=frame_index,
        timestamp_seconds=np.asarray(arrays["timestamp_seconds"], dtype=np.float64),
        tracking_state_code=state_code,
        source_raw_detection_index=raw_index,
        valid_kinematics=valid,
        valid_palm_frame=palm_valid,
        flexion_deg=flexion,
        adjacent_spread_deg=spread,
        palm_rotation_matrix=rotation,
        palm_quaternion_wxyz=quaternion,
        kinematic_flags_json=np.array(flags_json.tolist(), dtype=np.str_),
    )
