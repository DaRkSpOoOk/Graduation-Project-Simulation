"""Deterministic TASK-005-like fixtures for the TASK-006B benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import (
    CHAIN_ORDER,
    FINGER_ORDER,
    KinematicsInput,
    SPREAD_ORDER,
    TRACK_ORDER,
)
from .orientation import quaternion_wxyz_from_matrix, rotation_matrix_axis, rotation_matrix_xyz


KNOWN_ANGLE_VALUES_DEG = (0.0, 1.0, 30.0, 45.0, 90.0, 135.0, 179.0, 180.0)


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    """One valid or intentionally corrupted benchmark input."""

    fixture_id: str
    category: str
    description: str
    data: KinematicsInput
    expected_valid: bool = True
    expected_error: str | None = None
    coverage_kind: str | None = None
    coverage_key: tuple[str, ...] | None = None
    coverage_angle_deg: float | None = None


def _expand_array(
    value: object,
    shape: tuple[int, ...],
    *,
    broadcast_shapes: tuple[tuple[int, ...], ...] = (),
    dtype: Any = np.float64,
) -> np.ndarray:
    values = np.asarray(value, dtype=dtype)
    if values.shape == shape:
        return values.copy()
    for source_shape in broadcast_shapes:
        if values.shape == source_shape:
            candidate = values
            # A frame-major value such as [F,5,3] is shared by both hands;
            # insert the hand axis before broadcasting to [F,2,5,3].
            if values.ndim + 1 == len(shape) and values.shape[0] == shape[0]:
                candidate = values[:, None, ...]
            return np.broadcast_to(candidate, shape).copy()
    raise ValueError(f"cannot expand {values.shape} to {shape}")


def make_input(
    *,
    frame_count: int = 1,
    flexion_deg: object | None = None,
    adjacent_spread_deg: object | None = None,
    palm_rotation_matrix: object | None = None,
    palm_quaternion_wxyz: object | None = None,
    timestamps_seconds: object | None = None,
    frame_index: object | None = None,
    hand_present: object | None = None,
    valid_palm_frame: object | None = None,
    valid_kinematics: object | None = None,
    source_provenance: tuple[str, ...] | None = None,
    track_order: tuple[str, ...] = TRACK_ORDER,
) -> KinematicsInput:
    """Create a shape-correct synthetic input without calling production code."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    flexion = np.zeros((frame_count, 2, 5, 3), dtype=np.float64)
    if flexion_deg is not None:
        flexion = _expand_array(
            flexion_deg,
            flexion.shape,
            broadcast_shapes=((5, 3), (frame_count, 5, 3)),
        )
    spread = np.zeros((frame_count, 2, 4), dtype=np.float64)
    if adjacent_spread_deg is not None:
        spread = _expand_array(
            adjacent_spread_deg,
            spread.shape,
            broadcast_shapes=((4,), (frame_count, 4)),
        )

    identity = np.eye(3, dtype=np.float64)
    rotations = _expand_array(
        identity if palm_rotation_matrix is None else palm_rotation_matrix,
        (frame_count, 2, 3, 3),
        broadcast_shapes=((3, 3), (frame_count, 3, 3)),
    )
    if palm_quaternion_wxyz is None:
        quaternions = np.asarray(
            [
                [quaternion_wxyz_from_matrix(rotations[frame, hand]) for hand in range(2)]
                for frame in range(frame_count)
            ],
            dtype=np.float64,
        )
    else:
        quaternions = _expand_array(
            palm_quaternion_wxyz,
            (frame_count, 2, 4),
            broadcast_shapes=((4,), (frame_count, 4)),
        )

    timestamps = (
        np.arange(frame_count, dtype=np.float64)
        if timestamps_seconds is None
        else np.asarray(timestamps_seconds, dtype=np.float64).copy()
    )
    indices = (
        np.arange(frame_count, dtype=np.int64)
        if frame_index is None
        else np.asarray(frame_index).copy()
    )
    present = np.ones((frame_count, 2), dtype=bool)
    if hand_present is not None:
        present = _expand_array(
            hand_present,
            present.shape,
            broadcast_shapes=((2,),),
            dtype=bool,
        )
    palm_valid = np.ones((frame_count, 2), dtype=bool)
    if valid_palm_frame is not None:
        palm_valid = _expand_array(
            valid_palm_frame,
            palm_valid.shape,
            broadcast_shapes=((2,),),
            dtype=bool,
        )
    if valid_kinematics is None:
        strict_valid = (
            present
            & palm_valid
            & np.isfinite(flexion).all(axis=(2, 3))
            & np.isfinite(spread).all(axis=2)
        )
    else:
        strict_valid = _expand_array(
            valid_kinematics,
            present.shape,
            broadcast_shapes=((2,),),
            dtype=bool,
        )
    if source_provenance is None:
        provenance = tuple(f"synthetic://task006b/frame/{frame}" for frame in range(frame_count))
    else:
        provenance = source_provenance

    return KinematicsInput(
        flexion_deg=flexion,
        adjacent_spread_deg=spread,
        palm_rotation_matrix=rotations,
        palm_quaternion_wxyz=quaternions,
        timestamps_seconds=timestamps,
        frame_index=indices,
        hand_present=present,
        valid_palm_frame=palm_valid,
        valid_kinematics=strict_valid,
        source_provenance=provenance,
        track_order=tuple(track_order),
    )


def copy_input(data: KinematicsInput, **updates: object) -> KinematicsInput:
    """Copy a synthetic input before applying one intentional corruption."""

    fields: dict[str, object] = {
        "flexion_deg": np.asarray(data.flexion_deg).copy(),
        "adjacent_spread_deg": np.asarray(data.adjacent_spread_deg).copy(),
        "palm_rotation_matrix": np.asarray(data.palm_rotation_matrix).copy(),
        "palm_quaternion_wxyz": np.asarray(data.palm_quaternion_wxyz).copy(),
        "timestamps_seconds": np.asarray(data.timestamps_seconds).copy(),
        "frame_index": np.asarray(data.frame_index).copy(),
        "hand_present": np.asarray(data.hand_present).copy(),
        "valid_palm_frame": np.asarray(data.valid_palm_frame).copy(),
        "valid_kinematics": np.asarray(data.valid_kinematics).copy(),
        "source_provenance": data.source_provenance,
        "track_order": data.track_order,
        "schema_version": data.schema_version,
    }
    fields.update(updates)
    return KinematicsInput(**fields)


def _fixture(
    fixture_id: str,
    category: str,
    description: str,
    data: KinematicsInput,
    *,
    coverage_kind: str | None = None,
    coverage_key: tuple[str, ...] | None = None,
    coverage_angle_deg: float | None = None,
) -> SyntheticFixture:
    return SyntheticFixture(
        fixture_id=fixture_id,
        category=category,
        description=description,
        data=data,
        coverage_kind=coverage_kind,
        coverage_key=coverage_key,
        coverage_angle_deg=coverage_angle_deg,
    )


def build_valid_fixtures() -> tuple[SyntheticFixture, ...]:
    """Return deterministic valid coverage, invariance and validity cases."""

    fixtures: list[SyntheticFixture] = []
    fixtures.append(
        _fixture(
            "neutral",
            "neutral",
            "All 15 bend and 4 spread channels at zero with identity palm orientation.",
            make_input(),
        )
    )

    for finger_index, finger in enumerate(FINGER_ORDER):
        for joint_index, joint in enumerate(CHAIN_ORDER):
            for angle in KNOWN_ANGLE_VALUES_DEG:
                flexion = np.zeros((5, 3), dtype=np.float64)
                flexion[finger_index, joint_index] = angle
                fixtures.append(
                    _fixture(
                        f"bend_{finger}_{joint}_{int(angle):03d}deg",
                        "single_bend",
                        f"Only {finger} {joint} bend is {angle:g} degrees.",
                        make_input(flexion_deg=flexion),
                        coverage_kind="bend",
                        coverage_key=(finger, joint),
                        coverage_angle_deg=angle,
                    )
                )

    for spread_index, spread_name in enumerate(SPREAD_ORDER):
        for angle in KNOWN_ANGLE_VALUES_DEG:
            spreads = np.zeros(4, dtype=np.float64)
            spreads[spread_index] = angle
            fixtures.append(
                _fixture(
                    f"spread_{spread_name.replace('-', '_')}_{int(angle):03d}deg",
                    "single_spread",
                    f"Only {spread_name} spread is {angle:g} degrees.",
                    make_input(adjacent_spread_deg=spreads),
                    coverage_kind="spread",
                    coverage_key=(spread_name,),
                    coverage_angle_deg=angle,
                )
            )

    all_flexion = np.asarray(
        [[0.0, 30.0, 60.0], [1.0, 45.0, 90.0], [30.0, 90.0, 135.0], [45.0, 135.0, 179.0], [90.0, 179.0, 180.0]],
        dtype=np.float64,
    )
    fixtures.append(
        _fixture(
            "multi_all_channels",
            "multi_channel",
            "Every bend channel and every spread channel carry distinct in-contract values.",
            make_input(flexion_deg=all_flexion, adjacent_spread_deg=(5.0, 45.0, 90.0, 135.0)),
        )
    )

    side_rotations = np.asarray(
        [[rotation_matrix_axis("X", 25.0), rotation_matrix_axis("Y", -35.0)]],
        dtype=np.float64,
    )
    side_quaternions = np.asarray(
        [[quaternion_wxyz_from_matrix(side_rotations[0, hand]) for hand in range(2)]],
        dtype=np.float64,
    )
    fixtures.append(
        _fixture(
            "left_right_equivalent_local_values",
            "mirror",
            "LEFT and RIGHT have equivalent local Hall values but distinct proper palm orientations.",
            make_input(
                flexion_deg=all_flexion,
                adjacent_spread_deg=(10.0, 20.0, 30.0, 40.0),
                palm_rotation_matrix=side_rotations,
                palm_quaternion_wxyz=side_quaternions,
            ),
        )
    )

    sequence_flexion = np.asarray(
        [
            np.zeros((5, 3)),
            all_flexion,
            np.full((5, 3), 180.0),
        ],
        dtype=np.float64,
    )
    sequence_spread = np.asarray(
        [[0.0, 5.0, 10.0, 15.0], [45.0, 90.0, 135.0, 179.0], [180.0, 135.0, 90.0, 45.0]],
        dtype=np.float64,
    )
    sequence_rotations = np.asarray(
        [
            [rotation_matrix_axis("Z", 0.0), rotation_matrix_axis("Z", 0.0)],
            [rotation_matrix_axis("Z", 45.0), rotation_matrix_axis("Z", 45.0)],
            [rotation_matrix_axis("Z", 90.0), rotation_matrix_axis("Z", 90.0)],
        ],
        dtype=np.float64,
    )
    sequence_quaternions = np.asarray(
        [
            [quaternion_wxyz_from_matrix(sequence_rotations[frame, hand]) for hand in range(2)]
            for frame in range(3)
        ],
        dtype=np.float64,
    )
    fixtures.append(
        _fixture(
            "multi_frame_known_sequence",
            "multi_channel",
            "Three timestamped frames exercise simultaneous bend, spread and orientation changes.",
            make_input(
                frame_count=3,
                flexion_deg=sequence_flexion,
                adjacent_spread_deg=sequence_spread,
                palm_rotation_matrix=sequence_rotations,
                palm_quaternion_wxyz=sequence_quaternions,
                timestamps_seconds=(0.0, 0.5, 1.0),
                frame_index=(0, 1, 2),
            ),
        )
    )

    orientation_cases = (
        ("identity", np.eye(3, dtype=np.float64)),
        ("x90", rotation_matrix_axis("X", 90.0)),
        ("y90", rotation_matrix_axis("Y", 90.0)),
        ("z90", rotation_matrix_axis("Z", 90.0)),
        ("x180", rotation_matrix_axis("X", 180.0)),
        ("y180", rotation_matrix_axis("Y", 180.0)),
        ("z180", rotation_matrix_axis("Z", 180.0)),
        ("composed", rotation_matrix_xyz(25.0, -40.0, 70.0)),
    )
    for orientation_name, matrix in orientation_cases:
        fixtures.append(
            _fixture(
                f"orientation_{orientation_name}",
                "orientation",
                f"Known {orientation_name} proper palm rotation with direct IMU passthrough.",
                make_input(
                    flexion_deg=all_flexion,
                    adjacent_spread_deg=(5.0, 10.0, 20.0, 30.0),
                    palm_rotation_matrix=matrix,
                ),
            )
        )

    partial_spread = np.full((1, 2, 4), 45.0, dtype=np.float64)
    partial_spread[0, 0, 2] = np.nan
    partial_data = make_input(
        flexion_deg=np.full((5, 3), 60.0),
        adjacent_spread_deg=partial_spread,
    )
    fixtures.append(
        _fixture(
            "validity_partial_spread",
            "validity",
            "LEFT middle-ring spread is undefined while all bends and both IMUs remain usable.",
            partial_data,
        )
    )

    invalid_palm_rotation = np.full((1, 2, 3, 3), np.nan, dtype=np.float64)
    invalid_palm_rotation[0, 1] = np.eye(3)
    invalid_palm_quaternion = np.full((1, 2, 4), np.nan, dtype=np.float64)
    invalid_palm_quaternion[0, 1] = quaternion_wxyz_from_matrix(np.eye(3))
    palm_invalid_data = make_input(
        flexion_deg=np.full((5, 3), 30.0),
        adjacent_spread_deg=np.full(4, 30.0),
        palm_rotation_matrix=invalid_palm_rotation,
        palm_quaternion_wxyz=invalid_palm_quaternion,
        valid_palm_frame=((False, True),),
    )
    fixtures.append(
        _fixture(
            "validity_whole_palm_invalid",
            "validity",
            "LEFT palm orientation is unavailable but finite Hall channels are preserved; RIGHT is complete.",
            palm_invalid_data,
        )
    )

    missing_flexion = np.full((1, 2, 5, 3), np.nan, dtype=np.float64)
    missing_spread = np.full((1, 2, 4), np.nan, dtype=np.float64)
    missing_rotation = np.full((1, 2, 3, 3), np.nan, dtype=np.float64)
    missing_quaternion = np.full((1, 2, 4), np.nan, dtype=np.float64)
    missing_flexion[0, 0] = 45.0
    missing_spread[0, 0] = 45.0
    missing_rotation[0, 0] = np.eye(3)
    missing_quaternion[0, 0] = quaternion_wxyz_from_matrix(np.eye(3))
    fixtures.append(
        _fixture(
            "validity_missing_tracking_pose",
            "validity",
            "RIGHT pose is absent and carries no fabricated finite values.",
            make_input(
                flexion_deg=missing_flexion,
                adjacent_spread_deg=missing_spread,
                palm_rotation_matrix=missing_rotation,
                palm_quaternion_wxyz=missing_quaternion,
                hand_present=((True, False),),
                valid_palm_frame=((True, False),),
                valid_kinematics=((True, False),),
            ),
        )
    )
    return tuple(fixtures)


def build_invalid_fixtures() -> tuple[SyntheticFixture, ...]:
    """Return explicit corruptions that must hard-fail validation."""

    one_frame = make_input(flexion_deg=np.full((5, 3), 30.0), adjacent_spread_deg=np.full(4, 30.0))
    two_frames = make_input(
        frame_count=2,
        timestamps_seconds=(0.0, 1.0),
        frame_index=(0, 1),
    )
    invalid: list[SyntheticFixture] = []

    flexion_negative = np.asarray(one_frame.flexion_deg).copy()
    flexion_negative[0, 0, 0, 0] = -1.0
    invalid.append(
        SyntheticFixture(
            "invalid_bend_below_zero",
            "invalid_input",
            "Bend degree is below the contract range.",
            copy_input(one_frame, flexion_deg=flexion_negative),
            expected_valid=False,
            expected_error="flexion_deg_outside_0_180",
        )
    )
    flexion_above = np.asarray(one_frame.flexion_deg).copy()
    flexion_above[0, 1, 4, 2] = 181.0
    invalid.append(
        SyntheticFixture(
            "invalid_bend_above_180",
            "invalid_input",
            "Bend degree is above the contract range.",
            copy_input(one_frame, flexion_deg=flexion_above),
            expected_valid=False,
            expected_error="flexion_deg_outside_0_180",
        )
    )
    spread_negative = np.asarray(one_frame.adjacent_spread_deg).copy()
    spread_negative[0, 0, 0] = -1.0
    invalid.append(
        SyntheticFixture(
            "invalid_spread_below_zero",
            "invalid_input",
            "Spread degree is below the contract range.",
            copy_input(one_frame, adjacent_spread_deg=spread_negative),
            expected_valid=False,
            expected_error="adjacent_spread_deg_outside_0_180",
        )
    )
    spread_above = np.asarray(one_frame.adjacent_spread_deg).copy()
    spread_above[0, 1, 3] = 181.0
    invalid.append(
        SyntheticFixture(
            "invalid_spread_above_180",
            "invalid_input",
            "Spread degree is above the contract range.",
            copy_input(one_frame, adjacent_spread_deg=spread_above),
            expected_valid=False,
            expected_error="adjacent_spread_deg_outside_0_180",
        )
    )
    invalid.append(
        SyntheticFixture(
            "invalid_malformed_shape",
            "invalid_input",
            "Spread array has the wrong final channel dimension.",
            copy_input(one_frame, adjacent_spread_deg=np.asarray(one_frame.adjacent_spread_deg)[:, :, :3]),
            expected_valid=False,
            expected_error="adjacent_spread_deg_shape",
        )
    )
    invalid.append(
        SyntheticFixture(
            "invalid_non_monotonic_timestamps",
            "invalid_input",
            "Timestamp sequence decreases between frames.",
            copy_input(two_frames, timestamps_seconds=np.asarray((1.0, 0.0))),
            expected_valid=False,
            expected_error="timestamps_non_monotonic",
        )
    )
    invalid.append(
        SyntheticFixture(
            "invalid_duplicate_frame_index",
            "invalid_input",
            "Two rows carry the same frame index.",
            copy_input(two_frames, frame_index=np.asarray((0, 0), dtype=np.int64)),
            expected_valid=False,
            expected_error="frame_index_duplicate",
        )
    )
    nonfinite_rotation = np.asarray(one_frame.palm_rotation_matrix).copy()
    nonfinite_rotation[0, 0, 0, 0] = np.nan
    invalid.append(
        SyntheticFixture(
            "invalid_nonfinite_valid_orientation",
            "invalid_input",
            "A palm matrix marked valid contains NaN.",
            copy_input(one_frame, palm_rotation_matrix=nonfinite_rotation),
            expected_valid=False,
            expected_error="invalid_palm_rotation",
        )
    )
    invalid_quaternion = np.asarray(one_frame.palm_quaternion_wxyz).copy()
    invalid_quaternion[0, 1] *= 2.0
    invalid.append(
        SyntheticFixture(
            "invalid_quaternion_norm",
            "invalid_input",
            "A valid palm quaternion is not normalized.",
            copy_input(one_frame, palm_quaternion_wxyz=invalid_quaternion),
            expected_valid=False,
            expected_error="invalid_palm_quaternion",
        )
    )
    invalid.append(
        SyntheticFixture(
            "invalid_track_order",
            "invalid_input",
            "Track order is reversed.",
            copy_input(one_frame, track_order=("RIGHT", "LEFT")),
            expected_valid=False,
            expected_error="wrong_track_order",
        )
    )
    invalid.append(
        SyntheticFixture(
            "invalid_missing_provenance",
            "invalid_input",
            "Source provenance is absent.",
            copy_input(one_frame, source_provenance=None),
            expected_valid=False,
            expected_error="missing_or_malformed_provenance",
        )
    )
    finite_no_pose = np.asarray(one_frame.hand_present).copy()
    finite_no_pose[0, 1] = False
    invalid.append(
        SyntheticFixture(
            "invalid_missing_pose_with_finite_values",
            "invalid_input",
            "A missing pose incorrectly retains finite derived values.",
            copy_input(
                one_frame,
                hand_present=finite_no_pose,
                valid_palm_frame=((True, False),),
                valid_kinematics=((False, False),),
            ),
            expected_valid=False,
            expected_error="missing_pose_has_finite_derived",
        )
    )
    return tuple(invalid)


def build_fixture_catalog() -> tuple[SyntheticFixture, ...]:
    """Return valid and invalid fixtures in stable construction order."""

    return build_valid_fixtures() + build_invalid_fixtures()


__all__ = [
    "KNOWN_ANGLE_VALUES_DEG",
    "SyntheticFixture",
    "build_fixture_catalog",
    "build_invalid_fixtures",
    "build_valid_fixtures",
    "copy_input",
    "make_input",
]
