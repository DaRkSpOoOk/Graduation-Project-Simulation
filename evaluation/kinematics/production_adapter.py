"""Neutral adapter from TASK-005B fixtures to frozen TASK-005A math.

The adapter deliberately performs no kinematic calculation of its own.  It
only supplies each synthetic 21-joint frame to ``kinematics`` and maps the
returned field names into TASK-005B's result contract.  The production result
is returned even when it contains NaN channels so partial-validity and
conditioning behaviour can be audited rather than hidden by validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics import compute_hand_kinematics

from .benchmark_contract import KinematicsResult
from .synthetic_hand import SyntheticSequence


PRODUCTION_TRACK_ORDER = ("left", "right")
POSE_STATE = "OBSERVED"


@dataclass(frozen=True)
class ProductionSyntheticResult:
    """Production output plus validity/flag provenance for one fixture."""

    result: KinematicsResult
    valid_kinematics: np.ndarray
    valid_palm_frame: np.ndarray
    flags: tuple[tuple[tuple[str, ...], ...], ...]

    @property
    def all_channels_finite(self) -> bool:
        """Whether all four public numeric channels are finite."""

        return bool(
            np.isfinite(self.result.flexion_deg).all()
            and np.isfinite(self.result.adjacent_spread_deg).all()
            and np.isfinite(self.result.palm_rotation_matrix).all()
            and np.isfinite(self.result.palm_quaternion_wxyz).all()
        )


def extract_production_sequence(
    sequence: SyntheticSequence,
    *,
    state: str = POSE_STATE,
) -> ProductionSyntheticResult:
    """Run frozen TASK-005A per-frame math on a TASK-005B sequence.

    ``state`` is fixed to ``OBSERVED`` by the validation harness for valid
    synthetic geometry.  It is an argument so invalid-state behaviour remains
    explicit in tests and no tracking state is invented by this adapter.
    """

    joints = np.asarray(sequence.joints, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"expected [F,2,21,3] synthetic joints, got {joints.shape}")
    if tuple(sequence.sides) != ("LEFT", "RIGHT"):
        raise ValueError(f"unexpected benchmark track order: {sequence.sides!r}")

    flexion: list[list[np.ndarray]] = []
    spread: list[list[np.ndarray]] = []
    rotation: list[list[np.ndarray]] = []
    quaternion: list[list[np.ndarray]] = []
    valid: list[list[bool]] = []
    palm_valid: list[list[bool]] = []
    flags: list[tuple[tuple[str, ...], ...]] = []

    for frame in range(joints.shape[0]):
        flexion_row: list[np.ndarray] = []
        spread_row: list[np.ndarray] = []
        rotation_row: list[np.ndarray] = []
        quaternion_row: list[np.ndarray] = []
        valid_row: list[bool] = []
        palm_valid_row: list[bool] = []
        flags_row: list[tuple[str, ...]] = []
        for track, side in enumerate(PRODUCTION_TRACK_ORDER):
            hand = compute_hand_kinematics(joints[frame, track], side, state)
            flexion_row.append(np.asarray(hand.flexion_deg, dtype=np.float64).copy())
            spread_row.append(np.asarray(hand.spread_deg, dtype=np.float64).copy())
            rotation_row.append(np.asarray(hand.palm_rotation, dtype=np.float64).copy())
            quaternion_row.append(np.asarray(hand.palm_quaternion, dtype=np.float64).copy())
            valid_row.append(bool(hand.valid))
            palm_valid_row.append(bool(hand.palm_frame_valid))
            flags_row.append(tuple(hand.flags))
        flexion.append(flexion_row)
        spread.append(spread_row)
        rotation.append(rotation_row)
        quaternion.append(quaternion_row)
        valid.append(valid_row)
        palm_valid.append(palm_valid_row)
        flags.append(tuple(flags_row))

    return ProductionSyntheticResult(
        result=KinematicsResult(
            flexion_deg=np.asarray(flexion, dtype=np.float64),
            adjacent_spread_deg=np.asarray(spread, dtype=np.float64),
            palm_rotation_matrix=np.asarray(rotation, dtype=np.float64),
            palm_quaternion_wxyz=np.asarray(quaternion, dtype=np.float64),
        ),
        valid_kinematics=np.asarray(valid, dtype=bool),
        valid_palm_frame=np.asarray(palm_valid, dtype=bool),
        flags=tuple(flags),
    )
