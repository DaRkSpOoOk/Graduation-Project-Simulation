"""Versioned TASK-005 final contract and independent benchmark truth.

TASK-005B and TASK-005D are historical records.  This module is the explicit
post-integration contract selected for TASK-005E1.  It reuses the frozen
TASK-005B *fixture catalog* but derives every expected geometric channel from
the generated 21-joint coordinates.  It does not import the production
``kinematics`` package or call production formulas.

The orientation matrices retained on a fixture are the fixture's declared
analytic basis.  A side-specific, fixed comparison mapping is provided for
the production frame; it is an integration convention and never overwrites a
raw production matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .benchmark_contract import CONTRACT_TOLERANCES
from .synthetic_hand import (
    BenchmarkCase,
    SyntheticSequence,
    build_benchmark_catalog,
    quaternion_wxyz_from_matrix,
)


FINAL_CONTRACT_VERSION = "TASK-005-final-v2"
FINAL_TRACK_ORDER = ("LEFT", "RIGHT")
FINAL_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINAL_CHAIN_ORDER = ("proximal", "middle", "distal")
FINAL_SPREAD_ORDER = ("thumb-index", "index-middle", "middle-ring", "ring-pinky")

# This is the fixed 21-joint layout, repeated locally so the benchmark remains
# independent from kinematics/layout.py.  The first bend is deliberately
# defined using wrist -> base because no metacarpal-shaft joint is observed.
FINAL_FINGER_CHAINS: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINAL_SPREAD_SEGMENTS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (5, 6),
    (9, 10),
    (13, 14),
    (17, 18),
)
FINAL_SPREAD_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 4))

# The projection threshold is a contract choice, not a fitted parameter.  A
# unit direction at 15 degrees from the normal has a projected norm of sin(15)
#; below it the in-plane direction is considered undefined.
FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG = 15.0
FINAL_SPREAD_MIN_PROJECTED_NORM = float(
    math.sin(math.radians(FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG))
)
FINAL_GEOMETRY_EPSILON = 1e-8

# Keep the original TASK-005B numerical limits byte-for-byte in meaning.  The
# revised contract changes definitions and masks, not tolerance strictness.
FINAL_CONTRACT_TOLERANCES = dict(CONTRACT_TOLERANCES)

# TASK-005B's generator declares these side-local bases.  They remain the
# independent fixture orientation truth.  The following fixed matrices map
# the frozen TASK-005A [lateral | palmar normal | distal] frame into that
# fixture convention for comparison.  They were derived from the literal
# wrist/MCP fixture coordinates and the documented frame axes, not fitted to
# any result sample.
FIXTURE_BASIS_RIGHT = np.eye(3, dtype=np.float64)
FIXTURE_BASIS_LEFT = np.diag((-1.0, 1.0, -1.0)).astype(np.float64)
FROZEN_SYNTHETIC_PRODUCTION_BASIS = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SIDE_ORIENTATION_MAPPINGS = {
    "RIGHT": FROZEN_SYNTHETIC_PRODUCTION_BASIS.T @ FIXTURE_BASIS_RIGHT,
    "LEFT": FROZEN_SYNTHETIC_PRODUCTION_BASIS.T @ FIXTURE_BASIS_LEFT,
}


@dataclass(frozen=True, slots=True)
class FinalSyntheticSequence:
    """TASK-005-final-v2 truth for one frozen catalog case."""

    case_id: str
    joints: np.ndarray
    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    expected_valid: bool
    invalid_reasons: tuple[str, ...]
    valid_palm_frame: np.ndarray
    valid_kinematics: np.ndarray
    spread_direction_degenerate: np.ndarray
    sides: tuple[str, str] = FINAL_TRACK_ORDER


def _unit(vector: object, *, minimum: float = FINAL_GEOMETRY_EPSILON) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        return None
    norm = float(np.linalg.norm(values))
    if norm < minimum:
        return None
    return values / norm


def _unsigned_angle(first: object, second: object) -> float | None:
    """Analytic unsigned angle in [0, 180] using atan2 geometry."""

    first_unit = _unit(first)
    second_unit = _unit(second)
    if first_unit is None or second_unit is None:
        return None
    cross_norm = float(np.linalg.norm(np.cross(first_unit, second_unit)))
    dot = float(np.dot(first_unit, second_unit))
    return float(np.degrees(np.arctan2(cross_norm, dot)))


def _palm_normal_from_output_geometry(joints: np.ndarray) -> np.ndarray | None:
    """Return a normal from wrist/index-MCP/middle-MCP output geometry.

    The sign is immaterial for an unsigned projected angle.  Deriving the
    plane from the output points is important: the old fixture's orientation
    basis is a coordinate convention, while these three landmarks define the
    actual synthetic palm plane.
    """

    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    return _unit(np.cross(index_mcp - wrist, middle_mcp - wrist))


def _derive_flexion(joints: np.ndarray) -> np.ndarray:
    values = np.full((5, 3), np.nan, dtype=np.float64)
    for finger, chain in enumerate(FINAL_FINGER_CHAINS):
        for joint in range(3):
            value = _unsigned_angle(
                joints[chain[joint + 1]] - joints[chain[joint]],
                joints[chain[joint + 2]] - joints[chain[joint + 1]],
            )
            if value is not None:
                values[finger, joint] = value
    return values


def _derive_spread(
    joints: np.ndarray,
    palm_normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((4,), np.nan, dtype=np.float64)
    degenerate = np.zeros((5,), dtype=bool)
    if palm_normal is None:
        degenerate[:] = True
        return values, degenerate

    directions: list[np.ndarray | None] = []
    for finger, (start, end) in enumerate(FINAL_SPREAD_SEGMENTS):
        raw = _unit(joints[end] - joints[start])
        if raw is None:
            degenerate[finger] = True
            directions.append(None)
            continue
        projected = raw - float(np.dot(raw, palm_normal)) * palm_normal
        projected_norm = float(np.linalg.norm(projected))
        if not np.isfinite(projected_norm) or projected_norm < FINAL_SPREAD_MIN_PROJECTED_NORM:
            degenerate[finger] = True
            directions.append(None)
            continue
        directions.append(projected / projected_norm)

    for spread_index, (first, second) in enumerate(FINAL_SPREAD_PAIRS):
        if directions[first] is None or directions[second] is None:
            continue
        value = _unsigned_angle(directions[first], directions[second])
        if value is not None:
            values[spread_index] = value
    return values, degenerate


def build_final_sequence(case: BenchmarkCase) -> FinalSyntheticSequence:
    """Build final-v2 truth from a frozen case's generated joint coordinates."""

    source: SyntheticSequence = case.generate()
    frames = int(source.joints.shape[0])
    flexion = np.full((frames, 2, 5, 3), np.nan, dtype=np.float64)
    spread = np.full((frames, 2, 4), np.nan, dtype=np.float64)
    palm_valid = np.zeros((frames, 2), dtype=bool)
    spread_degenerate = np.zeros((frames, 2, 5), dtype=bool)

    for frame in range(frames):
        for track in range(2):
            joints = np.asarray(source.joints[frame, track], dtype=np.float64)
            normal = _palm_normal_from_output_geometry(joints)
            palm_valid[frame, track] = normal is not None
            flexion[frame, track] = _derive_flexion(joints)
            spread[frame, track], spread_degenerate[frame, track] = _derive_spread(joints, normal)

    # The fixture's known orientation is retained as independent analytic
    # truth. It is not recalculated from production output.
    palm_rotation = np.asarray(source.palm_rotation_matrix, dtype=np.float64).copy()
    palm_quaternion = np.asarray(source.palm_quaternion_wxyz, dtype=np.float64).copy()
    valid_kinematics = (
        np.asarray(palm_valid, dtype=bool)
        & np.isfinite(flexion).all(axis=(2, 3))
        & np.isfinite(spread).all(axis=2)
        & np.isfinite(palm_rotation).all(axis=(2, 3))
        & np.isfinite(palm_quaternion).all(axis=2)
    )
    if not source.expected_valid:
        # Invalid geometry is never promoted to a valid benchmark hand even
        # if an individual derived channel happens to remain finite.
        valid_kinematics[:] = False

    return FinalSyntheticSequence(
        case_id=source.case_id,
        joints=np.asarray(source.joints, dtype=np.float64).copy(),
        flexion_deg=flexion,
        adjacent_spread_deg=spread,
        palm_rotation_matrix=palm_rotation,
        palm_quaternion_wxyz=palm_quaternion,
        expected_valid=bool(source.expected_valid),
        invalid_reasons=tuple(source.invalid_reasons),
        valid_palm_frame=palm_valid,
        valid_kinematics=valid_kinematics,
        spread_direction_degenerate=spread_degenerate,
        sides=FINAL_TRACK_ORDER,
    )


def build_final_catalog() -> tuple[BenchmarkCase, ...]:
    """Return the frozen 86-case catalog without changing its history."""

    return build_benchmark_catalog()


def orientation_mapping_for_side(side: str) -> np.ndarray:
    normalized = str(side).upper()
    try:
        return SIDE_ORIENTATION_MAPPINGS[normalized].copy()
    except KeyError as error:
        raise ValueError(f"side must be LEFT or RIGHT, got {side!r}") from error


def map_production_rotation(rotation: object, side: str) -> np.ndarray:
    """Map a production frame to the fixed fixture convention for scoring."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in (3,3), got {matrix.shape}")
    return np.matmul(matrix, orientation_mapping_for_side(side))


def map_production_sequence_rotations(rotations: object) -> np.ndarray:
    """Apply the fixed LEFT/RIGHT comparison mappings to [F,2,3,3] data."""

    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (2, 3, 3):
        raise ValueError(f"rotations must have shape [F,2,3,3], got {values.shape}")
    mapped = np.empty_like(values)
    mapped[:, 0] = map_production_rotation(values[:, 0], "LEFT")
    mapped[:, 1] = map_production_rotation(values[:, 1], "RIGHT")
    return mapped


def mapped_quaternion_wxyz(rotation: object) -> np.ndarray:
    """Return independent [w,x,y,z] truth for a mapped rotation matrix."""

    quaternion = quaternion_wxyz_from_matrix(np.asarray(rotation, dtype=np.float64))
    return np.asarray(quaternion, dtype=np.float64)


__all__ = [
    "FINAL_CONTRACT_VERSION",
    "FINAL_TRACK_ORDER",
    "FINAL_FINGER_ORDER",
    "FINAL_CHAIN_ORDER",
    "FINAL_SPREAD_ORDER",
    "FINAL_FINGER_CHAINS",
    "FINAL_SPREAD_SEGMENTS",
    "FINAL_SPREAD_PAIRS",
    "FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG",
    "FINAL_SPREAD_MIN_PROJECTED_NORM",
    "FINAL_GEOMETRY_EPSILON",
    "FINAL_CONTRACT_TOLERANCES",
    "FIXTURE_BASIS_RIGHT",
    "FIXTURE_BASIS_LEFT",
    "FROZEN_SYNTHETIC_PRODUCTION_BASIS",
    "SIDE_ORIENTATION_MAPPINGS",
    "FinalSyntheticSequence",
    "build_final_catalog",
    "build_final_sequence",
    "map_production_rotation",
    "map_production_sequence_rotations",
    "mapped_quaternion_wxyz",
    "orientation_mapping_for_side",
]
