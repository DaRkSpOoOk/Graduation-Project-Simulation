"""Convert frozen TASK-005 kinematics into ideal virtual-glove sensor output.

Per-frame and per-channel. Nothing is interpolated, forward filled, smoothed,
copied between hands, or invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .imu import angular_velocity_body_frame
from .layout import (
    CHAIN_ORDER,
    FINGER_ORDER,
    SPREAD_PAIRS,
    TRACK_ORDER,
)
from .signals import (
    describe_violations,
    normalize_angles,
    to_adc_12bit,
)

# The frozen TASK-005 schema this stage consumes.
REQUIRED_KINEMATICS_ARRAYS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "tracking_state_code",
    "source_raw_detection_index",
    "valid_kinematics",
    "valid_palm_frame",
    "flexion_deg",
    "adjacent_spread_deg",
    "palm_rotation_matrix",
    "palm_quaternion_wxyz",
)

N_TRACKS = len(TRACK_ORDER)
N_FINGERS = len(FINGER_ORDER)
N_CHAIN = len(CHAIN_ORDER)
N_SPREAD = len(SPREAD_PAIRS)


class GloveInputError(ValueError):
    """The kinematics input does not satisfy the frozen TASK-005 contract."""


@dataclass(slots=True)
class GloveSequence:
    """Virtual-glove output for one video, in the fixed output contract."""

    sample_id: str
    frame_index: np.ndarray
    timestamp_seconds: np.ndarray
    bend_angle_deg: np.ndarray
    bend_normalized: np.ndarray
    bend_valid: np.ndarray
    spread_angle_deg: np.ndarray
    spread_normalized: np.ndarray
    spread_valid: np.ndarray
    imu_rotation_matrix: np.ndarray
    imu_quaternion_wxyz: np.ndarray
    palm_imu_valid: np.ndarray
    tracking_state_code: np.ndarray
    source_raw_detection_index: np.ndarray
    bend_adc_12bit: np.ndarray
    spread_adc_12bit: np.ndarray
    imu_angular_velocity_rad_s: np.ndarray
    imu_angular_velocity_valid: np.ndarray
    source_valid_kinematics: np.ndarray
    source_valid_palm_frame: np.ndarray
    contract_violations: list[dict[str, Any]] = field(default_factory=list)


def _check_input(arrays: dict[str, np.ndarray], metadata: dict, sample_id: str) -> int:
    missing = [name for name in REQUIRED_KINEMATICS_ARRAYS if name not in arrays]
    if missing:
        raise GloveInputError(f"{sample_id}: kinematics input missing {missing}")

    track_order = tuple(str(t).lower() for t in metadata.get("track_order", ()))
    expected = tuple(t.lower() for t in TRACK_ORDER)
    if track_order != expected:
        raise GloveInputError(
            f"{sample_id}: track order {track_order} != frozen TASK-005 order {expected}"
        )
    finger_order = tuple(metadata.get("finger_order", ()))
    if finger_order != FINGER_ORDER:
        raise GloveInputError(f"{sample_id}: finger order {finger_order} != {FINGER_ORDER}")
    chain_order = tuple(metadata.get("chain_joint_order", ()))
    if chain_order != CHAIN_ORDER:
        raise GloveInputError(f"{sample_id}: chain order {chain_order} != {CHAIN_ORDER}")
    if str(metadata.get("quaternion_order", "")).lower() != "wxyz":
        raise GloveInputError(
            f"{sample_id}: quaternion order {metadata.get('quaternion_order')!r} != 'wxyz'"
        )

    frames = int(np.asarray(arrays["frame_index"]).shape[0])
    shapes = {
        "flexion_deg": (frames, N_TRACKS, N_FINGERS, N_CHAIN),
        "adjacent_spread_deg": (frames, N_TRACKS, N_SPREAD),
        "palm_rotation_matrix": (frames, N_TRACKS, 3, 3),
        "palm_quaternion_wxyz": (frames, N_TRACKS, 4),
        "valid_kinematics": (frames, N_TRACKS),
        "valid_palm_frame": (frames, N_TRACKS),
    }
    for name, shape in shapes.items():
        actual = tuple(np.asarray(arrays[name]).shape)
        if actual != shape:
            raise GloveInputError(f"{sample_id}: {name} has shape {actual}, expected {shape}")
    return frames


def extract_glove_sequence(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    sample_id: str,
    *,
    on_contract_violation: str = "raise",
) -> GloveSequence:
    """Build the virtual-glove sequence for one video.

    Validity is per channel, inheriting TASK-005 Model-B semantics: a hand is
    never discarded wholesale because one channel is absent. ``bend_valid`` and
    ``spread_valid`` come from the finiteness of the frozen channels themselves,
    and ``palm_imu_valid`` from the frozen ``valid_palm_frame``. The strict
    ``valid_kinematics`` flag is carried through for provenance but is
    deliberately NOT used to mask any sensor: a hand with a usable palm and 15
    finite bends keeps all 16 of those sensors even when one spread channel is
    absent and the strict flag is therefore false.
    """

    frames = _check_input(arrays, metadata, sample_id)

    bend_deg = np.asarray(arrays["flexion_deg"], dtype=np.float64)
    spread_deg = np.asarray(arrays["adjacent_spread_deg"], dtype=np.float64)

    bend_norm, bend_violations = normalize_angles(
        bend_deg, channel="bend", on_violation=on_contract_violation
    )
    spread_norm, spread_violations = normalize_angles(
        spread_deg, channel="spread", on_violation=on_contract_violation
    )

    # Masks derive from the frozen channel state, never from a substitute.
    bend_valid = np.isfinite(bend_deg) & ~bend_violations
    spread_valid = np.isfinite(spread_deg) & ~spread_violations
    palm_imu_valid = np.asarray(arrays["valid_palm_frame"], dtype=bool).copy()

    # Orientation is copied verbatim: no re-normalization, no sign change, no
    # basis mapping. The evaluation-only comparison bases are not applied.
    imu_rotation = np.array(arrays["palm_rotation_matrix"], dtype=np.float32, copy=True)
    imu_quaternion = np.array(arrays["palm_quaternion_wxyz"], dtype=np.float32, copy=True)

    omega = np.full((frames, N_TRACKS, 3), np.nan, dtype=np.float64)
    omega_valid = np.zeros((frames, N_TRACKS), dtype=bool)
    for track in range(N_TRACKS):
        track_omega, track_valid = angular_velocity_body_frame(
            np.asarray(arrays["palm_rotation_matrix"], dtype=np.float64)[:, track],
            np.asarray(arrays["timestamp_seconds"], dtype=np.float64),
            palm_imu_valid[:, track],
            np.asarray(arrays["frame_index"]),
        )
        omega[:, track] = track_omega
        omega_valid[:, track] = track_valid

    violations: list[dict[str, Any]] = []
    if bend_violations.any():
        violations.extend(describe_violations(bend_deg, "bend"))
    if spread_violations.any():
        violations.extend(describe_violations(spread_deg, "spread"))

    return GloveSequence(
        sample_id=sample_id,
        frame_index=np.asarray(arrays["frame_index"], dtype=np.int32).copy(),
        timestamp_seconds=np.asarray(arrays["timestamp_seconds"], dtype=np.float64).copy(),
        bend_angle_deg=bend_deg.astype(np.float32),
        bend_normalized=bend_norm.astype(np.float32),
        bend_valid=bend_valid,
        spread_angle_deg=spread_deg.astype(np.float32),
        spread_normalized=spread_norm.astype(np.float32),
        spread_valid=spread_valid,
        imu_rotation_matrix=imu_rotation,
        imu_quaternion_wxyz=imu_quaternion,
        palm_imu_valid=palm_imu_valid,
        tracking_state_code=np.asarray(arrays["tracking_state_code"], dtype=np.int32).copy(),
        source_raw_detection_index=np.asarray(
            arrays["source_raw_detection_index"], dtype=np.int32
        ).copy(),
        bend_adc_12bit=to_adc_12bit(bend_norm, bend_valid),
        spread_adc_12bit=to_adc_12bit(spread_norm, spread_valid),
        imu_angular_velocity_rad_s=omega.astype(np.float32),
        imu_angular_velocity_valid=omega_valid,
        source_valid_kinematics=np.asarray(arrays["valid_kinematics"], dtype=bool).copy(),
        source_valid_palm_frame=np.asarray(arrays["valid_palm_frame"], dtype=bool).copy(),
        contract_violations=violations,
    )
