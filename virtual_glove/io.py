"""Persistence for the virtual-glove stage.

All arrays are plain numeric NumPy arrays, written and read with
``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .extractor import GloveSequence
from .imu import DERIVED_ANGULAR_VELOCITY_UNITS, accelerometer_feasibility
from .layout import (
    CHAIN_ORDER,
    EXPECTED_BEND_SENSORS,
    EXPECTED_HALL_SENSORS,
    EXPECTED_IMU_SENSORS,
    EXPECTED_SENSOR_PACKAGES,
    EXPECTED_SPREAD_SENSORS,
    FINGER_ORDER,
    LAYOUT_VERSION,
    SPREAD_PAIRS,
    TRACK_ORDER,
    layout_document,
)
from .signals import (
    ADC_BITS,
    ADC_INVALID_SENTINEL,
    ADC_MAX,
    ADC_MIN,
    CONTRACT_MAX_DEG,
    CONTRACT_MIN_DEG,
    NORMALIZATION_DIVISOR,
)

SCHEMA_VERSION = "virtual_glove_v1"
GLOVE_NPZ_NAME = "virtual_glove.npz"
GLOVE_META_NAME = "virtual_glove_meta.json"
SENSOR_LAYOUT_NAME = "sensor_layout.json"

ARRAY_ORDER: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "bend_angle_deg",
    "bend_normalized",
    "bend_valid",
    "spread_angle_deg",
    "spread_normalized",
    "spread_valid",
    "imu_rotation_matrix",
    "imu_quaternion_wxyz",
    "palm_imu_valid",
    "tracking_state_code",
    "source_raw_detection_index",
    # optional / provenance
    "bend_adc_12bit",
    "spread_adc_12bit",
    "imu_angular_velocity_rad_s",
    "imu_angular_velocity_valid",
    "source_valid_kinematics",
    "source_valid_palm_frame",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    sequence: GloveSequence,
    *,
    kinematics_dir: Path,
    kinematics_sha256: str,
    kinematics_metadata: dict[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "virtual_glove",
        "task": "TASK-006A",
        "sample_id": sequence.sample_id,
        "total_frames": int(sequence.frame_index.shape[0]),
        "layout_version": LAYOUT_VERSION,
        "track_order": list(TRACK_ORDER),
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "spread_pairs": [list(pair) for pair in SPREAD_PAIRS],
        "sensor_counts_per_hand": {
            "bend_hall_sensors": EXPECTED_BEND_SENSORS,
            "spread_hall_sensors": EXPECTED_SPREAD_SENSORS,
            "hall_sensors_total": EXPECTED_HALL_SENSORS,
            "imu_packages": EXPECTED_IMU_SENSORS,
            "logical_sensing_packages": EXPECTED_SENSOR_PACKAGES,
        },
        "representations": {
            "geometric_angle": {
                "arrays": ["bend_angle_deg", "spread_angle_deg"],
                "units": "degrees",
                "note": "frozen TASK-005 values, copied verbatim",
            },
            "normalized_ideal_sensor": {
                "arrays": ["bend_normalized", "spread_normalized"],
                "formula": f"degrees / {NORMALIZATION_DIVISOR}",
                "range": [0.0, 1.0],
                "authoritative_for_ml": True,
                "fitted_to_dataset": False,
                "note": (
                    "fixed a priori from the TASK-005 0..180 degree contract; no "
                    "pilot min/max normalization, no per-finger neutral offset, "
                    "no per-subject or per-dataset term"
                ),
            },
            "optional_adc": {
                "arrays": ["bend_adc_12bit", "spread_adc_12bit"],
                "bits": ADC_BITS,
                "range": [ADC_MIN, ADC_MAX],
                "formula": f"normalized * {ADC_MAX}",
                "rounding": "half-up, floor(x + 0.5)",
                "invalid_sentinel": ADC_INVALID_SENTINEL,
                "authoritative_for_ml": False,
                "note": (
                    "ideal full-scale 12-bit encoding. Deliberately NOT the old "
                    "physical prototype's ~850-1700 count range, which described "
                    "that hardware and is not the ideal simulated-glove contract."
                ),
            },
        },
        "contract": {
            "input_contract": "TASK-005-final-v2",
            "angle_range_deg": [CONTRACT_MIN_DEG, CONTRACT_MAX_DEG],
            "on_finite_value_outside_contract": (
                "raise SensorContractViolation; never clamped, never repaired"
            ),
        },
        "validity_policy": {
            "model": "TASK-005 Model-B, per channel",
            "bend_valid": "isfinite(frozen flexion_deg) for that exact channel",
            "spread_valid": "isfinite(frozen adjacent_spread_deg) for that exact channel",
            "palm_imu_valid": "frozen valid_palm_frame",
            "strict_valid_kinematics": (
                "carried through as source_valid_kinematics for provenance only; "
                "never used to mask a sensor. A hand with a usable palm and 15 "
                "finite bends keeps all 16 of those sensors even when one spread "
                "channel is absent and the strict flag is therefore false."
            ),
            "no_interpolation": True,
            "no_forward_fill": True,
            "no_cross_hand_copying": True,
            "no_invented_values": True,
        },
        "imu": {
            "orientation_source": "frozen TASK-005 palm_rotation_matrix / palm_quaternion_wxyz",
            "orientation_transform": "none - copied verbatim",
            "quaternion_order": "wxyz",
            "evaluation_only_basis_applied": False,
            "evaluation_only_basis_note": (
                "the LEFT/RIGHT comparison bases in "
                "evaluation.kinematics.final_contract are an evaluation "
                "convention and are not applied to stored production orientation"
            ),
            "angular_velocity": {
                "arrays": ["imu_angular_velocity_rad_s", "imu_angular_velocity_valid"],
                "status": "DERIVED",
                "units": DERIVED_ANGULAR_VELOCITY_UNITS,
                "frame": "body frame of the earlier sample",
                "formula": "axis_angle(R[k]^T @ R[k+1]) / (t[k+1] - t[k])",
                "smoothing": "none",
                "bridging": (
                    "never: requires both orientations valid, adjacent frame_index, "
                    "and a finite positive timestamp delta"
                ),
            },
            "accelerometer": accelerometer_feasibility(),
        },
        "ml_contract": {
            "primary_channels_per_hand_per_timestep": 23,
            "composition": {
                "bend_normalized": 15,
                "spread_normalized": 4,
                "imu_quaternion_wxyz": 4,
            },
            "plus": "validity masks",
            "not_done_here": [
                "no LSTM tensor is built",
                "no feature ablation is decided",
                "no classifier is trained",
                "no final training dataset is generated",
            ],
        },
        "source": {
            "kinematics_dir": str(kinematics_dir),
            "kinematics_npz_sha256": kinematics_sha256,
            "kinematics_schema_version": kinematics_metadata.get("schema_version"),
            "kinematics_implementation_commit": kinematics_metadata.get(
                "implementation_commit"
            ),
            "kinematics_sample_id": kinematics_metadata.get("sample_id"),
            "tracked_source": (kinematics_metadata.get("source") or {}),
        },
        "contract_violations": sequence.contract_violations,
        "implementation_commit": implementation_commit,
    }


def save_glove_sequence(
    directory: str | Path, sequence: GloveSequence, metadata: dict[str, Any]
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(sequence, name) for name in ARRAY_ORDER}
    npz_path = path / GLOVE_NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    (path / GLOVE_META_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (path / SENSOR_LAYOUT_NAME).write_text(
        json.dumps(layout_document(), indent=2, sort_keys=True) + "\n"
    )
    return npz_path


def load_glove_sequence(directory: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(directory)
    npz_path = path / GLOVE_NPZ_NAME if path.is_dir() else path
    meta_path = npz_path.parent / GLOVE_META_NAME
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return arrays, metadata
