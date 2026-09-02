"""Neutral TASK-006D adapters and final-contract checks.

The independent TASK-006B oracle remains the source of expected values.  This
module only adapts its synthetic TASK-005-like arrays to the frozen TASK-006A
entry point and compares serialized production views with an explicitly
float32-quantized oracle.  It does not repeat the sensor-transfer formula.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.virtual_glove.contract import (
    HALL_PER_HAND,
    KinematicsInput,
    IdealSensorOutput,
    build_sensor_catalog,
    validate_kinematics_input,
)
from evaluation.virtual_glove.orientation import rotation_matrix_axis, rotation_matrix_xyz
from evaluation.virtual_glove.reference import reference_ideal_sensor_model
from evaluation.virtual_glove.synthetic import (
    SyntheticFixture,
    build_invalid_fixtures,
    build_valid_fixtures,
)
from virtual_glove import (
    CHAIN_ORDER,
    FINGER_ORDER,
    TRACK_ORDER,
    angular_velocity_body_frame,
    extract_glove_sequence,
    layout_document,
)


FROZEN_INPUTS: dict[str, Any] = {
    "main_base": "bf3d678f734cdf3fb62c6acdaf1cd774083df159",
    "task006a": {
        "pr": 18,
        "commits": (
            "c438fab3f81402b899e38d4990bb5f86973ae59d",
            "8646ec03cd0a220a42f8e06db7d7b1f4d52cdea9",
            "cf6fcb1e9568b215fe6c872d83b133bc27464eb6",
        ),
        "branch": "opus/task-006a-core-virtual-glove",
    },
    "task006b": {
        "pr": 19,
        "commit": "f0e1cb215f4881a84c528f885a3f5ada6f6c1026",
        "branch": "luna/task-006b-virtual-glove-benchmark",
    },
    "task006c": {
        "pr": 20,
        "commit": "f91361096e7e670fce6819599cc3be4ec3745f92",
        "branch": "opus/task-006c-virtual-glove-qa-tooling",
    },
}


def _production_arrays(data: KinematicsInput) -> dict[str, np.ndarray]:
    """Adapt a benchmark input to TASK-006A's frozen array names.

    Synthetic provenance and detection fields are deliberately neutral because
    TASK-006A consumes the geometric arrays and their TASK-005 validity state.
    The benchmark's own input validator runs before this adapter is called.
    """

    present = np.asarray(data.hand_present, dtype=bool)
    return {
        "frame_index": np.asarray(data.frame_index, dtype=np.int32).copy(),
        "timestamp_seconds": np.asarray(data.timestamps_seconds, dtype=np.float64).copy(),
        "tracking_state_code": np.where(present, 1, 0).astype(np.int32),
        "source_raw_detection_index": np.where(present, 0, -1).astype(np.int32),
        "valid_kinematics": np.asarray(data.valid_kinematics, dtype=bool).copy(),
        "valid_palm_frame": np.asarray(data.valid_palm_frame, dtype=bool).copy(),
        "flexion_deg": np.asarray(data.flexion_deg, dtype=np.float64).copy(),
        "adjacent_spread_deg": np.asarray(data.adjacent_spread_deg, dtype=np.float64).copy(),
        "palm_rotation_matrix": np.asarray(data.palm_rotation_matrix, dtype=np.float64).copy(),
        "palm_quaternion_wxyz": np.asarray(data.palm_quaternion_wxyz, dtype=np.float64).copy(),
    }


def _production_metadata(data: KinematicsInput) -> dict[str, Any]:
    return {
        "track_order": list(data.track_order),
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "quaternion_order": "wxyz",
    }


def _production_output(data: KinematicsInput) -> IdealSensorOutput:
    """Run the actual converter and expose its slots in the oracle shape."""

    sequence = extract_glove_sequence(
        _production_arrays(data),
        _production_metadata(data),
        "synthetic-task006d",
    )
    hall_normalized = np.concatenate(
        (
            sequence.bend_normalized.reshape(sequence.bend_normalized.shape[0], 2, 15),
            sequence.spread_normalized,
        ),
        axis=2,
    )
    hall_valid = np.concatenate(
        (
            sequence.bend_valid.reshape(sequence.bend_valid.shape[0], 2, 15),
            sequence.spread_valid,
        ),
        axis=2,
    )
    return IdealSensorOutput(
        sensor_catalog=build_sensor_catalog(),
        hall_normalized=hall_normalized,
        hall_valid=hall_valid,
        palm_imu_rotation_matrix=sequence.imu_rotation_matrix,
        palm_imu_quaternion_wxyz=sequence.imu_quaternion_wxyz,
        palm_imu_valid=sequence.palm_imu_valid,
    )


def _max_finite_abs_error(actual: object, expected: object) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _compare_outputs(
    actual: IdealSensorOutput,
    expected: IdealSensorOutput,
) -> dict[str, Any]:
    """Compare production output to raw and float32-quantized oracle views."""

    quantized = IdealSensorOutput(
        sensor_catalog=expected.sensor_catalog,
        hall_normalized=expected.hall_normalized.astype(np.float32),
        hall_valid=expected.hall_valid.copy(),
        palm_imu_rotation_matrix=expected.palm_imu_rotation_matrix.astype(np.float32),
        palm_imu_quaternion_wxyz=expected.palm_imu_quaternion_wxyz.astype(np.float32),
        palm_imu_valid=expected.palm_imu_valid.copy(),
    )
    errors: list[str] = []
    if actual.sensor_catalog != expected.sensor_catalog:
        errors.append("sensor_catalog_mismatch")
    for name in ("hall_normalized", "palm_imu_rotation_matrix", "palm_imu_quaternion_wxyz"):
        if not np.array_equal(
            getattr(actual, name), getattr(quantized, name), equal_nan=True
        ):
            errors.append(f"{name}_serialized_mismatch")
    for name in ("hall_valid", "palm_imu_valid"):
        if not np.array_equal(getattr(actual, name), getattr(expected, name)):
            errors.append(f"{name}_mask_mismatch")

    raw_errors = [
        _max_finite_abs_error(actual.hall_normalized, expected.hall_normalized),
        _max_finite_abs_error(
            actual.palm_imu_rotation_matrix, expected.palm_imu_rotation_matrix
        ),
        _max_finite_abs_error(
            actual.palm_imu_quaternion_wxyz, expected.palm_imu_quaternion_wxyz
        ),
    ]
    quantization_errors = [
        _max_finite_abs_error(expected.hall_normalized, quantized.hall_normalized),
        _max_finite_abs_error(
            expected.palm_imu_rotation_matrix, quantized.palm_imu_rotation_matrix
        ),
        _max_finite_abs_error(
            expected.palm_imu_quaternion_wxyz, quantized.palm_imu_quaternion_wxyz
        ),
    ]
    serialized_errors = [
        _max_finite_abs_error(actual.hall_normalized, quantized.hall_normalized),
        _max_finite_abs_error(
            actual.palm_imu_rotation_matrix, quantized.palm_imu_rotation_matrix
        ),
        _max_finite_abs_error(
            actual.palm_imu_quaternion_wxyz, quantized.palm_imu_quaternion_wxyz
        ),
    ]
    return {
        "passed": not errors,
        "errors": errors,
        "raw_float64_oracle_max_abs_error": max(raw_errors),
        "float32_quantization_max_abs_error": max(quantization_errors),
        "serialized_comparison_max_abs_error": max(serialized_errors),
        "serialized_exact_after_quantization": not any(
            error.endswith("_serialized_mismatch") for error in errors
        ),
        "hall_valid": bool(np.array_equal(actual.hall_valid, expected.hall_valid)),
        "imu_valid": bool(np.array_equal(actual.palm_imu_valid, expected.palm_imu_valid)),
    }


def run_valid_fixture_validation(
    fixtures: tuple[SyntheticFixture, ...] | None = None,
) -> dict[str, Any]:
    """Run all expected-valid fixtures through production and the oracle."""

    selected = build_valid_fixtures() if fixtures is None else fixtures
    results: list[dict[str, Any]] = []
    category_counts: dict[str, dict[str, int]] = {}
    raw_max = quantized_max = serialized_max = 0.0
    for fixture in selected:
        category = category_counts.setdefault(fixture.category, {"total": 0, "passed": 0, "failed": 0})
        category["total"] += 1
        result: dict[str, Any] = {
            "fixture_id": fixture.fixture_id,
            "category": fixture.category,
            "passed": False,
        }
        try:
            validate_kinematics_input(fixture.data)
            expected = reference_ideal_sensor_model(fixture.data)
            actual = _production_output(fixture.data)
            comparison = _compare_outputs(actual, expected)
            result.update(comparison)
        except Exception as error:  # pragma: no cover - recorded as a fixture failure
            result["errors"] = [f"exception:{type(error).__name__}:{error}"]
        if result["passed"]:
            category["passed"] += 1
        else:
            category["failed"] += 1
        raw_max = max(raw_max, float(result.get("raw_float64_oracle_max_abs_error", 0.0)))
        quantized_max = max(quantized_max, float(result.get("float32_quantization_max_abs_error", 0.0)))
        serialized_max = max(serialized_max, float(result.get("serialized_comparison_max_abs_error", 0.0)))
        results.append(result)

    coverage = {
        "bend": {
            "fixture_count": sum(item.coverage_kind == "bend" for item in selected),
            "channel_instances": sum(item.coverage_kind == "bend" for item in selected) * 2,
        },
        "spread": {
            "fixture_count": sum(item.coverage_kind == "spread" for item in selected),
            "channel_instances": sum(item.coverage_kind == "spread" for item in selected) * 2,
        },
    }
    # The catalog is deterministic and the two track slots are included in
    # every coverage fixture.  Count category checks rather than inventing a
    # second truth source for the sensor transfer.
    for kind, details in coverage.items():
        covered_ids = {item.fixture_id for item in selected if item.coverage_kind == kind}
        details["passed"] = all(
            row["passed"] for row in results if row.get("fixture_id") in covered_ids
        )
    return {
        "fixture_count": len(selected),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "category_counts": category_counts,
        "coverage": coverage,
        "raw_float64_oracle_max_abs_error": raw_max,
        "float32_quantization_max_abs_error": quantized_max,
        "serialized_comparison_max_abs_error": serialized_max,
        "failures": [result for result in results if not result["passed"]],
        "per_fixture": results,
    }


def run_invalid_fixture_validation(
    fixtures: tuple[SyntheticFixture, ...] | None = None,
) -> dict[str, Any]:
    """Reject all corruptions at the neutral TASK-006 input boundary.

    The frozen production converter assumes a validated TASK-005 source and
    consequently does not validate every temporal/provenance property itself.
    We record its direct outcome for transparency, while the integration
    boundary remains the required hard-fail gate.
    """

    selected = build_invalid_fixtures() if fixtures is None else fixtures
    results: list[dict[str, Any]] = []
    for fixture in selected:
        result: dict[str, Any] = {
            "fixture_id": fixture.fixture_id,
            "expected_error": fixture.expected_error,
            "boundary_rejected": False,
            "production_direct_rejected": False,
        }
        try:
            validate_kinematics_input(fixture.data)
            result["boundary_error"] = "accepted_by_input_contract"
        except Exception as error:
            result["boundary_rejected"] = True
            result["boundary_error"] = f"{type(error).__name__}:{error}"
        try:
            _production_output(fixture.data)
            result["production_direct_error"] = "accepted_without_neutral_boundary"
        except Exception as error:
            result["production_direct_rejected"] = True
            result["production_direct_error"] = f"{type(error).__name__}:{error}"
        result["expected_error_present"] = str(fixture.expected_error) in str(result["boundary_error"])
        result["passed"] = bool(result["boundary_rejected"] and result["expected_error_present"])
        results.append(result)
    return {
        "fixture_count": len(selected),
        "rejected": sum(item["boundary_rejected"] for item in results),
        "passed": sum(item["passed"] for item in results),
        "failed": sum(not item["passed"] for item in results),
        "production_directly_rejected": sum(item["production_direct_rejected"] for item in results),
        "production_directly_accepted": sum(not item["production_direct_rejected"] for item in results),
        "results": results,
    }


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Independent principal rotation vector for a known proper matrix."""

    skew_vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    sine = 0.5 * float(np.linalg.norm(skew_vector))
    cosine = 0.5 * (float(np.trace(matrix)) - 1.0)
    angle = float(np.arctan2(sine, np.clip(cosine, -1.0, 1.0)))
    norm = float(np.linalg.norm(skew_vector))
    if norm <= 1e-14:
        return np.zeros(3, dtype=np.float64)
    return skew_vector / norm * angle


def _body_rate_from_known_local_rotation(
    initial: np.ndarray, local_delta: np.ndarray, delta_seconds: float
) -> np.ndarray:
    """Analytic body rate for ``R_current = R_initial @ local_delta``."""

    return _rotation_vector(local_delta) / float(delta_seconds)


def run_gyro_convention_checks() -> dict[str, Any]:
    """Prove the final body-frame convention against independent rotations."""

    cases = [
        ("identity_to_z90", np.eye(3), rotation_matrix_axis("Z", 90.0), 1.0, "LEFT"),
        (
            "already_rotated_local_y30",
            rotation_matrix_xyz(35.0, -20.0, 15.0),
            rotation_matrix_axis("Y", 30.0),
            0.5,
            "RIGHT",
        ),
        (
            "noncommuting_composed_local",
            rotation_matrix_xyz(-25.0, 40.0, 15.0),
            rotation_matrix_xyz(30.0, -20.0, 35.0),
            2.3,
            "LEFT",
        ),
        (
            "right_track_varying_dt",
            rotation_matrix_axis("X", -40.0),
            rotation_matrix_xyz(-15.0, 25.0, 10.0),
            0.1,
            "RIGHT",
        ),
    ]
    results: list[dict[str, Any]] = []
    world_frame_differences: list[float] = []
    for case_id, initial, local_delta, delta_seconds, track in cases:
        current = initial @ local_delta
        actual, valid = angular_velocity_body_frame(
            np.asarray([initial, current]),
            np.asarray([0.0, delta_seconds]),
            np.asarray([True, True]),
            np.asarray([0, 1]),
        )
        expected = _body_rate_from_known_local_rotation(initial, local_delta, delta_seconds)
        error = float(np.max(np.abs(actual[1] - expected)))
        # The old TASK-006B quaternion helper's q2 * conj(q1) is the world
        # delta R_current @ R_initial.T.  Compare it only as a diagnostic.
        world_vector = _rotation_vector(initial @ local_delta @ initial.T) / delta_seconds
        world_difference = float(np.linalg.norm(expected - world_vector))
        world_frame_differences.append(world_difference)
        results.append(
            {
                "case": case_id,
                "track": track,
                "delta_seconds": delta_seconds,
                "body_expected_rad_s": expected.tolist(),
                "production_rad_s": actual[1].tolist(),
                "max_abs_error_rad_s": error,
                "valid": bool(valid[1]),
                "world_frame_difference_norm": world_difference,
                "passed": bool(valid[1] and error <= 1e-10),
            }
        )
    return {
        "convention": "BODY-FRAME: R_previous.T @ R_current, expressed in the earlier palm body axes",
        "passed": all(item["passed"] for item in results),
        "noncommuting_world_body_difference_observed": max(world_frame_differences, default=0.0) > 1e-8,
        "cases": results,
    }


def run_layout_reconciliation() -> dict[str, Any]:
    """Verify 20 physical template definitions expand bijectively to 40 IDs."""

    document = layout_document()
    template = list(document.get("sensors", []))
    runtime: list[dict[str, Any]] = []
    for hand in TRACK_ORDER:
        for entry in template:
            runtime.append(
                {
                    "sensor_id": f"{hand}.{entry['sensor_id']}",
                    "template_sensor_id": entry["sensor_id"],
                    "hand": hand,
                    "display_marker": entry["display_marker"],
                    "array": entry["array"],
                    "array_index": tuple(entry["array_index"]),
                    "kind": "IMU" if entry["display_marker"] == "IMU" else "HALL",
                }
            )
    expected_slots = {
        ("bend_angle_deg", tuple([finger, joint]))
        for finger in range(5)
        for joint in range(3)
    } | {("spread_angle_deg", (pair,)) for pair in range(4)} | {("imu_quaternion_wxyz", ())}
    observed_slots = {
        (entry["array"], tuple(entry["array_index"])) for entry in template
    }
    marker_ok = all(
        (entry["display_marker"] == ("IMU" if entry["kind"] == "IMU" else "H"))
        for entry in runtime
    )
    return {
        "passed": bool(
            len(template) == 20
            and len(runtime) == 40
            and len({entry["sensor_id"] for entry in runtime}) == 40
            and marker_ok
            and observed_slots == expected_slots
        ),
        "physical_template_definitions_per_hand": len(template),
        "runtime_identity_count": len(runtime),
        "runtime_hall_count": sum(entry["kind"] == "HALL" for entry in runtime),
        "runtime_imu_count": sum(entry["kind"] == "IMU" for entry in runtime),
        "markers": {
            "hall": sorted({entry["display_marker"] for entry in runtime if entry["kind"] == "HALL"}),
            "imu": sorted({entry["display_marker"] for entry in runtime if entry["kind"] == "IMU"}),
        },
        "unique_runtime_ids": len({entry["sensor_id"] for entry in runtime}),
        "slot_bijection": observed_slots == expected_slots,
    }


def summarize_pilot_run(
    kinematics_run: str | Path,
    virtual_glove_run: str | Path,
    qa_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute compact independent accounting over a frozen pilot run."""

    source_root = Path(kinematics_run)
    glove_root = Path(virtual_glove_run)
    sample_ids = sorted(
        entry.name
        for entry in glove_root.iterdir()
        if entry.is_dir() and (entry / "virtual_glove.npz").is_file()
    )
    total_frames = hand_instances = 0
    bend_valid = bend_total = spread_valid = spread_total = 0
    imu_valid = imu_total = 0
    retained_partial = 0
    partial_instances = 0
    flexion_nan = spread_nan = conditioning_nan = 0
    max_bend_normalized = -math.inf
    max_spread_normalized = -math.inf
    gyro_magnitudes: list[float] = []
    alignment_mismatches = 0
    source_sha256: dict[str, str] = {}
    for sample_id in sample_ids:
        glove_dir = glove_root / sample_id
        source_dir = source_root / sample_id
        with np.load(glove_dir / "virtual_glove.npz", allow_pickle=False) as glove_data:
            glove = {key: glove_data[key] for key in glove_data.files}
        with np.load(source_dir / "hand_kinematics.npz", allow_pickle=False) as source_data:
            source = {key: source_data[key] for key in source_data.files}
        frames = int(glove["frame_index"].shape[0])
        total_frames += frames
        hand_instances += frames * 2
        bend_valid += int(glove["bend_valid"].sum())
        bend_total += int(glove["bend_valid"].size)
        spread_valid += int(glove["spread_valid"].sum())
        spread_total += int(glove["spread_valid"].size)
        imu_valid += int(glove["palm_imu_valid"].sum())
        imu_total += int(glove["palm_imu_valid"].size)
        flexion_nan += int(np.isnan(glove["bend_angle_deg"]).sum())
        spread_nan += int(np.isnan(glove["spread_angle_deg"]).sum())
        conditioning_nan += int(
            np.isnan(source["adjacent_spread_deg"])[source["valid_palm_frame"]].sum()
        )
        max_bend_normalized = max(
            max_bend_normalized,
            float(np.nanmax(glove["bend_normalized"])),
        )
        max_spread_normalized = max(
            max_spread_normalized,
            float(np.nanmax(glove["spread_normalized"])),
        )
        strict_false = ~glove["source_valid_kinematics"]
        partial = (
            strict_false
            & glove["source_valid_palm_frame"]
            & glove["bend_valid"].all(axis=(2, 3))
            & (~glove["spread_valid"]).any(axis=2)
        )
        partial_instances += int(partial.sum())
        retained_partial += int(
            (
                glove["bend_valid"].sum(axis=(2, 3))
                + glove["spread_valid"].sum(axis=2)
                + glove["palm_imu_valid"]
            )[strict_false].sum()
        )
        if "imu_angular_velocity_rad_s" in glove:
            gyro = glove["imu_angular_velocity_rad_s"]
            valid = glove["imu_angular_velocity_valid"]
            gyro_magnitudes.extend(np.linalg.norm(gyro[valid], axis=1).astype(float).tolist())
        for field in ("frame_index", "timestamp_seconds", "tracking_state_code", "source_raw_detection_index"):
            if not np.array_equal(glove[field], source[field]):
                alignment_mismatches += 1
        source_sha256[sample_id] = str(json.loads((glove_dir / "virtual_glove_meta.json").read_text())["source"]["kinematics_npz_sha256"])

    def percentile(values: list[float], q: float) -> float | None:
        return float(np.percentile(np.asarray(values), q)) if values else None

    return {
        "samples": len(sample_ids),
        "sample_ids": sample_ids,
        "frames": total_frames,
        "hand_instances": hand_instances,
        "bend": {"valid": bend_valid, "total": bend_total},
        "spread": {"valid": spread_valid, "total": spread_total},
        "imu": {"valid": imu_valid, "total": imu_total},
        "strict_false_partial_instances": partial_instances,
        "retained_channels_on_strict_false": retained_partial,
        "nan_accounting": {
            "flexion": flexion_nan,
            "spread": spread_nan,
            "spread_conditioning_with_valid_palm": conditioning_nan,
            "no_pose_spread": spread_nan - conditioning_nan,
        },
        "normalization_observed_max": {
            "bend": max_bend_normalized if np.isfinite(max_bend_normalized) else None,
            "spread": max_spread_normalized if np.isfinite(max_spread_normalized) else None,
        },
        "gyro_magnitude_rad_s": {
            "count": len(gyro_magnitudes),
            "p99": percentile(gyro_magnitudes, 99),
            "max": max(gyro_magnitudes, default=None),
        },
        "gyro_magnitude_deg_s": {
            "count": len(gyro_magnitudes),
            "p99": percentile([value * 180.0 / math.pi for value in gyro_magnitudes], 99),
            "max": max((value * 180.0 / math.pi for value in gyro_magnitudes), default=None),
        },
        "alignment_mismatch_count": alignment_mismatches,
        "source_sha256_count": len(source_sha256),
        "qa_passed": bool(qa_summary.get("passed")) if qa_summary is not None else None,
    }


__all__ = [
    "FROZEN_INPUTS",
    "run_gyro_convention_checks",
    "run_invalid_fixture_validation",
    "run_layout_reconciliation",
    "run_valid_fixture_validation",
    "summarize_pilot_run",
]
