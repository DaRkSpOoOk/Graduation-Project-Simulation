"""Run-level QA for extractor-independent TASK-006 virtual-glove outputs.

The validator consumes a TASK-005 run as a read-only source and a derived
TASK-006 run.  It checks serialized data and declared metadata; it never
recomputes production kinematics or sensor mathematics.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.kinematics_qa.rotation_checks import (
    quaternion_to_matrix_wxyz,
    rotation_delta_degrees,
    rotation_orthogonality_error,
)

from .contract import (
    ALIGNMENT_ARRAYS,
    BEND_SENSOR_COUNT,
    FINGER_NAMES,
    JOINT_NAMES,
    KinematicsSample,
    MATRIX_QUATERNION_TOLERANCE,
    NORMALIZATION_ATOL,
    NORMALIZATION_RTOL,
    OPTIONAL_ARRAYS,
    PALM_IMU_COUNT,
    QUATERNION_NORM_TOLERANCE,
    REQUIRED_ARRAYS,
    ROTATION_DETERMINANT_TOLERANCE,
    ROTATION_ORTHOGONALITY_TOLERANCE,
    SPREAD_NAMES,
    TRACK_NAMES,
    ContractError,
    SensorDescriptor,
    VirtualGloveSample,
    VIRTUAL_GLOVE_NPZ_NAME,
    canonical_track_order,
    list_sample_ids,
    load_kinematics_sample,
    load_virtual_glove_sample,
    parse_sensor_layout,
    sha256_file,
    validate_kinematics_input,
    validate_sample_contract,
)

TOOL_VERSION = "TASK-006C-v1"
VERDICT_READY = "VIRTUAL-GLOVE QA TOOLING READY"
VERDICT_NEEDS_REVISION = "VIRTUAL-GLOVE QA TOOLING NEEDS REVISION"

CSV_FIELDS = (
    "metric",
    "sensor_id",
    "hand",
    "finger",
    "joint",
    "spread_pair",
    "component",
    "count",
    "min",
    "p1",
    "p50",
    "p95",
    "p99",
    "max",
    "missing_count",
)


def _safe_number(value: Any) -> int | float | None:
    """Convert a scalar to strict-JSON-safe data without repairing it."""

    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, bool):
        return value
    return None


def _percentile_summary(values: Iterable[float]) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "p95": float(np.percentile(vector, 95)),
        "p99": float(np.percentile(vector, 99)),
        "max": float(vector.max()),
    }


def _stats(values: Iterable[float], missing_count: int) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "missing_count": int(missing_count),
        }
    return {
        "count": int(vector.size),
        "min": float(vector.min()),
        "p1": float(np.percentile(vector, 1)),
        "p50": float(np.percentile(vector, 50)),
        "p95": float(np.percentile(vector, 95)),
        "p99": float(np.percentile(vector, 99)),
        "max": float(vector.max()),
        "missing_count": int(missing_count),
    }


def _reference(
    sample_id: str,
    frame_index: int,
    hand: int | str,
    channel: str,
    *,
    sensor_id: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    track = TRACK_NAMES[hand] if isinstance(hand, int) else str(hand).upper()
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "frame_index": int(frame_index),
        "track": track,
        "channel": channel,
    }
    if sensor_id is not None:
        result["sensor_id"] = sensor_id
    if value is not None:
        result["value"] = _safe_number(value)
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            result["value_state"] = "non_finite"
    return result


def _first_mismatch(left: np.ndarray, right: np.ndarray) -> tuple[tuple[int, ...] | None, Any, Any]:
    if left.shape != right.shape:
        return None, left.shape, right.shape
    equal = np.equal(left, right)
    if equal.ndim == 0:
        return (() if not bool(equal) else None), left.item(), right.item()
    if bool(np.all(equal)):
        return None, None, None
    positions = np.argwhere(~equal)
    position = tuple(int(value) for value in positions[0])
    return position, left[position], right[position]


def _shape_ready(sample: VirtualGloveSample) -> bool:
    arrays = sample.arrays
    frame_index = np.asarray(arrays.get("frame_index", np.empty(0)))
    if frame_index.ndim != 1:
        return False
    frame_count = int(frame_index.shape[0])
    for name, specification in REQUIRED_ARRAYS.items():
        if name not in arrays:
            return False
        shape = tuple(np.asarray(arrays[name]).shape)
        if len(shape) != len(specification):
            return False
        for actual, expected in zip(shape, specification):
            if expected == "F" and actual != frame_count:
                return False
            if expected != "F" and actual != expected:
                return False
    return True


def _artifact_sample_ids(run_dir: Path, npz_name: str, metadata_name: str) -> list[str]:
    """Include artifact-only directories so missing companion files are reported."""

    return sorted(
        entry.name
        for entry in run_dir.iterdir()
        if entry.is_dir()
        and ((entry / npz_name).is_file() or (entry / metadata_name).is_file())
    )


def _data_ready(sample: VirtualGloveSample, contract: dict[str, Any]) -> bool:
    if not _shape_ready(sample):
        return False
    arrays = sample.arrays
    numeric = (
        "timestamp_seconds",
        "bend_angle_deg",
        "bend_normalized",
        "spread_angle_deg",
        "spread_normalized",
        "imu_rotation_matrix",
        "imu_quaternion_wxyz",
    )
    if any(not np.issubdtype(np.asarray(arrays[name]).dtype, np.floating) for name in numeric):
        return False
    if any(np.asarray(arrays[name]).dtype != np.dtype(np.bool_) for name in ("bend_valid", "spread_valid", "palm_imu_valid")):
        return False
    frame_count = int(np.asarray(arrays["frame_index"]).shape[0])
    for name, specification in OPTIONAL_ARRAYS.items():
        if name not in arrays:
            continue
        shape = tuple(np.asarray(arrays[name]).shape)
        if len(shape) != len(specification) or any(
            actual != frame_count if expected == "F" else actual != expected
            for actual, expected in zip(shape, specification)
        ):
            return False
    for name in ("bend_adc_12bit", "spread_adc_12bit"):
        if name in arrays:
            dtype = np.asarray(arrays[name]).dtype
            if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
                return False
    if "imu_angular_velocity_rad_s" in arrays:
        if not np.issubdtype(np.asarray(arrays["imu_angular_velocity_rad_s"]).dtype, np.floating):
            return False
        if "imu_angular_velocity_valid" not in arrays or np.asarray(arrays["imu_angular_velocity_valid"]).dtype != np.dtype(np.bool_):
            return False
    if "imu_angular_velocity_valid" in arrays and "imu_angular_velocity_rad_s" not in arrays:
        return False
    return bool(contract.get("frame_count", 0) > 0)


def _frame_integrity(sample: VirtualGloveSample | KinematicsSample, label: str) -> dict[str, Any]:
    arrays = sample.arrays
    failures: list[str] = []
    frames = np.asarray(arrays.get("frame_index", np.empty(0)))
    timestamps_numeric = True
    try:
        timestamps = np.asarray(arrays.get("timestamp_seconds", np.empty(0)), dtype=np.float64)
    except (TypeError, ValueError):
        timestamps = np.asarray(arrays.get("timestamp_seconds", np.empty(0)))
        timestamps_numeric = False
        failures.append(f"{label} timestamp_seconds must be numeric")
    if frames.ndim != 1:
        failures.append(f"{label} frame_index must be rank-1, got {frames.shape}")
        frame_count = 0
    else:
        frame_count = int(frames.size)
        if not np.issubdtype(frames.dtype, np.integer):
            failures.append(f"{label} frame_index must be integer, got {frames.dtype}")
        elif frame_count > 1 and np.any(np.diff(frames.astype(np.int64, copy=False)) <= 0):
            failures.append(f"{label} frame_index must be strictly increasing")
    if timestamps.ndim != 1 or timestamps.size != frame_count:
        failures.append(
            f"{label} timestamp_seconds must be rank-1 with F={frame_count}, got {timestamps.shape}"
        )
    elif timestamps_numeric and not np.isfinite(timestamps).all():
        failures.append(f"{label} timestamp_seconds contains non-finite values")
    elif timestamps_numeric and timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        failures.append(f"{label} timestamp_seconds must be strictly increasing")
    for name in ("tracking_state_code", "source_raw_detection_index"):
        if name in arrays and not np.issubdtype(np.asarray(arrays[name]).dtype, np.integer):
            failures.append(f"{label} {name} must have integer dtype")
    metadata = getattr(sample, "metadata", {})
    declared_frames = metadata.get("total_frames") if isinstance(metadata, dict) else None
    if declared_frames is not None and declared_frames != frame_count:
        failures.append(f"{label} metadata total_frames is {declared_frames!r}, expected {frame_count}")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "frame_count": frame_count,
        "first_frame": int(frames[0]) if frames.size else None,
        "last_frame": int(frames[-1]) if frames.size else None,
    }


def _metadata_normalization_check(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("normalization") or metadata.get("normalization_contract")
    if raw is None:
        # TASK-006A stores the authoritative fixed transfer under the
        # representation catalogue rather than duplicating a top-level QA
        # alias.  Accept that canonical production location without relaxing
        # the required divisor/range/fitting checks below.
        representations = metadata.get("representations")
        if isinstance(representations, dict):
            raw = representations.get("normalized_ideal_sensor")
    failures: list[str] = []
    evidence: list[str] = []
    if not isinstance(raw, dict):
        return {
            "passed": False,
            "declared": False,
            "run_specific_evidence": [],
            "failures": ["metadata normalization contract is missing or not an object"],
        }

    def scan(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                value = node[key]
                key_text = str(key).lower().replace("-", "_")
                child_path = f"{path}.{key}" if path else str(key)
                if key_text in {"run_specific", "per_run", "per_sample", "fit_per_run", "fit_per_sample"} and value is True:
                    evidence.append(f"{child_path}=true")
                if key_text in {"fit_scope", "normalization_scope", "scope", "fit_domain"} and isinstance(value, str):
                    normalized = value.lower().replace("-", "_").replace(" ", "_")
                    if normalized not in {"global", "fixed", "contract", "all_runs", "dataset_independent"}:
                        evidence.append(f"{child_path}={value}")
                if key_text in {"method", "type", "strategy", "kind"} and isinstance(value, str):
                    normalized = value.lower().replace("-", "_").replace(" ", "_")
                    if any(token in normalized for token in ("min_max", "minmax", "percentile", "fitted", "fit_")):
                        evidence.append(f"{child_path}={value}")
                if isinstance(value, str):
                    normalized = value.lower().replace(" ", "")
                    if "run_specific" in normalized or "per_run" in normalized or "min_max" in normalized or "minmax" in normalized:
                        evidence.append(f"{child_path}={value}")
                scan(value, child_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                scan(value, f"{path}[{index}]")

    scan(raw, "normalization")
    evidence = sorted(set(evidence))
    if evidence:
        failures.append("metadata indicates run-specific or fitted min/max normalization")

    divisor = raw.get("angle_divisor_deg", raw.get("denominator_deg", raw.get("scale")))
    if divisor is None:
        formula = raw.get("formula")
        if isinstance(formula, str) and re.search(r"/\s*180(?:\.0*)?", formula):
            divisor = 180.0
    if divisor is None or not isinstance(divisor, (int, float)) or float(divisor) != 180.0:
        failures.append(f"normalization must declare a fixed 180-degree divisor, got {divisor!r}")

    declared_range = raw.get("range", raw.get("normalized_range"))
    if declared_range is not None:
        try:
            range_values = [float(value) for value in declared_range]
        except (TypeError, ValueError):
            range_values = []
        if range_values != [0.0, 1.0]:
            failures.append(f"normalization declared range must be [0, 1], got {declared_range!r}")
    method = raw.get("method", raw.get("type"))
    if isinstance(method, str):
        normalized_method = method.lower().replace("-", "_").replace(" ", "_")
        if normalized_method in {"min_max", "minmax", "percentile", "fitted_min_max"}:
            failures.append(f"normalization method cannot be run-fitted: {method!r}")
    return {
        "passed": not failures,
        "declared": True,
        "run_specific_evidence": evidence,
        "failures": sorted(set(failures)),
    }


def _normalization_values(samples: list[VirtualGloveSample]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    metadata_results: dict[str, dict[str, Any]] = {}
    for sample in samples:
        metadata_results[sample.sample_id] = _metadata_normalization_check(sample.metadata)
        arrays = sample.arrays
        if not _shape_ready(sample):
            continue
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        bend_angle = np.asarray(arrays["bend_angle_deg"], dtype=np.float64)
        bend_norm = np.asarray(arrays["bend_normalized"], dtype=np.float64)
        bend_valid = np.asarray(arrays["bend_valid"], dtype=bool)
        spread_angle = np.asarray(arrays["spread_angle_deg"], dtype=np.float64)
        spread_norm = np.asarray(arrays["spread_normalized"], dtype=np.float64)
        spread_valid = np.asarray(arrays["spread_valid"], dtype=bool)
        for row in range(frames.size):
            for hand in range(2):
                for finger, finger_name in enumerate(FINGER_NAMES):
                    for joint, joint_name in enumerate(JOINT_NAMES):
                        if not bend_valid[row, hand, finger, joint]:
                            continue
                        angle = bend_angle[row, hand, finger, joint]
                        normalized = bend_norm[row, hand, finger, joint]
                        ref = _reference(
                            sample.sample_id,
                            int(frames[row]),
                            hand,
                            f"bend.{finger_name}.{joint_name}",
                        )
                        if not np.isfinite(angle) or not np.isfinite(normalized):
                            continue
                        if normalized < 0.0 or normalized > 1.0:
                            violations.append({**ref, "reason": "normalized_out_of_range", "value": float(normalized)})
                        if angle < 0.0 or angle > 180.0:
                            violations.append({**ref, "reason": "angle_out_of_range", "value": float(angle)})
                        expected = angle / 180.0
                        if not np.isclose(normalized, expected, atol=NORMALIZATION_ATOL, rtol=NORMALIZATION_RTOL):
                            violations.append(
                                {
                                    **ref,
                                    "reason": "normalized_angle_disagreement",
                                    "angle_deg": float(angle),
                                    "normalized": float(normalized),
                                    "expected": float(expected),
                                    "absolute_error": float(abs(normalized - expected)),
                                }
                            )
                for spread, pair_name in enumerate(SPREAD_NAMES):
                    if not spread_valid[row, hand, spread]:
                        continue
                    angle = spread_angle[row, hand, spread]
                    normalized = spread_norm[row, hand, spread]
                    ref = _reference(sample.sample_id, int(frames[row]), hand, f"spread.{pair_name}")
                    if not np.isfinite(angle) or not np.isfinite(normalized):
                        continue
                    if normalized < 0.0 or normalized > 1.0:
                        violations.append({**ref, "reason": "normalized_out_of_range", "value": float(normalized)})
                    if angle < 0.0 or angle > 180.0:
                        violations.append({**ref, "reason": "angle_out_of_range", "value": float(angle)})
                    expected = angle / 180.0
                    if not np.isclose(normalized, expected, atol=NORMALIZATION_ATOL, rtol=NORMALIZATION_RTOL):
                        violations.append(
                            {
                                **ref,
                                "reason": "normalized_angle_disagreement",
                                "angle_deg": float(angle),
                                "normalized": float(normalized),
                                "expected": float(expected),
                                "absolute_error": float(abs(normalized - expected)),
                            }
                        )
    violations.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"], item["channel"], item["reason"]))
    metadata_failures = {
        sample_id: result["failures"]
        for sample_id, result in sorted(metadata_results.items())
        if result["failures"]
    }
    return {
        "passed": not violations and not metadata_failures,
        "tolerance": {"absolute": NORMALIZATION_ATOL, "relative": NORMALIZATION_RTOL},
        "metadata": metadata_results,
        "metadata_failures": metadata_failures,
        "violations": violations,
        "count": len(violations),
    }


def _provenance_check(
    source: KinematicsSample,
    glove: VirtualGloveSample,
    kinematics_run: Path,
) -> dict[str, Any]:
    metadata = glove.metadata
    raw = metadata.get("source", metadata.get("source_provenance", metadata.get("provenance")))
    failures: list[str] = []
    if not isinstance(raw, dict):
        return {"passed": False, "failures": ["metadata source provenance is missing or not an object"]}
    nested = raw.get("kinematics") if isinstance(raw.get("kinematics"), dict) else raw.get("kinematics_input", {})
    if not isinstance(nested, dict):
        nested = {}

    def source_value(names: tuple[str, ...], default: Any = None) -> Any:
        value = next((raw[name] for name in names if name in raw), default)
        return next((nested[name] for name in names if name in nested), value)

    source_id = source_value(("sample_id", "kinematics_sample_id", "source_sample_id"))
    if source_id != source.sample_id:
        failures.append(f"source sample_id must be {source.sample_id!r}, got {source_id!r}")
    task = source_value(("task", "source_task"))
    # TASK-006A uses an explicit ``kinematics_*`` namespace and the frozen
    # TASK-005 schema, but does not repeat a task label in this nested object.
    # Infer TASK-005 only from that unambiguous pair; arbitrary missing labels
    # must still fail provenance validation.
    if task is None:
        declared_schema = source_value(("kinematics_schema_version", "source_schema_version"))
        declared_sample = source_value(("kinematics_sample_id", "source_sample_id"))
        if declared_schema == source.metadata.get("schema_version") and declared_sample == source.sample_id:
            task = "TASK-005 (inferred from explicit kinematics_* provenance)"
    if not isinstance(task, str) or not task.upper().startswith("TASK-005"):
        failures.append(f"source task must identify TASK-005, got {task!r}")
    stage = source_value(("stage", "source_stage"))
    if stage is not None and (not isinstance(stage, str) or "kinematic" not in stage.lower()):
        failures.append(f"source stage must identify kinematics, got {stage!r}")

    digest = source_value(("kinematics_npz_sha256", "source_kinematics_npz_sha256", "source_npz_sha256", "npz_sha256", "sha256"))
    actual_digest = sha256_file(source.path / "hand_kinematics.npz")
    if not isinstance(digest, str) or digest.lower() != actual_digest:
        failures.append("source kinematics_npz_sha256 does not match the TASK-005 input")

    declared_run = source_value(("kinematics_run", "source_run", "run_dir", "run"))
    if declared_run is not None:
        try:
            declared_path = Path(str(declared_run)).expanduser()
            candidates = [declared_path.resolve()]
            if not declared_path.is_absolute():
                candidates.append((glove.path.parent / declared_path).resolve())
            if kinematics_run.resolve() not in candidates:
                failures.append(f"source run does not match --kinematics-run: {declared_run!r}")
        except OSError:
            failures.append(f"source run path is not usable: {declared_run!r}")

    declared_sample_dir = source_value(("kinematics_dir", "source_sample_dir"))
    if declared_sample_dir is not None:
        try:
            declared_path = Path(str(declared_sample_dir)).expanduser().resolve()
            if declared_path != source.path.resolve() and declared_path.parent != kinematics_run.resolve():
                failures.append(f"source sample directory does not match TASK-005 input: {declared_sample_dir!r}")
        except OSError:
            failures.append(f"source sample directory is not usable: {declared_sample_dir!r}")

    source_schema = source_value(("kinematics_schema_version", "source_schema_version"))
    actual_schema = source.metadata.get("schema_version")
    if source_schema is not None and source_schema != actual_schema:
        failures.append("source schema version does not match TASK-005 metadata")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "declared_sample_id": source_id,
        "declared_task": task,
        "declared_sha256": digest,
        "actual_sha256": actual_digest,
    }


def _alignment_check(
    kinematics_run: Path,
    source_samples: dict[str, KinematicsSample],
    glove_samples: dict[str, VirtualGloveSample],
    source_ids: list[str],
    glove_ids: list[str],
) -> dict[str, Any]:
    missing_in_virtual = sorted(set(source_ids) - set(glove_ids))
    extra_in_virtual = sorted(set(glove_ids) - set(source_ids))
    mismatches: list[dict[str, Any]] = []
    track_order_failures: list[dict[str, Any]] = []

    for sample_id in sorted(set(source_ids) & set(glove_ids)):
        source = source_samples.get(sample_id)
        glove = glove_samples.get(sample_id)
        if source is None or glove is None:
            continue
        source_arrays = source.arrays
        glove_arrays = glove.arrays
        source_order = source.metadata.get("track_order")
        glove_order = glove.metadata.get("track_order")
        normalized_source = canonical_track_order(source_order)
        normalized_glove = canonical_track_order(glove_order)
        if normalized_source != normalized_glove or normalized_glove != TRACK_NAMES:
            track_order_failures.append(
                {"sample_id": sample_id, "source": source_order, "virtual_glove": glove_order}
            )

        for field in ALIGNMENT_ARRAYS:
            if field not in source_arrays or field not in glove_arrays:
                mismatches.append({"sample_id": sample_id, "field": field, "reason": "field_missing_for_alignment"})
                continue
            source_value = np.asarray(source_arrays[field])
            glove_value = np.asarray(glove_arrays[field])
            if source_value.shape != glove_value.shape:
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": "frame_count" if field == "frame_index" else field,
                        "reason": "shape_mismatch",
                        "source_shape": list(source_value.shape),
                        "virtual_glove_shape": list(glove_value.shape),
                    }
                )
                continue
            position, left, right = _first_mismatch(source_value, glove_value)
            if position is None:
                continue
            if field in {"frame_index", "timestamp_seconds"}:
                mismatch: dict[str, Any] = {
                    "sample_id": sample_id,
                    "field": field,
                    "first_mismatch_position": list(position) if position is not None else None,
                    "source": _safe_number(left),
                    "virtual_glove": _safe_number(right),
                }
                if field == "timestamp_seconds":
                    mismatch["max_abs_diff"] = float(np.max(np.abs(source_value.astype(np.float64) - glove_value.astype(np.float64))))
                mismatches.append(mismatch)
            else:
                row = int(position[0]) if position else 0
                hand = int(position[1]) if len(position) > 1 else 0
                frame_values = np.asarray(glove_arrays.get("frame_index", np.empty(0)))
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "frame_index": int(frame_values[row]) if row < frame_values.size else None,
                        "track": TRACK_NAMES[hand] if hand < len(TRACK_NAMES) else None,
                        "source": _safe_number(left),
                        "virtual_glove": _safe_number(right),
                    }
                )
    mismatches.sort(key=lambda item: (item["sample_id"], item["field"], item.get("frame_index", -1)))
    track_order_failures.sort(key=lambda item: item["sample_id"])
    return {
        "passed": bool(source_ids) and bool(glove_ids) and not missing_in_virtual and not extra_in_virtual and not mismatches and not track_order_failures,
        "missing_in_virtual_glove": missing_in_virtual,
        "extra_in_virtual_glove": extra_in_virtual,
        "track_order_mismatches": track_order_failures,
        "mismatches": mismatches,
        "sample_count_kinematics": len(source_ids),
        "sample_count_virtual_glove": len(glove_ids),
    }


def _layout_summary(
    glove_samples: dict[str, VirtualGloveSample],
    contract_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[SensorDescriptor]]:
    per_sample: dict[str, Any] = {}
    failures: list[str] = []
    signatures: dict[str, tuple[tuple[Any, ...], ...]] = {}
    first_sensors: list[SensorDescriptor] = []
    for sample_id in sorted(glove_samples):
        result = contract_results.get(sample_id, {})
        layout = result.get("layout")
        if layout is None:
            layout = parse_sensor_layout(glove_samples[sample_id].metadata)
        sensors = list(layout.sensors)
        signature = tuple(
            (
                sensor.sensor_id,
                sensor.sensor_type,
                sensor.hand,
                sensor.family,
                sensor.finger,
                sensor.joint,
                sensor.spread_pair,
                sensor.display_marker,
                sensor.description,
                json.dumps(sensor.logical_location, sort_keys=True, separators=(",", ":")),
            )
            for sensor in sensors
        )
        signatures[sample_id] = signature
        if not first_sensors and layout.sensors:
            first_sensors = sensors
        per_sample[sample_id] = {
            "passed": layout.passed,
            "failures": list(layout.failures),
            "sensor_count": len(sensors),
            "sensor_ids": [sensor.sensor_id for sensor in sensors],
            "representation": layout.representation,
            "physical_template_count": layout.physical_template_count,
            "runtime_identity_count": len(sensors),
        }
        failures.extend(f"{sample_id}: {failure}" for failure in layout.failures)
    if signatures:
        reference_id = sorted(signatures)[0]
        for sample_id in sorted(signatures)[1:]:
            if signatures[sample_id] != signatures[reference_id]:
                failures.append(f"{sample_id}: sensor layout differs from {reference_id}")
    section = {
        "passed": not failures,
        "expected": {
            "physical_template_per_hand": 20,
            "runtime_identity_count": 40,
            "per_hand": {
                "bend_hall": BEND_SENSOR_COUNT,
                "spread_hall": 4,
                "hall_total": 19,
                "palm_imu": PALM_IMU_COUNT,
                "total": 20,
            },
            "run_total": {"hall": 38, "palm_imu": 2, "all_sensors": 40},
        },
        "per_sample": per_sample,
        "failures": sorted(set(failures)),
    }
    return section, first_sensors


def _validity_checks(samples: list[VirtualGloveSample]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    partial_examples: list[dict[str, Any]] = []
    for sample in samples:
        if not _shape_ready(sample):
            continue
        arrays = sample.arrays
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        bend_angle = np.asarray(arrays["bend_angle_deg"], dtype=np.float64)
        bend_norm = np.asarray(arrays["bend_normalized"], dtype=np.float64)
        bend_valid = np.asarray(arrays["bend_valid"], dtype=bool)
        spread_angle = np.asarray(arrays["spread_angle_deg"], dtype=np.float64)
        spread_norm = np.asarray(arrays["spread_normalized"], dtype=np.float64)
        spread_valid = np.asarray(arrays["spread_valid"], dtype=bool)
        matrices = np.asarray(arrays["imu_rotation_matrix"], dtype=np.float64)
        quaternions = np.asarray(arrays["imu_quaternion_wxyz"], dtype=np.float64)
        palm_valid = np.asarray(arrays["palm_imu_valid"], dtype=bool)

        for row in range(frames.size):
            for hand in range(2):
                if palm_valid[row, hand] and bend_valid[row, hand].any() and (~spread_valid[row, hand]).any():
                    partial_examples.append(
                        {
                            "sample_id": sample.sample_id,
                            "frame_index": int(frames[row]),
                            "track": TRACK_NAMES[hand],
                            "bend_valid_count": int(bend_valid[row, hand].sum()),
                            "spread_valid_count": int(spread_valid[row, hand].sum()),
                            "palm_imu_valid": True,
                        }
                    )
                for finger, finger_name in enumerate(FINGER_NAMES):
                    for joint, joint_name in enumerate(JOINT_NAMES):
                        channel = f"bend.{finger_name}.{joint_name}"
                        valid = bool(bend_valid[row, hand, finger, joint])
                        values = (bend_angle[row, hand, finger, joint], bend_norm[row, hand, finger, joint])
                        if valid and not all(np.isfinite(value) for value in values):
                            violations.append({**_reference(sample.sample_id, int(frames[row]), hand, channel), "reason": "valid_channel_has_non_finite_value"})
                        if not valid and not all(np.isnan(value) for value in values):
                            violations.append({**_reference(sample.sample_id, int(frames[row]), hand, channel), "reason": "invalid_channel_must_be_all_nan"})
                for spread, pair_name in enumerate(SPREAD_NAMES):
                    channel = f"spread.{pair_name}"
                    valid = bool(spread_valid[row, hand, spread])
                    values = (spread_angle[row, hand, spread], spread_norm[row, hand, spread])
                    if valid and not all(np.isfinite(value) for value in values):
                        violations.append({**_reference(sample.sample_id, int(frames[row]), hand, channel), "reason": "valid_channel_has_non_finite_value"})
                    if not valid and not all(np.isnan(value) for value in values):
                        violations.append({**_reference(sample.sample_id, int(frames[row]), hand, channel), "reason": "invalid_channel_must_be_all_nan"})
                orientation_values = tuple(matrices[row, hand].reshape(-1).tolist()) + tuple(quaternions[row, hand].tolist())
                if palm_valid[row, hand] and not all(np.isfinite(value) for value in orientation_values):
                    violations.append({**_reference(sample.sample_id, int(frames[row]), hand, "palm_imu.orientation"), "reason": "valid_orientation_has_non_finite_value"})
                if not palm_valid[row, hand] and not all(np.isnan(value) for value in orientation_values):
                    violations.append({**_reference(sample.sample_id, int(frames[row]), hand, "palm_imu.orientation"), "reason": "invalid_orientation_must_be_all_nan"})

        if "imu_angular_velocity_rad_s" in arrays and "imu_angular_velocity_valid" in arrays:
            gyro = np.asarray(arrays["imu_angular_velocity_rad_s"], dtype=np.float64)
            gyro_valid = np.asarray(arrays["imu_angular_velocity_valid"], dtype=bool)
            for row in range(frames.size):
                for hand in range(2):
                    if gyro_valid[row, hand] and not np.isfinite(gyro[row, hand]).all():
                        violations.append({**_reference(sample.sample_id, int(frames[row]), hand, "palm_imu.angular_velocity"), "reason": "valid_gyro_has_non_finite_value"})
                    if not gyro_valid[row, hand] and not np.isnan(gyro[row, hand]).all():
                        violations.append({**_reference(sample.sample_id, int(frames[row]), hand, "palm_imu.angular_velocity"), "reason": "invalid_gyro_must_be_all_nan"})
    violations.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"], item["channel"], item["reason"]))
    partial_examples.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"]))
    return {
        "passed": not violations,
        "violations": violations,
        "count": len(violations),
        "partial_channel_examples": partial_examples,
        "model_b_note": "Bend, spread, palm orientation, and optional gyro validity are checked independently; no all-or-nothing hand rule is applied.",
    }


def _nan_propagation_checks(
    source_samples: dict[str, KinematicsSample],
    glove_samples: dict[str, VirtualGloveSample],
    data_sample_ids: Iterable[str],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for sample_id in sorted(data_sample_ids):
        source = source_samples.get(sample_id)
        glove = glove_samples.get(sample_id)
        if source is None or glove is None or not _shape_ready(glove):
            continue
        source_check = validate_kinematics_input(source)
        if not source_check["passed"]:
            continue
        source_arrays = source.arrays
        arrays = glove.arrays
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        expected_bend = np.isfinite(np.asarray(source_arrays["flexion_deg"], dtype=np.float64))
        actual_bend = np.asarray(arrays["bend_valid"], dtype=bool)
        expected_spread = np.isfinite(np.asarray(source_arrays["adjacent_spread_deg"], dtype=np.float64))
        actual_spread = np.asarray(arrays["spread_valid"], dtype=bool)
        source_palm = np.asarray(source_arrays["valid_palm_frame"], dtype=bool)
        source_matrix_finite = np.isfinite(np.asarray(source_arrays["palm_rotation_matrix"], dtype=np.float64)).all(axis=(2, 3))
        source_quat_finite = np.isfinite(np.asarray(source_arrays["palm_quaternion_wxyz"], dtype=np.float64)).all(axis=2)
        expected_palm = source_palm & source_matrix_finite & source_quat_finite
        actual_palm = np.asarray(arrays["palm_imu_valid"], dtype=bool)

        for row in range(frames.size):
            for hand in range(2):
                for finger, finger_name in enumerate(FINGER_NAMES):
                    for joint, joint_name in enumerate(JOINT_NAMES):
                        ref = _reference(sample_id, int(frames[row]), hand, f"bend.{finger_name}.{joint_name}")
                        if bool(expected_bend[row, hand, finger, joint]) != bool(actual_bend[row, hand, finger, joint]):
                            violations.append({**ref, "reason": "TASK-005 bend finite-state not propagated", "source_finite": bool(expected_bend[row, hand, finger, joint]), "virtual_glove_valid": bool(actual_bend[row, hand, finger, joint])})
                for spread, pair_name in enumerate(SPREAD_NAMES):
                    ref = _reference(sample_id, int(frames[row]), hand, f"spread.{pair_name}")
                    if bool(expected_spread[row, hand, spread]) != bool(actual_spread[row, hand, spread]):
                        violations.append({**ref, "reason": "TASK-005 spread finite-state not propagated", "source_finite": bool(expected_spread[row, hand, spread]), "virtual_glove_valid": bool(actual_spread[row, hand, spread])})
                ref = _reference(sample_id, int(frames[row]), hand, "palm_imu.orientation")
                if bool(expected_palm[row, hand]) != bool(actual_palm[row, hand]):
                    violations.append({**ref, "reason": "TASK-005 palm orientation validity not propagated", "source_valid": bool(expected_palm[row, hand]), "virtual_glove_valid": bool(actual_palm[row, hand])})
    violations.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"], item["channel"]))
    return {
        "passed": not violations,
        "violations": violations,
        "count": len(violations),
        "rule": "NaN/finite state is propagated per bend, spread, and palm-orientation channel; TASK-005 strict hand validity is never used as an all-or-nothing gate.",
    }


def _rotation_checks(samples: list[VirtualGloveSample]) -> dict[str, Any]:
    orthogonality: list[float] = []
    determinant_error: list[float] = []
    quaternion_norm_error: list[float] = []
    matrix_quaternion_element_error: list[float] = []
    matrix_quaternion_angle_error: list[float] = []
    violations: list[dict[str, Any]] = []
    non_finite_count = 0
    non_positive: list[dict[str, Any]] = []
    worst: dict[str, dict[str, Any] | None] = {
        "orthogonality": None,
        "determinant_abs_error": None,
        "quaternion_norm_abs_error": None,
        "matrix_quaternion_element_abs_error": None,
        "matrix_quaternion_angular_disagreement_deg": None,
    }

    def observe(name: str, value: float, ref: dict[str, Any], value_key: str = "value") -> None:
        current = worst[name]
        if current is None or value > float(current[value_key]):
            worst[name] = {**ref, value_key: float(value)}

    for sample in samples:
        if not _shape_ready(sample):
            continue
        arrays = sample.arrays
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        matrices = np.asarray(arrays["imu_rotation_matrix"], dtype=np.float64)
        quaternions = np.asarray(arrays["imu_quaternion_wxyz"], dtype=np.float64)
        valid = np.asarray(arrays["palm_imu_valid"], dtype=bool)
        for row in range(frames.size):
            for hand in range(2):
                if not valid[row, hand]:
                    continue
                ref = _reference(sample.sample_id, int(frames[row]), hand, "palm_imu.orientation")
                matrix = matrices[row, hand]
                quaternion = quaternions[row, hand]
                if not np.isfinite(matrix).all() or not np.isfinite(quaternion).all():
                    non_finite_count += 1
                    continue
                orth = rotation_orthogonality_error(matrix)
                det = float(np.linalg.det(matrix))
                det_error = abs(det - 1.0)
                norm = float(np.linalg.norm(quaternion))
                norm_error = abs(norm - 1.0)
                orthogonality.append(orth)
                determinant_error.append(det_error)
                quaternion_norm_error.append(norm_error)
                observe("orthogonality", orth, ref)
                observe("determinant_abs_error", det_error, ref)
                observe("quaternion_norm_abs_error", norm_error, ref)
                if orth > ROTATION_ORTHOGONALITY_TOLERANCE:
                    violations.append({**ref, "field": "imu_rotation_matrix", "reason": "orthogonality_tolerance", "value": orth, "tolerance": ROTATION_ORTHOGONALITY_TOLERANCE})
                if det <= 0.0:
                    event = {**ref, "field": "imu_rotation_matrix", "determinant": det, "reason": "non_positive_determinant"}
                    non_positive.append(event)
                    violations.append(event)
                if det_error > ROTATION_DETERMINANT_TOLERANCE:
                    violations.append({**ref, "field": "imu_rotation_matrix", "reason": "determinant_tolerance", "value": det_error, "determinant": det, "tolerance": ROTATION_DETERMINANT_TOLERANCE})
                if norm_error > QUATERNION_NORM_TOLERANCE:
                    violations.append({**ref, "field": "imu_quaternion_wxyz", "reason": "quaternion_norm_tolerance", "value": norm_error, "norm": norm, "tolerance": QUATERNION_NORM_TOLERANCE})
                if norm <= np.finfo(np.float64).eps:
                    continue
                quaternion_matrix = quaternion_to_matrix_wxyz(quaternion / norm)
                element_error = float(np.max(np.abs(matrix - quaternion_matrix)))
                angular_error = rotation_delta_degrees(quaternion_matrix, matrix)
                matrix_quaternion_element_error.append(element_error)
                matrix_quaternion_angle_error.append(angular_error)
                observe("matrix_quaternion_element_abs_error", element_error, ref)
                observe("matrix_quaternion_angular_disagreement_deg", angular_error, ref, "value_deg")
                if element_error > MATRIX_QUATERNION_TOLERANCE:
                    violations.append({**ref, "field": "imu_quaternion_wxyz", "reason": "matrix_quaternion_element_tolerance", "value": element_error, "tolerance": MATRIX_QUATERNION_TOLERANCE})
    violations.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"], item.get("field", ""), item["reason"]))
    non_positive.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"]))
    return {
        "passed": not violations and non_finite_count == 0,
        "tolerances": {
            "orthogonality": ROTATION_ORTHOGONALITY_TOLERANCE,
            "determinant_abs_error": ROTATION_DETERMINANT_TOLERANCE,
            "quaternion_norm_abs_error": QUATERNION_NORM_TOLERANCE,
            "matrix_quaternion_element_abs_error": MATRIX_QUATERNION_TOLERANCE,
        },
        "orthogonality": _percentile_summary(orthogonality),
        "determinant_abs_error": _percentile_summary(determinant_error),
        "quaternion_norm_abs_error": _percentile_summary(quaternion_norm_error),
        "matrix_quaternion_element_abs_error": _percentile_summary(matrix_quaternion_element_error),
        "matrix_quaternion_angular_disagreement_deg": _percentile_summary(matrix_quaternion_angle_error),
        "non_finite_valid_orientations": non_finite_count,
        "non_positive_determinant": {"count": len(non_positive), "violations": non_positive},
        "worst": worst,
        "violations": violations,
        "count": len(violations),
    }


def _adc_transfer(metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    raw = metadata.get("adc_transfer", metadata.get("adc_contract"))
    if raw is None:
        # TASK-006A declares the optional compatibility transfer beside the
        # authoritative normalized representation.  Keep this fallback strict
        # about bit depth, rails, mapping and invalid-channel semantics.
        representations = metadata.get("representations")
        if isinstance(representations, dict):
            raw = representations.get("optional_adc")
    if raw is None:
        return None, ["ADC output is present but metadata adc_transfer contract is missing"]
    if not isinstance(raw, dict):
        return None, ["metadata adc_transfer must be an object"]
    failures: list[str] = []
    bits = raw.get("bits", raw.get("bit_depth"))
    declared_range = raw.get("range")
    range_min = declared_range[0] if isinstance(declared_range, (list, tuple)) and len(declared_range) == 2 else 0
    range_max = declared_range[1] if isinstance(declared_range, (list, tuple)) and len(declared_range) == 2 else 4095
    minimum = raw.get("min_code", raw.get("minimum", range_min))
    maximum = raw.get("max_code", raw.get("maximum", range_max))
    if bits != 12:
        failures.append(f"ADC bits must be 12, got {bits!r}")
    if minimum != 0 or maximum != 4095:
        failures.append(f"ADC code range must be 0..4095, got {minimum!r}..{maximum!r}")
    rounding = str(raw.get("rounding", "")).lower().replace("-", "_").replace(" ", "_")
    default_tolerance = 0.0 if "half_up" in rounding or "floor(x_+_0.5)" in rounding else 1.0
    tolerance = raw.get("tolerance_codes", raw.get("tolerance", default_tolerance))
    try:
        tolerance_float = float(tolerance)
    except (TypeError, ValueError):
        tolerance_float = -1.0
    if not np.isfinite(tolerance_float) or tolerance_float < 0.0:
        failures.append(f"ADC tolerance_codes must be a non-negative number, got {tolerance!r}")
    mapping = str(raw.get("mapping", raw.get("formula", "linear_full_scale"))).lower().replace("-", "_").replace(" ", "_")
    if not any(token in mapping for token in ("linear", "full_scale", "normalized")):
        failures.append(f"unsupported ADC transfer mapping: {raw.get('mapping', raw.get('formula'))!r}")
    return {
        "bits": bits,
        "min_code": minimum,
        "max_code": maximum,
        "tolerance_codes": tolerance_float,
        "mapping": mapping,
        "invalid_value": raw.get("invalid_value", raw.get("invalid_sentinel")),
        "rounding": rounding,
    }, sorted(set(failures))


def _adc_checks(samples: list[VirtualGloveSample], sensor_map: dict[tuple[str, str, str, str], SensorDescriptor]) -> dict[str, Any]:
    present = any(name in sample.arrays for sample in samples for name in ("bend_adc_12bit", "spread_adc_12bit"))
    if not present:
        return {"present": False, "passed": True, "failures": [], "violations": [], "count": 0}
    failures: list[str] = []
    violations: list[dict[str, Any]] = []
    transfers: dict[str, dict[str, Any]] = {}
    for sample in samples:
        transfer, transfer_failures = _adc_transfer(sample.metadata)
        if transfer is None:
            failures.extend(f"{sample.sample_id}: {item}" for item in transfer_failures)
        else:
            transfers[sample.sample_id] = transfer
            failures.extend(f"{sample.sample_id}: {item}" for item in transfer_failures)
        if not _shape_ready(sample):
            continue
        arrays = sample.arrays
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        for name, mask_name, norm_name, shape in (
            ("bend_adc_12bit", "bend_valid", "bend_normalized", "bend"),
            ("spread_adc_12bit", "spread_valid", "spread_normalized", "spread"),
        ):
            if name not in arrays:
                continue
            adc = np.asarray(arrays[name])
            if not np.issubdtype(adc.dtype, np.number):
                failures.append(f"{sample.sample_id}: {name} must be numeric")
                continue
            mask = np.asarray(arrays[mask_name], dtype=bool)
            normalized = np.asarray(arrays[norm_name], dtype=np.float64)
            transfer = transfers.get(sample.sample_id)
            if transfer is None or transfer["tolerance_codes"] < 0:
                continue
            tolerance = transfer["tolerance_codes"]
            invalid_sentinel = transfer.get("invalid_value")
            for row in range(frames.size):
                for hand in range(2):
                    if shape == "bend":
                        locations = ((finger, joint, f"bend.{FINGER_NAMES[finger]}.{JOINT_NAMES[joint]}") for finger in range(5) for joint in range(3))
                    else:
                        locations = ((spread, None, f"spread.{SPREAD_NAMES[spread]}") for spread in range(4))
                    for first, second, channel in locations:
                        index = (row, hand, first, second) if second is not None else (row, hand, first)
                        code = adc[index]
                        valid = bool(mask[index])
                        sensor_key = (
                            TRACK_NAMES[hand],
                            "bend" if shape == "bend" else "spread",
                            FINGER_NAMES[first] if shape == "bend" else SPREAD_NAMES[first],
                            JOINT_NAMES[second] if second is not None else "",
                        )
                        sensor_id = sensor_map.get(sensor_key).sensor_id if sensor_key in sensor_map else None
                        ref = _reference(sample.sample_id, int(frames[row]), hand, channel, sensor_id=sensor_id)
                        if valid:
                            if not np.isfinite(code) or float(code) < 0.0 or float(code) > 4095.0:
                                violations.append({**ref, "reason": "adc_out_of_range", "value": _safe_number(code)})
                                continue
                            expected = float(normalized[index]) * 4095.0
                            if "half_up" in transfer["rounding"] or "floor(x_+_0.5)" in transfer["rounding"]:
                                expected = float(np.floor(expected + 0.5))
                            if not np.isfinite(expected) or abs(float(code) - expected) > tolerance:
                                violations.append({**ref, "reason": "adc_normalized_disagreement", "value": _safe_number(code), "expected": float(expected) if np.isfinite(expected) else None, "tolerance_codes": tolerance})
                        else:
                            invalid_ok = bool(np.isnan(code)) if np.issubdtype(adc.dtype, np.floating) else invalid_sentinel is not None and code == invalid_sentinel
                            if not invalid_ok:
                                violations.append({**ref, "reason": "invalid_adc_channel_must_be_explicitly_invalid", "value": _safe_number(code)})
    failures = sorted(set(failures))
    violations.sort(key=lambda item: (item["sample_id"], item["frame_index"], item["track"], item["channel"], item["reason"]))
    return {
        "present": True,
        "passed": not failures and not violations,
        "declared_transfers": transfers,
        "failures": failures,
        "violations": violations,
        "count": len(violations),
        "range": [0, 4095],
    }


def _delta_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(entry["value"]) for entry in entries]
    summary = _percentile_summary(values)
    if entries:
        maximum = max(entries, key=lambda item: (float(item["value"]), tuple(str(item.get(key, "")) for key in ("sample_id", "frame_index", "track", "channel"))))
        summary["maximum"] = {
            "value": float(maximum["value"]),
            "sample_id": maximum["sample_id"],
            "frame_index": int(maximum["frame_index"]),
            "track": maximum["track"],
            "channel": maximum["channel"],
            "sensor_id": maximum.get("sensor_id"),
        }
    else:
        summary["maximum"] = None
    return summary


def _temporal_checks(samples: list[VirtualGloveSample], sensor_map: dict[tuple[str, str, str, str], SensorDescriptor]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "bend_angle_deg": {},
        "spread_angle_deg": {},
        "palm_orientation": {},
        "imu_angular_velocity_rad_s": {},
    }
    for sample in samples:
        if not _shape_ready(sample):
            continue
        arrays = sample.arrays
        frames = np.asarray(arrays["frame_index"], dtype=np.int64)
        bend = np.asarray(arrays["bend_angle_deg"], dtype=np.float64)
        bend_valid = np.asarray(arrays["bend_valid"], dtype=bool)
        spread = np.asarray(arrays["spread_angle_deg"], dtype=np.float64)
        spread_valid = np.asarray(arrays["spread_valid"], dtype=bool)
        matrices = np.asarray(arrays["imu_rotation_matrix"], dtype=np.float64)
        palm_valid = np.asarray(arrays["palm_imu_valid"], dtype=bool)
        for hand in range(2):
            for finger, finger_name in enumerate(FINGER_NAMES):
                for joint, joint_name in enumerate(JOINT_NAMES):
                    channel = f"{finger_name}.{joint_name}"
                    key = f"{TRACK_NAMES[hand]}.{channel}"
                    sensor = sensor_map.get((TRACK_NAMES[hand], "bend", finger_name, joint_name))
                    entries = groups["bend_angle_deg"].setdefault(key, [])
                    for row in range(1, frames.size):
                        if bend_valid[row - 1, hand, finger, joint] and bend_valid[row, hand, finger, joint]:
                            previous = bend[row - 1, hand, finger, joint]
                            current = bend[row, hand, finger, joint]
                            if np.isfinite(previous) and np.isfinite(current):
                                entries.append({"value": abs(float(current - previous)), "sample_id": sample.sample_id, "frame_index": int(frames[row]), "track": TRACK_NAMES[hand], "channel": channel, "sensor_id": sensor.sensor_id if sensor else None})
            for spread_index, pair_name in enumerate(SPREAD_NAMES):
                key = f"{TRACK_NAMES[hand]}.{pair_name}"
                sensor = sensor_map.get((TRACK_NAMES[hand], "spread", pair_name, ""))
                entries = groups["spread_angle_deg"].setdefault(key, [])
                for row in range(1, frames.size):
                    if spread_valid[row - 1, hand, spread_index] and spread_valid[row, hand, spread_index]:
                        previous = spread[row - 1, hand, spread_index]
                        current = spread[row, hand, spread_index]
                        if np.isfinite(previous) and np.isfinite(current):
                            entries.append({"value": abs(float(current - previous)), "sample_id": sample.sample_id, "frame_index": int(frames[row]), "track": TRACK_NAMES[hand], "channel": pair_name, "sensor_id": sensor.sensor_id if sensor else None})
            key = TRACK_NAMES[hand]
            sensor = sensor_map.get((TRACK_NAMES[hand], "imu", "palm", ""))
            entries = groups["palm_orientation"].setdefault(key, [])
            for row in range(1, frames.size):
                if palm_valid[row - 1, hand] and palm_valid[row, hand] and np.isfinite(matrices[row - 1, hand]).all() and np.isfinite(matrices[row, hand]).all():
                    entries.append({"value": rotation_delta_degrees(matrices[row - 1, hand], matrices[row, hand]), "sample_id": sample.sample_id, "frame_index": int(frames[row]), "track": TRACK_NAMES[hand], "channel": "palm_orientation", "sensor_id": sensor.sensor_id if sensor else None})
            if "imu_angular_velocity_rad_s" in arrays and "imu_angular_velocity_valid" in arrays:
                gyro = np.asarray(arrays["imu_angular_velocity_rad_s"], dtype=np.float64)
                gyro_valid = np.asarray(arrays["imu_angular_velocity_valid"], dtype=bool)
                for component, component_name in enumerate(("x", "y", "z")):
                    component_key = f"{TRACK_NAMES[hand]}.{component_name}"
                    entries = groups["imu_angular_velocity_rad_s"].setdefault(component_key, [])
                    for row in range(1, frames.size):
                        if gyro_valid[row - 1, hand] and gyro_valid[row, hand] and np.isfinite(gyro[row - 1, hand, component]) and np.isfinite(gyro[row, hand, component]):
                            entries.append({"value": abs(float(gyro[row, hand, component] - gyro[row - 1, hand, component])), "sample_id": sample.sample_id, "frame_index": int(frames[row]), "track": TRACK_NAMES[hand], "channel": component_name, "sensor_id": sensor.sensor_id if sensor else None})
    summaries = {
        group: {key: _delta_summary(entries) for key, entries in sorted(channel_entries.items())}
        for group, channel_entries in groups.items()
        if group != "imu_angular_velocity_rad_s"
        or any("imu_angular_velocity_rad_s" in sample.arrays for sample in samples)
    }
    return {
        "passed": True,
        "gap_policy": "Only adjacent source frames with both channel-valid endpoints are compared; missing values are never bridged, smoothed, or filled.",
        "bend_angle_deg": summaries.get("bend_angle_deg", {}),
        "spread_angle_deg": summaries.get("spread_angle_deg", {}),
        "palm_orientation": summaries.get("palm_orientation", {}),
        "imu_angular_velocity_rad_s": summaries.get("imu_angular_velocity_rad_s", {}),
    }


def _distribution_checks(
    samples: list[VirtualGloveSample],
    sensor_map: dict[tuple[str, str, str, str], SensorDescriptor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_frames = sum(int(np.asarray(sample.arrays["frame_index"]).shape[0]) for sample in samples if _shape_ready(sample))
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def record(key: tuple[str, str, str, str], metric: str, values: list[float], missing: int, descriptor: SensorDescriptor | None) -> None:
        record_key = key + (metric,)
        records[record_key] = {
            "metric": metric,
            "sensor_id": descriptor.sensor_id if descriptor else None,
            "hand": key[0],
            "finger": key[2] if key[1] == "bend" else "",
            "joint": key[3] if key[1] == "bend" else "",
            "spread_pair": key[2] if key[1] == "spread" else "",
            "component": key[3] if key[1] == "imu" else "",
            **_stats(values, missing),
        }

    for hand in TRACK_NAMES:
        for finger_index, finger_name in enumerate(FINGER_NAMES):
            for joint_name in JOINT_NAMES:
                key = (hand, "bend", finger_name, joint_name)
                angle_values: list[float] = []
                normalized_values: list[float] = []
                valid_count = 0
                for sample in samples:
                    if not _shape_ready(sample):
                        continue
                    arrays = sample.arrays
                    valid = np.asarray(arrays["bend_valid"], dtype=bool)[:, TRACK_NAMES.index(hand), finger_index, JOINT_NAMES.index(joint_name)]
                    angle = np.asarray(arrays["bend_angle_deg"], dtype=np.float64)[:, TRACK_NAMES.index(hand), finger_index, JOINT_NAMES.index(joint_name)]
                    normalized = np.asarray(arrays["bend_normalized"], dtype=np.float64)[:, TRACK_NAMES.index(hand), finger_index, JOINT_NAMES.index(joint_name)]
                    valid_count += int(valid.sum())
                    angle_values.extend(angle[valid & np.isfinite(angle)].tolist())
                    normalized_values.extend(normalized[valid & np.isfinite(normalized)].tolist())
                descriptor = sensor_map.get(key)
                record(key, "bend_angle_deg", angle_values, total_frames - len(angle_values), descriptor)
                record(key, "bend_normalized", normalized_values, total_frames - len(normalized_values), descriptor)
        for spread_name in SPREAD_NAMES:
            key = (hand, "spread", spread_name, "")
            angle_values = []
            normalized_values = []
            spread_index = SPREAD_NAMES.index(spread_name)
            hand_index = TRACK_NAMES.index(hand)
            for sample in samples:
                if not _shape_ready(sample):
                    continue
                arrays = sample.arrays
                valid = np.asarray(arrays["spread_valid"], dtype=bool)[:, hand_index, spread_index]
                angle = np.asarray(arrays["spread_angle_deg"], dtype=np.float64)[:, hand_index, spread_index]
                normalized = np.asarray(arrays["spread_normalized"], dtype=np.float64)[:, hand_index, spread_index]
                angle_values.extend(angle[valid & np.isfinite(angle)].tolist())
                normalized_values.extend(normalized[valid & np.isfinite(normalized)].tolist())
            descriptor = sensor_map.get(key)
            record(key, "spread_angle_deg", angle_values, total_frames - len(angle_values), descriptor)
            record(key, "spread_normalized", normalized_values, total_frames - len(normalized_values), descriptor)

    if any("imu_angular_velocity_rad_s" in sample.arrays for sample in samples):
        for hand in TRACK_NAMES:
            hand_index = TRACK_NAMES.index(hand)
            descriptor = sensor_map.get((hand, "imu", "palm", ""))
            for component, component_name in enumerate(("x", "y", "z")):
                values: list[float] = []
                for sample in samples:
                    arrays = sample.arrays
                    if "imu_angular_velocity_rad_s" not in arrays or "imu_angular_velocity_valid" not in arrays or not _shape_ready(sample):
                        continue
                    valid = np.asarray(arrays["imu_angular_velocity_valid"], dtype=bool)[:, hand_index]
                    gyro = np.asarray(arrays["imu_angular_velocity_rad_s"], dtype=np.float64)[:, hand_index, component]
                    values.extend(gyro[valid & np.isfinite(gyro)].tolist())
                key = (hand, "imu", "palm", component_name)
                record(key, "imu_angular_velocity_rad_s", values, total_frames - len(values), descriptor)

    rows = [records[key] for key in sorted(records)]
    nested: dict[str, Any] = {"bend": {}, "spread": {}, "imu": {}}
    for row in rows:
        if row["metric"].startswith("bend"):
            hand = nested["bend"].setdefault(row["hand"], {})
            finger = hand.setdefault(row["finger"], {})
            channel = finger.setdefault(row["joint"], {"sensor_id": row["sensor_id"]})
            channel[row["metric"]] = {key: row[key] for key in ("count", "min", "p1", "p50", "p95", "p99", "max", "missing_count")}
        elif row["metric"].startswith("spread"):
            hand = nested["spread"].setdefault(row["hand"], {})
            channel = hand.setdefault(row["spread_pair"], {"sensor_id": row["sensor_id"]})
            channel[row["metric"]] = {key: row[key] for key in ("count", "min", "p1", "p50", "p95", "p99", "max", "missing_count")}
        else:
            hand = nested["imu"].setdefault(row["hand"], {})
            channel = hand.setdefault(row["component"], {"sensor_id": row["sensor_id"]})
            channel[row["metric"]] = {key: row[key] for key in ("count", "min", "p1", "p50", "p95", "p99", "max", "missing_count")}
    return {"by_channel": nested, "channel_statistics": rows, "total_frame_opportunities": total_frames}, rows


def _source_schema_summary(
    source_samples: dict[str, KinematicsSample],
    source_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_sample: dict[str, Any] = {}
    failures: dict[str, list[str]] = {}
    for sample_id in source_ids:
        sample = source_samples.get(sample_id)
        if sample is None:
            continue
        check = validate_kinematics_input(sample)
        frame_check = _frame_integrity(sample, "TASK-005")
        sample_failures = sorted(set(check["failures"] + frame_check["failures"]))
        per_sample[sample_id] = {
            "passed": not sample_failures,
            "frame_count": check["frame_count"],
            "failures": sample_failures,
        }
        if sample_failures:
            failures[sample_id] = sample_failures
    return {"passed": bool(source_ids) and not failures and len(per_sample) == len(source_ids), "per_sample": per_sample, "failures": failures}, failures


def validate_runs(
    kinematics_run: str | Path,
    virtual_glove_run: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a TASK-006 run against its TASK-005 source run."""

    kinematics_path = Path(kinematics_run)
    virtual_path = Path(virtual_glove_run)
    if not kinematics_path.is_dir():
        raise ContractError(f"kinematics run does not exist: {kinematics_path}")
    if not virtual_path.is_dir():
        raise ContractError(f"virtual-glove run does not exist: {virtual_path}")

    source_ids = _artifact_sample_ids(kinematics_path, "hand_kinematics.npz", "hand_kinematics_meta.json")
    glove_ids = _artifact_sample_ids(virtual_path, VIRTUAL_GLOVE_NPZ_NAME, "virtual_glove_meta.json")
    source_samples: dict[str, KinematicsSample] = {}
    glove_samples: dict[str, VirtualGloveSample] = {}
    source_load_failures: dict[str, str] = {}
    glove_load_failures: dict[str, str] = {}
    for sample_id in source_ids:
        try:
            source_samples[sample_id] = load_kinematics_sample(kinematics_path, sample_id)
        except ContractError as error:
            source_load_failures[sample_id] = str(error)
    for sample_id in glove_ids:
        try:
            glove_samples[sample_id] = load_virtual_glove_sample(virtual_path, sample_id)
        except ContractError as error:
            glove_load_failures[sample_id] = str(error)

    source_schema, _ = _source_schema_summary(source_samples, source_ids)
    contract_results: dict[str, dict[str, Any]] = {}
    glove_sample_failures: dict[str, list[str]] = {}
    frame_integrity: dict[str, Any] = {"TASK-005": {}, "TASK-006": {}}
    for sample_id in source_ids:
        if sample_id in source_samples:
            frame_integrity["TASK-005"][sample_id] = _frame_integrity(source_samples[sample_id], "TASK-005")
    for sample_id in glove_ids:
        if sample_id in glove_samples:
            result = validate_sample_contract(glove_samples[sample_id])
            contract_results[sample_id] = result
            glove_sample_failures[sample_id] = list(result["failures"])
            frame_integrity["TASK-006"][sample_id] = _frame_integrity(glove_samples[sample_id], "TASK-006")

    schema_failures = {
        sample_id: failures for sample_id, failures in sorted(glove_sample_failures.items()) if failures
    }
    schema_section = {
        "passed": bool(glove_ids) and not glove_load_failures and not schema_failures and len(glove_samples) == len(glove_ids),
        "required_arrays": {name: list(specification) for name, specification in REQUIRED_ARRAYS.items()},
        "optional_arrays": {name: list(specification) for name, specification in OPTIONAL_ARRAYS.items()},
        "load_failures": glove_load_failures,
        "sample_failures": schema_failures,
        "checked_samples": len(glove_samples),
    }
    frame_failures = {
        stage: {
            sample_id: result["failures"]
            for sample_id, result in sorted(per_sample.items())
            if result["failures"]
        }
        for stage, per_sample in frame_integrity.items()
    }
    frame_section = {
        "passed": not any(frame_failures.values()),
        "per_stage": frame_integrity,
        "failures": frame_failures,
    }

    alignment = _alignment_check(kinematics_path, source_samples, glove_samples, source_ids, glove_ids)
    provenance_results: dict[str, Any] = {}
    for sample_id in sorted(set(source_ids) & set(glove_ids)):
        if sample_id in source_samples and sample_id in glove_samples:
            provenance_results[sample_id] = _provenance_check(source_samples[sample_id], glove_samples[sample_id], kinematics_path)
    provenance_failures = {
        sample_id: result["failures"]
        for sample_id, result in provenance_results.items()
        if result["failures"]
    }
    common_sample_count = len(set(source_ids) & set(glove_ids))
    provenance = {"passed": common_sample_count > 0 and not provenance_failures and len(provenance_results) == common_sample_count, "per_sample": provenance_results, "failures": provenance_failures}

    layout, layout_sensors = _layout_summary(glove_samples, contract_results)
    sensor_map = {sensor.key: sensor for sensor in layout_sensors}
    analysis_ids = [
        sample_id
        for sample_id in sorted(glove_samples)
        if sample_id in contract_results and _data_ready(glove_samples[sample_id], contract_results[sample_id])
    ]
    analysis_samples = [glove_samples[sample_id] for sample_id in analysis_ids]
    normalization = _normalization_values(analysis_samples)
    validity = _validity_checks(analysis_samples)
    nan_propagation = _nan_propagation_checks(source_samples, glove_samples, analysis_ids)
    rotation = _rotation_checks(analysis_samples)
    adc = _adc_checks(analysis_samples, sensor_map)
    temporal = _temporal_checks(analysis_samples, sensor_map)
    distributions, csv_rows = _distribution_checks(analysis_samples, sensor_map)

    summary: dict[str, Any] = {
        "tool": {"name": "TASK-006 virtual-glove QA", "version": TOOL_VERSION},
        "inputs": {"kinematics_run": str(kinematics_path), "virtual_glove_run": str(virtual_path)},
        "schema_validation": schema_section,
        "task005_source_schema": source_schema,
        "frame_timestamp_integrity": frame_section,
        "alignment": alignment,
        "provenance": provenance,
        "sensor_layout": layout,
        "normalization": normalization,
        "validity_masks": validity,
        "nan_propagation": nan_propagation,
        "rotation_quality": rotation,
        "adc": adc,
        "temporal_diagnostics": temporal,
        "distributions": distributions,
        "analysis_samples": analysis_ids,
        "load_failures": {"kinematics": source_load_failures, "virtual_glove": glove_load_failures},
    }
    summary["passed"] = all(
        section["passed"]
        for section in (
            schema_section,
            source_schema,
            frame_section,
            alignment,
            provenance,
            layout,
            normalization,
            validity,
            nan_propagation,
            rotation,
            adc,
        )
    )
    summary["verdict"] = VERDICT_READY if summary["passed"] else VERDICT_NEEDS_REVISION
    return summary, csv_rows


def write_json(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a stable, strict-JSON summary."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the compact deterministic channel summary, never full frames."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in CSV_FIELDS})


__all__ = [
    "CSV_FIELDS",
    "TOOL_VERSION",
    "VERDICT_NEEDS_REVISION",
    "VERDICT_READY",
    "validate_runs",
    "write_csv",
    "write_json",
]
