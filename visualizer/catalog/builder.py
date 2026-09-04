"""Deterministic Core-28 exemplar selection from TASK-008 artifacts.

This module only audits already-produced arrays and metadata.  It does not
derive angles, normalize sensor values, fill masks, or alter production
artifacts.  The production sensor mathematics remains upstream in TASK-005/6.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from evaluation.dataset.core28 import load_label_records, validate_core28_records
from evaluation.virtual_glove.orientation import quaternion_matrix_wxyz

from .catalog import CATALOG_VERSION, ExemplarEntry, write_catalog, write_catalog_csv
from .descriptor import SequenceDescriptor

EXPECTED_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
EXPECTED_JOINTS = ("proximal", "middle", "distal")
EXPECTED_SPREAD_PAIRS = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)
EXPECTED_BEND_SHAPE = (2, 5, 3)
EXPECTED_SPREAD_SHAPE = (2, 4)
EXPECTED_HAND_SHAPE = (2,)
EXPECTED_ROTATION_SHAPE = (2, 3, 3)
EXPECTED_QUATERNION_SHAPE = (2, 4)
POSE_BEARING_CODES = (1, 2)
ANGLE_TOLERANCE = 1e-5
NORMALIZATION_TOLERANCE = 1e-5
ORIENTATION_TOLERANCE = 1e-4
ADC_TOLERANCE = 0


class CatalogBuildError(ValueError):
    """Raised when a reproducible catalog cannot be built from the source."""


@dataclass(slots=True)
class _Candidate:
    row: dict[str, str]
    descriptor: SequenceDescriptor
    sequence_length: int
    metrics: dict[str, Any]
    score: float = 0.0

    @property
    def sign_id(self) -> str:
        return self.row["sign_id"]

    @property
    def sample_id(self) -> str:
        return self.row["sample_id"]

    @property
    def signer_id(self) -> str:
        return self.row.get("signer_id", "")

    def entry(self, *, reason: str = "") -> ExemplarEntry:
        return ExemplarEntry(
            character=self.row["label_ar"],
            sign_id=self.sign_id,
            label_index=int(self.row["label_index"]),
            sample_id=self.sample_id,
            signer_id=self.signer_id,
            official_partition=self.row.get("official_partition", ""),
            repetition_id=self.row.get("repetition_id", ""),
            sequence_length=self.sequence_length,
            descriptor=self.descriptor,
            score=float(self.score),
            metrics=self.metrics,
            selection_reason=reason,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: Any) -> float:
    return float(value)


def _fraction(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return float(values.mean()) if values.size else 0.0


def _safe_fraction(values: np.ndarray, mask: np.ndarray) -> tuple[float, list[str]]:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    problems: list[str] = []
    if values.shape != mask.shape:
        return 0.0, ["value_mask_shape_mismatch"]
    valid_finite = np.isfinite(values[mask])
    invalid_values = values[~mask]
    if not bool(valid_finite.all()):
        problems.append("valid_channel_is_non_finite")
    if invalid_values.size and bool(np.isfinite(invalid_values).any()):
        problems.append("invalid_channel_contains_finite_value")
    return _fraction(mask), problems


def _layout_audit(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.is_file():
        return False, {"present": False, "contract_valid": False, "problems": ["missing_sensor_layout"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, {"present": True, "contract_valid": False, "problems": [f"invalid_json:{error}"]}

    sensors = payload.get("sensors")
    problems: list[str] = []
    if payload.get("layout_version") not in {None, "ideal_virtual_glove_v1"}:
        problems.append(f"layout_version={payload.get('layout_version')!r}")
    if not isinstance(sensors, list):
        return False, {"present": True, "contract_valid": False, "problems": ["sensors_not_list"]}
    if len(sensors) != 20:
        problems.append(f"sensor_count={len(sensors)}")
    ids = [str(sensor.get("sensor_id", "")) for sensor in sensors]
    if any(not sensor_id for sensor_id in ids):
        problems.append("missing_sensor_id")
    if len(ids) != len(set(ids)):
        problems.append("duplicate_sensor_id")

    bend = [sensor for sensor in sensors if sensor.get("role") == "bend"]
    spread = [sensor for sensor in sensors if sensor.get("role") == "spread"]
    imu = [sensor for sensor in sensors if sensor.get("role") in {"orientation", "palm_orientation"}]
    if len(bend) != 15:
        problems.append(f"bend_sensor_count={len(bend)}")
    if len(spread) != 4:
        problems.append(f"spread_sensor_count={len(spread)}")
    if len(imu) != 1:
        problems.append(f"imu_sensor_count={len(imu)}")

    bend_assignments = set()
    for sensor in bend:
        finger = sensor.get("finger")
        joint = sensor.get("joint")
        if finger not in EXPECTED_FINGERS or joint not in EXPECTED_JOINTS:
            problems.append(f"invalid_bend_assignment={sensor.get('sensor_id', '')}")
        bend_assignments.add((finger, joint))
        if sensor.get("array") != "bend_angle_deg":
            problems.append(f"invalid_bend_array={sensor.get('sensor_id', '')}")
        if sensor.get("display_marker") != "H":
            problems.append(f"invalid_hall_marker={sensor.get('sensor_id', '')}")
        if not str(sensor.get("sensor_type", "")).startswith("hall"):
            problems.append(f"invalid_hall_type={sensor.get('sensor_id', '')}")

    spread_assignments = set()
    for sensor in spread:
        pair = tuple(sensor.get("pair") or ())
        if pair not in EXPECTED_SPREAD_PAIRS:
            problems.append(f"invalid_spread_assignment={sensor.get('sensor_id', '')}")
        spread_assignments.add(pair)
        if sensor.get("array") != "spread_angle_deg":
            problems.append(f"invalid_spread_array={sensor.get('sensor_id', '')}")
        if sensor.get("display_marker") != "H":
            problems.append(f"invalid_hall_marker={sensor.get('sensor_id', '')}")
        if not str(sensor.get("sensor_type", "")).startswith("hall"):
            problems.append(f"invalid_hall_type={sensor.get('sensor_id', '')}")

    if len(bend_assignments) != 15:
        problems.append("duplicate_or_incomplete_bend_coverage")
    if len(spread_assignments) != 4:
        problems.append("duplicate_or_incomplete_spread_coverage")
    for sensor in sensors:
        if not str(sensor.get("logical_location", "")).strip():
            problems.append(f"missing_logical_location={sensor.get('sensor_id', '')}")
        if not str(sensor.get("description", "")).strip():
            problems.append(f"missing_description={sensor.get('sensor_id', '')}")

    if imu:
        sensor = imu[0]
        if sensor.get("display_marker") != "IMU":
            problems.append(f"invalid_imu_marker={sensor.get('sensor_id', '')}")
        if not str(sensor.get("sensor_type", "")).startswith("imu"):
            problems.append(f"invalid_imu_type={sensor.get('sensor_id', '')}")
        if sensor.get("array") != "imu_quaternion_wxyz":
            problems.append(f"invalid_imu_array={sensor.get('sensor_id', '')}")

    counts = payload.get("per_hand_counts", {})
    expected_counts = {
        "bend_hall_sensors": 15,
        "spread_hall_sensors": 4,
        "hall_sensors_total": 19,
        "imu_packages": 1,
        "logical_sensing_packages": 20,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            problems.append(f"{key}={counts.get(key)!r}")
    result = {
        "present": True,
        "contract_valid": not problems,
        "problems": sorted(set(problems)),
        "sensor_count": len(sensors),
        "hall_count": len(bend) + len(spread),
        "imu_count": len(imu),
    }
    return not problems, result


def _orientation_audit(matrix: np.ndarray, quaternion: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid_positions = np.argwhere(np.asarray(valid, dtype=bool))
    if valid_positions.size == 0:
        return {
            "valid_count": 0,
            "quality_count": 0,
            "quality_fraction": 0.0,
            "matrix_orthogonality_max": None,
            "determinant_min": None,
            "quaternion_norm_error_max": None,
            "matrix_quaternion_error_max": None,
        }
    orth_errors: list[float] = []
    determinants: list[float] = []
    norm_errors: list[float] = []
    consistency_errors: list[float] = []
    quality_count = 0
    identity = np.eye(3)
    for frame, hand in valid_positions:
        rotation = np.asarray(matrix[frame, hand], dtype=np.float64)
        quat = np.asarray(quaternion[frame, hand], dtype=np.float64)
        if not np.isfinite(rotation).all() or not np.isfinite(quat).all():
            orth_errors.append(float("inf"))
            determinants.append(float("nan"))
            norm_errors.append(float("inf"))
            consistency_errors.append(float("inf"))
            continue
        orth_error = float(np.max(np.abs(rotation.T @ rotation - identity)))
        determinant = float(np.linalg.det(rotation))
        norm_error = abs(float(np.linalg.norm(quat)) - 1.0)
        try:
            reconstructed = quaternion_matrix_wxyz(quat)
            consistency_error = float(np.max(np.abs(rotation - reconstructed)))
        except ValueError:
            consistency_error = float("inf")
        orth_errors.append(orth_error)
        determinants.append(determinant)
        norm_errors.append(norm_error)
        consistency_errors.append(consistency_error)
        if (
            orth_error <= ORIENTATION_TOLERANCE
            and abs(determinant - 1.0) <= ORIENTATION_TOLERANCE
            and determinant > 0.0
            and norm_error <= ORIENTATION_TOLERANCE
            and consistency_error <= ORIENTATION_TOLERANCE
        ):
            quality_count += 1
    finite_determinants = [value for value in determinants if math.isfinite(value)]
    return {
        "valid_count": int(len(valid_positions)),
        "quality_count": quality_count,
        "quality_fraction": quality_count / len(valid_positions),
        "matrix_orthogonality_max": max(orth_errors),
        "determinant_min": min(finite_determinants) if finite_determinants else None,
        "quaternion_norm_error_max": max(norm_errors),
        "matrix_quaternion_error_max": max(consistency_errors),
    }


def _adc_audit(
    adc: np.ndarray | None,
    degrees: np.ndarray,
    normalized: np.ndarray,
    valid: np.ndarray,
    name: str,
) -> tuple[dict[str, Any], list[str]]:
    if adc is None:
        return {"present": False}, []
    values = np.asarray(adc)
    problems: list[str] = []
    if values.shape != valid.shape:
        return {"present": True, "shape": list(values.shape)}, [f"{name}_shape_mismatch"]
    valid_values = values[valid]
    invalid_values = values[~valid]
    if valid_values.size and (np.any(valid_values < 0) or np.any(valid_values > 4095)):
        problems.append(f"{name}_outside_0_4095")
    if invalid_values.size and np.any(invalid_values != -1):
        problems.append(f"{name}_invalid_not_minus_one")
    finite_contract = valid & np.isfinite(normalized) & np.isfinite(degrees)
    expected_from_normalized = np.floor(
        np.asarray(normalized[finite_contract], dtype=np.float64) * 4095.0 + 0.5
    ).astype(np.int64)
    expected_from_degrees = np.floor(
        np.asarray(degrees[finite_contract], dtype=np.float64) / 180.0 * 4095.0 + 0.5
    ).astype(np.int64)
    actual = np.asarray(adc[finite_contract], dtype=np.int64)
    agrees_with_normalized = actual == expected_from_normalized
    agrees_with_degrees = actual == expected_from_degrees
    agreement = int(np.sum(~agrees_with_normalized)) if actual.size else 0
    float32_rounding = int(np.sum(~agrees_with_normalized & agrees_with_degrees)) if actual.size else 0
    true_mismatch = int(np.sum(~agrees_with_normalized & ~agrees_with_degrees)) if actual.size else 0
    if true_mismatch:
        problems.append(f"{name}_normalization_disagreement={true_mismatch}")
    return {
        "present": True,
        "valid_count": int(valid_values.size),
        "invalid_count": int(invalid_values.size),
        "agreement_mismatches": agreement,
        "float32_rounding_mismatches": float32_rounding,
    }, problems


def _validate_shapes_and_values(
    arrays: Mapping[str, np.ndarray],
    row: Mapping[str, str],
    layout_result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    required = {
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
    }
    missing = sorted(required - set(arrays))
    if missing:
        return {}, [f"missing_array={name}" for name in missing]
    frame_index = np.asarray(arrays["frame_index"])
    if frame_index.ndim != 1:
        problems.append(f"frame_index_shape={frame_index.shape}")
        return {}, problems
    frames = int(frame_index.shape[0])
    expected = {
        "timestamp_seconds": (frames,),
        "bend_angle_deg": (frames,) + EXPECTED_BEND_SHAPE,
        "bend_normalized": (frames,) + EXPECTED_BEND_SHAPE,
        "bend_valid": (frames,) + EXPECTED_BEND_SHAPE,
        "spread_angle_deg": (frames,) + EXPECTED_SPREAD_SHAPE,
        "spread_normalized": (frames,) + EXPECTED_SPREAD_SHAPE,
        "spread_valid": (frames,) + EXPECTED_SPREAD_SHAPE,
        "imu_rotation_matrix": (frames,) + EXPECTED_ROTATION_SHAPE,
        "imu_quaternion_wxyz": (frames,) + EXPECTED_QUATERNION_SHAPE,
        "palm_imu_valid": (frames,) + EXPECTED_HAND_SHAPE,
        "tracking_state_code": (frames,) + EXPECTED_HAND_SHAPE,
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            problems.append(f"{name}_shape={np.asarray(arrays[name]).shape}")
    if problems:
        return {"frames": frames}, problems
    if np.any(np.diff(frame_index.astype(np.int64)) <= 0):
        problems.append("frame_index_not_strictly_increasing")
    timestamps = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        problems.append("timestamp_not_finite_or_strictly_increasing")
    for key in ("bend_valid", "spread_valid", "palm_imu_valid"):
        if np.asarray(arrays[key]).dtype.kind not in {"b", "i", "u"}:
            problems.append(f"{key}_not_boolean_like")

    bend_deg = np.asarray(arrays["bend_angle_deg"], dtype=np.float64)
    bend_norm = np.asarray(arrays["bend_normalized"], dtype=np.float64)
    bend_valid = np.asarray(arrays["bend_valid"], dtype=bool)
    spread_deg = np.asarray(arrays["spread_angle_deg"], dtype=np.float64)
    spread_norm = np.asarray(arrays["spread_normalized"], dtype=np.float64)
    spread_valid = np.asarray(arrays["spread_valid"], dtype=bool)
    for values, mask, name in (
        (bend_deg, bend_valid, "bend_angle_deg"),
        (bend_norm, bend_valid, "bend_normalized"),
        (spread_deg, spread_valid, "spread_angle_deg"),
        (spread_norm, spread_valid, "spread_normalized"),
    ):
        _, mask_problems = _safe_fraction(values, mask)
        problems.extend(f"{name}:{problem}" for problem in mask_problems)
    for values, name in ((bend_deg, "bend_angle_deg"), (spread_deg, "spread_angle_deg")):
        finite = np.isfinite(values)
        if np.any(values[finite] < -ANGLE_TOLERANCE) or np.any(values[finite] > 180.0 + ANGLE_TOLERANCE):
            problems.append(f"{name}_outside_0_180")
    for values, name in ((bend_norm, "bend_normalized"), (spread_norm, "spread_normalized")):
        finite = np.isfinite(values)
        if np.any(values[finite] < -NORMALIZATION_TOLERANCE) or np.any(values[finite] > 1.0 + NORMALIZATION_TOLERANCE):
            problems.append(f"{name}_outside_0_1")
    for degrees, normalized, mask, name in (
        (bend_deg, bend_norm, bend_valid, "bend"),
        (spread_deg, spread_norm, spread_valid, "spread"),
    ):
        valid_values = mask & np.isfinite(degrees) & np.isfinite(normalized)
        if np.any(np.abs(normalized[valid_values] - degrees[valid_values] / 180.0) > NORMALIZATION_TOLERANCE):
            problems.append(f"{name}_normalization_disagreement")

    orientation = _orientation_audit(
        np.asarray(arrays["imu_rotation_matrix"]),
        np.asarray(arrays["imu_quaternion_wxyz"]),
        np.asarray(arrays["palm_imu_valid"], dtype=bool),
    )
    if orientation["quality_count"] != orientation["valid_count"]:
        problems.append("orientation_quality_violation")
    adc_details: dict[str, Any] = {}
    for adc_name, normalized, valid in (
        ("bend_adc_12bit", bend_norm, bend_valid),
        ("spread_adc_12bit", spread_norm, spread_valid),
    ):
        degrees = bend_deg if adc_name.startswith("bend_") else spread_deg
        detail, adc_problems = _adc_audit(arrays.get(adc_name), degrees, normalized, valid, adc_name)
        adc_details[adc_name] = detail
        problems.extend(adc_problems)

    tracking = np.asarray(arrays["tracking_state_code"])
    pose = np.isin(tracking, POSE_BEARING_CODES)
    hand_metrics = {
        "LEFT": _fraction(pose[:, 0]),
        "RIGHT": _fraction(pose[:, 1]),
        "BOTH": _fraction(pose[:, 0] & pose[:, 1]),
    }
    hand_metrics["ACTIVE"] = max(hand_metrics["LEFT"], hand_metrics["RIGHT"])
    manifest_frames = row.get("source_frame_count") or row.get("sequence_length")
    if manifest_frames and int(manifest_frames) != frames:
        problems.append(f"manifest_frame_count={manifest_frames},output_frame_count={frames}")
    if layout_result.get("contract_valid") is not True:
        problems.append("sensor_layout_contract_violation")
    metrics = {
        "adc": adc_details,
        "bend_valid_fraction": _fraction(bend_valid),
        "frames": frames,
        "geometry_available": False,
        "hand_availability_fraction": hand_metrics,
        "imu_valid_fraction": _fraction(np.asarray(arrays["palm_imu_valid"], dtype=bool)),
        "orientation_quality": orientation,
        "spread_valid_fraction": _fraction(spread_valid),
        "tracking_quality_fraction": hand_metrics["ACTIVE"],
    }
    return metrics, sorted(set(problems))


def _artifact_flags(run_root: Path, sample_id: str, layout_path: Path, metadata_path: Path) -> dict[str, Any]:
    paths = {
        "pose": run_root / "pose" / "raw" / sample_id / "wilor_raw.npz",
        "tracking": run_root / "tracking" / sample_id / "wilor_tracked.npz",
        "kinematics": run_root / "kinematics" / sample_id / "hand_kinematics.npz",
        "virtual_glove": run_root / "virtual_glove" / sample_id / "virtual_glove.npz",
        "sensor_layout": layout_path,
        "metadata": metadata_path,
    }
    mesh_candidates = []
    sample_root = run_root / "virtual_glove" / sample_id
    if sample_root.is_dir():
        for suffix in (".obj", ".ply", ".glb", ".gltf"):
            mesh_candidates.extend(sample_root.glob(f"*{suffix}"))
    flags = {name: path.is_file() for name, path in paths.items()}
    flags["surface_triangle_topology_available"] = bool(mesh_candidates)
    flags["embedded_mano_vertices_available"] = False
    if flags["pose"]:
        try:
            with np.load(paths["pose"], allow_pickle=False) as data:
                flags["embedded_mano_vertices_available"] = (
                    "vertices" in data.files and "vertices_keys" in data.files
                )
        except (OSError, ValueError, EOFError):
            flags["embedded_mano_vertices_available"] = False
    flags["tracked_landmarks_3d_available"] = False
    if flags["tracking"]:
        try:
            with np.load(paths["tracking"], allow_pickle=False) as data:
                flags["tracked_landmarks_3d_available"] = "landmarks_3d" in data.files
        except (OSError, ValueError, EOFError):
            flags["tracked_landmarks_3d_available"] = False
    return flags


def _audit_candidate(row: Mapping[str, str], run_root: Path) -> tuple[_Candidate | None, list[str]]:
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        return None, ["missing_sample_id"]
    try:
        descriptor = SequenceDescriptor.from_manifest_row(row, run_root)
    except ValueError as error:
        return None, [str(error)]
    npz_path = descriptor.absolute_path("virtual_glove")
    layout_path = npz_path.parent / "sensor_layout.json"
    metadata_path = npz_path.parent / "virtual_glove_meta.json"
    stage_path = npz_path.parent / "task008a_stage.json"
    if npz_path.parent.name != sample_id:
        return None, ["virtual_glove_path_sample_id_mismatch"]
    if row.get("virtual_glove_status") and row.get("virtual_glove_status") != "VIRTUAL_GLOVE_DONE":
        return None, [f"manifest_virtual_glove_status={row.get('virtual_glove_status')}"]
    if not npz_path.is_file():
        return None, [f"missing_virtual_glove={npz_path}"]
    layout_ok, layout = _layout_audit(layout_path)
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as error:
        return None, [f"cannot_read_virtual_glove={error}"]
    metrics, problems = _validate_shapes_and_values(arrays, row, layout)
    if not layout_ok and "sensor_layout_contract_violation" not in problems:
        problems.append("sensor_layout_contract_violation")
    flags = _artifact_flags(run_root, sample_id, layout_path, metadata_path)
    metrics["artifact_flags"] = flags
    metrics["geometry_components"] = {
        "pose": flags["pose"],
        "tracking": flags["tracking"],
        "kinematics": flags["kinematics"],
        "embedded_mano_vertices": flags["embedded_mano_vertices_available"],
        "tracked_landmarks_3d": flags["tracked_landmarks_3d_available"],
        "surface_triangle_topology": flags["surface_triangle_topology_available"],
    }
    metrics["geometry_available"] = all(flags[name] for name in ("pose", "tracking", "kinematics"))
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sample_id") not in {None, sample_id}:
                problems.append("metadata_sample_id_mismatch")
            if metadata.get("schema_version") not in {None, "virtual_glove_v1"}:
                problems.append("metadata_schema_version_mismatch")
            normalized_contract = metadata.get("representations", {}).get("normalized_ideal_sensor", {})
            if normalized_contract.get("fitted_to_dataset") is True:
                problems.append("metadata_run_specific_normalization")
            if normalized_contract.get("formula") and "180" not in str(normalized_contract["formula"]):
                problems.append("metadata_normalization_formula_mismatch")
            angle_range = metadata.get("contract", {}).get("angle_range_deg")
            if angle_range is not None:
                try:
                    normalized_range = list(angle_range)
                except TypeError:
                    normalized_range = None
                if normalized_range not in ([0.0, 180.0], [0, 180]):
                    problems.append("metadata_angle_range_mismatch")
            metrics["metadata_schema_version"] = metadata.get("schema_version")
            raw_source = metadata.get("source", {}).get("tracked_source", {}).get("raw_source", {})
            for key, expected in (
                ("manifest_sha256", row.get("manifest_sha256", "")),
                ("source_video", row.get("source_relative_path", "")),
                ("source_video_sha256", row.get("source_sha256", "")),
            ):
                if expected and raw_source.get(key) not in {None, expected}:
                    problems.append(f"metadata_{key}_mismatch")
        except (OSError, json.JSONDecodeError):
            problems.append("invalid_virtual_glove_metadata")
    if stage_path.is_file():
        try:
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            for key, expected in (
                ("sample_id", sample_id),
                ("manifest_sha256", row.get("manifest_sha256", "")),
                ("source_relative_path", row.get("source_relative_path", "")),
                ("source_sha256", row.get("source_sha256", "")),
            ):
                if expected and stage.get(key) not in {None, expected}:
                    problems.append(f"stage_{key}_mismatch")
        except (OSError, json.JSONDecodeError):
            problems.append("invalid_task008_stage_metadata")
    metrics["artifact_completeness"] = sum(
        flags[name] for name in ("pose", "tracking", "kinematics", "virtual_glove", "sensor_layout", "metadata")
    ) / 6.0
    metrics["layout_contract"] = {
        "contract_valid": layout.get("contract_valid", False),
        "hall_count": layout.get("hall_count", 0),
        "imu_count": layout.get("imu_count", 0),
        "sensor_count": layout.get("sensor_count", 0),
    }
    if problems:
        return None, sorted(set(problems))
    return _Candidate(dict(row), descriptor, int(metrics.pop("frames", descriptor.sequence_length)), metrics), []


def _duration_score(length: int, lengths: list[int]) -> tuple[float, float, float, float]:
    values = np.asarray(lengths, dtype=np.float64)
    median = float(np.median(values))
    q1, q3 = (float(item) for item in np.percentile(values, (25, 75)))
    scale = max(q3 - q1, 1.0)
    score = max(0.0, 1.0 - abs(float(length) - median) / scale)
    return score, median, q1, q3


def _assign_scores(candidates: list[_Candidate]) -> None:
    by_class: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate.sign_id].append(candidate)
    for class_candidates in by_class.values():
        lengths = [candidate.sequence_length for candidate in class_candidates]
        expected_two_hand = float(
            np.mean([candidate.metrics["hand_availability_fraction"]["BOTH"] for candidate in class_candidates])
        ) >= 0.5
        for candidate in class_candidates:
            hand = candidate.metrics["hand_availability_fraction"]
            hand_score = hand["BOTH"] if expected_two_hand else hand["ACTIVE"]
            duration, median, q1, q3 = _duration_score(candidate.sequence_length, lengths)
            geometry = 1.0 if candidate.metrics["geometry_available"] else 0.0
            completeness = float(candidate.metrics["artifact_completeness"])
            components = {
                "artifact_completeness": completeness,
                "geometry_availability": geometry,
                "hand_tracking_quality": hand_score,
                "bend_validity": candidate.metrics["bend_valid_fraction"],
                "spread_validity": candidate.metrics["spread_valid_fraction"],
                "imu_validity": candidate.metrics["imu_valid_fraction"],
                "duration_proximity": duration,
            }
            weights = {
                "artifact_completeness": 0.10,
                "geometry_availability": 0.10,
                "hand_tracking_quality": 0.20,
                "bend_validity": 0.25,
                "spread_validity": 0.15,
                "imu_validity": 0.15,
                "duration_proximity": 0.05,
            }
            candidate.score = sum(components[name] * weights[name] for name in weights)
            candidate.metrics["class_profile"] = {
                "duration_median": median,
                "duration_q1": q1,
                "duration_q3": q3,
                "expected_two_hand": expected_two_hand,
                "hand_score_basis": "BOTH" if expected_two_hand else "ACTIVE",
            }
            candidate.metrics["duration_proximity"] = duration
            candidate.metrics["score_components"] = components


def _ranking(candidate: _Candidate) -> tuple[float, str]:
    return (-candidate.score, candidate.sample_id)


def _selection_reason(candidate: _Candidate, mode: str) -> str:
    basis = candidate.metrics["class_profile"]["hand_score_basis"]
    return (
        f"{mode}: highest weighted contract-quality score; complete artifact/layout audit, "
        f"{basis.lower()} hand-tracking basis, per-channel bend/spread/IMU validity, "
        "median-near duration, then sample_id ascending tie-break."
    )


def _candidate_reference(candidate: _Candidate) -> dict[str, Any]:
    """Keep random-selection pools compact while retaining loadable paths."""

    row = candidate.row
    return {
        "character": row["label_ar"],
        "label_index": int(row["label_index"]),
        "official_partition": row.get("official_partition", ""),
        "repetition_id": row.get("repetition_id", ""),
        "sample_id": candidate.sample_id,
        "score": float(candidate.score),
        "sequence_length": candidate.sequence_length,
        "sign_id": candidate.sign_id,
        "signer_id": candidate.signer_id,
        "manifest_sha256": row.get("manifest_sha256", ""),
        "source_relative_path": row.get("source_relative_path", ""),
        "source_sha256": row.get("source_sha256", ""),
        "virtual_glove_relative_path": candidate.descriptor.virtual_glove_relative_path,
    }


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise CatalogBuildError(f"manifest is empty: {path}")
    required = {"sample_id", "sign_id", "label_ar", "label_index", "virtual_glove_relative_path"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise CatalogBuildError(f"virtual-glove manifest is missing {missing}")
    return rows


def _authoritative_labels(path: Path) -> dict[str, tuple[str, int]]:
    records = validate_core28_records(load_label_records(path))
    return {f"{record.sign_id:04d}": (record.label_ar, index) for index, record in enumerate(records)}


def build_catalog_payload(
    *,
    manifest_path: str | Path,
    run_root: str | Path,
    labels_path: str | Path,
) -> dict[str, Any]:
    """Audit source rows and return a deterministic catalog payload."""

    manifest = Path(manifest_path).resolve()
    root = Path(run_root).expanduser().resolve()
    labels = Path(labels_path).resolve()
    authoritative = _authoritative_labels(labels)
    rows = _manifest_rows(manifest)
    candidates: list[_Candidate] = []
    rejected: dict[str, list[str]] = {}
    seen_samples: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id in seen_samples:
            raise CatalogBuildError(f"duplicate sample_id in manifest: {sample_id}")
        seen_samples.add(sample_id)
        sign_id = str(row.get("sign_id", "")).zfill(4)
        if sign_id not in authoritative:
            rejected[sample_id] = [f"not_core28_sign_id={sign_id}"]
            continue
        expected_label, expected_index = authoritative[sign_id]
        if row.get("label_ar") != expected_label or int(row.get("label_index", -1)) != expected_index:
            raise CatalogBuildError(
                f"manifest label mapping mismatch for {sample_id}: "
                f"got ({row.get('label_ar')!r}, {row.get('label_index')!r}), "
                f"expected ({expected_label!r}, {expected_index})"
            )
        row["sign_id"] = sign_id
        candidate, problems = _audit_candidate(row, root)
        if candidate is None:
            rejected[sample_id] = problems
        else:
            candidates.append(candidate)
    _assign_scores(candidates)

    by_class: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate.sign_id].append(candidate)
    missing_classes = sorted(set(authoritative) - set(by_class))
    if missing_classes:
        details = {key: rejected.get(key, []) for key in sorted(rejected)[:10]}
        raise CatalogBuildError(f"no valid exemplar candidates for SignIDs {missing_classes}; rejected={details}")

    canonical: list[ExemplarEntry] = []
    for sign_id, (character, label_index) in sorted(authoritative.items(), key=lambda item: item[1][1]):
        ranked = sorted(by_class[sign_id], key=_ranking)
        selected = ranked[0]
        canonical.append(selected.entry(reason=_selection_reason(selected, "canonical")))

    signer_exemplars: dict[str, list[ExemplarEntry]] = {}
    available_signers = sorted({candidate.signer_id for candidate in candidates})
    for signer in available_signers:
        selected_for_signer: list[ExemplarEntry] = []
        for sign_id, (_, _) in sorted(authoritative.items(), key=lambda item: item[1][1]):
            signer_candidates = [candidate for candidate in by_class[sign_id] if candidate.signer_id == signer]
            if signer_candidates:
                selected = sorted(signer_candidates, key=_ranking)[0]
                selected_for_signer.append(
                    selected.entry(reason=_selection_reason(selected, signer))
                )
        signer_exemplars[signer] = selected_for_signer

    candidate_index = {
        sign_id: [_candidate_reference(candidate) for candidate in sorted(values, key=lambda item: item.sample_id)]
        for sign_id, values in sorted(by_class.items())
    }
    class_profiles: dict[str, Any] = {}
    for sign_id, values in sorted(by_class.items()):
        lengths = [candidate.sequence_length for candidate in values]
        both_rate = float(np.mean([candidate.metrics["hand_availability_fraction"]["BOTH"] for candidate in values]))
        class_profiles[sign_id] = {
            "candidate_count": len(values),
            "duration_median": float(np.median(lengths)),
            "duration_max": max(lengths),
            "duration_min": min(lengths),
            "expected_two_hand": both_rate >= 0.5,
            "mean_both_hand_fraction": both_rate,
            "signer_counts": dict(sorted(Counter(candidate.signer_id for candidate in values).items())),
        }

    source = {
        "labels_manifest": str(labels.relative_to(Path.cwd())) if labels.is_relative_to(Path.cwd()) else str(labels),
        "labels_manifest_sha256": sha256_file(labels),
        "run_root": str(root),
        "virtual_glove_manifest": str(manifest.relative_to(Path.cwd())) if manifest.is_relative_to(Path.cwd()) else str(manifest),
        "virtual_glove_manifest_sha256": sha256_file(manifest),
        "rows_audited": len(rows),
        "candidates_accepted": len(candidates),
        "candidates_rejected": len(rejected),
        "rejected_samples": {sample_id: reasons for sample_id, reasons in sorted(rejected.items())},
    }
    selection_policy = {
        "algorithm_version": "task007b_quality_rank_v1",
        "eligibility": [
            "required virtual_glove.npz arrays and exact [F,2,5,3]/[F,2,4]/[F,2,3,3]/[F,2,4] shapes",
            "strict frame/timestamp order and fixed deg/180 normalized relation",
            "per-channel invalid values remain NaN and valid values remain finite",
            "machine-readable 15 bend + 4 spread + 1 palm IMU layout with H/IMU markers",
            "geometry terminology distinguishes embedded MANO vertices, tracked 21-joint landmarks, and surface topology",
            "valid orientation matrices/quaternions and optional ADC transfer agreement",
        ],
        "weights": {
            "artifact_completeness": 0.10,
            "geometry_availability": 0.10,
            "hand_tracking_quality": 0.20,
            "bend_validity": 0.25,
            "spread_validity": 0.15,
            "imu_validity": 0.15,
            "duration_proximity": 0.05,
        },
        "hand_policy": (
            "If a class is predominantly two-hand, use the BOTH-frame fraction. "
            "Otherwise use the best active-hand fraction, so a legitimate one-hand sign "
            "is not penalized for the absent physical hand."
        ),
        "tie_break": "descending score, then ascending sample_id",
        "duration_policy": "1 - distance from class median divided by class IQR (minimum scale 1), floored at 0",
        "random_policy": "candidate_index sorted by sample_id; per-SignID SHA-256-derived RNG seed; explicit rng_seed required",
    }
    return {
        "candidate_index": candidate_index,
        "catalog_version": CATALOG_VERSION,
        "class_profiles": class_profiles,
        "entries": [entry.to_dict() for entry in canonical],
        "selection_policy": selection_policy,
        "signer_exemplars": {
            signer: [entry.to_dict() for entry in entries]
            for signer, entries in sorted(signer_exemplars.items())
        },
        "source": source,
    }


def build_catalog(
    *,
    manifest_path: str | Path,
    run_root: str | Path,
    labels_path: str | Path,
    output_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_catalog_payload(
        manifest_path=manifest_path,
        run_root=run_root,
        labels_path=labels_path,
    )
    if output_path is not None:
        write_catalog(output_path, payload)
    if output_csv_path is not None:
        write_catalog_csv(output_csv_path, payload)
    return payload
