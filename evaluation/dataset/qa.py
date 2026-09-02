"""Extractor-independent QA for external TASK-008A sensor runs.

The checker reads stage artifacts; it never loads WiLoR and never rewrites any
stage.  It is intentionally conservative: a missing sidecar, provenance
mismatch, frame alignment mismatch or malformed array is a sample failure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .manifest import load_manifest, manifest_sha256, sha256_file
from .orchestrator import RunPaths, STAGES, validate_stage_artifact


class DatasetQAError(ValueError):
    """The source manifest or external stage output violates the contract."""


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {name: None for name in ("min", "p1", "median", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "p1": int(math.floor(np.percentile(array, 1))),
        "median": int(math.floor(np.percentile(array, 50))),
        "p95": int(math.floor(np.percentile(array, 95))),
        "p99": int(math.floor(np.percentile(array, 99))),
        "max": int(array.max()),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _stage_sidecar(paths: RunPaths, stage: str, sample_id: str) -> dict[str, Any]:
    path = paths.stage_sidecar(stage, sample_id)
    if not path.is_file():
        raise DatasetQAError(f"Missing {stage} provenance sidecar for {sample_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetQAError(f"Malformed {stage} provenance sidecar for {sample_id}") from error


def _assert_equal(name: str, left: np.ndarray, right: np.ndarray, sample_id: str) -> None:
    if not np.array_equal(left, right):
        raise DatasetQAError(f"{sample_id}: {name} differs between pipeline stages")


def _validate_sample(
    row: Mapping[str, str],
    *,
    data_root: Path,
    paths: RunPaths,
    manifest_hash_value: str,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    source = data_root / row["source_relative_path"]
    if not source.is_file() or source.stat().st_size <= 0:
        raise DatasetQAError(f"{sample_id}: source video is missing or empty")
    declared_hash = row.get("source_sha256", "").lower()
    actual_hash = sha256_file(source)
    if declared_hash and actual_hash != declared_hash:
        raise DatasetQAError(f"{sample_id}: source SHA-256 mismatch")

    for stage in STAGES:
        if validate_stage_artifact(
            paths, stage, row, manifest_hash_value, source_sha256=actual_hash
        ) is None:
            raise DatasetQAError(f"{sample_id}: {stage} artifact is missing or invalid")
        sidecar = _stage_sidecar(paths, stage, sample_id)
        if sidecar.get("source_sha256") != actual_hash:
            raise DatasetQAError(f"{sample_id}: {stage} source provenance mismatch")

    pose_dir = paths.pose / sample_id
    tracking_dir = paths.tracking / sample_id
    kinematics_dir = paths.kinematics / sample_id
    glove_dir = paths.virtual_glove / sample_id
    pose = _load_npz(pose_dir / "wilor_raw.npz")
    tracking = _load_npz(tracking_dir / "wilor_tracked.npz")
    kinematics = _load_npz(kinematics_dir / "hand_kinematics.npz")
    glove = _load_npz(glove_dir / "virtual_glove.npz")
    frame_index = np.asarray(kinematics["frame_index"])
    timestamps = np.asarray(kinematics["timestamp_seconds"])
    frames = int(frame_index.shape[0])
    if frames <= 0:
        raise DatasetQAError(f"{sample_id}: zero output frames")
    if row.get("frame_count") and int(row["frame_count"]) != frames:
        raise DatasetQAError(
            f"{sample_id}: frame count {frames} != manifest {row['frame_count']}"
        )
    _assert_equal("frame_index", np.asarray(pose["frame_index"]), frame_index, sample_id)
    _assert_equal("timestamp_seconds", np.asarray(pose["timestamp_seconds"]), timestamps, sample_id)
    _assert_equal("frame_index", np.asarray(tracking["frame_index"]), frame_index, sample_id)
    _assert_equal("frame_index", np.asarray(glove["frame_index"]), frame_index, sample_id)
    _assert_equal("timestamp_seconds", np.asarray(tracking["timestamp_seconds"]), timestamps, sample_id)
    _assert_equal("timestamp_seconds", np.asarray(glove["timestamp_seconds"]), timestamps, sample_id)
    _assert_equal("tracking_state_code", np.asarray(tracking["state_code"]), np.asarray(kinematics["tracking_state_code"]), sample_id)
    _assert_equal("source_raw_detection_index", np.asarray(tracking["raw_detection_index"]), np.asarray(kinematics["source_raw_detection_index"]), sample_id)
    _assert_equal("tracking_state_code", np.asarray(glove["tracking_state_code"]), np.asarray(kinematics["tracking_state_code"]), sample_id)
    _assert_equal(
        "source_raw_detection_index",
        np.asarray(glove["source_raw_detection_index"]),
        np.asarray(kinematics["source_raw_detection_index"]),
        sample_id,
    )
    _assert_equal(
        "source_valid_kinematics",
        np.asarray(glove["source_valid_kinematics"]),
        np.asarray(kinematics["valid_kinematics"]),
        sample_id,
    )
    _assert_equal(
        "source_valid_palm_frame",
        np.asarray(glove["source_valid_palm_frame"]),
        np.asarray(kinematics["valid_palm_frame"]),
        sample_id,
    )

    expected_shapes = {
        "bend_normalized": (frames, 2, 5, 3),
        "spread_normalized": (frames, 2, 4),
        "bend_valid": (frames, 2, 5, 3),
        "spread_valid": (frames, 2, 4),
        "imu_rotation_matrix": (frames, 2, 3, 3),
        "imu_quaternion_wxyz": (frames, 2, 4),
        "palm_imu_valid": (frames, 2),
    }
    for name, shape in expected_shapes.items():
        if tuple(glove[name].shape) != shape:
            raise DatasetQAError(f"{sample_id}: {name} shape {glove[name].shape} != {shape}")

    bend_deg = np.asarray(glove["bend_angle_deg"], dtype=np.float64)
    spread_deg = np.asarray(glove["spread_angle_deg"], dtype=np.float64)
    bend_norm = np.asarray(glove["bend_normalized"], dtype=np.float64)
    spread_norm = np.asarray(glove["spread_normalized"], dtype=np.float64)
    bend_valid = np.asarray(glove["bend_valid"], dtype=bool)
    spread_valid = np.asarray(glove["spread_valid"], dtype=bool)
    imu_valid = np.asarray(glove["palm_imu_valid"], dtype=bool)
    if not np.allclose(bend_norm[bend_valid], (bend_deg / 180.0)[bend_valid], rtol=0, atol=2e-7):
        raise DatasetQAError(f"{sample_id}: bend normalization mismatch")
    if not np.allclose(spread_norm[spread_valid], (spread_deg / 180.0)[spread_valid], rtol=0, atol=2e-7):
        raise DatasetQAError(f"{sample_id}: spread normalization mismatch")
    if not np.array_equal(bend_valid, np.isfinite(bend_deg)):
        raise DatasetQAError(f"{sample_id}: bend mask does not preserve finite channels")
    if not np.array_equal(spread_valid, np.isfinite(spread_deg)):
        raise DatasetQAError(f"{sample_id}: spread mask does not preserve finite channels")
    if not np.isfinite(bend_norm[bend_valid]).all() or not np.isfinite(spread_norm[spread_valid]).all():
        raise DatasetQAError(f"{sample_id}: valid normalized channel is non-finite")
    if (np.any(bend_norm[bend_valid] < 0) or np.any(bend_norm[bend_valid] > 1) or
            np.any(spread_norm[spread_valid] < 0) or np.any(spread_norm[spread_valid] > 1)):
        raise DatasetQAError(f"{sample_id}: normalized channel is outside [0, 1]")
    if not np.isfinite(glove["imu_rotation_matrix"][imu_valid]).all() or not np.isfinite(glove["imu_quaternion_wxyz"][imu_valid]).all():
        raise DatasetQAError(f"{sample_id}: valid IMU orientation is non-finite")
    for matrix, quaternion in zip(
        glove["imu_rotation_matrix"][imu_valid], glove["imu_quaternion_wxyz"][imu_valid]
    ):
        if np.max(np.abs(matrix.T @ matrix - np.eye(3))) > 1e-4 or np.linalg.det(matrix) <= 0:
            raise DatasetQAError(f"{sample_id}: invalid IMU rotation matrix")
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
            raise DatasetQAError(f"{sample_id}: invalid IMU quaternion norm")

    strict_valid = np.asarray(glove["source_valid_kinematics"], dtype=bool)
    partial = ~strict_valid & (bend_valid.any(axis=(2, 3)) | spread_valid.any(axis=2) | imu_valid)
    retained = int(bend_valid[~strict_valid].sum() + spread_valid[~strict_valid].sum() + imu_valid[~strict_valid].sum())
    sidecar = _stage_sidecar(paths, "VIRTUAL_GLOVE", sample_id)
    return {
        "sample_id": sample_id,
        "sign_id": row["sign_id"],
        "label_ar": row["label_ar"],
        "signer_id": row["signer_id"],
        "official_partition": row["official_partition"],
        "frames": frames,
        "bend_valid": int(bend_valid.sum()),
        "bend_total": int(bend_valid.size),
        "spread_valid": int(spread_valid.sum()),
        "spread_total": int(spread_valid.size),
        "imu_valid": int(imu_valid.sum()),
        "imu_total": int(imu_valid.size),
        "partial_instances": int(partial.sum()),
        "retained_channels_on_strict_invalid": retained,
        "contract_violation_count": int(sidecar.get("contract_violation_count", 0)),
    }


def validate_run(
    manifest: str | Path,
    data_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    """Validate all four external stages and return a compact summary."""

    rows = load_manifest(manifest)
    if not rows:
        raise DatasetQAError("Cannot validate an empty/schema-only Core-28 manifest")
    root = Path(data_root).resolve()
    paths = RunPaths(Path(run_root).resolve())
    manifest_hash_value = manifest_sha256(manifest)
    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    stage_counts = {
        stage.lower(): {"success": 0, "failed": 0} for stage in STAGES
    }
    for row in rows:
        source = root / row["source_relative_path"]
        actual_source_hash = None
        if source.is_file() and source.stat().st_size > 0:
            actual_source_hash = sha256_file(source)
        for stage in STAGES:
            valid = actual_source_hash is not None and validate_stage_artifact(
                paths,
                stage,
                row,
                manifest_hash_value,
                source_sha256=actual_source_hash,
            ) is not None
            stage_counts[stage.lower()]["success" if valid else "failed"] += 1
        try:
            sample_results.append(
                _validate_sample(
                    row,
                    data_root=root,
                    paths=paths,
                    manifest_hash_value=manifest_hash_value,
                )
            )
        except (DatasetQAError, OSError, ValueError) as error:
            failures.append({"sample_id": row["sample_id"], "error": str(error)})

    success = sample_results
    frames = sum(item["frames"] for item in success)
    summary = {
        "schema_version": "task008a_dataset_qa_v1",
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_hash_value,
        "data_root": str(root),
        "run_root": str(paths.root),
        "requested_samples": len(rows),
        "successful_samples": len(success),
        "failed_samples": len(failures),
        "failures": failures,
        "stages": stage_counts,
        "frames": frames,
        "hand_instances": frames * 2,
        "validity": {
            "bend_valid": sum(item["bend_valid"] for item in success),
            "bend_total": sum(item["bend_total"] for item in success),
            "spread_valid": sum(item["spread_valid"] for item in success),
            "spread_total": sum(item["spread_total"] for item in success),
            "imu_valid": sum(item["imu_valid"] for item in success),
            "imu_total": sum(item["imu_total"] for item in success),
            "partial_instances": sum(item["partial_instances"] for item in success),
            "retained_channels_on_strict_invalid": sum(item["retained_channels_on_strict_invalid"] for item in success),
        },
        "by_label": {
            key: {"samples": sum(item["label_ar"] == key for item in success), "frames": sum(item["frames"] for item in success if item["label_ar"] == key)}
            for key in sorted({row["label_ar"] for row in rows})
        },
        "by_signer": {
            key: {"samples": sum(item["signer_id"] == key for item in success), "frames": sum(item["frames"] for item in success if item["signer_id"] == key)}
            for key in sorted({row["signer_id"] for row in rows})
        },
        "sequence_length": _percentiles([item["frames"] for item in success]),
        "contract_violations": sum(item["contract_violation_count"] for item in success),
        "provenance_mismatches": len(failures),
        "sample_results": sample_results,
    }
    return summary
