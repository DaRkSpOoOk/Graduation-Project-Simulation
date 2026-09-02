"""Hard input checks for the frozen TASK-005 pilot artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.kinematics_qa.contract import (
    KINEMATICS_META_NAME,
    KINEMATICS_NPZ_NAME,
    TRACKED_META_NAME,
    TRACKED_NPZ_NAME,
    ContractError,
    list_sample_ids,
    load_kinematics_sample,
    validate_sample_contract,
)


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "datasets" / "manifests" / "karsl_milestone1_pilot.csv"
FROZEN_KINEMATICS_COMMIT = "564167420c7f5b4f12197fe36e7d2b59ae08ace0"
EXPECTED_SAMPLE_COUNT = 18
EXPECTED_TOTAL_FRAMES = 894


class PilotInputError(ContractError):
    """Raised when a comparison input is not the exact frozen pilot."""


def _manifest_sample_ids(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        raise PilotInputError(f"pilot manifest does not exist: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise PilotInputError(f"pilot manifest has missing sample_id values: {manifest_path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise PilotInputError("pilot manifest contains duplicate sample_id values")
    return sample_ids


def _load_tracked_arrays(run_dir: Path, sample_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_dir = run_dir / sample_id
    npz_path = sample_dir / TRACKED_NPZ_NAME
    meta_path = sample_dir / TRACKED_META_NAME
    if not npz_path.is_file() or not meta_path.is_file():
        raise PilotInputError(f"{sample_id}: missing tracked NPZ or metadata")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads(meta_path.read_text(encoding="utf-8"))


def validate_frozen_pilot_inputs(
    tracked_run: str | Path,
    kinematics_run: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    expected_implementation_commit: str = FROZEN_KINEMATICS_COMMIT,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    expected_total_frames: int = EXPECTED_TOTAL_FRAMES,
) -> dict[str, Any]:
    """Hard-validate exact sample/frame provenance before comparison metrics.

    The two run directories are explicit inputs.  Only the exact file names
    ``wilor_tracked.npz`` and ``hand_kinematics.npz`` are considered; a
    detector-only Phase-A directory cannot be selected by a broad NPZ glob.
    """

    tracked_path = Path(tracked_run)
    kinematics_path = Path(kinematics_run)
    manifest = Path(manifest_path)
    if not tracked_path.is_dir():
        raise PilotInputError(f"tracked run does not exist: {tracked_path}")
    if not kinematics_path.is_dir():
        raise PilotInputError(f"kinematics run does not exist: {kinematics_path}")

    expected_ids = _manifest_sample_ids(manifest)
    if len(expected_ids) != expected_sample_count:
        raise PilotInputError(
            f"manifest sample count is {len(expected_ids)}, expected {expected_sample_count}"
        )
    expected_set = set(expected_ids)
    tracked_ids = list_sample_ids(tracked_path, TRACKED_NPZ_NAME)
    kinematics_ids = list_sample_ids(kinematics_path, KINEMATICS_NPZ_NAME)
    if set(tracked_ids) != expected_set:
        raise PilotInputError(
            f"tracked sample IDs do not equal manifest: missing={sorted(expected_set - set(tracked_ids))}, "
            f"extra={sorted(set(tracked_ids) - expected_set)}"
        )
    if set(kinematics_ids) != expected_set:
        raise PilotInputError(
            f"kinematics sample IDs do not equal manifest: missing={sorted(expected_set - set(kinematics_ids))}, "
            f"extra={sorted(set(kinematics_ids) - expected_set)}"
        )

    tracked_frames = 0
    kinematics_frames = 0
    metadata_failures: list[str] = []
    frame_counts: dict[str, int] = {}
    for sample_id in expected_ids:
        tracked_arrays, tracked_meta = _load_tracked_arrays(tracked_path, sample_id)
        tracked_frame_index = np.asarray(tracked_arrays.get("frame_index"))
        if tracked_frame_index.ndim != 1:
            raise PilotInputError(f"{sample_id}: tracked frame_index is not rank 1")
        tracked_count = int(tracked_frame_index.shape[0])
        tracked_frames += tracked_count

        sample = load_kinematics_sample(kinematics_path, sample_id)
        contract = validate_sample_contract(sample)
        if not contract["passed"]:
            metadata_failures.extend(f"{sample_id}: {failure}" for failure in contract["failures"])
        kine_frame_index = np.asarray(sample.arrays["frame_index"])
        kine_count = int(kine_frame_index.shape[0])
        kinematics_frames += kine_count
        frame_counts[sample_id] = kine_count
        if tracked_count != kine_count or not np.array_equal(tracked_frame_index, kine_frame_index):
            raise PilotInputError(f"{sample_id}: tracked and kinematics frame indices differ")

        if tracked_meta.get("sample_id") != sample_id:
            metadata_failures.append(f"{sample_id}: tracked metadata sample_id mismatch")
        if sample.metadata.get("sample_id") != sample_id:
            metadata_failures.append(f"{sample_id}: kinematics metadata sample_id mismatch")
        if sample.metadata.get("implementation_commit") != expected_implementation_commit:
            metadata_failures.append(
                f"{sample_id}: implementation_commit={sample.metadata.get('implementation_commit')!r}, "
                f"expected {expected_implementation_commit!r}"
            )
        source = sample.metadata.get("source", {})
        if not isinstance(source, dict) or source.get("tracked_sample_id") != sample_id:
            metadata_failures.append(f"{sample_id}: kinematics source does not identify tracked sample")

    if metadata_failures:
        raise PilotInputError("pilot metadata/contract failures: " + "; ".join(metadata_failures))
    if tracked_frames != expected_total_frames or kinematics_frames != expected_total_frames:
        raise PilotInputError(
            f"pilot frame total mismatch: tracked={tracked_frames}, kinematics={kinematics_frames}, "
            f"expected={expected_total_frames}"
        )

    return {
        "passed": True,
        "manifest": str(manifest),
        "sample_count": len(expected_ids),
        "sample_ids": expected_ids,
        "tracked_frames": tracked_frames,
        "kinematics_frames": kinematics_frames,
        "frame_counts": frame_counts,
        "implementation_commit": expected_implementation_commit,
    }
