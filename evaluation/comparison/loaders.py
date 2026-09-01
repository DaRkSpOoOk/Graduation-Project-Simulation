"""Strict loaders for the two frozen raw-output formats.

The comparison layer accepts explicit paths from the shared manifest. It never
glob-selects ``runs/wilor*`` because that could silently ingest the historical
detector-only Phase-A output.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common_contract import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MEDIAPIPE_MODEL_SHA256,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_TOTAL_FRAMES,
    HandRecord,
    normalize_label,
    reconstructed_hand,
)


class InputValidationError(RuntimeError):
    """Raised when comparison inputs are absent, ambiguous, or malformed."""


@dataclass(frozen=True, slots=True)
class ManifestContract:
    path: Path
    rows: tuple[dict[str, str], ...]
    video_paths: dict[str, Path]
    sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row["sample_id"] for row in self.rows)


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    system: str
    run_dir: Path
    records_by_sample: dict[str, tuple[HandRecord, ...]]
    frame_counts: dict[str, int]
    metadata: dict[str, Any]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.records_by_sample)

    @property
    def total_frames(self) -> int:
        return sum(self.frame_counts.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sample_ids() -> set[str]:
    return {
        f"karsl_test_s{signer}_sign{sign}_repfirst"
        for signer in ("01", "02", "03")
        for sign in ("0171", "0172", "0173", "0174", "0175", "0176")
    }


def _read_json_scalar(array: Any) -> Any:
    value = array.item() if getattr(array, "ndim", None) == 0 else array
    return json.loads(str(value))


def _read_json_row(array: Any, index: int, *, allow_empty: bool = False) -> Any:
    raw = str(array[index])
    if not raw and allow_empty:
        return None
    if not raw:
        raise InputValidationError(f"Empty JSON row at index {index}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise InputValidationError(f"Invalid JSON row at index {index}: {error}") from error


def validate_manifest(
    manifest_path: str | Path,
    *,
    video_root: str | Path | None = None,
    expected_sha256: str | None = EXPECTED_MANIFEST_SHA256,
    verify_video_checksums: bool = True,
) -> ManifestContract:
    """Validate the exact 18-row pilot contract and optionally its videos."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise InputValidationError(f"Manifest not found: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise InputValidationError(
            f"Manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_id",
        "local_relative_path",
        "checksum_sha256",
        "sign_id",
        "signer_id",
        "split",
        "repetition_id",
    }
    if not rows or not required <= set(rows[0]):
        raise InputValidationError(f"Manifest lacks required columns: {sorted(required)}")
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise InputValidationError(f"Expected {EXPECTED_SAMPLE_COUNT} rows, got {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != _expected_sample_ids():
        raise InputValidationError("Manifest sample IDs are not the frozen 18-clip pilot set")
    for row in rows:
        if row.get("split") != "test" or row.get("repetition_id") != "lexicographically_first_valid_mp4":
            raise InputValidationError(f"Manifest selection rule drifted for {row['sample_id']}")
        relative = Path(row["local_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise InputValidationError(f"Unsafe manifest video path: {relative}")

    root = Path(video_root).expanduser().resolve() if video_root is not None else None
    video_paths: dict[str, Path] = {}
    if root is not None:
        for row in rows:
            video_path = root / row["local_relative_path"]
            if not video_path.is_file():
                raise InputValidationError(f"Manifest video not found: {video_path}")
            if verify_video_checksums:
                expected = row["checksum_sha256"].strip()
                if not expected:
                    raise InputValidationError(f"Manifest checksum is empty: {row['sample_id']}")
                actual = sha256_file(video_path)
                if actual != expected:
                    raise InputValidationError(
                        f"Video checksum mismatch for {row['sample_id']}: expected {expected}, got {actual}"
                    )
            video_paths[row["sample_id"]] = video_path
    return ManifestContract(path=path, rows=tuple(rows), video_paths=video_paths, sha256=actual_sha256)


def _manifest_by_sample(contract: ManifestContract) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in contract.rows}


def _expected_frame_counts_from_inspection(path: Path, sample_ids: Iterable[str]) -> dict[str, int]:
    if not path.is_file():
        raise InputValidationError(f"Video inspection output not found: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputValidationError(f"Invalid video inspection JSON: {path}") from error
    if not isinstance(rows, list):
        raise InputValidationError("Video inspection JSON must contain a list of rows")
    expected_ids = set(sample_ids)
    observed_ids = [row.get("sample_id") for row in rows if isinstance(row, dict)]
    if len(observed_ids) != len(rows) or len(observed_ids) != len(expected_ids) or set(observed_ids) != expected_ids:
        raise InputValidationError("Video inspection sample IDs do not match the manifest")
    result: dict[str, int] = {}
    for row in rows:
        if not row.get("decoder_success") or not row.get("checksum_matches_manifest"):
            raise InputValidationError(f"Video inspection failed for {row.get('sample_id')}")
        count = int(row.get("decoded_frame_count", 0))
        if count <= 0 or row.get("frame_count_mismatch"):
            raise InputValidationError(f"Invalid decoded frame count for {row.get('sample_id')}")
        result[row["sample_id"]] = count
    if sum(result.values()) != EXPECTED_TOTAL_FRAMES:
        raise InputValidationError(
            f"Expected {EXPECTED_TOTAL_FRAMES} inspected frames, got {sum(result.values())}"
        )
    return result


def _load_mediapipe_npz(path: Path, sample_id: str) -> tuple[tuple[HandRecord, ...], int, dict[str, Any]]:
    if not path.is_file():
        raise InputValidationError(f"MediaPipe raw NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_indices",
            "hand_landmarks_image",
            "hand_landmarks_world",
            "hand_present",
            "handedness_labels",
            "handedness_scores",
            "metadata_json",
        }
        if not required <= set(data.files):
            raise InputValidationError(f"MediaPipe NPZ missing keys: {sorted(required - set(data.files))}")
        frame_indices = np.asarray(data["frame_indices"])
        image = np.asarray(data["hand_landmarks_image"])
        world = np.asarray(data["hand_landmarks_world"])
        present = np.asarray(data["hand_present"], dtype=bool)
        labels = np.asarray(data["handedness_labels"])
        scores = np.asarray(data["handedness_scores"], dtype=np.float64)
        if frame_indices.ndim != 1 or image.ndim != 4 or image.shape[1:] != (2, 21, 3):
            raise InputValidationError(f"Unexpected MediaPipe NPZ shape in {path}")
        if world.shape != image.shape or present.shape != image.shape[:2] or labels.shape != present.shape or scores.shape != present.shape:
            raise InputValidationError(f"Inconsistent MediaPipe NPZ arrays in {path}")
        if not np.array_equal(frame_indices, np.arange(len(frame_indices), dtype=frame_indices.dtype)):
            raise InputValidationError(f"MediaPipe frame indices are not contiguous in {path}")
        metadata = _read_json_scalar(data["metadata_json"])
        if metadata.get("sample_id") != sample_id or metadata.get("stage") != "raw_pose":
            raise InputValidationError(f"MediaPipe metadata identity/stage mismatch in {path}")
        if metadata.get("running_mode") != "VIDEO" or int(metadata.get("num_hands", 0)) != 2:
            raise InputValidationError(f"MediaPipe metadata does not describe the frozen VIDEO/2-hand run: {path}")
        records: list[HandRecord] = []
        for frame_position, frame_index in enumerate(frame_indices):
            for hand_index in range(2):
                score = float(scores[frame_position, hand_index])
                records.append(
                    HandRecord(
                        system="mediapipe",
                        frame_index=int(frame_index),
                        hand_present=bool(present[frame_position, hand_index]),
                        handedness_label=normalize_label(labels[frame_position, hand_index]),
                        confidence=score if np.isfinite(score) else None,
                        detection_confidence=None,
                        image_landmarks=image[frame_position, hand_index].copy(),
                        landmarks_3d=world[frame_position, hand_index].copy(),
                        mano_params=None,
                        mano_references=None,
                        mode=str(metadata.get("running_mode")),
                        source_index=hand_index,
                    )
                )
        return tuple(records), len(frame_indices), metadata


def validate_mediapipe_run(run_dir: str | Path, contract: ManifestContract) -> ValidatedRun:
    """Validate the frozen MediaPipe raw run against the exact contract."""

    path = Path(run_dir).expanduser().resolve()
    metadata_path = path / "run_metadata.json"
    if not metadata_path.is_file():
        raise InputValidationError(f"MediaPipe run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != contract.sha256:
        raise InputValidationError("MediaPipe run consumed a different manifest")
    if metadata.get("videos_requested") != EXPECTED_SAMPLE_COUNT or metadata.get("videos_successfully_processed") != EXPECTED_SAMPLE_COUNT:
        raise InputValidationError("MediaPipe run does not contain 18 successful videos")
    if metadata.get("model_sha256") != EXPECTED_MEDIAPIPE_MODEL_SHA256:
        raise InputValidationError("MediaPipe run model hash does not match the frozen experiment")
    config = metadata.get("config") or {}
    if config.get("running_mode") != "VIDEO" or config.get("num_hands") != 2 or config.get("delegate") != "CPU":
        raise InputValidationError("MediaPipe run configuration drifted")
    for key in ("min_hand_detection_confidence", "min_hand_presence_confidence", "min_tracking_confidence"):
        if float(config.get(key, -1)) != 0.5:
            raise InputValidationError(f"MediaPipe threshold drifted: {key}")

    frame_counts = _expected_frame_counts_from_inspection(path / "video_inspection.json", contract.sample_ids)
    records_by_sample: dict[str, tuple[HandRecord, ...]] = {}
    for sample_id in contract.sample_ids:
        records, count, npz_metadata = _load_mediapipe_npz(path / "raw_pose" / f"{sample_id}.npz", sample_id)
        if count != frame_counts[sample_id]:
            raise InputValidationError(f"MediaPipe frame count mismatch for {sample_id}")
        records_by_sample[sample_id] = records
        if npz_metadata.get("source_metadata", {}).get("checksum_sha256") != _manifest_by_sample(contract)[sample_id]["checksum_sha256"]:
            raise InputValidationError(f"MediaPipe NPZ source checksum mismatch for {sample_id}")
    if sum(frame_counts.values()) != EXPECTED_TOTAL_FRAMES:
        raise InputValidationError("MediaPipe run total frame count is not 894")
    return ValidatedRun("mediapipe", path, records_by_sample, frame_counts, metadata)


def _load_wilor_npz(path: Path, sample_id: str) -> tuple[tuple[HandRecord, ...], int, dict[str, Any]]:
    if not path.is_file():
        raise InputValidationError(f"WiLoR raw NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_index",
            "timestamp_seconds",
            "hand_present",
            "handedness_label",
            "detection_confidence",
            "landmarks_3d",
            "quality_flags_json",
            "mano_params_json",
            "mano_references_json",
            "extractor_metadata_json",
            "run_metadata_json",
        }
        if not required <= set(data.files):
            raise InputValidationError(f"WiLoR NPZ missing keys: {sorted(required - set(data.files))}")
        frame_indices = np.asarray(data["frame_index"])
        present = np.asarray(data["hand_present"], dtype=bool)
        landmarks = np.asarray(data["landmarks_3d"])
        if frame_indices.ndim != 1 or landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
            raise InputValidationError(f"Unexpected WiLoR NPZ shape in {path}")
        if len(frame_indices) != len(present) or len(frame_indices) != len(landmarks):
            raise InputValidationError(f"Inconsistent WiLoR NPZ arrays in {path}")
        run_metadata = _read_json_scalar(data["run_metadata_json"])
        if run_metadata.get("mode") != "full":
            raise InputValidationError(f"WiLoR input is not exact full mode: {path}")
        frame_count = len(set(int(value) for value in frame_indices))
        if not np.array_equal(np.unique(frame_indices), np.arange(frame_count, dtype=np.unique(frame_indices).dtype)):
            raise InputValidationError(f"WiLoR frame indices are not contiguous in {path}")
        records: list[HandRecord] = []
        frame_hand_indices: dict[int, int] = {}
        for index in range(len(frame_indices)):
            frame_index = int(frame_indices[index])
            hand_index = frame_hand_indices.get(frame_index, 0)
            frame_hand_indices[frame_index] = hand_index + 1
            metadata = _read_json_row(data["extractor_metadata_json"], index)
            flags = tuple(json.loads(str(data["quality_flags_json"][index])))
            mano = _read_json_row(data["mano_params_json"], index, allow_empty=True)
            refs = _read_json_row(data["mano_references_json"], index, allow_empty=True)
            raw_confidence = float(data["detection_confidence"][index])
            record = HandRecord(
                system="wilor",
                frame_index=frame_index,
                hand_present=bool(present[index]),
                handedness_label=normalize_label(data["handedness_label"][index]),
                confidence=raw_confidence if np.isfinite(raw_confidence) else None,
                detection_confidence=raw_confidence if np.isfinite(raw_confidence) else None,
                image_landmarks=None,
                landmarks_3d=landmarks[index].copy(),
                mano_params=mano,
                mano_references=refs,
                mode=metadata.get("mode"),
                quality_flags=flags,
                source_index=hand_index,
            )
            if record.hand_present:
                if "detector_only_no_mano" in flags:
                    raise InputValidationError(f"Detector-only flag on a present WiLoR row: {path}:{index}")
                if not reconstructed_hand(record):
                    raise InputValidationError(f"Invalid full WiLoR reconstruction row: {path}:{index}")
            records.append(record)

        if "vertices" not in data.files or "vertices_keys" not in data.files:
            raise InputValidationError(f"Full WiLoR mesh vertices are missing: {path}")
        vertices = np.asarray(data["vertices"])
        vertex_keys = {str(value) for value in np.asarray(data["vertices_keys"])}
        expected_vertex_keys = {
            f"{int(frame_indices[index])}:{source_index}"
            for index, source_index in enumerate(
                _source_indices_by_frame(frame_indices, present)
            )
            if present[index]
        }
        if vertices.ndim != 3 or vertices.shape[1:] != (778, 3) or not np.isfinite(vertices).all():
            raise InputValidationError(f"WiLoR mesh vertices are malformed: {path}")
        if len(vertices) != len(expected_vertex_keys) or vertex_keys != expected_vertex_keys:
            raise InputValidationError(f"WiLoR mesh keys do not cover reconstructed rows: {path}")
        return tuple(records), frame_count, run_metadata


def _source_indices_by_frame(frame_indices: np.ndarray, present: np.ndarray) -> list[int]:
    counts: dict[int, int] = {}
    result: list[int] = []
    for frame_index in frame_indices:
        key = int(frame_index)
        result.append(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
    return result


def validate_wilor_run(run_dir: str | Path, contract: ManifestContract) -> ValidatedRun:
    """Validate exact WiLoR Phase-B/full artifacts; reject Phase-A inputs."""

    path = Path(run_dir).expanduser().resolve()
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise InputValidationError(f"WiLoR summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "full":
        raise InputValidationError(
            f"Refusing WiLoR run with mode={summary.get('mode')!r}; exact full mode is required"
        )
    if summary.get("n_videos") != EXPECTED_SAMPLE_COUNT or summary.get("n_processed") != EXPECTED_SAMPLE_COUNT or summary.get("n_failed") != 0:
        raise InputValidationError("WiLoR summary does not contain 18 successful videos")
    summary_rows = {row.get("sample_id"): row for row in summary.get("per_video", [])}
    if set(summary_rows) != set(contract.sample_ids):
        raise InputValidationError("WiLoR summary sample IDs do not match the manifest")
    if any(row.get("mode") != "full" or row.get("frame_errors") for row in summary_rows.values()):
        raise InputValidationError("WiLoR summary contains non-full or failed per-video rows")
    frame_counts = {sample_id: int(summary_rows[sample_id]["total_frames_decoded"]) for sample_id in contract.sample_ids}
    if sum(frame_counts.values()) != EXPECTED_TOTAL_FRAMES:
        raise InputValidationError(f"WiLoR summary total is not {EXPECTED_TOTAL_FRAMES} frames")
    summary_manifest = str(summary.get("manifest", ""))
    if not summary_manifest.endswith("datasets/manifests/karsl_milestone1_pilot.csv"):
        raise InputValidationError("WiLoR summary does not identify the shared pilot manifest")

    records_by_sample: dict[str, tuple[HandRecord, ...]] = {}
    manifest_rows = _manifest_by_sample(contract)
    for sample_id in contract.sample_ids:
        records, frame_count, npz_run_metadata = _load_wilor_npz(
            path / "raw" / sample_id / "wilor_raw.npz", sample_id
        )
        if frame_count != frame_counts[sample_id]:
            raise InputValidationError(f"WiLoR frame count mismatch for {sample_id}")
        source_video = npz_run_metadata.get("source_video")
        if source_video and str(source_video) != manifest_rows[sample_id]["local_relative_path"]:
            raise InputValidationError(f"WiLoR source path mismatch for {sample_id}")
        records_by_sample[sample_id] = records
    return ValidatedRun("wilor", path, records_by_sample, frame_counts, summary)


def validate_all_inputs(
    manifest_path: str | Path,
    *,
    video_root: str | Path,
    mediapipe_run: str | Path,
    wilor_run: str | Path,
    verify_video_checksums: bool = True,
) -> tuple[ManifestContract, ValidatedRun, ValidatedRun]:
    """Validate both exact frozen runs before any comparison metric runs."""

    contract = validate_manifest(
        manifest_path,
        video_root=video_root,
        verify_video_checksums=verify_video_checksums,
    )
    mediapipe = validate_mediapipe_run(mediapipe_run, contract)
    wilor = validate_wilor_run(wilor_run, contract)
    if mediapipe.total_frames != EXPECTED_TOTAL_FRAMES or wilor.total_frames != EXPECTED_TOTAL_FRAMES:
        raise InputValidationError("Both validated runs must contain exactly 894 frames")
    return contract, mediapipe, wilor
