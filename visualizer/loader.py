"""Read one TASK-008 production sequence without loading the dataset bulk."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import (
    TRACK_ORDER,
    FrameData,
    HandGeometry,
    PlaybackSequence,
    validate_sensor_layout,
)


class ArtifactValidationError(ValueError):
    """Raised when a stored sequence is incomplete or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON artifact must contain an object: {path}")
    return value


def _read_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing NPZ artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ArtifactValidationError(f"{path} is missing arrays: {missing}")
            return {name: np.array(data[name], copy=True) for name in data.files}
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise ArtifactValidationError(f"could not read NPZ artifact: {path}") from exc


def _freeze(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _assert_shape(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        raise ArtifactValidationError(f"{name} has shape {value.shape}, expected {shape}")


def _assert_aligned(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ArtifactValidationError(f"{name} is not frame-aligned with virtual_glove.npz")


def _parse_vertex_key(value: Any) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ArtifactValidationError(f"invalid vertices_keys entry: {value!r}")
    try:
        frame_index, raw_index = (int(part) for part in parts)
    except ValueError as exc:
        raise ArtifactValidationError(f"invalid vertices_keys entry: {value!r}") from exc
    if frame_index < 0 or raw_index < 0:
        raise ArtifactValidationError(f"negative vertices_keys entry: {value!r}")
    return frame_index, raw_index


def _vertex_map(raw: Mapping[str, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    if "vertices" not in raw or "vertices_keys" not in raw:
        return {}
    vertices = np.asarray(raw["vertices"])
    keys = np.asarray(raw["vertices_keys"])
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ArtifactValidationError(f"vertices must have shape (N,V,3), got {vertices.shape}")
    if keys.ndim != 1 or len(keys) != len(vertices):
        raise ArtifactValidationError("vertices and vertices_keys must have the same row count")
    if vertices.shape[1] <= 0:
        raise ArtifactValidationError("vertices must contain at least one vertex")
    result: dict[tuple[int, int], np.ndarray] = {}
    seen: set[tuple[int, int]] = set()
    for key, value in zip(keys, vertices):
        parsed = _parse_vertex_key(key)
        if parsed in seen:
            raise ArtifactValidationError(f"duplicate mesh key: {parsed}")
        seen.add(parsed)
        if not np.isfinite(value).all():
            # Keep a missing/invalid mesh distinguishable; tracked landmarks
            # can still be rendered for this hand.
            continue
        result[parsed] = _freeze(np.asarray(value, dtype=np.float32))
    return result


def _manifest_record(path: Path, sample_id: str) -> dict[str, str]:
    if not path.is_file():
        raise ArtifactValidationError(f"manifest not found: {path}")
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row.get("sample_id") == sample_id]
    if len(records) != 1:
        raise ArtifactValidationError(f"manifest must contain exactly one row for {sample_id!r}")
    return records[0]


def _state_names(metadata: Mapping[str, Any]) -> dict[int, str]:
    raw = metadata.get("state_codes")
    if not isinstance(raw, Mapping):
        raise ArtifactValidationError("tracking metadata lacks state_codes")
    result: dict[int, str] = {}
    for name, code in raw.items():
        try:
            result[int(code)] = str(name).upper()
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError("tracking state code is not an integer") from exc
    if not result:
        raise ArtifactValidationError("tracking state_codes is empty")
    return result


def load_sequence(
    run_root: str | Path,
    sample_id: str,
    *,
    manifest_path: str | Path | None = None,
) -> PlaybackSequence:
    """Load exactly one stored sequence from the TASK-008 run.

    The loader validates frame/timestamp alignment across raw pose, tracking,
    kinematics, and virtual-glove stages.  It never mutates any source array or
    writes output files.
    """

    if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in {".", ".."}:
        raise ArtifactValidationError(f"invalid sample_id: {sample_id!r}")
    root = Path(run_root)
    paths = {
        "raw": root / "pose" / "raw" / sample_id / "wilor_raw.npz",
        "tracking": root / "tracking" / sample_id,
        "kinematics": root / "kinematics" / sample_id,
        "glove": root / "virtual_glove" / sample_id,
    }
    raw = _read_npz(
        paths["raw"],
        ("frame_index", "timestamp_seconds", "hand_present", "landmarks_3d"),
    )
    tracking = _read_npz(
        paths["tracking"] / "wilor_tracked.npz",
        (
            "frame_index",
            "timestamp_seconds",
            "state_code",
            "raw_detection_index",
            "landmarks_3d",
        ),
    )
    kinematics = _read_npz(
        paths["kinematics"] / "hand_kinematics.npz",
        (
            "frame_index",
            "timestamp_seconds",
            "tracking_state_code",
            "source_raw_detection_index",
        ),
    )
    glove = _read_npz(
        paths["glove"] / "virtual_glove.npz",
        (
            "frame_index",
            "timestamp_seconds",
            "bend_normalized",
            "spread_normalized",
            "bend_valid",
            "spread_valid",
            "imu_quaternion_wxyz",
            "palm_imu_valid",
            "tracking_state_code",
            "source_raw_detection_index",
        ),
    )
    tracking_meta = _read_json(paths["tracking"] / "wilor_tracked_meta.json")
    glove_meta = _read_json(paths["glove"] / "virtual_glove_meta.json")
    layout = validate_sensor_layout(_read_json(paths["glove"] / "sensor_layout.json"))

    frame_index = np.asarray(glove["frame_index"], dtype=np.int64)
    timestamps = np.asarray(glove["timestamp_seconds"], dtype=np.float64)
    _assert_shape("frame_index", frame_index, (None,))
    if len(frame_index) == 0:
        raise ArtifactValidationError("sequence contains no frames")
    if not np.isfinite(timestamps).all() or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ArtifactValidationError("virtual-glove timestamps must be finite and strictly increasing")
    if any(right <= left for left, right in zip(frame_index, frame_index[1:])):
        raise ArtifactValidationError("virtual-glove frame indices must be strictly increasing")

    raw_frame_index = np.asarray(raw["frame_index"], dtype=np.int64)
    raw_timestamps = np.asarray(raw["timestamp_seconds"], dtype=np.float64)
    _assert_shape("raw.frame_index", raw_frame_index, (None,))
    _assert_shape("raw.timestamp_seconds", raw_timestamps, (None,))
    _assert_shape("raw.landmarks_3d", np.asarray(raw["landmarks_3d"]), (None, 21, 3))
    if len(raw_frame_index) != len(raw_timestamps):
        raise ArtifactValidationError("raw frame_index and timestamp_seconds row counts differ")
    raw_frames = np.unique(raw_frame_index)
    if not np.array_equal(raw_frames, frame_index):
        raise ArtifactValidationError("raw pose does not cover the virtual-glove frame indices")
    for expected_frame, expected_timestamp in zip(frame_index, timestamps):
        rows = raw_timestamps[raw_frame_index == expected_frame]
        if not len(rows) or not np.isfinite(rows).all() or not np.allclose(
            rows, expected_timestamp, rtol=0.0, atol=1e-6
        ):
            raise ArtifactValidationError(f"raw timestamp mismatch for frame {int(expected_frame)}")

    for name, arrays in (("tracking", tracking), ("kinematics", kinematics)):
        _assert_aligned(f"{name}.frame_index", np.asarray(arrays["frame_index"]), frame_index)
        _assert_aligned(f"{name}.timestamp_seconds", np.asarray(arrays["timestamp_seconds"]), timestamps)
    _assert_aligned("kinematics.tracking_state_code", np.asarray(kinematics["tracking_state_code"]), np.asarray(glove["tracking_state_code"]))
    _assert_aligned("kinematics.source_raw_detection_index", np.asarray(kinematics["source_raw_detection_index"]), np.asarray(glove["source_raw_detection_index"]))

    n_frames = len(frame_index)
    _assert_shape("tracking.state_code", np.asarray(tracking["state_code"]), (n_frames, 2))
    _assert_shape("tracking.raw_detection_index", np.asarray(tracking["raw_detection_index"]), (n_frames, 2))
    _assert_shape("tracking.landmarks_3d", np.asarray(tracking["landmarks_3d"]), (n_frames, 2, 21, 3))
    for name, expected in (
        ("bend_normalized", (n_frames, 2, 5, 3)),
        ("spread_normalized", (n_frames, 2, 4)),
        ("bend_valid", (n_frames, 2, 5, 3)),
        ("spread_valid", (n_frames, 2, 4)),
        ("imu_quaternion_wxyz", (n_frames, 2, 4)),
        ("palm_imu_valid", (n_frames, 2)),
        ("tracking_state_code", (n_frames, 2)),
        ("source_raw_detection_index", (n_frames, 2)),
    ):
        _assert_shape(f"glove.{name}", np.asarray(glove[name]), expected)

    tracking_order = tuple(str(value).upper() for value in tracking_meta.get("track_order", ()))
    glove_order = tuple(str(value).upper() for value in glove_meta.get("track_order", ()))
    if tracking_order != TRACK_ORDER or glove_order != TRACK_ORDER:
        raise ArtifactValidationError("tracking and virtual-glove track order must be LEFT, RIGHT")
    state_names = _state_names(tracking_meta)
    mesh_field_presence = ("vertices" in raw, "vertices_keys" in raw)
    if mesh_field_presence[0] != mesh_field_presence[1]:
        raise ArtifactValidationError("raw mesh vertices and vertices_keys must be provided together")
    mesh_by_key = _vertex_map(raw)
    raw_has_mesh_arrays = "vertices" in raw and "vertices_keys" in raw
    if raw_has_mesh_arrays:
        raw_vertices = np.asarray(raw["vertices"])
        raw_keys = [_parse_vertex_key(key) for key in np.asarray(raw["vertices_keys"])]
        if len(raw_keys) != len(raw_frame_index) or len(raw_vertices) != len(raw_frame_index):
            raise ArtifactValidationError("raw mesh rows must match raw detection rows")
        raw_frame_set = set(int(value) for value in raw_frame_index)
        if any(frame not in raw_frame_set for frame, _ in raw_keys):
            raise ArtifactValidationError("raw mesh key references an unknown frame")
        for frame in raw_frame_set:
            indices = sorted(raw_index for key_frame, raw_index in raw_keys if key_frame == frame)
            expected = list(range(int(np.count_nonzero(raw_frame_index == frame))))
            if indices != expected:
                raise ArtifactValidationError(f"raw mesh detection keys are not contiguous for frame {frame}")

    manifest = None
    if manifest_path is None:
        candidate = Path(__file__).resolve().parents[1] / "datasets" / "manifests" / "karsl_core28.csv"
        if candidate.is_file():
            manifest_path = candidate
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = _manifest_record(Path(manifest_path), sample_id)

    frames: list[FrameData] = []
    mesh_rows = 0
    for position, (frame_value, timestamp) in enumerate(zip(frame_index, timestamps)):
        state_row = np.asarray(tracking["state_code"])[position]
        raw_index_row = np.asarray(tracking["raw_detection_index"])[position]
        landmark_row = np.asarray(tracking["landmarks_3d"])[position]
        hands: list[HandGeometry] = []
        for track_index, track in enumerate(TRACK_ORDER):
            state = state_names.get(int(state_row[track_index]), f"UNKNOWN({int(state_row[track_index])})")
            raw_index = int(raw_index_row[track_index])
            landmarks = landmark_row[track_index]
            has_landmarks = raw_index >= 0 and np.isfinite(landmarks).all()
            mesh = mesh_by_key.get((int(frame_value), raw_index)) if raw_index >= 0 else None
            # A tracking state without a usable source detection is missing;
            # never render an imputed hand for it.
            if state in {"MISSING", "LIKELY_OCCLUDED", "REJECTED_QUALITY"} or raw_index < 0:
                has_landmarks = False
                mesh = None
            landmarks_value = _freeze(np.asarray(landmarks, dtype=np.float32)) if has_landmarks else None
            if mesh is not None:
                mesh_rows += 1
            hands.append(HandGeometry(track, state, raw_index, landmarks_value, mesh))
        frames.append(
            FrameData(
                position=position,
                frame_index=int(frame_value),
                timestamp_seconds=float(timestamp),
                hands=(hands[0], hands[1]),
                bend_normalized=_freeze(np.asarray(glove["bend_normalized"][position], dtype=np.float32)),
                spread_normalized=_freeze(np.asarray(glove["spread_normalized"][position], dtype=np.float32)),
                bend_valid=_freeze(np.asarray(glove["bend_valid"][position], dtype=bool)),
                spread_valid=_freeze(np.asarray(glove["spread_valid"][position], dtype=bool)),
                palm_quaternion_wxyz=_freeze(np.asarray(glove["imu_quaternion_wxyz"][position], dtype=np.float32)),
                palm_imu_valid=_freeze(np.asarray(glove["palm_imu_valid"][position], dtype=bool)),
            )
        )

    geometry_source = "stored_mano_vertices+tracked_landmarks_3d" if mesh_rows else "tracked_landmarks_3d"
    metadata: dict[str, Any] = {
        "run_root": str(root),
        "sample_id": sample_id,
        "tracking": tracking_meta,
        "virtual_glove": glove_meta,
        "mesh": {
            "stored_vertices_arrays": raw_has_mesh_arrays,
            "embedded_mano_vertices_available": bool(raw_has_mesh_arrays and mesh_by_key),
            "tracked_landmarks_3d_available": True,
            # TASK-008 stores MANO vertices but not a trusted triangle index
            # buffer.  Keep this explicit so callers do not mistake a vertex
            # cloud for a renderable surface mesh.
            "surface_triangle_topology_available": False,
            "mesh_rows_mapped_to_track": mesh_rows,
            "vertex_count": int(next(iter(mesh_by_key.values())).shape[0]) if mesh_by_key else None,
            "vertices_key_format": "frame_index:raw_detection_index",
        },
    }
    if manifest is not None:
        metadata["manifest"] = manifest
    label_index = None
    if manifest and manifest.get("label_index") not in (None, ""):
        label_index = int(manifest["label_index"])
    return PlaybackSequence(
        sample_id=sample_id,
        label_ar=manifest.get("label_ar") if manifest else None,
        label_index=label_index,
        signer_id=manifest.get("signer_id") if manifest else None,
        frames=tuple(frames),
        sensor_layout=layout,
        geometry_source=geometry_source,
        metadata=metadata,
    )
