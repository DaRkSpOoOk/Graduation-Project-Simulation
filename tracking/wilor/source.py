"""Read-only access to immutable raw WiLoR full-mode output.

The loader opens the raw NPZ, copies what it needs into plain arrays, and
closes it. Nothing here writes to, or can write to, the raw artifact.

The full-mode validity rule intentionally mirrors
``evaluation.comparison.common_contract.reconstructed_hand`` for WiLoR
records. It is restated here rather than imported so that ``tracking`` does
not depend on ``evaluation`` at runtime (the pipeline order is
extraction -> tracking -> evaluation). ``tests/test_tracking_source.py``
asserts the two predicates agree, so they cannot silently diverge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import RawDetection


class RawInputError(RuntimeError):
    """Raised when a raw input is absent, not full-mode, or malformed."""


_REQUIRED_KEYS = frozenset(
    {
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
)


def normalize_label(value: Any) -> str | None:
    label = str(value).strip().casefold()
    return label if label in {"left", "right"} else None


def _finite(array: Any, shape: tuple[int, ...] | None = None) -> bool:
    if array is None:
        return False
    try:
        value = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if shape is not None and value.shape != shape:
        return False
    return value.size > 0 and bool(np.isfinite(value).all())


def is_complete_wilor_pose(
    mode: str | None,
    landmarks_3d: Any,
    mano_params: dict[str, Any] | None,
    quality_flags: tuple[str, ...],
) -> bool:
    """Full-mode completeness predicate (mirrors the comparison contract)."""

    if mode != "full" or "detector_only_no_mano" in quality_flags:
        return False
    if not _finite(landmarks_3d, (21, 3)):
        return False
    mano = mano_params or {}
    return all(
        _finite(mano.get(name))
        for name in ("hand_pose_rotmat", "global_orient_rotmat", "betas")
    )


def _row_json(array: Any, index: int, *, allow_empty: bool = False) -> Any:
    raw = str(array[index])
    if not raw:
        if allow_empty:
            return {}
        raise RawInputError(f"Empty JSON payload at row {index}")
    return json.loads(raw)


class RawSequence:
    """All raw detections of one video, grouped by frame, plus timestamps."""

    __slots__ = ("sample_id", "frame_indices", "timestamps", "detections_by_frame", "run_metadata")

    def __init__(
        self,
        sample_id: str,
        frame_indices: list[int],
        timestamps: dict[int, float],
        detections_by_frame: dict[int, list[RawDetection]],
        run_metadata: dict[str, Any],
    ) -> None:
        self.sample_id = sample_id
        self.frame_indices = frame_indices
        self.timestamps = timestamps
        self.detections_by_frame = detections_by_frame
        self.run_metadata = run_metadata

    @property
    def total_frames(self) -> int:
        return len(self.frame_indices)

    def detections(self, frame_index: int) -> list[RawDetection]:
        return self.detections_by_frame.get(frame_index, [])


def load_raw_sequence(npz_path: str | Path, sample_id: str | None = None) -> RawSequence:
    """Load one immutable raw WiLoR full-mode NPZ.

    Raises RawInputError for a detector-only (Phase-A) input, a non-``full``
    run mode, a missing key, or an incomplete reconstruction on a row that
    claims ``hand_present``.
    """

    path = Path(npz_path)
    if not path.is_file():
        raise RawInputError(f"Raw WiLoR NPZ not found: {path}")
    sample = sample_id or path.parent.name

    with np.load(path, allow_pickle=False) as data:
        missing = _REQUIRED_KEYS - set(data.files)
        if missing:
            raise RawInputError(f"Raw NPZ {path} is missing keys: {sorted(missing)}")

        run_metadata = json.loads(str(data["run_metadata_json"]))
        if run_metadata.get("mode") != "full":
            raise RawInputError(
                f"Raw NPZ {path} is not exact full mode (mode={run_metadata.get('mode')!r}); "
                "tracking refuses detector-only input"
            )

        frame_index_array = np.asarray(data["frame_index"]).astype(int)
        timestamp_array = np.asarray(data["timestamp_seconds"]).astype(float)
        present = np.asarray(data["hand_present"], dtype=bool)
        landmarks = np.asarray(data["landmarks_3d"], dtype=np.float64)
        confidence = np.asarray(data["detection_confidence"], dtype=np.float64)

        if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
            raise RawInputError(f"Unexpected landmarks_3d shape in {path}: {landmarks.shape}")
        lengths = {len(frame_index_array), len(timestamp_array), len(present), len(landmarks), len(confidence)}
        if len(lengths) != 1:
            raise RawInputError(f"Inconsistent row counts in {path}")

        timestamps: dict[int, float] = {}
        detections_by_frame: dict[int, list[RawDetection]] = {}
        per_frame_counter: dict[int, int] = {}

        for row in range(len(frame_index_array)):
            frame_index = int(frame_index_array[row])
            timestamps.setdefault(frame_index, float(timestamp_array[row]))
            detection_index = per_frame_counter.get(frame_index, 0)
            per_frame_counter[frame_index] = detection_index + 1
            detections_by_frame.setdefault(frame_index, [])

            if not bool(present[row]):
                continue

            flags = tuple(json.loads(str(data["quality_flags_json"][row])))
            metadata = _row_json(data["extractor_metadata_json"], row)
            mano = _row_json(data["mano_params_json"], row, allow_empty=True)
            references = _row_json(data["mano_references_json"], row, allow_empty=True)

            if not is_complete_wilor_pose(metadata.get("mode"), landmarks[row], mano, flags):
                raise RawInputError(
                    f"Row {row} of {path} claims hand_present but is not a complete full-mode pose"
                )

            image_size = references.get("img_size_wh")
            box_center = references.get("box_center_xy")
            box_size = references.get("box_size")
            if not image_size or not box_center or not box_size:
                raise RawInputError(f"Row {row} of {path} lacks bbox/image references")

            raw_confidence = float(confidence[row])
            detections_by_frame[frame_index].append(
                RawDetection(
                    frame_index=frame_index,
                    raw_detection_index=detection_index,
                    detector_label=normalize_label(data["handedness_label"][row]),
                    detector_confidence=raw_confidence if np.isfinite(raw_confidence) else None,
                    landmarks_3d=landmarks[row].copy(),
                    hand_pose_rotmat=np.asarray(mano["hand_pose_rotmat"], dtype=np.float64).reshape(15, 3, 3),
                    global_orient_rotmat=np.asarray(mano["global_orient_rotmat"], dtype=np.float64).reshape(3, 3),
                    betas=np.asarray(mano["betas"], dtype=np.float64).reshape(-1),
                    camera_translation=np.asarray(
                        references["camera_translation_xyz"], dtype=np.float64
                    ).reshape(3),
                    box_center_xy=(float(box_center[0]), float(box_center[1])),
                    box_size=float(box_size),
                    image_size_wh=(float(image_size[0]), float(image_size[1])),
                    focal_length=float(references.get("focal_length", 0.0)),
                    raw_quality_flags=flags,
                )
            )

    ordered_frames = sorted(detections_by_frame)
    return RawSequence(
        sample_id=sample,
        frame_indices=ordered_frames,
        timestamps=timestamps,
        detections_by_frame=detections_by_frame,
        run_metadata=run_metadata,
    )
