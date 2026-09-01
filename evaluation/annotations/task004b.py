"""Schema and validation for the independent TASK-004B benchmark.

The annotation file is deliberately extractor-neutral.  It contains only
human judgments from the original KArSL RGB frames and never stores model
labels, tracks, or pose predictions.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VISIBILITY_STATES = frozenset(
    {
        "VISIBLE",
        "PARTIALLY_OCCLUDED",
        "FULLY_OCCLUDED",
        "OUT_OF_FRAME",
        "AMBIGUOUS",
    }
)
ANNOTATOR_CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM", "LOW"})
SCENE_FLAGS = frozenset(
    {
        "HAND_CROSSING",
        "MOTION_BLUR",
        "FRAME_EDGE",
        "SELF_OCCLUSION",
        "HAND_HAND_OCCLUSION",
        "IDENTITY_AMBIGUOUS",
    }
)
ANNOTATION_COLUMNS = (
    "sample_id",
    "frame_index",
    "left_visibility",
    "left_x",
    "left_y",
    "right_visibility",
    "right_x",
    "right_y",
    "scene_flags",
    "annotator_confidence",
    "notes",
)


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """A source clip that is part of the locked benchmark contract."""

    sample_id: str
    frame_count: int
    role: str
    rationale: str


# Frame counts were obtained by decoding the original source files before
# annotation.  The order is the order used in the report and CSV.
CLIP_SPECS: tuple[ClipSpec, ...] = (
    ClipSpec(
        "karsl_test_s01_sign0174_repfirst",
        34,
        "challenge",
        "Required occlusion/crossing clip; frames 11-33 contain sustained hand overlap.",
    ),
    ClipSpec(
        "karsl_test_s02_sign0172_repfirst",
        36,
        "challenge",
        "Required close-contact and motion-blur clip while both hands converge.",
    ),
    ClipSpec(
        "karsl_test_s02_sign0175_repfirst",
        48,
        "challenge",
        "Required asymmetric-visibility clip with one hand low and the other moving toward the image edge.",
    ),
    ClipSpec(
        "karsl_test_s02_sign0176_repfirst",
        57,
        "challenge",
        "Required extra-detection stress clip; frame 39 is checked for two physical hands and blur.",
    ),
    ClipSpec(
        "karsl_test_s03_sign0173_repfirst",
        38,
        "challenge",
        "Required extended-hand/edge-proximity clip with one hand resting low in frame.",
    ),
    ClipSpec(
        "karsl_test_s03_sign0174_repfirst",
        38,
        "challenge",
        "Required crossing/hand-hand-occlusion clip with a prolonged central overlap.",
    ),
    ClipSpec(
        "karsl_test_s02_sign0171_repfirst",
        67,
        "control",
        "Preselected clean bilateral control with both physical hands visible across the clip.",
    ),
    ClipSpec(
        "karsl_test_s01_sign0171_repfirst",
        81,
        "control",
        "Preselected bilateral control; later hand contact tests continuity without leaving the frame.",
    ),
)
_CLIP_BY_ID = {spec.sample_id: spec for spec in CLIP_SPECS}


class AnnotationError(ValueError):
    """Raised when the benchmark file violates its locked contract."""


@dataclass(frozen=True, slots=True)
class AnnotationRow:
    sample_id: str
    frame_index: int
    left_visibility: str
    left_x: float | None
    left_y: float | None
    right_visibility: str
    right_x: float | None
    right_y: float | None
    scene_flags: tuple[str, ...]
    annotator_confidence: str
    notes: str

    @property
    def is_crossing(self) -> bool:
        return "HAND_CROSSING" in self.scene_flags

    @property
    def is_motion_blurred(self) -> bool:
        return "MOTION_BLUR" in self.scene_flags

    @property
    def has_identity_ambiguity(self) -> bool:
        return (
            "IDENTITY_AMBIGUOUS" in self.scene_flags
            or self.left_visibility == "AMBIGUOUS"
            or self.right_visibility == "AMBIGUOUS"
        )


def _required_columns(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise AnnotationError("annotation file has no data rows")
    missing = set(ANNOTATION_COLUMNS).difference(rows[0])
    if missing:
        raise AnnotationError(f"annotation file is missing columns: {sorted(missing)}")


def _optional_coordinate(value: str, *, sample_id: str, frame_index: int, name: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        coordinate = float(text)
    except ValueError as error:
        raise AnnotationError(f"{sample_id} frame {frame_index}: {name} is not numeric") from error
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise AnnotationError(f"{sample_id} frame {frame_index}: {name} must be finite and in [0, 1]")
    return coordinate


def _parse_row(raw: Mapping[str, str], line_number: int) -> AnnotationRow:
    sample_id = raw["sample_id"].strip()
    if not sample_id:
        raise AnnotationError(f"line {line_number}: sample_id is empty")
    try:
        frame_index = int(raw["frame_index"])
    except ValueError as error:
        raise AnnotationError(f"line {line_number}: frame_index is not an integer") from error
    if frame_index < 0:
        raise AnnotationError(f"line {line_number}: frame_index must be non-negative")

    left_visibility = raw["left_visibility"].strip()
    right_visibility = raw["right_visibility"].strip()
    for side, visibility in (("left", left_visibility), ("right", right_visibility)):
        if visibility not in VISIBILITY_STATES:
            raise AnnotationError(
                f"line {line_number}: {side}_visibility {visibility!r} is not a valid state"
            )

    left_x = _optional_coordinate(
        raw["left_x"], sample_id=sample_id, frame_index=frame_index, name="left_x"
    )
    left_y = _optional_coordinate(
        raw["left_y"], sample_id=sample_id, frame_index=frame_index, name="left_y"
    )
    right_x = _optional_coordinate(
        raw["right_x"], sample_id=sample_id, frame_index=frame_index, name="right_x"
    )
    right_y = _optional_coordinate(
        raw["right_y"], sample_id=sample_id, frame_index=frame_index, name="right_y"
    )
    if (left_x is None) != (left_y is None):
        raise AnnotationError(f"line {line_number}: left_x and left_y must be both set or both blank")
    if (right_x is None) != (right_y is None):
        raise AnnotationError(f"line {line_number}: right_x and right_y must be both set or both blank")

    scene_flags = tuple(flag for flag in raw["scene_flags"].split(";") if flag)
    unknown_flags = set(scene_flags).difference(SCENE_FLAGS)
    if unknown_flags:
        raise AnnotationError(f"line {line_number}: unknown scene flags {sorted(unknown_flags)}")
    confidence = raw["annotator_confidence"].strip()
    if confidence not in ANNOTATOR_CONFIDENCE_LEVELS:
        raise AnnotationError(f"line {line_number}: invalid annotator confidence {confidence!r}")
    notes = raw["notes"].strip()
    if not notes:
        raise AnnotationError(f"line {line_number}: notes must explain the frame judgment")

    return AnnotationRow(
        sample_id=sample_id,
        frame_index=frame_index,
        left_visibility=left_visibility,
        left_x=left_x,
        left_y=left_y,
        right_visibility=right_visibility,
        right_x=right_x,
        right_y=right_y,
        scene_flags=scene_flags,
        annotator_confidence=confidence,
        notes=notes,
    )


def validate_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    clip_specs: Sequence[ClipSpec] = CLIP_SPECS,
    require_complete: bool = True,
) -> tuple[AnnotationRow, ...]:
    """Parse and validate rows, optionally requiring every selected frame."""

    raw_rows = list(rows)
    _required_columns(raw_rows)
    clips = {spec.sample_id: spec for spec in clip_specs}
    parsed: list[AnnotationRow] = []
    seen: set[tuple[str, int]] = set()
    for line_number, raw in enumerate(raw_rows, start=2):
        row = _parse_row(raw, line_number)
        if row.sample_id not in clips:
            raise AnnotationError(f"line {line_number}: sample_id is not in the selected benchmark")
        key = (row.sample_id, row.frame_index)
        if key in seen:
            raise AnnotationError(f"duplicate annotation for {row.sample_id} frame {row.frame_index}")
        seen.add(key)
        spec = clips[row.sample_id]
        if row.frame_index >= spec.frame_count:
            raise AnnotationError(
                f"{row.sample_id} frame {row.frame_index} is outside 0..{spec.frame_count - 1}"
            )
        parsed.append(row)

    if require_complete:
        observed_samples = {row.sample_id for row in parsed}
        expected_samples = set(clips)
        if observed_samples != expected_samples:
            raise AnnotationError(
                f"selected videos mismatch: missing={sorted(expected_samples - observed_samples)}, "
                f"unexpected={sorted(observed_samples - expected_samples)}"
            )
        for spec in clip_specs:
            observed_frames = {row.frame_index for row in parsed if row.sample_id == spec.sample_id}
            expected_frames = set(range(spec.frame_count))
            if observed_frames != expected_frames:
                raise AnnotationError(
                    f"{spec.sample_id}: frame coverage mismatch; "
                    f"missing={sorted(expected_frames - observed_frames)}, "
                    f"unexpected={sorted(observed_frames - expected_frames)}"
                )
    return tuple(parsed)


def read_annotations(path: str | Path, *, require_complete: bool = True) -> tuple[AnnotationRow, ...]:
    """Read the committed CSV and validate it against the benchmark schema."""

    annotation_path = Path(path)
    with annotation_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return validate_rows(rows, require_complete=require_complete)


def validate_against_manifest(
    annotation_path: str | Path,
    manifest_path: str | Path,
) -> tuple[AnnotationRow, ...]:
    """Validate annotations and ensure each selected ID exists in the shared manifest."""

    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_id = {row.get("sample_id", "").strip(): row for row in manifest_rows}
    for spec in CLIP_SPECS:
        if spec.sample_id not in manifest_by_id:
            raise AnnotationError(f"selected sample is absent from the shared manifest: {spec.sample_id}")
        manifest_row = manifest_by_id[spec.sample_id]
        if manifest_row.get("source_dataset") != "KArSL-502":
            raise AnnotationError(f"{spec.sample_id}: source dataset is not KArSL-502")
        if manifest_row.get("modality") != "RGB":
            raise AnnotationError(f"{spec.sample_id}: source modality is not RGB")
        if not manifest_row.get("local_relative_path"):
            raise AnnotationError(f"{spec.sample_id}: manifest local path is empty")
    return read_annotations(annotation_path)


def annotation_statistics(rows: Iterable[AnnotationRow]) -> dict[str, object]:
    """Return transparent counts used by the TASK-004B report."""

    materialized = tuple(rows)
    sample_ids = {row.sample_id for row in materialized}
    flags = {
        name: len({(row.sample_id, row.frame_index) for row in materialized if name in row.scene_flags})
        for name in sorted(SCENE_FLAGS)
    }
    state_counts = {
        "left": dict(Counter(row.left_visibility for row in materialized)),
        "right": dict(Counter(row.right_visibility for row in materialized)),
    }
    return {
        "videos": len(sample_ids),
        "frames": len(materialized),
        "left_state_counts": state_counts["left"],
        "right_state_counts": state_counts["right"],
        "visible_left_frames": sum(row.left_visibility == "VISIBLE" for row in materialized),
        "visible_right_frames": sum(row.right_visibility == "VISIBLE" for row in materialized),
        "partially_occluded_hand_labels": sum(
            row.left_visibility == "PARTIALLY_OCCLUDED" for row in materialized
        )
        + sum(row.right_visibility == "PARTIALLY_OCCLUDED" for row in materialized),
        "fully_occluded_hand_labels": sum(
            row.left_visibility == "FULLY_OCCLUDED" for row in materialized
        )
        + sum(row.right_visibility == "FULLY_OCCLUDED" for row in materialized),
        "ambiguous_identity_frames": len(
            {(row.sample_id, row.frame_index) for row in materialized if row.has_identity_ambiguity}
        ),
        "flags_frame_counts": flags,
        "confidence_counts": dict(Counter(row.annotator_confidence for row in materialized)),
    }


def selected_clip(sample_id: str) -> ClipSpec:
    """Return the locked spec for a sample ID, or raise a clear error."""

    try:
        return _CLIP_BY_ID[sample_id]
    except KeyError as error:
        raise AnnotationError(f"unknown TASK-004B sample_id: {sample_id}") from error
