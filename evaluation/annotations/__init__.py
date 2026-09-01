"""Independent, human-authored tracking benchmark annotations."""

from .task004b import (
    ANNOTATION_COLUMNS,
    ANNOTATOR_CONFIDENCE_LEVELS,
    CLIP_SPECS,
    SCENE_FLAGS,
    VISIBILITY_STATES,
    AnnotationError,
    AnnotationRow,
    annotation_statistics,
    read_annotations,
    validate_against_manifest,
    validate_rows,
)

__all__ = [
    "ANNOTATION_COLUMNS",
    "ANNOTATOR_CONFIDENCE_LEVELS",
    "CLIP_SPECS",
    "SCENE_FLAGS",
    "VISIBILITY_STATES",
    "AnnotationError",
    "AnnotationRow",
    "annotation_statistics",
    "read_annotations",
    "validate_against_manifest",
    "validate_rows",
]
