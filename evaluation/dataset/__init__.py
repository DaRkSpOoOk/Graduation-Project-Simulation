"""TASK-008A dataset acquisition and reproducibility helpers."""

from .core28 import (
    CORE28_SIGN_IDS,
    EXTENDED_LETTER_SIGN_IDS,
    OFFICIAL_LABELS_URL,
    OFFICIAL_RGB_PAGE_URL,
    OFFICIAL_SITE_URL,
    LabelRecord,
    core28_records,
    load_label_records,
    validate_core28_records,
)
from .manifest import (
    CORE28_MANIFEST_FIELDS,
    VideoRecord,
    build_manifest_from_video_root,
    inspect_manifest_videos,
    load_manifest,
    manifest_sha256,
    write_manifest,
)
from .splits import (
    FOLD_SIGNERS,
    build_loso_splits,
    validate_split_rows,
    write_split_manifests,
)
from .orchestrator import (
    STAGES,
    RunPaths,
    assign_shards,
    sample_ids_for_shard,
    stable_shard_index,
    status_snapshot,
)
from .qa import DatasetQAError, validate_run

__all__ = [
    "CORE28_MANIFEST_FIELDS",
    "CORE28_SIGN_IDS",
    "EXTENDED_LETTER_SIGN_IDS",
    "FOLD_SIGNERS",
    "LabelRecord",
    "OFFICIAL_LABELS_URL",
    "OFFICIAL_RGB_PAGE_URL",
    "OFFICIAL_SITE_URL",
    "VideoRecord",
    "build_manifest_from_video_root",
    "build_loso_splits",
    "core28_records",
    "inspect_manifest_videos",
    "load_label_records",
    "load_manifest",
    "manifest_sha256",
    "validate_core28_records",
    "validate_split_rows",
    "write_manifest",
    "write_split_manifests",
    "DatasetQAError",
    "RunPaths",
    "STAGES",
    "assign_shards",
    "sample_ids_for_shard",
    "stable_shard_index",
    "status_snapshot",
    "validate_run",
]
