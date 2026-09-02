"""Extractor-independent QA for TASK-006 virtual-glove artifacts."""

from .contract import (
    OPTIONAL_ARRAYS,
    REQUIRED_ARRAYS,
    VIRTUAL_GLOVE_META_NAME,
    VIRTUAL_GLOVE_NPZ_NAME,
)
from .validator import validate_runs, write_csv, write_json

__all__ = [
    "OPTIONAL_ARRAYS",
    "REQUIRED_ARRAYS",
    "VIRTUAL_GLOVE_META_NAME",
    "VIRTUAL_GLOVE_NPZ_NAME",
    "validate_runs",
    "write_csv",
    "write_json",
]
