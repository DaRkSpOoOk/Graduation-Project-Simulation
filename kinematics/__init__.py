"""TASK-005A: canonical hand kinematics derived from TASK-004 tracked poses.

Per-frame geometric kinematics only: finger flexion, adjacent-finger spread,
and a canonical palm frame. No virtual sensors, no normalization, no temporal
filtering, no recognition features.
"""

from .extractor import (
    HandKinematics,
    SequenceKinematics,
    compute_flexion,
    compute_hand_kinematics,
    compute_spread,
    extract_sequence,
)
from .hand_frame import PalmFrame, build_palm_frame
from .io import (
    ARRAY_ORDER,
    KINEMATICS_META_NAME,
    KINEMATICS_NPZ_NAME,
    build_metadata,
    load_kinematics,
    save_kinematics,
    sha256_file,
)
from .layout import (
    CHAIN_ORDER,
    FINGER_CHAINS,
    FINGER_ORDER,
    SCHEMA_VERSION,
    SPREAD_PAIRS,
    TRACK_ORDER,
)

__all__ = [
    "ARRAY_ORDER",
    "CHAIN_ORDER",
    "FINGER_CHAINS",
    "FINGER_ORDER",
    "HandKinematics",
    "KINEMATICS_META_NAME",
    "KINEMATICS_NPZ_NAME",
    "PalmFrame",
    "SCHEMA_VERSION",
    "SPREAD_PAIRS",
    "SequenceKinematics",
    "TRACK_ORDER",
    "build_metadata",
    "build_palm_frame",
    "compute_flexion",
    "compute_hand_kinematics",
    "compute_spread",
    "extract_sequence",
    "load_kinematics",
    "save_kinematics",
    "sha256_file",
]
