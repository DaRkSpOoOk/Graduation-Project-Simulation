"""NPZ + metadata persistence for the kinematics stage.

Everything written is a plain numeric or string NumPy array: the NPZ is saved
and loaded with ``allow_pickle=False`` so no pickled Python object can enter
the artefact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .extractor import POSE_BEARING_STATES, SequenceKinematics
from .layout import (
    CHAIN_ORDER,
    FINGER_ORDER,
    SCHEMA_VERSION,
    SPREAD_PAIRS,
    TRACK_ORDER,
)

KINEMATICS_NPZ_NAME = "hand_kinematics.npz"
KINEMATICS_META_NAME = "hand_kinematics_meta.json"

ARRAY_ORDER: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "tracking_state_code",
    "source_raw_detection_index",
    "valid_kinematics",
    "valid_palm_frame",
    "flexion_deg",
    "adjacent_spread_deg",
    "palm_rotation_matrix",
    "palm_quaternion_wxyz",
    "kinematic_flags_json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    sequence: SequenceKinematics,
    *,
    tracked_dir: Path,
    tracked_sha256: str,
    tracked_metadata: dict[str, Any],
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "kinematics",
        "task": "TASK-005A",
        "sample_id": sequence.sample_id,
        "total_frames": int(sequence.frame_index.shape[0]),
        "track_order": list(TRACK_ORDER),
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "chain_joint_note": (
            "For the four non-thumb fingers proximal/middle/distal correspond "
            "approximately to MCP/PIP/DIP. The thumb chain is rooted at the "
            "wrist so its three angles sit approximately at CMC/MCP/IP; the "
            "generic names are kept because that mapping is not asserted as "
            "anatomically exact."
        ),
        "spread_pairs": [list(pair) for pair in SPREAD_PAIRS],
        "flexion_definition": {
            "type": "unsigned_polyline_turn_angle",
            "units": "degrees",
            "range": [0.0, 180.0],
            "formula": "angle(normalize(j[k+1]-j[k]), normalize(j[k+2]-j[k+1]))",
            "convention": "straight chain = 0 deg; increasing bend = increasing positive",
            "source": "3D joint positions only; MANO local rotations are NOT used",
            "caveat": (
                "The proximal angle is a total inter-bone bend and therefore "
                "couples flexion with abduction. Spread is reported separately "
                "so a consumer can account for it."
            ),
        },
        "spread_definition": {
            "type": "unsigned_palm_plane_angle",
            "units": "degrees",
            "range": [0.0, 180.0],
            "signed": False,
            "formula": (
                "angle between adjacent proximal-phalanx directions after "
                "projection onto the plane perpendicular to the palmar normal"
            ),
            "source": "3D joint positions only",
        },
        "palm_frame": {
            "landmarks": ["wrist", "index_MCP", "middle_MCP", "pinky_MCP"],
            "distal_axis": "normalize(middle_MCP - wrist)",
            "lateral_raw_right": "index_MCP - pinky_MCP",
            "lateral_raw_left": "pinky_MCP - index_MCP",
            "normal": "normalize(distal x lateral_raw)  [palmar/volar direction]",
            "lateral": "normal x distal",
            "matrix_columns": ["lateral", "palmar_normal", "distal"],
            "determinant": "+1 by construction",
            "anatomically_consistent_axes": ["distal", "palmar_normal"],
            "handedness_carrying_axis": "lateral (radial on RIGHT, ulnar on LEFT)",
        },
        "quaternion_order": "wxyz",
        "quaternion_convention": "normalized; sign fixed so w >= 0",
        "left_right_canonicalization": {
            "method": "lateral_raw_direction_reversed_for_left_hand",
            "location": "kinematics/hand_frame.py::build_palm_frame",
            "identity": "R_left = diag(-1,1,1) @ R_right @ diag(-1,1,1)",
            "flexion_and_spread": (
                "no correction applied or needed: both are built from dot "
                "products of joint differences and are therefore invariant "
                "under any orthogonal transform, reflections included"
            ),
            "upstream_note": (
                "WiLoR reconstructs every hand with the canonical MANO right-hand "
                "model and mirrors the joint x axis for a left hand "
                "(pose/wilor/frame_extraction.py). The exported 3D joints are "
                "true left-hand geometry; hand_pose_rotmat is not, which is why "
                "no output channel is derived from MANO local rotations."
            ),
        },
        "invalid_state_policy": {
            "pose_bearing_states": sorted(POSE_BEARING_STATES),
            "rule": (
                "valid_kinematics is True only when every float channel for that "
                "hand-instance is finite. States without a usable pose "
                "(MISSING, LIKELY_OCCLUDED, REJECTED_QUALITY) yield "
                "valid_kinematics=False and all-NaN channels."
            ),
            "ambiguous_states": (
                "AMBIGUOUS carries a real reconstructed pose, so kinematics are "
                "computed and flagged TRACK_STATE_AMBIGUOUS; the tracking state "
                "is preserved in tracking_state_code."
            ),
            "validity_flags": {
                "valid_kinematics": (
                    "strict: True only when every float channel for this "
                    "hand-instance is finite"
                ),
                "valid_palm_frame": (
                    "True when the pose was usable and the palm frame and "
                    "quaternion are well formed, i.e. the orientation channels "
                    "are trustworthy even if one per-channel quantity such as a "
                    "single spread pair is undefined"
                ),
                "per_channel": (
                    "channels that are geometrically sound are always written, "
                    "so mask per channel with np.isfinite(...) rather than "
                    "relying on valid_kinematics alone"
                ),
            },
            "spread_conditioning": {
                "min_projection_angle_deg": 15.0,
                "rationale": (
                    "a finger within 15 deg of the palm normal has no "
                    "meaningful direction in the palm plane; below that the "
                    "projection amplifies input angular error by more than "
                    "3.9x and is dominated by reconstruction noise"
                ),
                "chosen": "a priori conditioning limit, not fitted to the data; "
                          "the observed distribution is a smooth continuum with "
                          "no natural gap",
            },
            "no_interpolation": True,
            "no_forward_fill": True,
            "no_temporal_filtering": True,
            "no_cross_hand_copying": True,
        },
        "source": {
            "tracked_dir": str(tracked_dir),
            "tracked_npz_sha256": tracked_sha256,
            "tracked_schema_version": tracked_metadata.get("schema_version"),
            "tracked_sample_id": tracked_metadata.get("sample_id"),
            "raw_source": tracked_metadata.get("source", {}),
        },
        "implementation_commit": implementation_commit,
    }


def save_kinematics(
    directory: str | Path, sequence: SequenceKinematics, metadata: dict[str, Any]
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_index": sequence.frame_index,
        "timestamp_seconds": sequence.timestamp_seconds,
        "tracking_state_code": sequence.tracking_state_code,
        "source_raw_detection_index": sequence.source_raw_detection_index,
        "valid_kinematics": sequence.valid_kinematics,
        "valid_palm_frame": sequence.valid_palm_frame,
        "flexion_deg": sequence.flexion_deg,
        "adjacent_spread_deg": sequence.adjacent_spread_deg,
        "palm_rotation_matrix": sequence.palm_rotation_matrix,
        "palm_quaternion_wxyz": sequence.palm_quaternion_wxyz,
        "kinematic_flags_json": sequence.kinematic_flags_json,
    }
    npz_path = path / KINEMATICS_NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    (path / KINEMATICS_META_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return npz_path


def load_kinematics(directory: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(directory)
    npz_path = path / KINEMATICS_NPZ_NAME if path.is_dir() else path
    meta_path = npz_path.parent / KINEMATICS_META_NAME
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return arrays, metadata
