"""TASK-006 virtual-glove file and sensor-layout contract.

This module only describes and validates serialized artifacts.  It does not
derive angles, generate Hall values, or simulate an IMU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

VIRTUAL_GLOVE_NPZ_NAME = "virtual_glove.npz"
VIRTUAL_GLOVE_META_NAME = "virtual_glove_meta.json"
SENSOR_LAYOUT_NAME = "sensor_layout.json"
KINEMATICS_NPZ_NAME = "hand_kinematics.npz"
KINEMATICS_META_NAME = "hand_kinematics_meta.json"

TRACK_NAMES = ("LEFT", "RIGHT")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
JOINT_NAMES = ("MCP", "PIP", "DIP")
SPREAD_NAMES = (
    "thumb_index",
    "index_middle",
    "middle_ring",
    "ring_pinky",
)

BEND_SENSOR_COUNT = 15
SPREAD_SENSOR_COUNT = 4
HALL_SENSOR_COUNT = 19
PALM_IMU_COUNT = 1
SENSORS_PER_HAND = 20
TOTAL_SENSOR_COUNT = 2 * SENSORS_PER_HAND

# ``F`` is replaced by the frame count discovered from frame_index.
REQUIRED_ARRAYS: dict[str, tuple[int, ... | str]] = {
    "frame_index": ("F",),
    "timestamp_seconds": ("F",),
    "tracking_state_code": ("F", 2),
    "source_raw_detection_index": ("F", 2),
    "bend_angle_deg": ("F", 2, 5, 3),
    "bend_normalized": ("F", 2, 5, 3),
    "bend_valid": ("F", 2, 5, 3),
    "spread_angle_deg": ("F", 2, 4),
    "spread_normalized": ("F", 2, 4),
    "spread_valid": ("F", 2, 4),
    "imu_rotation_matrix": ("F", 2, 3, 3),
    "imu_quaternion_wxyz": ("F", 2, 4),
    "palm_imu_valid": ("F", 2),
}

OPTIONAL_ARRAYS: dict[str, tuple[int, ... | str]] = {
    "bend_adc_12bit": ("F", 2, 5, 3),
    "spread_adc_12bit": ("F", 2, 4),
    "imu_angular_velocity_rad_s": ("F", 2, 3),
    "imu_angular_velocity_valid": ("F", 2),
}

ALIGNMENT_ARRAYS = (
    "frame_index",
    "timestamp_seconds",
    "tracking_state_code",
    "source_raw_detection_index",
)

# Corresponding TASK-005 fields.  These are names in the input contract, not
# an import of production kinematics code.
SOURCE_ARRAYS = {
    "bend_angle_deg": "flexion_deg",
    "spread_angle_deg": "adjacent_spread_deg",
    "palm_imu_valid": "valid_palm_frame",
}

NORMALIZATION_ATOL = 1e-6
NORMALIZATION_RTOL = 1e-6
ROTATION_ORTHOGONALITY_TOLERANCE = 1e-5
ROTATION_DETERMINANT_TOLERANCE = 1e-5
QUATERNION_NORM_TOLERANCE = 1e-5
MATRIX_QUATERNION_TOLERANCE = 1e-5


class ContractError(RuntimeError):
    """Raised when a run cannot be read at all."""


@dataclass(frozen=True, slots=True)
class VirtualGloveSample:
    sample_id: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True, slots=True)
class KinematicsSample:
    sample_id: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True, slots=True)
class SensorDescriptor:
    """Canonical representation of one physical sensor assignment."""

    sensor_id: str
    sensor_type: str
    hand: str
    family: str
    finger: str | None
    joint: str | None
    spread_pair: str | None
    display_marker: str
    description: str
    logical_location: Any

    @property
    def key(self) -> tuple[str, str, str, str]:
        logical_name = self.finger or self.spread_pair or ""
        if self.family == "imu":
            logical_name = "palm"
        return (
            self.hand,
            self.family,
            logical_name,
            self.joint or "",
        )


@dataclass(frozen=True, slots=True)
class LayoutResult:
    passed: bool
    failures: tuple[str, ...]
    sensors: tuple[SensorDescriptor, ...]
    representation: str = "runtime_identities"
    physical_template_count: int | None = None

    @property
    def by_key(self) -> dict[tuple[str, str, str, str], SensorDescriptor]:
        return {sensor.key: sensor for sensor in self.sensors}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a serialized input artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_sample_ids(run_dir: str | Path, npz_name: str) -> list[str]:
    """List directory-style sample IDs in deterministic order."""

    directory = Path(run_dir)
    if not directory.is_dir():
        raise ContractError(f"run does not exist or is not a directory: {directory}")
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir() and (entry / npz_name).is_file()
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read JSON metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"metadata must be a JSON object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, TypeError) as error:
        raise ContractError(f"unable to read NPZ {path}: {error}") from error


def load_virtual_glove_sample(run_dir: str | Path, sample_id: str) -> VirtualGloveSample:
    sample_dir = Path(run_dir) / sample_id
    npz_path = sample_dir / VIRTUAL_GLOVE_NPZ_NAME
    meta_path = sample_dir / VIRTUAL_GLOVE_META_NAME
    if not npz_path.is_file():
        raise ContractError(f"{sample_id}: missing {VIRTUAL_GLOVE_NPZ_NAME}")
    if not meta_path.is_file():
        raise ContractError(f"{sample_id}: missing {VIRTUAL_GLOVE_META_NAME}")
    metadata = _load_json(meta_path)
    # TASK-006A writes the reusable physical layout beside each sample rather
    # than duplicating it into every metadata document.  Read that companion
    # artifact as part of the serialized contract; this is a read-only QA
    # compatibility path, not a mutation of the production output.
    if not any(key in metadata for key in ("sensor_layout", "sensorLayout", "layout")):
        for layout_path in (sample_dir / SENSOR_LAYOUT_NAME, Path(run_dir) / SENSOR_LAYOUT_NAME):
            if layout_path.is_file():
                metadata["sensor_layout"] = _load_json(layout_path)
                metadata["sensor_layout_source"] = str(layout_path)
                break
    return VirtualGloveSample(
        sample_id=sample_id,
        arrays=_load_npz(npz_path),
        metadata=metadata,
        path=sample_dir,
    )


def load_kinematics_sample(run_dir: str | Path, sample_id: str) -> KinematicsSample:
    sample_dir = Path(run_dir) / sample_id
    npz_path = sample_dir / KINEMATICS_NPZ_NAME
    meta_path = sample_dir / KINEMATICS_META_NAME
    if not npz_path.is_file():
        raise ContractError(f"{sample_id}: missing {KINEMATICS_NPZ_NAME}")
    if not meta_path.is_file():
        raise ContractError(f"{sample_id}: missing {KINEMATICS_META_NAME}")
    return KinematicsSample(
        sample_id=sample_id,
        arrays=_load_npz(npz_path),
        metadata=_load_json(meta_path),
        path=sample_dir,
    )


def _shape_matches(shape: tuple[int, ...], specification: tuple[int, ... | str], frame_count: int) -> bool:
    if len(shape) != len(specification):
        return False
    return all(
        actual == frame_count if expected == "F" else actual == expected
        for actual, expected in zip(shape, specification)
    )


def canonical_track(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in TRACK_NAMES else None


def canonical_track_order(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized = tuple(canonical_track(item) for item in value)
    if any(item is None for item in normalized):
        return None
    return tuple(item for item in normalized if item is not None)


def _canonical_finger(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    aliases = {"little": "pinky", "small": "pinky"}
    normalized = value.strip().lower()
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in FINGER_NAMES else None


def _canonical_joint(value: Any) -> str | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        index = int(value)
        return JOINT_NAMES[index] if 0 <= index < len(JOINT_NAMES) else None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "0": "MCP",
        "proximal": "MCP",
        "mcp": "MCP",
        "1": "PIP",
        "middle": "PIP",
        "pip": "PIP",
        "2": "DIP",
        "distal": "DIP",
        "dip": "DIP",
    }
    return aliases.get(normalized)


def _canonical_family(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bend": "bend",
        "bends": "bend",
        "flexion": "bend",
        "bend_angle": "bend",
        "bend_hall": "bend",
        "spread": "spread",
        "spreads": "spread",
        "adjacent_spread": "spread",
        "spread_angle": "spread",
        "spread_hall": "spread",
        "hall_bend_angular": "bend",
        "hall_spread_angular": "spread",
        "imu": "imu",
        "palm_imu": "imu",
        "palm": "imu",
        "imu_package": "imu",
    }
    return aliases.get(normalized)


def _canonical_pair(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first = _canonical_finger(value[0])
        second = _canonical_finger(value[1])
        if first is not None and second is not None:
            candidate = f"{first}_{second}"
            return candidate if candidate in SPREAD_NAMES else None
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "thumbindex": "thumb_index",
        "indexmiddle": "index_middle",
        "middlering": "middle_ring",
        "ringpinky": "ring_pinky",
        "thumb_index": "thumb_index",
        "index_middle": "index_middle",
        "middle_ring": "middle_ring",
        "ring_pinky": "ring_pinky",
    }
    return aliases.get(normalized)


def _first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _layout_entries(raw_layout: Any) -> list[tuple[dict[str, Any], str | None]]:
    """Flatten accepted layout containers without changing their contents."""

    if isinstance(raw_layout, list):
        return [(item, None) for item in raw_layout if isinstance(item, dict)]
    if not isinstance(raw_layout, dict):
        return []

    sensors = _first(raw_layout, ("sensors", "entries"))
    if isinstance(sensors, list):
        return [(item, None) for item in sensors if isinstance(item, dict)]

    grouped = _first(raw_layout, ("hands", "tracks", "by_hand"))
    if isinstance(grouped, dict):
        entries: list[tuple[dict[str, Any], str | None]] = []
        for group_name in sorted(grouped):
            group = grouped[group_name]
            if isinstance(group, dict):
                group = _first(group, ("sensors", "entries"), [])
            if isinstance(group, list):
                entries.extend((item, str(group_name)) for item in group if isinstance(item, dict))
        return entries

    entries = []
    for group_name in TRACK_NAMES:
        for spelling in (group_name, group_name.lower()):
            group = raw_layout.get(spelling)
            if isinstance(group, dict):
                group = _first(group, ("sensors", "entries"), [])
            if isinstance(group, list):
                entries.extend((item, group_name) for item in group if isinstance(item, dict))
                break
    return entries


def _location_parts(entry: dict[str, Any]) -> tuple[Any, dict[str, Any] | None, str | None]:
    location = _first(entry, ("logical_location", "logicalLocation", "location"))
    if isinstance(location, dict):
        return location, location, None
    if isinstance(location, str):
        return location, None, location
    return location, None, None


def _string_location_tokens(location: str) -> list[str]:
    return [token for token in location.replace("/", ".").replace(":", ".").split(".") if token]


def _descriptor_from_entry(
    entry: dict[str, Any],
    grouped_hand: str | None,
    failures: list[str],
    seen_ids: set[str],
    seen_assignments: set[tuple[str, str, str, str]],
) -> SensorDescriptor | None:
    sensor_id_value = _first(entry, ("sensor_id", "sensorId", "id"))
    sensor_id = sensor_id_value.strip() if isinstance(sensor_id_value, str) else ""
    if not sensor_id:
        failures.append("layout entry is missing a non-empty sensor_id")
    elif sensor_id in seen_ids:
        failures.append(f"layout duplicate sensor ID: {sensor_id}")
    else:
        seen_ids.add(sensor_id)

    location, location_dict, location_string = _location_parts(entry)
    if location is None or location == "" or (isinstance(location_dict, dict) and not location_dict):
        failures.append(f"{sensor_id or '<missing-id>'}: logical_location is missing")

    tokens = _string_location_tokens(location_string) if location_string else []
    token_lower = [token.strip().lower() for token in tokens]
    location_get = location_dict.get if location_dict is not None else lambda _name, _default=None: _default

    hand = canonical_track(
        _first(
            entry,
            ("hand", "track", "side"),
            _first(location_dict or {}, ("hand", "track", "side"), grouped_hand),
        )
    )
    if hand is None and grouped_hand is not None:
        hand = canonical_track(grouped_hand)
    if hand is None:
        for token in token_lower:
            hand = canonical_track(token)
            if hand is not None:
                break
    if hand is None:
        failures.append(f"{sensor_id or '<missing-id>'}: logical hand must be LEFT or RIGHT")

    sensor_type_value = _first(entry, ("sensor_type", "sensorType", "type"))
    sensor_type = sensor_type_value.strip().lower() if isinstance(sensor_type_value, str) else ""
    family_value = _first(
        entry,
        ("sensor_group", "channel_group", "family", "kind", "role", "sensor_role"),
        _first(location_dict or {}, ("sensor_group", "channel_group", "family", "kind", "role", "type")),
    )
    family = _canonical_family(family_value)
    if family is None:
        # TASK-006A's physical template uses the production sensor_type names
        # as the only type-level family hint for some entries.
        family = _canonical_family(sensor_type_value)
    if family is None:
        for token in token_lower:
            family = _canonical_family(token)
            if family is not None:
                break
    if family is None:
        failures.append(f"{sensor_id or '<missing-id>'}: logical sensor family is missing or invalid")

    valid_hall_types = {
        "hall",
        "magnetic",
        "hall_effect",
        "hall-effect",
        "hall/magnetic",
        "hall_bend_angular",
        "hall_spread_angular",
    }
    if family in {"bend", "spread"} and sensor_type not in valid_hall_types:
        failures.append(f"{sensor_id or '<missing-id>'}: Hall sensor has invalid sensor_type {sensor_type_value!r}")
    if family == "imu" and sensor_type not in {"imu", "imu_package"}:
        failures.append(f"{sensor_id or '<missing-id>'}: palm sensor has invalid sensor_type {sensor_type_value!r}")
    if family not in {"bend", "spread", "imu"}:
        sensor_type = sensor_type or "invalid"

    marker_value = _first(entry, ("display_marker", "displayMarker"))
    marker = marker_value if isinstance(marker_value, str) else ""
    expected_marker = "H" if family in {"bend", "spread"} else "IMU" if family == "imu" else ""
    if expected_marker and marker != expected_marker:
        failures.append(
            f"{sensor_id or '<missing-id>'}: display_marker must be {expected_marker!r}, got {marker_value!r}"
        )

    description_value = _first(entry, ("description", "label", "name"))
    description = description_value.strip() if isinstance(description_value, str) else ""
    if not description:
        failures.append(f"{sensor_id or '<missing-id>'}: description is missing")

    finger_value = _first(entry, ("finger",), _first(location_dict or {}, ("finger",)))
    if finger_value is None:
        for token in token_lower:
            if _canonical_finger(token) is not None:
                finger_value = token
                break
    finger = _canonical_finger(finger_value)

    joint_value = _first(entry, ("joint", "joint_name", "joint_index"), _first(location_dict or {}, ("joint", "joint_name", "joint_index")))
    if joint_value is None:
        for token in token_lower:
            if _canonical_joint(token) is not None:
                joint_value = token
                break
    joint = _canonical_joint(joint_value)

    pair_value = _first(entry, ("spread_pair", "pair", "adjacent_pair"), _first(location_dict or {}, ("spread_pair", "pair", "adjacent_pair")))
    if pair_value is None and len(token_lower) >= 2:
        for start in range(len(token_lower) - 1):
            candidate = _canonical_pair(token_lower[start : start + 2])
            if candidate is not None:
                pair_value = candidate
                break
    spread_pair = _canonical_pair(pair_value)

    if family == "bend":
        if finger is None or joint is None:
            failures.append(f"{sensor_id or '<missing-id>'}: bend logical_location lacks finger/joint coverage")
        assignment = (hand or "<missing-hand>", family, finger or "", joint or "")
    elif family == "spread":
        if spread_pair is None:
            failures.append(f"{sensor_id or '<missing-id>'}: spread logical_location lacks an adjacent pair")
        assignment = (hand or "<missing-hand>", family, spread_pair or "", "")
    elif family == "imu":
        # All palm IMUs use one logical assignment per hand.  A non-empty
        # location is still required above so a future visualizer can place it.
        assignment = (hand or "<missing-hand>", family, "palm", "")
    else:
        assignment = (hand or "<missing-hand>", family or "", "", "")

    if assignment in seen_assignments:
        failures.append(f"{sensor_id or '<missing-id>'}: duplicate logical sensor assignment {assignment}")
    else:
        seen_assignments.add(assignment)

    if not sensor_id or hand is None or family is None:
        return None
    return SensorDescriptor(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        hand=hand,
        family=family,
        finger=finger,
        joint=joint,
        spread_pair=spread_pair,
        display_marker=marker,
        description=description,
        logical_location=location,
    )


def parse_sensor_layout(metadata: dict[str, Any]) -> LayoutResult:
    """Validate and normalize the required two-hand sensor layout."""

    raw_layout = _first(metadata, ("sensor_layout", "sensorLayout", "layout"))
    failures: list[str] = []
    if raw_layout is None:
        return LayoutResult(False, ("metadata is missing sensor_layout",), ())

    entries = _layout_entries(raw_layout)
    # TASK-006A's layout is one physical placement template (15 bend + 4
    # spread + 1 IMU) reused by both canonical tracks.  Expand it only when it
    # is genuinely hand-independent and has exactly the expected template
    # cardinality.  Malformed or partially hand-labelled layouts continue
    # through the normal 40-entry validation and fail rather than being
    # repaired.
    template_expanded = False
    if len(entries) == SENSORS_PER_HAND and all(grouped_hand is None for _, grouped_hand in entries):
        has_explicit_hand = False
        for raw_entry, _ in entries:
            location = _first(raw_entry, ("logical_location", "logicalLocation", "location"))
            if any(key in raw_entry for key in ("hand", "track", "side")):
                has_explicit_hand = True
            if isinstance(location, dict) and any(key in location for key in ("hand", "track", "side")):
                has_explicit_hand = True
        if not has_explicit_hand:
            expanded: list[tuple[dict[str, Any], str | None]] = []
            for hand in TRACK_NAMES:
                for raw_entry, _ in entries:
                    entry = dict(raw_entry)
                    template_id = entry.get("sensor_id", entry.get("sensorId", entry.get("id")))
                    if isinstance(template_id, str) and template_id.strip():
                        entry["template_sensor_id"] = template_id.strip()
                        entry["sensor_id"] = f"{hand}.{template_id.strip()}"
                    entry["hand"] = hand
                    expanded.append((entry, None))
            entries = expanded
            template_expanded = True
    if not entries:
        failures.append("sensor_layout must contain a non-empty sensor list")

    seen_ids: set[str] = set()
    seen_assignments: set[tuple[str, str, str, str]] = set()
    sensors: list[SensorDescriptor] = []
    for raw_entry, grouped_hand in entries:
        entry = dict(raw_entry)
        if grouped_hand is not None:
            entry.setdefault("hand", grouped_hand)
        descriptor = _descriptor_from_entry(
            entry,
            grouped_hand,
            failures,
            seen_ids,
            seen_assignments,
        )
        if descriptor is not None:
            sensors.append(descriptor)

    if len(entries) != TOTAL_SENSOR_COUNT:
        failures.append(
            f"sensor_layout must contain exactly {TOTAL_SENSOR_COUNT} entries, got {len(entries)}"
        )

    expected_bend = {
        (hand, "bend", finger, joint)
        for hand in TRACK_NAMES
        for finger in FINGER_NAMES
        for joint in JOINT_NAMES
    }
    expected_spread = {
        (hand, "spread", pair, "") for hand in TRACK_NAMES for pair in SPREAD_NAMES
    }
    expected_imu = {(hand, "imu", "palm", "") for hand in TRACK_NAMES}
    observed = {sensor.key for sensor in sensors}
    for missing in sorted((expected_bend | expected_spread | expected_imu) - observed):
        failures.append(f"missing logical sensor assignment: {missing}")
    for extra in sorted(observed - (expected_bend | expected_spread | expected_imu)):
        failures.append(f"invalid logical sensor assignment: {extra}")

    for hand in TRACK_NAMES:
        hand_sensors = [sensor for sensor in sensors if sensor.hand == hand]
        bend_count = sum(sensor.family == "bend" for sensor in hand_sensors)
        spread_count = sum(sensor.family == "spread" for sensor in hand_sensors)
        imu_count = sum(sensor.family == "imu" for sensor in hand_sensors)
        if (bend_count, spread_count, imu_count) != (BEND_SENSOR_COUNT, SPREAD_SENSOR_COUNT, PALM_IMU_COUNT):
            failures.append(
                f"{hand}: expected 15 bend Hall + 4 spread Hall + 1 palm IMU, "
                f"got {bend_count} bend + {spread_count} spread + {imu_count} IMU"
            )

    declared_counts = metadata.get("sensor_counts")
    if declared_counts is None:
        declared_counts = metadata.get("sensor_counts_per_hand")
    if declared_counts is None and isinstance(raw_layout, dict):
        declared_counts = raw_layout.get("per_hand_counts")
    if declared_counts is not None:
        if not isinstance(declared_counts, dict):
            failures.append("sensor_counts must be a JSON object when present")
        else:
            expected_counts = {
                "bend_hall": (BEND_SENSOR_COUNT, "bend_hall_sensors"),
                "spread_hall": (SPREAD_SENSOR_COUNT, "spread_hall_sensors"),
                "hall_total": (HALL_SENSOR_COUNT, "hall_sensors_total"),
                "palm_imu": (PALM_IMU_COUNT, "imu_packages"),
            }
            for key, (expected, production_key) in expected_counts.items():
                declared = declared_counts.get(key, declared_counts.get(production_key))
                if declared is not None and declared != expected:
                    failures.append(
                        f"sensor_counts.{key} must be {expected}, got {declared!r}"
                    )

    sensors.sort(key=lambda sensor: sensor.key + (sensor.sensor_id,))
    return LayoutResult(
        not failures,
        tuple(sorted(failures)),
        tuple(sensors),
        representation=(
            "physical_template_expanded" if template_expanded else "runtime_identities"
        ),
        physical_template_count=SENSORS_PER_HAND if template_expanded else None,
    )


def validate_sample_contract(sample: VirtualGloveSample) -> dict[str, Any]:
    """Validate one virtual-glove sample's file, array, and layout structure."""

    arrays = sample.arrays
    metadata = sample.metadata
    failures: list[str] = []
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        failures.append(f"missing arrays: {missing}")

    frame_array = np.asarray(arrays.get("frame_index", np.empty(0)))
    if frame_array.ndim != 1:
        failures.append(f"frame_index must be rank-1, got shape {frame_array.shape}")
        frame_count = 0
    else:
        frame_count = int(frame_array.shape[0])
        if frame_count == 0:
            failures.append("virtual-glove output must contain at least one frame")
        if not np.issubdtype(frame_array.dtype, np.integer):
            failures.append(f"frame_index must have integer dtype, got {frame_array.dtype}")

    for name, specification in REQUIRED_ARRAYS.items():
        if name not in arrays:
            continue
        shape = tuple(np.asarray(arrays[name]).shape)
        if not _shape_matches(shape, specification, frame_count):
            failures.append(f"{name} shape mismatch: expected {specification} with F={frame_count}, got {shape}")

    for name in ("bend_valid", "spread_valid", "palm_imu_valid"):
        if name in arrays and np.asarray(arrays[name]).dtype != np.dtype(np.bool_):
            failures.append(f"{name} must have bool dtype, got {np.asarray(arrays[name]).dtype}")

    for name in ("tracking_state_code", "source_raw_detection_index"):
        if name in arrays and not np.issubdtype(np.asarray(arrays[name]).dtype, np.integer):
            failures.append(f"{name} must have integer dtype, got {np.asarray(arrays[name]).dtype}")

    numeric_fields = (
        "timestamp_seconds",
        "bend_angle_deg",
        "bend_normalized",
        "spread_angle_deg",
        "spread_normalized",
        "imu_rotation_matrix",
        "imu_quaternion_wxyz",
    )
    for name in numeric_fields:
        if name in arrays and not np.issubdtype(np.asarray(arrays[name]).dtype, np.floating):
            failures.append(f"{name} must have floating-point dtype, got {np.asarray(arrays[name]).dtype}")

    for name, specification in OPTIONAL_ARRAYS.items():
        if name not in arrays:
            continue
        shape = tuple(np.asarray(arrays[name]).shape)
        if not _shape_matches(shape, specification, frame_count):
            failures.append(f"{name} shape mismatch: expected {specification} with F={frame_count}, got {shape}")
    has_gyro = "imu_angular_velocity_rad_s" in arrays
    has_gyro_valid = "imu_angular_velocity_valid" in arrays
    if has_gyro != has_gyro_valid:
        failures.append("imu_angular_velocity_rad_s and imu_angular_velocity_valid must be provided together")
    if has_gyro_valid and np.asarray(arrays["imu_angular_velocity_valid"]).dtype != np.dtype(np.bool_):
        failures.append(
            "imu_angular_velocity_valid must have bool dtype, "
            f"got {np.asarray(arrays['imu_angular_velocity_valid']).dtype}"
        )
    for name in ("bend_adc_12bit", "spread_adc_12bit"):
        if name in arrays:
            dtype = np.asarray(arrays[name]).dtype
            if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
                failures.append(f"{name} must have real numeric dtype, got {dtype}")
    if has_gyro and not np.issubdtype(np.asarray(arrays["imu_angular_velocity_rad_s"]).dtype, np.floating):
        failures.append(
            "imu_angular_velocity_rad_s must have floating-point dtype, "
            f"got {np.asarray(arrays['imu_angular_velocity_rad_s']).dtype}"
        )

    if not isinstance(metadata, dict):
        failures.append("metadata must be a JSON object")
    else:
        if metadata.get("sample_id") != sample.sample_id:
            failures.append(
                f"metadata sample_id must be {sample.sample_id!r}, got {metadata.get('sample_id')!r}"
            )
        if not isinstance(metadata.get("schema_version"), str) or not metadata.get("schema_version"):
            failures.append("metadata schema_version is required")
        task = metadata.get("task")
        if not isinstance(task, str) or not task.upper().startswith("TASK-006"):
            failures.append(f"metadata task must identify TASK-006, got {task!r}")
        stage = metadata.get("stage")
        if not isinstance(stage, str) or "glove" not in stage.lower():
            failures.append(f"metadata stage must identify virtual glove output, got {stage!r}")
        if canonical_track_order(metadata.get("track_order")) != TRACK_NAMES:
            failures.append(
                f"metadata track_order must be {list(TRACK_NAMES)}, got {metadata.get('track_order')!r}"
            )

    layout = parse_sensor_layout(metadata if isinstance(metadata, dict) else {})
    failures.extend(layout.failures)
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "frame_count": frame_count,
        "track_count": 2 if frame_count and "tracking_state_code" in arrays else 0,
        "layout": layout,
    }


def validate_kinematics_input(sample: KinematicsSample) -> dict[str, Any]:
    """Check only the TASK-005 fields needed as a read-only QA source."""

    arrays = sample.arrays
    failures: list[str] = []
    required = {
        "frame_index": ("F",),
        "timestamp_seconds": ("F",),
        "tracking_state_code": ("F", 2),
        "source_raw_detection_index": ("F", 2),
        "flexion_deg": ("F", 2, 5, 3),
        "adjacent_spread_deg": ("F", 2, 4),
        "palm_rotation_matrix": ("F", 2, 3, 3),
        "palm_quaternion_wxyz": ("F", 2, 4),
        "valid_palm_frame": ("F", 2),
    }
    missing = sorted(set(required) - set(arrays))
    if missing:
        failures.append(f"missing TASK-005 arrays: {missing}")
    frame_array = np.asarray(arrays.get("frame_index", np.empty(0)))
    frame_count = int(frame_array.shape[0]) if frame_array.ndim == 1 else 0
    if frame_array.ndim != 1:
        failures.append(f"TASK-005 frame_index must be rank-1, got shape {frame_array.shape}")
    elif frame_count == 0:
        failures.append("TASK-005 input must contain at least one frame")
    for name, specification in required.items():
        if name in arrays and not _shape_matches(tuple(np.asarray(arrays[name]).shape), specification, frame_count):
            failures.append(f"TASK-005 {name} shape mismatch: expected {specification} with F={frame_count}, got {np.asarray(arrays[name]).shape}")
    if "valid_palm_frame" in arrays and np.asarray(arrays["valid_palm_frame"]).dtype != np.dtype(np.bool_):
        failures.append("TASK-005 valid_palm_frame must have bool dtype")
    for name in ("tracking_state_code", "source_raw_detection_index"):
        if name in arrays and not np.issubdtype(np.asarray(arrays[name]).dtype, np.integer):
            failures.append(f"TASK-005 {name} must have integer dtype")
    for name in ("timestamp_seconds", "flexion_deg", "adjacent_spread_deg", "palm_rotation_matrix", "palm_quaternion_wxyz"):
        if name in arrays and not np.issubdtype(np.asarray(arrays[name]).dtype, np.floating):
            failures.append(f"TASK-005 {name} must have floating-point dtype")
    if sample.metadata.get("sample_id") != sample.sample_id:
        failures.append("TASK-005 metadata sample_id does not match its directory")
    track_order = canonical_track_order(sample.metadata.get("track_order"))
    if track_order != TRACK_NAMES:
        failures.append(f"TASK-005 metadata track_order must be {list(TRACK_NAMES)}")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "frame_count": frame_count,
    }


__all__ = [
    "ALIGNMENT_ARRAYS",
    "BEND_SENSOR_COUNT",
    "ContractError",
    "FINGER_NAMES",
    "JOINT_NAMES",
    "KinematicsSample",
    "KINEMATICS_META_NAME",
    "KINEMATICS_NPZ_NAME",
    "LayoutResult",
    "NORMALIZATION_ATOL",
    "NORMALIZATION_RTOL",
    "OPTIONAL_ARRAYS",
    "PALM_IMU_COUNT",
    "REQUIRED_ARRAYS",
    "SensorDescriptor",
    "SPREAD_NAMES",
    "TRACK_NAMES",
    "TOTAL_SENSOR_COUNT",
    "VirtualGloveSample",
    "VIRTUAL_GLOVE_META_NAME",
    "VIRTUAL_GLOVE_NPZ_NAME",
    "SENSOR_LAYOUT_NAME",
    "canonical_track_order",
    "list_sample_ids",
    "load_kinematics_sample",
    "load_virtual_glove_sample",
    "parse_sensor_layout",
    "sha256_file",
    "validate_kinematics_input",
    "validate_sample_contract",
]
