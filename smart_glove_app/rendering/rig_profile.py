"""Validated, project-owned runtime mapping for the Blender hand asset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


SIDES = ("LEFT", "RIGHT")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = ("proximal", "middle", "distal")
SPREADS = ("thumb-index", "index-middle", "middle-ring", "ring-pinky")
AXES = frozenset({"X", "Y", "Z"})


class RigProfileError(ValueError):
    """Raised when a runtime rig profile is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ChannelCalibration:
    bone: str
    axis: str
    sign: int
    neutral_offset_deg: float
    safe_min_deg: float
    safe_max_deg: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, channel: str, side: str) -> "ChannelCalibration":
        try:
            bone = str(value["bone"])
            axis = str(value["axis"]).upper()
            sign = int(value["sign"])
            neutral = float(value.get("neutral_offset_deg", 0.0))
            safe_min = float(value["safe_min_deg"])
            safe_max = float(value["safe_max_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RigProfileError(f"invalid calibration for {channel}/{side}") from exc
        if not bone or axis not in AXES or sign not in (-1, 1):
            raise RigProfileError(f"invalid bone/axis/sign for {channel}/{side}")
        if not safe_min <= safe_max:
            raise RigProfileError(f"safe range is inverted for {channel}/{side}")
        return cls(bone, axis, sign, neutral, safe_min, safe_max)


@dataclass(frozen=True, slots=True)
class RigProfile:
    """Immutable mapping consumed by the presentation-only retargeter."""

    path: Path
    raw: Mapping[str, Any]
    presentation_roots: Mapping[str, str]
    armatures: Mapping[str, str]
    runtime_node_paths: Mapping[str, Mapping[str, Any]]
    root_bone: str
    palm_bone: str
    required_deform_bones: tuple[str, ...]
    bends: Mapping[str, Mapping[str, ChannelCalibration]]
    spreads: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_path(cls, path: str | Path) -> "RigProfile":
        resolved = Path(path).expanduser().resolve()
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RigProfileError(f"rig profile is not readable: {resolved}") from exc
        except json.JSONDecodeError as exc:
            raise RigProfileError(f"rig profile is not valid JSON: {resolved}") from exc
        if not isinstance(raw, Mapping):
            raise RigProfileError("rig profile root must be an object")

        roots = _side_mapping(raw, "presentation_roots")
        arms = _side_mapping(raw, "armatures")
        runtime_paths = _runtime_paths(raw)
        root_bone = _required_string(raw, "root_bone")
        palm_bone = _required_string(raw, "palm_bone")
        required = tuple(str(value) for value in raw.get("required_deform_bones", ()))
        if len(required) != 22 or len(set(required)) != 22:
            raise RigProfileError("required_deform_bones must contain 22 unique names")

        raw_bends = raw.get("bend_channels")
        if not isinstance(raw_bends, Mapping):
            raise RigProfileError("bend_channels must be an object")
        expected_bends = {f"{finger}[{index}]" for finger in FINGERS for index in range(3)}
        if set(raw_bends) != expected_bends:
            raise RigProfileError(f"bend channels must be exactly {sorted(expected_bends)}")
        bends: dict[str, dict[str, ChannelCalibration]] = {}
        for channel, item in raw_bends.items():
            if not isinstance(item, Mapping):
                raise RigProfileError(f"bend channel {channel} must be an object")
            parsed_bend: dict[str, ChannelCalibration] = {}
            for side in SIDES:
                side_value = item.get(side)
                if not isinstance(side_value, Mapping):
                    raise RigProfileError(f"bend channel {channel} must contain {side}")
                parsed_bend[side] = ChannelCalibration.from_mapping(
                    side_value,
                    channel=str(channel),
                    side=side,
                )
            bends[str(channel)] = parsed_bend

        raw_spreads = raw.get("spread_channels")
        if not isinstance(raw_spreads, Mapping) or set(raw_spreads) != set(SPREADS):
            raise RigProfileError(f"spread channels must be exactly {list(SPREADS)}")
        spreads: dict[str, dict[str, Any]] = {}
        for channel, item in raw_spreads.items():
            if not isinstance(item, Mapping):
                raise RigProfileError(f"spread channel {channel} must be an object")
            driver = str(item.get("driver_bone", ""))
            reference = str(item.get("reference_bone", ""))
            if not driver or not reference:
                raise RigProfileError(f"spread channel {channel} lacks a driver/reference bone")
            parsed_spread: dict[str, Any] = {
                "driver_bone": driver,
                "reference_bone": reference,
            }
            for side in SIDES:
                side_value = item.get(side)
                if not isinstance(side_value, Mapping):
                    raise RigProfileError(f"spread channel {channel} must contain {side}")
                side_mapping = dict(side_value)
                side_mapping["bone"] = driver
                parsed_spread[side] = ChannelCalibration.from_mapping(
                    side_mapping,
                    channel=str(channel),
                    side=side,
                )
            spreads[str(channel)] = parsed_spread

        profile = cls(resolved, raw, roots, arms, runtime_paths, root_bone, palm_bone, required, bends, spreads)
        profile.validate()
        return profile

    def validate(self) -> None:
        for side in SIDES:
            if not self.presentation_roots[side] or not self.armatures[side]:
                raise RigProfileError(f"missing {side} presentation root or armature")
            side_paths = self.runtime_node_paths[side]
            if not _valid_path(side_paths.get("presentation_root")) or not _valid_path(side_paths.get("armature")):
                raise RigProfileError(f"invalid runtime node path for {side} root/armature")
            bone_paths = side_paths.get("bones")
            if not isinstance(bone_paths, Mapping) or any(
                not _valid_path(bone_paths.get(bone)) for bone in self.required_deform_bones
            ):
                raise RigProfileError(f"runtime node paths must cover all deform bones for {side}")
        if self.root_bone not in self.required_deform_bones or self.palm_bone not in self.required_deform_bones:
            raise RigProfileError("root and palm bones must be required deform bones")
        for channel, sides in self.bends.items():
            for side, calibration in sides.items():
                if calibration.bone not in self.required_deform_bones:
                    raise RigProfileError(f"{channel}/{side} targets a non-deform bone")
        for channel, item in self.spreads.items():
            if item["driver_bone"] not in self.required_deform_bones:
                raise RigProfileError(f"{channel} driver is not a required deform bone")

    @property
    def source_asset(self) -> Mapping[str, Any]:
        return self.raw.get("asset", {})

    @property
    def direct_mode(self) -> Mapping[str, Any]:
        return self.raw.get("direct_mode", {})

    @property
    def profile_id(self) -> str:
        return str(self.raw.get("profile_id", self.path.stem))

    def qml_metadata(self) -> dict[str, Any]:
        """Return only simple QVariant-compatible profile metadata for QML."""

        return {
            "profile_id": self.profile_id,
            "presentation_roots": dict(self.presentation_roots),
            "armatures": dict(self.armatures),
            "runtime_node_paths": {
                side: {
                    "presentation_root": list(self.runtime_node_paths[side]["presentation_root"]),
                    "armature": list(self.runtime_node_paths[side]["armature"]),
                    "bones": {
                        bone: list(path)
                        for bone, path in self.runtime_node_paths[side]["bones"].items()
                    },
                }
                for side in SIDES
            },
            "root_bone": self.root_bone,
            "palm_bone": self.palm_bone,
            "required_deform_bones": list(self.required_deform_bones),
        }


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = str(value.get(key, ""))
    if not item:
        raise RigProfileError(f"rig profile lacks {key}")
    return item


def _side_mapping(value: Mapping[str, Any], key: str) -> dict[str, str]:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise RigProfileError(f"{key} must be an object")
    result = {side: str(raw.get(side, "")) for side in SIDES}
    if any(not item for item in result.values()):
        raise RigProfileError(f"{key} must contain LEFT and RIGHT")
    return result


def _valid_path(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(index, int) and index >= 0 for index in value
    )


def _runtime_paths(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("runtime_node_paths")
    if not isinstance(raw, Mapping):
        raise RigProfileError("runtime_node_paths must be an object")
    result: dict[str, Mapping[str, Any]] = {}
    for side in SIDES:
        item = raw.get(side)
        if not isinstance(item, Mapping):
            raise RigProfileError(f"runtime_node_paths must contain {side}")
        result[side] = item
    return result


def default_rig_profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "rig_profiles" / "blendswap_hands_v1.json"


def load_rig_profile(path: str | Path | None = None) -> RigProfile:
    return RigProfile.from_path(default_rig_profile_path() if path is None else path)


__all__ = [
    "AXES",
    "CHAINS",
    "ChannelCalibration",
    "FINGERS",
    "RigProfile",
    "RigProfileError",
    "SIDES",
    "SPREADS",
    "default_rig_profile_path",
    "load_rig_profile",
]
