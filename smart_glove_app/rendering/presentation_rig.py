"""TASK-007G presentation rig profile.

The profile describes only how the frozen TASK-005/TASK-006 channels are shown
on a rigged hand.  It carries no scientific contract: nothing here is fed back
into tracking, kinematics, the virtual glove or the recognizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SIDES: tuple[str, str] = ("LEFT", "RIGHT")
FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
SPREAD_PAIRS: tuple[str, ...] = ("thumb-index", "index-middle", "middle-ring", "ring-pinky")
VIEW_MODES: tuple[str, str] = ("PALM", "BACK")

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "assets" / "rig_profiles" / "task007g_hands.json"


class PresentationRigError(ValueError):
    """Raised when a rig profile cannot drive the presentation layer."""


@dataclass(frozen=True, slots=True)
class FingerChain:
    metacarpal: str
    joints: tuple[str, str, str]
    joint_limits_deg: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SpreadTarget:
    bone: str
    sign: float
    sum_of: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PresentationRig:
    profile_id: str
    raw: Mapping[str, Any]
    roots: Mapping[str, str]
    armatures: Mapping[str, str]
    meshes: Mapping[str, str]
    required_bones: tuple[str, ...]
    wrist_bone: str
    wrist_max_angle_deg: float
    bend_axis: str
    bend_sign: float
    spread_axis: str
    chains: Mapping[str, FingerChain]
    spread_neutral_deg: Mapping[str, float]
    spread_clamp_deg: tuple[float, float]
    spread_targets: tuple[SpreadTarget, ...]
    glb_filenames: Mapping[str, str]

    # ---- presentation layout -------------------------------------------------

    @property
    def presentation(self) -> Mapping[str, Any]:
        return self.raw["presentation"]

    def root_position(self, side: str) -> tuple[float, float, float]:
        key = "left_position" if side.upper() == "LEFT" else "right_position"
        return tuple(float(v) for v in self.presentation[key])  # type: ignore[return-value]

    def view_euler_deg(self, view: str) -> tuple[float, float, float]:
        mode = str(view).upper()
        if mode not in VIEW_MODES:
            raise PresentationRigError(f"unknown view mode: {view!r}")
        return tuple(float(v) for v in self.presentation["views"][mode]["root_euler_deg"])  # type: ignore[return-value]

    def as_qml(self) -> dict[str, Any]:
        """Plain JSON-compatible view for the QML scene."""

        return {
            "profile_id": self.profile_id,
            "roots": dict(self.roots),
            "armatures": dict(self.armatures),
            "meshes": dict(self.meshes),
            "required_bones": list(self.required_bones),
            "glbFilenames": dict(self.glb_filenames),
            "presentation": json.loads(json.dumps(self.presentation)),
        }


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise PresentationRigError(f"rig profile is missing required key: {key!r}")
    return payload[key]


def load_presentation_rig(path: str | Path | None = None) -> PresentationRig:
    target = Path(path).expanduser() if path is not None else DEFAULT_PROFILE_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PresentationRigError(f"cannot read rig profile {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PresentationRigError(f"rig profile {target} is not valid JSON: {exc}") from exc

    if str(payload.get("schema_version", "")) != "task007g.presentation-rig.v1":
        raise PresentationRigError(
            f"unsupported rig profile schema: {payload.get('schema_version')!r}"
        )

    roots = {str(k).upper(): str(v) for k, v in _require(payload, "roots").items()}
    armatures = {str(k).upper(): str(v) for k, v in _require(payload, "armatures").items()}
    meshes = {str(k).upper(): str(v) for k, v in _require(payload, "meshes").items()}
    for mapping, name in ((roots, "roots"), (armatures, "armatures"), (meshes, "meshes")):
        if set(mapping) != set(SIDES):
            raise PresentationRigError(f"rig profile {name} must define exactly LEFT and RIGHT")

    articulation = _require(payload, "articulation")
    chains: dict[str, FingerChain] = {}
    for finger in FINGERS:
        if finger not in articulation["chains"]:
            raise PresentationRigError(f"rig profile has no chain for finger {finger!r}")
        entry = articulation["chains"][finger]
        joints = tuple(str(v) for v in entry["joints"])
        limits = tuple(float(v) for v in entry["joint_limits_deg"])
        if len(joints) != 3 or len(limits) != 3:
            raise PresentationRigError(f"finger chain {finger!r} must declare 3 joints and 3 limits")
        chains[finger] = FingerChain(str(entry["metacarpal"]), joints, limits)  # type: ignore[arg-type]

    spread = articulation["spread"]
    neutral = {str(k): float(v) for k, v in spread["neutral_deg"].items()}
    if set(neutral) != set(SPREAD_PAIRS):
        raise PresentationRigError("spread neutral_deg must cover exactly the four TASK-006 pairs")
    targets = tuple(
        SpreadTarget(str(bone), float(entry["sign"]), tuple(str(v) for v in entry["sum_of"]))
        for bone, entry in spread["targets"].items()
    )
    for spread_target in targets:
        unknown = set(spread_target.sum_of) - set(SPREAD_PAIRS)
        if unknown:
            raise PresentationRigError(
                f"spread target {spread_target.bone!r} references unknown pairs: {sorted(unknown)}"
            )

    required = tuple(str(v) for v in _require(payload, "required_bones"))
    declared = {chain.metacarpal for chain in chains.values()} | {
        joint for chain in chains.values() for joint in chain.joints
    }
    missing = declared - set(required)
    if missing:
        raise PresentationRigError(f"required_bones is missing articulated bones: {sorted(missing)}")

    wrist = _require(payload, "wrist")
    if str(wrist["bone"]) not in required:
        raise PresentationRigError("wrist bone is not listed in required_bones")

    presentation = _require(payload, "presentation")
    for mode in VIEW_MODES:
        if mode not in presentation.get("views", {}):
            raise PresentationRigError(f"rig profile does not define the {mode} view")

    glb_filenames = {
        str(k).upper(): str(v) for k, v in payload.get("asset", {}).get("glb_filenames", {}).items()
    }
    if set(glb_filenames) != set(SIDES):
        raise PresentationRigError("rig profile must name one GLB file per hand")

    clamp = tuple(float(v) for v in spread["clamp_deg"])
    if len(clamp) != 2 or clamp[0] >= clamp[1]:
        raise PresentationRigError("spread clamp_deg must be an ordered [min, max] pair")

    return PresentationRig(
        profile_id=str(payload.get("profile_id", target.stem)),
        raw=payload,
        roots=roots,
        armatures=armatures,
        meshes=meshes,
        required_bones=required,
        wrist_bone=str(wrist["bone"]),
        wrist_max_angle_deg=float(wrist.get("max_angle_deg", 25.0)),
        bend_axis=str(articulation["bend_axis"]).upper(),
        bend_sign=float(articulation["bend_sign"]),
        spread_axis=str(articulation["spread_axis"]).upper(),
        chains=chains,
        spread_neutral_deg=neutral,
        spread_clamp_deg=clamp,  # type: ignore[arg-type]
        spread_targets=targets,
        glb_filenames=glb_filenames,
    )


__all__ = [
    "FINGERS",
    "SIDES",
    "SPREAD_PAIRS",
    "VIEW_MODES",
    "FingerChain",
    "PresentationRig",
    "PresentationRigError",
    "SpreadTarget",
    "load_presentation_rig",
]
