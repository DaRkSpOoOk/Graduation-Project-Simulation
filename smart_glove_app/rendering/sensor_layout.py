"""Authoritative presentation anchors for the frozen virtual-glove layout.

The scientific sensor contract lives in :mod:`virtual_glove.layout` and is
validated again when a TASK-008 sequence is loaded.  This module adds only the
display-side information needed to attach a marker to the already rendered
TASK-007G/I skeleton.  It deliberately does not create, transform, or persist
scientific sensor values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from visualizer.contract import SensorSpec, validate_sensor_layout
from virtual_glove.layout import layout_document


DEFAULT_SENSOR_LAYOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "sensor_layouts"
    / "core28_virtual_glove_v1.json"
)


class SensorPresentationLayoutError(ValueError):
    """Raised when a presentation map could point at the wrong channel."""


@dataclass(frozen=True, slots=True)
class SensorPresentationSpec:
    """One frozen-contract sensor plus its display-only rig anchor."""

    sensor_id: str
    sensor_type: str
    marker: str
    group: str
    display_name: str
    short_id: str
    description: str
    finger: str | None
    joint: str | None
    pair: tuple[str, str] | None
    array: str
    array_index: tuple[int, ...]
    anchor_kind: str
    anchor_bone: str | None
    anchor_bones: tuple[str, ...]
    local_offset: tuple[float, float, float]
    palm_blend: float

    @property
    def is_imu(self) -> bool:
        return self.sensor_type == "imu_package"

    @property
    def has_angle(self) -> bool:
        return self.sensor_type in {"hall_bend_angular", "hall_spread_angular"}

    def as_qml(self) -> dict[str, Any]:
        """Return a plain QVariant-friendly definition for QML."""

        return {
            "sensorId": self.sensor_id,
            "sensorType": self.sensor_type,
            "marker": self.marker,
            "group": self.group,
            "displayName": self.display_name,
            "shortId": self.short_id,
            "description": self.description,
            "finger": self.finger or "",
            "joint": self.joint or "",
            "pair": list(self.pair) if self.pair is not None else [],
            "array": self.array,
            "arrayIndex": list(self.array_index),
            "anchorKind": self.anchor_kind,
            "anchorBone": self.anchor_bone or "",
            "anchorBones": list(self.anchor_bones),
            "localOffset": list(self.local_offset),
            "palmBlend": self.palm_blend,
        }


def _as_float_triplet(value: Any, *, field: str, sensor_id: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SensorPresentationLayoutError(
            f"{sensor_id}: {field} must contain three finite numbers"
        ) from exc
    if len(result) != 3 or not all(item == item and abs(item) != float("inf") for item in result):
        raise SensorPresentationLayoutError(
            f"{sensor_id}: {field} must contain three finite numbers"
        )
    return result  # type: ignore[return-value]


def _contract_by_id() -> tuple[SensorSpec, ...]:
    return validate_sensor_layout(layout_document())


def load_sensor_presentation_layout(
    path: str | Path | None = None,
) -> tuple[SensorPresentationSpec, ...]:
    """Load and cross-check the project-owned 20-entry presentation map.

    The ID/order/type/array checks are intentionally strict.  A typo in a QML
    anchor must never silently move a numeric value to another scientific
    channel.
    """

    target = Path(path).expanduser() if path is not None else DEFAULT_SENSOR_LAYOUT_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SensorPresentationLayoutError(f"cannot read sensor layout {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SensorPresentationLayoutError(f"sensor layout {target} is not valid JSON: {exc}") from exc

    if payload.get("schema_version") != "task007j.sensor-presentation.v1":
        raise SensorPresentationLayoutError(
            f"unsupported sensor presentation schema: {payload.get('schema_version')!r}"
        )
    hands = tuple(str(value).upper() for value in payload.get("hand_applicability", ()))
    if hands != ("LEFT", "RIGHT"):
        raise SensorPresentationLayoutError(
            "sensor presentation map must apply to exactly LEFT and RIGHT hands"
        )
    raw_sensors = payload.get("sensors")
    if not isinstance(raw_sensors, list) or len(raw_sensors) != 20:
        raise SensorPresentationLayoutError("sensor presentation map must contain exactly 20 sensors")

    contract = _contract_by_id()
    contract_by_id = {sensor.sensor_id: sensor for sensor in contract}
    specs: list[SensorPresentationSpec] = []
    for index, raw in enumerate(raw_sensors):
        if not isinstance(raw, Mapping):
            raise SensorPresentationLayoutError(f"sensor entry {index} is not an object")
        sensor_id = str(raw.get("sensor_id", ""))
        if sensor_id not in contract_by_id:
            raise SensorPresentationLayoutError(f"unknown sensor ID in presentation map: {sensor_id!r}")
        expected = contract_by_id[sensor_id]
        if index != contract.index(expected):
            raise SensorPresentationLayoutError(
                f"sensor order is not the frozen TASK-006 order at {sensor_id}"
            )
        for field, expected_value in (
            ("sensor_type", expected.sensor_type),
            ("marker", expected.display_marker),
            ("array", expected.array),
        ):
            if raw.get(field) != expected_value:
                raise SensorPresentationLayoutError(
                    f"{sensor_id}: {field} disagrees with the frozen contract"
                )
        array_index = tuple(int(item) for item in raw.get("array_index", ()))
        if array_index != expected.array_index:
            raise SensorPresentationLayoutError(
                f"{sensor_id}: array_index disagrees with the frozen contract"
            )
        anchor = raw.get("anchor")
        if not isinstance(anchor, Mapping):
            raise SensorPresentationLayoutError(f"{sensor_id}: missing anchor object")
        anchor_kind = str(anchor.get("kind", ""))
        if anchor_kind not in {"bone", "spread_midpoint"}:
            raise SensorPresentationLayoutError(f"{sensor_id}: unsupported anchor kind {anchor_kind!r}")
        anchor_bone = str(anchor["bone"]) if anchor.get("bone") is not None else None
        anchor_bones = tuple(str(item) for item in anchor.get("bones", ()))
        if anchor_kind == "bone" and not anchor_bone:
            raise SensorPresentationLayoutError(f"{sensor_id}: bone anchor needs a bone")
        if anchor_kind == "spread_midpoint" and len(anchor_bones) != 2:
            raise SensorPresentationLayoutError(f"{sensor_id}: spread anchor needs two bones")
        offset = _as_float_triplet(
            anchor.get("local_offset", (0.0, 0.0, 0.0)),
            field="local_offset",
            sensor_id=sensor_id,
        )
        palm_blend = float(anchor.get("palm_blend", 0.0))
        if not 0.0 <= palm_blend <= 1.0:
            raise SensorPresentationLayoutError(f"{sensor_id}: palm_blend must be within [0, 1]")
        specs.append(
            SensorPresentationSpec(
                sensor_id=sensor_id,
                sensor_type=expected.sensor_type,
                marker=expected.display_marker,
                group=str(raw.get("group", "BEND")),
                display_name=str(raw.get("display_name", sensor_id)),
                short_id=str(raw.get("short_id", sensor_id)),
                description=expected.description,
                finger=expected.finger,
                joint=expected.joint,
                pair=expected.pair,
                array=expected.array,
                array_index=array_index,
                anchor_kind=anchor_kind,
                anchor_bone=anchor_bone,
                anchor_bones=anchor_bones,
                local_offset=offset,
                palm_blend=palm_blend,
            )
        )
    if len({spec.sensor_id for spec in specs}) != 20:
        raise SensorPresentationLayoutError("sensor IDs must be unique")
    return tuple(specs)


__all__ = [
    "DEFAULT_SENSOR_LAYOUT_PATH",
    "SensorPresentationLayoutError",
    "SensorPresentationSpec",
    "load_sensor_presentation_layout",
]
