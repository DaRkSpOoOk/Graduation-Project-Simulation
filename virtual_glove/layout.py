"""Ideal virtual smart-glove sensor layout, V1.

One hand carries 15 bend Hall sensors, 4 adjacent-finger spread Hall sensors
and 1 palm IMU package: 19 Hall-type channels plus 1 IMU, so 20 logical
sensing packages per hand.

This layout is deliberately machine-readable and richer than the old physical
five-Hall prototype. That prototype is not an architectural constraint here;
the ideal glove is designed first and later work may ablate it downwards.

Every entry carries enough information for a future visualization task to draw
where each sensor sits, plus the exact array slot it maps to, so a consumer can
go from ``sensor_id`` to a position in the output arrays without guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LAYOUT_VERSION = "ideal_virtual_glove_v1"

# Inherited verbatim from the frozen TASK-005 contract. Never reordered.
TRACK_ORDER: tuple[str, str] = ("LEFT", "RIGHT")
FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
CHAIN_ORDER: tuple[str, str, str] = ("proximal", "middle", "distal")
SPREAD_PAIRS: tuple[tuple[str, str], ...] = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)

SENSOR_TYPE_BEND = "hall_bend_angular"
SENSOR_TYPE_SPREAD = "hall_spread_angular"
SENSOR_TYPE_IMU = "imu_package"

MARKER_HALL = "H"
MARKER_IMU = "IMU"

EXPECTED_BEND_SENSORS = 15
EXPECTED_SPREAD_SENSORS = 4
EXPECTED_HALL_SENSORS = EXPECTED_BEND_SENSORS + EXPECTED_SPREAD_SENSORS  # 19
EXPECTED_IMU_SENSORS = 1
EXPECTED_SENSOR_PACKAGES = EXPECTED_HALL_SENSORS + EXPECTED_IMU_SENSORS  # 20

# Anatomical site each bend sensor straddles. The chain names stay generic for
# the thumb: TASK-005 roots the thumb chain at the wrist, so its three angles
# sit approximately at CMC/MCP/IP, and that mapping is not asserted as exact.
_BEND_SITE = {
    ("thumb", "proximal"): ("dorsal_thumb_carpometacarpal", "approximately the thumb CMC"),
    ("thumb", "middle"): ("dorsal_thumb_metacarpophalangeal", "approximately the thumb MCP"),
    ("thumb", "distal"): ("dorsal_thumb_interphalangeal", "approximately the thumb IP"),
}
_FINGER_SITE = {
    "proximal": ("metacarpophalangeal", "MCP"),
    "middle": ("proximal_interphalangeal", "PIP"),
    "distal": ("distal_interphalangeal", "DIP"),
}


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """One virtual sensing package."""

    sensor_id: str
    sensor_type: str
    finger: str | None
    pair: tuple[str, str] | None
    joint: str | None
    role: str
    logical_location: str
    display_marker: str
    description: str
    # Where this sensor's value lives in the output arrays.
    array: str
    array_index: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair"] = list(self.pair) if self.pair is not None else None
        payload["array_index"] = list(self.array_index)
        return payload


def _bend_sensors() -> list[SensorSpec]:
    sensors: list[SensorSpec] = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        for chain_index, chain in enumerate(CHAIN_ORDER):
            if finger == "thumb":
                location, anatomy = _BEND_SITE[(finger, chain)]
            else:
                suffix, short = _FINGER_SITE[chain]
                location = f"dorsal_{finger}_{suffix}"
                anatomy = f"the {finger} {short}"
            sensors.append(
                SensorSpec(
                    sensor_id=f"H_{finger.upper()}_{chain.upper()}",
                    sensor_type=SENSOR_TYPE_BEND,
                    finger=finger,
                    pair=None,
                    joint=chain,
                    role="bend",
                    logical_location=location,
                    display_marker=MARKER_HALL,
                    description=(
                        f"Hall/magnetic angular bend sensor spanning {anatomy}, "
                        f"reading the TASK-005 unsigned inter-bone angle for the "
                        f"{finger} {chain} chain joint."
                    ),
                    array="bend_angle_deg",
                    array_index=(finger_index, chain_index),
                )
            )
    return sensors


def _spread_sensors() -> list[SensorSpec]:
    sensors: list[SensorSpec] = []
    for pair_index, (first, second) in enumerate(SPREAD_PAIRS):
        sensors.append(
            SensorSpec(
                sensor_id=f"H_SPREAD_{first.upper()}_{second.upper()}",
                sensor_type=SENSOR_TYPE_SPREAD,
                finger=None,
                pair=(first, second),
                joint=None,
                role="spread",
                logical_location=f"interdigital_web_{first}_{second}",
                display_marker=MARKER_HALL,
                description=(
                    f"Hall/magnetic angular spread sensor in the web between the "
                    f"{first} and {second} fingers, reading the TASK-005 unsigned "
                    f"palm-plane angle between their proximal directions."
                ),
                array="spread_angle_deg",
                array_index=(pair_index,),
            )
        )
    return sensors


def _imu_sensor() -> SensorSpec:
    return SensorSpec(
        sensor_id="IMU_PALM",
        sensor_type=SENSOR_TYPE_IMU,
        finger=None,
        pair=None,
        joint=None,
        role="palm_orientation",
        logical_location="dorsal_palm_centre",
        display_marker=MARKER_IMU,
        description=(
            "Palm IMU package on the dorsum of the hand. Its authoritative output "
            "is the TASK-005 palm orientation, copied without any convention "
            "change; angular velocity is derived from consecutive valid "
            "orientations. No accelerometer channel is emitted (see the report)."
        ),
        array="imu_quaternion_wxyz",
        array_index=(),
    )


SENSOR_LAYOUT: tuple[SensorSpec, ...] = tuple(
    _bend_sensors() + _spread_sensors() + [_imu_sensor()]
)

BEND_SENSOR_IDS: tuple[str, ...] = tuple(
    s.sensor_id for s in SENSOR_LAYOUT if s.sensor_type == SENSOR_TYPE_BEND
)
SPREAD_SENSOR_IDS: tuple[str, ...] = tuple(
    s.sensor_id for s in SENSOR_LAYOUT if s.sensor_type == SENSOR_TYPE_SPREAD
)
HALL_SENSOR_IDS: tuple[str, ...] = BEND_SENSOR_IDS + SPREAD_SENSOR_IDS
IMU_SENSOR_IDS: tuple[str, ...] = tuple(
    s.sensor_id for s in SENSOR_LAYOUT if s.sensor_type == SENSOR_TYPE_IMU
)


def sensor_by_id(sensor_id: str) -> SensorSpec:
    for sensor in SENSOR_LAYOUT:
        if sensor.sensor_id == sensor_id:
            return sensor
    raise KeyError(f"unknown sensor_id: {sensor_id}")


def layout_document() -> dict[str, Any]:
    """The machine-readable layout written to ``sensor_layout.json``."""

    return {
        "layout_version": LAYOUT_VERSION,
        "task": "TASK-006A",
        "per_hand_counts": {
            "bend_hall_sensors": EXPECTED_BEND_SENSORS,
            "spread_hall_sensors": EXPECTED_SPREAD_SENSORS,
            "hall_sensors_total": EXPECTED_HALL_SENSORS,
            "imu_packages": EXPECTED_IMU_SENSORS,
            "logical_sensing_packages": EXPECTED_SENSOR_PACKAGES,
        },
        "track_order": list(TRACK_ORDER),
        "finger_order": list(FINGER_ORDER),
        "chain_joint_order": list(CHAIN_ORDER),
        "spread_pairs": [list(pair) for pair in SPREAD_PAIRS],
        "display_markers": {"hall": MARKER_HALL, "imu": MARKER_IMU},
        "notes": {
            "independence": (
                "The 15 bend channels are kept independent. Three joints of one "
                "finger are never aggregated into a single finger value."
            ),
            "thumb_chain": (
                "Thumb chain names stay generic. TASK-005 roots the thumb chain "
                "at the wrist, so the three angles sit approximately at CMC/MCP/IP; "
                "that mapping is not asserted as anatomically exact."
            ),
            "visualization": (
                "logical_location and display_marker exist so a later "
                "visualization task can show where every Hall sensor sits. No "
                "visualizer is built in TASK-006A."
            ),
            "ablation": (
                "This 19-Hall layout is intentionally information-rich. Later "
                "work may ablate 19 -> 15 -> 10 -> 5 to find how many sensors "
                "physical hardware actually needs. No ablation is done here."
            ),
        },
        "sensors": [sensor.to_dict() for sensor in SENSOR_LAYOUT],
    }
