"""Independent analytic 21-joint hand fixtures for TASK-005B.

This module is deliberately separate from the future production kinematics
package.  It creates simple, deterministic geometry and records the values
used to create that geometry as ground truth.  No production extractor is
imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COUNT = len(FINGER_NAMES)
JOINTS_PER_FINGER = 4
CHAIN_JOINT_COUNT = 3
TRACK_ORDER = ("LEFT", "RIGHT")

# The same index convention is used by the common 21-joint hand contract:
# wrist, then thumb CMC/MCP/IP/TIP, followed by index, middle, ring, pinky.
BENCHMARK_JOINT_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
FINGER_JOINTS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGER_BASE_JOINTS = tuple(chain[0] for chain in FINGER_JOINTS)

# LEFT is produced by mirroring x and reversing the palm normal.  The matrix
# is proper (determinant +1), so each side still has a right-handed local palm
# frame.  Applying it twice returns the original points exactly.
LEFT_MIRROR = np.diag([-1.0, 1.0, -1.0])
RIGHT_BASIS = np.eye(3, dtype=np.float64)
MIN_BONE_LENGTH = 1e-8
MIN_PALM_AREA = 1e-8


def _as_float_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _as_rotation_matrix(value: object) -> np.ndarray:
    matrix = _as_float_array(value, (3, 3), "global_rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError("global_rotation must be orthogonal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-8):
        raise ValueError("global_rotation must have determinant +1")
    return matrix


@dataclass(frozen=True)
class GeometryOptions:
    """Optional geometry perturbations used to generate invalid fixtures."""

    segment_lengths: tuple[float, float, float] = (0.36, 0.30, 0.24)
    palm_depth: float = 0.85
    coincident_mcp_points: bool = False
    zero_length_bone: tuple[int, int] | None = None
    invalid_joint: int | None = None
    invalid_value: float | None = None

    def __post_init__(self) -> None:
        lengths = tuple(float(value) for value in self.segment_lengths)
        if len(lengths) != 3 or any(not math.isfinite(value) or value < 0.0 for value in lengths):
            raise ValueError("segment_lengths must contain three finite non-negative values")
        object.__setattr__(self, "segment_lengths", lengths)
        if not math.isfinite(float(self.palm_depth)) or float(self.palm_depth) < 0.0:
            raise ValueError("palm_depth must be finite and non-negative")
        object.__setattr__(self, "palm_depth", float(self.palm_depth))
        if self.zero_length_bone is not None:
            if len(self.zero_length_bone) != 2:
                raise ValueError("zero_length_bone must be (finger_index, segment_index)")
            finger, segment = (int(value) for value in self.zero_length_bone)
            if finger not in range(FINGER_COUNT) or segment not in range(CHAIN_JOINT_COUNT):
                raise ValueError("zero_length_bone indices are out of range")
            object.__setattr__(self, "zero_length_bone", (finger, segment))
        if self.invalid_joint is not None:
            joint = int(self.invalid_joint)
            if joint not in range(len(BENCHMARK_JOINT_NAMES)):
                raise ValueError("invalid_joint is out of range")
            if self.invalid_value is None or math.isfinite(float(self.invalid_value)):
                raise ValueError("invalid_value must be NaN or Inf for an invalid-joint fixture")
            object.__setattr__(self, "invalid_joint", joint)


@dataclass(frozen=True)
class FrameParameters:
    """Parameters for one synthetic frame, independent of any extractor."""

    flexion_deg: object = field(default_factory=lambda: np.zeros((5, 3), dtype=np.float64))
    adjacent_spread_deg: object = (10.0, 10.0, 10.0, 10.0)
    global_rotation: object = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation: object = (0.0, 0.0, 0.0)
    scale: float = 1.0
    geometry: GeometryOptions = field(default_factory=GeometryOptions)

    def normalized(self) -> "FrameParameters":
        flexion = _as_float_array(self.flexion_deg, (FINGER_COUNT, CHAIN_JOINT_COUNT), "flexion_deg")
        spread = _as_float_array(self.adjacent_spread_deg, (4,), "adjacent_spread_deg")
        rotation = _as_rotation_matrix(self.global_rotation)
        translation = _as_float_array(self.translation, (3,), "translation")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        return FrameParameters(
            flexion_deg=flexion,
            adjacent_spread_deg=spread,
            global_rotation=rotation,
            translation=translation,
            scale=scale,
            geometry=self.geometry,
        )


@dataclass(frozen=True)
class SyntheticHand:
    """One generated hand and the analytic values used to create it."""

    joints: np.ndarray
    side: str
    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    base_heading_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    global_rotation: np.ndarray
    translation: np.ndarray
    scale: float

    @property
    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.joints)))


@dataclass(frozen=True)
class SyntheticSequence:
    """A sequence-shaped [F,2,...] fixture for the later adapter."""

    case_id: str
    joints: np.ndarray
    flexion_deg: np.ndarray
    adjacent_spread_deg: np.ndarray
    palm_rotation_matrix: np.ndarray
    palm_quaternion_wxyz: np.ndarray
    expected_valid: bool
    invalid_reasons: tuple[str, ...]
    sides: tuple[str, str] = TRACK_ORDER


@dataclass(frozen=True)
class BenchmarkCase:
    """Metadata plus parameters for one independent benchmark case."""

    case_id: str
    category: str
    description: str
    frames: tuple[FrameParameters, ...]
    expected_valid: bool = True
    mirror_equivalent: bool = False

    def generate(self) -> SyntheticSequence:
        sequence = generate_sequence(self.case_id, self.frames)
        if sequence.expected_valid != self.expected_valid:
            raise AssertionError(
                f"{self.case_id}: declared validity {self.expected_valid} "
                f"does not match generated validity {sequence.expected_valid}"
            )
        return sequence


def rotation_matrix_axis(axis: str, degrees: float) -> np.ndarray:
    """Return an exact right-handed rotation for a principal axis."""

    radians = math.radians(float(degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    axis = axis.upper()
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "Y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    if axis == "Z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"unknown rotation axis: {axis!r}")


def rotation_matrix_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Compose column-vector rotations as Rz @ Ry @ Rx."""

    return rotation_matrix_axis("Z", z_deg) @ rotation_matrix_axis("Y", y_deg) @ rotation_matrix_axis("X", x_deg)


def quaternion_wxyz_from_matrix(matrix: object) -> np.ndarray:
    """Convert a proper rotation matrix to canonical-sign [w,x,y,z]."""

    rotation = _as_rotation_matrix(matrix)
    trace = float(np.trace(rotation))
    quaternion = np.empty(4, dtype=np.float64)
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        quaternion[0] = 0.5 * root
        root = 0.5 / root
        quaternion[1] = (rotation[2, 1] - rotation[1, 2]) * root
        quaternion[2] = (rotation[0, 2] - rotation[2, 0]) * root
        quaternion[3] = (rotation[1, 0] - rotation[0, 1]) * root
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            quaternion[1] = 0.5 * root
            root = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (rotation[2, 1] - rotation[1, 2]) * root
            quaternion[2] = (rotation[0, 1] + rotation[1, 0]) * root
            quaternion[3] = (rotation[0, 2] + rotation[2, 0]) * root
        elif index == 1:
            root = math.sqrt(max(0.0, 1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]))
            quaternion[2] = 0.5 * root
            root = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (rotation[0, 2] - rotation[2, 0]) * root
            quaternion[1] = (rotation[0, 1] + rotation[1, 0]) * root
            quaternion[3] = (rotation[1, 2] + rotation[2, 1]) * root
        else:
            root = math.sqrt(max(0.0, 1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]))
            quaternion[3] = 0.5 * root
            root = 0.5 / root if root > 0.0 else 0.0
            quaternion[0] = (rotation[1, 0] - rotation[0, 1]) * root
            quaternion[1] = (rotation[0, 2] + rotation[2, 0]) * root
            quaternion[2] = (rotation[1, 2] + rotation[2, 1]) * root
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_matrix_wxyz(quaternion: object) -> np.ndarray:
    """Convert [w,x,y,z] to a rotation matrix."""

    values = _as_float_array(quaternion, (4,), "quaternion_wxyz")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("quaternion cannot have zero norm")
    w, x, y, z = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def mirror_points(points: object) -> np.ndarray:
    """Mirror points using the locked LEFT/RIGHT convention."""

    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("points must have a final dimension of 3")
    return values @ LEFT_MIRROR.T


def _canonical_points(parameters: FrameParameters) -> tuple[np.ndarray, np.ndarray]:
    flexion = _as_float_array(parameters.flexion_deg, (FINGER_COUNT, CHAIN_JOINT_COUNT), "flexion_deg")
    spread = _as_float_array(parameters.adjacent_spread_deg, (4,), "adjacent_spread_deg")
    options = parameters.geometry
    points = np.zeros((len(BENCHMARK_JOINT_NAMES), 3), dtype=np.float64)
    points[0] = (0.0, -options.palm_depth, 0.0)
    base_x = np.array([-0.60, -0.30, 0.0, 0.30, 0.60], dtype=np.float64)
    bases = np.column_stack((base_x, np.zeros(5), np.zeros(5)))
    if options.coincident_mcp_points:
        bases[2] = bases[1]
    headings = np.empty(FINGER_COUNT, dtype=np.float64)
    headings[0] = -0.5 * float(np.sum(spread))
    for finger in range(1, FINGER_COUNT):
        headings[finger] = headings[finger - 1] + spread[finger - 1]

    for finger, chain in enumerate(FINGER_JOINTS):
        points[chain[0]] = bases[finger]
        straight = np.array([math.sin(math.radians(headings[finger])), math.cos(math.radians(headings[finger])), 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        cursor = points[chain[0]].copy()
        cumulative_bend = 0.0
        for segment in range(CHAIN_JOINT_COUNT):
            cumulative_bend += flexion[finger, segment]
            radians = math.radians(cumulative_bend)
            direction = math.cos(radians) * straight + math.sin(radians) * normal
            length = options.segment_lengths[segment]
            if options.zero_length_bone == (finger, segment):
                length = 0.0
            cursor = cursor + length * direction
            points[chain[segment + 1]] = cursor
    return points, headings


def _side_basis(side: str) -> np.ndarray:
    normalized = side.upper()
    if normalized == "RIGHT":
        return RIGHT_BASIS.copy()
    if normalized == "LEFT":
        return LEFT_MIRROR.copy()
    raise ValueError("side must be LEFT or RIGHT")


def generate_hand(
    *,
    side: str = "RIGHT",
    flexion_deg: object | None = None,
    adjacent_spread_deg: object = (10.0, 10.0, 10.0, 10.0),
    global_rotation: object | None = None,
    translation: object = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    geometry: GeometryOptions | None = None,
) -> SyntheticHand:
    """Generate one deterministic hand from analytic parameters."""

    parameters = FrameParameters(
        flexion_deg=np.zeros((5, 3), dtype=np.float64) if flexion_deg is None else flexion_deg,
        adjacent_spread_deg=adjacent_spread_deg,
        global_rotation=np.eye(3, dtype=np.float64) if global_rotation is None else global_rotation,
        translation=translation,
        scale=scale,
        geometry=GeometryOptions() if geometry is None else geometry,
    ).normalized()
    canonical, headings = _canonical_points(parameters)
    basis = _side_basis(side)
    rotation = _as_rotation_matrix(parameters.global_rotation)
    translation_array = _as_float_array(parameters.translation, (3,), "translation")
    local = canonical @ basis.T
    world = (parameters.scale * local) @ rotation.T + translation_array
    if parameters.geometry.invalid_joint is not None:
        world[parameters.geometry.invalid_joint, :] = float(parameters.geometry.invalid_value)
    palm_rotation = rotation @ basis
    return SyntheticHand(
        joints=world,
        side=side.upper(),
        flexion_deg=np.asarray(parameters.flexion_deg, dtype=np.float64).copy(),
        adjacent_spread_deg=np.abs(np.asarray(parameters.adjacent_spread_deg, dtype=np.float64)),
        base_heading_deg=headings,
        palm_rotation_matrix=palm_rotation,
        palm_quaternion_wxyz=quaternion_wxyz_from_matrix(palm_rotation),
        global_rotation=rotation,
        translation=translation_array,
        scale=parameters.scale,
    )


def _bone_lengths(joints: np.ndarray) -> np.ndarray:
    lengths: list[float] = []
    for chain in FINGER_JOINTS:
        for first, second in zip(chain, chain[1:]):
            lengths.append(float(np.linalg.norm(joints[second] - joints[first])))
    return np.asarray(lengths, dtype=np.float64)


def geometry_validity(hand: SyntheticHand) -> tuple[bool, str]:
    """Return whether geometry is numerically suitable for local angles."""

    if not hand.finite:
        return False, "non_finite_joint"
    bone_lengths = _bone_lengths(hand.joints)
    if np.any(bone_lengths <= MIN_BONE_LENGTH):
        return False, "zero_or_tiny_bone"
    palm_a = hand.joints[5] - hand.joints[0]
    palm_b = hand.joints[9] - hand.joints[0]
    if float(np.linalg.norm(np.cross(palm_a, palm_b))) <= MIN_PALM_AREA:
        return False, "collinear_or_coincident_palm"
    return True, "valid"


def generate_sequence(case_id: str, frames: Sequence[FrameParameters]) -> SyntheticSequence:
    """Generate a [F,2,21,3] LEFT/RIGHT sequence and its analytic truth."""

    if not frames:
        raise ValueError("a sequence must contain at least one frame")
    hands_by_frame: list[tuple[SyntheticHand, SyntheticHand]] = []
    reasons: list[str] = []
    for parameters in frames:
        left = generate_hand(side="LEFT", **_parameters_kwargs(parameters))
        right = generate_hand(side="RIGHT", **_parameters_kwargs(parameters))
        hands_by_frame.append((left, right))
        for hand in (left, right):
            valid, reason = geometry_validity(hand)
            if not valid:
                reasons.append(f"{hand.side}:{reason}")
    joints = np.asarray([[hand.joints for hand in pair] for pair in hands_by_frame], dtype=np.float64)
    flexion = np.asarray([[hand.flexion_deg for hand in pair] for pair in hands_by_frame], dtype=np.float64)
    spread = np.asarray([[hand.adjacent_spread_deg for hand in pair] for pair in hands_by_frame], dtype=np.float64)
    rotations = np.asarray([[hand.palm_rotation_matrix for hand in pair] for pair in hands_by_frame], dtype=np.float64)
    quaternions = np.asarray([[hand.palm_quaternion_wxyz for hand in pair] for pair in hands_by_frame], dtype=np.float64)
    return SyntheticSequence(
        case_id=case_id,
        joints=joints,
        flexion_deg=flexion,
        adjacent_spread_deg=spread,
        palm_rotation_matrix=rotations,
        palm_quaternion_wxyz=quaternions,
        expected_valid=not reasons,
        invalid_reasons=tuple(sorted(set(reasons))),
    )


def _parameters_kwargs(parameters: FrameParameters) -> dict[str, object]:
    return {
        "flexion_deg": parameters.flexion_deg,
        "adjacent_spread_deg": parameters.adjacent_spread_deg,
        "global_rotation": parameters.global_rotation,
        "translation": parameters.translation,
        "scale": parameters.scale,
        "geometry": parameters.geometry,
    }


def frame_parameters_to_dict(parameters: FrameParameters) -> dict[str, object]:
    """Serialize a fixture descriptor without serializing bulky joint arrays."""

    normalized = parameters.normalized()
    geometry = normalized.geometry
    return {
        "flexion_deg": np.asarray(normalized.flexion_deg).tolist(),
        "adjacent_spread_deg": np.asarray(normalized.adjacent_spread_deg).tolist(),
        "global_rotation": np.asarray(normalized.global_rotation).tolist(),
        "translation": np.asarray(normalized.translation).tolist(),
        "scale": normalized.scale,
        "geometry": {
            "segment_lengths": list(geometry.segment_lengths),
            "palm_depth": geometry.palm_depth,
            "coincident_mcp_points": geometry.coincident_mcp_points,
            "zero_length_bone": list(geometry.zero_length_bone) if geometry.zero_length_bone else None,
            "invalid_joint": geometry.invalid_joint,
            "invalid_value": geometry.invalid_value,
        },
    }


def frame_parameters_from_dict(payload: dict[str, object]) -> FrameParameters:
    """Reconstruct a descriptor written by :func:`frame_parameters_to_dict`."""

    geometry_payload = dict(payload.get("geometry", {}))
    zero_length = geometry_payload.get("zero_length_bone")
    geometry = GeometryOptions(
        segment_lengths=tuple(geometry_payload.get("segment_lengths", (0.36, 0.30, 0.24))),
        palm_depth=float(geometry_payload.get("palm_depth", 0.85)),
        coincident_mcp_points=bool(geometry_payload.get("coincident_mcp_points", False)),
        zero_length_bone=tuple(zero_length) if zero_length is not None else None,
        invalid_joint=geometry_payload.get("invalid_joint"),
        invalid_value=geometry_payload.get("invalid_value"),
    )
    return FrameParameters(
        flexion_deg=payload["flexion_deg"],
        adjacent_spread_deg=payload["adjacent_spread_deg"],
        global_rotation=payload["global_rotation"],
        translation=payload["translation"],
        scale=float(payload["scale"]),
        geometry=geometry,
    ).normalized()


def _zero_flexion() -> np.ndarray:
    return np.zeros((FINGER_COUNT, CHAIN_JOINT_COUNT), dtype=np.float64)


def _one_bend(finger: int, joint: int, degrees: float) -> np.ndarray:
    values = _zero_flexion()
    values[finger, joint] = degrees
    return values


def _set_finger(values: np.ndarray, finger: int, bends: Sequence[float]) -> np.ndarray:
    result = values.copy()
    result[finger, :] = np.asarray(bends, dtype=np.float64)
    return result


def _case(
    case_id: str,
    category: str,
    description: str,
    *,
    flexion: object | None = None,
    spread: object = (10.0, 10.0, 10.0, 10.0),
    rotation: object | None = None,
    translation: object = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    geometry: GeometryOptions | None = None,
    expected_valid: bool = True,
    mirror_equivalent: bool = False,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        category=category,
        description=description,
        frames=(
            FrameParameters(
                flexion_deg=_zero_flexion() if flexion is None else flexion,
                adjacent_spread_deg=spread,
                global_rotation=np.eye(3) if rotation is None else rotation,
                translation=translation,
                scale=scale,
                geometry=GeometryOptions() if geometry is None else geometry,
            ),
        ),
        expected_valid=expected_valid,
        mirror_equivalent=mirror_equivalent,
    )


def build_benchmark_catalog() -> tuple[BenchmarkCase, ...]:
    """Build the complete locked catalog before any production results exist."""

    cases: list[BenchmarkCase] = []
    cases.append(_case("neutral", "neutral", "All 15 chain bends are zero."))
    for finger, finger_name in enumerate(FINGER_NAMES):
        for joint in range(CHAIN_JOINT_COUNT):
            for degrees in (30.0, 60.0, 90.0):
                cases.append(
                    _case(
                        f"single_{finger_name}_joint{joint}_{int(degrees)}deg",
                        "single_bend",
                        f"Only {finger_name} chain joint {joint} is bent to {degrees:g} degrees.",
                        flexion=_one_bend(finger, joint, degrees),
                    )
                )

    for finger, bends in ((1, (30.0, 45.0, 60.0)), (3, (60.0, 75.0, 90.0)), (4, (90.0, 90.0, 90.0))):
        cases.append(
            _case(
                f"multi_curl_{FINGER_NAMES[finger]}",
                "multi_joint_curl",
                f"{FINGER_NAMES[finger]} uses the multi-joint curl {list(bends)}.",
                flexion=_set_finger(_zero_flexion(), finger, bends),
            )
        )
    index_only = _set_finger(_zero_flexion(), 1, (60.0, 60.0, 60.0))
    cases.append(_case("independent_index_only", "independent_fingers", "Index bent; all other fingers straight.", flexion=index_only))
    ring_pinky = _set_finger(_set_finger(_zero_flexion(), 3, (45.0, 60.0, 75.0)), 4, (45.0, 60.0, 75.0))
    cases.append(_case("independent_ring_pinky", "independent_fingers", "Ring and pinky bent; all other fingers straight.", flexion=ring_pinky))

    for spread in (5.0, 10.0, 20.0, 30.0):
        cases.append(
            _case(
                f"spread_{int(spread)}deg",
                "spread",
                f"Each of the four adjacent base-ray gaps is {spread:g} degrees.",
                spread=(spread,) * 4,
            )
        )

    mirror_parameters = (
        ("mirror_neutral", _zero_flexion(), (10.0,) * 4, np.eye(3)),
        ("mirror_curl", _set_finger(_zero_flexion(), 1, (35.0, 55.0, 75.0)), (20.0,) * 4, np.eye(3)),
        ("mirror_rotated_curl", _set_finger(_zero_flexion(), 0, (45.0, 30.0, 60.0)), (15.0,) * 4, rotation_matrix_xyz(25.0, -40.0, 70.0)),
    )
    for case_id, flexion, spread, rotation in mirror_parameters:
        cases.append(
            _case(
                case_id,
                "mirror",
                "Paired LEFT/RIGHT generation must preserve local flexion and spread.",
                flexion=flexion,
                spread=spread,
                rotation=rotation,
                mirror_equivalent=True,
            )
        )

    translations = ((12.5, -7.25, 3.0), (-100.0, 40.0, 250.0), (0.001, 0.002, -0.003))
    for index, translation in enumerate(translations, start=1):
        cases.append(_case(f"translation_{index}", "translation", "Rigid translation must not change local values.", translation=translation))
    for scale in (0.5, 1.0, 2.0, 5.0):
        cases.append(_case(f"scale_{str(scale).replace('.', '_')}x", "scale", "Uniform scale must not change local values.", scale=scale))

    rotations = (
        ("identity", np.eye(3)),
        ("x90", rotation_matrix_axis("X", 90.0)),
        ("y90", rotation_matrix_axis("Y", 90.0)),
        ("z90", rotation_matrix_axis("Z", 90.0)),
        ("x180", rotation_matrix_axis("X", 180.0)),
        ("y180", rotation_matrix_axis("Y", 180.0)),
        ("z180", rotation_matrix_axis("Z", 180.0)),
        ("composed", rotation_matrix_xyz(35.0, -20.0, 110.0)),
    )
    for name, rotation in rotations:
        cases.append(
            _case(
                f"orientation_{name}",
                "quaternion_orientation",
                "Known global rotation with matrix and [w,x,y,z] quaternion truth.",
                flexion=_set_finger(_zero_flexion(), 1, (20.0, 40.0, 60.0)),
                rotation=rotation,
            )
        )

    cases.extend(
        (
            _case(
                "degenerate_zero_length_bone",
                "degenerate",
                "Index PIP-to-DIP bone has exactly zero length.",
                geometry=GeometryOptions(zero_length_bone=(1, 1)),
                expected_valid=False,
            ),
            _case(
                "degenerate_coincident_mcp",
                "degenerate",
                "Index and middle MCP points coincide.",
                geometry=GeometryOptions(coincident_mcp_points=True),
                expected_valid=False,
            ),
            _case(
                "degenerate_collinear_palm",
                "degenerate",
                "Wrist and MCP points are collinear in the palm plane.",
                geometry=GeometryOptions(palm_depth=0.0),
                expected_valid=False,
            ),
            _case(
                "degenerate_nan_joint",
                "degenerate",
                "One joint is explicitly NaN after generation.",
                geometry=GeometryOptions(invalid_joint=10, invalid_value=math.nan),
                expected_valid=False,
            ),
            _case(
                "degenerate_inf_joint",
                "degenerate",
                "One joint is explicitly +Inf after generation.",
                geometry=GeometryOptions(invalid_joint=10, invalid_value=math.inf),
                expected_valid=False,
            ),
        )
    )

    cases.extend(
        (
            _case(
                "adversarial_almost_straight",
                "adversarial",
                "Index MCP bend is 0.1 degrees, not an exact zero.",
                flexion=_one_bend(1, 0, 0.1),
            ),
            _case(
                "adversarial_near_180",
                "adversarial",
                "Index MCP bend is 179.9 degrees.",
                flexion=_one_bend(1, 0, 179.9),
            ),
            _case(
                "adversarial_tiny_nonzero_bone",
                "adversarial",
                "One bone is nonzero but below the numerical validity floor.",
                geometry=GeometryOptions(segment_lengths=(1e-10, 0.30, 0.24)),
                expected_valid=False,
            ),
            _case(
                "adversarial_mirrored_composed_rotation",
                "adversarial",
                "Mirrored paired pose after an arbitrary composed global rotation.",
                flexion=_set_finger(_zero_flexion(), 2, (70.0, 80.0, 20.0)),
                spread=(-15.0, 35.0, 10.0, 25.0),
                rotation=rotation_matrix_xyz(-65.0, 33.0, 147.0),
                mirror_equivalent=True,
            ),
            _case(
                "adversarial_facing_away",
                "adversarial",
                "Palm normal is rotated away by a 180-degree X rotation.",
                rotation=rotation_matrix_axis("X", 180.0),
            ),
            _case(
                "adversarial_palm_upside_down",
                "adversarial",
                "Palm is upside down after a 180-degree Y rotation.",
                rotation=rotation_matrix_axis("Y", 180.0),
            ),
            _case(
                "adversarial_finger_crossing",
                "adversarial",
                "Signed adjacent base headings force a crossing-like arrangement.",
                spread=(70.0, -140.0, 70.0, 70.0),
            ),
            _case(
                "adversarial_thumb_opposition",
                "adversarial",
                "Thumb opposition-like spread and curl without anatomical assumptions.",
                flexion=_set_finger(_zero_flexion(), 0, (60.0, 45.0, 30.0)),
                spread=(-35.0, 20.0, 20.0, 20.0),
            ),
        )
    )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("benchmark case IDs must be unique")
    return tuple(cases)
