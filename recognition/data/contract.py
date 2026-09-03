"""TASK-009A frozen sequence-input contract.

This module is the single source of truth for what every tensor slot means.
Nothing here is inferred at runtime from directory ordering, file ordering or
array shapes: the orders below are copied from the frozen TASK-006 sensor layout
and are asserted against the production ``sensor_layout.json`` when a dataset is
constructed.

Polarity rule for every mask in this contract: **True means present/valid/real.**
There is no inverted twin of any mask, because a contract that ships both
polarities eventually gets one of them backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

CONTRACT_VERSION = "task009a_sequence_input_v1"

# --- Frozen orders (TASK-006 ideal_virtual_glove_v1) -------------------------
# Physical hand identity. Never reordered by image position, confidence,
# availability or channel count: LEFT is always slot 0 and RIGHT always slot 1.
HAND_ORDER: tuple[str, str] = ("LEFT", "RIGHT")
FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
CHAIN_ORDER: tuple[str, ...] = ("proximal", "middle", "distal")
SPREAD_PAIRS: tuple[tuple[str, str], ...] = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)
QUATERNION_ORDER: tuple[str, ...] = ("w", "x", "y", "z")

BEND_CHANNELS_PER_HAND = len(FINGER_ORDER) * len(CHAIN_ORDER)  # 15
SPREAD_CHANNELS_PER_HAND = len(SPREAD_PAIRS)  # 4
QUATERNION_CHANNELS_PER_HAND = len(QUATERNION_ORDER)  # 4

# Upstream schema versions this contract is pinned to.
SOURCE_SCHEMA_VERSION = "virtual_glove_v1"
SOURCE_LAYOUT_VERSION = "ideal_virtual_glove_v1"
SOURCE_CONTRACTS = ("TASK-005-final-v2", "TASK-006-ideal-virtual-glove-v1")

# Frozen TASK-004 tracking codes. Kept complete on purpose: the production
# corpus only ever produced OBSERVED and MISSING, but the schema must stay
# compatible with the frozen tracking contract if future data differs.
TRACKING_STATE_NAMES: dict[int, str] = {
    0: "MISSING",
    1: "OBSERVED",
    2: "AMBIGUOUS",
    3: "REJECTED_QUALITY",
    4: "LIKELY_OCCLUDED",
}
# A hand counts as physically present only in a pose-bearing state.
POSE_BEARING_CODES: tuple[int, ...] = (1, 2)

NUM_CLASSES = 28
LABEL_INDEX_RANGE = (0, NUM_CLASSES - 1)

FeatureSet = Literal["bend_only", "bend_spread", "full"]
QuaternionPolicy = Literal["absolute", "relative_first_valid"]

FEATURE_SETS: tuple[str, ...] = ("bend_only", "bend_spread", "full")
QUATERNION_POLICIES: tuple[str, ...] = ("absolute", "relative_first_valid")

# Which feature families each set includes, in this fixed order.
_FAMILIES_BY_SET: dict[str, tuple[str, ...]] = {
    "bend_only": ("bend",),
    "bend_spread": ("bend", "spread"),
    "full": ("bend", "spread", "quaternion"),
}
_FAMILY_WIDTH = {
    "bend": BEND_CHANNELS_PER_HAND,
    "spread": SPREAD_CHANNELS_PER_HAND,
    "quaternion": QUATERNION_CHANNELS_PER_HAND,
}


def families_for(feature_set: str) -> tuple[str, ...]:
    if feature_set not in _FAMILIES_BY_SET:
        raise ValueError(f"unknown feature_set {feature_set!r}; expected one of {FEATURE_SETS}")
    return _FAMILIES_BY_SET[feature_set]


def channels_per_hand(feature_set: str) -> int:
    return sum(_FAMILY_WIDTH[name] for name in families_for(feature_set))


def feature_dimension(feature_set: str) -> int:
    """Total channel count across both hands."""

    return len(HAND_ORDER) * channels_per_hand(feature_set)


def channel_names(feature_set: str) -> list[str]:
    """The name of every channel, in tensor order.

    ``channel_names(fs)[i]`` answers "``X[..., i]`` means what?" for every i,
    which is the whole point of freezing this contract.
    """

    names: list[str] = []
    for hand in HAND_ORDER:
        for family in families_for(feature_set):
            if family == "bend":
                for finger in FINGER_ORDER:
                    for joint in CHAIN_ORDER:
                        names.append(f"{hand}/bend/{finger}/{joint}")
            elif family == "spread":
                for first, second in SPREAD_PAIRS:
                    names.append(f"{hand}/spread/{first}-{second}")
            else:
                for component in QUATERNION_ORDER:
                    names.append(f"{hand}/palm_quaternion/{component}")
    return names


def channel_index(feature_set: str, name: str) -> int:
    """Index of a named channel; raises if the set does not carry it."""

    try:
        return channel_names(feature_set).index(name)
    except ValueError as error:
        raise KeyError(f"{name!r} is not a channel of feature_set {feature_set!r}") from error


def hand_slice(feature_set: str, hand: str) -> slice:
    """The contiguous channel block belonging to one physical hand."""

    if hand not in HAND_ORDER:
        raise ValueError(f"unknown hand {hand!r}; expected one of {HAND_ORDER}")
    width = channels_per_hand(feature_set)
    start = HAND_ORDER.index(hand) * width
    return slice(start, start + width)


def family_slice(feature_set: str, hand: str, family: str) -> slice:
    """The contiguous channel block of one feature family within one hand."""

    families = families_for(feature_set)
    if family not in families:
        raise KeyError(f"feature_set {feature_set!r} does not carry family {family!r}")
    offset = hand_slice(feature_set, hand).start
    for name in families:
        if name == family:
            return slice(offset, offset + _FAMILY_WIDTH[name])
        offset += _FAMILY_WIDTH[name]
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class SequenceInputConfig:
    """Everything TASK-009B needs to know to interpret a batch.

    Defaults are the frozen TASK-009A primary contract. Every field that changes
    tensor semantics is recorded in :meth:`to_dict`, so an experiment can never
    silently disagree with the report about what it trained on.
    """

    feature_set: FeatureSet = "full"
    quaternion_policy: QuaternionPolicy = "absolute"
    # TASK-008 already emits physically normalized values (deg / 180). The
    # primary contract applies no further transform, so there is nothing to fit
    # and therefore nothing to leak across a LOSO boundary.
    normalization: Literal["task008_physical"] = "task008_physical"
    # Mechanical fill for tensor slots with no measurement. Always paired with a
    # False entry in feature_valid, which is what makes it distinguishable from
    # a real 0.0 reading.
    invalid_fill_value: float = 0.0
    padding_fill_value: float = 0.0
    include_hand_present: bool = True
    include_tracking_state: bool = True  # metadata only, never a model feature
    # "first" checks the layout of the first sample at construction; "all" checks
    # every sample (used by tests and the full audit); "none" skips the check.
    verify_layout: Literal["first", "all", "none"] = "first"
    preload: bool = False

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if self.quaternion_policy not in QUATERNION_POLICIES:
            raise ValueError(f"quaternion_policy must be one of {QUATERNION_POLICIES}")
        if self.verify_layout not in ("first", "all", "none"):
            raise ValueError("verify_layout must be 'first', 'all' or 'none'")
        if self.quaternion_policy != "absolute" and "quaternion" not in families_for(self.feature_set):
            raise ValueError(
                f"quaternion_policy={self.quaternion_policy!r} is meaningless for "
                f"feature_set={self.feature_set!r}, which carries no quaternion channels"
            )

    @property
    def feature_dim(self) -> int:
        return feature_dimension(self.feature_set)

    @property
    def channels_per_hand(self) -> int:
        return channels_per_hand(self.feature_set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "feature_set": self.feature_set,
            "quaternion_policy": self.quaternion_policy,
            "normalization": self.normalization,
            "invalid_fill_value": self.invalid_fill_value,
            "padding_fill_value": self.padding_fill_value,
            "include_hand_present": self.include_hand_present,
            "include_tracking_state": self.include_tracking_state,
            "feature_dim": self.feature_dim,
            "channels_per_hand": self.channels_per_hand,
        }


def contract_document(config: SequenceInputConfig | None = None) -> dict[str, Any]:
    """The machine-readable contract TASK-009B consumes instead of re-deciding."""

    config = config or SequenceInputConfig()
    return {
        "contract_version": CONTRACT_VERSION,
        "task": "TASK-009A",
        "purpose": (
            "freeze the transformation from the finalized TASK-008 virtual-glove "
            "dataset to variable-length, explicitly masked, signer-independent "
            "PyTorch batches"
        ),
        "source": {
            "index": "datasets/manifests/karsl_core28_virtual_glove.csv",
            "splits": "datasets/splits/karsl_core28_loso_s{01,02,03}.csv",
            "npz_schema_version": SOURCE_SCHEMA_VERSION,
            "sensor_layout_version": SOURCE_LAYOUT_VERSION,
            "upstream_contracts": list(SOURCE_CONTRACTS),
        },
        "hand_order": list(HAND_ORDER),
        "feature_order": {
            "finger_order": list(FINGER_ORDER),
            "chain_joint_order": list(CHAIN_ORDER),
            "spread_pairs": [list(pair) for pair in SPREAD_PAIRS],
            "quaternion_order": list(QUATERNION_ORDER),
            "layout": "hand-major, then family (bend, spread, quaternion), then the orders above",
        },
        "feature_sets": {
            name: {
                "families": list(families_for(name)),
                "channels_per_hand": channels_per_hand(name),
                "feature_dim": feature_dimension(name),
                "channel_names": channel_names(name),
            }
            for name in FEATURE_SETS
        },
        "masks": {
            "polarity": "True means present/valid/real for every mask in this contract",
            "feature_valid": {
                "shape": "[B, T_max, D]",
                "dtype": "bool",
                "meaning": "this exact channel carries a real measurement at this timestep",
            },
            "hand_present": {
                "shape": "[B, T_max, 2]",
                "dtype": "bool",
                "meaning": "the physical hand was reconstructed (pose-bearing tracking state)",
                "hand_order": list(HAND_ORDER),
            },
            "frame_valid": {
                "shape": "[B, T_max]",
                "dtype": "bool",
                "meaning": "a real source frame, not batch padding",
                "note": "~frame_valid is the transformer-style key_padding_mask",
            },
        },
        "three_distinct_states": {
            "real_observation": "feature_valid=True, frame_valid=True (value may legitimately be 0.0)",
            "invalid_or_missing_sensor": "feature_valid=False, frame_valid=True",
            "batch_padding": "frame_valid=False (and feature_valid False throughout)",
        },
        "tracking_state": {
            "decision": "hand presence as a 2-channel mask; raw state codes as metadata only",
            "codes": {str(k): v for k, v in TRACKING_STATE_NAMES.items()},
            "pose_bearing_codes": list(POSE_BEARING_CODES),
        },
        "quaternion_policies": {
            "absolute": "verbatim TASK-008 palm quaternion, WXYZ, camera frame, w >= 0",
            "relative_first_valid": (
                "conjugate(q_ref) * q(t) with q_ref the FIRST VALID palm quaternion of that "
                "physical hand in that sequence; strictly causal; re-canonicalized to w >= 0"
            ),
            "default": config.quaternion_policy,
        },
        "normalization": {
            "policy": config.normalization,
            "bend": "TASK-008 bend_normalized = bend_angle_deg / 180.0 (fixed divisor)",
            "spread": "TASK-008 spread_normalized = spread_angle_deg / 180.0 (fixed divisor)",
            "quaternion": "already unit-norm and dimensionless",
            "fitted_statistics": "none; nothing is fitted, therefore nothing can leak",
        },
        "temporal": {
            "policy": "original length preserved; no cropping, resampling or fixed windows",
            "observed_length_range": [9, 70],
            "batching": "right-padded dense tensors plus lengths and frame_valid",
            "pack_padded_sequence": "supported; lengths are int64 on CPU and need not be sorted",
        },
        "labels": {
            "source": "label_index column of the frozen index (never directory ordering)",
            "num_classes": NUM_CLASSES,
            "index_range": list(LABEL_INDEX_RANGE),
        },
        "splits": {
            "source": "frozen TASK-008B LOSO files",
            "roles": ["train", "validation", "test"],
            "held_out_signers": ["01", "02", "03"],
        },
        "config_defaults": config.to_dict(),
    }


__all__ = [
    "CONTRACT_VERSION", "HAND_ORDER", "FINGER_ORDER", "CHAIN_ORDER", "SPREAD_PAIRS",
    "QUATERNION_ORDER", "BEND_CHANNELS_PER_HAND", "SPREAD_CHANNELS_PER_HAND",
    "QUATERNION_CHANNELS_PER_HAND", "SOURCE_SCHEMA_VERSION", "SOURCE_LAYOUT_VERSION",
    "SOURCE_CONTRACTS", "TRACKING_STATE_NAMES", "POSE_BEARING_CODES", "NUM_CLASSES",
    "LABEL_INDEX_RANGE", "FEATURE_SETS", "QUATERNION_POLICIES", "SequenceInputConfig",
    "families_for", "channels_per_hand", "feature_dimension", "channel_names",
    "channel_index", "hand_slice", "family_slice", "contract_document",
]
