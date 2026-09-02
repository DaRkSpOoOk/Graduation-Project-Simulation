"""Canonical index layout for the TASK-005A kinematic representation.

Joint indices follow the 21-point OpenPose hand order that WiLoR emits via
``mano_to_openpose`` (see ``pose/wilor/npz_io.py``). Verified empirically on
the pilot: metacarpal lengths run 0.082-0.095 m with the middle finger longest
and the pinky shortest, phalanges shorten distally, and the wrist plus four
finger MCPs are near-coplanar.

Nothing here is derived from MANO local rotations. See ``kinematics.geometry``
for why.
"""

from __future__ import annotations

SCHEMA_VERSION = "hand_kinematics_v1"

# Track order is inherited from TASK-004 and must not be reordered.
TRACK_ORDER: tuple[str, str] = ("left", "right")
N_TRACKS = 2

FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
N_FINGERS = 5

# Generic chain-joint names. For the four non-thumb fingers these correspond
# approximately to MCP / PIP / DIP. The thumb keeps the generic names: its
# chain here is rooted at the wrist, so the three angles sit approximately at
# CMC / MCP / IP, but that mapping is not asserted as anatomically exact.
CHAIN_ORDER: tuple[str, str, str] = ("proximal", "middle", "distal")
N_CHAIN = 3

WRIST = 0

# Five points per finger: root, then the three chain joints, then the tip.
# The three flexion angles are the turn angles of this 5-point polyline.
FINGER_CHAINS: dict[str, tuple[int, int, int, int, int]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

# Proximal-phalanx direction used for spread: (from, to) per finger.
SPREAD_SEGMENTS: dict[str, tuple[int, int]] = {
    "thumb": (1, 2),
    "index": (5, 6),
    "middle": (9, 10),
    "ring": (13, 14),
    "pinky": (17, 18),
}

# Adjacent finger pairs whose in-plane angle becomes adjacent_spread_deg.
SPREAD_PAIRS: tuple[tuple[str, str], ...] = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)
N_SPREAD = 4

# Palm-frame defining landmarks.
PALM_INDEX_MCP = 5
PALM_MIDDLE_MCP = 9
PALM_PINKY_MCP = 17
PALM_POINTS: tuple[int, int, int, int] = (WRIST, PALM_INDEX_MCP, PALM_MIDDLE_MCP, PALM_PINKY_MCP)

N_JOINTS = 21
