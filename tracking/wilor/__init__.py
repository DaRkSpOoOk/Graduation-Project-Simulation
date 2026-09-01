"""WiLoR-only temporal dual-hand tracking (TASK-004A).

Derives two canonical subject-anatomical tracks (LEFT, RIGHT) from
immutable raw WiLoR full-mode output. This stage establishes temporal
identity, state and quality metadata only: it computes no joint angles, no
simulated sensor values and no learning features.
"""

from .config import DEFAULT_CONFIG, TrackerConfig
from .metrics import TrackingMetrics, aggregate_metrics, compute_metrics
from .npz_io import load_tracked_sequence, save_tracked_sequence
from .schema import TRACK_NAMES, TrackedFrame, TrackedHand, TrackedSequence, TrackState
from .source import RawInputError, RawSequence, load_raw_sequence
from .tracker import track_sequence

__all__ = [
    "DEFAULT_CONFIG",
    "TRACK_NAMES",
    "RawInputError",
    "RawSequence",
    "TrackedFrame",
    "TrackedHand",
    "TrackedSequence",
    "TrackState",
    "TrackerConfig",
    "TrackingMetrics",
    "aggregate_metrics",
    "compute_metrics",
    "load_raw_sequence",
    "load_tracked_sequence",
    "save_tracked_sequence",
    "track_sequence",
]
