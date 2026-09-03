"""Interactive playback foundation for one stored virtual-glove sequence.

The package deliberately keeps sequence loading/playback independent from any
Arabic keyboard or recognizer.  Those later components can resolve a
``sample_id`` and hand it to :func:`load_sequence`.
"""

from .contract import (
    CHAIN_ORDER,
    FINGER_ORDER,
    SPREAD_PAIRS,
    TRACK_ORDER,
    FrameData,
    HandGeometry,
    PlaybackSequence,
    SensorReading,
    SensorSpec,
    validate_sensor_layout,
)
from .loader import ArtifactValidationError, load_sequence
from .playback import PlaybackController, PlaybackError

__all__ = [
    "ArtifactValidationError",
    "CHAIN_ORDER",
    "FINGER_ORDER",
    "FrameData",
    "HandGeometry",
    "PlaybackController",
    "PlaybackError",
    "PlaybackSequence",
    "SPREAD_PAIRS",
    "SensorReading",
    "SensorSpec",
    "TRACK_ORDER",
    "load_sequence",
    "validate_sensor_layout",
]
