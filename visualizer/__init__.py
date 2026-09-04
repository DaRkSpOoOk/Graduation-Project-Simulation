"""Core visualizer contracts plus non-graphical Core-28 queue components.

Rendering/playback and keyboard/catalog/queue code remain separate subpackages.
The package-level exports keep the TASK-007A renderer API available while the
TASK-007B subpackages provide independent text-to-exemplar resolution.
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
    "catalog",
    "keyboard",
    "load_sequence",
    "mapping",
    "queue",
    "validate_sensor_layout",
]
