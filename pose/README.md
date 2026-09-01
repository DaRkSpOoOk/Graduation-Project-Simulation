# pose/

Video/image → hand pose extraction.

- `common/`: extractor-agnostic schemas/interfaces
- `mediapipe/`, `wilor/`, `hamer/`, `omnihands/`: isolated model-specific implementations

## MediaPipe pilot

`pose/mediapipe/extractor.py` uses the official MediaPipe Hand Landmarker
Python Tasks API in `VIDEO` mode with two hands. Raw detector order,
detector-provided handedness, handedness scores, image landmarks, world
landmarks, timestamps, and returned-hand presence are stored in one NPZ per
video. Missing per-result detection/presence/tracking scores are represented
as unavailable rather than inferred.
