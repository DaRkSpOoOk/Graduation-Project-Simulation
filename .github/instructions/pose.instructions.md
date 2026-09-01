# pose/ instructions

Path focus: `pose/**`

- `pose/common/` contains extractor-agnostic interfaces/schemas.
- Keep model-specific logic isolated in `mediapipe/`, `wilor/`, `hamer/`, or `omnihands/`.
- Do not force MediaPipe-only or MANO-only assumptions in common contracts.
