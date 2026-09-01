# pose/mediapipe/

MediaPipe Hand Landmarker baseline adapter for Milestone 1.

The pilot runner is:

```bash
python scripts/run_mediapipe_pilot.py \
  --manifest datasets/manifests/karsl_milestone1_pilot.csv \
  --model datasets/raw/models/hand_landmarker.task
```

The default delegate is CPU and is recorded in every run. Use `--delegate GPU`
only when the MediaPipe GPU delegate is known to initialize in the execution
environment. The task model is downloaded separately with
`scripts/download_mediapipe_model.py` and remains ignored by Git.
