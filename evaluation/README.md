# evaluation/

Shared metrics and benchmark protocols across competing extraction/tracking approaches.

`metrics/mediapipe_baseline.py` reports detection/missingness, handedness
changes, identity-instability heuristics, wrist jumps, approximate temporal
jitter, world-space bone-length variation, coordinate completeness, and
runtime. Wrist-jump summaries include both the raw detector-order sequence and
the derived handedness-labelled sequence, because label reassignment can
itself create a large convenience-channel jump. Diagnostic jump values are
descriptive pilot statistics; this bootstrap task sets no acceptance
thresholds.
