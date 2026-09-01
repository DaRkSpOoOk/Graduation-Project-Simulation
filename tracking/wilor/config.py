"""Tunable constants for the WiLoR temporal dual-hand tracker.

Every default below was calibrated against the validated 18-video /
894-frame WiLoR full-mode pilot run (1,779 reconstructed rows), not chosen
arbitrarily. The measured distribution that motivates each value is quoted
in the field comment and reproduced in
reports/tracking/TASK-004A-wilor-temporal-dual-hand-tracking.md.

Positions are expressed in image-normalized units (pixel coordinate divided
by the frame width/height from ``mano_references['img_size_wh']``), so the
thresholds are resolution independent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Deterministic tracker parameters. No value is learned or adaptive."""

    # --- association gating -------------------------------------------------
    # Measured frame-to-frame normalized bbox-centre displacement for a hand
    # keeping the same detector label: p50 0.0015, p95 0.0153, p99 0.0262,
    # p99.9 0.0434, max 0.0535. 0.08 is ~1.5x the observed maximum.
    base_gate_radius: float = 0.08
    # A track that has been unobserved for k frames may have moved further.
    # Gate grows linearly and is capped so a stale track can never capture an
    # arbitrarily distant detection.
    gate_growth_per_missing_frame: float = 0.5
    max_gate_radius: float = 0.30

    # --- association cost weights (sum to 1.0 for interpretability) ---------
    weight_position: float = 0.55
    weight_label: float = 0.25
    weight_scale: float = 0.10
    weight_confidence: float = 0.10
    # |log(size ratio)| is divided by this before clipping to [0, 1].
    # Measured consecutive same-label box size ratio: p1 0.878, p99 1.138,
    # so |log ratio| is normally < 0.13; log(2) tolerates a doubling.
    scale_log_reference: float = 0.6931471805599453  # log(2)

    # --- ambiguity ----------------------------------------------------------
    # If the best and second-best feasible assignments are within this total
    # cost of each other, the frame's bound tracks are flagged AMBIGUOUS.
    ambiguity_margin: float = 0.10
    # Two candidates closer than this (normalized) are spatially
    # unresolvable regardless of assignment cost.
    proximity_ambiguity_radius: float = 0.02

    # --- duplicate suppression ---------------------------------------------
    # Applied ONLY to detection pairs that share a detector label. Measured
    # same-frame bbox IoU between the two genuinely different hands reaches
    # p90 0.688 / max 0.924 in this pilot, so cross-label suppression would
    # be destructive; same-label overlap above this threshold occurred
    # exactly once (frame 39 of karsl_test_s02_sign0176_repfirst, IoU 0.747).
    duplicate_iou_threshold: float = 0.5

    # --- cross-label duplicate ("ghost") suppression ------------------------
    # TASK-004D. A weak detection can land on top of an already-detected hand
    # while carrying the OPPOSITE handedness label, so same-label suppression
    # cannot see it. Validated across all 885 same-frame cross-label detection
    # pairs in the 894-frame pilot: the ghost pairs occupy IoU 0.833-0.924,
    # centre-separation/box 0.0275-0.0613 and confidence ratio 0.381-0.511,
    # while every genuine two-hand pair stays at IoU <= 0.778,
    # separation/box >= 0.0831 and confidence ratio >= 0.638. All three
    # conditions must hold together, and each threshold sits inside the gap
    # between those two populations rather than on either boundary.
    cross_label_duplicate_iou: float = 0.80
    cross_label_duplicate_separation_ratio: float = 0.07
    cross_label_duplicate_confidence_ratio: float = 0.55
    # A track returning from absence must not be bound to a candidate that is
    # ghost-suspect against the other track's detection in the same frame.
    require_distinct_candidate_for_reacquisition: bool = True

    # --- quality gate (conservative; marks rather than fabricates) ----------
    # Detector operating point is fixed by TASK-002/003B and is NOT retuned
    # here; this floor only rejects rows below the detector's own threshold.
    min_detection_confidence: float = 0.30
    # Measured joint-span / palm-length ratio: min 1.28, p1 1.37, p99 2.28,
    # max 2.34. The gate is deliberately far outside the observed range.
    min_span_palm_ratio: float = 1.0
    max_span_palm_ratio: float = 3.0
    # Measured |projected joint centroid - bbox centre| / bbox size:
    # max 0.151. 0.35 is a conservative safety net.
    max_projection_centre_offset: float = 0.35
    # Reported as a LOW_QUALITY_POSE_JUMP flag only; never a rejection.
    # Measured consecutive global-orientation geodesic jump: p50 1.8 deg,
    # p95 16.5 deg, p99 38.1 deg, max 102.2 deg.
    orientation_jump_warn_degrees: float = 60.0

    # --- occlusion heuristic ------------------------------------------------
    # A MISSING track is only *suggested* as LIKELY_OCCLUDED when the other
    # track is observed and the missing track was last seen within this
    # normalized distance of the other hand's current position.
    occlusion_proximity_radius: float = 0.10
    # ... and only for this many frames after the last observation.
    occlusion_max_age_frames: int = 15

    # --- seeding ------------------------------------------------------------
    # When the first usable frame has two same-label detections, seeding falls
    # back to image position. TASK-003A measured that for this facing,
    # non-mirrored capture the subject's RIGHT hand appears at the smaller
    # image x in ~95-97% of two-hand frames. Used for seeding only.
    seed_right_hand_at_smaller_x: bool = True

    schema_version: str = "wilor_tracked_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrackerConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown tracker config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrackerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if not 0 < self.base_gate_radius <= self.max_gate_radius:
            raise ValueError("base_gate_radius must be positive and <= max_gate_radius")
        weights = (
            self.weight_position,
            self.weight_label,
            self.weight_scale,
            self.weight_confidence,
        )
        if any(w < 0 for w in weights):
            raise ValueError("cost weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("cost weights must sum to 1.0")
        if not 0 <= self.duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be within [0, 1]")
        if self.min_span_palm_ratio >= self.max_span_palm_ratio:
            raise ValueError("span/palm ratio bounds are inverted")
        if not 0 <= self.cross_label_duplicate_iou <= 1:
            raise ValueError("cross_label_duplicate_iou must be within [0, 1]")
        if self.cross_label_duplicate_iou < self.duplicate_iou_threshold:
            raise ValueError(
                "cross_label_duplicate_iou must be at least as strict as "
                "duplicate_iou_threshold: suppressing an opposite-label "
                "detection needs more evidence, not less"
            )
        if self.cross_label_duplicate_separation_ratio <= 0:
            raise ValueError("cross_label_duplicate_separation_ratio must be positive")
        if not 0 < self.cross_label_duplicate_confidence_ratio <= 1:
            raise ValueError("cross_label_duplicate_confidence_ratio must be within (0, 1]")


DEFAULT_CONFIG = TrackerConfig()
