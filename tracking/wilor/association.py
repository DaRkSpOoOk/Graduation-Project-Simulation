"""Deterministic 2-track association for WiLoR detections.

There are exactly two canonical tracks (LEFT, RIGHT), so the assignment
problem is 2 x N. For N <= a handful of detections the optimal assignment is
found by exhaustive enumeration of all feasible track->detection injections,
which is both exact (identical to Hungarian for this size) and fully
deterministic, with no external solver dependency. Ties are broken by a
fixed lexicographic rule so repeated runs produce byte-identical output.

Detector handedness is used as *evidence with a cost*, never as an
override: TASK-003B showed the two extractors disagreeing on handedness
during occlusion, and MediaPipe flickering its label on a stationary hand,
so a label alone is not trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

from .config import TrackerConfig
from .schema import TRACK_NAMES, RawDetection

INFEASIBLE = float("inf")


@dataclass(frozen=True, slots=True)
class TrackPrior:
    """What the tracker remembers about one canonical track."""

    track: str
    last_centre: tuple[float, float] | None
    last_normalized_size: float | None
    frames_since_observed: int

    @property
    def is_initialized(self) -> bool:
        return self.last_centre is not None


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Chosen track -> detection mapping plus the evidence behind it."""

    mapping: dict[str, int]                 # track -> index into candidates
    costs: dict[str, float]                 # track -> individual cost
    total_cost: float
    runner_up_total_cost: float | None
    margin: float | None
    ambiguous: bool

    @property
    def assigned_indices(self) -> set[int]:
        return set(self.mapping.values())


def bbox_iou(first: RawDetection, second: RawDetection) -> float:
    """IoU of two square detector boxes given centre and side length."""

    def corners(detection: RawDetection) -> tuple[float, float, float, float]:
        half = detection.box_size / 2.0
        cx, cy = detection.box_center_xy
        return (cx - half, cy - half, cx + half, cy + half)

    ax0, ay0, ax1, ay1 = corners(first)
    bx0, by0, bx1, by1 = corners(second)
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = overlap_w * overlap_h
    union = first.box_size**2 + second.box_size**2 - intersection
    return intersection / union if union > 0 else 0.0


def suppress_same_label_duplicates(
    detections: list[RawDetection], config: TrackerConfig
) -> tuple[list[int], dict[int, str]]:
    """Drop near-duplicate boxes that share a detector label.

    Suppression is restricted to same-label pairs on purpose. In this pilot
    the two genuinely different hands reach a bbox IoU of 0.92, so a
    label-agnostic non-maximum suppression would delete real hands.

    Returns the surviving candidate indices (input order preserved) and a
    reason per suppressed index.
    """

    order = sorted(
        range(len(detections)),
        key=lambda i: (
            -(detections[i].detector_confidence or 0.0),
            detections[i].raw_detection_index,
        ),
    )
    suppressed: dict[int, str] = {}
    kept: list[int] = []
    for index in order:
        candidate = detections[index]
        for keeper in kept:
            other = detections[keeper]
            if candidate.detector_label != other.detector_label:
                continue
            if candidate.detector_label is None:
                continue
            iou = bbox_iou(candidate, other)
            if iou >= config.duplicate_iou_threshold:
                suppressed[index] = (
                    f"duplicate_same_label_iou={iou:.3f}>=({config.duplicate_iou_threshold})"
                    f"_of_detection_{other.raw_detection_index}"
                )
                break
        else:
            kept.append(index)
    return sorted(kept), suppressed


def gate_radius(prior: TrackPrior, config: TrackerConfig) -> float:
    """Gate widens with staleness but is bounded."""

    radius = config.base_gate_radius * (
        1.0 + config.gate_growth_per_missing_frame * max(0, prior.frames_since_observed)
    )
    return min(radius, config.max_gate_radius)


def pair_cost(
    prior: TrackPrior, detection: RawDetection, config: TrackerConfig
) -> tuple[float, float | None]:
    """Cost of binding ``detection`` to ``prior``'s track.

    Returns ``(cost, distance)``. ``cost`` is INFEASIBLE when the detection
    lies outside the track's gate. Components, each normalized to [0, 1] and
    weighted to sum to 1:

    * position: normalized bbox-centre distance / gate radius
    * label:    1 when the detector label contradicts the track identity
    * scale:    |log(size ratio)| / log(2), clipped
    * confidence: 1 - detector confidence
    """

    centre = detection.normalized_centre
    distance: float | None = None

    if prior.is_initialized:
        assert prior.last_centre is not None
        distance = math.hypot(centre[0] - prior.last_centre[0], centre[1] - prior.last_centre[1])
        radius = gate_radius(prior, config)
        if distance > radius:
            return INFEASIBLE, distance
        position_term = distance / radius
    else:
        # An uninitialized track has no spatial prior; position is neutral.
        position_term = 0.5

    label_term = 0.0
    if detection.detector_label is None:
        label_term = 0.5
    elif detection.detector_label != prior.track:
        label_term = 1.0

    scale_term = 0.0
    if prior.last_normalized_size and detection.normalized_size > 0:
        ratio = detection.normalized_size / prior.last_normalized_size
        if ratio > 0:
            scale_term = min(1.0, abs(math.log(ratio)) / config.scale_log_reference)

    confidence_term = 1.0 - (detection.detector_confidence or 0.0)
    confidence_term = min(1.0, max(0.0, confidence_term))

    cost = (
        config.weight_position * position_term
        + config.weight_label * label_term
        + config.weight_scale * scale_term
        + config.weight_confidence * confidence_term
    )
    return cost, distance


def solve_assignment(
    priors: dict[str, TrackPrior],
    detections: list[RawDetection],
    config: TrackerConfig,
) -> AssignmentResult:
    """Exact minimum-cost assignment of up to two tracks to detections.

    Every subset of tracks (including the empty and single-track cases) is
    enumerated, so a track is left unassigned when no feasible detection
    exists. Determinism: candidate options are generated in a fixed order and
    ties resolve to the lexicographically smallest mapping.
    """

    cost_table: dict[tuple[str, int], float] = {}
    for track in TRACK_NAMES:
        for index, detection in enumerate(detections):
            cost, _ = pair_cost(priors[track], detection, config)
            cost_table[(track, index)] = cost

    options: list[tuple[float, int, tuple[tuple[str, int], ...]]] = []
    indices = range(len(detections))
    for size in (2, 1, 0):
        if size > min(len(TRACK_NAMES), len(detections)):
            continue
        for tracks in (TRACK_NAMES[:1], TRACK_NAMES[1:], TRACK_NAMES) if size else ((),):
            if len(tracks) != size:
                continue
            for chosen in permutations(indices, size):
                pairs = tuple(zip(tracks, chosen))
                total = sum(cost_table[pair] for pair in pairs)
                if math.isinf(total):
                    continue
                # Prefer assignments that bind more tracks; among equal
                # counts prefer lower cost. Encoded as (-size, total).
                options.append((total, -size, pairs))

    if not options:
        return AssignmentResult({}, {}, 0.0, None, None, False)

    options.sort(key=lambda item: (item[1], round(item[0], 12), item[2]))
    best_total, best_negative_size, best_pairs = options[0]

    runner_up: float | None = None
    for total, negative_size, pairs in options[1:]:
        if negative_size == best_negative_size and pairs != best_pairs:
            runner_up = total
            break

    margin = None if runner_up is None else runner_up - best_total
    ambiguous = margin is not None and margin < config.ambiguity_margin

    # Two candidates that are spatially on top of each other are
    # unresolvable regardless of cost separation.
    if len(best_pairs) == 2:
        first = detections[best_pairs[0][1]].normalized_centre
        second = detections[best_pairs[1][1]].normalized_centre
        if math.hypot(first[0] - second[0], first[1] - second[1]) < config.proximity_ambiguity_radius:
            ambiguous = True

    mapping = {track: index for track, index in best_pairs}
    costs = {track: cost_table[(track, index)] for track, index in best_pairs}
    return AssignmentResult(mapping, costs, best_total, runner_up, margin, ambiguous)
