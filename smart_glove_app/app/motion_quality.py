"""Presentation-only motion quality helpers for TASK-007H.

The objects in this module operate on solved :class:`HandPose` values.  They
never accept or return a ``PlaybackSequence`` and therefore cannot put a
render-time transition into the scientific or recognition path.

The transition is deliberately absolute: every sample is evaluated from the
two immutable endpoint poses, with quaternion SLERP and an eased scalar
parameter.  No sample is composed onto the previous displayed sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from smart_glove_app.rendering.presentation_rig import PresentationRig, SIDES

from .hand_pose_solver import (
    HandPose,
    quaternion_angle_deg,
    quaternion_inverse,
    quaternion_multiply,
    slerp,
)


def _copy_pose(pose: HandPose) -> HandPose:
    """Return an immutable-value snapshot of one solved presentation pose."""

    return HandPose(
        side=pose.side,
        bones_wxyz={
            name: tuple(float(value) for value in quaternion)
            for name, quaternion in pose.bones_wxyz.items()
        },
        state=pose.state,
        dimmed=pose.dimmed,
        bend_valid_count=pose.bend_valid_count,
        spread_valid_count=pose.spread_valid_count,
        wrist_valid=pose.wrist_valid,
        wrist_angle_deg=float(pose.wrist_angle_deg),
    )


def copy_pose_map(poses: Mapping[str, HandPose]) -> dict[str, HandPose]:
    """Snapshot both endpoint poses without retaining mutable mapping state."""

    return {side: _copy_pose(poses[side]) for side in SIDES}


def smoothstep(value: float) -> float:
    """Cubic ease-in/ease-out with a bounded presentation parameter."""

    t = min(1.0, max(0.0, float(value)))
    return t * t * (3.0 - 2.0 * t)


def _angular_difference_deg(
    first: tuple[float, ...], second: tuple[float, ...]
) -> float:
    delta = quaternion_multiply(quaternion_inverse(first), second)
    return quaternion_angle_deg(delta)


def pose_distance_degrees(
    first: Mapping[str, HandPose],
    second: Mapping[str, HandPose],
    rig: PresentationRig,
) -> float:
    """Compute the weighted endpoint distance used for transition timing.

    The metric averages the shortest angular distance over both hands.  Wrist
    orientation has weight 2.0, spread metacarpals 1.25, and each finger
    phalange 1.0.  Root and presentation-only placement bones are intentionally
    excluded: a transition must describe a hand-shape change, not a camera or
    layout change.
    """

    spread_bones = {target.bone for target in rig.spread_targets}
    bend_bones = {joint for chain in rig.chains.values() for joint in chain.joints}
    weighted_total = 0.0
    total_weight = 0.0
    for side in SIDES:
        first_pose = first[side]
        second_pose = second[side]
        for bone in rig.required_bones:
            if bone == rig.wrist_bone:
                weight = 2.0
            elif bone in spread_bones:
                weight = 1.25
            elif bone in bend_bones:
                weight = 1.0
            else:
                continue
            if bone not in first_pose.bones_wxyz or bone not in second_pose.bones_wxyz:
                continue
            weighted_total += weight * _angular_difference_deg(
                first_pose.bones_wxyz[bone], second_pose.bones_wxyz[bone]
            )
            total_weight += weight
    return weighted_total / total_weight if total_weight else 0.0


@dataclass(frozen=True, slots=True)
class TransitionConfig:
    """Presentation timing policy, independent of scientific timestamps."""

    boundary_hold_ms: float = 80.0
    minimum_duration_ms: float = 150.0
    maximum_duration_ms: float = 350.0
    distance_for_max_duration_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.boundary_hold_ms < 0.0:
            raise ValueError("boundary_hold_ms must be non-negative")
        if self.minimum_duration_ms < 0.0:
            raise ValueError("minimum_duration_ms must be non-negative")
        if self.maximum_duration_ms < self.minimum_duration_ms:
            raise ValueError("maximum_duration_ms must be at least minimum_duration_ms")
        if self.distance_for_max_duration_deg <= 0.0:
            raise ValueError("distance_for_max_duration_deg must be positive")


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    """Resolved timing for one endpoint-to-endpoint presentation transition."""

    distance_degrees: float
    duration_ms: float
    hold_ms: float


def make_transition_plan(
    first: Mapping[str, HandPose],
    second: Mapping[str, HandPose],
    rig: PresentationRig,
    config: TransitionConfig,
) -> TransitionPlan:
    distance = pose_distance_degrees(first, second, rig)
    fraction = min(1.0, max(0.0, distance / config.distance_for_max_duration_deg))
    duration = config.minimum_duration_ms + fraction * (
        config.maximum_duration_ms - config.minimum_duration_ms
    )
    return TransitionPlan(distance, duration, config.boundary_hold_ms)


def interpolate_pose_maps(
    first: Mapping[str, HandPose],
    second: Mapping[str, HandPose],
    alpha: float,
    *,
    wrist_bone: str = "palm",
) -> dict[str, HandPose]:
    """SLERP two solved pose maps using one eased display-time parameter."""

    t = smoothstep(alpha)
    result: dict[str, HandPose] = {}
    for side in SIDES:
        first_pose = first[side]
        second_pose = second[side]
        bones = {
            bone: tuple(
                float(value)
                for value in slerp(
                    first_pose.bones_wxyz[bone], second_pose.bones_wxyz[bone], t
                )
            )
            for bone in first_pose.bones_wxyz
            if bone in second_pose.bones_wxyz
        }
        wrist = bones.get(
            wrist_bone, first_pose.bones_wxyz.get(wrist_bone, (1.0, 0.0, 0.0, 0.0))
        )
        result[side] = HandPose(
            side=side,
            bones_wxyz=bones,
            state="TRANSITION",
            dimmed=second_pose.dimmed,
            bend_valid_count=second_pose.bend_valid_count,
            spread_valid_count=second_pose.spread_valid_count,
            wrist_valid=second_pose.wrist_valid,
            wrist_angle_deg=quaternion_angle_deg(wrist),
        )
    return result


@dataclass(frozen=True, slots=True)
class TransitionSample:
    """One presentation-only sample from a boundary transition."""

    poses: Mapping[str, HandPose]
    phase: str
    alpha: float
    done: bool


class PresentationTransition:
    """Time-addressable transition between two absolute solved pose maps."""

    def __init__(
        self,
        first: Mapping[str, HandPose],
        second: Mapping[str, HandPose],
        *,
        started_at: float,
        rig: PresentationRig,
        config: TransitionConfig,
    ) -> None:
        self.first = copy_pose_map(first)
        self.second = copy_pose_map(second)
        self.started_at = float(started_at)
        self._wrist_bone = rig.wrist_bone
        self.plan = make_transition_plan(self.first, self.second, rig, config)
        self._blend_start = self.started_at + self.plan.hold_ms / 1000.0
        self._blend_duration = self.plan.duration_ms / 1000.0

    def sample(self, now: float) -> TransitionSample:
        current = float(now)
        if current < self._blend_start:
            return TransitionSample(self.first, "BOUNDARY_HOLD", 0.0, False)
        if self._blend_duration <= 0.0:
            return TransitionSample(self.second, "COMPLETE", 1.0, True)
        alpha = (current - self._blend_start) / self._blend_duration
        if alpha >= 1.0:
            return TransitionSample(self.second, "COMPLETE", 1.0, True)
        return TransitionSample(
            interpolate_pose_maps(
                self.first, self.second, alpha, wrist_bone=self._wrist_bone
            ),
            "BLENDING",
            min(1.0, max(0.0, alpha)),
            False,
        )


__all__ = [
    "PresentationTransition",
    "TransitionConfig",
    "TransitionPlan",
    "TransitionSample",
    "copy_pose_map",
    "interpolate_pose_maps",
    "make_transition_plan",
    "pose_distance_degrees",
    "smoothstep",
]
