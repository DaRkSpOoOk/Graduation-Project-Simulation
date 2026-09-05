"""Application playback adapter over the frozen timestamp-aware controller."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

from visualizer.playback import PlaybackController


@dataclass(frozen=True, slots=True)
class PlaybackDisplayState:
    """Presentation clock state; positions still reference source frames."""

    position: int
    frame_index: int
    timestamp_seconds: float
    interpolation_alpha: float
    playing: bool


@dataclass(slots=True)
class PlaybackBoundaryTrace:
    """Evidence record for one sign's source-frame/queue boundary.

    ``displayed_positions`` contains unique source positions in the order in
    which the application published them.  It is deliberately separate from
    the presentation interpolation clock: an interpolated pose can be drawn
    between two anchors without becoming a scientific frame.
    """

    sample_id: str
    character: str
    source_frame_indices: tuple[int, ...]
    displayed_positions: list[int]
    displayed_frame_indices: list[int]
    queue_advanced: bool = False
    queue_advanced_after_final: bool = False
    early_queue_advance: bool = False
    transition_distance_degrees: float | None = None
    transition_duration_ms: float | None = None
    boundary_hold_ms: float | None = None

    @classmethod
    def for_sequence(
        cls, sample_id: str, character: str, frame_indices: Sequence[int]
    ) -> "PlaybackBoundaryTrace":
        return cls(
            sample_id=str(sample_id),
            character=str(character),
            source_frame_indices=tuple(int(value) for value in frame_indices),
            displayed_positions=[],
            displayed_frame_indices=[],
        )

    def record(self, position: int) -> None:
        value = int(position)
        if not 0 <= value < len(self.source_frame_indices):
            raise IndexError(f"source position out of range: {position}")
        if self.displayed_positions and self.displayed_positions[-1] == value:
            return
        self.displayed_positions.append(value)
        self.displayed_frame_indices.append(self.source_frame_indices[value])

    @property
    def first_source_frame(self) -> int | None:
        return self.displayed_frame_indices[0] if self.displayed_frame_indices else None

    @property
    def final_source_frame(self) -> int | None:
        return (
            self.displayed_frame_indices[-1] if self.displayed_frame_indices else None
        )

    @property
    def last_frame_presented(self) -> bool:
        return (
            bool(self.source_frame_indices)
            and self.final_source_frame == self.source_frame_indices[-1]
        )

    @property
    def all_source_positions_presented(self) -> bool:
        return self.displayed_positions == list(range(len(self.source_frame_indices)))

    def mark_queue_advance(self) -> None:
        self.queue_advanced = True
        self.queue_advanced_after_final = self.last_frame_presented
        self.early_queue_advance = not self.queue_advanced_after_final

    def set_transition_plan(
        self, *, distance_degrees: float, duration_ms: float, hold_ms: float
    ) -> None:
        """Attach presentation timing without changing source-frame evidence."""

        self.transition_distance_degrees = float(distance_degrees)
        self.transition_duration_ms = float(duration_ms)
        self.boundary_hold_ms = float(hold_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "character": self.character,
            "source_sequence_length": len(self.source_frame_indices),
            "source_frame_indices": list(self.source_frame_indices),
            "displayed_positions": list(self.displayed_positions),
            "displayed_frame_indices": list(self.displayed_frame_indices),
            "first_source_frame": self.first_source_frame,
            "final_source_frame": self.final_source_frame,
            "all_source_positions_presented": self.all_source_positions_presented,
            "last_frame_presented": self.last_frame_presented,
            "queue_advanced": self.queue_advanced,
            "queue_advanced_after_final": self.queue_advanced_after_final,
            "early_queue_advance": self.early_queue_advance,
            "transition_distance_degrees": self.transition_distance_degrees,
            "transition_duration_ms": self.transition_duration_ms,
            "boundary_hold_ms": self.boundary_hold_ms,
        }


class PersistentPlaybackController:
    """Keep scientific frame selection exact while exposing render interpolation."""

    def __init__(
        self,
        timestamps: Sequence[float],
        frame_indices: Sequence[int],
        *,
        speed: float = 1.0,
    ) -> None:
        self.scientific = PlaybackController(timestamps, frame_indices, speed=speed)
        self._wall_anchor = 0.0
        self._timeline_anchor = float(self.scientific.timestamps[0])

    @property
    def position(self) -> int:
        return self.scientific.position

    @property
    def frame_index(self) -> int:
        return self.scientific.frame_index

    @property
    def playing(self) -> bool:
        return self.scientific.playing

    @property
    def at_end(self) -> bool:
        return self.scientific.at_end

    def _now(self, now: float | None) -> float:
        return time.monotonic() if now is None else float(now)

    def _display_state(self, target: float | None = None) -> PlaybackDisplayState:
        position = self.scientific.position
        alpha = 0.0
        if (
            target is not None
            and self.scientific.playing
            and position < len(self.scientific.timestamps) - 1
        ):
            left = self.scientific.timestamps[position]
            right = self.scientific.timestamps[position + 1]
            alpha = min(1.0, max(0.0, (target - left) / (right - left)))
        return PlaybackDisplayState(
            position=position,
            frame_index=self.scientific.frame_index,
            timestamp_seconds=self.scientific.timestamps[position],
            interpolation_alpha=alpha,
            playing=self.scientific.playing,
        )

    def play(self, now: float | None = None) -> PlaybackDisplayState:
        current = self._now(now)
        self.scientific.play(current)
        self._wall_anchor = current
        self._timeline_anchor = self.scientific.timestamps[self.scientific.position]
        return self._display_state(self._timeline_anchor)

    def pause(self, now: float | None = None) -> PlaybackDisplayState:
        current = self._now(now)
        if self.scientific.playing:
            self.tick(current)
            self.scientific.pause(current)
        self._wall_anchor = current
        self._timeline_anchor = self.scientific.timestamps[self.scientific.position]
        return self._display_state()

    def restart(self) -> PlaybackDisplayState:
        self.scientific.restart()
        self._wall_anchor = 0.0
        self._timeline_anchor = self.scientific.timestamps[0]
        return self._display_state()

    def set_speed(self, speed: float, now: float | None = None) -> PlaybackDisplayState:
        current = self._now(now)
        if self.scientific.playing:
            self.tick(current)
        self.scientific.set_speed(speed, current)
        self._wall_anchor = current
        self._timeline_anchor = self.scientific.timestamps[self.scientific.position]
        return self._display_state()

    def tick(self, now: float | None = None) -> PlaybackDisplayState:
        current = self._now(now)
        if not self.scientific.playing:
            return self._display_state()
        target = (
            self._timeline_anchor
            + max(0.0, current - self._wall_anchor) * self.scientific.speed
        )
        self.scientific.tick(current)
        return self._display_state(target)


__all__ = [
    "PersistentPlaybackController",
    "PlaybackBoundaryTrace",
    "PlaybackDisplayState",
]
