"""Application playback adapter over the frozen timestamp-aware controller."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

from visualizer.playback import PlaybackController


@dataclass(frozen=True, slots=True)
class PlaybackDisplayState:
    """Presentation clock state; positions still reference source frames."""

    position: int
    frame_index: int
    timestamp_seconds: float
    interpolation_alpha: float
    playing: bool


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
        if target is not None and self.scientific.playing and position < len(self.scientific.timestamps) - 1:
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
        target = self._timeline_anchor + max(0.0, current - self._wall_anchor) * self.scientific.speed
        self.scientific.tick(current)
        return self._display_state(target)


__all__ = ["PersistentPlaybackController", "PlaybackDisplayState"]
