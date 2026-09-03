"""Timestamp-aware playback state machine, independent of GUI code."""

from __future__ import annotations

import bisect
import time
from typing import Sequence

import numpy as np


class PlaybackError(ValueError):
    """Raised when playback timing inputs are not usable."""


class PlaybackController:
    """Play, pause, restart and scrub one sequence using stored timestamps.

    ``tick`` accepts an optional monotonic wall-clock value, which makes the
    timing logic deterministic to unit-test without a GUI.  Frame selection is
    always an index into the original stored sequence; no resampling occurs.
    """

    def __init__(
        self,
        timestamps: Sequence[float],
        frame_indices: Sequence[int],
        *,
        speed: float = 1.0,
    ) -> None:
        self.timestamps = tuple(float(value) for value in timestamps)
        self.frame_indices = tuple(int(value) for value in frame_indices)
        if not self.timestamps or len(self.timestamps) != len(self.frame_indices):
            raise PlaybackError("timestamps and frame indices must be non-empty and have equal length")
        if not np.isfinite(self.timestamps).all():
            raise PlaybackError("timestamps must be finite")
        if any(right <= left for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise PlaybackError("timestamps must be strictly increasing")
        if any(right <= left for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise PlaybackError("frame indices must be strictly increasing")
        self.position = 0
        self.playing = False
        self.speed = 1.0
        self._wall_anchor = 0.0
        self._timeline_anchor = self.timestamps[0]
        self.set_speed(speed)

    @property
    def frame_index(self) -> int:
        return self.frame_indices[self.position]

    @property
    def at_end(self) -> bool:
        return self.position == len(self.timestamps) - 1

    def set_speed(self, speed: float, now: float | None = None) -> None:
        value = float(speed)
        if not np.isfinite(value) or value <= 0:
            raise PlaybackError("playback speed must be finite and positive")
        if self.playing:
            current_now = time.monotonic() if now is None else float(now)
            self.tick(current_now)
            self._wall_anchor = current_now
            self._timeline_anchor = self.timestamps[self.position]
        self.speed = value

    def play(self, now: float | None = None) -> None:
        current_now = time.monotonic() if now is None else float(now)
        if self.at_end:
            self.position = 0
        self._wall_anchor = current_now
        self._timeline_anchor = self.timestamps[self.position]
        self.playing = True

    def pause(self, now: float | None = None) -> None:
        if self.playing:
            current_now = time.monotonic() if now is None else float(now)
            self.tick(current_now)
        self.playing = False

    def restart(self) -> None:
        self.position = 0
        self.playing = False
        self._timeline_anchor = self.timestamps[0]

    def seek(self, position: int, now: float | None = None) -> int:
        value = int(position)
        if not 0 <= value < len(self.timestamps):
            raise IndexError(f"playback position out of range: {position}")
        self.position = value
        if self.playing:
            current_now = time.monotonic() if now is None else float(now)
            self._wall_anchor = current_now
            self._timeline_anchor = self.timestamps[value]
        return self.position

    def seek_frame(self, frame_index: int, now: float | None = None) -> int:
        try:
            position = self.frame_indices.index(int(frame_index))
        except ValueError as exc:
            raise KeyError(f"stored frame index not found: {frame_index}") from exc
        return self.seek(position, now=now)

    def tick(self, now: float | None = None) -> int:
        if not self.playing:
            return self.position
        current_now = time.monotonic() if now is None else float(now)
        if not np.isfinite(current_now):
            raise PlaybackError("wall-clock timestamp must be finite")
        target = self._timeline_anchor + max(0.0, current_now - self._wall_anchor) * self.speed
        if target >= self.timestamps[-1]:
            self.position = len(self.timestamps) - 1
            self.playing = False
            return self.position
        self.position = max(0, bisect.bisect_right(self.timestamps, target) - 1)
        return self.position
