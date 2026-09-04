"""Stateful queue for isolated Core-28 sign sequences.

The queue never creates geometry or transition frames.  A separator is an
explicit neutral-gap item, allowing a renderer to create a visual boundary
without making that boundary part of the recognition dataset.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal

from visualizer.catalog import SequenceDescriptor
from visualizer.mapping import (
    CharacterResolution,
    Core28Resolver,
    UnsupportedCharacterError,
    unsupported_sequence_at,
)

GAP_DURATION_MS = 250
QueueItemType = Literal["sign", "gap"]


class QueueState(str, Enum):
    PENDING = "PENDING"
    PLAYING = "PLAYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class UnsupportedTextIssue:
    position: int
    character: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"character": self.character, "position": self.position, "reason": self.reason}


class UnsupportedTextError(ValueError):
    """Atomic text-enqueue failure with every unsupported position reported."""

    def __init__(self, issues: Iterable[UnsupportedTextIssue]) -> None:
        self.issues = tuple(issues)
        message = "; ".join(
            f"position {issue.position}: Unsupported Core-28 character: {issue.character}"
            for issue in self.issues
        )
        super().__init__(message or "text contains unsupported Core-28 characters")


@dataclass(slots=True)
class PlaybackQueueItem:
    """One sign request or explicit renderer gap."""

    item_type: QueueItemType
    character: str
    state: QueueState = QueueState.PENDING
    sign_id: str | None = None
    label_index: int | None = None
    sample_id: str | None = None
    signer_id: str | None = None
    sequence_descriptor: SequenceDescriptor | None = None
    gap_after_ms: int | None = None
    transition_policy: str | None = None
    failure_reason: str | None = None

    @classmethod
    def from_resolution(cls, resolution: CharacterResolution) -> "PlaybackQueueItem":
        return cls(
            item_type="sign",
            character=resolution.character,
            sign_id=resolution.sign_id,
            label_index=resolution.label_index,
            sample_id=resolution.sample_id,
            signer_id=resolution.signer_id,
            sequence_descriptor=resolution.sequence_descriptor,
        )

    @classmethod
    def neutral_gap(cls, character: str, *, duration_ms: int = GAP_DURATION_MS) -> "PlaybackQueueItem":
        if duration_ms < 0:
            raise ValueError("gap duration must be non-negative")
        return cls(
            item_type="gap",
            character=character,
            gap_after_ms=duration_ms,
            transition_policy="neutral_gap",
        )

    @property
    def is_sign(self) -> bool:
        return self.item_type == "sign"

    @property
    def sequence_path_or_descriptor(self) -> SequenceDescriptor | None:
        return self.sequence_descriptor

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "failure_reason": self.failure_reason,
            "gap_after_ms": self.gap_after_ms,
            "item_type": self.item_type,
            "label_index": self.label_index,
            "sample_id": self.sample_id,
            "sequence_descriptor": (
                self.sequence_descriptor.to_dict() if self.sequence_descriptor is not None else None
            ),
            "sign_id": self.sign_id,
            "signer_id": self.signer_id,
            "state": self.state.value,
            "transition_policy": self.transition_policy,
        }


def _is_separator(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


class PlaybackQueue:
    """Ordered queue preserving repeated characters and explicit separators.

    ``current`` and ``peek`` point at the first non-completed item.  ``start``
    marks it PLAYING.  ``advance`` completes it and starts the next item;
    ``pop`` completes and returns it.  ``reset`` restores all item states and
    ``clear`` removes the queue.
    """

    def __init__(self, resolver: Core28Resolver | None = None, *, gap_duration_ms: int = GAP_DURATION_MS) -> None:
        if gap_duration_ms < 0:
            raise ValueError("gap_duration_ms must be non-negative")
        self.resolver = resolver or Core28Resolver()
        self.gap_duration_ms = gap_duration_ms
        self._items: list[PlaybackQueueItem] = []
        self._cursor = 0
        self.last_unsupported: tuple[UnsupportedTextIssue, ...] = ()

    @property
    def items(self) -> tuple[PlaybackQueueItem, ...]:
        return tuple(self._items)

    @property
    def current(self) -> PlaybackQueueItem | None:
        return self._items[self._cursor] if self._cursor < len(self._items) else None

    def current_item(self) -> PlaybackQueueItem | None:
        """Method alias for callers that prefer callable access."""

        return self.current

    def peek(self) -> PlaybackQueueItem | None:
        return self.current

    @property
    def remaining(self) -> int:
        return len(self._items) - self._cursor

    @property
    def completed(self) -> int:
        return sum(item.state == QueueState.COMPLETED for item in self._items)

    @property
    def is_complete(self) -> bool:
        return self._cursor >= len(self._items) and bool(self._items)

    def completed_items(self) -> tuple[PlaybackQueueItem, ...]:
        return tuple(item for item in self._items if item.state == QueueState.COMPLETED)

    def remaining_items(self) -> tuple[PlaybackQueueItem, ...]:
        return tuple(self._items[self._cursor :])

    def _append(self, item: PlaybackQueueItem) -> PlaybackQueueItem:
        self._items.append(item)
        return item

    def enqueue_character(
        self,
        character: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
    ) -> PlaybackQueueItem:
        resolution = self.resolver.resolve_character(character, mode=mode, rng_seed=rng_seed)
        return self._append(PlaybackQueueItem.from_resolution(resolution))

    def enqueue_gap(self, character: str, *, duration_ms: int | None = None) -> PlaybackQueueItem:
        return self._append(
            PlaybackQueueItem.neutral_gap(
                character, duration_ms=self.gap_duration_ms if duration_ms is None else duration_ms
            )
        )

    def _plan_text(
        self,
        text: str,
        *,
        mode: str,
        rng_seed: int | None,
    ) -> tuple[list[PlaybackQueueItem], tuple[UnsupportedTextIssue, ...]]:
        planned: list[PlaybackQueueItem] = []
        issues: list[UnsupportedTextIssue] = []
        position = 0
        while position < len(text):
            compound = unsupported_sequence_at(text, position)
            if compound is not None:
                issues.append(
                    UnsupportedTextIssue(
                        position,
                        compound,
                        f"Unsupported Core-28 sequence: {compound}",
                    )
                )
                position += len(compound)
                continue
            character = text[position]
            if _is_separator(character):
                planned.append(PlaybackQueueItem.neutral_gap(character, duration_ms=self.gap_duration_ms))
                position += 1
                continue
            try:
                resolution = self.resolver.resolve_character(character, mode=mode, rng_seed=rng_seed)
            except UnsupportedCharacterError as error:
                issues.append(UnsupportedTextIssue(position, character, str(error)))
                position += 1
                continue
            planned.append(PlaybackQueueItem.from_resolution(resolution))
            position += 1
        return planned, tuple(issues)

    def enqueue_text(
        self,
        text: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
        unsupported_policy: Literal["reject", "report"] = "reject",
    ) -> tuple[PlaybackQueueItem, ...]:
        """Enqueue text atomically, or report unsupported positions explicitly.

        ``reject`` leaves the queue unchanged and raises ``UnsupportedTextError``.
        ``report`` enqueues supported characters and separator gaps, returns the
        created items, and exposes unsupported positions through
        ``last_unsupported``.  Unsupported characters are never silently lost.
        """

        if unsupported_policy not in {"reject", "report"}:
            raise ValueError("unsupported_policy must be 'reject' or 'report'")
        planned, issues = self._plan_text(text, mode=mode, rng_seed=rng_seed)
        self.last_unsupported = issues
        if issues and unsupported_policy == "reject":
            raise UnsupportedTextError(issues)
        self._items.extend(planned)
        return tuple(planned)

    def start(self) -> PlaybackQueueItem | None:
        item = self.current
        if item is not None and item.state == QueueState.PENDING:
            item.state = QueueState.PLAYING
        return item

    def advance(self) -> PlaybackQueueItem | None:
        item = self.current
        if item is None:
            return None
        if item.state not in {QueueState.PENDING, QueueState.PLAYING}:
            self._cursor += 1
            return self.advance()
        item.state = QueueState.COMPLETED
        self._cursor += 1
        return self.start()

    def pop(self) -> PlaybackQueueItem | None:
        item = self.current
        if item is None:
            return None
        if item.state in {QueueState.PENDING, QueueState.PLAYING}:
            item.state = QueueState.COMPLETED
        self._cursor += 1
        return item

    def fail(self, reason: str, *, unavailable: bool = False) -> PlaybackQueueItem | None:
        item = self.current
        if item is None:
            return None
        item.state = QueueState.UNAVAILABLE if unavailable else QueueState.FAILED
        item.failure_reason = str(reason)
        self._cursor += 1
        return item

    def clear(self) -> None:
        self._items.clear()
        self._cursor = 0
        self.last_unsupported = ()

    def reset(self) -> None:
        for item in self._items:
            item.state = QueueState.PENDING
            item.failure_reason = None
        self._cursor = 0
        self.last_unsupported = ()


# Renderer integrations may use the conceptual name from the task contract;
# the concrete object remains the same plain data item.
ResolvedPlaybackItem = PlaybackQueueItem
