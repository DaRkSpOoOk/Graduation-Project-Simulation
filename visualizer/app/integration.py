"""Renderer-neutral integration between the Core-28 queue and TASK-007A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from visualizer.contract import PlaybackSequence
from visualizer.loader import ArtifactValidationError, load_sequence
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueue, PlaybackQueueItem, QueueState

if TYPE_CHECKING:
    from visualizer.recognition import RecognizerAdapter


class VisualizerIntegrationError(ValueError):
    """Raised when a queue descriptor cannot be consumed by the renderer."""


def load_sequence_for_item(
    item: PlaybackQueueItem,
    *,
    run_root: str | Path,
    manifest_path: str | Path | None = None,
) -> PlaybackSequence | None:
    """Resolve one queue sign item through the TASK-007A loader.

    Neutral gaps intentionally return ``None``: they are presentation
    transitions and do not correspond to a TASK-008 sample or synthetic frame.
    The descriptor's stored run root is not used as an authority; the caller's
    configured run root is used so a catalog can remain portable across hosts.
    """

    if item.item_type == "gap":
        return None
    if item.item_type != "sign":
        raise VisualizerIntegrationError(f"unsupported queue item type: {item.item_type!r}")
    descriptor = item.sequence_descriptor
    if descriptor is None or item.sample_id is None:
        raise VisualizerIntegrationError(f"sign item {item.character!r} has no sequence descriptor")
    if descriptor.sample_id != item.sample_id:
        raise VisualizerIntegrationError(
            f"queue item/descriptor sample mismatch: {item.sample_id!r} != {descriptor.sample_id!r}"
        )
    try:
        sequence = load_sequence(run_root, descriptor.sample_id, manifest_path=manifest_path)
    except ArtifactValidationError as exc:
        raise VisualizerIntegrationError(
            f"cannot load queued sample {descriptor.sample_id!r}: {exc}"
        ) from exc
    if len(sequence) != descriptor.sequence_length:
        raise VisualizerIntegrationError(
            f"sequence length mismatch for {descriptor.sample_id!r}: "
            f"descriptor={descriptor.sequence_length}, loaded={len(sequence)}"
        )
    if item.sign_id is not None and sequence.metadata.get("manifest", {}).get("sign_id") not in {
        None,
        "",
        item.sign_id,
    }:
        raise VisualizerIntegrationError(f"SignID mismatch for queued sample {descriptor.sample_id!r}")
    if item.label_index is not None and sequence.label_index not in {None, item.label_index}:
        raise VisualizerIntegrationError(f"label-index mismatch for queued sample {descriptor.sample_id!r}")
    if not any(hand.present for frame in sequence.frames for hand in frame.hands):
        raise VisualizerIntegrationError(f"queued sample {descriptor.sample_id!r} has no geometry")
    return sequence


@dataclass(frozen=True, slots=True)
class HeadlessPlaybackResult:
    """Compact result from deterministic queue traversal for tests/CI."""

    requested_text: str
    played: tuple[dict[str, Any], ...]
    completed_indices: tuple[int, ...]
    queue_complete: bool


@dataclass(frozen=True, slots=True)
class HeadlessRecognitionPlaybackResult:
    """Queue traversal plus demo-only sequence-level recognizer results."""

    requested_text: str
    played: tuple[dict[str, Any], ...]
    completed_indices: tuple[int, ...]
    queue_complete: bool
    recognition_enabled: bool
    checkpoint_metadata: dict[str, Any] | None


class QueuePlaybackSession:
    """Queue state plus on-demand loading of the currently playing sample."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        resolver: Core28Resolver | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.queue = PlaybackQueue(resolver)
        self.current_item: PlaybackQueueItem | None = None
        self.current_sequence: PlaybackSequence | None = None
        self.completed_items: list[PlaybackQueueItem] = []

    def enqueue_text(
        self,
        text: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
        unsupported_policy: str = "reject",
    ) -> tuple[PlaybackQueueItem, ...]:
        return self.queue.enqueue_text(
            text,
            mode=mode,
            rng_seed=rng_seed,
            unsupported_policy=unsupported_policy,  # type: ignore[arg-type]
        )

    def enqueue_character(
        self,
        character: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
    ) -> PlaybackQueueItem:
        return self.queue.enqueue_character(character, mode=mode, rng_seed=rng_seed)

    def _activate(self, item: PlaybackQueueItem) -> PlaybackQueueItem:
        self.current_item = item
        self.current_sequence = load_sequence_for_item(
            item,
            run_root=self.run_root,
            manifest_path=self.manifest_path,
        )
        return item

    def start(self) -> PlaybackQueueItem | None:
        """Mark the current item playing and load it if it is a sign."""

        item = self.queue.current
        if item is None:
            return None
        if item.state == QueueState.PENDING:
            item = self.queue.start()
            if item is None:  # pragma: no cover - guarded by current above
                return None
            return self._activate(item)
        if self.current_item is not item:
            return self._activate(item)
        return item

    def complete_current(self) -> PlaybackQueueItem | None:
        """Complete the current item and activate the next queue item."""

        previous = self.queue.current
        if previous is None:
            return None
        if previous.state == QueueState.PENDING:
            self.queue.start()
        next_item = self.queue.advance()
        self.completed_items.append(previous)
        if next_item is not None:
            self._activate(next_item)
        return next_item

    def reset(self) -> None:
        self.queue.reset()
        self.current_item = None
        self.current_sequence = None
        self.completed_items.clear()

    def clear(self) -> None:
        self.queue.clear()
        self.current_item = None
        self.current_sequence = None
        self.completed_items.clear()


def _item_index(items: tuple[PlaybackQueueItem, ...], item: PlaybackQueueItem) -> int:
    for index, candidate in enumerate(items):
        if candidate is item:
            return index
    raise VisualizerIntegrationError("queue item is not present in its queue")


def run_headless_queue(
    text: str,
    *,
    run_root: str | Path,
    resolver: Core28Resolver | None = None,
    manifest_path: str | Path | None = None,
    mode: str = "canonical",
    rng_seed: int | None = None,
) -> HeadlessPlaybackResult:
    """Traverse every queued item, loading signs and never fabricating gaps."""

    session = QueuePlaybackSession(
        run_root=run_root,
        resolver=resolver,
        manifest_path=manifest_path,
    )
    items = session.enqueue_text(text, mode=mode, rng_seed=rng_seed)
    if session.queue.last_unsupported:
        raise VisualizerIntegrationError(
            "headless queue input contains unsupported characters: "
            + "; ".join(issue.character for issue in session.queue.last_unsupported)
        )

    played: list[dict[str, Any]] = []
    completed_indices: list[int] = []
    session.start()
    while session.queue.current is not None:
        item = session.queue.current
        index = _item_index(items, item)
        sequence = session.current_sequence
        played.append(
            {
                "index": index,
                "item_type": item.item_type,
                "character": item.character,
                "sign_id": item.sign_id,
                "sample_id": item.sample_id,
                "sequence_length": len(sequence) if sequence is not None else None,
                "has_geometry": (
                    any(hand.present for frame in sequence.frames for hand in frame.hands)
                    if sequence is not None
                    else False
                ),
            }
        )
        session.complete_current()
        completed_indices.append(index)
    return HeadlessPlaybackResult(
        requested_text=text,
        played=tuple(played),
        completed_indices=tuple(completed_indices),
        queue_complete=session.queue.is_complete,
    )


def run_headless_recognizer_queue(
    text: str,
    *,
    run_root: str | Path,
    recognition_adapter: RecognizerAdapter | None = None,
    recognition_error: str | None = None,
    resolver: Core28Resolver | None = None,
    manifest_path: str | Path | None = None,
    mode: str = "canonical",
    rng_seed: int | None = None,
) -> HeadlessRecognitionPlaybackResult:
    """Traverse a queue and attach one cached sequence-level prediction per sign.

    This is an engineering/demo path.  It deliberately reports predictions
    without calculating accuracy or changing the queue's expected labels.
    Neutral gaps are recorded but never sent to the recognizer.
    """

    # Keep the existing visualization-only module importable in environments
    # without the optional TASK-009B/PyTorch dependency.
    from visualizer.recognition import RecognitionController

    session = QueuePlaybackSession(
        run_root=run_root,
        resolver=resolver,
        manifest_path=manifest_path,
    )
    controller = RecognitionController(
        recognition_adapter,
        disabled_reason=recognition_error,
    )
    items = session.enqueue_text(text, mode=mode, rng_seed=rng_seed)
    if session.queue.last_unsupported:
        raise VisualizerIntegrationError(
            "headless queue input contains unsupported characters: "
            + "; ".join(issue.character for issue in session.queue.last_unsupported)
        )

    played: list[dict[str, Any]] = []
    completed_indices: list[int] = []
    session.start()
    while session.queue.current is not None:
        item = session.queue.current
        index = _item_index(items, item)
        sequence = session.current_sequence
        recognition = controller.result_for(item)
        played.append(
            {
                "index": index,
                "item_type": item.item_type,
                "character": item.character,
                "sign_id": item.sign_id,
                "sample_id": item.sample_id,
                "sequence_length": len(sequence) if sequence is not None else None,
                "has_geometry": (
                    any(hand.present for frame in sequence.frames for hand in frame.hands)
                    if sequence is not None
                    else False
                ),
                "recognition": recognition.to_dict() if recognition is not None else None,
            }
        )
        session.complete_current()
        completed_indices.append(index)
    return HeadlessRecognitionPlaybackResult(
        requested_text=text,
        played=tuple(played),
        completed_indices=tuple(completed_indices),
        queue_complete=session.queue.is_complete,
        recognition_enabled=controller.enabled,
        checkpoint_metadata=(controller.metadata.to_dict() if controller.metadata is not None else None),
    )
