"""Non-graphical Arabic Core-28 keyboard specification."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any

from visualizer.mapping import Core28Mapping, unsupported_sequence_at
from visualizer.queue import PlaybackQueue, UnsupportedTextIssue


@dataclass(frozen=True, slots=True)
class KeyboardKey:
    character: str
    sign_id: str
    label_index: int
    label_en: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "label_en": self.label_en,
            "label_index": self.label_index,
            "sign_id": self.sign_id,
        }


@dataclass(frozen=True, slots=True)
class TextValidation:
    supported: tuple[tuple[int, str], ...]
    separators: tuple[tuple[int, str], ...]
    unsupported: tuple[UnsupportedTextIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.unsupported


class Core28Keyboard:
    """Keyboard data and queue actions; graphical buttons remain out of scope."""

    def __init__(self, mapping: Core28Mapping | None = None) -> None:
        self.mapping = mapping or Core28Mapping()
        self.keys = tuple(
            KeyboardKey(label.character, label.sign_id, label.label_index, label.label_en)
            for label in self.mapping.labels
        )

    @property
    def rtl_rows(self) -> tuple[tuple[KeyboardKey, ...], ...]:
        """Return three rows in right-to-left display order.

        The mapping/key tuple remains SignID/label-index order.  Reversing each
        visual row only describes placement for a future RTL-aware UI and never
        changes resolution semantics.
        """

        row_sizes = (10, 9, 9)
        rows: list[tuple[KeyboardKey, ...]] = []
        start = 0
        for size in row_sizes:
            rows.append(tuple(reversed(self.keys[start : start + size])))
            start += size
        return tuple(rows)

    def layout_spec(self) -> dict[str, Any]:
        """Return a renderer-agnostic RTL keyboard layout description."""

        return {
            "direction": "rtl",
            "rows": [[key.to_dict() for key in row] for row in self.rtl_rows],
            "supported_character_count": len(self.keys),
            "mapping_order": "authoritative label_index ascending",
        }

    def validate_text(self, text: str) -> TextValidation:
        supported: list[tuple[int, str]] = []
        separators: list[tuple[int, str]] = []
        unsupported: list[UnsupportedTextIssue] = []
        position = 0
        while position < len(text):
            compound = unsupported_sequence_at(text, position)
            if compound is not None:
                unsupported.append(
                    UnsupportedTextIssue(
                        position, compound, f"Unsupported Core-28 sequence: {compound}"
                    )
                )
                position += len(compound)
                continue
            character = text[position]
            if character.isspace() or unicodedata.category(character).startswith("P"):
                separators.append((position, character))
                position += 1
                continue
            try:
                self.mapping.resolve_label(character)
            except ValueError as error:
                unsupported.append(UnsupportedTextIssue(position, character, str(error)))
            else:
                supported.append((position, character))
            position += 1
        return TextValidation(tuple(supported), tuple(separators), tuple(unsupported))

    def enqueue_key(
        self,
        queue: PlaybackQueue,
        character: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
    ):
        return queue.enqueue_character(character, mode=mode, rng_seed=rng_seed)

    def enqueue_text(
        self,
        queue: PlaybackQueue,
        text: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
        unsupported_policy: str = "reject",
    ):
        return queue.enqueue_text(
            text,
            mode=mode,
            rng_seed=rng_seed,
            unsupported_policy=unsupported_policy,  # type: ignore[arg-type]
        )
