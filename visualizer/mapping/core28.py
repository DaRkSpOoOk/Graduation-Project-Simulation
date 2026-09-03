"""Core-28 mapping backed by the repository's authoritative label manifest."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.dataset.core28 import load_label_records, validate_core28_records

from visualizer.catalog import Core28ExemplarCatalog, ExemplarEntry, SequenceDescriptor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "visualizer" / "catalog" / "core28_exemplars.json"
UNSUPPORTED_CORE28_SEQUENCES = ("ئـ", "لا", "ال")


class UnsupportedCharacterError(ValueError):
    """Raised when a character is not one of the frozen Core-28 classes."""

    def __init__(self, character: str, *, position: int | None = None) -> None:
        self.character = character
        self.position = position
        location = "" if position is None else f" at position {position}"
        super().__init__(f"Unsupported Core-28 character{location}: {character}")


@dataclass(frozen=True, slots=True)
class Core28Label:
    character: str
    sign_id: str
    label_index: int
    label_en: str


@dataclass(frozen=True, slots=True)
class CharacterResolution:
    """A complete mapping result ready for queue construction."""

    character: str
    sign_id: str
    label_index: int
    sample_id: str
    signer_id: str
    sequence_descriptor: SequenceDescriptor
    mode: str
    selection_score: float | None = None

    @property
    def signer(self) -> str:
        """Short alias used by UI/control code."""

        return self.signer_id

    @property
    def descriptor(self) -> SequenceDescriptor:
        return self.sequence_descriptor

    @property
    def sequence_path_or_descriptor(self) -> SequenceDescriptor:
        return self.sequence_descriptor

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "label_index": self.label_index,
            "mode": self.mode,
            "sample_id": self.sample_id,
            "selection_score": self.selection_score,
            "sequence_descriptor": self.sequence_descriptor.to_dict(),
            "sign_id": self.sign_id,
            "signer_id": self.signer_id,
        }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def unsupported_sequence_at(text: str, position: int) -> str | None:
    """Return a frozen multi-codepoint token that must not be decomposed."""

    for sequence in UNSUPPORTED_CORE28_SEQUENCES:
        if text.startswith(sequence, position):
            return sequence
    return None


class Core28Mapping:
    """Read and validate labels without depending on filesystem ordering."""

    def __init__(self, labels_path: str | Path = DEFAULT_LABELS_PATH) -> None:
        self.labels_path = Path(labels_path)
        records = validate_core28_records(load_label_records(self.labels_path))
        self.labels = tuple(
            Core28Label(
                character=unicodedata.normalize("NFC", record.label_ar).strip(),
                sign_id=f"{record.sign_id:04d}",
                label_index=index,
                label_en=record.label_en,
            )
            for index, record in enumerate(records)
        )
        self._validate_source_indices()
        self._by_character = {label.character: label for label in self.labels}
        self._by_sign_id = {label.sign_id: label for label in self.labels}

    def _validate_source_indices(self) -> None:
        """Check extra CSV contract columns when the canonical CSV provides them."""

        if self.labels_path.suffix.lower() != ".csv":
            return
        with self.labels_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        core_rows = [row for row in rows if _truthy(row.get("is_core28", "true"))]
        if "is_core28" not in (rows[0] if rows else {}):
            core_ids = {f"{value:04d}" for value in range(32, 60)}
            core_rows = [row for row in rows if str(row.get("sign_id", "")).zfill(4) in core_ids]
        if len(core_rows) != 28:
            raise ValueError(f"labels manifest must identify exactly 28 Core-28 rows, got {len(core_rows)}")
        by_id = {str(row.get("sign_id", "")).zfill(4): row for row in core_rows}
        for index, label in enumerate(self.labels):
            row = by_id.get(label.sign_id)
            if row is None:
                raise ValueError(f"labels manifest has no Core-28 row for {label.sign_id}")
            declared = row.get("label_index", "").strip()
            if declared and int(declared) != index:
                raise ValueError(
                    f"label_index mismatch for {label.sign_id}: source={declared}, expected={index}"
                )

    def resolve_label(self, character: str) -> Core28Label:
        normalized = unicodedata.normalize("NFC", str(character))
        if len(normalized) != 1:
            raise UnsupportedCharacterError(str(character))
        try:
            return self._by_character[normalized]
        except KeyError as error:
            raise UnsupportedCharacterError(normalized) from error

    def label_for_sign_id(self, sign_id: str) -> Core28Label:
        try:
            return self._by_sign_id[str(sign_id).zfill(4)]
        except KeyError as error:
            raise ValueError(f"not a Core-28 SignID: {sign_id!r}") from error

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "label_ar": label.character,
                "label_en_if_available": label.label_en,
                "label_index": label.label_index,
                "sign_id": label.sign_id,
            }
            for label in self.labels
        ]


class Core28Resolver:
    """Resolve Unicode characters through labels and the selected catalog entry."""

    def __init__(
        self,
        *,
        labels_path: str | Path = DEFAULT_LABELS_PATH,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    ) -> None:
        self.mapping = Core28Mapping(labels_path)
        self.catalog = Core28ExemplarCatalog.load(catalog_path)
        self.catalog.assert_matches_labels(self.mapping.as_dicts())

    def resolve_character(
        self,
        character: str,
        *,
        mode: str = "canonical",
        rng_seed: int | None = None,
    ) -> CharacterResolution:
        label = self.mapping.resolve_label(character)
        entry: ExemplarEntry = self.catalog.select(label.sign_id, mode=mode, rng_seed=rng_seed)
        if entry.character != label.character or entry.label_index != label.label_index:
            raise ValueError(f"catalog mapping disagrees with labels for {label.sign_id}")
        return CharacterResolution(
            character=label.character,
            sign_id=label.sign_id,
            label_index=label.label_index,
            sample_id=entry.sample_id,
            signer_id=entry.signer_id,
            sequence_descriptor=entry.descriptor,
            mode=mode,
            selection_score=entry.score,
        )

    def supported_characters(self) -> tuple[str, ...]:
        return tuple(label.character for label in self.mapping.labels)


def resolve_character(
    character: str,
    *,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    mode: str = "canonical",
    rng_seed: int | None = None,
) -> CharacterResolution:
    """Convenience API for ``Arabic character -> SignID -> exemplar``."""

    return Core28Resolver(labels_path=labels_path, catalog_path=catalog_path).resolve_character(
        character, mode=mode, rng_seed=rng_seed
    )
