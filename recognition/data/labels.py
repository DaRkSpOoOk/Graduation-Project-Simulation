"""Authoritative Core-28 label mapping.

Read from the frozen label table, never reconstructed from directory ordering,
prediction order or the order classes happen to appear in a split.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .contract import NUM_CLASSES

DEFAULT_LABEL_TABLE = Path("datasets/manifests/karsl_core28_labels.csv")


@dataclass(frozen=True)
class Core28Label:
    label_index: int
    sign_id: str
    label_ar: str
    label_en: str


def load_label_table(path: str | Path = DEFAULT_LABEL_TABLE) -> dict[int, Core28Label]:
    """Map ``label_index`` -> the frozen Core-28 label record."""

    table: dict[int, Core28Label] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["label_index"])
            if index in table:
                raise ValueError(f"duplicate label_index {index} in {path}")
            table[index] = Core28Label(
                label_index=index,
                sign_id=row["sign_id"],
                label_ar=row["label_ar"],
                label_en=row.get("label_en_if_available", ""),
            )
    if len(table) != NUM_CLASSES or sorted(table) != list(range(NUM_CLASSES)):
        raise ValueError(
            f"{path} must define exactly {NUM_CLASSES} contiguous label_index values, "
            f"found {len(table)}"
        )
    return table


__all__ = ["DEFAULT_LABEL_TABLE", "Core28Label", "load_label_table"]
