"""Signer-independent KArSL Core-28 split construction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from .core28 import CORE28_SIGN_IDS
from .manifest import CORE28_MANIFEST_FIELDS, validate_manifest_rows

FOLD_SIGNERS: tuple[str, ...] = ("01", "02", "03")
SPLIT_FIELDS: tuple[str, ...] = CORE28_MANIFEST_FIELDS + ("fold", "role")
ROLES = {"train", "validation", "test"}


def _sort_key(row: Mapping[str, str]) -> tuple[str, str]:
    return (row["sample_id"], row["source_relative_path"])


def _class_ids(rows: Iterable[Mapping[str, str]]) -> set[int]:
    return {int(row["sign_id"]) for row in rows}


def _ensure_remaining_class_coverage(
    train: list[dict[str, str]], validation: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Repair only a missing class in one official partition deterministically.

    The normal KArSL distribution should already contain each class in both
    official partitions.  If a source export does not, the smallest fallback
    is one deterministic move from the other role for that class.  A move is
    refused when it would leave the donor role without that class, so a bad or
    incomplete source cannot be hidden by the split builder.
    """

    train = list(train)
    validation = list(validation)
    for _ in range(2):
        changed = False
        for target, donor in ((train, validation), (validation, train)):
            target_ids = _class_ids(target)
            donor_ids = _class_ids(donor)
            for sign_id in CORE28_SIGN_IDS:
                if sign_id in target_ids:
                    continue
                candidates = sorted(
                    (row for row in donor if int(row["sign_id"]) == sign_id),
                    key=_sort_key,
                )
                if not candidates:
                    raise ValueError(
                        f"Cannot provide sign {sign_id:04d} in both train and validation"
                    )
                if len([row for row in donor if int(row["sign_id"]) == sign_id]) < 2:
                    raise ValueError(
                        f"Sign {sign_id:04d} has only one non-held-out sample; "
                        "cannot preserve train and validation coverage"
                    )
                moved = candidates[0]
                donor.remove(moved)
                target.append(moved)
                target_ids.add(sign_id)
                changed = True
        if not changed:
            break
    if _class_ids(train) != set(CORE28_SIGN_IDS) or _class_ids(validation) != set(CORE28_SIGN_IDS):
        raise ValueError("LOSO repair did not achieve complete train/validation Core-28 coverage")
    return train, validation


def build_loso_splits(
    manifest_rows: Iterable[Mapping[str, str]], held_out_signer: str
) -> list[dict[str, str]]:
    """Build one complete LOSO fold from the committed source manifest."""

    if held_out_signer not in FOLD_SIGNERS:
        raise ValueError(f"held_out_signer must be one of {FOLD_SIGNERS}: {held_out_signer!r}")
    rows = validate_manifest_rows(manifest_rows)
    if not rows:
        raise ValueError("Cannot build a populated split from an empty manifest")
    test = [row for row in rows if row["signer_id"] == held_out_signer]
    remaining = [row for row in rows if row["signer_id"] != held_out_signer]
    train = [row for row in remaining if row["official_partition"] == "train"]
    validation = [row for row in remaining if row["official_partition"] == "test"]
    train, validation = _ensure_remaining_class_coverage(train, validation)

    output: list[dict[str, str]] = []
    for role, selected in (("train", train), ("validation", validation), ("test", test)):
        for row in sorted(selected, key=_sort_key):
            output.append({**row, "fold": f"S{held_out_signer}", "role": role})
    validate_split_rows(output, manifest_rows=rows, held_out_signer=held_out_signer)
    return output


def validate_split_rows(
    split_rows: Iterable[Mapping[str, str]],
    *,
    manifest_rows: Iterable[Mapping[str, str]] | None = None,
    held_out_signer: str | None = None,
) -> list[dict[str, str]]:
    rows = [dict(row) for row in split_rows]
    if not rows:
        raise ValueError("Split is empty")
    missing = set(SPLIT_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"Split is missing columns: {sorted(missing)}")
    validate_manifest_rows(rows)
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample assignment in split")
    if any(row["role"] not in ROLES for row in rows):
        raise ValueError("Invalid split role")
    if held_out_signer is None:
        fold_values = {row["fold"] for row in rows}
        if len(fold_values) != 1 or not fold_values <= {f"S{s}" for s in FOLD_SIGNERS}:
            raise ValueError(f"Invalid or mixed fold values: {fold_values}")
        held_out_signer = next(iter(fold_values))[1:]
    if held_out_signer not in FOLD_SIGNERS:
        raise ValueError(f"Invalid held-out signer: {held_out_signer}")
    if any(row["fold"] != f"S{held_out_signer}" for row in rows):
        raise ValueError("Rows from multiple folds were supplied")
    for row in rows:
        if row["role"] == "test" and row["signer_id"] != held_out_signer:
            raise ValueError(f"Held-out signer leakage in test assignment: {row['sample_id']}")
        if row["role"] in {"train", "validation"} and row["signer_id"] == held_out_signer:
            raise ValueError(f"Held-out signer leakage in {row['role']}: {row['sample_id']}")
    if manifest_rows is not None:
        expected = validate_manifest_rows(manifest_rows)
        expected_ids = {row["sample_id"] for row in expected}
        if set(sample_ids) != expected_ids:
            raise ValueError("Split has missing or unknown samples compared with the source manifest")
    train_ids = _class_ids(row for row in rows if row["role"] == "train")
    validation_ids = _class_ids(row for row in rows if row["role"] == "validation")
    if train_ids != set(CORE28_SIGN_IDS) or validation_ids != set(CORE28_SIGN_IDS):
        raise ValueError("Every LOSO train and validation role must contain all Core-28 classes")
    return rows


def write_split_manifests(output_dir: str | Path, manifest_rows: Iterable[Mapping[str, str]]) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = validate_manifest_rows(manifest_rows)
    paths: dict[str, Path] = {}
    for held_out in FOLD_SIGNERS:
        split = build_loso_splits(rows, held_out)
        path = destination / f"karsl_core28_loso_s{held_out}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(split)
        paths[held_out] = path
    return paths


def write_split_headers(output_dir: str | Path) -> None:
    """Create trackable schema-only split templates before source discovery."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for signer in FOLD_SIGNERS:
        path = destination / f"karsl_core28_loso_s{signer}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerow(SPLIT_FIELDS)
