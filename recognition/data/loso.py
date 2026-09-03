"""TASK-009A signer-independent fold selection.

Folds come from the frozen TASK-008B split files. Nothing is re-derived from
path ordering, and the held-out signer is checked against the rows rather than
assumed from the file name.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .sequence_dataset import SequenceContractError, SequenceRecord

FOLD_SIGNERS: tuple[str, ...] = ("01", "02", "03")
ROLES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class LosoFold:
    """One leave-one-signer-out fold, already verified against its own rows."""

    held_out_signer: str
    roles: dict[str, list[SequenceRecord]]

    def counts(self) -> dict[str, int]:
        return {role: len(self.roles[role]) for role in ROLES}

    def signers(self, role: str) -> set[str]:
        return {record.signer_id for record in self.roles[role]}

    def classes(self, role: str) -> set[str]:
        return {record.sign_id for record in self.roles[role]}


def split_file(splits_dir: str | Path, held_out_signer: str) -> Path:
    return Path(splits_dir) / f"karsl_core28_loso_s{held_out_signer}.csv"


def load_fold(
    splits_dir: str | Path,
    held_out_signer: str,
    records: Iterable[SequenceRecord],
) -> LosoFold:
    """Load one fold and enforce the signer-independence contract.

    Raises rather than warns: a fold with leakage would silently invalidate
    every downstream accuracy number, so it must never be loadable.
    """

    if held_out_signer not in FOLD_SIGNERS:
        raise SequenceContractError(f"unknown held-out signer {held_out_signer!r}")
    by_id = {record.sample_id: record for record in records}
    path = split_file(splits_dir, held_out_signer)
    if not path.is_file():
        raise SequenceContractError(f"missing frozen split file {path}")

    roles: dict[str, list[SequenceRecord]] = {role: [] for role in ROLES}
    assigned: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            role = row["role"]
            if role not in ROLES:
                raise SequenceContractError(f"{sample_id}: unknown split role {role!r}")
            if row.get("fold") and row["fold"] != f"S{held_out_signer}":
                raise SequenceContractError(
                    f"{sample_id}: fold {row['fold']!r} does not belong to S{held_out_signer}"
                )
            if sample_id in assigned:
                raise SequenceContractError(
                    f"{sample_id} appears in both {assigned[sample_id]!r} and {role!r}"
                )
            record = by_id.get(sample_id)
            if record is None:
                raise SequenceContractError(f"{sample_id} is in the split but not in the index")
            assigned[sample_id] = role
            roles[role].append(record)

    missing = set(by_id) - set(assigned)
    if missing:
        raise SequenceContractError(
            f"S{held_out_signer}: {len(missing)} indexed samples are absent from the split"
        )
    for role in ("train", "validation"):
        leaked = sorted(r.sample_id for r in roles[role] if r.signer_id == held_out_signer)
        if leaked:
            raise SequenceContractError(
                f"S{held_out_signer}: held-out signer leaks into {role} ({len(leaked)} samples)"
            )
    wrong_test = sorted(r.sample_id for r in roles["test"] if r.signer_id != held_out_signer)
    if wrong_test:
        raise SequenceContractError(
            f"S{held_out_signer}: test contains {len(wrong_test)} samples from another signer"
        )
    # Deterministic order, independent of file ordering and of dict iteration.
    for role in ROLES:
        roles[role].sort(key=lambda record: record.sample_id)
    return LosoFold(held_out_signer=held_out_signer, roles=roles)


def load_all_folds(
    splits_dir: str | Path, records: Iterable[SequenceRecord]
) -> dict[str, LosoFold]:
    materialized = list(records)
    return {signer: load_fold(splits_dir, signer, materialized) for signer in FOLD_SIGNERS}


__all__ = ["FOLD_SIGNERS", "ROLES", "LosoFold", "split_file", "load_fold", "load_all_folds"]
