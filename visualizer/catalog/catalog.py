"""Load and select the deterministic Core-28 exemplar catalog."""

from __future__ import annotations

import hashlib
import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .descriptor import SequenceDescriptor

CATALOG_VERSION = "task007b_core28_exemplars_v1"
_SIGNER_MODE = re.compile(r"^signer([0-9]{2})$")


class CatalogError(ValueError):
    """Raised when a catalog is missing, malformed, or contract-inconsistent."""


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise CatalogError(f"catalog field {field!r} is not an integer: {value!r}") from error


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CatalogError(f"catalog field {field!r} is not numeric: {value!r}") from error


@dataclass(frozen=True, slots=True)
class ExemplarEntry:
    """A selected or selectable sample, independent of rendering code."""

    character: str
    sign_id: str
    label_index: int
    sample_id: str
    signer_id: str
    official_partition: str
    repetition_id: str
    sequence_length: int
    descriptor: SequenceDescriptor
    score: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    selection_reason: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExemplarEntry":
        try:
            descriptor_payload = payload["sequence_descriptor"]
            descriptor = SequenceDescriptor(
                sample_id=str(descriptor_payload["sample_id"]),
                run_root=str(descriptor_payload["run_root"]),
                pose_relative_path=str(descriptor_payload["pose_relative_path"]),
                tracking_relative_path=str(descriptor_payload["tracking_relative_path"]),
                kinematics_relative_path=str(descriptor_payload["kinematics_relative_path"]),
                virtual_glove_relative_path=str(descriptor_payload["virtual_glove_relative_path"]),
                sequence_length=_as_int(descriptor_payload["sequence_length"], "sequence_length"),
                source_relative_path=str(descriptor_payload.get("source_relative_path", "")),
                source_sha256=str(descriptor_payload.get("source_sha256", "")),
                manifest_sha256=str(descriptor_payload.get("manifest_sha256", "")),
                signer_id=str(descriptor_payload.get("signer_id", "")),
                official_partition=str(descriptor_payload.get("official_partition", "")),
                repetition_id=str(descriptor_payload.get("repetition_id", "")),
            )
            sample_id = str(payload["sample_id"])
            if descriptor.sample_id != sample_id:
                raise CatalogError(f"descriptor/sample_id mismatch for {sample_id!r}")
            score_value = payload.get("score")
            score = None if score_value is None else _as_float(score_value, "score")
            return cls(
                character=str(payload["character"]),
                sign_id=str(payload["sign_id"]),
                label_index=_as_int(payload["label_index"], "label_index"),
                sample_id=sample_id,
                signer_id=str(payload.get("signer_id", descriptor.signer_id)),
                official_partition=str(payload.get("official_partition", descriptor.official_partition)),
                repetition_id=str(payload.get("repetition_id", descriptor.repetition_id)),
                sequence_length=_as_int(payload["sequence_length"], "sequence_length"),
                descriptor=descriptor,
                score=score,
                metrics=dict(payload.get("metrics", {})),
                selection_reason=str(payload.get("selection_reason", "")),
            )
        except KeyError as error:
            raise CatalogError(f"catalog exemplar is missing {error.args[0]!r}") from error

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "character": self.character,
            "label_index": self.label_index,
            "metrics": dict(self.metrics),
            "official_partition": self.official_partition,
            "repetition_id": self.repetition_id,
            "sample_id": self.sample_id,
            "score": self.score,
            "selection_reason": self.selection_reason,
            "sequence_descriptor": self.descriptor.to_dict(),
            "sequence_length": self.sequence_length,
            "sign_id": self.sign_id,
            "signer_id": self.signer_id,
        }
        return payload


def _candidate_entry(payload: Mapping[str, Any], source: Mapping[str, Any]) -> ExemplarEntry:
    """Expand a compact candidate reference into the common entry shape."""

    if "sequence_descriptor" in payload:
        return ExemplarEntry.from_dict(payload)
    try:
        sample_id = str(payload["sample_id"])
        run_root = str(payload.get("run_root", source.get("run_root", "")))
        descriptor = {
            "kinematics_relative_path": f"kinematics/{sample_id}/hand_kinematics.npz",
            "manifest_sha256": str(payload.get("manifest_sha256", source.get("virtual_glove_manifest_sha256", ""))),
            "official_partition": str(payload.get("official_partition", "")),
            "pose_relative_path": f"pose/raw/{sample_id}/wilor_raw.npz",
            "repetition_id": str(payload.get("repetition_id", "")),
            "run_root": run_root,
            "sample_id": sample_id,
            "sequence_length": payload["sequence_length"],
            "signer_id": str(payload.get("signer_id", "")),
            "source_relative_path": str(payload.get("source_relative_path", "")),
            "source_sha256": str(payload.get("source_sha256", "")),
            "tracking_relative_path": f"tracking/{sample_id}/wilor_tracked.npz",
            "virtual_glove_relative_path": str(payload["virtual_glove_relative_path"]),
        }
    except KeyError as error:
        raise CatalogError(f"compact candidate is missing {error.args[0]!r}") from error
    expanded = dict(payload)
    expanded["sequence_descriptor"] = descriptor
    expanded.setdefault("character", str(payload.get("character", "")))
    expanded.setdefault("sign_id", str(payload.get("sign_id", "")))
    expanded.setdefault("label_index", payload.get("label_index", -1))
    return ExemplarEntry.from_dict(expanded)


def _validate_unique(entries: Iterable[ExemplarEntry], context: str) -> dict[str, ExemplarEntry]:
    by_sign = {}
    by_char = {}
    for entry in entries:
        if entry.sign_id in by_sign:
            raise CatalogError(f"duplicate sign_id {entry.sign_id!r} in {context}")
        if entry.character in by_char:
            raise CatalogError(f"duplicate character {entry.character!r} in {context}")
        by_sign[entry.sign_id] = entry
        by_char[entry.character] = entry
    return by_sign


class Core28ExemplarCatalog:
    """Validated catalog with canonical, signer-specific and seeded selection."""

    def __init__(
        self,
        *,
        source: Mapping[str, Any],
        selection_policy: Mapping[str, Any],
        entries: Iterable[ExemplarEntry],
        signer_exemplars: Mapping[str, Iterable[ExemplarEntry]],
        candidate_index: Mapping[str, Iterable[ExemplarEntry]],
    ) -> None:
        self.source = dict(source)
        self.selection_policy = dict(selection_policy)
        self.entries = tuple(sorted(entries, key=lambda entry: entry.label_index))
        if len(self.entries) != 28:
            raise CatalogError(f"canonical catalog must contain exactly 28 entries, got {len(self.entries)}")
        _validate_unique(self.entries, "canonical_exemplars")
        if [entry.label_index for entry in self.entries] != list(range(28)):
            raise CatalogError("canonical label_index values must be exactly 0..27")
        if any(entry.sequence_length != entry.descriptor.sequence_length for entry in self.entries):
            raise CatalogError("canonical entry and descriptor sequence lengths disagree")
        self._by_sign_id = {entry.sign_id: entry for entry in self.entries}
        self._by_character = {entry.character: entry for entry in self.entries}

        self.signer_exemplars = {
            str(signer): tuple(sorted(values, key=lambda entry: entry.label_index))
            for signer, values in signer_exemplars.items()
        }
        for signer, values in self.signer_exemplars.items():
            if values:
                _validate_unique(values, f"signer_exemplars[{signer}]")
                if any(entry.sequence_length != entry.descriptor.sequence_length for entry in values):
                    raise CatalogError(f"signer_exemplars[{signer}] has a sequence length mismatch")

        self.candidate_index = {
            str(sign_id): tuple(sorted(values, key=lambda entry: entry.sample_id))
            for sign_id, values in candidate_index.items()
        }
        for sign_id, values in self.candidate_index.items():
            if not values:
                raise CatalogError(f"candidate_index[{sign_id!r}] is empty")
            if any(entry.sign_id != sign_id for entry in values):
                raise CatalogError(f"candidate_index[{sign_id!r}] contains a different sign_id")
            if len({entry.sample_id for entry in values}) != len(values):
                raise CatalogError(f"candidate_index[{sign_id!r}] contains duplicate sample_id")
            if any(entry.sequence_length != entry.descriptor.sequence_length for entry in values):
                raise CatalogError(f"candidate_index[{sign_id!r}] has a sequence length mismatch")

        missing_pools = sorted(set(self._by_sign_id) - set(self.candidate_index))
        if missing_pools:
            raise CatalogError(f"catalog has no candidate pool for {missing_pools}")
        extra_pools = sorted(set(self.candidate_index) - set(self._by_sign_id))
        if extra_pools:
            raise CatalogError(f"catalog has non-Core-28 candidate pools for {extra_pools}")

    @classmethod
    def load(cls, path: str | Path) -> "Core28ExemplarCatalog":
        source_path = Path(path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise CatalogError(f"cannot read catalog {source_path}: {error}") from error
        except json.JSONDecodeError as error:
            raise CatalogError(f"catalog is not valid JSON: {source_path}: {error}") from error
        if payload.get("catalog_version") != CATALOG_VERSION:
            raise CatalogError(
                f"unsupported catalog_version {payload.get('catalog_version')!r}; expected {CATALOG_VERSION!r}"
            )
        try:
            entries = [ExemplarEntry.from_dict(item) for item in payload["entries"]]
            signer_exemplars = {
                str(signer): [ExemplarEntry.from_dict(item) for item in values]
                for signer, values in payload.get("signer_exemplars", {}).items()
            }
            candidate_index = {
                str(sign_id): [_candidate_entry(item, payload.get("source", {})) for item in values]
                for sign_id, values in payload["candidate_index"].items()
            }
        except KeyError as error:
            raise CatalogError(f"catalog is missing {error.args[0]!r}") from error
        return cls(
            source=payload.get("source", {}),
            selection_policy=payload.get("selection_policy", {}),
            entries=entries,
            signer_exemplars=signer_exemplars,
            candidate_index=candidate_index,
        )

    def assert_matches_labels(self, labels: Iterable[Mapping[str, Any]]) -> None:
        """Ensure the catalog cannot silently use a different Core-28 mapping."""

        expected = {
            str(item["sign_id"]): (str(item["label_ar"]), int(item["label_index"])) for item in labels
        }
        actual = {entry.sign_id: (entry.character, entry.label_index) for entry in self.entries}
        if actual != expected:
            raise CatalogError("catalog labels do not exactly match the authoritative Core-28 labels manifest")

    def entry_for_sign_id(self, sign_id: str) -> ExemplarEntry:
        try:
            return self._by_sign_id[sign_id]
        except KeyError as error:
            raise CatalogError(f"no Core-28 exemplar for SignID {sign_id!r}") from error

    def _seed_for(self, rng_seed: int, sign_id: str) -> int:
        material = f"task007b:{int(rng_seed)}:{sign_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def select(self, sign_id: str, mode: str = "canonical", rng_seed: int | None = None) -> ExemplarEntry:
        """Select an exemplar without filesystem or insertion-order dependence."""

        if mode == "canonical":
            return self.entry_for_sign_id(sign_id)
        signer_match = _SIGNER_MODE.fullmatch(mode)
        if signer_match:
            signer = signer_match.group(1)
            for entry in self.signer_exemplars.get(signer, ()):
                if entry.sign_id == sign_id:
                    return entry
            raise CatalogError(f"no exemplar for SignID {sign_id!r} from signer {signer!r}")
        if mode == "random":
            if rng_seed is None:
                raise CatalogError("random exemplar mode requires an explicit rng_seed")
            candidates = self.candidate_index.get(sign_id, ())
            if not candidates:
                raise CatalogError(f"no random candidate pool for SignID {sign_id!r}")
            return candidates[random.Random(self._seed_for(rng_seed, sign_id)).randrange(len(candidates))]
        raise CatalogError(
            f"unsupported exemplar mode {mode!r}; expected canonical, random, or signerNN"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": {
                sign_id: [entry.to_dict() for entry in values]
                for sign_id, values in sorted(self.candidate_index.items())
            },
            "catalog_version": CATALOG_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
            "selection_policy": self.selection_policy,
            "signer_exemplars": {
                signer: [entry.to_dict() for entry in values]
                for signer, values in sorted(self.signer_exemplars.items())
            },
            "source": self.source,
        }


def write_catalog(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic catalog JSON payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_catalog_csv(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a compact, canonical-only CSV companion to the JSON catalog."""

    fields = (
        "character", "sign_id", "label_index", "sample_id", "signer_id",
        "official_partition", "repetition_id", "sequence_length", "selection_score",
        "bend_valid_fraction", "spread_valid_fraction", "imu_valid_fraction",
        "tracking_quality_fraction", "geometry_available", "pose_relative_path",
        "tracking_relative_path", "kinematics_relative_path", "virtual_glove_relative_path",
    )
    entries = sorted(payload.get("entries", ()), key=lambda entry: int(entry["label_index"]))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            metrics = entry.get("metrics", {})
            descriptor = entry.get("sequence_descriptor", {})
            writer.writerow({
                "character": entry.get("character", ""),
                "sign_id": entry.get("sign_id", ""),
                "label_index": entry.get("label_index", ""),
                "sample_id": entry.get("sample_id", ""),
                "signer_id": entry.get("signer_id", ""),
                "official_partition": entry.get("official_partition", ""),
                "repetition_id": entry.get("repetition_id", ""),
                "sequence_length": entry.get("sequence_length", ""),
                "selection_score": entry.get("score", ""),
                "bend_valid_fraction": metrics.get("bend_valid_fraction", ""),
                "spread_valid_fraction": metrics.get("spread_valid_fraction", ""),
                "imu_valid_fraction": metrics.get("imu_valid_fraction", ""),
                "tracking_quality_fraction": metrics.get("tracking_quality_fraction", ""),
                "geometry_available": metrics.get("geometry_available", ""),
                "pose_relative_path": descriptor.get("pose_relative_path", ""),
                "tracking_relative_path": descriptor.get("tracking_relative_path", ""),
                "kinematics_relative_path": descriptor.get("kinematics_relative_path", ""),
                "virtual_glove_relative_path": descriptor.get("virtual_glove_relative_path", ""),
            })
