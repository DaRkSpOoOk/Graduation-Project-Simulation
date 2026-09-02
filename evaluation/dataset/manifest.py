"""Portable KArSL Core-28 source-manifest construction and validation."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .core28 import CORE28_SIGN_IDS, LabelRecord, core28_records, normalize_label

CORE28_MANIFEST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "source_dataset",
    "dataset_version",
    "modality",
    "sign_id",
    "label_ar",
    "label_en_if_available",
    "label_index",
    "signer_id",
    "official_partition",
    "repetition_id",
    "source_relative_path",
    "source_file_name",
    "source_url",
    "source_sha256",
    "source_size_bytes",
    "container",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "skeleton_available",
)

_SIGNER_RE = re.compile(r"^(?:signer[_-]?)?0*([123])$", re.IGNORECASE)
_SIGN_RE = re.compile(r"^0*(\d{1,4})$")
_PARTITIONS = {"train", "test"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def _portable_relative(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Manifest path must be relative and confined to its root: {path!r}")
    if "\\" in path:
        raise ValueError(f"Manifest path must use POSIX separators: {path!r}")
    return candidate.as_posix()


def _parse_int(value: str, field: str, *, allow_blank: bool = True) -> int | None:
    if allow_blank and not str(value).strip():
        return None
    try:
        return int(str(value))
    except ValueError as error:
        raise ValueError(f"Manifest field {field} is not an integer: {value!r}") from error


def _parse_float(value: str, field: str, *, allow_blank: bool = True) -> float | None:
    if allow_blank and not str(value).strip():
        return None
    try:
        number = float(str(value))
    except ValueError as error:
        raise ValueError(f"Manifest field {field} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Manifest field {field} must be finite: {value!r}")
    return number


def _parse_positive_int(value: str, field: str, *, allow_blank: bool = True) -> int | None:
    number = _parse_int(value, field, allow_blank=allow_blank)
    if number is not None and number <= 0:
        raise ValueError(f"Manifest field {field} must be positive: {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class VideoRecord:
    sample_id: str
    sign_id: int
    label_ar: str
    label_en: str
    label_index: int
    signer_id: str
    official_partition: str
    repetition_id: str
    source_relative_path: str
    source_file_name: str
    source_url: str = ""
    source_sha256: str = ""
    source_size_bytes: int | None = None
    container: str = ".mp4"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    skeleton_available: str = "unknown"

    def to_row(self) -> dict[str, str]:
        values: dict[str, object] = {
            "sample_id": self.sample_id,
            "source_dataset": "KArSL",
            "dataset_version": "KArSL-502",
            "modality": "RGB",
            "sign_id": f"{self.sign_id:04d}",
            "label_ar": self.label_ar,
            "label_en_if_available": self.label_en,
            "label_index": self.label_index,
            "signer_id": self.signer_id,
            "official_partition": self.official_partition,
            "repetition_id": self.repetition_id,
            "source_relative_path": self.source_relative_path,
            "source_file_name": self.source_file_name,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "container": self.container,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "skeleton_available": self.skeleton_available,
        }
        return {field: "" if values[field] is None else str(values[field]) for field in CORE28_MANIFEST_FIELDS}


def validate_manifest_rows(rows: Iterable[Mapping[str, str]], *, require_core28: bool = True) -> list[dict[str, str]]:
    """Validate identity/path/label invariants without touching video files."""

    result = [dict(row) for row in rows]
    if not result:
        return result
    missing = set(CORE28_MANIFEST_FIELDS) - set(result[0])
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    expected_labels = {record.sign_id: record for record in core28_records()}
    sample_ids: set[str] = set()
    source_paths: set[str] = set()
    for row in result:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"Duplicate or empty sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        raw_source = row.get("source_relative_path", "").strip()
        if not raw_source:
            raise ValueError(f"Empty source_relative_path for {sample_id}")
        source = _portable_relative(raw_source)
        if source in {".", ""} or source in source_paths:
            raise ValueError(f"Duplicate or empty source_relative_path: {source!r}")
        source_paths.add(source)

        sign_id = _parse_int(row.get("sign_id", ""), "sign_id", allow_blank=False)
        assert sign_id is not None
        if require_core28 and sign_id not in CORE28_SIGN_IDS:
            raise ValueError(f"Non-Core-28 sign_id in manifest: {sign_id:04d}")
        label_index = _parse_int(row.get("label_index", ""), "label_index", allow_blank=False)
        assert label_index is not None
        if require_core28 and not 0 <= label_index < len(CORE28_SIGN_IDS):
            raise ValueError(f"Invalid Core-28 label_index: {label_index}")
        if require_core28:
            expected = expected_labels[sign_id]
            if normalize_label(row.get("label_ar", "")) != normalize_label(expected.label_ar):
                raise ValueError(f"label_ar does not match sign_id for {sample_id}")
            if normalize_label(row.get("label_en_if_available", "")) != normalize_label(expected.label_en):
                raise ValueError(f"label_en_if_available does not match sign_id for {sample_id}")
            if label_index != CORE28_SIGN_IDS.index(sign_id):
                raise ValueError(f"label_index does not match sign_id for {sample_id}")
        if row.get("modality", "").upper() != "RGB":
            raise ValueError(f"Only RGB source rows are accepted: {sample_id}")
        signer = row.get("signer_id", "").strip()
        if signer not in {"01", "02", "03"}:
            raise ValueError(f"Invalid signer_id for {sample_id}: {signer!r}")
        partition = row.get("official_partition", "").strip().lower()
        if partition not in _PARTITIONS:
            raise ValueError(f"Invalid official_partition for {sample_id}: {partition!r}")
        for field in ("source_size_bytes", "width", "height", "frame_count"):
            _parse_positive_int(row.get(field, ""), field)
        for field in ("fps", "duration_seconds"):
            _parse_float(row.get(field, ""), field)
        checksum = row.get("source_sha256", "").strip()
        if checksum and (len(checksum) != 64 or any(character not in "0123456789abcdefABCDEF" for character in checksum)):
            raise ValueError(f"Invalid SHA-256 for {sample_id}")
    return result


def load_manifest(path: str | Path, *, require_core28: bool = True) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source}")
        missing = set(CORE28_MANIFEST_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    return validate_manifest_rows(rows, require_core28=require_core28)


def write_manifest(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = []
    for row in rows:
        materialized.append({field: "" if row.get(field) is None else str(row.get(field)) for field in CORE28_MANIFEST_FIELDS})
    validate_manifest_rows(materialized)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORE28_MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _signer_from_token(token: str) -> str | None:
    match = _SIGNER_RE.match(token)
    return f"0{match.group(1)}" if match else None


def _sign_id_from_token(token: str) -> int | None:
    match = _SIGN_RE.match(token)
    if not match:
        return None
    value = int(match.group(1))
    return value if value in CORE28_SIGN_IDS else None


# The official KArSL-502 archives place an extra directory between the
# partition and the class, e.g. ``01/train/videos/0041/<file>.mp4``. A layout
# without it is also accepted so earlier flat fixtures keep working. At most one
# intervening segment is tolerated, which keeps the match unambiguous.
_MAX_INTERVENING_SEGMENTS = 1


def _locate_layout(relative: Path) -> tuple[str, str, int] | None:
    parts = relative.parts
    candidates: list[tuple[str, str, int]] = []
    for index in range(len(parts) - 2):
        signer = _signer_from_token(parts[index])
        partition = parts[index + 1].lower()
        if not signer or partition not in _PARTITIONS:
            continue
        for skip in range(_MAX_INTERVENING_SEGMENTS + 1):
            position = index + 2 + skip
            if position >= len(parts) - 1:
                break
            sign_id = _sign_id_from_token(parts[position])
            if sign_id is not None:
                candidates.append((signer, partition, sign_id))
                break
    unique = sorted(set(candidates))
    if len(unique) > 1:
        raise ValueError(
            "Ambiguous <signer>/<train|test>[/<dir>]/<four-digit-sign>/video layout "
            f"for {relative.as_posix()}, found {unique}"
        )
    if not unique:
        # No Core-28 class anchor in this path. The official root also holds the
        # number and extended-letter chapters, so this is an ordinary skip, not
        # a malformed layout. Genuine ambiguity still raises above.
        return None
    return unique[0]


def build_manifest_from_video_root(
    data_root: str | Path,
    video_root: str | Path,
    labels: Iterable[LabelRecord],
    *,
    source_urls: Mapping[tuple[str, str], str] | None = None,
    inspect: bool = True,
    hash_files: bool = True,
    skeleton_available: str = "unknown",
) -> list[dict[str, str]]:
    """Discover only valid Core-28 RGB videos in a deterministic order."""

    root = Path(data_root).resolve()
    videos = Path(video_root).resolve()
    if not videos.is_dir():
        raise FileNotFoundError(f"RGB video root does not exist: {videos}")
    label_by_id = {record.sign_id: record for record in labels}
    if set(label_by_id) != set(CORE28_SIGN_IDS):
        raise ValueError("Manifest discovery requires exactly the validated Core-28 label mapping")

    candidates = sorted(
        (path for path in videos.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: (
            path.relative_to(videos).as_posix().casefold(),
            path.relative_to(videos).as_posix(),
        ),
    )
    groups: dict[tuple[str, str, int], list[Path]] = {}
    parsed: dict[Path, tuple[str, str, int]] = {}
    for path in candidates:
        relative = path.relative_to(videos)
        key = _locate_layout(relative)
        if key is None:
            continue
        parsed[path] = key
        groups.setdefault(key, []).append(path)
    candidates = [path for path in candidates if path in parsed]

    rows: list[dict[str, str]] = []
    for path in candidates:
        signer, partition, sign_id = parsed[path]
        group = sorted(groups[(signer, partition, sign_id)], key=lambda item: (item.name.casefold(), item.name))
        repetition = group.index(path) + 1
        record = label_by_id[sign_id]
        try:
            source_relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Video {path} is outside data root {root}") from error
        row = VideoRecord(
            sample_id=f"karsl_core28_s{signer}_sign{sign_id:04d}_{partition}_rep{repetition:03d}",
            sign_id=sign_id,
            label_ar=record.label_ar,
            label_en=record.label_en,
            label_index=CORE28_SIGN_IDS.index(sign_id),
            signer_id=signer,
            official_partition=partition,
            repetition_id=f"rep{repetition:03d}",
            source_relative_path=source_relative,
            source_file_name=path.name,
            source_url=(source_urls or {}).get((signer, partition), ""),
            source_sha256=sha256_file(path) if hash_files else "",
            source_size_bytes=path.stat().st_size,
            skeleton_available=skeleton_available,
        )
        if inspect:
            from video_io.reader import inspect_video

            info = inspect_video(path)
            row_values = {field: getattr(row, field) for field in row.__dataclass_fields__}
            row_values.update(
                {
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "frame_count": info.decoded_frame_count,
                    "duration_seconds": info.duration_seconds,
                }
            )
            row = VideoRecord(
                **row_values
            )
            if not info.decoder_success:
                raise ValueError(f"Video decoder validation failed for {path}: {info.error}")
        rows.append(row.to_row())
    validate_manifest_rows(rows)
    return rows


def inspect_manifest_videos(
    manifest_rows: Iterable[Mapping[str, str]], data_root: str | Path
) -> list[dict[str, object]]:
    """Return per-video integrity observations without changing the manifest."""

    from video_io.reader import inspect_video

    root = Path(data_root).resolve()
    results: list[dict[str, object]] = []
    for row in validate_manifest_rows(manifest_rows):
        path = root / row["source_relative_path"]
        observation: dict[str, object] = {
            "sample_id": row["sample_id"],
            "path": row["source_relative_path"],
            "exists": path.is_file(),
        }
        if not path.is_file():
            observation["decoder_success"] = False
            observation["error"] = "missing_file"
            results.append(observation)
            continue
        info = inspect_video(path)
        observation.update(info.to_dict())
        observation["size_bytes"] = path.stat().st_size
        observation["sha256"] = sha256_file(path)
        results.append(observation)
    return results
