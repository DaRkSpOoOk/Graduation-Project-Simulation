"""Small, resumable HTTP acquisition primitives for official KArSL assets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CHUNK_BYTES = 4 * 1024 * 1024
USER_AGENT = "Graduation-Project-Simulation/TASK-008A"


class AcquisitionError(RuntimeError):
    """A source could not be downloaded or verified safely."""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    source_url: str
    destination_relative_path: str
    expected_sha256: str = ""
    expected_size_bytes: int | None = None
    sample_id: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    if not value.strip():
        raise AcquisitionError("Catalog destination path must not be empty")
    path = Path(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts or "\\" in value:
        raise AcquisitionError(f"Unsafe catalog destination path: {value!r}")
    return path


def load_source_catalog(path: str | Path) -> list[CatalogEntry]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"source_url", "destination_relative_path"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise AcquisitionError(f"Source catalog requires columns: {sorted(required)}")
        result: list[CatalogEntry] = []
        destinations: set[str] = set()
        sample_ids: set[str] = set()
        for row in reader:
            url = row.get("source_url", "").strip()
            destination = row.get("destination_relative_path", "").strip()
            if not url:
                raise AcquisitionError("Source catalog contains an empty source_url")
            safe = _safe_relative(destination)
            normalized = safe.as_posix()
            if normalized in destinations:
                raise AcquisitionError(f"Duplicate source destination: {normalized}")
            destinations.add(normalized)
            sample_id = row.get("sample_id", "").strip()
            if sample_id:
                if sample_id in sample_ids:
                    raise AcquisitionError(f"Duplicate source catalog sample_id: {sample_id}")
                sample_ids.add(sample_id)
            expected_size = row.get("expected_size_bytes", "").strip()
            try:
                size = int(expected_size) if expected_size else None
            except ValueError as error:
                raise AcquisitionError(f"Invalid expected_size_bytes for {normalized}") from error
            if size is not None and size <= 0:
                raise AcquisitionError(f"expected_size_bytes must be positive for {normalized}")
            checksum = row.get("expected_sha256", "").strip()
            if checksum and (len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum)):
                raise AcquisitionError(f"Invalid expected_sha256 for {normalized}")
            result.append(
                CatalogEntry(
                    source_url=url,
                    destination_relative_path=normalized,
                    expected_sha256=checksum,
                    expected_size_bytes=size,
                    sample_id=sample_id,
                )
            )
    return result


def _response_length(response: object) -> int | None:
    value = getattr(response, "headers", {}).get("Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def download_file(
    url: str,
    destination: str | Path,
    *,
    resume: bool = True,
    expected_sha256: str = "",
    expected_size_bytes: int | None = None,
    retries: int = 3,
    progress: Callable[[int, int | None], None] | None = None,
) -> dict[str, object]:
    """Download one direct binary URL using a sidecar ``.part`` file.

    A server that returns an HTML folder/share page is rejected.  A resumed
    request is accepted only when the server answers with HTTP 206; if it
    ignores the Range header, the partial file is restarted safely.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        existing_size = target.stat().st_size
        if expected_size_bytes is not None and existing_size != expected_size_bytes:
            # A complete-looking but incorrect file is not trusted.  It is
            # replaced through the normal .part path below.
            pass
        elif expected_sha256:
            existing_sha = sha256_file(target)
            if existing_sha.lower() == expected_sha256.lower():
                return {
                    "url": url,
                    "path": str(target),
                    "size_bytes": existing_size,
                    "sha256": existing_sha,
                    "attempt": 0,
                    "resumed": False,
                    "skipped_existing": True,
                }
        elif expected_size_bytes is not None:
            return {
                "url": url,
                "path": str(target),
                "size_bytes": existing_size,
                "sha256": sha256_file(target),
                "attempt": 0,
                "resumed": False,
                "skipped_existing": True,
            }
        elif not resume or not partial.exists():
            return {
                "url": url,
                "path": str(target),
                "size_bytes": existing_size,
                "sha256": sha256_file(target),
                "attempt": 0,
                "resumed": False,
                "skipped_existing": True,
            }
    partial = target.with_name(target.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            response = urlopen(Request(url, headers=headers), timeout=180)
            status = getattr(response, "status", None)
            if offset and status != 206:
                response.close()
                partial.unlink(missing_ok=True)
                offset = 0
                response = urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=180)
                status = getattr(response, "status", None)
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "text/html" in content_type or "text/plain" in content_type and not url.lower().endswith(('.mp4', '.7z', '.zip', '.xlsx')):
                response.close()
                raise AcquisitionError(
                    f"Source returned {content_type or 'text'} rather than a direct binary: {url}"
                )
            length = _response_length(response)
            total = offset + length if length is not None else expected_size_bytes
            mode = "ab" if offset else "wb"
            received = offset
            with response, partial.open(mode) as handle:
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
                    received += len(block)
                    if progress:
                        progress(received, total)
            if expected_size_bytes is not None and received != expected_size_bytes:
                raise AcquisitionError(
                    f"Size mismatch for {url}: received {received}, expected {expected_size_bytes}"
                )
            checksum = sha256_file(partial)
            if expected_sha256 and checksum.lower() != expected_sha256.lower():
                raise AcquisitionError(
                    f"SHA-256 mismatch for {url}: received {checksum}, expected {expected_sha256}"
                )
            partial.replace(target)
            return {
                "url": url,
                "path": str(target),
                "size_bytes": received,
                "sha256": checksum,
                "attempt": attempt,
                "resumed": bool(offset),
            }
        except (HTTPError, URLError, TimeoutError, OSError, AcquisitionError) as error:
            last_error = error
            # A checksum/size/content-type failure invalidates the partial
            # byte stream.  Network interruptions remain resumable; integrity
            # failures must restart rather than append to a bad .part file.
            if isinstance(error, AcquisitionError):
                partial.unlink(missing_ok=True)
            elif isinstance(error, HTTPError) and error.code == 416:
                partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise AcquisitionError(f"Download failed after {retries} attempts: {url}: {last_error}") from last_error


def preflight_catalog(
    entries: Iterable[CatalogEntry], data_root: str | Path
) -> dict[str, object]:
    """Calculate known bytes before any catalog download is started."""

    entries = list(entries)
    root = Path(data_root).resolve()
    expected = 0
    known = True
    existing = 0
    existing_files = 0
    for entry in entries:
        if entry.expected_size_bytes is None:
            known = False
        else:
            expected += entry.expected_size_bytes
        target = root / _safe_relative(entry.destination_relative_path)
        if target.is_file():
            existing += target.stat().st_size
            existing_files += 1
    usage_path = root
    while not usage_path.exists() and usage_path != usage_path.parent:
        usage_path = usage_path.parent
    usage = shutil.disk_usage(usage_path)
    remaining = max(expected - existing, 0) if known else None
    return {
        "data_root": str(root),
        "entries": len(entries),
        "expected_bytes_known": known,
        "expected_total_bytes": expected if known else None,
        "existing_bytes": existing,
        "existing_files": existing_files,
        "remaining_known_bytes": remaining,
        "free_bytes": usage.free,
        "sufficient_for_known_bytes": remaining is None or usage.free >= remaining,
    }


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def acquisition_status(data_root: str | Path, manifest: str | Path | None = None) -> dict[str, object]:
    root = Path(data_root).resolve()
    status_path = root / "acquisition" / "status.json"
    payload: dict[str, object] = {
        "data_root": str(root),
        "status_file": str(status_path),
        "status": json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {},
    }
    if manifest:
        manifest_path = Path(manifest)
        payload["manifest"] = str(manifest_path.resolve())
        payload["manifest_exists"] = manifest_path.is_file()
    return payload
