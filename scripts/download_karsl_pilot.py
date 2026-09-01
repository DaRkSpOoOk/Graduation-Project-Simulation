#!/usr/bin/env python3
"""Acquire the deterministic KArSL milestone-1 pilot without bulk download.

The official KArSL Google Drive distribution stores RGB clips in solid 7z
archives.  A solid archive normally requires the preceding compressed stream
to reach a selected member, so this downloader range-fetches only the prefix
needed by the manifest and the small archive header at the end.  It then asks
the system ``7z`` executable to extract only the exact members in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ARCHIVE_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
SEVEN_Z_SIGNATURE = b"7z\xbc\xaf'\x1c"
HEADER_BYTES = 32
TAIL_BYTES = 1024 * 1024
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    archive_id: str
    archive_member: str
    local_relative_path: str
    archive_prefix_bytes: int
    archive_total_bytes: int


def _read_manifest(path: Path) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        required = {
            "sample_id",
            "source_archive_id",
            "source_archive_member",
            "local_relative_path",
            "archive_prefix_bytes",
            "source_archive_total_bytes",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    rows: list[ManifestRow] = []
    for raw in raw_rows:
        local = Path(raw["local_relative_path"])
        if local.is_absolute() or ".." in local.parts:
            raise ValueError(f"Unsafe local path in manifest: {local}")
        member = Path(raw["source_archive_member"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe archive member in manifest: {member}")
        rows.append(
            ManifestRow(
                sample_id=raw["sample_id"],
                archive_id=raw["source_archive_id"],
                archive_member=raw["source_archive_member"],
                local_relative_path=raw["local_relative_path"],
                archive_prefix_bytes=int(raw["archive_prefix_bytes"]),
                archive_total_bytes=int(raw["source_archive_total_bytes"]),
            )
        )
    return rows, raw_rows


def _range_response(url: str, start: int, end: int):
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "karsl-milestone1-pilot/1.0"},
    )
    response = urllib.request.urlopen(request, timeout=180)
    content_range = response.headers.get("Content-Range", "")
    expected = f"bytes {start}-{end}/"
    if response.status != 206 or not content_range.startswith(expected):
        response.close()
        raise RuntimeError(
            "Source did not honor the bounded HTTP range request; refusing "
            f"an unbounded download (status={response.status}, "
            f"content-range={content_range!r})."
        )
    return response


def _write_range(response, output, expected_bytes: int) -> None:
    written = 0
    while True:
        chunk = response.read(CHUNK_BYTES)
        if not chunk:
            break
        output.write(chunk)
        written += len(chunk)
    response.close()
    if written != expected_bytes:
        raise IOError(f"Range truncated: expected {expected_bytes} bytes, got {written}")


def _fetch_partial_archive(archive_id: str, prefix_bytes: int, expected_total: int, destination: Path) -> dict[str, int]:
    url = ARCHIVE_URL.format(file_id=archive_id)
    with _range_response(url, 0, HEADER_BYTES - 1) as response:
        header = response.read()
    if len(header) != HEADER_BYTES or header[:6] != SEVEN_Z_SIGNATURE:
        raise ValueError(f"Source {archive_id} is not a 7z archive")
    next_header_offset, next_header_size, _ = struct.unpack("<QQI", header[12:32])
    archive_total = HEADER_BYTES + next_header_offset + next_header_size
    if archive_total != expected_total:
        raise ValueError(
            f"Manifest archive size mismatch for {archive_id}: "
            f"manifest={expected_total}, header={archive_total}"
        )
    if not HEADER_BYTES <= prefix_bytes < archive_total:
        raise ValueError(f"Invalid bounded prefix {prefix_bytes} for archive size {archive_total}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(header)
        with _range_response(url, HEADER_BYTES, prefix_bytes - 1) as response:
            _write_range(response, output, prefix_bytes - HEADER_BYTES)
        output.truncate(archive_total)

    tail_start = max(prefix_bytes, archive_total - TAIL_BYTES)
    with destination.open("r+b") as output:
        output.seek(tail_start)
        with _range_response(url, tail_start, archive_total - 1) as response:
            _write_range(response, output, archive_total - tail_start)
    return {
        "archive_total_bytes": archive_total,
        "prefix_bytes": prefix_bytes,
        "tail_bytes": archive_total - tail_start,
        "downloaded_bytes": prefix_bytes + archive_total - tail_start,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_members(archive: Path, members: Iterable[str], destination: Path) -> None:
    if shutil.which("7z") is None:
        raise RuntimeError("The bounded KArSL downloader requires the system 7z executable")
    destination.mkdir(parents=True, exist_ok=True)
    command = ["7z", "x", "-y", "-bd", str(archive), *members, f"-o{destination}"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "7z could not extract the requested manifest members.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _update_manifest(path: Path, raw_rows: list[dict[str, str]], checksums: dict[str, str]) -> None:
    fieldnames = list(raw_rows[0].keys())
    if "checksum_sha256" not in fieldnames:
        fieldnames.append("checksum_sha256")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in raw_rows:
            row["checksum_sha256"] = checksums.get(row["sample_id"], row.get("checksum_sha256", ""))
            writer.writerow(row)
    temporary.replace(path)


def download(manifest_path: Path, repository_root: Path, keep_partials: bool = False, update_manifest: bool = True) -> dict:
    rows, raw_rows = _read_manifest(manifest_path)
    if not rows:
        raise ValueError("Manifest is empty")

    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.archive_id, []).append(row)
    checksums: dict[str, str] = {}
    archive_logs: list[dict] = []

    cache_root = repository_root / "datasets" / "raw" / ".cache" / "karsl_milestone1_pilot"
    cache_root.mkdir(parents=True, exist_ok=True)
    for archive_id, archive_rows in sorted(grouped.items()):
        prefixes = {row.archive_prefix_bytes for row in archive_rows}
        totals = {row.archive_total_bytes for row in archive_rows}
        if len(prefixes) != 1 or len(totals) != 1:
            raise ValueError(f"Inconsistent archive bounds for {archive_id}")
        prefix_bytes = prefixes.pop()
        total_bytes = totals.pop()
        archive_path = cache_root / f"{archive_id}.7z"
        extraction_root = Path(tempfile.mkdtemp(prefix=f"karsl_{archive_id}_", dir=str(cache_root)))
        try:
            fetch_log = _fetch_partial_archive(archive_id, prefix_bytes, total_bytes, archive_path)
            _extract_members(archive_path, [row.archive_member for row in archive_rows], extraction_root)
            for row in archive_rows:
                extracted = extraction_root / row.archive_member
                if not extracted.is_file():
                    raise FileNotFoundError(f"Manifest member was not extracted: {row.archive_member}")
                local_path = repository_root / row.local_relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if local_path.exists():
                    if _sha256(local_path) != _sha256(extracted):
                        raise FileExistsError(f"Refusing to overwrite a different existing file: {local_path}")
                else:
                    shutil.copy2(extracted, local_path)
                checksums[row.sample_id] = _sha256(local_path)
            archive_logs.append({"archive_id": archive_id, "members": len(archive_rows), **fetch_log})
        finally:
            shutil.rmtree(extraction_root, ignore_errors=True)
            if not keep_partials:
                archive_path.unlink(missing_ok=True)

    if update_manifest:
        _update_manifest(manifest_path, raw_rows, checksums)
    return {"videos": len(rows), "archives": archive_logs, "checksums": checksums}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/karsl_milestone1_pilot.csv"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--keep-partials", action="store_true", help="Keep bounded partial 7z files for debugging")
    parser.add_argument("--no-update-manifest", action="store_true", help="Do not write downloaded SHA-256 values")
    args = parser.parse_args()
    try:
        result = download(
            args.manifest.resolve(),
            args.repository_root.resolve(),
            keep_partials=args.keep_partials,
            update_manifest=not args.no_update_manifest,
        )
    except Exception as error:  # CLI boundary: return a useful message and non-zero status.
        print(f"KArSL pilot acquisition failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
