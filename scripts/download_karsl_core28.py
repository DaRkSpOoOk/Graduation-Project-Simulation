#!/usr/bin/env python3
"""Acquire and validate the official KArSL Core-28 RGB source.

The official KArSL distribution exposes SharePoint folders, not a stable
individual-video API.  This command deliberately accepts a catalog of direct
binary URLs for video assets and never guesses how a folder should be
scraped.  A disabled/HTML source is reported as an acquisition error rather
than being replaced by a mirror.

Typical workflow on Ibex::

    python scripts/download_karsl_core28.py --data-root "$KARSL_DATA_ROOT" \
        --labels-only
    python scripts/download_karsl_core28.py --data-root "$KARSL_DATA_ROOT" \
        --source-catalog "$KARSL_DATA_ROOT/acquisition/source_catalog.csv"
    python scripts/download_karsl_core28.py --data-root "$KARSL_DATA_ROOT" \
        --discover --verify

The command does not run WiLoR or any downstream extraction stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.acquisition import (  # noqa: E402
    AcquisitionError,
    acquisition_status,
    download_file,
    load_source_catalog,
    preflight_catalog,
    write_json,
)
from evaluation.dataset.core28 import (  # noqa: E402
    OFFICIAL_LABELS_URL,
    OFFICIAL_RGB_SOURCES,
    core28_records,
    extended_letter_records,
    load_label_records,
    validate_core28_records,
)
from evaluation.dataset.manifest import (  # noqa: E402
    build_manifest_from_video_root,
    inspect_manifest_videos,
    load_manifest,
    manifest_sha256,
    sha256_file,
    write_manifest,
)
from evaluation.dataset.splits import write_split_headers, write_split_manifests  # noqa: E402


def _default_data_root() -> Path:
    import os

    return Path(os.environ.get("KARSL_DATA_ROOT", ROOT / "datasets" / "external" / "karsl"))


def _default_run_manifest() -> Path:
    return ROOT / "datasets" / "manifests" / "karsl_core28.csv"


def _labels_target(data_root: Path, value: Path | None) -> Path:
    return (value or (data_root / "labels" / "KARSL-502_Labels.xlsx")).resolve()


def _label_verification_path(data_root: Path) -> Path:
    return data_root / "labels" / "official_labels_verification.json"


def _load_verified_or_committed_labels(data_root: Path, labels_path: Path) -> tuple[list[Any], bool, str]:
    """Return labels and whether an official workbook has been validated."""

    if labels_path.is_file():
        validated = validate_core28_records(load_label_records(labels_path))
        verification = {
            "status": "verified",
            "source": str(labels_path),
            "source_sha256": sha256_file(labels_path),
        }
        marker = _label_verification_path(data_root)
        if not marker.is_file():
            return validated, False, "official workbook exists but --labels-only verification marker is missing"
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("status") != "verified" or existing.get("source_sha256") != verification["source_sha256"]:
                return validated, False, "official workbook exists but its verification marker is stale"
        except (OSError, json.JSONDecodeError):
            return validated, False, "official workbook exists but verification marker is malformed"
        return validated, True, "official workbook and verification marker validated"
    return core28_records(), False, "official workbook is not available; committed mapping is unverified"


def _progress(received: int, total: int | None) -> None:
    if total:
        print(f"\r  downloaded {received / 1048576:.1f}/{total / 1048576:.1f} MiB", end="", flush=True)
    else:
        print(f"\r  downloaded {received / 1048576:.1f} MiB", end="", flush=True)


def _download_labels(args: argparse.Namespace, data_root: Path, labels_path: Path) -> dict[str, Any]:
    if labels_path.is_file():
        try:
            records = validate_core28_records(load_label_records(labels_path))
        except (OSError, ValueError):
            records = []
        if records:
            payload = {
                "status": "verified",
                "source_url": OFFICIAL_LABELS_URL,
                "local_path": str(labels_path),
                "source_sha256": sha256_file(labels_path),
                "class_count": len(records),
                "retrieved_utc": None,
                "note": "Existing file validated; no network request was made.",
            }
            if not args.dry_run:
                write_json(_label_verification_path(data_root), payload)
            return payload
    if args.dry_run:
        return {
            "status": "dry_run",
            "source_url": OFFICIAL_LABELS_URL,
            "local_path": str(labels_path),
            "class_count": 28,
        }
    result = download_file(OFFICIAL_LABELS_URL, labels_path, resume=args.resume, progress=_progress)
    print()
    records = validate_core28_records(load_label_records(labels_path))
    payload = {
        "status": "verified",
        "source_url": OFFICIAL_LABELS_URL,
        "local_path": str(labels_path),
        "source_sha256": result["sha256"],
        "size_bytes": result["size_bytes"],
        "class_count": len(records),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(_label_verification_path(data_root), payload)
    return payload


def _download_catalog(args: argparse.Namespace, data_root: Path, catalog_path: Path) -> dict[str, Any]:
    entries = load_source_catalog(catalog_path)
    if not entries and not args.dry_run:
        raise AcquisitionError(
            f"Source catalog {catalog_path} is empty; populate it with official direct binary assets first"
        )
    preflight = preflight_catalog(entries, data_root)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if not preflight["sufficient_for_known_bytes"]:
        raise AcquisitionError("Available disk space is insufficient for the known catalog bytes")
    if args.dry_run:
        return {"status": "dry_run", "preflight": preflight, "entries": len(entries)}
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in entries:
        destination = data_root / entry.destination_relative_path
        try:
            print(f"Downloading {entry.destination_relative_path} from {entry.source_url}")
            result = download_file(
                entry.source_url,
                destination,
                resume=args.resume,
                expected_sha256=entry.expected_sha256,
                expected_size_bytes=entry.expected_size_bytes,
                progress=_progress,
            )
            print()
            outcomes.append({"sample_id": entry.sample_id, **result})
        except AcquisitionError as error:
            print()
            failures.append({"sample_id": entry.sample_id, "url": entry.source_url, "error": str(error)})
            if not args.retry_failed:
                break
    payload = {
        "status": "failed" if failures else "complete",
        "catalog": str(catalog_path.resolve()),
        "preflight": preflight,
        "completed": outcomes,
        "failures": failures,
    }
    write_json(data_root / "acquisition" / "status.json", payload)
    if failures:
        raise AcquisitionError(f"{len(failures)} catalog asset(s) failed; see acquisition/status.json")
    return payload


def _discover(args: argparse.Namespace, data_root: Path, labels_path: Path) -> dict[str, Any]:
    video_root = (args.video_root or data_root / "raw").resolve()
    labels, verified, label_status = _load_verified_or_committed_labels(data_root, labels_path)
    if not verified and not args.allow_unverified_labels:
        raise AcquisitionError(
            "Official labels are not verified. Run --labels-only successfully before --discover, "
            "or use --allow-unverified-labels for a non-production development discovery."
        )
    rows = build_manifest_from_video_root(
        data_root,
        video_root,
        labels,
        source_urls=OFFICIAL_RGB_SOURCES,
        inspect=not args.skip_inspection,
        hash_files=not args.skip_hash,
        skeleton_available="unknown",
    )
    destination = args.manifest.resolve()
    split_dir = args.split_dir.resolve()
    split_paths = {
        signer: str(split_dir / f"karsl_core28_loso_s{signer}.csv") for signer in ("01", "02", "03")
    }
    if not args.dry_run:
        write_manifest(destination, rows)
        if rows:
            split_paths = {key: str(value) for key, value in write_split_manifests(split_dir, rows).items()}
        else:
            write_split_headers(split_dir)
    payload = {
        "status": "dry_run" if args.dry_run else "complete",
        "video_root": str(video_root),
        "manifest": str(destination),
        "manifest_sha256": manifest_sha256(destination) if destination.is_file() else None,
        "sample_count": len(rows),
        "label_status": label_status,
        "official_labels_verified": verified,
        "split_paths": split_paths,
        "extended_letter_candidates": [record.to_dict() for record in extended_letter_records()],
    }
    if not args.dry_run:
        write_json(data_root / "acquisition" / "discovery.json", payload)
    return payload


def _verify(args: argparse.Namespace, data_root: Path) -> dict[str, Any]:
    rows = load_manifest(args.manifest)
    if not rows:
        raise AcquisitionError("Cannot verify a populated dataset from an empty/schema-only manifest")
    observations = inspect_manifest_videos(rows, data_root)
    rows_by_id = {row["sample_id"]: row for row in rows}
    failures: list[dict[str, Any]] = []
    for observation in observations:
        row = rows_by_id[observation["sample_id"]]
        expected_hash = row.get("source_sha256", "").lower()
        actual_hash = str(observation.get("sha256", "")).lower()
        expected_size = row.get("source_size_bytes", "")
        size_mismatch = bool(expected_size) and int(expected_size) != int(observation.get("size_bytes", 0) or 0)
        hash_mismatch = bool(expected_hash) and expected_hash != actual_hash
        if (
            not observation.get("exists")
            or not observation.get("decoder_success")
            or not actual_hash
            or observation.get("size_bytes", 0) <= 0
            or size_mismatch
            or hash_mismatch
        ):
            failures.append(
                {
                    **observation,
                    "expected_sha256": expected_hash or None,
                    "expected_size_bytes": int(expected_size) if expected_size else None,
                    "hash_mismatch": hash_mismatch,
                    "size_mismatch": size_mismatch,
                }
            )
    payload = {
        "status": "failed" if failures else "verified",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(args.manifest),
        "data_root": str(data_root.resolve()),
        "sample_count": len(rows),
        "readable_count": len(observations) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "observations": observations,
    }
    if not args.dry_run:
        write_json(data_root / "checksums" / "karsl_core28_integrity.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--manifest", type=Path, default=_default_run_manifest())
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--source-catalog", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "datasets" / "splits")
    parser.add_argument("--resume", action="store_true", help="resume partial downloads and preserve existing files")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show sources/preflight without downloading or writing")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--discover", action="store_true", help="discover local RGB files into the portable manifest")
    parser.add_argument("--allow-unverified-labels", action="store_true", help="development-only discovery override")
    parser.add_argument("--skip-inspection", action="store_true")
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--status", action="store_true", help="read acquisition status only")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    labels_path = _labels_target(data_root, args.labels_file)
    if args.status:
        print(json.dumps(acquisition_status(data_root, args.manifest), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        if args.labels_only:
            print(json.dumps(_download_labels(args, data_root, labels_path), ensure_ascii=False, indent=2, sort_keys=True))
        if args.source_catalog:
            _download_catalog(args, data_root, args.source_catalog.resolve())
        if args.discover:
            print(json.dumps(_discover(args, data_root, labels_path), ensure_ascii=False, indent=2, sort_keys=True))
        if args.verify:
            result = _verify(args, data_root)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            if result["failure_count"]:
                return 2
        if not any((args.labels_only, args.source_catalog, args.discover, args.verify)):
            parser.error("choose --labels-only, --source-catalog, --discover, --verify, or --status")
    except (AcquisitionError, FileNotFoundError, ValueError, OSError) as error:
        print(f"TASK-008A acquisition error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
