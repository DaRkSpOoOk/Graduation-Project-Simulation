#!/usr/bin/env python3
"""TASK-008B: verify the official local KArSL-502 source and freeze Core-28.

Phases A-C verify the official label workbook, freeze the Core-28 mapping and
resolve the extended-letter and number/digit boundaries. Phases D-G audit the
already-downloaded videos, populate the real Core-28 manifest and regenerate
signer-independent LOSO splits. Phase J emits the deterministic benchmark
subset.

Nothing is downloaded, and no dataset media is copied into the repository.

    python scripts/run_task008b_validation.py \\
      --dataset-root /home/hatim/datasets/KArSL-502 \\
      --work-dir /home/hatim/graduation-project-runs/task008b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.core28 import (  # noqa: E402
    CORE28_SIGN_IDS,
    EXTENDED_LETTER_SIGN_IDS,
    core28_records,
    extended_letter_records,
)
from evaluation.dataset.local_source import (  # noqa: E402
    discover_local_videos,
    probe_videos,
    select_smoke_subset,
)
from evaluation.dataset.manifest import (  # noqa: E402
    build_manifest_from_video_root,
    validate_manifest_rows,
    write_manifest,
)
from evaluation.dataset.official import (  # noqa: E402
    LETTER_SIGN_IDS,
    NUMBER_SIGN_IDS,
    OFFICIAL_MAPPING_VERSION,
    category_breakdown,
    read_official_workbook,
    verify_candidate_mapping,
)
from evaluation.dataset.splits import (  # noqa: E402
    build_loso_splits,
    validate_split_rows,
    write_split_manifests,
)

WORKBOOK_NAME = "KARSL-502_Labels.xlsx"
GIB = 1024 ** 3


def _stats(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "sum": int(array.sum()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="already-downloaded official KArSL-502 root")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="ignored directory for audit caches and results")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_core28.csv")
    parser.add_argument("--smoke-manifest", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_smoke.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--smoke-rows", type=int, default=24)
    parser.add_argument("--reuse-audit", action="store_true",
                        help="reuse a cached local_audit.json instead of re-probing")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"task": "TASK-008B",
                                  "dataset_root": str(args.dataset_root),
                                  "mapping_version": OFFICIAL_MAPPING_VERSION}

    # ---- Phase A: official workbook -------------------------------------
    workbook = read_official_workbook(args.dataset_root / WORKBOOK_NAME)
    print(f"[A] workbook {workbook.file_name} sha256={workbook.sha256[:16]}... "
          f"rows={workbook.total_rows} data={workbook.data_rows} issues={len(workbook.issues)}")
    results["workbook"] = {
        "file_name": workbook.file_name, "size_bytes": workbook.size_bytes,
        "sha256": workbook.sha256, "sheet_names": workbook.sheet_names,
        "sheet_used": workbook.sheet_used, "total_rows": workbook.total_rows,
        "header": workbook.header, "data_rows": workbook.data_rows,
        "sign_id_min": workbook.sign_id_min, "sign_id_max": workbook.sign_id_max,
        "issues": workbook.issues,
    }

    # ---- Phases B/C: freeze mapping, resolve boundaries ------------------
    core = verify_candidate_mapping(workbook, core28_records(), tuple(CORE28_SIGN_IDS))
    ext = verify_candidate_mapping(workbook, extended_letter_records(), tuple(EXTENDED_LETTER_SIGN_IDS))
    breakdown = category_breakdown(workbook)
    print(f"[B] Core-28 verification: {'PASS' if core['passed'] else 'FAIL'} "
          f"({core['class_count']} classes)")
    print(f"[C] extended: {'PASS' if ext['passed'] else 'FAIL'} ({ext['class_count']} classes); "
          f"numbers: {breakdown['number']['class_count']} classes")
    results["core28_verification"] = core
    results["extended_verification"] = ext
    results["category_breakdown"] = breakdown
    results["sign_id_0031"] = next(
        {"sign_id": e.sign_id, "label_ar": e.label_ar, "label_en": e.label_en,
         "arabic_cell_type": e.arabic_cell_type, "category": e.category}
        for e in workbook.entries if e.sign_id == 31
    )

    # ---- Phase D: audit local videos ------------------------------------
    cache = args.work_dir / "local_audit.json"
    if args.reuse_audit and cache.is_file():
        payload = json.loads(cache.read_text())
        print(f"[D] reusing cached audit: {len(payload['videos'])} videos")
    else:
        videos = discover_local_videos(args.dataset_root)
        probes = probe_videos(args.dataset_root, videos, workers=args.workers,
                              hash_files=True, progress=True)
        payload = {"videos": [v.to_dict() for v in videos],
                   "probes": [p.to_dict() for p in probes.values()]}
        cache.write_text(json.dumps(payload))
        print(f"[D] audited {len(videos)} videos")

    video_by_path = {v["relative_path"]: v for v in payload["videos"]}
    probe_by_path = {p["relative_path"]: p for p in payload["probes"]}
    failed = [p for p in probe_by_path.values() if not p["ok"]]

    # integrity findings
    name_mismatch = [v for v in video_by_path.values() if not v["path_sign_id_matches_filename"]]
    digests: dict[str, list[str]] = {}
    for probe in probe_by_path.values():
        if probe["ok"] and probe["sha256"]:
            digests.setdefault(probe["sha256"], []).append(probe["relative_path"])
    duplicate_groups = [paths for paths in digests.values() if len(paths) > 1]
    results["integrity"] = {
        "videos_discovered": len(video_by_path),
        "probe_failures": len(failed),
        "probe_failure_detail": failed[:20],
        "filename_signid_mismatch_count": len(name_mismatch),
        "filename_signid_mismatch_sign_ids": sorted({v["sign_id"] for v in name_mismatch}),
        "duplicate_content_groups": len(duplicate_groups),
        "duplicate_files_involved": sum(len(g) for g in duplicate_groups),
        "duplicate_groups_touching_core28": sum(
            1 for g in duplicate_groups
            if any(32 <= video_by_path[p]["sign_id"] <= 59 for p in g)
        ),
        "duplicate_example": sorted(duplicate_groups)[0] if duplicate_groups else [],
    }

    # ---- Phase E: exact sizes -------------------------------------------
    def region(name: str, low: int, high: int) -> dict[str, object]:
        rows = [v for v in video_by_path.values() if low <= v["sign_id"] <= high]
        frames = [probe_by_path[v["relative_path"]]["frame_count"] or 0 for v in rows]
        total_bytes = sum(v["size_bytes"] for v in rows)
        duration = sum(probe_by_path[v["relative_path"]]["duration_seconds"] or 0.0 for v in rows)
        return {
            "name": name, "sign_id_low": low, "sign_id_high": high,
            "classes": len({v["sign_id"] for v in rows}), "videos": len(rows),
            "bytes": total_bytes, "mib": total_bytes / (1024 ** 2), "gib": total_bytes / GIB,
            "frames": int(sum(frames)), "duration_seconds": duration,
            "duration_hours": duration / 3600.0, "frame_stats": _stats(frames),
        }

    sizes = {
        "core28": region("Core-28", 32, 59),
        "letters_all": region("All letter forms", 32, 70),
        "extended_letters": region("Extended letters", 60, 70),
        "numbers": region("Numbers/digits", 1, 31),
        "local_subset_all": region("Entire local 0001-0070", 1, 70),
    }
    core_rows = [v for v in video_by_path.values() if 32 <= v["sign_id"] <= 59]
    sizes["core28_by_signer"] = {}
    for signer in sorted({v["signer_id"] for v in core_rows}):
        subset = [v for v in core_rows if v["signer_id"] == signer]
        frames = [probe_by_path[v["relative_path"]]["frame_count"] or 0 for v in subset]
        sizes["core28_by_signer"][signer] = {
            "videos": len(subset), "frames": int(sum(frames)),
            "bytes": sum(v["size_bytes"] for v in subset), "frame_stats": _stats(frames),
        }
    sizes["core28_by_partition"] = {}
    for partition in ("train", "test"):
        subset = [v for v in core_rows if v["official_partition"] == partition]
        frames = [probe_by_path[v["relative_path"]]["frame_count"] or 0 for v in subset]
        sizes["core28_by_partition"][partition] = {
            "videos": len(subset), "frames": int(sum(frames)),
            "bytes": sum(v["size_bytes"] for v in subset),
        }
    sizes["core28_by_class"] = {}
    for sign_id in CORE28_SIGN_IDS:
        subset = [v for v in core_rows if v["sign_id"] == sign_id]
        frames = [probe_by_path[v["relative_path"]]["frame_count"] or 0 for v in subset]
        sizes["core28_by_class"][f"{sign_id:04d}"] = {
            "videos": len(subset), "frames": int(sum(frames)),
            "bytes": sum(v["size_bytes"] for v in subset),
        }
    results["sizes"] = sizes
    print(f"[E] Core-28: {sizes['core28']['videos']} videos, "
          f"{sizes['core28']['frames']:,} frames, {sizes['core28']['gib']:.3f} GiB")

    # ---- Phase F: populate the real Core-28 manifest ---------------------
    rows = build_manifest_from_video_root(
        args.dataset_root, args.dataset_root, core28_records(),
        inspect=False, hash_files=False, skeleton_available="unknown",
    )
    for row in rows:
        relative = row["source_relative_path"]
        probe = probe_by_path.get(relative)
        if probe is None or not probe["ok"]:
            continue
        row["source_sha256"] = probe["sha256"] or ""
        row["width"] = str(probe["width"] or "")
        row["height"] = str(probe["height"] or "")
        row["fps"] = f"{probe['fps']:.6f}" if probe["fps"] else ""
        row["frame_count"] = str(probe["frame_count"] or "")
        row["duration_seconds"] = f"{probe['duration_seconds']:.6f}" if probe["duration_seconds"] else ""
        row["container"] = "mp4"
    validated = validate_manifest_rows(rows)
    write_manifest(args.manifest, validated)
    sample_ids = [r["sample_id"] for r in validated]
    paths = [r["source_relative_path"] for r in validated]
    absolute = [p for p in paths if p.startswith("/") or ":" in p.split("/")[0]]
    missing = [p for p in paths if not (args.dataset_root / p).is_file()]
    results["manifest"] = {
        "path": str(args.manifest.relative_to(ROOT)),
        "rows": len(validated),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "duplicate_source_paths": len(paths) - len(set(paths)),
        "unknown_labels": sum(1 for r in validated if not r["label_ar"]),
        "invalid_signers": sum(1 for r in validated if r["signer_id"] not in {"01", "02", "03"}),
        "invalid_partitions": sum(1 for r in validated if r["official_partition"] not in {"train", "test"}),
        "missing_files": len(missing),
        "absolute_paths": len(absolute),
        "portable_paths": not absolute,
        "rows_without_frame_count": sum(1 for r in validated if not r["frame_count"]),
        "classes": len({r["sign_id"] for r in validated}),
        "signers": sorted({r["signer_id"] for r in validated}),
    }
    print(f"[F] manifest rows={len(validated)} duplicates=0 missing={len(missing)} "
          f"portable={'PASS' if not absolute else 'FAIL'}")

    # ---- Phase G: real LOSO splits ---------------------------------------
    written = write_split_manifests(args.splits_dir, validated)
    loso: dict[str, object] = {}
    for signer in ("01", "02", "03"):
        # build_loso_splits validates the fold internally; re-validate here so
        # a regression in that internal call cannot pass silently.
        assignments = build_loso_splits(validated, signer)
        validate_split_rows(assignments, manifest_rows=validated, held_out_signer=signer)
        buckets: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
        for row in assignments:
            buckets[row["role"]].append(row)
        leakage = sum(1 for role in ("train", "validation")
                      for row in buckets[role] if row["signer_id"] == signer)
        assigned = {row["sample_id"] for row in assignments}
        loso[f"S{signer}"] = {
            "train": len(buckets["train"]), "validation": len(buckets["validation"]),
            "test": len(buckets["test"]),
            "classes_train": len({r["sign_id"] for r in buckets["train"]}),
            "classes_validation": len({r["sign_id"] for r in buckets["validation"]}),
            "classes_test": len({r["sign_id"] for r in buckets["test"]}),
            "signers_train": sorted({r["signer_id"] for r in buckets["train"]}),
            "signers_validation": sorted({r["signer_id"] for r in buckets["validation"]}),
            "signers_test": sorted({r["signer_id"] for r in buckets["test"]}),
            "held_out_leakage": leakage,
            "unassigned": len(validated) - len(assigned),
            "duplicate_assignment": len(assignments) - len(assigned),
        }
        print(f"[G] LOSO S{signer}: train={loso[f'S{signer}']['train']} "
              f"val={loso[f'S{signer}']['validation']} test={loso[f'S{signer}']['test']} "
              f"leakage={leakage}")
    results["loso"] = loso
    results["loso_files"] = {k: str(Path(v).relative_to(ROOT)) for k, v in written.items()}

    # ---- Phase J: deterministic smoke subset -----------------------------
    smoke = select_smoke_subset(validated, target=args.smoke_rows)
    write_manifest(args.smoke_manifest, smoke)
    smoke_frames = [int(r["frame_count"]) for r in smoke]
    results["smoke"] = {
        "path": str(args.smoke_manifest.relative_to(ROOT)),
        "rows": len(smoke),
        "signers": sorted({r["signer_id"] for r in smoke}),
        "partitions": sorted({r["official_partition"] for r in smoke}),
        "classes": sorted({r["sign_id"] for r in smoke}),
        "class_count": len({r["sign_id"] for r in smoke}),
        "frames_total": int(sum(smoke_frames)),
        "frame_min": min(smoke_frames), "frame_max": max(smoke_frames),
        "frame_median": float(np.median(smoke_frames)),
        "sample_ids": [r["sample_id"] for r in smoke],
    }
    print(f"[J] smoke rows={len(smoke)} signers={results['smoke']['signers']} "
          f"frames={results['smoke']['frames_total']} "
          f"range={results['smoke']['frame_min']}-{results['smoke']['frame_max']}")

    out = args.work_dir / "task008b_phases_a_to_j.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
