#!/usr/bin/env python3
"""Run or inspect the resumable TASK-008A Core-28 extraction pipeline.

This is an orchestration entry point.  It deliberately keeps the validated
WiLoR, tracking, kinematics and virtual-glove implementations in their
existing packages and loads WiLoR only in worker mode, once per shard.

Status and dry-run modes are safe to use from a login node and never load a
model or alter run artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.manifest import load_manifest, manifest_sha256, validate_manifest_rows  # noqa: E402
from evaluation.dataset.orchestrator import (  # noqa: E402
    STAGES,
    RunPaths,
    assign_shards,
    format_status,
    run_worker,
    status_snapshot,
    validate_stage_artifact,
)


def _default_data_root() -> Path:
    return Path(os.environ.get("KARSL_DATA_ROOT", ROOT / "datasets" / "external" / "karsl"))


def _default_manifest() -> Path:
    return ROOT / "datasets" / "manifests" / "karsl_core28.csv"


def _default_run_root() -> Path:
    return Path(
        os.environ.get(
            "TASK008A_RUN_ROOT",
            "/ibex/user/$USER/graduation-project-runs/task008a-karsl-core28".replace("$USER", os.environ.get("USER", "user")),
        )
    )


def _resolved_shard_index(value: int | None, num_shards: int) -> int:
    if value is None:
        value = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if value < 0 or value >= num_shards:
        raise ValueError(f"shard_index {value} outside [0, {num_shards})")
    return value


def _dry_run(rows: list[dict[str, str]], args: argparse.Namespace, shard_index: int) -> dict[str, Any]:
    shards = assign_shards(rows, args.num_shards)
    selected = shards[shard_index]
    if args.sample_id:
        selected = [row for row in selected if row["sample_id"] == args.sample_id]
        if not selected:
            raise ValueError(f"Sample {args.sample_id!r} is not assigned to shard {shard_index}")
    if args.limit is not None:
        selected = selected[: args.limit]
    return {
        "mode": "dry-run",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(args.manifest),
        "data_root": str(args.data_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "num_shards": args.num_shards,
        "shard_index": shard_index,
        "assigned_samples": len(selected),
        "sample_ids": [row["sample_id"] for row in selected],
        "stage": args.stage,
        "device": args.device,
        "model_loaded": False,
    }


def _verify_only(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    manifest_hash = manifest_sha256(args.manifest)
    paths = RunPaths(args.run_root.resolve())
    stages = STAGES if args.stage == "ALL" else (args.stage,)
    samples = rows
    if args.sample_id:
        samples = [row for row in rows if row["sample_id"] == args.sample_id]
        if not samples:
            raise ValueError(f"Unknown sample_id: {args.sample_id}")
    result: list[dict[str, Any]] = []
    for row in samples:
        statuses: dict[str, str] = {}
        for stage in stages:
            statuses[stage] = (
                "VALID"
                if validate_stage_artifact(
                    paths,
                    stage,
                    row,
                    manifest_hash,
                    source_sha256=row.get("source_sha256", ""),
                )
                else "MISSING_OR_INVALID"
            )
        result.append({"sample_id": row["sample_id"], "stages": statuses})
    invalid = sum(
        any(value != "VALID" for value in sample["stages"].values())
        for sample in result
    )
    return {
        "mode": "verify-only",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "run_root": str(args.run_root.resolve()),
        "sample_count": len(result),
        "invalid_samples": invalid,
        "samples": result,
        "model_loaded": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--stage", choices=("ALL", *STAGES), default="ALL")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.status and (args.dry_run or args.verify_only):
        parser.error("--status cannot be combined with --dry-run or --verify-only")
    try:
        if args.status:
            print(format_status(status_snapshot(args.run_root, args.manifest)))
            return 0

        rows = validate_manifest_rows(load_manifest(args.manifest))
        shard_index = _resolved_shard_index(args.shard_index, args.num_shards)
        if args.dry_run:
            print(json.dumps(_dry_run(rows, args, shard_index), ensure_ascii=False, indent=2))
            return 0
        if not rows:
            raise ValueError(
                "The Core-28 manifest is schema-only/empty. Complete official label verification "
                "and source discovery before submitting an extraction worker."
            )
        if args.verify_only:
            result = _verify_only(rows, args)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["invalid_samples"] == 0 else 2
        result = run_worker(
            manifest=args.manifest.resolve(),
            data_root=args.data_root.resolve(),
            run_root=args.run_root.resolve(),
            num_shards=args.num_shards,
            shard_index=shard_index,
            resume=args.resume,
            retry_failed=args.retry_failed,
            limit=args.limit,
            sample_id=args.sample_id,
            stage=args.stage,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result["failed"] == 0 else 2
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"TASK-008A worker error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
