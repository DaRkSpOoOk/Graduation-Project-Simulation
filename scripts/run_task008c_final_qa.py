#!/usr/bin/env python3
"""TASK-008C: independently finalize the completed Core-28 production run.

The expensive extraction is already finished. This driver never loads WiLoR and
never rewrites a stage artifact; it re-derives every claim from the run root,
the frozen manifest and the frozen split/label tables, and writes the machine
readable statistics the TASK-008C report cites.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.final_qa import (  # noqa: E402
    aggregate_dataset,
    coverage_accounting,
    sequence_length_statistics,
    verify_dataset_contract,
    verify_label_integrity,
    verify_loso_folds,
)
from evaluation.dataset.manifest import load_manifest, manifest_sha256  # noqa: E402
from evaluation.dataset.qa import validate_run  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_core28.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "datasets/manifests/karsl_core28_labels.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dataset/TASK-008C-final-qa.json")
    parser.add_argument("--skip-per-sample", action="store_true", help="skip the slow per-sample re-hash pass")
    args = parser.parse_args(argv)

    run_root = args.run_root.resolve()
    rows = load_manifest(args.manifest)
    started = time.perf_counter()
    payload: dict[str, object] = {
        "schema_version": "task008c_final_qa_v1",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256(args.manifest),
        "data_root": str(args.data_root.resolve()),
        "run_root": str(run_root),
        "requested_samples": len(rows),
    }

    state_path = run_root / "state" / "shard-00.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["worker_state"] = {
        "path": str(state_path),
        "manifest_sha256": state_payload.get("manifest_sha256"),
        "manifest_hash_matches": state_payload.get("manifest_sha256") == payload["manifest_sha256"],
        "updated_utc": state_payload.get("updated_utc"),
        "frames_processed_total": sum(
            int(entry.get("frames_processed") or 0) for entry in state_payload.get("samples", {}).values()
        ),
        "attempts_greater_than_one": sum(
            int(entry.get("attempts") or 0) > 1 for entry in state_payload.get("samples", {}).values()
        ),
    }
    payload["coverage"] = coverage_accounting(rows, state_payload)

    provenance_path = run_root / "provenance" / "shard-00.json"
    if provenance_path.is_file():
        payload["run_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))

    on_disk = {stage: sorted(p.name for p in (run_root / rel).iterdir() if p.is_dir())
               for stage, rel in (("pose", "pose/raw"), ("tracking", "tracking"),
                                  ("kinematics", "kinematics"), ("virtual_glove", "virtual_glove"))}
    requested_ids = {row["sample_id"] for row in rows}
    payload["on_disk"] = {
        stage: {
            "directories": len(names),
            "extra_not_in_manifest": sorted(set(names) - requested_ids)[:20],
            "missing_from_disk": sorted(requested_ids - set(names))[:20],
            "duplicate_directory_names": len(names) - len(set(names)),
        }
        for stage, names in on_disk.items()
    }

    print("[1/6] dataset aggregation", flush=True)
    payload["aggregate"] = aggregate_dataset(rows, run_root)
    print("[2/6] contract / temporal / value verification", flush=True)
    payload["contract"] = verify_dataset_contract(rows, run_root)
    print("[3/6] label integrity", flush=True)
    payload["labels"] = verify_label_integrity(rows, _read_csv(args.labels))
    print("[4/6] LOSO folds", flush=True)
    fold_rows = {
        signer: _read_csv(args.splits_dir / f"karsl_core28_loso_s{signer}.csv")
        for signer in ("01", "02", "03")
        if (args.splits_dir / f"karsl_core28_loso_s{signer}.csv").is_file()
    }
    payload["loso"] = verify_loso_folds(fold_rows, rows)

    print("[5/6] sequence-length statistics", flush=True)
    lengths_by_signer: dict[str, list[int]] = {}
    # Read from the finalized index so the report's length statistics and the
    # committed index cannot drift apart; build the index before this driver.
    index_rows = _read_csv(ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    lengths_all: list[int] = []
    for row in index_rows:
        if row.get("sequence_length"):
            lengths_all.append(int(row["sequence_length"]))
            lengths_by_signer.setdefault(row["signer_id"], []).append(int(row["sequence_length"]))
    if lengths_all:
        payload["sequence_length_statistics"] = {
            "overall": sequence_length_statistics(lengths_all),
            "by_signer": {k: sequence_length_statistics(v) for k, v in sorted(lengths_by_signer.items())},
        }

    payload["storage"] = {
        "run_root_bytes": _directory_bytes(run_root),
        "by_stage_bytes": {
            stage: _directory_bytes(run_root / rel)
            for stage, rel in (("pose", "pose"), ("tracking", "tracking"),
                               ("kinematics", "kinematics"), ("virtual_glove", "virtual_glove"),
                               ("state", "state"), ("provenance", "provenance"))
        },
        "source_video_bytes": sum(int(row["source_size_bytes"]) for row in rows if row.get("source_size_bytes")),
    }

    if not args.skip_per_sample:
        print("[6/6] per-sample QA over every sample (re-hashes every source video)", flush=True)
        per_sample = validate_run(args.manifest, args.data_root, run_root)
        # The 4,222 per-sample rows are already carried, per sample, by the
        # committed index CSV. Keeping them here too would make this statistics
        # file 2 MB of duplicated detail, so only the aggregates are retained.
        per_sample.pop("sample_results", None)
        payload["per_sample_qa"] = per_sample
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} in {payload['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
