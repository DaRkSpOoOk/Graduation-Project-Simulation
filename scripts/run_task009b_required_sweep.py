#!/usr/bin/env python3
"""TASK-009B sweep helper: the required experiment matrix, run sequentially.

Prints every command before running it, skips experiments already marked
COMPLETE, and stops on the first genuine failure rather than burying it. Use
--dry-run to see the matrix without spending GPU time.

This is invoked by the user. The agent that wrote it does not run it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/run_task009b_lstm_baseline.py"
DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")
DEFAULT_DATA_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")

# (feature_set, quaternion_policy, pooling, purpose)
MATRIX: tuple[tuple[str, str, str, str], ...] = (
    ("bend_only", "absolute", "masked_mean", "sensor ablation: 15 bend channels per hand"),
    ("bend_spread", "absolute", "masked_mean", "sensor ablation: + 4 spread channels per hand"),
    ("full", "absolute", "masked_mean", "primary: + palm orientation, absolute"),
    ("full", "relative_first_valid", "masked_mean", "quaternion policy comparison"),
    ("full", "absolute", "final_hidden", "duration Control B: pooling comparison"),
)


def experiment_dir(run_root: Path, feature_set: str, quaternion: str, pooling: str,
                   fold: str, seed: int) -> Path:
    tag = quaternion if feature_set == "full" else "na"
    return run_root / feature_set / f"q-{tag}" / pooling / f"fold{fold}" / f"seed{seed}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-run-root", type=Path, default=DEFAULT_DATA_RUN_ROOT)
    parser.add_argument("--folds", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    parser.add_argument("--resume", action="store_true", help="pass --resume to each experiment")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these feature sets")
    args = parser.parse_args(argv)

    jobs: list[tuple[list[str], Path, str]] = []
    for feature_set, quaternion, pooling, purpose in MATRIX:
        if args.only and feature_set not in args.only:
            continue
        for fold in args.folds:
            for seed in args.seeds:
                directory = experiment_dir(args.run_root, feature_set, quaternion,
                                           pooling, fold, seed)
                command = [
                    sys.executable, str(TRAINER),
                    "--run-root", str(args.run_root),
                    "--data-run-root", str(args.data_run_root),
                    "--fold", fold,
                    "--feature-set", feature_set,
                    "--quaternion-policy", quaternion,
                    "--pooling", pooling,
                    "--seed", str(seed),
                    "--device", args.device,
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                ]
                if args.resume:
                    command.append("--resume")
                jobs.append((command, directory, purpose))

    print(f"TASK-009B required sweep: {len(jobs)} experiments")
    print(f"run root : {args.run_root}")
    print(f"data root: {args.data_run_root}")
    print()

    started = time.perf_counter()
    completed = skipped = 0
    for position, (command, directory, purpose) in enumerate(jobs, start=1):
        status_path = directory / "status.json"
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if payload.get("status") == "COMPLETE" and not args.dry_run:
                print(f"[{position}/{len(jobs)}] SKIP (already COMPLETE) {directory.relative_to(args.run_root)}")
                skipped += 1
                continue
        print(f"[{position}/{len(jobs)}] {purpose}")
        print("    " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"\nFAILED with exit code {result.returncode}. "
                  f"Nothing after this ran. Fix, then re-run with --resume.")
            return result.returncode
        completed += 1

    elapsed = time.perf_counter() - started
    print()
    print(f"sweep finished: {completed} run, {skipped} already complete, "
          f"{elapsed/3600:.2f} h")
    if not args.dry_run:
        print("STATUS:")
        print("COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
