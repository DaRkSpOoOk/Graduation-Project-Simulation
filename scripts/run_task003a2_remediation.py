#!/usr/bin/env python3
"""Validate frozen runs and emit extractor-neutral pilot metrics.

All run and video paths are explicit arguments. In particular, this script
never searches for or glob-selects a WiLoR run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.comparison.harmonized_metrics import aggregate_metrics, evaluate_video
from evaluation.comparison.loaders import validate_all_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--mediapipe-run", type=Path, required=True)
    parser.add_argument("--wilor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        contract, mediapipe, wilor = validate_all_inputs(
            args.manifest,
            video_root=args.video_root,
            mediapipe_run=args.mediapipe_run,
            wilor_run=args.wilor_run,
        )
        output: dict[str, object] = {
            "manifest": str(contract.path),
            "manifest_sha256": contract.sha256,
            "expected_frames": 894,
            "systems": {},
        }
        for run in (mediapipe, wilor):
            per_video = [
                evaluate_video(sample_id, run.records_by_sample[sample_id], run.frame_counts[sample_id])
                for sample_id in contract.sample_ids
            ]
            output["systems"][run.system] = {
                "run_dir": str(run.run_dir),
                "validated_total_frames": run.total_frames,
                "per_video": per_video,
                "aggregate": aggregate_metrics(per_video),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "validation": "PASS", "videos": 18, "frames": 894}, indent=2))
    except Exception as error:  # CLI boundary: input failures are hard failures.
        print(f"TASK-003A2 remediation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
