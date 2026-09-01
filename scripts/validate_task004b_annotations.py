#!/usr/bin/env python3
"""Validate and summarize the independent TASK-004B annotation CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.annotations import annotation_statistics, validate_against_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("evaluation/annotations/task004_hand_identity_visibility.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/karsl_milestone1_pilot.csv"),
    )
    args = parser.parse_args()
    rows = validate_against_manifest(args.annotations, args.manifest)
    print(json.dumps(annotation_statistics(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
