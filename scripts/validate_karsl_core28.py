#!/usr/bin/env python3
"""Run CPU-only final QA over an external TASK-008A Core-28 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.qa import DatasetQAError, validate_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_core28.csv")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("KARSL_DATA_ROOT", "datasets/external/karsl")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TASK008A_RUN_ROOT", "runs/task008a-karsl-core28")))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_run(args.manifest, args.data_root, args.run_root)
    except (DatasetQAError, FileNotFoundError, OSError, ValueError) as error:
        print(f"TASK-008A QA error: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["failed_samples"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
