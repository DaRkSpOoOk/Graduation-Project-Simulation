#!/usr/bin/env python3
"""Run the independent TASK-006B ideal virtual-glove benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.virtual_glove import run_self_check  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/virtual_glove/TASK-006B-benchmark-results.json",
    )
    args = parser.parse_args()
    result = run_self_check()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "fixture_count": result["fixture_count"],
                "valid_fixture_count": result["valid_fixture_count"],
                "invalid_fixture_count": result["invalid_fixture_count"],
                "failed_checks": [name for name, passed in result["checks"].items() if not passed],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
