#!/usr/bin/env python3
"""Validate TASK-006 virtual-glove outputs against a TASK-005 run.

Example:

    python scripts/validate_task006_virtual_glove.py \
      --kinematics-run /path/to/task005-run \
      --virtual-glove-run /path/to/task006-run \
      --output-json /path/to/summary.json \
      --output-csv /path/to/channel-summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.virtual_glove_qa.contract import ContractError  # noqa: E402
from evaluation.virtual_glove_qa.validator import validate_runs, write_csv, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics-run", type=Path, required=True, help="TASK-005 run (read-only)")
    parser.add_argument("--virtual-glove-run", type=Path, required=True, help="TASK-006 virtual-glove run")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary, csv_rows = validate_runs(args.kinematics_run, args.virtual_glove_run)
    except ContractError as error:
        print(f"VALIDATION FAILURE: {error}", file=sys.stderr)
        return 2

    write_json(args.output_json, summary)
    write_csv(args.output_csv, csv_rows)

    print(f"schema passed       : {summary['schema_validation']['passed']}")
    print(f"alignment passed    : {summary['alignment']['passed']}")
    print(f"provenance passed   : {summary['provenance']['passed']}")
    print(f"layout passed       : {summary['sensor_layout']['passed']}")
    print(f"normalization issues: {summary['normalization']['count']}")
    print(f"validity issues     : {summary['validity_masks']['count']}")
    print(f"rotation issues     : {summary['rotation_quality']['count']}")
    print(f"ADC present         : {summary['adc']['present']}")
    print(f"verdict             : {summary['verdict']}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
