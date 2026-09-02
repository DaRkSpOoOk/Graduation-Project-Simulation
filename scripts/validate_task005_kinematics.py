#!/usr/bin/env python3
"""Validate TASK-005 kinematics contract and diagnostics against TASK-004 tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.kinematics_qa.contract import ContractError  # noqa: E402
from evaluation.kinematics_qa.validator import (  # noqa: E402
    validate_runs,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-run", type=Path, required=True)
    parser.add_argument("--kinematics-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary, csv_rows = validate_runs(args.tracked_run, args.kinematics_run)
    except ContractError as error:
        print(f"VALIDATION FAILURE: {error}", file=sys.stderr)
        return 2

    write_json(args.output_json, summary)
    write_csv(args.output_csv, csv_rows)

    print(f"contract passed     : {summary['contract_validation']['passed']}")
    print(f"alignment passed    : {summary['tracking_alignment']['passed']}")
    print(f"invalid-mask issues : {summary['invalid_mask_violations']['count']}")
    print(f"non-finite issues   : {summary['non_finite_violations']['count']}")
    print(f"rotation det<=0     : {summary['rotation_errors']['determinant_non_positive']['count']}")
    print(f"suspicious values   : {summary['suspicious_value_flags']['count']}")
    print(f"verdict             : {summary['verdict']}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
