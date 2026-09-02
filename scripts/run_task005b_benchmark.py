#!/usr/bin/env python3
"""Run the independent TASK-005B catalog self-check.

This command validates the fixture generator against its own analytic truth.
TASK-005D can reuse ``evaluate_sequence`` with a production adapter result;
this script intentionally does not import or execute a production extractor.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.kinematics import CONTRACT_TOLERANCES, KinematicsResult, build_benchmark_catalog, evaluate_sequence


def main() -> int:
    cases = build_benchmark_catalog()
    valid_cases = 0
    invalid_cases = 0
    scores = []
    for case in cases:
        sequence = case.generate()
        if not sequence.expected_valid:
            invalid_cases += 1
            continue
        valid_cases += 1
        result = KinematicsResult(
            sequence.flexion_deg,
            sequence.adjacent_spread_deg,
            sequence.palm_rotation_matrix,
            sequence.palm_quaternion_wxyz,
        )
        score = evaluate_sequence(result, sequence)
        if not all(score[name] for name in ("flexion_pass", "spread_pass", "orientation_pass", "quaternion_pass")):
            raise SystemExit(f"analytic self-check failed: {case.case_id}: {score}")
        scores.append(score)
    print(
        json.dumps(
            {
                "catalog_cases": len(cases),
                "valid_cases_self_checked": valid_cases,
                "invalid_cases": invalid_cases,
                "categories": dict(Counter(case.category for case in cases)),
                "tolerances": CONTRACT_TOLERANCES,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
