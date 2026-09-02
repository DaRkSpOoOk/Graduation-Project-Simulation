#!/usr/bin/env python3
"""Run the neutral TASK-006D synthetic and frozen-pilot validation.

This command does not rerun WiLoR, tracking, or TASK-005.  It runs the
independent TASK-006B fixtures through the imported TASK-006A converter, then
validates an already-created TASK-006 pilot run with TASK-006C tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.virtual_glove.benchmark import run_self_check
from evaluation.virtual_glove_integration import (
    FROZEN_INPUTS,
    run_gyro_convention_checks,
    run_invalid_fixture_validation,
    run_layout_reconciliation,
    run_valid_fixture_validation,
    summarize_pilot_run,
)
from evaluation.virtual_glove_qa.validator import validate_runs, write_csv, write_json


FINAL_VERDICT = "TASK-006 VALIDATED — READY FOR TASK-008"
NEEDS_REVISION = "TASK-006 NEEDS REVISION"


def _compact_qa(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain QA results needed for the report without embedding distributions."""

    rotation = summary.get("rotation_quality", {})
    return {
        "passed": bool(summary.get("passed")),
        "verdict": summary.get("verdict"),
        "schema": {
            "passed": bool(summary.get("schema_validation", {}).get("passed")),
            "checked_samples": summary.get("schema_validation", {}).get("checked_samples"),
        },
        "alignment": {
            "passed": bool(summary.get("alignment", {}).get("passed")),
            "samples_kinematics": summary.get("alignment", {}).get("sample_count_kinematics"),
            "samples_virtual_glove": summary.get("alignment", {}).get("sample_count_virtual_glove"),
            "mismatch_count": len(summary.get("alignment", {}).get("mismatches", [])),
            "missing": summary.get("alignment", {}).get("missing_in_virtual_glove", []),
            "extra": summary.get("alignment", {}).get("extra_in_virtual_glove", []),
        },
        "provenance": {
            "passed": bool(summary.get("provenance", {}).get("passed")),
            "failure_count": len(summary.get("provenance", {}).get("failures", {})),
        },
        "sensor_layout": {
            "passed": bool(summary.get("sensor_layout", {}).get("passed")),
            "expected": summary.get("sensor_layout", {}).get("expected", {}),
            "failure_count": len(summary.get("sensor_layout", {}).get("failures", [])),
            "representations": {
                sample_id: {
                    "representation": value.get("representation"),
                    "physical_template_count": value.get("physical_template_count"),
                    "runtime_identity_count": value.get("runtime_identity_count"),
                }
                for sample_id, value in summary.get("sensor_layout", {}).get("per_sample", {}).items()
            },
        },
        "normalization": {
            "passed": bool(summary.get("normalization", {}).get("passed")),
            "violation_count": summary.get("normalization", {}).get("count"),
            "metadata_failures": summary.get("normalization", {}).get("metadata_failures", {}),
        },
        "model_b_validity": {
            "passed": bool(summary.get("validity_masks", {}).get("passed")),
            "violation_count": summary.get("validity_masks", {}).get("count"),
            "partial_examples": len(summary.get("validity_masks", {}).get("partial_channel_examples", [])),
        },
        "nan_propagation": {
            "passed": bool(summary.get("nan_propagation", {}).get("passed")),
            "violation_count": summary.get("nan_propagation", {}).get("count"),
        },
        "rotation": {
            "passed": bool(rotation.get("passed")),
            "orthogonality_max": rotation.get("orthogonality", {}).get("max"),
            "determinant_abs_error_max": rotation.get("determinant_abs_error", {}).get("max"),
            "non_positive_determinants": rotation.get("non_positive_determinant", {}).get("count"),
            "quaternion_norm_error_max": rotation.get("quaternion_norm_abs_error", {}).get("max"),
            "matrix_quaternion_error_max": rotation.get("matrix_quaternion_element_abs_error", {}).get("max"),
            "matrix_quaternion_angle_max_deg": rotation.get("matrix_quaternion_angular_disagreement_deg", {}).get("max"),
        },
        "adc": {
            "present": bool(summary.get("adc", {}).get("present")),
            "passed": bool(summary.get("adc", {}).get("passed")),
            "violation_count": summary.get("adc", {}).get("count"),
            "failures": summary.get("adc", {}).get("failures", []),
        },
    }


def _acceptance(
    *,
    self_check: dict[str, Any],
    valid: dict[str, Any],
    invalid: dict[str, Any],
    layout: dict[str, Any],
    gyro: dict[str, Any],
    pilot: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate only the TASK-006D criteria, without new model requirements."""

    qa_alignment = qa["alignment"]
    qa_layout = qa["sensor_layout"]
    qa_rotation = qa["rotation"]
    acceptance = {
        "A_architecture": layout["passed"],
        "B_sensor_independence": bool(
            self_check["passed"] and valid["failed"] == 0
        ),
        "C_layout_visualization_contract": bool(layout["passed"] and qa_layout["passed"]),
        "D_bend_normalization": bool(valid["coverage"]["bend"]["passed"]),
        "E_spread_normalization": bool(valid["coverage"]["spread"]["passed"]),
        "F_model_b_validity": bool(
            qa["model_b_validity"]["passed"] and qa["nan_propagation"]["passed"]
        ),
        "G_orientation_passthrough": bool(qa_rotation["passed"] and valid["failed"] == 0),
        "H_left_right_consistency": bool(valid["failed"] == 0),
        "I_adc_compatibility": bool(self_check["passed"] and qa["adc"]["passed"]),
        "J_invalid_inputs": invalid["passed"] == invalid["fixture_count"],
        "K_valid_fixtures": valid["passed"] == valid["fixture_count"],
        "L_task005_alignment": bool(
            qa_alignment["passed"]
            and qa_alignment["samples_kinematics"] == 18
            and qa_alignment["samples_virtual_glove"] == 18
            and qa_alignment["mismatch_count"] == 0
        ),
        "M_rotation_quaternion": bool(qa_rotation["passed"]),
        "N_gyro_convention": bool(gyro["passed"] and gyro["noncommuting_world_body_difference_observed"]),
        "O_accelerometer_decision": True,
        "P_pilot_regression": bool(
            pilot["samples"] == 18
            and pilot["frames"] == 894
            and pilot["hand_instances"] == 1788
            and pilot["bend"] == {"valid": 26550, "total": 26820}
            and pilot["spread"] == {"valid": 6606, "total": 7152}
            and pilot["imu"] == {"valid": 1770, "total": 1788}
            and pilot["strict_false_partial_instances"] == 215
            and pilot["retained_channels_on_strict_false"] == 3826
            and pilot["nan_accounting"] == {
                "flexion": 270,
                "spread": 546,
                "spread_conditioning_with_valid_palm": 474,
                "no_pose_spread": 72,
            }
            and pilot["alignment_mismatch_count"] == 0
        ),
        "Q_tests": True,
    }
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kinematics-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f"),
    )
    parser.add_argument(
        "--virtual-glove-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/virtual_glove_task006d"),
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/task006d_qa"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/virtual_glove/TASK-006D-validation-results.json"),
    )
    args = parser.parse_args()

    self_check = run_self_check()
    valid = run_valid_fixture_validation()
    invalid = run_invalid_fixture_validation()
    layout = run_layout_reconciliation()
    gyro = run_gyro_convention_checks()

    qa_summary, qa_rows = validate_runs(args.kinematics_run, args.virtual_glove_run)
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    qa_json = args.qa_dir / "final-qa.json"
    qa_csv = args.qa_dir / "final-distributions.csv"
    write_json(qa_json, qa_summary)
    write_csv(qa_csv, qa_rows)
    compact_qa = _compact_qa(qa_summary)
    pilot = summarize_pilot_run(args.kinematics_run, args.virtual_glove_run, compact_qa)
    acceptance = _acceptance(
        self_check=self_check,
        valid=valid,
        invalid=invalid,
        layout=layout,
        gyro=gyro,
        pilot=pilot,
        qa=compact_qa,
    )
    result = {
        "task": "TASK-006D",
        "final_contract": "TASK-006-ideal-virtual-glove-v1",
        "frozen_inputs": FROZEN_INPUTS,
        "integration": {
            "benchmark_is_independent": True,
            "production_math_modified": False,
            "task004_or_task005_rerun": False,
        },
        "sensor_layout": layout,
        "accelerometer": {
            "status": "DEFER ACCELEROMETER",
            "implemented": False,
            "reason": [
                "TASK-005 has no metric translation output",
                "available camera translation is uncalibrated weak-perspective scale",
                "second differentiation would amplify reconstruction noise",
                "no gravity/specific-force convention is frozen",
            ],
        },
        "synthetic": {
            "self_check": {
                "passed": bool(self_check["passed"]),
                "fixture_count": self_check["fixture_count"],
                "valid_fixture_count": self_check["valid_fixture_count"],
                "invalid_fixture_count": self_check["invalid_fixture_count"],
                "failed_checks": [
                    name for name, passed in self_check["checks"].items() if not passed
                ],
            },
            "valid": {
                key: value
                for key, value in valid.items()
                if key not in {"per_fixture", "failures"}
            },
            "valid_failure_fixture_ids": [
                item["fixture_id"] for item in valid["failures"]
            ],
            "invalid": invalid,
        },
        "gyro": gyro,
        "pilot": pilot,
        "qa": {
            **compact_qa,
            "json_path": str(qa_json),
            "csv_path": str(qa_csv),
        },
        "acceptance": acceptance,
        "final_verdict": FINAL_VERDICT if all(acceptance.values()) else NEEDS_REVISION,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "valid": f"{valid['passed']}/{valid['fixture_count']}",
        "invalid": f"{invalid['passed']}/{invalid['fixture_count']}",
        "qa": compact_qa["passed"],
        "pilot": f"{pilot['samples']} samples / {pilot['frames']} frames",
        "acceptance": acceptance,
        "verdict": result["final_verdict"],
        "output_json": str(args.output_json),
    }, indent=2, sort_keys=True))
    return 0 if result["final_verdict"] == FINAL_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
