#!/usr/bin/env python3
"""Run the final TASK-005F contract and pilot revalidation.

This script does not run WiLoR or tracking.  It uses the existing E2-generated
kinematics run, the existing TASK-004D tracked run, the versioned TASK-005E1
synthetic contract, and the corrected neutral QA validator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.kinematics import (  # noqa: E402
    FINAL_CONTRACT_TOLERANCES,
    FROZEN_KINEMATICS_COMMIT,
    build_final_catalog,
    extract_production_sequence,
    validate_frozen_pilot_inputs,
)
from evaluation.kinematics_qa.validator import validate_runs  # noqa: E402
from scripts.run_task005e1_contract import run as run_final_contract  # noqa: E402


EXPECTED_SAMPLES = 18
EXPECTED_FRAMES = 894
EXPECTED_HAND_INSTANCES = EXPECTED_FRAMES * 2
EXPECTED_VALID_PALM = 1770
EXPECTED_STRICT_VALID = 1555
EXPECTED_PARTIAL = 215
EXPECTED_FLEXION_NAN = 270
EXPECTED_SPREAD_NAN = 546
EXPECTED_SPREAD_CONDITIONING_NAN = 474


def _max_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.max(array)) if array.size else None


def _run_tests() -> dict[str, Any]:
    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "evaluation", "tracking", "kinematics", "scripts", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    unit_output = f"{unit.stdout}\n{unit.stderr}".strip()
    test_count = None
    for line in unit_output.splitlines():
        if line.startswith("Ran ") and " tests" in line:
            try:
                test_count = int(line.split()[1])
            except (IndexError, ValueError):
                pass
    return {
        "unit_test_command": "python -m unittest discover -s tests -p 'test_*.py'",
        "unit_test_returncode": unit.returncode,
        "unit_test_count": test_count,
        "unit_test_output_tail": unit_output[-500:],
        "compileall_command": "python -m compileall -q evaluation tracking kinematics scripts tests",
        "compileall_returncode": compile_result.returncode,
        "compileall_output": f"{compile_result.stdout}\n{compile_result.stderr}".strip(),
        "passed": unit.returncode == 0 and compile_result.returncode == 0,
    }


def _npz_arrays_equal(first_path: Path, second_path: Path) -> bool:
    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        if set(first.files) != set(second.files):
            return False
        for name in first.files:
            left = first[name]
            right = second[name]
            if left.dtype.kind in "fc" or right.dtype.kind in "fc":
                if not np.array_equal(left, right, equal_nan=True):
                    return False
            elif not np.array_equal(left, right):
                return False
    return True


def _load_pilot_counts(
    kinematics_run: Path,
    previous_run: Path,
    sample_ids: list[str],
) -> dict[str, Any]:
    frames = 0
    hand_instances = 0
    valid_palm = 0
    strict_valid = 0
    flexion_nan = 0
    spread_nan = 0
    no_palm_all_nan = True
    new_palm_invalid = 0
    recovered_palm = 0
    npz_equal = True

    for sample_id in sample_ids:
        current_npz = kinematics_run / sample_id / "hand_kinematics.npz"
        previous_npz = previous_run / sample_id / "hand_kinematics.npz"
        with np.load(current_npz, allow_pickle=False) as current:
            frame_count = int(current["frame_index"].shape[0])
            palm = np.asarray(current["valid_palm_frame"], dtype=bool)
            strict = np.asarray(current["valid_kinematics"], dtype=bool)
            flexion = np.asarray(current["flexion_deg"], dtype=np.float64)
            spread = np.asarray(current["adjacent_spread_deg"], dtype=np.float64)
            frames += frame_count
            hand_instances += int(palm.size)
            valid_palm += int(palm.sum())
            strict_valid += int(strict.sum())
            flexion_nan += int((~np.isfinite(flexion)).sum())
            spread_nan += int((~np.isfinite(spread)).sum())
            no_palm = ~palm
            if no_palm.any():
                no_palm_all_nan = no_palm_all_nan and bool(
                    np.isnan(flexion[no_palm]).all()
                    and np.isnan(spread[no_palm]).all()
                    and np.isnan(current["palm_rotation_matrix"][no_palm]).all()
                    and np.isnan(current["palm_quaternion_wxyz"][no_palm]).all()
                )
        if previous_npz.is_file():
            with np.load(previous_npz, allow_pickle=False) as previous:
                previous_palm = np.asarray(previous["valid_palm_frame"], dtype=bool)
                new_palm_invalid += int((previous_palm & ~palm).sum())
                recovered_palm += int((~previous_palm & palm).sum())
            npz_equal = npz_equal and _npz_arrays_equal(current_npz, previous_npz)
        else:
            npz_equal = False

    strict_false_palm_true = valid_palm - strict_valid
    return {
        "samples": len(sample_ids),
        "frames": frames,
        "hand_instances": hand_instances,
        "valid_palm_frames": valid_palm,
        "invalid_from_no_pose_or_invalid_palm": hand_instances - valid_palm,
        "strict_valid_kinematics": strict_valid,
        "partial_spread_instances": strict_false_palm_true,
        "flexion_nan": flexion_nan,
        "spread_nan": spread_nan,
        "spread_nan_from_no_pose": (hand_instances - valid_palm) * 4,
        "spread_conditioning_nan": spread_nan - (hand_instances - valid_palm) * 4,
        "no_palm_all_derived_nan": no_palm_all_nan,
        "previous_run_npz_arrays_exactly_equal": npz_equal,
        "new_real_palm_frames_rejected": new_palm_invalid,
        "real_palm_frames_recovered": recovered_palm,
        "counts_match_expected": (
            len(sample_ids) == EXPECTED_SAMPLES
            and frames == EXPECTED_FRAMES
            and hand_instances == EXPECTED_HAND_INSTANCES
            and valid_palm == EXPECTED_VALID_PALM
            and strict_valid == EXPECTED_STRICT_VALID
            and strict_false_palm_true == EXPECTED_PARTIAL
            and flexion_nan == EXPECTED_FLEXION_NAN
            and spread_nan == EXPECTED_SPREAD_NAN
            and spread_nan - (hand_instances - valid_palm) * 4 == EXPECTED_SPREAD_CONDITIONING_NAN
        ),
    }


def _temporal_summary(qa_summary: dict[str, Any], pilot_summary: dict[str, Any]) -> dict[str, Any]:
    by_channel = qa_summary["temporal_statistics"]

    def pooled(name: str) -> dict[str, Any]:
        entries = [entry for entry in by_channel[name].values() if entry["count"]]
        maxima = [entry["maximum"] for entry in entries if entry["maximum"] is not None]
        values = [
            (entry["maximum"]["value"], entry["maximum"])
            for entry in entries
            if entry["maximum"] is not None
        ]
        worst = max(values, key=lambda pair: pair[0])[1] if values else None
        # The final output runner stores pooled p95/p99/counts. QA supplies the
        # exact sample/frame/channel location for the largest event.
        runner_name = {
            "flexion_abs_delta_deg": "flexion_deg",
            "spread_abs_delta_deg": "adjacent_spread_deg",
            "palm_orientation_abs_delta_deg": "palm_orientation_deg",
        }[name]
        runner = pilot_summary.get("temporal_change", {}).get(runner_name, {})
        return {
            "count": int(runner.get("count", sum(entry["count"] for entry in entries))),
            "p95": runner.get("p95"),
            "p99": runner.get("p99"),
            "maximum": runner.get("max"),
            "maximum_event": worst,
            "channel_count": len(entries),
        }

    return {
        "flexion": pooled("flexion_abs_delta_deg"),
        "spread": pooled("spread_abs_delta_deg"),
        "palm_orientation": pooled("palm_orientation_abs_delta_deg"),
    }


def _coincident_mcp_validation() -> dict[str, Any]:
    """Verify the E2 invalid-geometry contract without fixture hardcoding.

    The production result is inspected directly so the final report can
    distinguish a generic 6/6 invalid-case flag from the stronger E2
    requirement: an invalid palm frame must emit all derived float channels as
    NaN and expose a coincidence flag.
    """

    case = next(
        candidate
        for candidate in build_final_catalog()
        if "coincident" in candidate.case_id.lower()
    )
    production = extract_production_sequence(case.generate())
    result = production.result
    all_derived_nan = bool(
        np.isnan(result.flexion_deg).all()
        and np.isnan(result.adjacent_spread_deg).all()
        and np.isnan(result.palm_rotation_matrix).all()
        and np.isnan(result.palm_quaternion_wxyz).all()
    )
    flag_names = sorted(
        {
            flag
            for row in production.flags
            for hand_flags in row
            for flag in hand_flags
        }
    )
    coincident_flag_present = any(
        flag.startswith("PALM_LANDMARKS_COINCIDENT_") for flag in flag_names
    )
    passed = bool(
        not production.valid_palm_frame.any()
        and not production.valid_kinematics.any()
        and all_derived_nan
        and coincident_flag_present
    )
    return {
        "case_id": case.case_id,
        "valid_palm_frame": production.valid_palm_frame.tolist(),
        "valid_kinematics": production.valid_kinematics.tolist(),
        "all_derived_floats_nan": all_derived_nan,
        "flag_names": flag_names,
        "coincident_flag_present": coincident_flag_present,
        "pass": passed,
    }


def _pilot_qa(
    tracked_run: Path,
    kinematics_run: Path,
    previous_run: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_contract = validate_frozen_pilot_inputs(
        tracked_run,
        kinematics_run,
        expected_implementation_commit=FROZEN_KINEMATICS_COMMIT,
    )
    qa_summary, _ = validate_runs(tracked_run, kinematics_run)
    counts = _load_pilot_counts(kinematics_run, previous_run, input_contract["sample_ids"])

    previous_summary_path = previous_run / "kinematics_summary.json"
    with (kinematics_run / "kinematics_summary.json").open(encoding="utf-8") as handle:
        pilot_summary = json.load(handle)
    if previous_summary_path.is_file():
        with previous_summary_path.open(encoding="utf-8") as handle:
            previous_summary = json.load(handle)
    else:
        previous_summary = {}

    alignment = qa_summary["tracking_alignment"]
    alignment_mismatches = alignment["mismatches"]
    structural = qa_summary["sample_frame_counts"]
    rotation = qa_summary["rotation_errors"]
    quaternion = qa_summary["quaternion_errors"]
    limits = FINAL_CONTRACT_TOLERANCES
    rotation_pass = bool(
        rotation["non_finite_matrices"] == 0
        and rotation["determinant_non_positive"]["count"] == 0
        and rotation["orthogonality"]["max"] <= limits["rotation_matrix_orthogonality"]
        and rotation["determinant_abs_error"]["max"] <= limits["rotation_matrix_determinant"]
        and quaternion["non_finite_quaternions"] == 0
        and quaternion["norm_abs_error"]["max"] <= limits["quaternion_norm"]
        and quaternion["matrix_quaternion_element_abs_error"]["max"] <= limits["matrix_quaternion_consistency"]
    )
    qa = {
        "passed": bool(qa_summary["passed"]),
        "validity_contract": qa_summary["validity_contract"],
        "structural": {
            "samples": structural["kinematics_samples"],
            "frames": structural["total_kinematics_frames"],
            "samples_exact": structural["kinematics_samples"] == EXPECTED_SAMPLES,
            "frames_exact": structural["total_kinematics_frames"] == EXPECTED_FRAMES,
        },
        "alignment": {
            "passed": bool(alignment["passed"]),
            "mismatch_count": len(alignment_mismatches),
            "tracking_state_mismatches": sum(item["field"] == "tracking_state_code" for item in alignment_mismatches),
            "source_provenance_mismatches": sum(item["field"] == "source_raw_detection_index" for item in alignment_mismatches),
            "frame_index_mismatches": sum(item["field"] == "frame_index" for item in alignment_mismatches),
            "timestamp_mismatches": sum(item["field"] == "timestamp_seconds" for item in alignment_mismatches),
        },
        "model_b": {
            "invalid_mask_violations": qa_summary["invalid_mask_violations"]["count"],
            "non_finite_violations": qa_summary["non_finite_violations"]["count"],
            "partial_channel_instances": len(qa_summary["invalid_mask_violations"]["partial_channel_instances"]),
            "passed": qa_summary["invalid_mask_violations"]["count"] == 0
            and qa_summary["non_finite_violations"]["count"] == 0,
        },
        "rotation": {
            "max_orthogonality_error": rotation["orthogonality"]["max"],
            "max_abs_det_minus_1": rotation["determinant_abs_error"]["max"],
            "non_positive_determinants": rotation["determinant_non_positive"]["count"],
            "pass": rotation_pass,
        },
        "quaternion": {
            "max_norm_error": quaternion["norm_abs_error"]["max"],
            "max_matrix_quaternion_element_error": quaternion["matrix_quaternion_element_abs_error"]["max"],
            "max_matrix_quaternion_angular_error_deg": quaternion["matrix_quaternion_angular_disagreement_deg"]["max"],
            "pass": rotation_pass,
        },
        "suspicious_flexion_values": qa_summary["suspicious_value_flags"]["count"],
    }
    pilot = {
        "input_contract": input_contract,
        "counts": counts,
        "source_summary_implementation_commit": pilot_summary.get("implementation_commit"),
        "source_summary": {
            "videos": pilot_summary.get("videos"),
            "frames": pilot_summary.get("frames"),
            "valid_palm_frame_instances": pilot_summary.get("valid_palm_frame_instances"),
            "valid_hand_instances": pilot_summary.get("valid_hand_instances"),
            "nan_counts": pilot_summary.get("nan_counts"),
        },
        "previous_summary_implementation_commit": previous_summary.get("implementation_commit"),
        "temporal": _temporal_summary(qa_summary, pilot_summary),
    }
    return pilot, qa


def _acceptance(
    synthetic: dict[str, Any],
    self_check: dict[str, Any],
    pilot: dict[str, Any],
    qa: dict[str, Any],
    tests: dict[str, Any],
) -> dict[str, bool]:
    invalid = synthetic["invalid_geometry"]
    spread = synthetic["spread"]
    acceptance = {
        "A_final_analytic_flexion": synthetic["flexion"]["case_pass_count"] == 80,
        "B_final_analytic_spread": (
            spread["case_pass_count"] == 80
            and spread["numeric_pass_values"] == spread["numeric_defined_values"]
            and spread["expected_nan_values"] == 18
            and spread["unexpected_nan_values"] == 0
            and spread["unexpected_finite_values"] == 0
        ),
        "C_mirror_consistency": bool(self_check["checks"].get("mirror_local_truth_is_equivalent")),
        "D_rigid_invariance": bool(self_check["checks"].get("rigid_transform_local_truth_is_invariant")),
        "E_palm_orientation": synthetic["orientation"]["case_pass_count"] == 80,
        "F_rotation_quality": bool(qa["rotation"]["pass"] and qa["quaternion"]["pass"]),
        "G_task004_alignment": bool(
            qa["alignment"]["passed"]
            and qa["alignment"]["tracking_state_mismatches"] == 0
            and qa["alignment"]["source_provenance_mismatches"] == 0
            and pilot["counts"]["samples"] == EXPECTED_SAMPLES
            and pilot["counts"]["frames"] == EXPECTED_FRAMES
        ),
        "H_missing_occlusion_policy": bool(pilot["counts"]["no_palm_all_derived_nan"]),
        "I_invalid_geometry": (
            invalid["case_pass_count"] == invalid["case_count"] == 6
            and bool(synthetic["coincident_mcp_validation"]["pass"])
        ),
        "J_model_b_validity": bool(qa["model_b"]["passed"]),
        "K_pilot_regression": bool(
            pilot["counts"]["counts_match_expected"]
            and pilot["counts"]["new_real_palm_frames_rejected"] == 0
            and pilot["counts"]["previous_run_npz_arrays_exactly_equal"]
        ),
        "L_tests": bool(tests["passed"]),
    }
    return acceptance


def run(
    tracked_run: Path,
    kinematics_run: Path,
    previous_run: Path,
    output_path: Path,
) -> dict[str, Any]:
    # Use a temporary result path so the historical E1 result file is never
    # rewritten by this final validation.
    temporary_result = ROOT / ".task005f_e1_result.tmp.json"
    try:
        e1_result = run_final_contract(temporary_result)
    finally:
        temporary_result.unlink(missing_ok=True)

    pilot, qa = _pilot_qa(tracked_run, kinematics_run, previous_run)
    tests = _run_tests()
    synthetic = e1_result["production_compatibility"]
    self_check = e1_result["self_check"]
    synthetic["coincident_mcp_validation"] = _coincident_mcp_validation()
    synthetic["invalid_case_results"] = [
        {
            "case_id": record["case_id"],
            "expected_reasons": record.get("invalid_reasons", []),
            "production_valid_palm_frame": record.get("production_valid_palm_frame"),
            "production_valid_kinematics": record.get("production_valid_kinematics"),
            "production_flag_names": record.get("production_flag_names", []),
            "invalid_case_pass": record.get("invalid_case_pass"),
        }
        for record in e1_result["case_results"]
        if not record["expected_valid"]
    ]
    acceptance = _acceptance(synthetic, self_check, pilot, qa, tests)
    data: dict[str, Any] = {
        "task": "TASK-005F",
        "final_verdict": "TASK-005 VALIDATED — READY FOR TASK-006" if all(acceptance.values()) else "TASK-005 FINAL VALIDATION FAILED",
        "final_inputs": {
            "task005d_commit": "b41bc1808d09b1987ebdcf417e1bdadc42962f6d",
            "task005e1_contract_commit": "e6029e35e49516389356fdca159f6dddc1bcfda2",
            "task005e2_production_commit": FROZEN_KINEMATICS_COMMIT,
            "task005f_integration_branch_base": "e6029e35e49516389356fdca159f6dddc1bcfda2",
            "task005e2_integrated_cherry_pick": "248dcf5660fc2aa9237adf081a4b45ff5a021b5b",
        },
        "artifact_paths": {
            "tracked_run": str(tracked_run),
            "final_kinematics_run": str(kinematics_run),
            "previous_kinematics_run": str(previous_run),
            "qa_json": "/home/hatim/graduation-project-runs/task005f_qa/final-qa.json",
            "qa_distributions_csv": "/home/hatim/graduation-project-runs/task005f_qa/final-distributions.csv",
        },
        "final_contract_version": e1_result["contract_version"],
        "synthetic_benchmark": {
            "catalog_count": self_check["catalog_count"],
            "valid_case_count": self_check["valid_case_count"],
            "invalid_case_count": self_check["invalid_case_count"],
            "self_check": self_check,
            "flexion": synthetic["flexion"],
            "spread": synthetic["spread"],
            "orientation": synthetic["orientation"],
            "quaternion": synthetic["quaternion"],
            "invalid_geometry": synthetic["invalid_geometry"],
            "invalid_case_results": synthetic["invalid_case_results"],
            "coincident_mcp_validation": synthetic["coincident_mcp_validation"],
            "mirror_pass": bool(self_check["checks"].get("mirror_local_truth_is_equivalent")),
            "rigid_invariance_pass": bool(self_check["checks"].get("rigid_transform_local_truth_is_invariant")),
            "orientation_mappings": e1_result["orientation_mappings"],
        },
        "pilot": pilot,
        "qa": qa,
        "tests": tests,
        "acceptance_criteria": acceptance,
        "tolerances": e1_result["tolerances"],
        "no_model_or_tracking_rerun": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d"),
    )
    parser.add_argument(
        "--kinematics-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f"),
    )
    parser.add_argument(
        "--previous-kinematics-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/kinematics/TASK-005F-validation-results.json",
    )
    args = parser.parse_args()
    data = run(args.tracked_run, args.kinematics_run, args.previous_kinematics_run, args.output_json)
    print(
        json.dumps(
            {
                "verdict": data["final_verdict"],
                "implementation_commit": data["final_inputs"]["task005e2_production_commit"],
                "synthetic": {
                    "valid": data["synthetic_benchmark"]["valid_case_count"],
                    "invalid": data["synthetic_benchmark"]["invalid_case_count"],
                    "invalid_pass": data["synthetic_benchmark"]["invalid_geometry"]["case_pass_count"],
                },
                "pilot": data["pilot"]["counts"],
                "qa": data["qa"],
                "acceptance": data["acceptance_criteria"],
                "tests": data["tests"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if data["final_verdict"] == "TASK-005 VALIDATED — READY FOR TASK-006" else 1


if __name__ == "__main__":
    raise SystemExit(main())
