#!/usr/bin/env python3
"""Run TASK-005D synthetic integration and frozen-pilot validation.

This harness does not rerun WiLoR or TASK-004.  It calls the frozen
TASK-005A per-frame extractor on TASK-005B's generated fixtures, and it reads
the existing pilot kinematics/tracking runs for neutral QA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.kinematics import (  # noqa: E402
    CONTRACT_TOLERANCES,
    build_benchmark_catalog,
    extract_production_sequence,
    rotation_angle_error_deg,
    validate_frozen_pilot_inputs,
    validate_result,
)
from evaluation.kinematics_qa.validator import validate_runs  # noqa: E402


TRACKS = ("LEFT", "RIGHT")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = ("proximal", "middle", "distal")
SPREADS = ("thumb-index", "index-middle", "middle-ring", "ring-pinky")

# These matrices follow directly from the two frozen documentation blocks.
# TASK-005B's canonical right basis is I and its left basis is diag(-1,1,-1).
# For the synthetic palm's wrist/MCP landmarks, TASK-005A's [lateral,normal,
# distal] construction produces this constant P before the benchmark's global
# rotation is applied.  The adapter is diagnostic only; raw production R is
# never overwritten.
BENCHMARK_BASIS = {
    "RIGHT": np.eye(3, dtype=np.float64),
    "LEFT": np.diag([-1.0, 1.0, -1.0]),
}
PRODUCTION_SYNTHETIC_PALM_BASIS = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SIDE_FIXED_MAPPINGS = {
    side: PRODUCTION_SYNTHETIC_PALM_BASIS.T @ basis
    for side, basis in BENCHMARK_BASIS.items()
}


def _finite_values(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)[np.isfinite(array)]


def _max_or_none(values: Iterable[float]) -> float | None:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return None
    return float(np.max(values_array))


def _error_summary(predicted: np.ndarray, expected: np.ndarray, limit: float) -> dict[str, Any]:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    mask = np.isfinite(predicted_array) & np.isfinite(expected_array)
    errors = np.abs(predicted_array - expected_array)
    finite_errors = errors[mask]
    return {
        "finite_values": int(mask.sum()),
        "total_values": int(mask.size),
        "non_finite_values": int((~mask).sum()),
        "max_error": _max_or_none(finite_errors),
        "pass": bool(mask.all() and np.all(finite_errors <= limit)),
    }


def _matrix_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            rotation_angle_error_deg(predicted[frame, track], expected[frame, track])
            for frame in range(predicted.shape[0])
            for track in range(2)
        ],
        dtype=np.float64,
    )


def _quaternion_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    errors: list[float] = []
    for frame in range(predicted.shape[0]):
        for track in range(2):
            first = np.asarray(predicted[frame, track], dtype=np.float64)
            second = np.asarray(expected[frame, track], dtype=np.float64)
            first = first / np.linalg.norm(first)
            second = second / np.linalg.norm(second)
            cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
            errors.append(float(np.degrees(2.0 * np.arccos(cosine))))
    return np.asarray(errors, dtype=np.float64)


def _case_result(case: Any) -> dict[str, Any]:
    sequence = case.generate()
    adapted = extract_production_sequence(sequence)
    result = adapted.result
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "expected_valid": bool(case.expected_valid),
        "production_valid_kinematics": adapted.valid_kinematics.tolist(),
        "production_valid_palm_frame": adapted.valid_palm_frame.tolist(),
        "production_all_public_channels_finite": adapted.all_channels_finite,
    }

    if not case.expected_valid:
        record.update(
            {
                "invalid_reasons": list(sequence.invalid_reasons),
                "production_flags": [[list(flags) for flags in row] for row in adapted.flags],
                "invalid_hand_rejected_or_flagged": bool(
                    not adapted.valid_kinematics.any()
                    or any(adapted.flags[frame][track] for frame in range(len(adapted.flags)) for track in range(2))
                ),
                "invalid_case_pass": bool(not adapted.valid_kinematics.any()),
            }
        )
        return record

    flexion = np.asarray(result.flexion_deg)
    spread = np.asarray(result.adjacent_spread_deg)
    rotation = np.asarray(result.palm_rotation_matrix)
    quaternion = np.asarray(result.palm_quaternion_wxyz)
    flex_summary = _error_summary(
        flexion,
        sequence.flexion_deg,
        CONTRACT_TOLERANCES["known_flexion_abs_error_deg"],
    )
    spread_summary = _error_summary(
        spread,
        sequence.adjacent_spread_deg,
        CONTRACT_TOLERANCES["known_spread_abs_error_deg"],
    )
    orientation_errors = _matrix_errors(rotation, sequence.palm_rotation_matrix)
    quaternion_errors = _quaternion_errors(quaternion, sequence.palm_quaternion_wxyz)
    record.update(
        {
            "flexion": flex_summary,
            "spread": spread_summary,
            "orientation": {
                "finite_values": int(np.isfinite(rotation).all(axis=(2, 3)).sum()),
                "total_values": int(rotation.shape[0] * rotation.shape[1]),
                "max_error_deg": _max_or_none(orientation_errors),
                "pass": bool(np.isfinite(rotation).all() and np.all(orientation_errors <= 1.0)),
            },
            "quaternion": {
                "finite_values": int(np.isfinite(quaternion).all(axis=2).sum()),
                "total_values": int(quaternion.shape[0] * quaternion.shape[1]),
                "max_error_deg": _max_or_none(quaternion_errors),
                "pass": bool(np.isfinite(quaternion).all() and np.all(quaternion_errors <= 1.0)),
            },
            "strict_result_contract": None,
        }
    )
    try:
        validate_result(result, expected_frames=sequence.joints.shape[0])
        record["strict_result_contract"] = True
    except Exception as error:  # benchmark contract errors are part of the report
        record["strict_result_contract"] = False
        record["strict_result_contract_error"] = str(error)
    record["full_case_pass"] = bool(
        record["strict_result_contract"]
        and record["flexion"]["pass"]
        and record["spread"]["pass"]
        and record["orientation"]["pass"]
        and record["quaternion"]["pass"]
    )
    return record


def _category_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(record["category"]) for record in case_results})
    summary: dict[str, Any] = {}
    for category in categories:
        records = [record for record in case_results if record["category"] == category]
        valid_records = [record for record in records if record["expected_valid"]]
        summary[category] = {
            "cases": len(records),
            "valid_cases": len(valid_records),
            "invalid_cases": sum(not record["expected_valid"] for record in records),
            "full_case_pass": sum(bool(record.get("full_case_pass")) for record in valid_records),
            "flexion_pass": sum(bool(record.get("flexion", {}).get("pass")) for record in valid_records),
            "spread_pass": sum(bool(record.get("spread", {}).get("pass")) for record in valid_records),
            "orientation_pass": sum(bool(record.get("orientation", {}).get("pass")) for record in valid_records),
            "quaternion_pass": sum(bool(record.get("quaternion", {}).get("pass")) for record in valid_records),
        }
    return summary


def _worst_case(case_results: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates: list[tuple[float, str]] = []
    for record in case_results:
        if not record.get("expected_valid"):
            continue
        section = record.get(metric, {})
        value = section.get("max_error_deg", section.get("max_error"))
        if value is not None:
            candidates.append((float(value), str(record["case_id"])))
    if not candidates:
        return None
    value, case_id = max(candidates)
    return {"case_id": case_id, "value": value}


def _local_rigid_invariance(cases: dict[str, dict[str, Any]], baseline_id: str, ids: list[str]) -> dict[str, Any]:
    baseline = cases[baseline_id]
    baseline_sequence = baseline["sequence"]
    baseline_result = baseline["adapted"].result
    entries: list[dict[str, Any]] = []
    for case_id in ids:
        current = cases[case_id]
        current_result = current["adapted"].result
        flex_error = np.abs(current_result.flexion_deg - baseline_result.flexion_deg)
        spread_error = np.abs(current_result.adjacent_spread_deg - baseline_result.adjacent_spread_deg)
        flex_mask = np.isfinite(flex_error)
        spread_mask = np.isfinite(spread_error)
        entries.append(
            {
                "case_id": case_id,
                "max_flexion_delta_deg": _max_or_none(flex_error[flex_mask]),
                "max_spread_delta_deg": _max_or_none(spread_error[spread_mask]),
            }
        )
    all_flex = [entry["max_flexion_delta_deg"] for entry in entries if entry["max_flexion_delta_deg"] is not None]
    all_spread = [entry["max_spread_delta_deg"] for entry in entries if entry["max_spread_delta_deg"] is not None]
    return {
        "baseline": baseline_id,
        "cases": entries,
        "max_flexion_delta_deg": _max_or_none(all_flex),
        "max_spread_delta_deg": _max_or_none(all_spread),
        "pass": bool(
            all(value <= 1e-8 for value in all_flex)
            and all(value <= 1e-8 for value in all_spread)
        ),
    }


def run_synthetic_validation() -> dict[str, Any]:
    catalog = build_benchmark_catalog()
    if len(catalog) != 86:
        raise RuntimeError(f"frozen TASK-005B catalog changed: {len(catalog)} cases")
    valid_cases = [case for case in catalog if case.expected_valid]
    invalid_cases = [case for case in catalog if not case.expected_valid]
    if len(valid_cases) != 80 or len(invalid_cases) != 6:
        raise RuntimeError("frozen TASK-005B validity split changed")

    case_results = [_case_result(case) for case in catalog]
    generated: dict[str, dict[str, Any]] = {}
    for case in valid_cases:
        sequence = case.generate()
        generated[case.case_id] = {
            "sequence": sequence,
            "adapted": extract_production_sequence(sequence),
        }

    flex_errors: list[float] = []
    spread_errors: list[float] = []
    orientation_errors: list[float] = []
    quaternion_errors: list[float] = []
    prox_errors: list[float] = []
    nonprox_errors: list[float] = []
    prox_pass = 0
    prox_total = 0
    nonprox_pass = 0
    nonprox_total = 0
    for case in valid_cases:
        sequence = generated[case.case_id]["sequence"]
        result = generated[case.case_id]["adapted"].result
        flex_error = np.abs(result.flexion_deg - sequence.flexion_deg)
        spread_error = np.abs(result.adjacent_spread_deg - sequence.adjacent_spread_deg)
        flex_mask = np.isfinite(flex_error)
        spread_mask = np.isfinite(spread_error)
        flex_errors.extend(flex_error[flex_mask].tolist())
        spread_errors.extend(spread_error[spread_mask].tolist())
        prox = flex_error[:, :, :, 0]
        nonprox = flex_error[:, :, :, 1:]
        prox_errors.extend(prox[np.isfinite(prox)].tolist())
        nonprox_errors.extend(nonprox[np.isfinite(nonprox)].tolist())
        prox_pass += int(np.sum(np.isfinite(prox) & (prox <= CONTRACT_TOLERANCES["known_flexion_abs_error_deg"])))
        prox_total += int(np.isfinite(prox).sum())
        nonprox_pass += int(np.sum(np.isfinite(nonprox) & (nonprox <= CONTRACT_TOLERANCES["known_flexion_abs_error_deg"])))
        nonprox_total += int(np.isfinite(nonprox).sum())
        orientation_errors.extend(_matrix_errors(result.palm_rotation_matrix, sequence.palm_rotation_matrix).tolist())
        quaternion_errors.extend(_quaternion_errors(result.palm_quaternion_wxyz, sequence.palm_quaternion_wxyz).tolist())

    mirror_entries: list[dict[str, Any]] = []
    for case in valid_cases:
        if not case.mirror_equivalent:
            continue
        adapted = generated[case.case_id]["adapted"].result
        flex_delta = np.abs(adapted.flexion_deg[:, 0] - adapted.flexion_deg[:, 1])
        spread_delta = np.abs(adapted.adjacent_spread_deg[:, 0] - adapted.adjacent_spread_deg[:, 1])
        flex_delta = flex_delta[np.isfinite(flex_delta)]
        spread_delta = spread_delta[np.isfinite(spread_delta)]
        mirror_entries.append(
            {
                "case_id": case.case_id,
                "max_flexion_delta_deg": _max_or_none(flex_delta),
                "max_spread_delta_deg": _max_or_none(spread_delta),
            }
        )
    mirror_values = [
        value
        for entry in mirror_entries
        for value in (entry["max_flexion_delta_deg"], entry["max_spread_delta_deg"])
        if value is not None
    ]

    orientation_mapping: dict[str, Any] = {
        "production_synthetic_palm_basis_P": PRODUCTION_SYNTHETIC_PALM_BASIS.tolist(),
        "benchmark_basis": {side: matrix.tolist() for side, matrix in BENCHMARK_BASIS.items()},
        "side_fixed_mappings_C": {side: matrix.tolist() for side, matrix in SIDE_FIXED_MAPPINGS.items()},
        "single_fixed_mapping_exists": bool(np.allclose(SIDE_FIXED_MAPPINGS["LEFT"], SIDE_FIXED_MAPPINGS["RIGHT"], atol=1e-12)),
        "single_mapping_diagnostic": {},
        "per_side_mapping_max_error_deg": {},
    }
    all_mapping_residuals = {"LEFT": [], "RIGHT": []}
    for candidate_name, candidate in (("C_RIGHT", SIDE_FIXED_MAPPINGS["RIGHT"]), ("C_LEFT", SIDE_FIXED_MAPPINGS["LEFT"])):
        candidate_errors: list[float] = []
        for case in valid_cases:
            sequence = generated[case.case_id]["sequence"]
            production = generated[case.case_id]["adapted"].result.palm_rotation_matrix
            mapped = np.einsum("ftij,jk->ftik", production, candidate)
            for frame in range(mapped.shape[0]):
                for track in range(2):
                    candidate_errors.append(
                        rotation_angle_error_deg(mapped[frame, track], sequence.palm_rotation_matrix[frame, track])
                    )
        orientation_mapping["single_mapping_diagnostic"][candidate_name] = {
            "max_error_deg": _max_or_none(candidate_errors),
            "right_max_error_deg": _max_or_none(candidate_errors[1::2]),
            "left_max_error_deg": _max_or_none(candidate_errors[0::2]),
        }
    for track, side in enumerate(TRACKS):
        mapping = SIDE_FIXED_MAPPINGS[side]
        errors: list[float] = []
        for case in valid_cases:
            sequence = generated[case.case_id]["sequence"]
            production = generated[case.case_id]["adapted"].result.palm_rotation_matrix
            mapped = np.einsum("ftij,jk->ftik", production, mapping)
            for frame in range(mapped.shape[0]):
                errors.append(rotation_angle_error_deg(mapped[frame, track], sequence.palm_rotation_matrix[frame, track]))
        all_mapping_residuals[side].extend(errors)
        orientation_mapping["per_side_mapping_max_error_deg"][side] = _max_or_none(errors)
    orientation_mapping["single_mapping_diagnostic"]["C_RIGHT_applied_to_both"] = orientation_mapping["single_mapping_diagnostic"]["C_RIGHT"]
    orientation_mapping["single_mapping_diagnostic"]["C_LEFT_applied_to_both"] = orientation_mapping["single_mapping_diagnostic"]["C_LEFT"]
    orientation_mapping["per_side_mapping_pass"] = bool(
        all(value <= CONTRACT_TOLERANCES["known_orientation_error_deg"] for values in all_mapping_residuals.values() for value in values)
    )

    invalid_records = [record for record in case_results if not record["expected_valid"]]
    finite_valid_cases = sum(record["production_all_public_channels_finite"] for record in case_results if record["expected_valid"])
    conditioning_case_ids = sorted(
        record["case_id"]
        for record in case_results
        if record["expected_valid"]
        and any("SPREAD_DIRECTION_DEGENERATE" in flag for row in generated[record["case_id"]]["adapted"].flags for flagset in row for flag in flagset)
    )
    return {
        "catalog": {
            "total_cases": len(catalog),
            "valid_cases": len(valid_cases),
            "invalid_cases": len(invalid_cases),
            "category_summary": _category_summary(case_results),
        },
        "case_results": case_results,
        "aggregate_errors": {
            "flexion": {"finite_values": len(flex_errors), "max_error_deg": _max_or_none(flex_errors)},
            "spread": {"finite_values": len(spread_errors), "max_error_deg": _max_or_none(spread_errors)},
            "orientation": {"values": len(orientation_errors), "max_error_deg": _max_or_none(orientation_errors)},
            "quaternion": {"values": len(quaternion_errors), "max_error_deg": _max_or_none(quaternion_errors)},
            "worst_flexion": _worst_case(case_results, "flexion"),
            "worst_spread": _worst_case(case_results, "spread"),
            "worst_orientation": _worst_case(case_results, "orientation"),
            "worst_quaternion": _worst_case(case_results, "quaternion"),
            "failing_fixture_ids": {
                "full_case": [
                    record["case_id"]
                    for record in case_results
                    if record["expected_valid"] and not record.get("full_case_pass", False)
                ],
                "flexion": [
                    record["case_id"]
                    for record in case_results
                    if record["expected_valid"] and not record.get("flexion", {}).get("pass", False)
                ],
                "spread": [
                    record["case_id"]
                    for record in case_results
                    if record["expected_valid"] and not record.get("spread", {}).get("pass", False)
                ],
                "orientation": [
                    record["case_id"]
                    for record in case_results
                    if record["expected_valid"] and not record.get("orientation", {}).get("pass", False)
                ],
                "quaternion": [
                    record["case_id"]
                    for record in case_results
                    if record["expected_valid"] and not record.get("quaternion", {}).get("pass", False)
                ],
            },
        },
        "proximal_analysis": {
            "proximal_finite_values": len(prox_errors),
            "proximal_pass": prox_pass,
            "proximal_max_error_deg": _max_or_none(prox_errors),
            "nonproximal_finite_values": len(nonprox_errors),
            "nonproximal_pass": nonprox_pass,
            "nonproximal_max_error_deg": _max_or_none(nonprox_errors),
            "neutral_production_proximal_deg": generated["neutral"]["adapted"].result.flexion_deg[0, :, :, 0].tolist(),
        },
        "mirror_validation": {
            "cases": mirror_entries,
            "max_local_difference": _max_or_none(mirror_values),
            "pass": bool(mirror_values and max(mirror_values) <= 1e-8),
        },
        "rigid_transform_validation": {
            "translation": _local_rigid_invariance(
                generated,
                "neutral",
                [case.case_id for case in valid_cases if case.category == "translation"],
            ),
            "scale": _local_rigid_invariance(
                generated,
                "neutral",
                [case.case_id for case in valid_cases if case.category == "scale"],
            ),
            "global_rotation": _local_rigid_invariance(
                generated,
                "orientation_identity",
                [case.case_id for case in valid_cases if case.category == "quaternion_orientation" and case.case_id != "orientation_identity"],
            ),
        },
        "orientation_mapping": orientation_mapping,
        "invalid_cases": invalid_records,
        "conditioning_cases": conditioning_case_ids,
        "valid_cases_with_all_public_channels_finite": finite_valid_cases,
        "locked_tolerances": dict(CONTRACT_TOLERANCES),
    }


def _pilot_diagnostics(qa: dict[str, Any]) -> dict[str, Any]:
    tracked_run = Path(qa["_tracked_run"])
    kinematics_run = Path(qa["_kinematics_run"])
    strict = 0
    palm = 0
    hand_instances = 0
    flex_nan = 0
    flex_total = 0
    spread_nan = 0
    spread_total = 0
    state_invalid = 0
    conditioning = 0
    no_pose_no_invention_pass = True
    flag_counts: dict[str, int] = {}
    for sample_dir in sorted(kinematics_run.iterdir(), key=lambda path: path.name):
        if not sample_dir.is_dir() or not (sample_dir / "hand_kinematics.npz").is_file():
            continue
        with np.load(sample_dir / "hand_kinematics.npz", allow_pickle=False) as data:
            valid = np.asarray(data["valid_kinematics"], dtype=bool)
            palm_valid = np.asarray(data["valid_palm_frame"], dtype=bool)
            states = np.asarray(data["tracking_state_code"], dtype=np.int32)
            flex = np.asarray(data["flexion_deg"])
            spread = np.asarray(data["adjacent_spread_deg"])
            rotation = np.asarray(data["palm_rotation_matrix"])
            quaternion = np.asarray(data["palm_quaternion_wxyz"])
            flags_json = np.asarray(data["kinematic_flags_json"])
        hand_instances += int(valid.size)
        strict += int(valid.sum())
        palm += int(palm_valid.sum())
        state_invalid += int((~palm_valid).sum())
        conditioning += int((palm_valid & ~valid).sum())
        for row, hand in zip(*np.where(~palm_valid)):
            no_pose_no_invention_pass &= not any(
                np.isfinite(view[row, hand]).any()
                for view in (flex, spread, rotation, quaternion)
            )
        flex_nan += int(np.isnan(flex).sum())
        flex_total += int(flex.size)
        spread_nan += int(np.isnan(spread).sum())
        spread_total += int(spread.size)
        for value in flags_json.flat:
            for flag in json.loads(str(value)):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    qa_copy = dict(qa)
    qa_copy.pop("_tracked_run", None)
    qa_copy.pop("_kinematics_run", None)
    invalid_mask = dict(qa_copy.get("invalid_mask_violations", {}))
    partial_instances = invalid_mask.pop("partial_channel_instances", [])
    invalid_mask.pop("violations", None)
    invalid_mask["partial_channel_instance_count"] = len(partial_instances)
    qa_copy["invalid_mask_violations"] = invalid_mask
    non_finite = dict(qa_copy.get("non_finite_violations", {}))
    non_finite.pop("violations", None)
    qa_copy["non_finite_violations"] = non_finite
    alignment = dict(qa_copy.get("tracking_alignment", {}))
    alignment.pop("mismatches", None)
    qa_copy["tracking_alignment"] = alignment
    suspicious = dict(qa_copy.get("suspicious_value_flags", {}))
    suspicious.pop("violations", None)
    qa_copy["suspicious_value_flags"] = suspicious
    return {
        "qa_summary": qa_copy,
        "hand_instances": hand_instances,
        "strict_valid_hand_instances": strict,
        "valid_palm_frames": palm,
        "invalid_from_tracking_state": state_invalid,
        "partial_spread_conditioned_instances": conditioning,
        "flexion_nan": flex_nan,
        "flexion_total": flex_total,
        "flexion_nan_expected_from_tracking": state_invalid * 15,
        "flexion_nan_accounting_pass": flex_nan == state_invalid * 15,
        "spread_nan": spread_nan,
        "spread_total": spread_total,
        "spread_nan_from_tracking": state_invalid * 4,
        "spread_nan_additional_conditioning": spread_nan - state_invalid * 4,
        "spread_nan_accounting_pass": spread_nan - state_invalid * 4 == 474,
        "no_pose_no_invention_pass": no_pose_no_invention_pass,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-run", type=Path, required=True)
    parser.add_argument("--kinematics-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--qa-json", type=Path, default=None)
    args = parser.parse_args()

    pilot_contract = validate_frozen_pilot_inputs(args.tracked_run, args.kinematics_run)
    qa_summary, _ = validate_runs(args.tracked_run, args.kinematics_run)
    qa_summary["_tracked_run"] = str(args.tracked_run)
    qa_summary["_kinematics_run"] = str(args.kinematics_run)
    synthetic = run_synthetic_validation()
    pilot_diagnostics = _pilot_diagnostics(qa_summary)
    rotation_qa = qa_summary["rotation_errors"]
    quaternion_qa = qa_summary["quaternion_errors"]
    rotation_quality_pass = bool(
        rotation_qa["orthogonality"]["max"] <= CONTRACT_TOLERANCES["rotation_matrix_orthogonality"]
        and rotation_qa["determinant_abs_error"]["max"] <= CONTRACT_TOLERANCES["rotation_matrix_determinant"]
        and rotation_qa["determinant_non_positive"]["count"] == 0
        and quaternion_qa["norm_abs_error"]["max"] <= CONTRACT_TOLERANCES["quaternion_norm"]
        and quaternion_qa["matrix_quaternion_element_abs_error"]["max"]
        <= CONTRACT_TOLERANCES["matrix_quaternion_consistency"]
    )
    acceptance = {
        "A_analytic_flexion": {
            "status": "PASS" if all(record.get("flexion", {}).get("pass", False) for record in synthetic["case_results"] if record["expected_valid"]) else "FAIL",
            "evidence": "full frozen fixture flexion comparison, including the proximal channel",
        },
        "B_analytic_spread": {
            "status": "PASS" if all(record.get("spread", {}).get("pass", False) for record in synthetic["case_results"] if record["expected_valid"]) else "FAIL",
            "evidence": "full frozen fixture spread comparison; exact conditioning exceptions remain reported",
        },
        "C_mirror_consistency": {
            "status": "PASS" if synthetic["mirror_validation"]["pass"] else "FAIL",
            "evidence": "shared production local flexion/spread on four mirror-equivalent cases",
        },
        "D_rigid_invariance": {
            "status": "PASS" if all(section["pass"] for section in synthetic["rigid_transform_validation"].values()) else "FAIL",
            "evidence": "translation, scale and global-rotation local deltas",
        },
        "E_palm_orientation_convention": {
            "status": "PASS" if synthetic["orientation_mapping"]["single_fixed_mapping_exists"] and synthetic["orientation_mapping"]["per_side_mapping_pass"] else "FAIL",
            "evidence": "one common fixed mapping is required; only distinct LEFT/RIGHT mappings are exact",
        },
        "F_rotation_quality": {
            "status": "PASS" if rotation_quality_pass else "FAIL",
            "evidence": "full pilot valid-palm rotation/quaternion QA against locked thresholds",
        },
        "G_tracking_alignment": {
            "status": "PASS" if pilot_contract["passed"] and qa_summary["tracking_alignment"]["passed"] else "FAIL",
            "evidence": "exact manifest IDs, 18 samples, 894 frames, state/source alignment",
        },
        "H_missing_occlusion_handling": {
            "status": "PASS" if pilot_diagnostics["no_pose_no_invention_pass"] else "FAIL",
            "evidence": "all 18 no-pose hand instances remain all-NaN and invalid",
        },
        "I_invalid_geometry": {
            "status": "PASS" if all(record.get("invalid_case_pass", False) for record in synthetic["invalid_cases"]) else "FAIL",
            "evidence": "all six intentionally invalid fixture outcomes",
        },
        "J_validity_nan_contract": {
            "status": "PASS" if qa_summary["passed"] and pilot_diagnostics["flexion_nan_accounting_pass"] and pilot_diagnostics["spread_nan_accounting_pass"] else "FAIL",
            "evidence": "channel-level valid_palm_frame contract with exact pilot NaN accounting",
        },
    }
    final_verdict = "TASK-005 VALIDATED — READY FOR TASK-006" if all(item["status"] == "PASS" for item in acceptance.values()) else "TASK-005 NEEDS REVISION"
    payload = {
        "task": "TASK-005D",
        "frozen_inputs": {
            "main_base": "ba6389f334ea5277b303c3f7795c919def4bf08e",
            "task005a_implementation": "564167420c7f5b4f12197fe36e7d2b59ae08ace0",
            "task005a_report": "60480a6",
            "task005b": "5f981d9f8c44408488f02b74f73a378197422830",
            "task005c_pr": 11,
            "task005c_commit": "59d49f49321834160943cf94d216a1d26e4979e8",
        },
        "pilot_input_contract": pilot_contract,
        "synthetic_validation": synthetic,
        "pilot_diagnostics": pilot_diagnostics,
        "copilot_qa": {
            "tool": "evaluation/kinematics_qa",
            "validator_result": "PASS" if qa_summary["passed"] else "FAIL",
            "qa_json": str(args.qa_json) if args.qa_json else None,
        },
        "acceptance_criteria": acceptance,
        "required_follow_up": [
            "INTEGRATION/CONTRACT FIX: settle the proximal-channel definition and one-common-vs-side-specific palm basis mapping before TASK-006.",
            "OPUS CORE FIX: reject or explicitly flag coincident MCP geometry instead of returning a fully valid hand.",
        ],
        "spread_conditioning_recommendation": "KEEP for the frozen run; retain per-channel masks and gather more evidence before changing 15 degrees.",
        "final_verdict": final_verdict,
    }
    output = args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"synthetic valid/invalid: {synthetic['catalog']['valid_cases']}/{synthetic['catalog']['invalid_cases']}")
    print(f"synthetic full-case pass: {sum(record.get('full_case_pass', False) for record in synthetic['case_results'] if record['expected_valid'])}/80")
    print(f"pilot QA: {payload['copilot_qa']['validator_result']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
