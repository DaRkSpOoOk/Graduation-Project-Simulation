#!/usr/bin/env python3
"""Self-check and integration diagnostic for the final TASK-005 contract.

The self-check uses only the independent fixture generator and the
geometry-derived TASK-005-final-v2 truth.  The optional production diagnostic
is run by default against the frozen TASK-005A entrypoint, but it never edits
production output or changes its validity flags.
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

from evaluation.kinematics.benchmark_contract import quaternion_matrix_wxyz  # noqa: E402
from evaluation.kinematics.final_contract import (  # noqa: E402
    FINAL_CONTRACT_TOLERANCES,
    FINAL_CONTRACT_VERSION,
    FINAL_FINGER_ORDER,
    FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG,
    FINAL_SPREAD_ORDER,
    FINAL_TRACK_ORDER,
    SIDE_ORIENTATION_MAPPINGS,
    build_final_catalog,
    build_final_sequence,
    map_production_sequence_rotations,
    mapped_quaternion_wxyz,
)
from evaluation.kinematics.metrics import (  # noqa: E402
    quaternion_angle_error_deg,
    rotation_angle_error_deg,
)
from evaluation.kinematics.production_adapter import (  # noqa: E402
    ProductionSyntheticResult,
    extract_production_sequence,
)


def _max_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.max(array))


def _error_summary(predicted: object, expected: object, tolerance: float) -> dict[str, Any]:
    actual = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(expected, dtype=np.float64)
    if actual.shape != truth.shape:
        return {
            "pass": False,
            "shape_match": False,
            "predicted_shape": list(actual.shape),
            "expected_shape": list(truth.shape),
        }
    finite_truth = np.isfinite(truth)
    finite_actual = np.isfinite(actual)
    comparable = finite_truth & finite_actual
    errors = np.abs(actual - truth)
    finite_errors = errors[comparable]
    return {
        "pass": bool(finite_actual[finite_truth].all() and finite_errors.size == int(finite_truth.sum()) and np.all(finite_errors <= tolerance)),
        "shape_match": True,
        "expected_finite_values": int(finite_truth.sum()),
        "predicted_finite_values_at_expected_positions": int((finite_truth & finite_actual).sum()),
        "max_error": _max_or_none(finite_errors),
        "tolerance": float(tolerance),
    }


def _spread_summary(predicted: object, expected: object, tolerance: float) -> dict[str, Any]:
    actual = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(expected, dtype=np.float64)
    if actual.shape != truth.shape:
        return {
            "pass": False,
            "shape_match": False,
            "predicted_shape": list(actual.shape),
            "expected_shape": list(truth.shape),
        }
    expected_finite = np.isfinite(truth)
    predicted_finite = np.isfinite(actual)
    mask_match = bool(np.array_equal(expected_finite, predicted_finite))
    comparable = expected_finite & predicted_finite
    errors = np.abs(actual - truth)
    finite_errors = errors[comparable]
    return {
        "pass": bool(mask_match and np.all(finite_errors <= tolerance)),
        "shape_match": True,
        "finite_mask_match": mask_match,
        "expected_finite_values": int(expected_finite.sum()),
        "predicted_finite_values": int(predicted_finite.sum()),
        "expected_nan_values": int((~expected_finite).sum()),
        "predicted_nan_values": int((~predicted_finite).sum()),
        "max_error": _max_or_none(finite_errors),
        "tolerance": float(tolerance),
    }


def _orientation_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            rotation_angle_error_deg(predicted[frame, track], expected[frame, track])
            for frame in range(predicted.shape[0])
            for track in range(2)
        ],
        dtype=np.float64,
    )


def _quaternion_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            quaternion_angle_error_deg(predicted[frame, track], expected[frame, track])
            for frame in range(predicted.shape[0])
            for track in range(2)
        ],
        dtype=np.float64,
    )


def _rotation_quality(production: ProductionSyntheticResult) -> dict[str, Any]:
    matrices = np.asarray(production.result.palm_rotation_matrix, dtype=np.float64)
    quaternions = np.asarray(production.result.palm_quaternion_wxyz, dtype=np.float64)
    orthogonality: list[float] = []
    determinant: list[float] = []
    quaternion_norm: list[float] = []
    matrix_quaternion: list[float] = []
    non_finite = 0
    non_positive_det = 0
    for frame in range(matrices.shape[0]):
        for track in range(matrices.shape[1]):
            matrix = matrices[frame, track]
            quaternion = quaternions[frame, track]
            if not np.isfinite(matrix).all() or not np.isfinite(quaternion).all():
                non_finite += 1
                continue
            orthogonality.append(float(np.max(np.abs(matrix.T @ matrix - np.eye(3)))))
            det_error = abs(float(np.linalg.det(matrix)) - 1.0)
            determinant.append(det_error)
            if float(np.linalg.det(matrix)) <= 0.0:
                non_positive_det += 1
            quaternion_norm.append(abs(float(np.linalg.norm(quaternion)) - 1.0))
            converted = quaternion_matrix_wxyz(quaternion)
            matrix_quaternion.append(float(np.max(np.abs(matrix - converted))))

    limits = FINAL_CONTRACT_TOLERANCES
    return {
        "checked_hand_instances": int(matrices.shape[0] * matrices.shape[1] - non_finite),
        "non_finite_instances": non_finite,
        "non_positive_determinants": non_positive_det,
        "maximum_orthogonality_error": _max_or_none(orthogonality),
        "maximum_determinant_abs_error": _max_or_none(determinant),
        "maximum_quaternion_norm_error": _max_or_none(quaternion_norm),
        "maximum_matrix_quaternion_error": _max_or_none(matrix_quaternion),
        "pass": bool(
            non_finite == 0
            and non_positive_det == 0
            and max(orthogonality, default=0.0) <= limits["rotation_matrix_orthogonality"]
            and max(determinant, default=0.0) <= limits["rotation_matrix_determinant"]
            and max(quaternion_norm, default=0.0) <= limits["quaternion_norm"]
            and max(matrix_quaternion, default=0.0) <= limits["matrix_quaternion_consistency"]
        ),
    }


def _self_check(catalog: tuple[Any, ...]) -> dict[str, Any]:
    valid = [case for case in catalog if case.expected_valid]
    invalid = [case for case in catalog if not case.expected_valid]
    checks: dict[str, bool] = {
        "catalog_has_86_cases": len(catalog) == 86,
        "catalog_has_80_valid_cases": len(valid) == 80,
        "catalog_has_6_invalid_cases": len(invalid) == 6,
    }
    final_sequences = {case.case_id: build_final_sequence(case) for case in catalog}
    checks["all_valid_flexion_truth_is_finite"] = all(
        np.isfinite(final_sequences[case.case_id].flexion_deg).all() for case in valid
    )
    checks["requested_middle_distal_turns_match_geometry"] = all(
        np.allclose(
            final_sequences[case.case_id].flexion_deg[0, :, :, 1:],
            np.asarray(case.frames[0].normalized().flexion_deg, dtype=np.float64)[None, :, 1:],
            atol=1e-8,
            rtol=0.0,
        )
        for case in valid
    )
    checks["valid_palm_plane_is_geometry_derived"] = all(
        bool(final_sequences[case.case_id].valid_palm_frame.all()) for case in valid
    )
    def _channel_level_truth_is_consistent(case: Any) -> bool:
        sequence = final_sequences[case.case_id]
        for frame in range(sequence.joints.shape[0]):
            for track in range(2):
                spread_nan = np.isnan(sequence.adjacent_spread_deg[frame, track])
                if spread_nan.any():
                    if sequence.valid_kinematics[frame, track]:
                        return False
                    if not sequence.valid_palm_frame[frame, track]:
                        return False
                    if not np.isfinite(sequence.flexion_deg[frame, track]).all():
                        return False
                    if not np.isfinite(sequence.palm_rotation_matrix[frame, track]).all():
                        return False
                    if not np.isfinite(sequence.palm_quaternion_wxyz[frame, track]).all():
                        return False
                    if not np.isfinite(sequence.adjacent_spread_deg[frame, track]).any():
                        return False
                elif not sequence.valid_kinematics[frame, track]:
                    return False
        return True

    checks["conditioning_truth_is_channel_level"] = all(
        _channel_level_truth_is_consistent(case) for case in valid
    )
    neutral = final_sequences["neutral"]
    expected_neutral_proximal = np.asarray(
        [15.217592968193, 9.440034828176, 0.0, 9.440034828176, 15.217592968193],
        dtype=np.float64,
    )
    checks["neutral_proximal_truth_uses_wrist_to_base_geometry"] = bool(
        np.allclose(neutral.flexion_deg[0, 0, :, 0], expected_neutral_proximal, atol=1e-9, rtol=0.0)
    )
    near_180 = final_sequences["adversarial_near_180"]
    checks["near_180_spread_uses_actual_proximal_direction"] = bool(
        np.allclose(
            near_180.adjacent_spread_deg,
            np.asarray([[[170.0, 170.0, 10.0, 10.0], [170.0, 170.0, 10.0, 10.0]]]),
            atol=1e-9,
            rtol=0.0,
        )
    )
    conditioning_ids = {
        "single_thumb_joint0_90deg",
        "single_index_joint0_90deg",
        "single_middle_joint0_90deg",
        "single_ring_joint0_90deg",
        "single_pinky_joint0_90deg",
        "multi_curl_pinky",
    }
    actual_conditioning_ids = {
        case.case_id
        for case in valid
        if np.isnan(final_sequences[case.case_id].adjacent_spread_deg).any()
    }
    checks["six_conditioning_cases_are_explicit"] = actual_conditioning_ids == conditioning_ids
    checks["spread_nan_mask_matches_degenerate_directions"] = all(
        np.array_equal(
            np.isnan(final_sequences[case.case_id].adjacent_spread_deg[0, track]),
            np.asarray(
                [
                    final_sequences[case.case_id].spread_direction_degenerate[0, track, first]
                    or final_sequences[case.case_id].spread_direction_degenerate[0, track, second]
                    for first, second in ((0, 1), (1, 2), (2, 3), (3, 4))
                ],
                dtype=bool,
            ),
        )
        for case in valid
        for track in range(2)
    )
    checks["invalid_catalog_expectations_preserved"] = all(
        not final_sequences[case.case_id].expected_valid for case in invalid
    )
    checks["orientation_mappings_are_fixed_proper_rotations"] = all(
        matrix.shape == (3, 3)
        and np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0.0)
        and abs(float(np.linalg.det(matrix)) - 1.0) <= 1e-12
        for matrix in SIDE_ORIENTATION_MAPPINGS.values()
    )
    checks["left_right_mappings_are_documented_separately"] = not np.array_equal(
        SIDE_ORIENTATION_MAPPINGS["LEFT"], SIDE_ORIENTATION_MAPPINGS["RIGHT"]
    )

    mirror_cases = [case for case in valid if case.mirror_equivalent]
    checks["mirror_local_truth_is_equivalent"] = all(
        np.allclose(
            final_sequences[case.case_id].flexion_deg[:, 0],
            final_sequences[case.case_id].flexion_deg[:, 1],
            atol=1e-8,
            rtol=0.0,
            equal_nan=True,
        )
        and np.allclose(
            final_sequences[case.case_id].adjacent_spread_deg[:, 0],
            final_sequences[case.case_id].adjacent_spread_deg[:, 1],
            atol=1e-8,
            rtol=0.0,
            equal_nan=True,
        )
        for case in mirror_cases
    )

    def _same_local_truth(first_id: str, other_ids: set[str]) -> bool:
        first = final_sequences[first_id]
        return all(
            np.allclose(
                first.flexion_deg,
                final_sequences[case_id].flexion_deg,
                atol=1e-8,
                rtol=0.0,
                equal_nan=True,
            )
            and np.allclose(
                first.adjacent_spread_deg,
                final_sequences[case_id].adjacent_spread_deg,
                atol=1e-8,
                rtol=0.0,
                equal_nan=True,
            )
            for case_id in other_ids
        )

    checks["rigid_transform_local_truth_is_invariant"] = (
        _same_local_truth("neutral", {"translation_1", "translation_2", "translation_3"})
        and _same_local_truth("neutral", {"scale_0_5x", "scale_1_0x", "scale_2_0x", "scale_5_0x"})
        and _same_local_truth(
            "orientation_identity",
            {
                "orientation_identity",
                "orientation_x90",
                "orientation_y90",
                "orientation_z90",
                "orientation_x180",
                "orientation_y180",
                "orientation_z180",
                "orientation_composed",
            },
        )
    )

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "catalog_count": len(catalog),
        "valid_case_count": len(valid),
        "invalid_case_count": len(invalid),
        "conditioning_case_ids": sorted(actual_conditioning_ids),
        "conditioning_min_projected_angle_deg": FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG,
    }


def _production_case_result(case: Any) -> dict[str, Any]:
    expected = build_final_sequence(case)
    production = extract_production_sequence(case.generate())
    result = production.result
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "expected_valid": bool(expected.expected_valid),
        "production_valid_kinematics": production.valid_kinematics.tolist(),
        "production_valid_palm_frame": production.valid_palm_frame.tolist(),
    }
    if not expected.expected_valid:
        flagged = any(
            production.flags[frame][track]
            for frame in range(len(production.flags))
            for track in range(2)
        )
        record.update(
            {
                "invalid_reasons": list(expected.invalid_reasons),
                "production_flag_names": sorted(
                    {
                        flag
                        for row in production.flags
                        for flags in row
                        for flag in flags
                    }
                ),
                "rejected_or_flagged": bool(not production.valid_kinematics.any() or flagged),
                "invalid_case_pass": bool(not production.valid_kinematics.any() or flagged),
            }
        )
        return record

    flexion = _error_summary(
        result.flexion_deg,
        expected.flexion_deg,
        FINAL_CONTRACT_TOLERANCES["known_flexion_abs_error_deg"],
    )
    spread = _spread_summary(
        result.adjacent_spread_deg,
        expected.adjacent_spread_deg,
        FINAL_CONTRACT_TOLERANCES["known_spread_abs_error_deg"],
    )
    mapped_rotation = map_production_sequence_rotations(result.palm_rotation_matrix)
    orientation_errors = _orientation_errors(mapped_rotation, expected.palm_rotation_matrix)
    mapped_quaternions = np.asarray(
        [
            [mapped_quaternion_wxyz(mapped_rotation[frame, track]) for track in range(2)]
            for frame in range(mapped_rotation.shape[0])
        ],
        dtype=np.float64,
    )
    quaternion_errors = _quaternion_errors(mapped_quaternions, expected.palm_quaternion_wxyz)
    orientation_pass = bool(
        np.isfinite(mapped_rotation).all()
        and np.all(orientation_errors <= FINAL_CONTRACT_TOLERANCES["known_orientation_error_deg"])
    )
    quaternion_pass = bool(
        np.isfinite(mapped_quaternions).all()
        and np.all(quaternion_errors <= FINAL_CONTRACT_TOLERANCES["known_orientation_error_deg"])
    )
    palm_match = bool(np.array_equal(production.valid_palm_frame, expected.valid_palm_frame))
    strict_match = bool(np.array_equal(production.valid_kinematics, expected.valid_kinematics))
    quality = _rotation_quality(production)
    record.update(
        {
            "flexion": flexion,
            "spread": spread,
            "orientation": {
                "pass": orientation_pass,
                "max_error_deg": _max_or_none(orientation_errors),
            },
            "quaternion": {
                "pass": quaternion_pass,
                "max_error_deg": _max_or_none(quaternion_errors),
            },
            "validity": {
                "expected_valid_palm_frame": expected.valid_palm_frame.tolist(),
                "palm_frame_match": palm_match,
                "expected_valid_kinematics": expected.valid_kinematics.tolist(),
                "strict_validity_match": strict_match,
            },
            "rotation_quality": quality,
        }
    )
    record["full_case_pass"] = bool(
        flexion["pass"]
        and spread["pass"]
        and orientation_pass
        and quaternion_pass
        and palm_match
        and strict_match
        and quality["pass"]
    )
    return record


def _aggregate_production(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in case_results if record["expected_valid"]]
    invalid = [record for record in case_results if not record["expected_valid"]]

    def count_pass(metric: str) -> int:
        return sum(bool(record.get(metric, {}).get("pass")) for record in valid)

    def worst(metric: str, key: str) -> float | None:
        values = [record[metric].get(key) for record in valid if record.get(metric, {}).get(key) is not None]
        return max((float(value) for value in values), default=None)

    return {
        "valid_case_count": len(valid),
        "invalid_case_count": len(invalid),
        "final_valid_case_pass_count": sum(bool(record.get("full_case_pass")) for record in valid),
        "flexion": {
            "case_pass_count": count_pass("flexion"),
            "value_count": int(sum(record.get("flexion", {}).get("expected_finite_values", 0) for record in valid)),
            "worst_error_deg": worst("flexion", "max_error"),
        },
        "spread": {
            "case_pass_count": count_pass("spread"),
            "expected_nan_values": int(sum(record.get("spread", {}).get("expected_nan_values", 0) for record in valid)),
            "conditioning_case_count": sum(
                bool(record.get("spread", {}).get("expected_nan_values", 0)) for record in valid
            ),
            "worst_finite_error_deg": worst("spread", "max_error"),
        },
        "orientation": {
            "case_pass_count": count_pass("orientation"),
            "worst_error_deg": worst("orientation", "max_error_deg"),
        },
        "quaternion": {
            "case_pass_count": count_pass("quaternion"),
            "worst_error_deg": worst("quaternion", "max_error_deg"),
        },
        "invalid_geometry": {
            "case_pass_count": sum(bool(record.get("invalid_case_pass")) for record in invalid),
            "case_count": len(invalid),
            "failed_case_ids": [record["case_id"] for record in invalid if not record.get("invalid_case_pass")],
        },
        "invalid_case_results": [
            {
                "case_id": record["case_id"],
                "expected_reasons": record.get("invalid_reasons", []),
                "production_valid_kinematics": record.get("production_valid_kinematics"),
                "production_valid_palm_frame": record.get("production_valid_palm_frame"),
                "invalid_case_pass": record.get("invalid_case_pass"),
            }
            for record in invalid
        ],
    }


def run(output_path: Path) -> dict[str, Any]:
    catalog = build_final_catalog()
    self_check = _self_check(catalog)
    case_results = [_production_case_result(case) for case in catalog]
    production = _aggregate_production(case_results)
    data: dict[str, Any] = {
        "task": "TASK-005E1",
        "contract_version": FINAL_CONTRACT_VERSION,
        "frozen_inputs": {
            "task005d_commit": "b41bc1808d09b1987ebdcf417e1bdadc42962f6d",
            "task005a_implementation": "564167420c7f5b4f12197fe36e7d2b59ae08ace0",
            "task005a_report": "60480a6",
            "task005b_commit": "5f981d9f8c44408488f02b74f73a378197422830",
            "task005d_pr": 14,
        },
        "contract": {
            "track_order": list(FINAL_TRACK_ORDER),
            "finger_order": list(FINAL_FINGER_ORDER),
            "spread_order": list(FINAL_SPREAD_ORDER),
            "flexion": "unsigned geometric bend magnitude in degrees, range 0..180",
            "proximal": "angle(wrist -> finger base, finger base -> next joint); geometric proxy, not isolated clinical MCP flexion",
            "spread": "unsigned angle between actual proximal-phalanx directions projected into the output-geometry palm plane",
            "spread_conditioning_angle_deg": FINAL_SPREAD_MIN_PROJECTED_ANGLE_DEG,
            "validity": "Model B channel-level validity",
            "orientation": "fixture convention scored using one fixed proper mapping per hand side; raw production frame is not overwritten",
            "quaternion_order": "wxyz",
        },
        "tolerances": {key: float(value) for key, value in FINAL_CONTRACT_TOLERANCES.items()},
        "orientation_mappings": {
            side: matrix.tolist() for side, matrix in SIDE_ORIENTATION_MAPPINGS.items()
        },
        "self_check": self_check,
        "production_compatibility": production,
        "case_results": case_results,
        "qa_contract": {
            "version": "TASK-005-final-v2-model-B",
            "status": "SELECTED",
            "rule": "valid_palm_frame gates orientation; valid flexion/spread channels are evaluated independently and undefined channels are NaN",
        },
        "remaining_production_defect": "OPUS CORE FIX — coincident-MCP invalid geometry",
        "final_readiness": "WAITING_FOR_TASK005E2",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/kinematics/TASK-005E1-final-contract-results.json",
    )
    args = parser.parse_args()
    data = run(args.output_json)
    print(
        json.dumps(
            {
                "contract_version": data["contract_version"],
                "self_check": data["self_check"],
                "production_compatibility": data["production_compatibility"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if data["self_check"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
