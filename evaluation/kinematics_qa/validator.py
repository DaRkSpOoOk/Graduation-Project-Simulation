"""TASK-005 kinematics QA validation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    KINEMATICS_NPZ_NAME,
    TRACKED_NPZ_NAME,
    ContractError,
    SampleKinematics,
    VALIDITY_CONTRACT_NAME,
    VALIDITY_CONTRACT_VERSION,
    list_sample_ids,
    load_kinematics_sample,
    validate_sample_contract,
)
from .rotation_checks import (
    percentile_summary,
    quaternion_to_matrix_wxyz,
    rotation_delta_degrees,
    rotation_orthogonality_error,
)
from .statistics import FINGER_NAMES, HAND_NAMES, JOINT_NAMES, SPREAD_NAMES, distribution


def _tracked_arrays(tracked_run: Path, sample_id: str) -> dict[str, np.ndarray]:
    npz = tracked_run / sample_id / TRACKED_NPZ_NAME
    with np.load(npz, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _iter_valid_positions(sample: SampleKinematics):
    valid = np.asarray(sample.arrays["valid_kinematics"], dtype=bool)
    frames = np.asarray(sample.arrays["frame_index"], dtype=np.int64)
    for row in range(valid.shape[0]):
        for hand in range(valid.shape[1]):
            if valid[row, hand]:
                yield row, hand, int(frames[row])


def _iter_valid_palm_positions(sample: SampleKinematics):
    """Yield rows with a valid canonical palm frame.

    TASK-005A deliberately separates strict all-channel validity from the
    orientation/palm-frame validity flag.  Rotation QA therefore includes
    the 215 observed hand instances whose spread channels are undefined but
    whose palm frame and quaternion remain finite.
    """

    valid = np.asarray(sample.arrays["valid_palm_frame"], dtype=bool)
    frames = np.asarray(sample.arrays["frame_index"], dtype=np.int64)
    for row in range(valid.shape[0]):
        for hand in range(valid.shape[1]):
            if valid[row, hand]:
                yield row, hand, int(frames[row])


def _hand_name(index: int) -> str:
    return HAND_NAMES[index]


def _state_nan_checks(samples: list[SampleKinematics]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the TASK-005D channel-level validity contract.

    ``valid_kinematics`` remains a strict convenience flag: it is true only
    when every float channel is finite.  ``valid_palm_frame`` independently
    certifies the rotation/quaternion channels.  A strict-false/palm-true
    instance may therefore retain finite flexion and orientation while only
    the geometrically undefined spread channels are NaN.  When the palm frame
    is invalid, all derived float fields must be NaN.
    """

    invalid_mask_violations: list[dict[str, Any]] = []
    non_finite_violations: list[dict[str, Any]] = []
    partial_channel_instances: list[dict[str, Any]] = []

    fields = (
        "flexion_deg",
        "adjacent_spread_deg",
        "palm_rotation_matrix",
        "palm_quaternion_wxyz",
    )

    for sample in samples:
        valid = np.asarray(sample.arrays["valid_kinematics"], dtype=bool)
        palm_valid = np.asarray(sample.arrays["valid_palm_frame"], dtype=bool)
        frame_index = np.asarray(sample.arrays["frame_index"], dtype=np.int64)
        for row in range(valid.shape[0]):
            for hand in range(valid.shape[1]):
                reference = {
                    "sample_id": sample.sample_id,
                    "frame_index": int(frame_index[row]),
                    "track": _hand_name(hand),
                }
                finite_by_field: dict[str, bool] = {}
                for field in fields:
                    view = np.asarray(sample.arrays[field])[row, hand]
                    finite_by_field[field] = bool(np.isfinite(view).all())
                    if valid[row, hand] and not finite_by_field[field]:
                        non_finite_violations.append({**reference, "field": field})

                if valid[row, hand] and not palm_valid[row, hand]:
                    invalid_mask_violations.append(
                        {**reference, "field": "validity_flags", "reason": "strict_valid_without_palm_frame"}
                    )
                if palm_valid[row, hand]:
                    # Model B is intentionally channel-level, but its
                    # permitted partial state is narrow: a valid palm frame
                    # must still have finite flexion and finite orientation;
                    # only geometrically undefined spread channels may be
                    # NaN. This prevents an arbitrary partial pose from
                    # being mistaken for the selected contract.
                    if not finite_by_field["flexion_deg"]:
                        invalid_mask_violations.append(
                            {**reference, "field": "flexion_deg", "reason": "palm_valid_requires_finite_flexion"}
                        )
                    for field in ("palm_rotation_matrix", "palm_quaternion_wxyz"):
                        if not finite_by_field[field]:
                            non_finite_violations.append({**reference, "field": field})
                    if not valid[row, hand]:
                        if all(finite_by_field.values()):
                            invalid_mask_violations.append(
                                {
                                    **reference,
                                    "field": "validity_flags",
                                    "reason": "strict_false_with_all_float_channels_finite",
                                }
                            )
                        elif (
                            finite_by_field["flexion_deg"]
                            and finite_by_field["palm_rotation_matrix"]
                            and finite_by_field["palm_quaternion_wxyz"]
                        ):
                            partial_channel_instances.append(
                                {
                                    **reference,
                                    "finite_fields": [field for field, finite in finite_by_field.items() if finite],
                                    "non_finite_fields": [field for field, finite in finite_by_field.items() if not finite],
                                }
                            )
                else:
                    # A missing/invalid palm frame has no trustworthy derived
                    # channel. This preserves the old strict rule for actual
                    # no-pose and frame-degenerate rows.
                    for field in fields:
                        view = np.asarray(sample.arrays[field])[row, hand]
                        if np.isfinite(view).any():
                            invalid_mask_violations.append({**reference, "field": field})

    invalid_mask_violations.sort(key=lambda x: (x["sample_id"], x["frame_index"], x["track"], x["field"]))
    non_finite_violations.sort(key=lambda x: (x["sample_id"], x["frame_index"], x["track"], x["field"]))

    return (
        {
            "count": len(invalid_mask_violations),
            "violations": invalid_mask_violations,
            "partial_channel_instances": partial_channel_instances,
        },
        {
            "count": len(non_finite_violations),
            "violations": non_finite_violations,
        },
    )


def _rotation_checks(samples: list[SampleKinematics]) -> dict[str, Any]:
    orth_errors: list[float] = []
    det_errors: list[float] = []
    non_finite = 0
    non_positive_det: list[dict[str, Any]] = []
    worst_orth: dict[str, Any] | None = None
    worst_det: dict[str, Any] | None = None

    for sample in samples:
        matrices = np.asarray(sample.arrays["palm_rotation_matrix"], dtype=np.float64)
        for row, hand, frame in _iter_valid_palm_positions(sample):
            matrix = matrices[row, hand]
            if not np.isfinite(matrix).all():
                non_finite += 1
                continue
            orth = rotation_orthogonality_error(matrix)
            det = float(np.linalg.det(matrix))
            det_error = abs(det - 1.0)
            orth_errors.append(orth)
            det_errors.append(det_error)
            ref = {
                "sample_id": sample.sample_id,
                "frame_index": frame,
                "track": _hand_name(hand),
            }
            if worst_orth is None or orth > worst_orth["value"]:
                worst_orth = {**ref, "value": orth}
            if worst_det is None or det_error > worst_det["value"]:
                worst_det = {**ref, "value": det_error}
            if det <= 0.0:
                non_positive_det.append({**ref, "determinant": det})

    return {
        "orthogonality": percentile_summary(orth_errors),
        "determinant_abs_error": percentile_summary(det_errors),
        "non_finite_matrices": non_finite,
        "determinant_non_positive": {
            "count": len(non_positive_det),
            "violations": sorted(non_positive_det, key=lambda x: (x["sample_id"], x["frame_index"], x["track"])),
        },
        "worst_orthogonality": worst_orth,
        "worst_determinant_error": worst_det,
    }


def _quaternion_checks(samples: list[SampleKinematics]) -> dict[str, Any]:
    norm_errors: list[float] = []
    angle_errors: list[float] = []
    element_errors: list[float] = []
    non_finite = 0
    worst_angle: dict[str, Any] | None = None

    for sample in samples:
        quats = np.asarray(sample.arrays["palm_quaternion_wxyz"], dtype=np.float64)
        mats = np.asarray(sample.arrays["palm_rotation_matrix"], dtype=np.float64)
        for row, hand, frame in _iter_valid_palm_positions(sample):
            quat = quats[row, hand]
            matrix = mats[row, hand]
            if not np.isfinite(quat).all():
                non_finite += 1
                continue
            norm = float(np.linalg.norm(quat))
            norm_error = abs(norm - 1.0)
            norm_errors.append(norm_error)
            if norm == 0.0 or not np.isfinite(matrix).all():
                continue
            q_matrix = quaternion_to_matrix_wxyz(quat / norm)
            element_errors.append(float(np.max(np.abs(matrix - q_matrix))))
            angle = rotation_delta_degrees(q_matrix, matrix)
            angle_errors.append(angle)
            if worst_angle is None or angle > worst_angle["value_deg"]:
                worst_angle = {
                    "sample_id": sample.sample_id,
                    "frame_index": frame,
                    "track": _hand_name(hand),
                    "value_deg": angle,
                }

    return {
        "norm_abs_error": percentile_summary(norm_errors),
        "non_finite_quaternions": non_finite,
        "matrix_quaternion_element_abs_error": percentile_summary(element_errors),
        "matrix_quaternion_angular_disagreement_deg": percentile_summary(angle_errors),
        "worst_matrix_quaternion_disagreement": worst_angle,
    }


def _distribution_checks(samples: list[SampleKinematics]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    flex_channels: dict[tuple[int, int, int], list[float]] = {(h, f, j): [] for h in range(2) for f in range(5) for j in range(3)}
    spread_channels: dict[tuple[int, int], list[float]] = {(h, s): [] for h in range(2) for s in range(4)}
    suspicious: list[dict[str, Any]] = []

    for sample in samples:
        frames = np.asarray(sample.arrays["frame_index"], dtype=np.int64)
        flex = np.asarray(sample.arrays["flexion_deg"], dtype=np.float64)
        spread = np.asarray(sample.arrays["adjacent_spread_deg"], dtype=np.float64)
        for row in range(flex.shape[0]):
            for hand in range(2):
                for finger in range(5):
                    for joint in range(3):
                        value = float(flex[row, hand, finger, joint])
                        if np.isfinite(value):
                            flex_channels[(hand, finger, joint)].append(value)
                            if value < -10.0 or value > 180.0:
                                suspicious.append(
                                    {
                                        "sample_id": sample.sample_id,
                                        "frame_index": int(frames[row]),
                                        "track": _hand_name(hand),
                                        "channel": f"flexion.{FINGER_NAMES[finger]}.{JOINT_NAMES[joint]}",
                                        "value_deg": value,
                                    }
                                )
                for spread_idx in range(4):
                    value = float(spread[row, hand, spread_idx])
                    if np.isfinite(value):
                        spread_channels[(hand, spread_idx)].append(value)

    flexion_stats: dict[str, Any] = {}
    spread_stats: dict[str, Any] = {}
    left_right: dict[str, Any] = {}

    for hand in range(2):
        hand_name = _hand_name(hand)
        all_flex: list[float] = []
        all_spread: list[float] = []
        for finger in range(5):
            for joint in range(3):
                channel_values = np.asarray(flex_channels[(hand, finger, joint)], dtype=np.float64)
                all_flex.extend(channel_values.tolist())
                key = f"{hand_name}.{FINGER_NAMES[finger]}.{JOINT_NAMES[joint]}"
                flexion_stats[key] = distribution(channel_values)
        for spread_idx in range(4):
            channel_values = np.asarray(spread_channels[(hand, spread_idx)], dtype=np.float64)
            all_spread.extend(channel_values.tolist())
            key = f"{hand_name}.{SPREAD_NAMES[spread_idx]}"
            spread_stats[key] = distribution(channel_values)
        left_right[hand_name] = {
            "flexion": distribution(np.asarray(all_flex, dtype=np.float64)),
            "adjacent_spread": distribution(np.asarray(all_spread, dtype=np.float64)),
        }

    suspicious.sort(key=lambda x: (x["sample_id"], x["frame_index"], x["track"], x["channel"]))
    return flexion_stats, spread_stats, left_right, suspicious


def _temporal_checks(samples: list[SampleKinematics]) -> dict[str, Any]:
    flex_deltas: dict[tuple[int, int, int], list[tuple[float, str, int]]] = {
        (h, f, j): [] for h in range(2) for f in range(5) for j in range(3)
    }
    spread_deltas: dict[tuple[int, int], list[tuple[float, str, int]]] = {(h, s): [] for h in range(2) for s in range(4)}
    orient_deltas: dict[int, list[tuple[float, str, int]]] = {0: [], 1: []}

    for sample in samples:
        frames = np.asarray(sample.arrays["frame_index"], dtype=np.int64)
        flex = np.asarray(sample.arrays["flexion_deg"], dtype=np.float64)
        spread = np.asarray(sample.arrays["adjacent_spread_deg"], dtype=np.float64)
        rot = np.asarray(sample.arrays["palm_rotation_matrix"], dtype=np.float64)

        for row in range(1, flex.shape[0]):
            for hand in range(2):
                for finger in range(5):
                    for joint in range(3):
                        prev = flex[row - 1, hand, finger, joint]
                        curr = flex[row, hand, finger, joint]
                        if np.isfinite(prev) and np.isfinite(curr):
                            flex_deltas[(hand, finger, joint)].append(
                                (abs(float(curr - prev)), sample.sample_id, int(frames[row]))
                            )
                for spread_idx in range(4):
                    prev = spread[row - 1, hand, spread_idx]
                    curr = spread[row, hand, spread_idx]
                    if np.isfinite(prev) and np.isfinite(curr):
                        spread_deltas[(hand, spread_idx)].append(
                            (abs(float(curr - prev)), sample.sample_id, int(frames[row]))
                        )
                r_prev = rot[row - 1, hand]
                r_curr = rot[row, hand]
                if np.isfinite(r_prev).all() and np.isfinite(r_curr).all():
                    orient_deltas[hand].append(
                        (
                            rotation_delta_degrees(r_prev, r_curr),
                            sample.sample_id,
                            int(frames[row]),
                        )
                    )

    def summarize(entries: list[tuple[float, str, int]], channel: str) -> dict[str, Any]:
        if not entries:
            return {"count": 0, "mean": None, "p95": None, "p99": None, "maximum": None}
        values = np.asarray([value for value, _, _ in entries], dtype=np.float64)
        max_value, max_sample, max_frame = max(entries, key=lambda item: item[0])
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "maximum": {
                "value": float(max_value),
                "sample_id": max_sample,
                "frame_index": int(max_frame),
                "channel": channel,
            },
        }

    flexion: dict[str, Any] = {}
    spread_summary: dict[str, Any] = {}
    orientation: dict[str, Any] = {}
    for hand in range(2):
        hand_name = _hand_name(hand)
        for finger in range(5):
            for joint in range(3):
                key = f"{hand_name}.{FINGER_NAMES[finger]}.{JOINT_NAMES[joint]}"
                flexion[key] = summarize(flex_deltas[(hand, finger, joint)], key)
        for spread_idx in range(4):
            key = f"{hand_name}.{SPREAD_NAMES[spread_idx]}"
            spread_summary[key] = summarize(spread_deltas[(hand, spread_idx)], key)
        orientation_key = f"{hand_name}.palm_orientation"
        orientation[orientation_key] = summarize(orient_deltas[hand], orientation_key)

    return {
        "flexion_abs_delta_deg": flexion,
        "spread_abs_delta_deg": spread_summary,
        "palm_orientation_abs_delta_deg": orientation,
    }


def _alignment_checks(
    tracked_run: Path,
    tracked_ids: list[str],
    kinematics_ids: list[str],
    samples: dict[str, SampleKinematics],
) -> dict[str, Any]:
    tracked_set = set(tracked_ids)
    kinematics_set = set(kinematics_ids)
    missing_in_kinematics = sorted(tracked_set - kinematics_set)
    extra_in_kinematics = sorted(kinematics_set - tracked_set)

    mismatches: list[dict[str, Any]] = []
    total_frames = 0

    for sample_id in sorted(tracked_set & kinematics_set):
        if sample_id not in samples:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "kinematics_unavailable_for_alignment",
                    "reason": "sample failed to load or failed contract checks",
                }
            )
            continue
        tracked = _tracked_arrays(tracked_run, sample_id)
        kine = samples[sample_id].arrays

        tracked_frame = np.asarray(tracked["frame_index"])
        kine_frame = np.asarray(kine["frame_index"])
        tracked_time = np.asarray(tracked["timestamp_seconds"], dtype=np.float64)
        kine_time = np.asarray(kine["timestamp_seconds"], dtype=np.float64)
        tracked_state = np.asarray(tracked["state_code"])
        kine_state = np.asarray(kine["tracking_state_code"])
        tracked_source = np.asarray(tracked["raw_detection_index"])
        kine_source = np.asarray(kine["source_raw_detection_index"])

        total_frames += int(kine_frame.size)

        if tracked_frame.shape[0] != kine_frame.shape[0]:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "frame_count",
                    "tracked": int(tracked_frame.shape[0]),
                    "kinematics": int(kine_frame.shape[0]),
                }
            )
            continue

        if not np.array_equal(tracked_frame, kine_frame):
            diff = np.where(tracked_frame != kine_frame)[0]
            idx = int(diff[0]) if diff.size else -1
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "frame_index",
                    "first_mismatch_position": idx,
                    "tracked": int(tracked_frame[idx]) if idx >= 0 else None,
                    "kinematics": int(kine_frame[idx]) if idx >= 0 else None,
                }
            )

        if not np.array_equal(tracked_time, kine_time):
            diff = np.where(tracked_time != kine_time)[0]
            idx = int(diff[0]) if diff.size else -1
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "timestamp_seconds",
                    "first_mismatch_position": idx,
                    "tracked": float(tracked_time[idx]) if idx >= 0 else None,
                    "kinematics": float(kine_time[idx]) if idx >= 0 else None,
                    "max_abs_diff": float(np.max(np.abs(tracked_time - kine_time))),
                }
            )

        if not np.array_equal(tracked_state, kine_state):
            pos = np.argwhere(tracked_state != kine_state)
            row, hand = [int(v) for v in pos[0]]
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "tracking_state_code",
                    "frame_index": int(kine_frame[row]),
                    "track": _hand_name(hand),
                    "tracked": int(tracked_state[row, hand]),
                    "kinematics": int(kine_state[row, hand]),
                }
            )

        if not np.array_equal(tracked_source, kine_source):
            pos = np.argwhere(tracked_source != kine_source)
            row, hand = [int(v) for v in pos[0]]
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "field": "source_raw_detection_index",
                    "frame_index": int(kine_frame[row]),
                    "track": _hand_name(hand),
                    "tracked": int(tracked_source[row, hand]),
                    "kinematics": int(kine_source[row, hand]),
                }
            )

    mismatches.sort(key=lambda x: (x["sample_id"], x["field"]))
    return {
        "passed": not (missing_in_kinematics or extra_in_kinematics or mismatches),
        "missing_in_kinematics": missing_in_kinematics,
        "extra_in_kinematics": extra_in_kinematics,
        "mismatches": mismatches,
        "sample_count_tracked": len(tracked_ids),
        "sample_count_kinematics": len(kinematics_ids),
        "total_frames_kinematics": total_frames,
    }


def _csv_rows(flexion: dict[str, Any], spread: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel, stats in sorted(flexion.items()):
        hand, finger, joint = channel.split(".")
        rows.append(
            {
                "metric": "flexion_deg",
                "hand": hand,
                "channel": channel,
                "finger": finger,
                "component": joint,
                **stats,
            }
        )
    for channel, stats in sorted(spread.items()):
        hand, pair = channel.split(".")
        rows.append(
            {
                "metric": "adjacent_spread_deg",
                "hand": hand,
                "channel": channel,
                "finger": "",
                "component": pair,
                **stats,
            }
        )
    return rows


def validate_runs(tracked_run: str | Path, kinematics_run: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tracked_path = Path(tracked_run)
    kinematics_path = Path(kinematics_run)

    if not tracked_path.is_dir():
        raise ContractError(f"tracked run does not exist: {tracked_path}")
    if not kinematics_path.is_dir():
        raise ContractError(f"kinematics run does not exist: {kinematics_path}")

    tracked_ids = list_sample_ids(tracked_path, TRACKED_NPZ_NAME)
    kinematics_ids = list_sample_ids(kinematics_path, KINEMATICS_NPZ_NAME)

    samples: dict[str, SampleKinematics] = {}
    structurally_valid_ids: list[str] = []
    contract_failures: dict[str, list[str]] = {}
    load_failures: dict[str, str] = {}

    for sample_id in kinematics_ids:
        try:
            sample = load_kinematics_sample(kinematics_path, sample_id)
        except ContractError as error:
            load_failures[sample_id] = str(error)
            continue
        samples[sample_id] = sample
        contract = validate_sample_contract(sample)
        if not contract["passed"]:
            contract_failures[sample_id] = contract["failures"]
        else:
            structurally_valid_ids.append(sample_id)

    loaded_samples = [samples[sid] for sid in sorted(structurally_valid_ids)]

    alignment_samples = {sample_id: samples[sample_id] for sample_id in structurally_valid_ids}
    alignment = _alignment_checks(tracked_path, tracked_ids, kinematics_ids, alignment_samples)
    invalid_mask, non_finite = _state_nan_checks(loaded_samples)
    rotation = _rotation_checks(loaded_samples)
    quaternion = _quaternion_checks(loaded_samples)
    flexion, spread, left_right, suspicious_values = _distribution_checks(loaded_samples)
    temporal = _temporal_checks(loaded_samples)

    contract_passed = not load_failures and not contract_failures

    summary = {
        "validity_contract": {
            "name": VALIDITY_CONTRACT_NAME,
            "version": VALIDITY_CONTRACT_VERSION,
            "semantics": "valid_palm_frame gates orientation; finite/NaN state is checked per channel",
        },
        "contract_validation": {
            "passed": contract_passed,
            "load_failures": load_failures,
            "sample_failures": contract_failures,
            "checked_samples": len(samples),
        },
        "tracking_alignment": alignment,
        "sample_frame_counts": {
            "tracked_samples": len(tracked_ids),
            "kinematics_samples": len(kinematics_ids),
            "checked_kinematics_samples": len(samples),
            "total_kinematics_frames": int(
                sum(np.asarray(sample.arrays["frame_index"]).shape[0] for sample in loaded_samples)
            ),
        },
        "invalid_mask_violations": invalid_mask,
        "non_finite_violations": non_finite,
        "rotation_errors": rotation,
        "quaternion_errors": quaternion,
        "flexion_statistics": flexion,
        "spread_statistics": spread,
        "temporal_statistics": temporal,
        "left_right_statistics": left_right,
        "suspicious_value_flags": {
            "count": len(suspicious_values),
            "violations": suspicious_values,
        },
    }

    summary["passed"] = (
        summary["contract_validation"]["passed"]
        and summary["tracking_alignment"]["passed"]
        and summary["invalid_mask_violations"]["count"] == 0
        and summary["non_finite_violations"]["count"] == 0
    )
    summary["verdict"] = "PASS" if summary["passed"] else "FAIL"

    return summary, _csv_rows(flexion, spread)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "hand",
        "channel",
        "finger",
        "component",
        "count",
        "min",
        "p1",
        "p50",
        "p95",
        "p99",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
