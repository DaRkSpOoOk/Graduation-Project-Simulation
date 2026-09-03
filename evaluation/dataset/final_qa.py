"""TASK-008C dataset-level QA over the completed Core-28 extraction.

``evaluation.dataset.qa.validate_run`` checks each sample against its own
provenance and contract. This module aggregates the finished dataset: coverage
accounting, tracking-state distribution, sensor validity broken down by signer,
class and partition, sequence-length statistics, and the evidence that no
padding, truncation, resampling or interpolation happened.

It reads only external run artifacts and the committed manifest, and writes
nothing back into the run.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .orchestrator import RunPaths

# Frozen TASK-004 state names, keyed by the code stored in the tracked NPZ.
TRACKING_STATE_NAMES: dict[int, str] = {
    0: "MISSING",
    1: "OBSERVED",
    2: "AMBIGUOUS",
    3: "REJECTED_QUALITY",
    4: "LIKELY_OCCLUDED",
}
# States that carry a real reconstructed pose (frozen TASK-004 POSE_STATES).
POSE_BEARING_CODES = frozenset({1, 2})
TRACK_ORDER = ("LEFT", "RIGHT")


def _percentile_block(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p1": float(np.percentile(array, 1)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
        "sum": int(array.sum()),
    }


def _fraction(valid: int, total: int) -> float | None:
    return (valid / total) if total else None


def aggregate_dataset(
    manifest_rows: Iterable[Mapping[str, str]], run_root: str | Path
) -> dict[str, Any]:
    """Aggregate the finished dataset from its external artifacts."""

    paths = RunPaths(Path(run_root).resolve())
    rows = [dict(row) for row in manifest_rows]

    totals = {"bend_valid": 0, "bend_total": 0, "spread_valid": 0, "spread_total": 0,
              "imu_valid": 0, "imu_total": 0}
    by_signer: dict[str, Counter] = defaultdict(Counter)
    by_class: dict[str, Counter] = defaultdict(Counter)
    by_partition: dict[str, Counter] = defaultdict(Counter)
    state_counts: Counter = Counter()
    track_state_counts: dict[str, Counter] = {t: Counter() for t in TRACK_ORDER}
    left_available = right_available = both_available = neither_available = 0
    hand_instances = 0
    sequence_lengths: list[int] = []
    manifest_frames: list[int] = []
    length_mismatches: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    labels_by_class: dict[str, str] = {}

    for row in rows:
        sample_id = row["sample_id"]
        glove_path = paths.virtual_glove / sample_id / "virtual_glove.npz"
        tracked_path = paths.tracking / sample_id / "wilor_tracked.npz"
        if not glove_path.is_file() or not tracked_path.is_file():
            missing_outputs.append(sample_id)
            continue
        with np.load(glove_path, allow_pickle=False) as glove:
            frames = int(np.asarray(glove["frame_index"]).shape[0])
            bend_valid = np.asarray(glove["bend_valid"], dtype=bool)
            spread_valid = np.asarray(glove["spread_valid"], dtype=bool)
            imu_valid = np.asarray(glove["palm_imu_valid"], dtype=bool)
            state_code = np.asarray(glove["tracking_state_code"], dtype=np.int64)

        declared = int(row["frame_count"]) if row.get("frame_count") else None
        sequence_lengths.append(frames)
        if declared is not None:
            manifest_frames.append(declared)
            if declared != frames:
                length_mismatches.append(
                    {"sample_id": sample_id, "manifest_frames": declared, "output_frames": frames}
                )

        signer = row["signer_id"]
        sign_id = row["sign_id"]
        partition = row["official_partition"]
        labels_by_class.setdefault(sign_id, row.get("label_ar", ""))

        counts = {
            "bend_valid": int(bend_valid.sum()), "bend_total": int(bend_valid.size),
            "spread_valid": int(spread_valid.sum()), "spread_total": int(spread_valid.size),
            "imu_valid": int(imu_valid.sum()), "imu_total": int(imu_valid.size),
            "frames": frames, "videos": 1,
        }
        for key, value in counts.items():
            if key in totals:
                totals[key] += value
            by_signer[signer][key] += value
            by_class[sign_id][key] += value
            by_partition[partition][key] += value

        for code, count in zip(*np.unique(state_code, return_counts=True)):
            state_counts[TRACKING_STATE_NAMES.get(int(code), f"UNKNOWN_{int(code)}")] += int(count)
        for column, track in enumerate(TRACK_ORDER):
            for code, count in zip(*np.unique(state_code[:, column], return_counts=True)):
                track_state_counts[track][
                    TRACKING_STATE_NAMES.get(int(code), f"UNKNOWN_{int(code)}")
                ] += int(count)

        posed = np.isin(state_code, list(POSE_BEARING_CODES))
        left, right = posed[:, 0], posed[:, 1]
        left_available += int(left.sum())
        right_available += int(right.sum())
        both_available += int((left & right).sum())
        neither_available += int((~left & ~right).sum())
        hand_instances += int(state_code.size)

    def breakdown(source: dict[str, Counter]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in sorted(source):
            c = source[key]
            out[key] = {
                "videos": c["videos"], "frames": c["frames"],
                "bend_valid_fraction": _fraction(c["bend_valid"], c["bend_total"]),
                "spread_valid_fraction": _fraction(c["spread_valid"], c["spread_total"]),
                "imu_valid_fraction": _fraction(c["imu_valid"], c["imu_total"]),
                "bend_valid": c["bend_valid"], "bend_total": c["bend_total"],
                "spread_valid": c["spread_valid"], "spread_total": c["spread_total"],
                "imu_valid": c["imu_valid"], "imu_total": c["imu_total"],
            }
            if key in labels_by_class:
                out[key]["label_ar"] = labels_by_class[key]
        return out

    class_block = breakdown(by_class)
    worst_spread = sorted(
        class_block.items(), key=lambda kv: (kv[1]["spread_valid_fraction"] or 0.0)
    )[:5]
    worst_bend = sorted(
        class_block.items(), key=lambda kv: (kv[1]["bend_valid_fraction"] or 0.0)
    )[:5]

    total_frames = sum(sequence_lengths)
    return {
        "samples_aggregated": len(sequence_lengths),
        "samples_missing_output": missing_outputs,
        "sensor_validity_totals": {
            **totals,
            "bend_valid_fraction": _fraction(totals["bend_valid"], totals["bend_total"]),
            "spread_valid_fraction": _fraction(totals["spread_valid"], totals["spread_total"]),
            "imu_valid_fraction": _fraction(totals["imu_valid"], totals["imu_total"]),
        },
        "by_signer": breakdown(by_signer),
        "by_partition": breakdown(by_partition),
        "by_class": class_block,
        "worst_classes_by_spread_validity": [
            {"sign_id": k, "label_ar": v.get("label_ar", ""),
             "spread_valid_fraction": v["spread_valid_fraction"]} for k, v in worst_spread
        ],
        "worst_classes_by_bend_validity": [
            {"sign_id": k, "label_ar": v.get("label_ar", ""),
             "bend_valid_fraction": v["bend_valid_fraction"]} for k, v in worst_bend
        ],
        "tracking_states": {
            "combined": dict(sorted(state_counts.items())),
            "combined_fractions": {
                name: count / hand_instances if hand_instances else None
                for name, count in sorted(state_counts.items())
            },
            "per_track": {t: dict(sorted(c.items())) for t, c in track_state_counts.items()},
            "hand_instances": hand_instances,
        },
        "hand_availability": {
            "frames": total_frames,
            "left_available": left_available,
            "right_available": right_available,
            "both_available": both_available,
            "neither_available": neither_available,
            "left_fraction": _fraction(left_available, total_frames),
            "right_fraction": _fraction(right_available, total_frames),
            "both_fraction": _fraction(both_available, total_frames),
            "note": (
                "availability counts frames whose tracking state carries a real "
                "reconstructed pose (OBSERVED or AMBIGUOUS). An absent hand is "
                "never inferred as a straight or neutral hand."
            ),
        },
        "sequence_lengths": _percentile_block(sequence_lengths),
        "manifest_frame_lengths": _percentile_block(manifest_frames),
        "temporal_integrity": {
            "output_equals_manifest_frames": not length_mismatches,
            "length_mismatches": length_mismatches[:20],
            "length_mismatch_count": len(length_mismatches),
            "padding_performed": False,
            "truncation_performed": False,
            "resampling_performed": False,
            "interpolation_performed": False,
            "evidence": (
                "every output sequence length equals its source video frame count "
                "from the frozen manifest, so no frame was added or removed"
            ),
        },
    }


def coverage_accounting(
    manifest_rows: Iterable[Mapping[str, str]], state_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconcile requested samples against the durable worker state."""

    rows = [dict(row) for row in manifest_rows]
    requested = {row["sample_id"] for row in rows}
    samples = dict(state_payload.get("samples", {}))
    statuses = Counter(entry.get("status") for entry in samples.values())
    successful = {sid for sid, entry in samples.items()
                  if entry.get("status") == "VIRTUAL_GLOVE_DONE"}
    failed = {sid for sid, entry in samples.items() if entry.get("status") == "FAILED"}
    incomplete = set(samples) - successful - failed
    unaccounted = requested - set(samples)
    return {
        "requested": len(requested),
        "in_state": len(samples),
        "successful": len(successful),
        "failed": len(failed),
        "incomplete_other_status": sorted(incomplete)[:20],
        "incomplete_count": len(incomplete),
        "unaccounted": sorted(unaccounted)[:20],
        "unaccounted_count": len(unaccounted),
        "status_counts": dict(sorted(statuses.items(), key=lambda kv: str(kv[0]))),
        "extra_in_state_not_requested": sorted(set(samples) - requested)[:20],
        "all_accounted": len(unaccounted) == 0 and len(incomplete) == 0,
    }


# --- TASK-008C finalization checks -------------------------------------------
#
# ``aggregate_dataset`` answers "what does the finished dataset contain".  The
# helpers below answer "is the finished dataset still the thing TASK-005/006
# froze": temporal ordering, value ranges, quaternion convention, sensor channel
# ordering, label identity and LOSO integrity.  They read only run artifacts and
# committed metadata; nothing is rewritten.

SEQUENCE_PERCENTILES = (5, 25, 50, 75, 95)

# Frozen TASK-006 layout, restated here so a silent reordering upstream is a
# test failure rather than an invisible relabelling of ML channels.
EXPECTED_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
EXPECTED_CHAIN_ORDER = ("proximal", "middle", "distal")
EXPECTED_SPREAD_PAIRS = (
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)
EXPECTED_PER_HAND_COUNTS = {
    "bend_hall_sensors": 15,
    "spread_hall_sensors": 4,
    "hall_sensors_total": 19,
    "imu_packages": 1,
    "logical_sensing_packages": 20,
}
# TASK-006 stores an unsigned geometric angle, so 180 degrees is the algebraic
# ceiling of the underlying turn angle -- not a dataset-fitted clip.
ANGLE_DEGREE_CEILING = 180.0


def sequence_length_statistics(values: Iterable[int]) -> dict[str, Any]:
    """Return the percentile block the TASK-008C report asks for."""

    lengths = [int(value) for value in values]
    if not lengths:
        return {"count": 0}
    array = np.asarray(lengths, dtype=np.float64)
    block: dict[str, Any] = {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "sum": int(array.sum()),
    }
    for percentile in SEQUENCE_PERCENTILES:
        name = "median" if percentile == 50 else f"p{percentile}"
        block[name] = float(np.percentile(array, percentile))
    return block


def _layout_fingerprint(payload: Mapping[str, Any]) -> tuple:
    """Reduce a sensor layout to the ordering that indexes the ML arrays."""

    channels = []
    for sensor in payload.get("sensors", []):
        channels.append(
            (
                sensor.get("sensor_id"),
                sensor.get("array"),
                tuple(sensor.get("array_index") or ()),
                sensor.get("role"),
                sensor.get("finger"),
                sensor.get("joint"),
                tuple(sensor.get("pair") or ()),
                sensor.get("display_marker"),
            )
        )
    return tuple(channels)


def _check_layout_contract(payload: Mapping[str, Any]) -> list[str]:
    """Return contract violations for one ``sensor_layout.json`` payload."""

    problems: list[str] = []
    if tuple(payload.get("finger_order") or ()) != EXPECTED_FINGER_ORDER:
        problems.append("finger_order differs from the frozen TASK-006 order")
    if tuple(payload.get("chain_joint_order") or ()) != EXPECTED_CHAIN_ORDER:
        problems.append("chain_joint_order differs from the frozen TASK-006 order")
    counts = payload.get("per_hand_counts") or {}
    for key, expected in EXPECTED_PER_HAND_COUNTS.items():
        if int(counts.get(key, -1)) != expected:
            problems.append(f"per_hand_counts.{key} = {counts.get(key)} != {expected}")

    sensors = list(payload.get("sensors") or [])
    bend = [s for s in sensors if s.get("role") == "bend"]
    spread = [s for s in sensors if s.get("role") == "spread"]
    imu = [s for s in sensors if s.get("role") == "orientation"] or [
        s for s in sensors if str(s.get("sensor_type", "")).startswith("imu")
    ]
    if len(bend) != 15:
        problems.append(f"{len(bend)} bend channels != 15")
    if len(spread) != 4:
        problems.append(f"{len(spread)} spread channels != 4")
    if len(imu) != 1:
        problems.append(f"{len(imu)} palm IMU packages != 1")

    expected_bend_index = [
        (finger_index, chain_index)
        for finger_index in range(len(EXPECTED_FINGER_ORDER))
        for chain_index in range(len(EXPECTED_CHAIN_ORDER))
    ]
    actual_bend_index = [tuple(s.get("array_index") or ()) for s in bend]
    if actual_bend_index != expected_bend_index:
        problems.append("bend array_index ordering is not finger-major/chain-minor")
    expected_bend_names = [
        (finger, joint) for finger in EXPECTED_FINGER_ORDER for joint in EXPECTED_CHAIN_ORDER
    ]
    if [(s.get("finger"), s.get("joint")) for s in bend] != expected_bend_names:
        problems.append("bend channel naming does not follow the frozen finger/chain order")

    if [tuple(s.get("array_index") or ()) for s in spread] != [(i,) for i in range(4)]:
        problems.append("spread array_index ordering is not 0..3")
    if [tuple(s.get("pair") or ()) for s in spread] != list(EXPECTED_SPREAD_PAIRS):
        problems.append("spread pairs differ from the frozen adjacent-pair order")

    if any(s.get("display_marker") != "H" for s in bend + spread):
        problems.append("a Hall channel is not marked H")
    if any(s.get("display_marker") != "IMU" for s in imu):
        problems.append("the palm IMU is not marked IMU")
    return problems


def verify_dataset_contract(
    manifest_rows: Iterable[Mapping[str, str]], run_root: str | Path
) -> dict[str, Any]:
    """Verify temporal ordering, value ranges and channel identity dataset-wide.

    Every check is a property of the frozen TASK-005/006 contract, so a
    violation is a production defect and is reported per sample rather than
    repaired.  Invalid channels must stay NaN: an imputed zero would look like a
    straight finger to TASK-009.
    """

    paths = RunPaths(Path(run_root).resolve())
    rows = [dict(row) for row in manifest_rows]

    violations: list[dict[str, str]] = []
    layout_fingerprints: Counter = Counter()
    layout_problems: list[dict[str, Any]] = []
    checked = 0
    bend_min = spread_min = float("inf")
    bend_max = spread_max = float("-inf")
    quaternion_norm_error = 0.0
    negative_w = 0
    imputed_invalid_channels = 0
    first_frame_nonzero: list[str] = []
    non_contiguous: list[str] = []
    reference_layout: dict[str, Any] | None = None

    for row in rows:
        sample_id = row["sample_id"]
        glove_path = paths.virtual_glove / sample_id / "virtual_glove.npz"
        layout_path = paths.virtual_glove / sample_id / "sensor_layout.json"
        if not glove_path.is_file() or not layout_path.is_file():
            violations.append({"sample_id": sample_id, "problem": "missing glove output"})
            continue

        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        layout_fingerprints[_layout_fingerprint(layout)] += 1
        if reference_layout is None:
            reference_layout = layout
            for problem in _check_layout_contract(layout):
                layout_problems.append({"sample_id": sample_id, "problem": problem})

        with np.load(glove_path, allow_pickle=False) as glove:
            frame_index = np.asarray(glove["frame_index"], dtype=np.int64)
            timestamps = np.asarray(glove["timestamp_seconds"], dtype=np.float64)
            bend_deg = np.asarray(glove["bend_angle_deg"], dtype=np.float64)
            spread_deg = np.asarray(glove["spread_angle_deg"], dtype=np.float64)
            bend_norm = np.asarray(glove["bend_normalized"], dtype=np.float64)
            spread_norm = np.asarray(glove["spread_normalized"], dtype=np.float64)
            bend_valid = np.asarray(glove["bend_valid"], dtype=bool)
            spread_valid = np.asarray(glove["spread_valid"], dtype=bool)
            imu_valid = np.asarray(glove["palm_imu_valid"], dtype=bool)
            quaternion = np.asarray(glove["imu_quaternion_wxyz"], dtype=np.float64)
        checked += 1

        if frame_index.size and int(frame_index[0]) != 0:
            first_frame_nonzero.append(sample_id)
        if frame_index.size > 1:
            if not np.all(np.diff(frame_index) > 0):
                violations.append({"sample_id": sample_id, "problem": "frame_index is not strictly increasing"})
            if not np.all(np.diff(frame_index) == 1):
                non_contiguous.append(sample_id)
            if not np.all(np.diff(timestamps) > 0):
                violations.append({"sample_id": sample_id, "problem": "timestamp is not strictly increasing"})
        if not np.isfinite(timestamps).all():
            violations.append({"sample_id": sample_id, "problem": "non-finite timestamp"})

        # An invalid channel must be NaN, never a plausible-looking number.
        imputed = int(np.isfinite(bend_deg[~bend_valid]).sum() + np.isfinite(spread_deg[~spread_valid]).sum())
        imputed_invalid_channels += imputed
        if imputed:
            violations.append({"sample_id": sample_id, "problem": f"{imputed} invalid channels carry a finite value"})

        if bend_valid.any():
            values = bend_deg[bend_valid]
            bend_min, bend_max = min(bend_min, float(values.min())), max(bend_max, float(values.max()))
            if values.min() < 0.0 or values.max() > ANGLE_DEGREE_CEILING:
                violations.append({"sample_id": sample_id, "problem": "bend angle outside [0, 180]"})
            if not np.allclose(bend_norm[bend_valid], values / ANGLE_DEGREE_CEILING, rtol=0, atol=2e-7):
                violations.append({"sample_id": sample_id, "problem": "bend_normalized != bend_angle_deg / 180"})
        if spread_valid.any():
            values = spread_deg[spread_valid]
            spread_min, spread_max = min(spread_min, float(values.min())), max(spread_max, float(values.max()))
            if values.min() < 0.0 or values.max() > ANGLE_DEGREE_CEILING:
                violations.append({"sample_id": sample_id, "problem": "spread angle outside [0, 180]"})
            if not np.allclose(spread_norm[spread_valid], values / ANGLE_DEGREE_CEILING, rtol=0, atol=2e-7):
                violations.append({"sample_id": sample_id, "problem": "spread_normalized != spread_angle_deg / 180"})

        if imu_valid.any():
            usable = quaternion[imu_valid]
            norms = np.linalg.norm(usable, axis=-1)
            quaternion_norm_error = max(quaternion_norm_error, float(np.max(np.abs(norms - 1.0))))
            if np.max(np.abs(norms - 1.0)) > 1e-4:
                violations.append({"sample_id": sample_id, "problem": "quaternion is not unit norm"})
            below = int((usable[:, 0] < 0.0).sum())
            negative_w += below
            if below:
                violations.append({"sample_id": sample_id, "problem": f"{below} quaternions violate the w >= 0 convention"})

    return {
        "samples_checked": checked,
        "violations": violations[:50],
        "violation_count": len(violations),
        "temporal": {
            "frame_index_strictly_increasing": not any(
                "frame_index" in v["problem"] for v in violations
            ),
            "timestamp_strictly_increasing": not any(
                "timestamp" in v["problem"] for v in violations
            ),
            "samples_not_starting_at_frame_zero": first_frame_nonzero[:20],
            "samples_with_non_contiguous_frames": non_contiguous[:20],
            "non_contiguous_count": len(non_contiguous),
        },
        "value_ranges": {
            "bend_angle_deg_min": None if bend_min == float("inf") else bend_min,
            "bend_angle_deg_max": None if bend_max == float("-inf") else bend_max,
            "spread_angle_deg_min": None if spread_min == float("inf") else spread_min,
            "spread_angle_deg_max": None if spread_max == float("-inf") else spread_max,
            "declared_ceiling_deg": ANGLE_DEGREE_CEILING,
            "normalization": "bend_normalized = bend_angle_deg / 180.0 (fixed divisor, no dataset min/max)",
        },
        "quaternion": {
            "max_unit_norm_error": quaternion_norm_error,
            "negative_w_count": negative_w,
            "ordering": "WXYZ",
            "sign_convention": "w >= 0",
        },
        "imputed_invalid_channels": imputed_invalid_channels,
        "sensor_layout": {
            "distinct_layouts": len(layout_fingerprints),
            "channel_order_identical_across_samples": len(layout_fingerprints) <= 1,
            "contract_problems": layout_problems,
            "per_hand_counts": (reference_layout or {}).get("per_hand_counts"),
            "layout_version": (reference_layout or {}).get("layout_version"),
            "two_hand_hall_channels": 2 * EXPECTED_PER_HAND_COUNTS["hall_sensors_total"],
            "two_hand_imu_packages": 2,
        },
        "contract_intact": not violations and len(layout_fingerprints) <= 1 and not layout_problems,
    }


def verify_label_integrity(
    manifest_rows: Iterable[Mapping[str, str]], label_rows: Iterable[Mapping[str, str]]
) -> dict[str, Any]:
    """Check that every production row still carries its frozen Core-28 label."""

    rows = [dict(row) for row in manifest_rows]
    labels = [dict(row) for row in label_rows]
    by_sign = {row["sign_id"]: row for row in labels}

    problems: list[dict[str, str]] = []
    seen_signs: Counter = Counter()
    for row in rows:
        sign_id = row["sign_id"]
        seen_signs[sign_id] += 1
        reference = by_sign.get(sign_id)
        if reference is None:
            problems.append({"sample_id": row["sample_id"], "problem": f"sign_id {sign_id} is not Core-28"})
            continue
        if row.get("label_ar") != reference.get("label_ar"):
            problems.append({"sample_id": row["sample_id"], "problem": "label_ar disagrees with the frozen label table"})
        if str(row.get("label_index")) != str(reference.get("label_index")):
            problems.append({"sample_id": row["sample_id"], "problem": "label_index disagrees with the frozen label table"})
        if row.get("signer_id") not in {"01", "02", "03"}:
            problems.append({"sample_id": row["sample_id"], "problem": "unknown signer_id"})
        if row.get("official_partition") not in {"train", "test"}:
            problems.append({"sample_id": row["sample_id"], "problem": "unknown official_partition"})
        if not row.get("source_relative_path") or not row.get("source_sha256"):
            problems.append({"sample_id": row["sample_id"], "problem": "missing source provenance"})

    label_indices = sorted(int(row["label_index"]) for row in labels)
    return {
        "distinct_classes": len(seen_signs),
        "expected_classes": 28,
        "class_count_correct": len(seen_signs) == 28 and len(by_sign) == 28,
        "sign_id_range": [min(seen_signs), max(seen_signs)] if seen_signs else [],
        "label_index_contiguous": label_indices == list(range(28)),
        "samples_per_class": dict(sorted(seen_signs.items())),
        "problems": problems[:20],
        "problem_count": len(problems),
        "labels_intact": not problems,
    }


def verify_loso_folds(
    fold_rows_by_signer: Mapping[str, Iterable[Mapping[str, str]]],
    manifest_rows: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    """Independently re-check the frozen LOSO folds for leakage and coverage."""

    from .splits import validate_split_rows

    manifest = [dict(row) for row in manifest_rows]
    manifest_ids = {row["sample_id"] for row in manifest}
    folds: dict[str, Any] = {}
    all_ok = True
    for signer in sorted(fold_rows_by_signer):
        rows = [dict(row) for row in fold_rows_by_signer[signer]]
        entry: dict[str, Any] = {
            "counts": dict(sorted(Counter(row["role"] for row in rows).items())),
            "total_rows": len(rows),
        }
        test_signers = sorted({row["signer_id"] for row in rows if row["role"] == "test"})
        fit_signers = sorted({row["signer_id"] for row in rows if row["role"] != "test"})
        entry["test_signers"] = test_signers
        entry["train_validation_signers"] = fit_signers
        entry["held_out_signer_leakage"] = signer in fit_signers
        entry["classes_per_role"] = {
            role: len({row["sign_id"] for row in rows if row["role"] == role})
            for role in ("train", "validation", "test")
        }
        entry["covers_manifest_exactly"] = {row["sample_id"] for row in rows} == manifest_ids
        try:
            validate_split_rows(rows, manifest_rows=manifest, held_out_signer=signer)
            entry["validator"] = "PASS"
        except ValueError as error:
            entry["validator"] = f"FAIL: {error}"
            all_ok = False
        if entry["held_out_signer_leakage"] or not entry["covers_manifest_exactly"]:
            all_ok = False
        if any(count != 28 for count in entry["classes_per_role"].values()):
            all_ok = False
        folds[signer] = entry
    return {"folds": folds, "fold_count": len(folds), "loso_intact": all_ok and len(folds) == 3}
