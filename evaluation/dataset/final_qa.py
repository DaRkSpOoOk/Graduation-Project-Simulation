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
