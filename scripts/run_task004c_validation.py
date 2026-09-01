#!/usr/bin/env python3
"""Validate the frozen TASK-004A tracker against the frozen TASK-004B benchmark.

Reads both frozen inputs read-only, aligns strictly on
``(sample_id, frame_index)``, computes every TASK-004C metric and writes the
machine-readable result. It never modifies the tracker, its config, or the
annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.annotations.task004b import (  # noqa: E402
    CLIP_SPECS,
    annotation_statistics,
    read_annotations,
)
from evaluation.tracking.validation import (  # noqa: E402
    ValidationError,
    align,
    ambiguity_calibration,
    extra_detection_case,
    false_presence,
    identity_accuracy,
    identity_switches,
    load_tracked_frames,
    occlusion_state_validity,
    per_clip_summary,
    quality_gate_evaluation,
    reacquisition,
    tracker_claimed_reacquisitions,
    stratified_identity,
    verify_raw_integrity,
    visibility_recall,
)

EXPECTED_ANNOTATION_STATS = {
    "videos": 8,
    "frames": 399,
    "visible_left_frames": 371,
    "visible_right_frames": 328,
    "fully_occluded_hand_labels": 18,
    "ambiguous_identity_frames": 10,
}
EXPECTED_FLAG_COUNTS = {"HAND_CROSSING": 124, "MOTION_BLUR": 26}
EXPECTED_LEFT_STATES = {"VISIBLE": 371, "PARTIALLY_OCCLUDED": 28}
EXPECTED_RIGHT_STATES = {
    "VISIBLE": 328,
    "PARTIALLY_OCCLUDED": 43,
    "FULLY_OCCLUDED": 18,
    "AMBIGUOUS": 10,
}


def check_annotation_integrity(stats: dict) -> dict:
    """Hard gate: recomputed statistics must match the locked contract."""

    failures: list[str] = []
    for key, expected in EXPECTED_ANNOTATION_STATS.items():
        if stats.get(key) != expected:
            failures.append(f"{key}: expected {expected}, got {stats.get(key)}")
    for flag, expected in EXPECTED_FLAG_COUNTS.items():
        got = stats.get("flags_frame_counts", {}).get(flag)
        if got != expected:
            failures.append(f"flag {flag}: expected {expected}, got {got}")
    if stats.get("left_state_counts") != EXPECTED_LEFT_STATES:
        failures.append(f"left states: {stats.get('left_state_counts')}")
    if stats.get("right_state_counts") != EXPECTED_RIGHT_STATES:
        failures.append(f"right states: {stats.get('right_state_counts')}")
    occluded = stats.get("partially_occluded_hand_labels", 0) + stats.get(
        "fully_occluded_hand_labels", 0
    )
    if occluded != 89:
        failures.append(f"total occluded labels: expected 89, got {occluded}")
    return {"passed": not failures, "failures": failures, "total_occluded_labels": occluded}


def evaluate_acceptance(results: dict) -> dict:
    """Acceptance criteria A-F, fixed before results were seen."""

    recall = results["visibility_recall"]
    criteria = {
        "A_identity_switches": {
            "requirement": "0 confirmed persistent LEFT/RIGHT identity switches",
            "observed": results["identity_switches"]["confirmed_switches"],
            "passed": results["identity_switches"]["confirmed_switches"] == 0,
        },
        "B_reacquisition": {
            "requirement": "100% correct reacquisition on unambiguous reappearances",
            "observed": results["reacquisition"]["accuracy_pct"],
            "annotated_events": results["reacquisition"]["events"],
            "tracker_claimed": results["tracker_claimed_reacquisitions"]["tracker_claimed_reacquisitions"],
            "tracker_claimed_contradicted": results["tracker_claimed_reacquisitions"][
                "contradicted_by_reference"
            ],
            "passed": (
                results["reacquisition"]["incorrect"] == 0
                and results["tracker_claimed_reacquisitions"]["contradicted_by_reference"] == 0
            ),
        },
        "C_full_occlusion": {
            "requirement": "0 fabricated poses on human FULLY_OCCLUDED/OUT_OF_FRAME",
            "observed": results["false_presence"]["false_presence_count"],
            "passed": results["false_presence"]["false_presence_count"] == 0,
        },
        "D_extra_detection": {
            "requirement": "s02_0176 frame 39 extra-detection handling PASS",
            "observed": results["extra_detection"]["status"],
            "passed": results["extra_detection"]["status"] == "PASS",
        },
        "E_raw_integrity": {
            "requirement": "18/18 source WiLoR raw NPZ unchanged",
            "observed": f"{results['raw_integrity']['raw_sha256_matched']}/"
            f"{results['raw_integrity']['tracked_samples_checked']}",
            "passed": results["raw_integrity"]["all_unchanged"]
            and results["raw_integrity"]["tracked_samples_checked"] == 18,
        },
        "F_visibility_recall": {
            "requirement": ">= 98% recall for clearly visible physical hands",
            "observed": recall["fully_visible"]["recall_pct"],
            "left_pct": recall["left"]["recall_pct"],
            "right_pct": recall["right"]["recall_pct"],
            "passed": (recall["fully_visible"]["recall_pct"] or 0.0) >= 98.0,
        },
    }
    criteria["all_passed"] = all(item["passed"] for item in criteria.values() if isinstance(item, dict))
    return criteria


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "evaluation/annotations/task004_hand_identity_visibility.csv",
    )
    parser.add_argument(
        "--tracked-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked"),
    )
    parser.add_argument(
        "--raw-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_full"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/tracking/TASK-004C-validation-results.json")
    parser.add_argument("--tracker-commit", default="00ec1d7de21837012fa3eb8faecbf635ac2503d6")
    parser.add_argument("--annotation-commit", default="012d58a989a079dbeca6e5cb49b26c384dd80c21")
    args = parser.parse_args()

    rows = read_annotations(args.annotations)
    stats = annotation_statistics(rows)
    integrity = check_annotation_integrity(stats)
    if not integrity["passed"]:
        print("ANNOTATION INTEGRITY FAILURE:", file=sys.stderr)
        for failure in integrity["failures"]:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    tracked = {
        sample_id: load_tracked_frames(args.tracked_run, sample_id)
        for sample_id in sorted({row.sample_id for row in rows})
    }
    try:
        pairs = align(rows, tracked)
    except ValidationError as error:
        print(f"ALIGNMENT FAILURE: {error}", file=sys.stderr)
        return 3

    results: dict = {
        "task": "TASK-004C",
        "frozen_inputs": {
            "tracker_commit": args.tracker_commit,
            "annotation_commit": args.annotation_commit,
            "tracked_run": str(args.tracked_run),
            "raw_run": str(args.raw_run),
            "annotations": str(args.annotations.relative_to(ROOT)),
        },
        "annotation_integrity": {**integrity, "statistics": stats},
        "aligned_frames": len(pairs),
        "visibility_recall": visibility_recall(pairs),
        "false_presence": false_presence(pairs),
        "identity_accuracy": identity_accuracy(pairs, position="box"),
        "identity_accuracy_wrist_crosscheck": identity_accuracy(pairs, position="wrist"),
        "identity_switches": identity_switches(pairs),
        "reacquisition": reacquisition(pairs),
        "tracker_claimed_reacquisitions": tracker_claimed_reacquisitions(pairs, args.tracked_run),
        "occlusion_state_validity": occlusion_state_validity(pairs),
        "ambiguity_calibration": ambiguity_calibration(pairs),
        "extra_detection": extra_detection_case(pairs, args.raw_run),
        "quality_gate": quality_gate_evaluation(pairs, args.tracked_run),
        "raw_integrity": verify_raw_integrity(args.tracked_run),
        "stratified": stratified_identity(pairs),
        "per_clip": per_clip_summary(pairs),
        "clip_roles": {spec.sample_id: spec.role for spec in CLIP_SPECS},
    }
    results["acceptance"] = evaluate_acceptance(results)
    results["verdict"] = (
        "TASK-004 TRACKING VALIDATED - READY FOR TASK-005"
        if results["acceptance"]["all_passed"]
        else "TASK-004 TRACKING NEEDS REVISION"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n")

    recall = results["visibility_recall"]
    identity = results["identity_accuracy"]
    print(f"aligned frames               : {results['aligned_frames']}")
    print(f"visible recall (L/R/overall) : {recall['left']['recall_pct']:.2f}% / "
          f"{recall['right']['recall_pct']:.2f}% / {recall['overall']['recall_pct']:.2f}%")
    print(f"identity accuracy            : {identity['correct_frames']}/{identity['evaluable_frames']} "
          f"= {identity['accuracy_pct']:.2f}%")
    print(f"confirmed switches           : {results['identity_switches']['confirmed_switches']}")
    print(f"false presence               : {results['false_presence']['false_presence_count']}")
    print(f"extra detection              : {results['extra_detection']['status']}")
    print(f"raw integrity                : {results['raw_integrity']['raw_sha256_matched']}"
          f"/{results['raw_integrity']['tracked_samples_checked']}")
    print(f"\nVERDICT: {results['verdict']}")
    print(f"wrote {args.output}")
    return 0 if results["acceptance"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
