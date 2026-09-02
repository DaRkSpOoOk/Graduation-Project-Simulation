#!/usr/bin/env python3
"""Derive TASK-005A hand kinematics over the KArSL pilot.

Reads the TASK-004D remediated tracked run by explicit path, derives per-frame
finger flexion, adjacent-finger spread and a canonical palm frame for both
tracks, and writes the kinematics stage plus a diagnostics summary to an
ignored output directory. The tracked input is opened read-only; its SHA-256
is recorded before and after the run and compared.

Example:

    python scripts/run_task005a_kinematics.py \\
      --tracked-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d \\
      --out-dir /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005a \\
      --strict-counts
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics import (  # noqa: E402
    CHAIN_ORDER,
    FINGER_ORDER,
    SPREAD_PAIRS,
    TRACK_ORDER,
    build_metadata,
    extract_sequence,
    save_kinematics,
    sha256_file,
)
from kinematics.geometry import orthonormality_error  # noqa: E402
from tracking.wilor import load_tracked_sequence  # noqa: E402
from tracking.wilor.npz_io import TRACKED_NPZ_NAME  # noqa: E402
from tracking.wilor.schema import CODE_TO_STATE  # noqa: E402

EXPECTED_SAMPLES = 18
EXPECTED_FRAMES = 894


def _sample_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise SystemExit(f"Manifest {manifest} has no sample_id column")
    return [row["sample_id"] for row in rows]


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _percentile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _channel_summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def _consecutive_deltas(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Absolute frame-to-frame change, only across consecutive valid pairs.

    Gaps are skipped rather than bridged: a jump measured across an occlusion
    would describe the occlusion, not the kinematics.
    """

    both = valid[:-1] & valid[1:]
    if not both.any():
        return np.empty((0,), dtype=np.float64)
    deltas = np.abs(values[1:] - values[:-1])
    deltas = deltas[both]
    return deltas[np.isfinite(deltas)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_milestone1_pilot.csv")
    parser.add_argument(
        "--tracked-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_tracked_task004d"),
        help="TASK-004D remediated tracked run (read-only)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="ignored output directory")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--strict-counts", action="store_true", help="require the exact 18-video / 894-frame pilot")
    args = parser.parse_args()

    sample_ids = args.sample_ids or _sample_ids(args.manifest)
    commit = _commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_video: list[dict] = []
    total_frames = 0
    state_to_valid: dict[str, dict[str, int]] = {}
    flag_counts: dict[str, int] = {}
    flexion_all: list[np.ndarray] = []
    spread_all: list[np.ndarray] = []
    flexion_by_track: dict[str, list[np.ndarray]] = {track: [] for track in TRACK_ORDER}
    spread_by_track: dict[str, list[np.ndarray]] = {track: [] for track in TRACK_ORDER}
    flexion_deltas: list[np.ndarray] = []
    spread_deltas: list[np.ndarray] = []
    quaternion_deltas: list[np.ndarray] = []
    quat_norm_errors: list[float] = []
    orthogonality_errors: list[float] = []
    determinant_errors: list[float] = []
    largest_jumps: list[dict] = []

    for sample_id in sample_ids:
        tracked_dir = args.tracked_run / sample_id
        tracked_npz = tracked_dir / TRACKED_NPZ_NAME
        if not tracked_npz.is_file():
            raise SystemExit(f"Tracked input missing: {tracked_npz}")
        sha_before = sha256_file(tracked_npz)

        arrays, tracked_meta = load_tracked_sequence(tracked_dir)
        sequence = extract_sequence(arrays, tracked_meta, sample_id)

        sha_after = sha256_file(tracked_npz)
        if sha_before != sha_after:
            raise SystemExit(f"Tracked input mutated during processing: {tracked_npz}")

        metadata = build_metadata(
            sequence,
            tracked_dir=tracked_dir,
            tracked_sha256=sha_before,
            tracked_metadata=tracked_meta,
            implementation_commit=commit,
        )
        save_kinematics(args.out_dir / sample_id, sequence, metadata)

        frames = int(sequence.frame_index.shape[0])
        total_frames += frames
        valid = sequence.valid_kinematics
        palm_valid = sequence.valid_palm_frame

        # --- state <-> validity consistency ---------------------------------
        for row in range(frames):
            for column, track in enumerate(TRACK_ORDER):
                state = CODE_TO_STATE[int(sequence.tracking_state_code[row, column])].value
                bucket = state_to_valid.setdefault(state, {"valid": 0, "invalid": 0})
                bucket["valid" if valid[row, column] else "invalid"] += 1
                for flag in json.loads(str(sequence.kinematic_flags_json[row, column])):
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # --- numeric diagnostics -------------------------------------------
        flexion_all.append(sequence.flexion_deg.reshape(-1))
        spread_all.append(sequence.adjacent_spread_deg.reshape(-1))
        for column, track in enumerate(TRACK_ORDER):
            flexion_by_track[track].append(sequence.flexion_deg[:, column].reshape(-1))
            spread_by_track[track].append(sequence.adjacent_spread_deg[:, column].reshape(-1))

            column_valid = valid[:, column]
            for finger in range(len(FINGER_ORDER)):
                for chain in range(len(CHAIN_ORDER)):
                    series = sequence.flexion_deg[:, column, finger, chain].astype(np.float64)
                    deltas = _consecutive_deltas(series, np.isfinite(series))
                    flexion_deltas.append(deltas)
                    if deltas.size:
                        peak = float(deltas.max())
                        largest_jumps.append({
                            "sample_id": sample_id, "track": track, "channel": "flexion",
                            "finger": FINGER_ORDER[finger], "chain": CHAIN_ORDER[chain],
                            "delta_deg": peak,
                        })
            for pair in range(len(SPREAD_PAIRS)):
                series = sequence.adjacent_spread_deg[:, column, pair].astype(np.float64)
                deltas = _consecutive_deltas(series, np.isfinite(series))
                spread_deltas.append(deltas)
                if deltas.size:
                    largest_jumps.append({
                        "sample_id": sample_id, "track": track, "channel": "spread",
                        "pair": "-".join(SPREAD_PAIRS[pair]),
                        "delta_deg": float(deltas.max()),
                    })

            # palm orientation change as a geodesic angle between quaternions
            quats = sequence.palm_quaternion_wxyz[:, column].astype(np.float64)
            column_palm_valid = palm_valid[:, column]
            both = column_palm_valid[:-1] & column_palm_valid[1:]
            if both.any():
                dot = np.abs(np.sum(quats[1:] * quats[:-1], axis=1))
                angle = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
                angle = angle[both]
                quaternion_deltas.append(angle[np.isfinite(angle)])

            for row in range(frames):
                if not palm_valid[row, column]:
                    continue
                quat = quats[row]
                quat_norm_errors.append(abs(float(np.linalg.norm(quat)) - 1.0))
                rotation = sequence.palm_rotation_matrix[row, column].astype(np.float64)
                orthogonality_errors.append(orthonormality_error(rotation))
                determinant_errors.append(abs(float(np.linalg.det(rotation)) - 1.0))

        valid_count = int(valid.sum())
        per_video.append({
            "sample_id": sample_id,
            "frames": frames,
            "hand_instances": frames * len(TRACK_ORDER),
            "valid_hand_instances": valid_count,
            "invalid_hand_instances": frames * len(TRACK_ORDER) - valid_count,
            "valid_left": int(valid[:, 0].sum()),
            "valid_right": int(valid[:, 1].sum()),
            "valid_palm_frame_instances": int(palm_valid.sum()),
            "tracked_npz_sha256": sha_before,
        })
        print(
            f"[{sample_id}] frames={frames} strict={valid_count}/{frames * 2} "
            f"palm={int(palm_valid.sum())}/{frames * 2} "
            f"L={int(valid[:, 0].sum())} R={int(valid[:, 1].sum())}"
        )

    if args.strict_counts:
        if len(per_video) != EXPECTED_SAMPLES or total_frames != EXPECTED_FRAMES:
            raise SystemExit(
                f"Strict count check failed: {len(per_video)} videos / {total_frames} frames "
                f"(expected {EXPECTED_SAMPLES} / {EXPECTED_FRAMES})"
            )

    flexion_flat = np.concatenate(flexion_all) if flexion_all else np.empty(0)
    spread_flat = np.concatenate(spread_all) if spread_all else np.empty(0)
    flexion_delta_flat = np.concatenate([d for d in flexion_deltas if d.size]) if flexion_deltas else np.empty(0)
    spread_delta_flat = np.concatenate([d for d in spread_deltas if d.size]) if spread_deltas else np.empty(0)
    quat_delta_flat = np.concatenate([d for d in quaternion_deltas if d.size]) if quaternion_deltas else np.empty(0)

    total_instances = sum(v["hand_instances"] for v in per_video)
    total_valid = sum(v["valid_hand_instances"] for v in per_video)
    total_palm_valid = sum(v["valid_palm_frame_instances"] for v in per_video)

    summary = {
        "task": "TASK-005A",
        "implementation_commit": commit,
        "tracked_run": str(args.tracked_run),
        "out_dir": str(args.out_dir),
        "videos": len(per_video),
        "frames": total_frames,
        "hand_instances": total_instances,
        "valid_hand_instances": total_valid,
        "invalid_hand_instances": total_instances - total_valid,
        "valid_pct": 100.0 * total_valid / total_instances if total_instances else None,
        "valid_palm_frame_instances": total_palm_valid,
        "valid_palm_frame_pct": 100.0 * total_palm_valid / total_instances if total_instances else None,
        "strictly_invalid_but_palm_frame_usable": total_palm_valid - total_valid,
        "tracking_state_to_validity": state_to_valid,
        "flag_counts": dict(sorted(flag_counts.items())),
        "nan_counts": {
            "flexion_deg": int(np.count_nonzero(~np.isfinite(flexion_flat))),
            "flexion_deg_total": int(flexion_flat.size),
            "adjacent_spread_deg": int(np.count_nonzero(~np.isfinite(spread_flat))),
            "adjacent_spread_deg_total": int(spread_flat.size),
        },
        "channel_ranges": {
            "flexion_deg": _channel_summary(flexion_flat),
            "adjacent_spread_deg": _channel_summary(spread_flat),
        },
        "per_finger_flexion": {},
        "per_pair_spread": {},
        "left_right": {
            track: {
                "flexion_deg": _channel_summary(np.concatenate(flexion_by_track[track])),
                "adjacent_spread_deg": _channel_summary(np.concatenate(spread_by_track[track])),
            }
            for track in TRACK_ORDER
        },
        "temporal_change": {
            "flexion_deg": {
                "count": int(flexion_delta_flat.size),
                "p50": _percentile(flexion_delta_flat, 50),
                "p95": _percentile(flexion_delta_flat, 95),
                "p99": _percentile(flexion_delta_flat, 99),
                "max": float(flexion_delta_flat.max()) if flexion_delta_flat.size else None,
            },
            "adjacent_spread_deg": {
                "count": int(spread_delta_flat.size),
                "p50": _percentile(spread_delta_flat, 50),
                "p95": _percentile(spread_delta_flat, 95),
                "p99": _percentile(spread_delta_flat, 99),
                "max": float(spread_delta_flat.max()) if spread_delta_flat.size else None,
            },
            "palm_orientation_deg": {
                "count": int(quat_delta_flat.size),
                "p50": _percentile(quat_delta_flat, 50),
                "p95": _percentile(quat_delta_flat, 95),
                "p99": _percentile(quat_delta_flat, 99),
                "max": float(quat_delta_flat.max()) if quat_delta_flat.size else None,
            },
            "note": "consecutive valid frame pairs only; gaps are skipped, never bridged",
        },
        "numerical_quality": {
            "quaternion_norm_error_max": max(quat_norm_errors) if quat_norm_errors else None,
            "rotation_orthogonality_error_max": max(orthogonality_errors) if orthogonality_errors else None,
            "rotation_determinant_error_max": max(determinant_errors) if determinant_errors else None,
            "samples": len(quat_norm_errors),
        },
        "largest_temporal_jumps": sorted(largest_jumps, key=lambda item: -item["delta_deg"])[:15],
        "per_video": per_video,
    }

    for finger_index, finger in enumerate(FINGER_ORDER):
        for chain_index, chain in enumerate(CHAIN_ORDER):
            values = np.concatenate([
                np.load(args.out_dir / v["sample_id"] / "hand_kinematics.npz")["flexion_deg"][
                    :, :, finger_index, chain_index
                ].reshape(-1)
                for v in per_video
            ])
            summary["per_finger_flexion"][f"{finger}_{chain}"] = _channel_summary(values)
    for pair_index, pair in enumerate(SPREAD_PAIRS):
        values = np.concatenate([
            np.load(args.out_dir / v["sample_id"] / "hand_kinematics.npz")["adjacent_spread_deg"][
                :, :, pair_index
            ].reshape(-1)
            for v in per_video
        ])
        summary["per_pair_spread"]["-".join(pair)] = _channel_summary(values)

    summary_path = args.out_dir / "kinematics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {summary_path}")
    print(
        f"videos={summary['videos']} frames={summary['frames']} "
        f"strictly_valid={total_valid}/{total_instances} ({summary['valid_pct']:.2f}%)  "
        f"palm_frame_valid={total_palm_valid}/{total_instances} "
        f"({summary['valid_palm_frame_pct']:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
