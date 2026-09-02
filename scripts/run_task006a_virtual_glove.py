#!/usr/bin/env python3
"""Derive the ideal virtual smart-glove signals over the KArSL pilot.

Reads the validated TASK-005F kinematics run by explicit path, converts every
sample into the 19-Hall + 1-IMU ideal glove representation, and writes the
stage plus a diagnostics summary to an ignored output directory. The
kinematics input is opened read-only and re-hashed after processing.

Example:

    python scripts/run_task006a_virtual_glove.py \\
      --kinematics-run /home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f \\
      --out-dir /home/hatim/graduation-project-runs/virtual_glove_task006a \\
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

from virtual_glove import (  # noqa: E402
    ADC_INVALID_SENTINEL,
    CHAIN_ORDER,
    EXPECTED_HALL_SENSORS,
    EXPECTED_SENSOR_PACKAGES,
    FINGER_ORDER,
    SPREAD_PAIRS,
    TRACK_ORDER,
    build_metadata,
    extract_glove_sequence,
    layout_document,
    save_glove_sequence,
    sha256_file,
)
from virtual_glove.layout import BEND_SENSOR_IDS, SPREAD_SENSOR_IDS  # noqa: E402

EXPECTED_SAMPLES = 18
EXPECTED_FRAMES = 894
KINEMATICS_NPZ = "hand_kinematics.npz"
KINEMATICS_META = "hand_kinematics_meta.json"


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


def _summary(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_milestone1_pilot.csv")
    parser.add_argument(
        "--kinematics-run",
        type=Path,
        default=Path("/home/hatim/graduation-project-runs/wilor_karsl_pilot_kinematics_task005f"),
        help="validated TASK-005F kinematics run (read-only)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="ignored output directory")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--strict-counts", action="store_true")
    args = parser.parse_args()

    sample_ids = args.sample_ids or _sample_ids(args.manifest)
    commit = _commit()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_video: list[dict] = []
    total_frames = 0
    bend_all: list[np.ndarray] = []
    spread_all: list[np.ndarray] = []
    bend_norm_all: list[np.ndarray] = []
    spread_norm_all: list[np.ndarray] = []
    bend_adc_all: list[np.ndarray] = []
    spread_adc_all: list[np.ndarray] = []
    omega_all: list[np.ndarray] = []
    per_track: dict[str, dict[str, list[np.ndarray]]] = {
        track: {"bend": [], "spread": []} for track in TRACK_ORDER
    }
    bend_valid_total = spread_valid_total = imu_valid_total = 0
    bend_total = spread_total = imu_total = 0
    quat_norm_error = 0.0
    orth_error = 0.0
    det_error = 0.0
    quat_matches = 0
    violations: list[dict] = []
    state_rows: dict[str, dict[str, int]] = {}
    kept_when_strict_false = 0

    for sample_id in sample_ids:
        kinematics_dir = args.kinematics_run / sample_id
        npz_path = kinematics_dir / KINEMATICS_NPZ
        if not npz_path.is_file():
            raise SystemExit(f"Kinematics input missing: {npz_path}")
        sha_before = sha256_file(npz_path)

        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        kinematics_meta = json.loads((kinematics_dir / KINEMATICS_META).read_text())

        sequence = extract_glove_sequence(arrays, kinematics_meta, sample_id)

        sha_after = sha256_file(npz_path)
        if sha_before != sha_after:
            raise SystemExit(f"Kinematics input mutated during processing: {npz_path}")

        metadata = build_metadata(
            sequence,
            kinematics_dir=kinematics_dir,
            kinematics_sha256=sha_before,
            kinematics_metadata=kinematics_meta,
            implementation_commit=commit,
        )
        save_glove_sequence(args.out_dir / sample_id, sequence, metadata)
        violations.extend(sequence.contract_violations)

        frames = int(sequence.frame_index.shape[0])
        total_frames += frames

        bend_all.append(sequence.bend_angle_deg.reshape(-1))
        spread_all.append(sequence.spread_angle_deg.reshape(-1))
        bend_norm_all.append(sequence.bend_normalized.reshape(-1))
        spread_norm_all.append(sequence.spread_normalized.reshape(-1))
        bend_adc_all.append(sequence.bend_adc_12bit.reshape(-1))
        spread_adc_all.append(sequence.spread_adc_12bit.reshape(-1))
        omega_all.append(
            np.linalg.norm(sequence.imu_angular_velocity_rad_s, axis=-1)[
                sequence.imu_angular_velocity_valid
            ]
        )
        for index, track in enumerate(TRACK_ORDER):
            per_track[track]["bend"].append(sequence.bend_angle_deg[:, index].reshape(-1))
            per_track[track]["spread"].append(sequence.spread_angle_deg[:, index].reshape(-1))

        bend_valid_total += int(sequence.bend_valid.sum())
        bend_total += int(sequence.bend_valid.size)
        spread_valid_total += int(sequence.spread_valid.sum())
        spread_total += int(sequence.spread_valid.size)
        imu_valid_total += int(sequence.palm_imu_valid.sum())
        imu_total += int(sequence.palm_imu_valid.size)

        # Model-B evidence: sensors retained where the strict flag is false.
        strict_false = ~sequence.source_valid_kinematics
        kept_when_strict_false += int(
            (sequence.bend_valid[strict_false].sum() + sequence.spread_valid[strict_false].sum())
        )

        # IMU orientation integrity, over the valid instances only.
        rotation = sequence.imu_rotation_matrix.astype(np.float64)
        quaternion = sequence.imu_quaternion_wxyz.astype(np.float64)
        for row in range(frames):
            for track in range(len(TRACK_ORDER)):
                if not sequence.palm_imu_valid[row, track]:
                    continue
                matrix = rotation[row, track]
                quat_norm_error = max(
                    quat_norm_error, abs(float(np.linalg.norm(quaternion[row, track])) - 1.0)
                )
                orth_error = max(
                    orth_error, float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
                )
                det_error = max(det_error, abs(float(np.linalg.det(matrix)) - 1.0))
                # verbatim copy check against the frozen source
                if np.array_equal(
                    sequence.imu_quaternion_wxyz[row, track],
                    arrays["palm_quaternion_wxyz"][row, track],
                ) and np.array_equal(
                    sequence.imu_rotation_matrix[row, track],
                    arrays["palm_rotation_matrix"][row, track],
                ):
                    quat_matches += 1

        codes = sequence.tracking_state_code
        for row in range(frames):
            for track in range(len(TRACK_ORDER)):
                bucket = state_rows.setdefault(
                    str(int(codes[row, track])),
                    {"bend_valid": 0, "bend_total": 0, "spread_valid": 0,
                     "spread_total": 0, "imu_valid": 0, "imu_total": 0},
                )
                bucket["bend_valid"] += int(sequence.bend_valid[row, track].sum())
                bucket["bend_total"] += int(sequence.bend_valid[row, track].size)
                bucket["spread_valid"] += int(sequence.spread_valid[row, track].sum())
                bucket["spread_total"] += int(sequence.spread_valid[row, track].size)
                bucket["imu_valid"] += int(sequence.palm_imu_valid[row, track])
                bucket["imu_total"] += 1

        per_video.append({
            "sample_id": sample_id,
            "frames": frames,
            "bend_valid": int(sequence.bend_valid.sum()),
            "bend_total": int(sequence.bend_valid.size),
            "spread_valid": int(sequence.spread_valid.sum()),
            "spread_total": int(sequence.spread_valid.size),
            "palm_imu_valid": int(sequence.palm_imu_valid.sum()),
            "palm_imu_total": int(sequence.palm_imu_valid.size),
            "angular_velocity_valid": int(sequence.imu_angular_velocity_valid.sum()),
            "kinematics_npz_sha256": sha_before,
        })
        print(
            f"[{sample_id}] frames={frames} bend={int(sequence.bend_valid.sum())}/"
            f"{sequence.bend_valid.size} spread={int(sequence.spread_valid.sum())}/"
            f"{sequence.spread_valid.size} imu={int(sequence.palm_imu_valid.sum())}/"
            f"{sequence.palm_imu_valid.size} gyro={int(sequence.imu_angular_velocity_valid.sum())}"
        )

    if args.strict_counts and (len(per_video) != EXPECTED_SAMPLES or total_frames != EXPECTED_FRAMES):
        raise SystemExit(
            f"Strict count check failed: {len(per_video)} videos / {total_frames} frames "
            f"(expected {EXPECTED_SAMPLES} / {EXPECTED_FRAMES})"
        )

    bend_flat = np.concatenate(bend_all)
    spread_flat = np.concatenate(spread_all)
    bend_norm_flat = np.concatenate(bend_norm_all)
    spread_norm_flat = np.concatenate(spread_norm_all)
    bend_adc_flat = np.concatenate(bend_adc_all)
    spread_adc_flat = np.concatenate(spread_adc_all)
    omega_flat = np.concatenate([o for o in omega_all if o.size]) if omega_all else np.empty(0)

    per_sensor: dict[str, dict] = {}
    for finger_index, finger in enumerate(FINGER_ORDER):
        for chain_index, chain in enumerate(CHAIN_ORDER):
            sensor_id = f"H_{finger.upper()}_{chain.upper()}"
            values = np.concatenate([
                np.load(args.out_dir / v["sample_id"] / "virtual_glove.npz")["bend_angle_deg"][
                    :, :, finger_index, chain_index
                ].reshape(-1)
                for v in per_video
            ])
            entry = _summary(values)
            entry["normalized_max"] = None if entry["max"] is None else entry["max"] / 180.0
            per_sensor[sensor_id] = entry
    for pair_index, pair in enumerate(SPREAD_PAIRS):
        sensor_id = f"H_SPREAD_{pair[0].upper()}_{pair[1].upper()}"
        values = np.concatenate([
            np.load(args.out_dir / v["sample_id"] / "virtual_glove.npz")["spread_angle_deg"][
                :, :, pair_index
            ].reshape(-1)
            for v in per_video
        ])
        entry = _summary(values)
        entry["normalized_max"] = None if entry["max"] is None else entry["max"] / 180.0
        per_sensor[sensor_id] = entry

    valid_adc_bend = bend_adc_flat[bend_adc_flat != ADC_INVALID_SENTINEL]
    valid_adc_spread = spread_adc_flat[spread_adc_flat != ADC_INVALID_SENTINEL]

    summary = {
        "task": "TASK-006A",
        "implementation_commit": commit,
        "layout_version": layout_document()["layout_version"],
        "input_contract": "TASK-005-final-v2",
        "kinematics_run": str(args.kinematics_run),
        "out_dir": str(args.out_dir),
        "videos": len(per_video),
        "frames": total_frames,
        "sensor_counts_per_hand": {
            "hall_sensors_total": EXPECTED_HALL_SENSORS,
            "logical_sensing_packages": EXPECTED_SENSOR_PACKAGES,
        },
        "validity": {
            "bend_valid": bend_valid_total, "bend_total": bend_total,
            "bend_valid_pct": 100.0 * bend_valid_total / bend_total,
            "spread_valid": spread_valid_total, "spread_total": spread_total,
            "spread_valid_pct": 100.0 * spread_valid_total / spread_total,
            "palm_imu_valid": imu_valid_total, "palm_imu_total": imu_total,
            "palm_imu_valid_pct": 100.0 * imu_valid_total / imu_total,
            "sensors_retained_where_strict_valid_kinematics_false": kept_when_strict_false,
        },
        "validity_by_tracking_state": state_rows,
        "ranges": {
            "bend_angle_deg": _summary(bend_flat),
            "spread_angle_deg": _summary(spread_flat),
            "bend_normalized": _summary(bend_norm_flat),
            "spread_normalized": _summary(spread_norm_flat),
        },
        "normalized_within_unit_interval": bool(
            np.all((bend_norm_flat[np.isfinite(bend_norm_flat)] >= 0.0)
                   & (bend_norm_flat[np.isfinite(bend_norm_flat)] <= 1.0))
            and np.all((spread_norm_flat[np.isfinite(spread_norm_flat)] >= 0.0)
                       & (spread_norm_flat[np.isfinite(spread_norm_flat)] <= 1.0))
        ),
        "optional_adc_12bit": {
            "bend": {"min": int(valid_adc_bend.min()), "max": int(valid_adc_bend.max()),
                     "count": int(valid_adc_bend.size)},
            "spread": {"min": int(valid_adc_spread.min()), "max": int(valid_adc_spread.max()),
                       "count": int(valid_adc_spread.size)},
            "invalid_sentinel": ADC_INVALID_SENTINEL,
            "invalid_count": int((bend_adc_flat == ADC_INVALID_SENTINEL).sum()
                                 + (spread_adc_flat == ADC_INVALID_SENTINEL).sum()),
        },
        "imu_orientation_integrity": {
            "quaternion_norm_error_max": quat_norm_error,
            "rotation_orthogonality_error_max": orth_error,
            "rotation_determinant_error_max": det_error,
            "verbatim_copies_confirmed": quat_matches,
            "valid_instances": imu_valid_total,
        },
        "derived_angular_velocity": {
            "status": "DERIVED",
            "units": "rad/s",
            "valid_samples": int(omega_flat.size),
            "possible_pairs": total_frames * len(TRACK_ORDER),
            "magnitude": _summary(omega_flat),
            "p50": float(np.percentile(omega_flat, 50)) if omega_flat.size else None,
            "p95": float(np.percentile(omega_flat, 95)) if omega_flat.size else None,
            "p99": float(np.percentile(omega_flat, 99)) if omega_flat.size else None,
            "smoothing": "none",
        },
        "accelerometer": "DEFER ACCELEROMETER",
        "per_track": {
            track: {
                "bend_angle_deg": _summary(np.concatenate(per_track[track]["bend"])),
                "spread_angle_deg": _summary(np.concatenate(per_track[track]["spread"])),
            }
            for track in TRACK_ORDER
        },
        "per_sensor": per_sensor,
        "contract_violations": violations,
        "contract_violation_count": len(violations),
        "per_video": per_video,
    }

    summary_path = args.out_dir / "virtual_glove_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "sensor_layout.json").write_text(
        json.dumps(layout_document(), indent=2, sort_keys=True) + "\n"
    )
    print(f"\nWrote {summary_path}")
    print(
        f"videos={summary['videos']} frames={summary['frames']} "
        f"bend={bend_valid_total}/{bend_total} spread={spread_valid_total}/{spread_total} "
        f"imu={imu_valid_total}/{imu_total} gyro={int(omega_flat.size)} "
        f"violations={len(violations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
