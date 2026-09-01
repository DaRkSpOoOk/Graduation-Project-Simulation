#!/usr/bin/env python3
"""Run validation, raw MediaPipe extraction, overlays, and baseline metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.metrics.mediapipe_baseline import aggregate_metrics, evaluate_npz
from pose.mediapipe.extractor import MediaPipeConfig, extract_video
from pose.mediapipe.hardware import collect_hardware_info
from video_io.reader import inspect_video


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_csv_row(manifest_row: dict[str, str], inspection: dict[str, Any], metric: dict[str, Any], extraction: dict[str, Any]) -> dict[str, Any]:
    instability = metric.get("potential_identity_instability") or {}
    jitter = metric.get("temporal_jitter_world") or {}
    jitter_second = jitter.get("second_difference_m_per_frame") or {}
    bones = metric.get("bone_length_variation") or {}
    wrist = metric.get("wrist_coordinate_jumps_world_m") or {}
    detector_wrist = metric.get("detector_order_wrist_coordinate_jumps_world_m") or {}
    return {
        "sample_id": manifest_row.get("sample_id"),
        "sign_id": manifest_row.get("sign_id"),
        "english_label": manifest_row.get("english_label"),
        "signer_id": manifest_row.get("signer_id"),
        "source_member": manifest_row.get("source_archive_member"),
        "local_relative_path": manifest_row.get("local_relative_path"),
        "decoder_success": inspection.get("decoder_success"),
        "reported_frame_count": inspection.get("reported_frame_count"),
        "decoded_frame_count": inspection.get("decoded_frame_count"),
        "fps": inspection.get("fps"),
        "width": inspection.get("width"),
        "height": inspection.get("height"),
        "duration_seconds": inspection.get("duration_seconds"),
        "timestamp_source": inspection.get("timestamp_source"),
        "total_frames": metric.get("total_frames"),
        "frames_with_no_hands": metric.get("frames_with_no_hands"),
        "frames_with_at_least_one_hand": metric.get("frames_with_at_least_one_hand"),
        "frames_with_left_hand": metric.get("frames_with_left_hand"),
        "frames_with_right_hand": metric.get("frames_with_right_hand"),
        "frames_with_both_hands": metric.get("frames_with_both_hands"),
        "missing_frame_percentage": metric.get("missing_frame_percentage"),
        "left_hand_detection_rate": metric.get("left_hand_detection_rate"),
        "right_hand_detection_rate": metric.get("right_hand_detection_rate"),
        "both_hand_detection_rate": metric.get("both_hand_detection_rate"),
        "longest_missing_streak_left_frames": metric.get("longest_missing_streak_left_frames"),
        "longest_missing_streak_right_frames": metric.get("longest_missing_streak_right_frames"),
        "handedness_label_set_changes": metric.get("handedness_label_set_changes"),
        "detector_order_handedness_changes": metric.get("detector_order_handedness_changes"),
        "duplicate_handedness_frames": metric.get("duplicate_handedness_frames"),
        "potential_identity_instability_events": instability.get("heuristic_event_count"),
        "wrist_jump_p95_world_m": wrist.get("p95"),
        "wrist_jump_max_world_m": wrist.get("max"),
        "wrist_jump_count_above_0_10m": wrist.get("count_above_diagnostic_threshold"),
        "detector_order_wrist_jump_p95_world_m": detector_wrist.get("p95"),
        "detector_order_wrist_jump_max_world_m": detector_wrist.get("max"),
        "detector_order_wrist_jump_count_above_0_10m": detector_wrist.get("count_above_diagnostic_threshold"),
        "temporal_jitter_mean_m_per_frame": jitter_second.get("mean"),
        "temporal_jitter_p95_m_per_frame": jitter_second.get("p95"),
        "temporal_acceleration_mean_m_per_s2": (jitter.get("acceleration_m_per_s2") or {}).get("mean"),
        "bone_length_cv_mean": bones.get("mean_coefficient_of_variation"),
        "bone_length_cv_max": bones.get("max_coefficient_of_variation"),
        "handedness_confidence_mean": (metric.get("handedness_confidence") or {}).get("mean"),
        "runtime_seconds": extraction.get("runtime_seconds"),
        "inference_seconds": extraction.get("inference_seconds"),
        "effective_processing_fps": extraction.get("effective_processing_fps"),
        "inference_fps": extraction.get("average_inference_fps"),
        "raw_npz_path": extraction.get("raw_npz_path"),
        "overlay_path": extraction.get("overlay_path"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _resolve(Path.cwd(), args.repository_root).resolve()
    manifest_path = _resolve(root, args.manifest).resolve()
    config_path = _resolve(root, args.config).resolve()
    output_dir = _resolve(root, args.output_dir).resolve()
    model_path = _resolve(root, args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe task model not found: {model_path}. Run scripts/download_mediapipe_model.py first.")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    delegate = args.delegate or config_data.get("delegate", "CPU")
    mp_config = MediaPipeConfig(
        model_path=model_path,
        num_hands=int(config_data.get("num_hands", 2)),
        min_hand_detection_confidence=float(config_data.get("min_hand_detection_confidence", 0.5)),
        min_hand_presence_confidence=float(config_data.get("min_hand_presence_confidence", 0.5)),
        min_tracking_confidence=float(config_data.get("min_tracking_confidence", 0.5)),
        delegate=delegate,
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    raw_dir = output_dir / "raw_pose"
    overlay_dir = output_dir / "overlays"
    inspection_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for manifest_row in manifest_rows:
        sample_id = manifest_row["sample_id"]
        video_path = _resolve(root, manifest_row["local_relative_path"]).resolve()
        inspection = inspect_video(video_path)
        inspection_dict = {"sample_id": sample_id, **inspection.to_dict()}
        if video_path.is_file() and manifest_row.get("checksum_sha256", "").strip():
            actual_checksum = _sha256(video_path)
            inspection_dict["checksum_matches_manifest"] = actual_checksum == manifest_row["checksum_sha256"].strip()
            inspection_dict["actual_checksum_sha256"] = actual_checksum
        else:
            inspection_dict["checksum_matches_manifest"] = None
            inspection_dict["actual_checksum_sha256"] = None
        inspection_rows.append(inspection_dict)
        if not inspection.decoder_success:
            extraction_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "skipped_decoder_failure",
                    "raw_npz_path": str(raw_dir / f"{sample_id}.npz"),
                    "overlay_path": str(overlay_dir / f"{sample_id}.mp4"),
                    "frame_count": inspection.decoded_frame_count,
                    "error": inspection.error,
                }
            )
            continue
        extraction = extract_video(
            video_path=video_path,
            raw_npz_path=raw_dir / f"{sample_id}.npz",
            overlay_path=overlay_dir / f"{sample_id}.mp4",
            config=mp_config,
            inspection=inspection,
            sample_id=sample_id,
            source_metadata=manifest_row,
        )
        extraction_dict = extraction.to_dict()
        extraction_rows.append(extraction_dict)
        if extraction.status == "success":
            metric = evaluate_npz(
                extraction.raw_npz_path,
                runtime_seconds=extraction.runtime_seconds,
                inference_seconds=extraction.inference_seconds,
            )
            metric["sample_id"] = sample_id
            metric["sign_id"] = manifest_row.get("sign_id")
            metric["english_label"] = manifest_row.get("english_label")
            metric["signer_id"] = manifest_row.get("signer_id")
            metric_records.append(metric)
            per_video_rows.append(_metric_csv_row(manifest_row, inspection_dict, metric, extraction_dict))

    aggregate = aggregate_metrics(metric_records, failed_videos=len(manifest_rows) - len(metric_records))
    run_seconds = time.perf_counter() - run_started
    hardware = collect_hardware_info(delegate)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "video_inspection.json", inspection_rows)
    _write_csv(output_dir / "video_inspection.csv", inspection_rows)
    _write_json(output_dir / "extraction_results.json", extraction_rows)
    _write_json(output_dir / "evaluation_per_video.json", metric_records)
    _write_csv(output_dir / "evaluation_per_video.csv", per_video_rows)
    _write_json(output_dir / "evaluation_aggregate.json", aggregate)
    _write_json(output_dir / "hardware.json", hardware)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "run_name": config_data.get("name"),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "config": config_data,
            "delegate_requested": delegate.upper(),
            "videos_requested": len(manifest_rows),
            "videos_successfully_processed": len(metric_records),
            "run_wall_seconds": run_seconds,
            "argv": sys.argv,
        },
    )
    return {
        "videos_requested": len(manifest_rows),
        "videos_successfully_processed": len(metric_records),
        "videos_failed_or_skipped": len(manifest_rows) - len(metric_records),
        "aggregate": aggregate,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/karsl_milestone1_pilot.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/pose/mediapipe_karsl_pilot.json"))
    parser.add_argument("--model", type=Path, default=Path("datasets/raw/models/hand_landmarker.task"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mediapipe_karsl_pilot"))
    parser.add_argument("--delegate", choices=["CPU", "GPU"], default=None)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:  # CLI boundary.
        print(f"MediaPipe pilot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
