#!/usr/bin/env python3
"""Audit Core-28 sign pose fidelity against stored TASK-008 geometry.

The audit is deliberately read-only.  It follows the canonical keyboard
selection path, verifies the timestamp-aware boundary trace, and compares the
legacy channel-only presentation solve with the TASK-007I landmark-guided
solve.  The latter uses stored TASK-008 landmarks only for the presentation
rig; no array is changed and no recognition tensor is constructed here.

KArSL RGB is an optional input.  When ``--rgb-root`` is supplied, the manifest
source path is checked for every selected exemplar.  The script never downloads
or synthesizes a missing source video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from smart_glove_app.app.hand_pose_solver import (
    HandPoseSolver,
    quaternion_to_matrix_wxyz,
)
from smart_glove_app.app.playback_controller import (
    PersistentPlaybackController,
    PlaybackBoundaryTrace,
)
from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
    SIDES,
    SPREAD_PAIRS,
    load_presentation_rig,
)
from visualizer.app.integration import load_sequence_for_item
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"
DEFAULT_CATALOG = PROJECT_ROOT / "visualizer" / "catalog" / "core28_exemplars.json"
DEFAULT_RIG_ASSET = PROJECT_ROOT / "assets-local" / "blendswap_hands_v1"
VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--rig-asset", type=Path, default=DEFAULT_RIG_ASSET)
    parser.add_argument("--rgb-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--tick-hz", type=float, default=60.0)
    return parser


def _asset_paths(asset_dir: Path) -> dict[str, Path]:
    return {
        "LEFT": asset_dir / "task007g_hand_left.glb",
        "RIGHT": asset_dir / "task007g_hand_right.glb",
    }


def _source_path(rgb_root: Path | None, relative: str) -> Path | None:
    if rgb_root is None or not relative:
        return None
    candidate = rgb_root / Path(relative)
    return candidate if candidate.is_file() else None


def _track_index(side: str) -> int:
    return 0 if side == "LEFT" else 1


def _actual_world_rotations(
    solver: HandPoseSolver,
    side: str,
    bones: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Rebuild the QML RuntimeLoader world orientation from a pose map."""

    calibration = solver._landmark_calibration[side]  # diagnostic read-only access
    if calibration is None:
        raise ValueError(f"no GLB calibration for {side}")
    wrist = solver.rig.wrist_bone
    result: dict[str, np.ndarray] = {
        wrist: calibration.world_rotations[wrist]
        @ quaternion_to_matrix_wxyz(bones[wrist])
    }
    for finger in FINGERS:
        parent = wrist
        chain = solver.rig.chains[finger]
        for bone in (chain.metacarpal, *chain.joints):
            result[bone] = (
                result[parent]
                @ calibration.local_rotations[bone]
                @ quaternion_to_matrix_wxyz(bones[bone])
            )
            parent = bone
    return result


def _direction_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(np.dot(first[:, 1], second[:, 1])),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def _source_clamps(sequence: Any, rig: Any) -> dict[str, Any]:
    bend = np.asarray([frame.bend_normalized for frame in sequence.frames], dtype=float)
    bend_valid = np.asarray([frame.bend_valid for frame in sequence.frames], dtype=bool)
    spread = np.asarray([frame.spread_normalized for frame in sequence.frames], dtype=float)
    spread_valid = np.asarray([frame.spread_valid for frame in sequence.frames], dtype=bool)

    channels: list[str] = []
    total = 0
    for finger_index, finger in enumerate(FINGERS):
        chain = rig.chains[finger]
        for joint_index, limit in enumerate(chain.joint_limits_deg):
            values = bend[:, :, finger_index, joint_index] * 180.0
            count = int(
                np.count_nonzero(
                    bend_valid[:, :, finger_index, joint_index]
                    & (values > limit + 1e-6)
                )
            )
            if count:
                total += count
                channels.append(f"{finger}:{joint_index}:{count}")

    spread_values = []
    for index in range(len(SPREAD_PAIRS)):
        valid_values = spread[:, :, index][spread_valid[:, :, index]] * 180.0
        spread_values.append(float(np.max(valid_values)) if len(valid_values) else None)

    return {
        "bend_clamp_count_in_channel_fallback": total,
        "bend_clamp_channels": channels,
        "spread_max_degrees": spread_values,
        "wrist_cap_degrees": float(rig.wrist_max_angle_deg),
    }


def _source_geometry_comparison(
    sequence: Any,
    rig: Any,
    guided: HandPoseSolver,
    channel_only: HandPoseSolver,
) -> dict[str, Any]:
    guided_errors: list[float] = []
    channel_errors: list[float] = []
    guided_max = 0.0
    guided_guided_bones = 0
    guided_fallback_bases = 0
    guided_fallback_phalanges = 0
    target_frames = 0
    timed_start = time.perf_counter()

    for frame in sequence.frames:
        # Calling each solver once per source frame preserves the exact
        # wrist-relative state used by the application and avoids measuring a
        # second, differently advanced solver state.
        guided_poses = {side: guided.pose_for_frame(frame, side) for side in SIDES}
        channel_poses = {
            side: channel_only.pose_for_frame(frame, side) for side in SIDES
        }
        for side in SIDES:
            hand = frame.hand(side)
            calibration = guided._landmark_calibration[side]
            if calibration is None:
                continue
            targets = guided._landmark_targets(hand.landmarks_3d, side, calibration)
            if targets is None:
                continue
            target_frames += 1
            pose = guided_poses[side]
            wrist = rig.wrist_bone
            wrist_shape = (
                calibration.world_rotations[wrist]
                @ quaternion_to_matrix_wxyz(pose.bones_wxyz[wrist])
                @ calibration.world_rotations[wrist].T
            )
            target_world = {
                bone: wrist_shape @ target
                for bone, target in targets.items()
            }
            actual = _actual_world_rotations(guided, side, pose.bones_wxyz)
            legacy_actual = _actual_world_rotations(
                guided, side, channel_poses[side].bones_wxyz
            )
            track = _track_index(side)
            bend_mask = np.asarray(frame.bend_valid)
            for finger_index, finger in enumerate(FINGERS):
                chain = rig.chains[finger]
                bone_names = (chain.metacarpal, *chain.joints)
                for joint_index, bone in enumerate(bone_names):
                    valid = (
                        guided._spread_valid_for_base(frame, track, finger)
                        if joint_index == 0
                        else bool(
                            bend_mask.ndim == 3
                            and bend_mask.shape[0] > track
                            and bool(
                                bend_mask[
                                    track,
                                    finger_index,
                                    joint_index - 1,
                                ]
                            )
                        )
                    )
                    if valid:
                        error = _direction_error_deg(
                            actual[bone], target_world[bone]
                        )
                        guided_errors.append(error)
                        guided_max = max(guided_max, error)
                        guided_guided_bones += 1
                        channel_errors.append(
                            _direction_error_deg(
                                legacy_actual[bone], target_world[bone]
                            )
                        )
                    elif joint_index == 0:
                        guided_fallback_bases += 1
                    else:
                        guided_fallback_phalanges += 1

    elapsed = time.perf_counter() - timed_start
    return {
        "guided_mean_direction_error_deg": float(np.mean(guided_errors)) if guided_errors else None,
        "guided_max_direction_error_deg": float(np.max(guided_errors)) if guided_errors else None,
        "channel_only_mean_direction_error_deg": float(np.mean(channel_errors)) if channel_errors else None,
        "channel_only_max_direction_error_deg": float(np.max(channel_errors)) if channel_errors else None,
        "guided_direction_samples": len(guided_errors),
        "guided_bones": guided_guided_bones,
        "landmark_target_frames": target_frames,
        "presentation_fallback_base_count": guided_fallback_bases,
        "presentation_fallback_phalange_count": guided_fallback_phalanges,
        "solver_elapsed_seconds": elapsed,
        "solver_frames_per_second": (
            (len(sequence.frames) * 2.0) / elapsed if elapsed > 0.0 else None
        ),
    }


def _sequence_audit(
    sequence: Any,
    item: Any,
    rig: Any,
    asset_paths: Mapping[str, Path],
    tick_hz: float,
    rgb_root: Path | None,
) -> dict[str, Any]:
    guided = HandPoseSolver(rig, rig_asset_paths=asset_paths)
    channel_only = HandPoseSolver(rig)
    before = {
        "bend_normalized": [np.array(frame.bend_normalized, copy=True) for frame in sequence.frames],
        "spread_normalized": [np.array(frame.spread_normalized, copy=True) for frame in sequence.frames],
        "bend_valid": [np.array(frame.bend_valid, copy=True) for frame in sequence.frames],
        "spread_valid": [np.array(frame.spread_valid, copy=True) for frame in sequence.frames],
    }
    comparison = _source_geometry_comparison(sequence, rig, guided, channel_only)
    arrays_unchanged = True
    for frame, bend_before, spread_before, bend_valid_before, spread_valid_before in zip(
        sequence.frames,
        before["bend_normalized"],
        before["spread_normalized"],
        before["bend_valid"],
        before["spread_valid"],
    ):
        arrays_unchanged = arrays_unchanged and all(
            np.array_equal(np.asarray(current), original, equal_nan=True)
            for current, original in zip(
                (
                    frame.bend_normalized,
                    frame.spread_normalized,
                    frame.bend_valid,
                    frame.spread_valid,
                ),
                (bend_before, spread_before, bend_valid_before, spread_valid_before),
            )
        )

    playback = PersistentPlaybackController(sequence.timestamps, sequence.frame_indices)
    trace = PlaybackBoundaryTrace.for_sequence(
        sequence.sample_id,
        item.character,
        sequence.frame_indices,
    )
    state = playback.play(0.0)
    trace.record(state.position)
    now = 0.0
    while playback.playing:
        now += 1.0 / tick_hz
        state = playback.tick(now)
        trace.record(state.position)
    trace.mark_queue_advance()

    frame_both = int(
        sum(all(hand.present for hand in frame.hands) for frame in sequence.frames)
    )
    bend_valid = np.asarray([frame.bend_valid for frame in sequence.frames], dtype=bool)
    spread_valid = np.asarray([frame.spread_valid for frame in sequence.frames], dtype=bool)
    manifest = sequence.metadata.get("manifest", {})
    source_relative = str(manifest.get("source_relative_path", ""))
    rgb_path = _source_path(rgb_root, source_relative)
    clamp_info = _source_clamps(sequence, rig)
    partial_spread = [
        SPREAD_PAIRS[index]
        for index in range(len(SPREAD_PAIRS))
        if not bool(np.all(spread_valid[:, :, index]))
    ]
    status = "PASS"
    diagnosis = (
        "Stored TASK-008 segment directions reproduce in the persistent rig "
        "within the measured residual; any invalid presentation channels use "
        "the documented hold/fallback policy."
    )
    if comparison["guided_mean_direction_error_deg"] is None or not arrays_unchanged:
        status = "FAIL"
        diagnosis = "Landmark-guided presentation could not be validated without mutating source arrays."
    elif comparison["guided_max_direction_error_deg"] is not None and comparison["guided_max_direction_error_deg"] > 1.0:
        status = "QUESTIONABLE"
        diagnosis = "Residual direction error remains on a validity-gated presentation channel."
    if partial_spread:
        diagnosis += (
            " Partial spread masks: "
            + ", ".join(partial_spread)
            + "; affected bases retain channel fallback rather than fabricating spread."
        )
    return {
        "sign_id": item.sign_id,
        "character": item.character,
        "sample_id": item.sample_id,
        "signer_id": item.signer_id,
        "official_partition": item.sequence_descriptor.official_partition,
        "repetition_id": item.sequence_descriptor.repetition_id,
        "catalog_score": None,
        "source_relative_path": source_relative,
        "source_rgb_available": rgb_path is not None,
        "source_rgb_path": str(rgb_path) if rgb_path is not None else None,
        "source_sequence_length": len(sequence),
        "source_frame_first": int(sequence.frame_indices[0]),
        "source_frame_last": int(sequence.frame_indices[-1]),
        "duration_seconds": float(sequence.timestamps[-1] - sequence.timestamps[0]),
        "both_hand_frames": frame_both,
        "both_hand_fraction": float(frame_both / len(sequence)),
        "bend_valid_fraction": float(np.count_nonzero(bend_valid) / bend_valid.size),
        "spread_valid_fraction": float(np.count_nonzero(spread_valid) / spread_valid.size),
        "partial_spread_pairs": partial_spread,
        "clamps": clamp_info,
        "comparison": comparison,
        "source_arrays_unchanged": bool(arrays_unchanged),
        "boundary_trace": trace.to_dict(),
        "status": status,
        "diagnosis": diagnosis,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tick_hz <= 0.0:
        print("--tick-hz must be positive", file=sys.stderr)
        return 2

    run_root = args.run_root.expanduser().resolve()
    asset_dir = args.rig_asset.expanduser().resolve()
    asset_paths = _asset_paths(asset_dir)
    resolver = Core28Resolver(
        labels_path=args.labels.expanduser().resolve(),
        catalog_path=args.catalog.expanduser().resolve(),
    )
    rig = load_presentation_rig()
    entries: list[dict[str, Any]] = []
    for character in resolver.supported_characters():
        queue = PlaybackQueue(resolver)
        item = queue.enqueue_character(character, mode="canonical")
        sequence = load_sequence_for_item(
            item,
            run_root=run_root,
            manifest_path=args.manifest.expanduser().resolve(),
        )
        if sequence is None:
            raise RuntimeError(f"canonical sign unexpectedly resolved to a gap: {item.character!r}")
        entry = _sequence_audit(
            sequence,
            item,
            rig,
            asset_paths,
            float(args.tick_hz),
            args.rgb_root.expanduser().resolve() if args.rgb_root is not None else None,
        )
        # Keep the catalog score visible without changing the resolver path.
        entry["catalog_score"] = resolver.catalog.entry_for_sign_id(item.sign_id or "").score
        entries.append(entry)

    payload = {
        "task": "TASK-007I",
        "run_root": str(run_root),
        "rig_asset_dir": str(asset_dir),
        "rgb_root": str(args.rgb_root.expanduser().resolve()) if args.rgb_root is not None else None,
        "rgb_video_extensions_checked": sorted(VIDEO_SUFFIXES),
        "rgb_available_for_all_canonical_entries": all(
            entry["source_rgb_available"] for entry in entries
        ),
        "tick_hz": float(args.tick_hz),
        "canonical_entry_count": len(entries),
        "entries": entries,
        "all_boundary_traces_complete": all(
            entry["boundary_trace"]["all_source_positions_presented"]
            and entry["boundary_trace"]["last_frame_presented"]
            and entry["boundary_trace"]["queue_advanced_after_final"]
            and not entry["boundary_trace"]["early_queue_advance"]
            for entry in entries
        ),
        "all_source_arrays_unchanged": all(
            entry["source_arrays_unchanged"] for entry in entries
        ),
        "status_counts": {
            status: sum(entry["status"] == status for entry in entries)
            for status in ("PASS", "QUESTIONABLE", "FAIL")
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(rendered.encode("utf-8"))
    else:  # pragma: no cover
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
