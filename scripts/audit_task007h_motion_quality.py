#!/usr/bin/env python3
"""Audit canonical Core-28 playback boundaries and presentation pose quality.

This is a read-only audit of existing TASK-008 artifacts.  It simulates the
same timestamp-aware playback clock used by the Qt application and reports
presentation clamp observations without writing a dataset or recognition
artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from smart_glove_app.app.hand_pose_solver import HandPoseSolver, quaternion_angle_deg
from smart_glove_app.app.playback_controller import (
    PersistentPlaybackController,
    PlaybackBoundaryTrace,
)
from smart_glove_app.rendering.presentation_rig import (
    FINGERS,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--tick-hz", type=float, default=60.0)
    return parser


def _sequence_audit(
    sequence: Any,
    item: Any,
    rig: Any,
    tick_hz: float,
    selection_score: float | None,
) -> dict[str, Any]:
    bend = np.asarray([frame.bend_normalized for frame in sequence.frames], dtype=float)
    bend_valid = np.asarray([frame.bend_valid for frame in sequence.frames], dtype=bool)
    spread = np.asarray(
        [frame.spread_normalized for frame in sequence.frames], dtype=float
    )
    spread_valid = np.asarray(
        [frame.spread_valid for frame in sequence.frames], dtype=bool
    )
    timestamps = np.asarray(sequence.timestamps, dtype=float)

    solver = HandPoseSolver(rig)
    poses = []
    for frame in sequence.frames:
        poses.append(solver.frame_pose(frame))
    wrist_max = max(
        quaternion_angle_deg(poses[position][side].bones_wxyz[rig.wrist_bone])
        for position in range(len(poses))
        for side in ("LEFT", "RIGHT")
    )

    bend_clamp_count = 0
    bend_clamp_channels: list[str] = []
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
                bend_clamp_count += count
                bend_clamp_channels.append(f"{finger}:{joint_index}:{count}")

    spread_max_deg = [
        float(
            np.nanmax(np.where(spread_valid[:, :, index], spread[:, :, index], np.nan))
            * 180.0
        )
        for index in range(len(SPREAD_PAIRS))
    ]

    playback = PersistentPlaybackController(sequence.timestamps, sequence.frame_indices)
    trace = PlaybackBoundaryTrace.for_sequence(
        sequence.sample_id, item.character, sequence.frame_indices
    )
    state = playback.play(0.0)
    trace.record(state.position)
    now = 0.0
    while playback.playing:
        now += 1.0 / tick_hz
        state = playback.tick(now)
        trace.record(state.position)
    trace.mark_queue_advance()

    return {
        "sign_id": item.sign_id,
        "character": item.character,
        "sample_id": item.sample_id,
        "signer_id": item.signer_id,
        "selection_score": selection_score,
        "source_sequence_length": len(sequence),
        "source_frame_first": int(sequence.frame_indices[0]),
        "source_frame_last": int(sequence.frame_indices[-1]),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "both_hand_frames": int(
            sum(all(hand.present for hand in frame.hands) for frame in sequence.frames)
        ),
        "bend_valid_fraction": float(np.count_nonzero(bend_valid) / bend_valid.size),
        "spread_valid_fraction": float(
            np.count_nonzero(spread_valid) / spread_valid.size
        ),
        "wrist_max_degrees": float(wrist_max),
        "bend_clamp_count": bend_clamp_count,
        "bend_clamp_channels": bend_clamp_channels,
        "spread_max_degrees": spread_max_deg,
        "source_indices_contiguous": list(sequence.frame_indices)
        == list(range(len(sequence))),
        "boundary_trace": trace.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tick_hz <= 0.0:
        print("--tick-hz must be positive", file=sys.stderr)
        return 2
    run_root = args.run_root.expanduser().resolve()
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
            raise RuntimeError(
                f"canonical sign unexpectedly resolved to a gap: {item.character!r}"
            )
        entry = resolver.catalog.entry_for_sign_id(item.sign_id or "")
        entries.append(
            _sequence_audit(sequence, item, rig, float(args.tick_hz), entry.score)
        )

    payload = {
        "task": "TASK-007H",
        "run_root": str(run_root),
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
