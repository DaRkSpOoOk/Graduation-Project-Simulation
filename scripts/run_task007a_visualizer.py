#!/usr/bin/env python3
"""Play one stored TASK-008 Core-28 sequence in the TASK-007A viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from visualizer import ArtifactValidationError, load_sequence
from visualizer.rendering import MatplotlibGloveViewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--frame-index", type=int, default=None, help="exact stored frame index to show initially")
    parser.add_argument("--speed", type=float, default=1.0, choices=(0.5, 1.0, 2.0))
    parser.add_argument("--save-png", type=Path, default=None, help="save the initial frame without requiring a GUI")
    parser.add_argument("--no-show", action="store_true", help="build/save the viewer and exit without opening a window")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sequence = load_sequence(args.run_root, args.sample_id, manifest_path=args.manifest)
    except (ArtifactValidationError, OSError, ValueError) as exc:
        print(f"TASK-007A artifact error: {exc}")
        return 2
    position = sequence.position_for_frame(args.frame_index) if args.frame_index is not None else 0
    viewer = MatplotlibGloveViewer(sequence, initial_position=position, speed=args.speed)
    if args.save_png is not None:
        viewer.save(args.save_png)
        print(f"saved: {args.save_png}")
    print(
        f"loaded {sequence.sample_id}: {len(sequence)} frames, "
        f"geometry={sequence.geometry_source}, label={sequence.label_ar or '(unknown)'}"
    )
    if not args.no_show:
        viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
