#!/usr/bin/env python3
"""Render human annotation markers over original RGB frames.

This helper deliberately has no pose-model or tracking dependency.  It reads
the source video and the locked human annotation CSV only, then writes JPEG
review frames to a caller-selected directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.annotations import AnnotationRow, validate_against_manifest


def _draw_label(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)


def _draw_hand(image, row: AnnotationRow, side: str, state: str) -> None:
    height, width = image.shape[:2]
    x = getattr(row, f"{side}_x")
    y = getattr(row, f"{side}_y")
    color = (50, 210, 50) if side == "left" else (255, 150, 40)
    if x is not None and y is not None:
        centre = (int(round(x * (width - 1))), int(round(y * (height - 1))))
        cv2.circle(image, centre, 13, color, 3, cv2.LINE_AA)
        _draw_label(image, f"{side.upper()} {state}", (centre[0] + 16, centre[1]), color)
    else:
        _draw_label(image, f"{side.upper()} {state}", (20, 70 if side == "left" else 105), color)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("evaluation/annotations/task004_hand_identity_visibility.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/karsl_milestone1_pilot.csv"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="directory joined with each manifest local_relative_path",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", help="render one selected clip; default renders all eight")
    args = parser.parse_args()

    rows = validate_against_manifest(args.annotations, args.manifest)
    rows_by_sample: dict[str, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        rows_by_sample[row.sample_id].append(row)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}

    selected = [args.sample_id] if args.sample_id else list(rows_by_sample)
    for sample_id in selected:
        if sample_id not in rows_by_sample:
            raise SystemExit(f"sample is not in TASK-004B annotations: {sample_id}")
        source_path = args.source_root / manifest[sample_id]["local_relative_path"]
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise SystemExit(f"could not open source video: {source_path}")
        destination = args.output_dir / sample_id
        destination.mkdir(parents=True, exist_ok=True)
        for row in sorted(rows_by_sample[sample_id], key=lambda item: item.frame_index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, row.frame_index)
            ok, image = capture.read()
            if not ok:
                raise SystemExit(f"decoder failed at {sample_id} frame {row.frame_index}")
            _draw_hand(image, row, "left", row.left_visibility)
            _draw_hand(image, row, "right", row.right_visibility)
            flags = ",".join(row.scene_flags) or "none"
            _draw_label(image, f"frame={row.frame_index} flags={flags}", (20, 35), (255, 255, 255))
            _draw_label(image, f"confidence={row.annotator_confidence}", (20, image.shape[0] - 20), (255, 255, 255))
            output_path = destination / f"frame_{row.frame_index:04d}.jpg"
            if not cv2.imwrite(str(output_path), image):
                raise SystemExit(f"could not write review frame: {output_path}")
        capture.release()
        print(f"rendered {len(rows_by_sample[sample_id])} frames: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
