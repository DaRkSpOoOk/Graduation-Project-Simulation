#!/usr/bin/env python3
"""Debug overlays for the derived WiLoR tracked representation.

Draws, per frame: the LEFT/RIGHT track skeletons (projected from the tracked
3D joints), the track state, the raw detector label the row came from, the
raw detection index and any quality warning. It is a diagnostic aid for
manual inspection, not the project GUI, and its output is ignored by Git.

No model is run and no metric is computed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
TRACK_COLORS = {"left": (255, 128, 0), "right": (0, 200, 0)}
STATE_COLORS = {
    "OBSERVED": (255, 255, 255),
    "AMBIGUOUS": (0, 215, 255),
    "MISSING": (120, 120, 120),
    "LIKELY_OCCLUDED": (0, 165, 255),
    "REJECTED_QUALITY": (0, 0, 255),
}


def project(joints: np.ndarray, translation: np.ndarray, focal: float, size_wh) -> np.ndarray:
    intrinsics = np.eye(3)
    intrinsics[0, 0] = intrinsics[1, 1] = focal
    intrinsics[0, 2] = size_wh[0] / 2.0
    intrinsics[1, 2] = size_wh[1] / 2.0
    points = joints + translation
    points = points / points[..., -1:]
    return (intrinsics @ points.T).T[..., :2]


def render(sample_id: str, video: Path, tracked_dir: Path, out_path: Path, focal: float) -> Path:
    with np.load(tracked_dir / "wilor_tracked.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads((tracked_dir / "wilor_tracked_meta.json").read_text())
    code_to_state = {code: name for name, code in meta["state_codes"].items()}
    tracks = meta["track_order"]

    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_rows = {int(value): row for row, value in enumerate(arrays["frame_index"])}
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            row = frame_rows.get(index)
            if row is not None:
                flags = json.loads(str(arrays["tracking_flags_json"][row]))
                for column, track in enumerate(tracks):
                    state = code_to_state[int(arrays["state_code"][row, column])]
                    colour = TRACK_COLORS[track]
                    joints = arrays["landmarks_3d"][row, column]
                    label_code = int(arrays["detector_label_code"][row, column])
                    detector_label = {0: "left", 1: "right"}.get(label_code, "-")
                    raw_index = int(arrays["raw_detection_index"][row, column])
                    text = f"{track.upper()} [{state}] raw#{raw_index} det={detector_label}"
                    if np.isfinite(joints).all():
                        points = project(
                            joints.astype(np.float64),
                            arrays["camera_translation"][row, column].astype(np.float64),
                            focal,
                            (width, height),
                        )
                        for start, end in BONES:
                            cv2.line(image, tuple(points[start].astype(int)),
                                     tuple(points[end].astype(int)), colour, 2, cv2.LINE_AA)
                        for point in points:
                            cv2.circle(image, tuple(point.astype(int)), 3, colour, -1)
                        anchor = (int(points[0][0]), max(0, int(points[0][1]) - 12))
                        cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
                    cv2.putText(image, text, (10, 40 + column * 34), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, STATE_COLORS.get(state, (255, 255, 255)), 2, cv2.LINE_AA)
                banner = f"{sample_id}  frame={index}  raw_det={int(arrays['number_of_raw_detections'][row])}"
                if flags:
                    banner += "  " + ",".join(flags)
                cv2.putText(image, banner, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
            index += 1
    finally:
        capture.release()
        writer.release()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--tracked-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--focal", type=float, default=37500.0)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}

    for sample_id in args.sample_ids:
        video = args.video_root / rows[sample_id]["local_relative_path"]
        output = render(
            sample_id,
            video,
            args.tracked_run / sample_id,
            args.out_dir / f"{sample_id}_tracked.mp4",
            args.focal,
        )
        print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
