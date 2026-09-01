#!/usr/bin/env python3
"""Render side-by-side MediaPipe / WiLoR frames for the TASK-003B visual review.

This is a *visualization* utility only. It computes no comparison metric and
does not touch the neutral fairness layer in ``evaluation/comparison``. It
reads the already-validated frozen artifacts and writes JPEG frames into an
ignored run directory so a human (or this task's manual qualitative review)
can inspect the same source frame under both extractors.

MediaPipe frames come from the existing rendered overlay MP4s. WiLoR frames
are drawn from the validated raw NPZ by projecting the reconstructed 21
joints with the same pinhole model the frozen WiLoR adapter uses
(``pose/wilor/geometry.py: project_points_full_img``), so clips without a
pre-rendered WiLoR MP4 can still be reviewed. No model is re-run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# OpenPose/MediaPipe 21-point hand skeleton (drawing only, not a metric).
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
COLOR_LEFT = (255, 128, 0)
COLOR_RIGHT = (0, 200, 0)


def project_points(points_3d: np.ndarray, cam_t: np.ndarray, focal: float, img_wh) -> np.ndarray:
    """Pinhole projection matching the frozen WiLoR visualization path."""
    k = np.eye(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = img_wh[0] / 2.0
    k[1, 2] = img_wh[1] / 2.0
    pts = points_3d + cam_t
    pts = pts / pts[..., -1:]
    return (k @ pts.T).T[..., :2]


def wilor_frame_overlay(video_path: Path, npz_path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    data = np.load(npz_path, allow_pickle=False)
    rows = defaultdict(list)
    for i in range(len(data["frame_index"])):
        if int(data["frame_index"][i]) != frame_index or not bool(data["hand_present"][i]):
            continue
        refs_raw = str(data["mano_references_json"][i])
        if not refs_raw:
            continue
        rows[frame_index].append((data["landmarks_3d"][i], json.loads(refs_raw), str(data["handedness_label"][i])))

    for joints, refs, label in rows.get(frame_index, []):
        cam_t = np.asarray(refs["camera_translation_xyz"], dtype=np.float64)
        pts = project_points(np.asarray(joints, dtype=np.float64), cam_t, float(refs["focal_length"]), refs["img_size_wh"])
        color = COLOR_LEFT if label == "left" else COLOR_RIGHT
        for a, b in BONES:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 3, color, -1)
        cv2.putText(frame, label, (int(pts[0][0]), max(0, int(pts[0][1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"WiLoR  frame={frame_index}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def mediapipe_frame(overlay_mp4: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(overlay_mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.putText(frame, f"MediaPipe  frame={frame_index}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--frames", required=True, help="comma-separated frame indices")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--mediapipe-run", required=True, type=Path)
    parser.add_argument("--wilor-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    npz = args.wilor_run / "raw" / args.sample_id / "wilor_raw.npz"
    overlay = args.mediapipe_run / "overlays" / f"{args.sample_id}.mp4"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in [int(v) for v in args.frames.split(",")]:
        mp_img = mediapipe_frame(overlay, frame_index)
        wl_img = wilor_frame_overlay(args.video, npz, frame_index)
        if mp_img is None or wl_img is None:
            print(f"skip frame {frame_index}: missing source")
            continue
        stacked = np.vstack([mp_img, wl_img])
        stacked = cv2.resize(stacked, None, fx=args.scale, fy=args.scale)
        out = args.out_dir / f"{args.sample_id}_f{frame_index:04d}.jpg"
        cv2.imwrite(str(out), stacked)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
