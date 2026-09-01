"""Visual validation (Task 7).

Renders detector bounding boxes (always available) and, when full MANO
reconstruction ran, projected 2D joint keypoints, handedness label, frame
index, and timestamp onto the source RGB frames. Runs after/separately from
inference -- never inside the timed extraction loop (Task 3).

Full triangle-mesh overlay via pyrender was intentionally NOT used: it
requires an OpenGL/EGL or OSMesa context, which is an extra, fragile
dependency uninvolved in the actual MANO parameter/joint output we care
about for Milestone 1. 2D keypoint + bbox overlay is sufficient for visual
validation of detection/handedness/temporal behavior; see the report's
"Visual validation" section for the documented trade-off.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose.common.schema import HandPoseFrame

_COLOR_LEFT = (255, 128, 0)  # BGR
_COLOR_RIGHT = (0, 200, 0)
_COLOR_UNKNOWN = (0, 0, 255)


def _color_for(frame: HandPoseFrame) -> tuple[int, int, int]:
    if frame.handedness_label == "left":
        return _COLOR_LEFT
    if frame.handedness_label == "right":
        return _COLOR_RIGHT
    return _COLOR_UNKNOWN


def draw_overlay(frame_bgr: np.ndarray, hand_frames: list[HandPoseFrame]) -> np.ndarray:
    out = frame_bgr.copy()
    for hf in hand_frames:
        color = _color_for(hf)
        if hf.landmarks_2d:
            pts = np.array([[int(p.x), int(p.y)] for p in hf.landmarks_2d], dtype=np.int32)
            if len(pts) == 4:  # detector-only bbox corners
                cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            else:
                for p in pts:
                    cv2.circle(out, tuple(p), 2, color, -1)
        if hf.mano_references and "vertices_ref" in (hf.mano_references or {}):
            # Full-mode: 2D keypoints are reprojected by the caller before
            # this function if desired; this module only draws what's on
            # hf.landmarks_2d to keep responsibilities separate.
            pass
        label = hf.handedness_label or "unknown"
        conf = f"{hf.detection_confidence:.2f}" if hf.detection_confidence is not None else "?"
        if hf.landmarks_2d:
            anchor = (int(hf.landmarks_2d[0].x), max(0, int(hf.landmarks_2d[0].y) - 6))
            cv2.putText(out, f"{label} {conf}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(
        out,
        f"frame={hand_frames[0].frame_index if hand_frames else '?'} "
        f"t={hand_frames[0].timestamp_seconds:.3f}s" if hand_frames else "",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_video_overlay(
    video_path: Path,
    frames_by_index: dict[int, list[HandPoseFrame]],
    out_path: Path,
    fps: float,
) -> Path:
    """Writes an MP4 with per-frame overlays. Output is a generated artifact
    (git-ignored per repository policy; caller must place it under runs/ or
    outputs/)."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (width, height))

    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            hand_frames = frames_by_index.get(frame_index, [])
            writer.write(draw_overlay(frame_bgr, hand_frames))
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    return out_path
