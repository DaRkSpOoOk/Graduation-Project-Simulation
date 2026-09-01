"""Visual validation (Task 7).

Renders detector bounding boxes (detector_only mode) or a real projected
21-joint MANO skeleton (full mode, from actual reconstructed
``landmarks_3d`` + ``mano_references['camera_translation_xyz']`` -- not
placeholder points), plus handedness label, confidence, frame index, and
timestamp. Runs after/separately from inference -- never inside the timed
extraction loop (Task 3).

Full triangle-mesh overlay via pyrender was intentionally NOT used: it
requires an OpenGL/EGL or OSMesa context, an extra fragile dependency, for a
visual that doesn't add reconstruction information beyond the skeleton
already drawn here from the same MANO output. Mesh *files* (.obj, via
`trimesh`, no OpenGL needed) are exported separately by
`export_mesh_obj` for anyone who wants to inspect the actual triangulated
surface offline. See the report's "Visual validation" section.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose.common.schema import HandPoseFrame

from .geometry import project_points_full_img

_COLOR_LEFT = (255, 128, 0)  # BGR
_COLOR_RIGHT = (0, 200, 0)
_COLOR_UNKNOWN = (0, 0, 255)

# Standard OpenPose 21-point hand skeleton connectivity (same topology as
# evaluation/metrics/hand_pose_metrics.py:HAND_BONES; duplicated here as a
# small constant to avoid pose/wilor depending on evaluation/).
_HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _color_for(frame: HandPoseFrame) -> tuple[int, int, int]:
    if frame.handedness_label == "left":
        return _COLOR_LEFT
    if frame.handedness_label == "right":
        return _COLOR_RIGHT
    return _COLOR_UNKNOWN


def _project_joints_2d(hf: HandPoseFrame) -> np.ndarray | None:
    """Real 2D reprojection of the reconstructed 21 3D joints into the
    source frame, using the same camera model WiLoR itself uses for its
    demo overlay (project_points_full_img, see geometry.py)."""
    if len(hf.landmarks_3d) < 21 or not hf.mano_references:
        return None
    cam_t = hf.mano_references.get("camera_translation_xyz")
    focal = hf.mano_references.get("focal_length")
    img_wh = hf.mano_references.get("img_size_wh")
    if not (cam_t and focal and img_wh):
        return None
    joints_3d = np.array([[j.x, j.y, j.z] for j in hf.landmarks_3d[:21]], dtype=np.float64)
    return project_points_full_img(joints_3d, np.array(cam_t, dtype=np.float64), focal, tuple(img_wh))


def draw_overlay(frame_bgr: np.ndarray, hand_frames: list[HandPoseFrame]) -> np.ndarray:
    out = frame_bgr.copy()
    for hf in hand_frames:
        color = _color_for(hf)
        joints_2d = _project_joints_2d(hf)
        if joints_2d is not None:
            for a, b in _HAND_BONES:
                pa = tuple(joints_2d[a].astype(int))
                pb = tuple(joints_2d[b].astype(int))
                cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)
            for p in joints_2d:
                cv2.circle(out, tuple(p.astype(int)), 3, color, -1)
        elif hf.landmarks_2d:
            pts = np.array([[int(p.x), int(p.y)] for p in hf.landmarks_2d], dtype=np.int32)
            if len(pts) == 4:  # detector-only bbox corners
                cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            else:
                for p in pts:
                    cv2.circle(out, tuple(p), 2, color, -1)
        label = hf.handedness_label or "unknown"
        conf = f"{hf.detection_confidence:.2f}" if hf.detection_confidence is not None else "?"
        anchor_pt = None
        if joints_2d is not None:
            anchor_pt = (int(joints_2d[0][0]), max(0, int(joints_2d[0][1]) - 10))
        elif hf.landmarks_2d:
            anchor_pt = (int(hf.landmarks_2d[0].x), max(0, int(hf.landmarks_2d[0].y) - 6))
        if anchor_pt is not None:
            cv2.putText(out, f"{label} {conf}", anchor_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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


def export_mesh_obj(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> Path:
    """Export one hand's reconstructed mesh as a plain .obj (via ``trimesh``,
    no OpenGL/EGL context required). ``faces`` comes from
    ``pipeline.model.mano.faces`` (shared across all hands/frames, since
    MANO's face topology is fixed). Output is a generated artifact
    (git-ignored), not committed."""
    import trimesh  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(out_path))
    return out_path
