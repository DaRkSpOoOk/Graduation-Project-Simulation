"""Small camera-math utilities vendored from WiLoR to avoid a hard
``pyrender`` (OpenGL/EGL) dependency for raw-output extraction.

``cam_crop_to_full`` below reproduces (unchanged) the function of the same
name in the official WiLoR repository, commit fcb911312a38fa8badd30d9656a167485d61b8f9,
at ``wilor/utils/renderer.py``:
https://github.com/rolpotamias/WiLoR/blob/main/wilor/utils/renderer.py

The upstream file bundles this pure coordinate-transform helper together
with an OpenGL mesh renderer, so importing it unconditionally pulls in
``pyrender`` even when only raw MANO camera translation is needed. We do not
need mesh rendering for Milestone 1 (see reports/pose/wilor/TASK-002-wilor-karsl-pilot.md,
"Visual validation"), so this module isolates just the math.

WiLoR's own code license (CC-BY-NC-ND) covers the model/weights/repository
as a whole; this is a short, standard camera-projection formula (crop-space
weak-perspective camera -> full-image translation), reproduced here for
non-commercial research use with clear attribution.
"""

from __future__ import annotations

import torch


def cam_crop_to_full(
    cam_bbox: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: float = 5000.0,
) -> torch.Tensor:
    """Convert a crop-relative weak-perspective camera to a full-image
    3D translation (tx, ty, tz)."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)
