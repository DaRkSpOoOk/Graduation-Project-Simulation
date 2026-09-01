"""Per-frame inference: turns detector + WiLoR model outputs into the
repository-common ``HandPoseFrame`` representation.

Two extraction modes are supported:

* :func:`extract_frame_full` -- YOLO detector + full WiLoR/MANO
  reconstruction. Requires all assets in :class:`WilorAssetPaths`,
  including the gated ``MANO_RIGHT.pkl``.
* :func:`extract_frame_detector_only` -- YOLO detector only. Used when MANO
  assets are unavailable (see reports/pose/wilor/TASK-002-wilor-karsl-pilot.md);
  produces hand presence/handedness/bbox/confidence but leaves MANO and 3D
  joint fields empty, and tags frames with the ``detector_only_no_mano``
  quality flag so downstream consumers cannot mistake this for a full
  reconstruction run.

Mesh vertices (778 x 3 per hand) are NOT embedded in ``HandPoseFrame`` --
they are large and per Task 4 belong in the immutable NPZ raw-output store.
This module returns them separately, keyed by ``(frame_index, hand_index)``,
for the caller to hand to :mod:`pose.wilor.npz_io`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pose.common.schema import HandPoseFrame, Landmark2D, Landmark3D

EXTRACTOR_NAME = "wilor"


def _handedness_label(is_right: float) -> str:
    return "right" if float(is_right) >= 0.5 else "left"


def extract_frame_detector_only(
    detector: Any,
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp_seconds: float,
    *,
    confidence_threshold: float,
    extractor_version: str,
) -> list[HandPoseFrame]:
    """Run only the YOLO hand detector (no MANO reconstruction).

    Detector classes follow the official WiLoR convention: ``cls`` encodes
    handedness directly (see demo.py: ``is_right = det.boxes.cls``).
    """
    results = detector(frame_bgr, conf=confidence_threshold, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return [
            HandPoseFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                hand_present=False,
                extractor_metadata={
                    "extractor": EXTRACTOR_NAME,
                    "extractor_version": extractor_version,
                    "mode": "detector_only",
                },
                quality_flags=["detector_only_no_mano", "no_hand_detected"],
            )
        ]

    frames: list[HandPoseFrame] = []
    for i in range(len(boxes)):
        data = boxes.data[i].detach().cpu().numpy()
        x1, y1, x2, y2, conf = data[:5]
        cls = float(boxes.cls[i].detach().cpu().item())
        bbox_2d = [
            Landmark2D(x=float(x1), y=float(y1)),
            Landmark2D(x=float(x2), y=float(y1)),
            Landmark2D(x=float(x2), y=float(y2)),
            Landmark2D(x=float(x1), y=float(y2)),
        ]
        frames.append(
            HandPoseFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                hand_present=True,
                handedness_label=_handedness_label(cls),
                detection_confidence=float(conf),
                landmarks_2d=bbox_2d,
                extractor_metadata={
                    "extractor": EXTRACTOR_NAME,
                    "extractor_version": extractor_version,
                    "mode": "detector_only",
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                },
                quality_flags=["detector_only_no_mano"],
            )
        )
    return frames


def extract_frame_full(
    pipeline: Any,
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp_seconds: float,
    *,
    runtime_confidence: float,
    rescale_factor: float,
    extractor_version: str,
    checkpoint_id: str,
) -> tuple[list[HandPoseFrame], dict[tuple[int, int], np.ndarray]]:
    """Run the full detector + WiLoR/MANO pipeline on one frame.

    Returns:
        (hand_pose_frames, vertices_by_hand) where ``vertices_by_hand`` maps
        ``(frame_index, hand_index)`` -> ``float32[778, 3]`` mesh vertices,
        to be persisted separately (see module docstring).
    """
    import torch  # noqa: PLC0415
    from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: PLC0415
    from wilor.utils import recursive_to  # noqa: PLC0415

    from .geometry import cam_crop_to_full

    model = pipeline.model
    detector = pipeline.detector
    model_cfg = pipeline.model_cfg
    device = pipeline.device

    detections = detector(frame_bgr, conf=runtime_confidence, verbose=False)[0]
    bboxes, is_right_list = [], []
    for det in detections:
        box = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(box[:4].tolist())

    if not bboxes:
        return (
            [
                HandPoseFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    hand_present=False,
                    extractor_metadata={
                        "extractor": EXTRACTOR_NAME,
                        "extractor_version": extractor_version,
                        "checkpoint_id": checkpoint_id,
                        "mode": "full",
                    },
                    quality_flags=["no_hand_detected"],
                )
            ],
            {},
        )

    boxes = np.stack(bboxes)
    right = np.stack(is_right_list)
    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)

    frames: list[HandPoseFrame] = []
    vertices_by_hand: dict[tuple[int, int], np.ndarray] = {}

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
            .detach()
            .cpu()
            .numpy()
        )

        batch_size = batch["img"].shape[0]
        for n in range(batch_size):
            is_right_n = float(batch["right"][n].cpu().numpy())
            verts = out["pred_vertices"][n].detach().cpu().numpy().copy()
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().copy()
            verts[:, 0] = (2 * is_right_n - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right_n - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n]

            mano_params = out["pred_mano_params"]
            hand_pose = mano_params["hand_pose"][n].detach().cpu().numpy().tolist()
            global_orient = mano_params["global_orient"][n].detach().cpu().numpy().tolist()
            betas = mano_params["betas"][n].detach().cpu().numpy().tolist()

            landmarks_3d = [Landmark3D(x=float(j[0]), y=float(j[1]), z=float(j[2])) for j in joints]
            wrist = landmarks_3d[0] if landmarks_3d else None

            det_conf = float(boxes[n][4]) if boxes.shape[1] > 4 else None

            frames.append(
                HandPoseFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    hand_present=True,
                    handedness_label=_handedness_label(is_right_n),
                    detection_confidence=det_conf,
                    landmarks_3d=landmarks_3d,
                    wrist_position=wrist,
                    mano_params={
                        "hand_pose_rotmat": hand_pose,
                        "global_orient_rotmat": global_orient,
                        "betas": betas,
                        "num_hand_joints": model_cfg.MANO.NUM_HAND_JOINTS,
                        "representation": "rotation_matrix",
                    },
                    mano_references={
                        "camera_translation_xyz": [float(c) for c in cam_t],
                        "focal_length": float(scaled_focal_length),
                        "box_center_xy": box_center[n].detach().cpu().tolist(),
                        "box_size": float(box_size[n].detach().cpu().item()),
                        "img_size_wh": img_size[n].detach().cpu().tolist(),
                        "vertices_ref": f"frame{frame_index:06d}_hand{n}",
                        "num_mesh_vertices": int(verts.shape[0]),
                    },
                    extractor_metadata={
                        "extractor": EXTRACTOR_NAME,
                        "extractor_version": extractor_version,
                        "checkpoint_id": checkpoint_id,
                        "mode": "full",
                        "fast_mode": bool(pipeline.fast_mode),
                    },
                )
            )
            vertices_by_hand[(frame_index, n)] = verts.astype(np.float32)

    return frames, vertices_by_hand
