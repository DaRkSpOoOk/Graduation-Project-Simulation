"""OpenCV overlay renderer for raw hand-landmarker validation videos."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def draw_hand_overlay(
    frame_bgr: np.ndarray,
    image_landmarks: np.ndarray,
    hand_present: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    frame_index: int,
    timestamp_seconds: float,
    connections: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Draw detector-order landmarks and metadata onto a copied BGR frame."""

    import cv2

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    colors = {"LEFT": (255, 80, 40), "RIGHT": (40, 180, 255), "": (180, 180, 180)}
    for hand_index in range(len(hand_present)):
        if not bool(hand_present[hand_index]):
            continue
        label = str(labels[hand_index]).upper()
        color = colors.get(label, (180, 180, 180))
        points: list[tuple[int, int] | None] = []
        for point in image_landmarks[hand_index]:
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y):
                points.append(None)
                continue
            pixel = (int(round(x * (width - 1))), int(round(y * (height - 1))))
            points.append(pixel)
            cv2.circle(annotated, pixel, 3, color, -1, lineType=cv2.LINE_AA)
        for start, end in connections:
            if start < len(points) and end < len(points) and points[start] and points[end]:
                cv2.line(annotated, points[start], points[end], color, 2, lineType=cv2.LINE_AA)
        wrist = points[0] if points else None
        score = float(scores[hand_index]) if np.isfinite(scores[hand_index]) else None
        text = label or "HAND"
        if score is not None:
            text += f" {score:.3f}"
        if wrist:
            cv2.putText(annotated, text, (wrist[0] + 8, wrist[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    overlay_text = f"frame={frame_index}  timestamp={timestamp_seconds:.6f}s"
    cv2.rectangle(annotated, (8, 8), (min(width - 8, 520), 42), (0, 0, 0), -1)
    cv2.putText(annotated, overlay_text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
