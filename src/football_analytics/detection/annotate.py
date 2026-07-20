"""Bounding-box sanitization and frame annotation helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def sanitize_detection(
    *,
    bbox_xyxy: list[float] | tuple[float, ...] | Any,
    confidence: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    """
    Clip bbox to frame bounds and validate geometry/confidence.

    Returns a dict with float bbox_xyxy and confidence, or None if invalid.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if conf < 0.0 or conf > 1.0:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(frame_width), x1))
    x2 = max(0.0, min(float(frame_width), x2))
    y1 = max(0.0, min(float(frame_height), y1))
    y2 = max(0.0, min(float(frame_height), y2))

    if not (x1 < x2 and y1 < y2):
        return None

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": conf,
    }


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw person boxes and confidence labels; no tracking IDs."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
        conf = det["confidence"]
        label = f"person {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = y1 - 8 if y1 > 20 else y1 + 16
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
