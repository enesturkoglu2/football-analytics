"""Frame annotation helpers for tracked person boxes."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_tracks(
    frame: np.ndarray,
    observations: list[dict[str, Any]],
) -> np.ndarray:
    """Draw person boxes with optional track_id labels; never invent IDs."""
    annotated = frame.copy()
    for obs in observations:
        x1, y1, x2, y2 = (int(round(v)) for v in obs["bbox_xyxy"])
        conf = obs["confidence"]
        track_id = obs.get("track_id")
        if track_id is None:
            label = f"person {conf:.2f}"
        else:
            label = f"id={track_id} person {conf:.2f}"
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
