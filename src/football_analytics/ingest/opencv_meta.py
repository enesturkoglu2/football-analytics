"""OpenCV-based readability checks and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class OpenCVMetaError(RuntimeError):
    """Raised when OpenCV cannot open or read the video."""


def read_opencv_metadata(video_path: Path) -> dict[str, Any]:
    """Open the video, read the first frame, and collect CAP_PROP values."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        opened = bool(capture.isOpened())
        if not opened:
            raise OpenCVMetaError(f"OpenCV could not open video: {video_path}")

        ok, _frame = capture.read()
        if not ok:
            raise OpenCVMetaError(f"OpenCV could not read the first frame: {video_path}")

        width = _positive_number(capture.get(cv2.CAP_PROP_FRAME_WIDTH), as_int=True)
        height = _positive_number(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), as_int=True)
        fps = _positive_number(capture.get(cv2.CAP_PROP_FPS), as_int=False)
        frame_count = _positive_number(capture.get(cv2.CAP_PROP_FRAME_COUNT), as_int=True)

        return {
            "opened": True,
            "first_frame_ok": True,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        }
    finally:
        capture.release()


def _positive_number(value: Any, *, as_int: bool) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if as_int:
        return int(number)
    return number
