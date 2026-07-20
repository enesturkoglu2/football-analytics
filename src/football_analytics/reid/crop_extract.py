"""Extract JPEG crops from video using selected manifest rows (single-pass read)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.crop_select import (
    CropSelectError,
    load_crop_selection_config,
    select_crops_from_tracks_file,
)
from football_analytics.reid.writers import (
    CROPS_DIRNAME,
    MANIFEST_NAME,
    ReIDWritersError,
    check_output_collision,
    cleanup_dir,
    create_temp_output_dir,
    finalize_output_dir,
    validate_manifest_disk_consistency,
    write_manifest_jsonl,
)

# Documented JPEG quality for ReID crops (OpenCV IMWRITE_JPEG_QUALITY).
JPEG_QUALITY = 95


class CropExtractError(RuntimeError):
    """Raised when crop extraction fails."""


def probe_video_size(video_path: str | Path) -> tuple[int, int, float, int]:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise CropExtractError(f"video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CropExtractError(f"could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise CropExtractError(f"invalid video dimensions for {path}: {width}x{height}")
    return width, height, fps, frame_count


def crop_frame_region(frame: np.ndarray, bbox_xyxy: Sequence[int]) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise CropExtractError("frame must be an HxWx3 array")
    height, width = frame.shape[:2]
    left, top, right, bottom = [int(v) for v in bbox_xyxy]
    if left < 0 or top < 0 or right > width or bottom > height:
        raise CropExtractError(
            f"crop bbox out of frame bounds: {[left, top, right, bottom]} "
            f"vs {(width, height)}"
        )
    if right <= left or bottom <= top:
        raise CropExtractError(f"empty crop bbox: {[left, top, right, bottom]}")
    crop = frame[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] <= 0 or crop.shape[1] <= 0:
        raise CropExtractError("empty crop array")
    return crop.copy()


def write_crop_jpeg(path: Path, crop_bgr: np.ndarray, *, jpeg_quality: int = JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        crop_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise CropExtractError(f"cv2.imwrite failed for {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise CropExtractError(f"JPEG not written or empty: {path}")


def extract_crops_single_pass(
    *,
    video_path: str | Path,
    selected_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    open_capture: Callable[[str], Any] | None = None,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[dict[str, Any]]:
    """Read the video once in order and write all selected crops under output_dir."""
    if not selected_rows:
        raise CropExtractError("no selected crops to extract")

    path = Path(video_path).expanduser().resolve()
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_frame[int(row["frame_index"])].append(row)

    needed_frames = sorted(by_frame.keys())
    opener = open_capture or (lambda p: cv2.VideoCapture(p))
    capture = opener(str(path))
    if capture is None or not capture.isOpened():
        raise CropExtractError(f"could not open video: {path}")

    written: list[dict[str, Any]] = []
    next_needed_idx = 0
    frame_index = 0
    try:
        while next_needed_idx < len(needed_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                missing = needed_frames[next_needed_idx:]
                raise CropExtractError(
                    f"failed reading required frame_index values: {missing}"
                )
            target = needed_frames[next_needed_idx]
            if frame_index == target:
                for row in by_frame[target]:
                    crop = crop_frame_region(frame, row["bbox_xyxy"])
                    rel = str(row["crop_relative_path"])
                    absolute = output_dir / rel
                    write_crop_jpeg(absolute, crop, jpeg_quality=jpeg_quality)
                    written.append(dict(row))
                next_needed_idx += 1
            frame_index += 1
    finally:
        capture.release()

    if len(written) != len(selected_rows):
        raise CropExtractError(
            f"wrote {len(written)} crops but expected {len(selected_rows)}"
        )
    return written


def run_select_reid_crops(
    *,
    video: str | Path,
    tracks: str | Path,
    config: str | Path,
    output_dir: str | Path,
    track_ids: Sequence[int] | None = None,
    overwrite: bool = False,
    open_capture: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    video_path = Path(video).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()

    check_output_collision(final_dir, overwrite=overwrite)
    cfg = load_crop_selection_config(config_path)
    width, height, _fps, _frame_count = probe_video_size(video_path)

    # Prefer a stable relative source_video for manifests when possible.
    try:
        source_video = str(video_path.relative_to(Path.cwd().resolve()))
    except ValueError:
        source_video = str(video_path)

    selection = select_crops_from_tracks_file(
        tracks_path=tracks_path,
        config=cfg,
        video_width=width,
        video_height=height,
        source_video=source_video,
        track_ids=track_ids,
    )
    selected = selection["selected"]
    if not selected:
        raise CropSelectError("no crops selected after filters")

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_output_dir(final_dir)
        extract_crops_single_pass(
            video_path=video_path,
            selected_rows=selected,
            output_dir=temp_dir,
            open_capture=open_capture,
        )
        write_manifest_jsonl(temp_dir / MANIFEST_NAME, selected)
        validate_manifest_disk_consistency(temp_dir, selected)
        finalized = finalize_output_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "manifest_path": str(finalized / MANIFEST_NAME),
        "crops_dir": str(finalized / CROPS_DIRNAME),
        "jpeg_quality": JPEG_QUALITY,
        "observations_read": selection["observations_read"],
        "tracks_examined": selection["tracks_examined"],
        "eligible_observations": selection["eligible_observations"],
        "tracks_with_crops": selection["tracks_with_crops"],
        "crops_written": selection["crops_selected"],
        "filter_reasons": selection["filter_reasons"],
        "selected": selected,
    }
