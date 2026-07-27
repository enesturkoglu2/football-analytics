"""Deterministic event-window review clips (no source overwrite, no public hosting)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from football_analytics.reid.hil.common import sha256_file


def _frame_to_seconds(frame: int, fps: float) -> float:
    return float(frame) / float(fps) if fps > 0 else 0.0


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()
    return fps if fps > 0 else 30.0


def extract_event_window_clip(
    *,
    source_video: Path,
    source_video_sha256: str,
    event: Mapping[str, Any],
    output_dir: Path,
    candidate_bboxes_by_frame: dict[int, list[list[float]]] | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Extract review clip for event review_window_* frames with optional bbox overlay.

    Window bounds come from the event contract (no invented padding).
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_video = source_video.resolve()
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source video SHA mismatch for clip extraction")

    start_f = int(event["review_window_start_frame"])
    end_f = int(event["review_window_end_frame"])
    if end_f < start_f:
        raise RuntimeError("invalid event review window")
    fps_v = float(fps or probe_fps(source_video))
    start_s = _frame_to_seconds(start_f, fps_v)
    # inclusive end frame → duration covers end_f
    duration_s = _frame_to_seconds(end_f - start_f + 1, fps_v)
    event_id = str(event["event_id"])

    # 1) Stream-copy cut (browser-friendly when source is already H.264)
    copy_path = output_dir / f"{event_id}_window_copy.mp4"
    if not copy_path.is_file():
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            str(source_video),
            "-t",
            f"{duration_s:.6f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(copy_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not copy_path.is_file():
            raise RuntimeError(f"ffmpeg copy clip failed: {proc.stderr[-500:]}")

    # 2) Overlay clip (re-encode H.264 for browser)
    overlay_path = output_dir / f"{event_id}_window_overlay.mp4"
    raw_overlay = output_dir / f"{event_id}_window_overlay_raw.mp4"
    if not overlay_path.is_file():
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {source_video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_overlay), fourcc, fps_v, (width, height))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_f))
            for frame_index in range(start_f, end_f + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                for bbox in (candidate_bboxes_by_frame or {}).get(frame_index, []):
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
                cv2.putText(
                    frame,
                    f"{event_id} f={frame_index}",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()
            cap.release()
        # Remux/transcode to H.264 for browsers
        cmd2 = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_overlay),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(overlay_path),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
        if proc2.returncode != 0 or not overlay_path.is_file():
            raise RuntimeError(f"ffmpeg h264 overlay failed: {proc2.stderr[-500:]}")
        if raw_overlay.is_file():
            raw_overlay.unlink()

    manifest = {
        "schema_version": "mehil_r1_event_window_clip_v1",
        "event_id": event_id,
        "event_type": event.get("event_type"),
        "source_video_path": str(source_video),
        "source_video_sha256": source_video_sha256,
        "review_window_start_frame": start_f,
        "review_window_end_frame": end_f,
        "fps": fps_v,
        "start_time_seconds": start_s,
        "duration_seconds": duration_s,
        "padding_invented": False,
        "requires_calibration_for_extra_padding": True,
        "copy_clip_path": str(copy_path),
        "copy_clip_sha256": sha256_file(copy_path),
        "overlay_clip_path": str(overlay_path),
        "overlay_clip_sha256": sha256_file(overlay_path),
        "browser_compatible_h264": True,
        "note": "Video playback is for review; confirming links existing tracklets and does not rerun tracking.",
    }
    man_path = output_dir / f"{event_id}_clip_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    return manifest


def bboxes_from_candidates_for_window(
    candidates: Sequence[Mapping[str, Any]],
    *,
    start_frame: int,
    end_frame: int,
) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    for cand in candidates:
        for ref in cand.get("bbox_references") or []:
            fi = int(ref["frame_index"])
            if start_frame <= fi <= end_frame:
                out.setdefault(fi, []).append([float(v) for v in ref["bbox_xyxy"]])
    return out
