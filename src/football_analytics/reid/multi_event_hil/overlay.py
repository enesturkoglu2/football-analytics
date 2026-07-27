"""Diagnostic timeline overlay video (no metrics / no Game State)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.hil.common import sha256_file


def _intervals_covering_frame(
    intervals: Sequence[Mapping[str, Any]], frame_index: int
) -> list[dict[str, Any]]:
    hits = []
    for iv in intervals:
        if int(iv["start_frame"]) <= frame_index <= int(iv["end_frame"]):
            hits.append(dict(iv))
    return hits


def _unresolved_covering(
    unresolved: Sequence[Mapping[str, Any]], frame_index: int
) -> bool:
    for iv in unresolved:
        if iv.get("metadata", {}).get("zero_width_event_marker"):
            continue
        if int(iv["start_frame"]) <= frame_index <= int(iv["end_frame"]):
            return True
    return False


def render_timeline_overlay_video(
    *,
    video_path: Path,
    timeline: Mapping[str, Any],
    output_path: Path,
    observation_lookup: dict[tuple[str, int], list[float]] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Burn verified / continuation / unresolved labels into a diagnostic MP4."""
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, int(max_frames))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open VideoWriter: {output_path}")

    intervals = list(timeline.get("intervals") or [])
    unresolved = list(timeline.get("unresolved_intervals") or [])
    frames_written = 0
    try:
        for frame_index in range(total):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            label = None
            color = (180, 180, 180)
            hits = _intervals_covering_frame(intervals, frame_index)
            if hits:
                # Prefer human_confirmed over continuation if both (should not overlap)
                hit = sorted(
                    hits,
                    key=lambda x: 0 if x.get("status") == "human_confirmed" else 1,
                )[0]
                status = hit.get("status")
                if status == "human_confirmed":
                    label = "TARGET_001 — HUMAN VERIFIED"
                    color = (0, 180, 0)
                elif status == "tracker_continuation":
                    label = "TARGET_001 — TRACKER CONTINUATION"
                    color = (0, 200, 255)
                else:
                    label = f"TARGET_001 — {status}"
                    color = (0, 200, 255)
                seg = hit.get("segment_id")
                bbox = None
                if observation_lookup and seg is not None:
                    bbox = observation_lookup.get((str(seg), frame_index))
                if bbox is None and hit.get("first_bbox") and frame_index == hit["start_frame"]:
                    bbox = hit.get("first_bbox")
                if bbox is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{seg} / {hit.get('raw_track_id')}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    label,
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                # provenance strip
                decs = ",".join(str(d) for d in (hit.get("source_decision_ids") or [])[:2])
                cv2.putText(
                    frame,
                    f"dec:{decs}",
                    (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            elif _unresolved_covering(unresolved, frame_index):
                cv2.putText(
                    frame,
                    "UNRESOLVED",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 220),
                    2,
                    cv2.LINE_AA,
                )
            # Always show timestamp / frame
            tsec = frame_index / fps if fps else 0.0
            cv2.putText(
                frame,
                f"f={frame_index} t={tsec:.2f}s",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()
        cap.release()

    return {
        "overlay_path": str(output_path),
        "overlay_sha256": sha256_file(output_path),
        "frames_written": frames_written,
        "fps": fps,
        "width": width,
        "height": height,
        "note": "Diagnostic overlay only; not model input; no spatial calibration.",
    }
