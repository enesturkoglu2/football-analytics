"""Diagnostic overlay MP4s for Target Tracking R1 (no annotation UI)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from football_analytics.reid.hil.common import sha256_file


def _transcode_h264(raw_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg h264 failed: {proc.stderr[-500:]}")
    if raw_path.is_file():
        raw_path.unlink()


def _draw_box(frame, bbox, color, label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def render_baseline_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    seed_raw_track_id: str,
    seed_last_frame: int,
    fps: float,
    frame_count: int,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source_video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rows = observations_by_frame.get(str(fi)) or []
            seed_seen = False
            for r in rows:
                tid = str(r.get("raw_track_id"))
                bbox = r["bbox_xyxy"]
                if tid == str(seed_raw_track_id):
                    _draw_box(frame, bbox, (0, 0, 255), f"TARGET RAW {tid}")
                    seed_seen = True
                else:
                    _draw_box(frame, bbox, (220, 220, 0), str(tid))
            banner = f"BASELINE · f={fi} t={fi/fps:.2f}s · seed_raw={seed_raw_track_id}"
            if fi > int(seed_last_frame):
                banner += " · TARGET RAW TRACK ENDED"
                cv2.putText(
                    frame,
                    "TARGET RAW TRACK ENDED — NO AUTO SWITCH",
                    (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif not seed_seen and fi <= int(seed_last_frame):
                cv2.putText(
                    frame,
                    "SEED TRACK ABSENT THIS FRAME",
                    (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
            cv2.putText(
                frame, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
            )
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    _transcode_h264(raw, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "kind": "baseline_target_raw_track_overlay",
    }


def render_stitched_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    timeline: Mapping[str, Any],
    stitch_events: Sequence[Mapping[str, Any]],
    fps: float,
    frame_count: int,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")

    # frame → active eligible interval
    intervals = list(timeline.get("intervals") or [])

    def interval_at(fi: int) -> dict[str, Any] | None:
        hit = None
        for iv in intervals:
            if int(iv["start_frame"]) <= fi <= int(iv["end_frame"]):
                # prefer analysis_eligible target intervals
                if iv.get("analysis_eligible"):
                    return iv
                hit = iv
        return hit

    stitch_start_frames = {}
    for ev in stitch_events:
        dec = ev.get("decision") or {}
        if dec.get("decision") == "AUTO_STITCH" and dec.get("selected"):
            sf = int(dec["selected"]["candidate_start_frame"])
            stitch_start_frames[sf] = {
                "prev": ev.get("previous_raw_track_id"),
                "new": dec["selected"]["candidate_raw_track_id"],
                "gap": dec["selected"].get("temporal_gap_frames"),
                "disp": dec["selected"].get("center_displacement_px"),
                "cost": dec["selected"].get("cost"),
            }

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source_video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    color_map = {
        "HUMAN_CONFIRMED": ((0, 0, 255), "TARGET — HUMAN SEED"),
        "RAW_TRACK_CONTINUATION": ((0, 0, 220), "TARGET — RAW TRACK"),
        "AUTO_STITCHED_CONTINUATION": ((0, 200, 0), "TARGET — AUTO STITCHED"),
        "UNRESOLVED": ((0, 165, 255), "TARGET UNRESOLVED"),
        "REVIEW_REQUIRED": ((180, 180, 0), "REVIEW REQUIRED"),
        "OUT_OF_FRAME": ((120, 120, 120), "OUT OF FRAME"),
    }
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rows = observations_by_frame.get(str(fi)) or []
            iv = interval_at(fi)
            target_tid = str(iv["raw_track_id"]) if iv and iv.get("raw_track_id") else None
            kind = iv["kind"] if iv else None
            for r in rows:
                tid = str(r.get("raw_track_id"))
                if target_tid and tid == target_tid:
                    color, label = color_map.get(kind or "", ((0, 0, 255), "TARGET"))
                    _draw_box(frame, r["bbox_xyxy"], color, f"{label} · {tid}")
                else:
                    _draw_box(frame, r["bbox_xyxy"], (220, 220, 0), tid)
            if iv and not target_tid:
                color, label = color_map.get(kind or "UNRESOLVED", ((0, 165, 255), "TARGET UNRESOLVED"))
                cv2.putText(
                    frame,
                    label,
                    (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if fi in stitch_start_frames:
                info = stitch_start_frames[fi]
                msg = (
                    f"AUTO STITCH · {info['prev']} → {info['new']} · "
                    f"gap={info['gap']} · disp={info['disp']:.1f}px · "
                    f"evidence=gap+displacement+scale+no_exact_frame_conflict"
                )
                cv2.putText(
                    frame,
                    msg,
                    (16, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            banner = f"STITCHED · f={fi} t={fi/fps:.2f}s · pt={timeline.get('persistent_target_id')}"
            cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
            cv2.putText(
                frame, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
            )
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    _transcode_h264(raw, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "kind": "stitched_persistent_target_overlay",
        "stitch_timestamps": [
            {
                "frame": sf,
                "time_sec": sf / fps,
                **info,
            }
            for sf, info in sorted(stitch_start_frames.items())
        ],
    }
