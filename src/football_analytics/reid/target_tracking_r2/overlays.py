"""R2 diagnostic overlays — purity split + segment-stitched target."""

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
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")
    if raw_path.is_file():
        raw_path.unlink()


def _box(frame, bbox, color, label: str, thick: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
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


def render_purity_split_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_raw_track_id: str,
    seed_segment: Mapping[str, Any],
    conflict_segment: Mapping[str, Any] | None,
    change_point: Mapping[str, Any] | None,
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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cp_f = int(change_point["change_point_frame"]) if change_point else None
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for r in observations_by_frame.get(str(fi)) or []:
                tid = str(r.get("raw_track_id"))
                if tid != str(parent_raw_track_id):
                    _box(frame, r["bbox_xyxy"], (200, 200, 0), tid, 1)
                    continue
                # parent raw track 10 — color by segment role (never paint conflict as target red)
                if (
                    int(seed_segment["start_frame"])
                    <= fi
                    <= int(seed_segment["end_frame"])
                ):
                    _box(
                        frame,
                        r["bbox_xyxy"],
                        (0, 220, 255),
                        f"TARGET SEED — PURE YELLOW · raw{tid}/{seed_segment['segment_id']}",
                        3,
                    )
                elif conflict_segment and int(conflict_segment["start_frame"]) <= fi <= int(
                    conflict_segment["end_frame"]
                ):
                    _box(
                        frame,
                        r["bbox_xyxy"],
                        (180, 180, 180),
                        f"IDENTITY CONFLICT — CROSS TEAM · raw{tid}/{conflict_segment['segment_id']}",
                        2,
                    )
                else:
                    _box(frame, r["bbox_xyxy"], (160, 160, 160), f"raw{tid}", 1)
            if cp_f is not None and abs(fi - cp_f) <= 1:
                cv2.putText(
                    frame,
                    f"PURITY BOUNDARY f={cp_f} t={cp_f/fps:.2f}s",
                    (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA,
                )
            banner = f"R2 PURITY SPLIT · f={fi} t={fi/fps:.2f}s · parent_raw={parent_raw_track_id} IMMUTABLE"
            cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
            cv2.putText(frame, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    _transcode_h264(raw, output_path)
    return {"path": str(output_path), "sha256": sha256_file(output_path), "bytes": output_path.stat().st_size}


def render_segment_stitched_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    timeline: Mapping[str, Any],
    rejected_cross_team: Sequence[Mapping[str, Any]],
    fps: float,
    frame_count: int,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")
    intervals = list(timeline.get("intervals") or [])

    def iv_at(fi: int):
        for iv in intervals:
            if int(iv["start_frame"]) <= fi <= int(iv["end_frame"]):
                if iv.get("analysis_eligible"):
                    return iv
        for iv in intervals:
            if int(iv["start_frame"]) <= fi <= int(iv["end_frame"]):
                return iv
        return None

    reject_ids = {str(c.get("candidate_raw_track_id")) for c in rejected_cross_team}

    cap = cv2.VideoCapture(str(source_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            iv = iv_at(fi)
            target_tid = str(iv["raw_track_id"]) if iv and iv.get("raw_track_id") and iv.get("analysis_eligible") else None
            for r in observations_by_frame.get(str(fi)) or []:
                tid = str(r.get("raw_track_id"))
                if target_tid and tid == target_tid:
                    label = iv.get("label") or "TARGET"
                    seg = iv.get("segment_id") or ""
                    _box(
                        frame,
                        r["bbox_xyxy"],
                        (0, 0, 255) if iv.get("kind") == "HUMAN_SEED_SEGMENT" else (0, 200, 0),
                        f"{label} · seg={seg} · raw={tid}",
                        3,
                    )
                elif tid in reject_ids:
                    _box(frame, r["bbox_xyxy"], (0, 0, 180), f"REJECTED CROSS-TEAM · raw={tid}", 2)
                elif iv and iv.get("kind") == "IDENTITY_CONFLICT_EXCLUDED" and tid == str(iv.get("raw_track_id")):
                    _box(
                        frame,
                        r["bbox_xyxy"],
                        (160, 160, 160),
                        "IDENTITY CONFLICT — CROSS TEAM (not target)",
                        2,
                    )
                else:
                    _box(frame, r["bbox_xyxy"], (200, 200, 0), tid, 1)
            if iv and not iv.get("analysis_eligible") and iv.get("kind") in {
                "TARGET_UNRESOLVED",
                "TARGET_TEMPORARILY_LOST",
            }:
                cv2.putText(
                    frame,
                    str(iv.get("label") or iv["kind"]),
                    (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 165, 255),
                    2,
                )
            banner = (
                f"R2 SEGMENT STITCH · f={fi} t={fi/fps:.2f}s · "
                f"pt={timeline.get('persistent_target_id')}"
            )
            cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
            cv2.putText(frame, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    _transcode_h264(raw, output_path)
    return {"path": str(output_path), "sha256": sha256_file(output_path), "bytes": output_path.stat().st_size}
