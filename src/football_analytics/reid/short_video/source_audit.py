"""Read-only source audit for product short videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from football_analytics.ingest.checksum import sha256_file


def _ffprobe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def _scene_cut_count(path: Path, *, threshold: float = 0.4) -> int | None:
    """Best-effort camera-cut count via ffmpeg scene filter; None if unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [
        ffmpeg,
        "-i",
        str(path),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # showinfo lines for selected frames appear on stderr
    return sum(1 for line in (proc.stderr or "").splitlines() if "pts_time:" in line and "showinfo" in line)


def audit_short_video(
    path: Path,
    *,
    windows_path_stated: str | None = None,
    windows_path_resolved: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    digest = sha256_file(path)
    size = path.stat().st_size
    probe = _ffprobe(path)
    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    vs = vstreams[0] if vstreams else {}
    fmt = probe.get("format", {})

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decoded = 0
    decode_failures = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or getattr(frame, "size", 0) == 0:
            decode_failures += 1
            continue
        decoded += 1
    cap.release()

    tags = vs.get("tags") or {}
    rotation = tags.get("rotate") or vs.get("rotation")
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = sd.get("rotation")

    duration = float(fmt.get("duration") or (decoded / fps if fps else 0.0))
    cuts = _scene_cut_count(path)

    return {
        "schema_version": "short_video_source_audit_v1",
        "absolute_path": str(path),
        "windows_path_stated": windows_path_stated,
        "windows_path_resolved": windows_path_resolved,
        "sha256": digest,
        "bytes": size,
        "duration_sec": duration,
        "fps": fps,
        "fps_ffprobe": vs.get("r_frame_rate") or vs.get("avg_frame_rate"),
        "frame_count_meta": meta_frames,
        "frame_count_decoded": decoded,
        "resolution": {"width": width, "height": height},
        "codec": vs.get("codec_name"),
        "codec_long": vs.get("codec_long_name"),
        "pix_fmt": vs.get("pix_fmt"),
        "container": fmt.get("format_name"),
        "decode_opened": True,
        "decode_failures": decode_failures,
        "decode_integrity_ok": decode_failures == 0 and decoded > 0,
        "orientation_rotation": rotation,
        "audio_streams": len(astreams),
        "audio_codec": astreams[0].get("codec_name") if astreams else None,
        "camera_cuts_estimate": cuts,
        "duration_in_preferred_30_60": 30.0 <= duration <= 60.0,
        "source_modified": False,
    }
