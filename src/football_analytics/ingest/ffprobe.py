"""FFprobe wrappers for container and stream metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class FfprobeError(RuntimeError):
    """Raised when ffprobe is missing or returns unusable output."""


def parse_frame_rate(value: str | None) -> float | None:
    """Parse an FFprobe frame-rate string such as '25/1' or '30000/1001'."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"0/0", "N/A", "nan"}:
        return None
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0:
                return None
            rate = numerator / denominator
        else:
            rate = float(text)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    return rate


def run_ffprobe(video_path: Path, timeout_sec: float = 60.0) -> dict[str, Any]:
    """Run ffprobe and return the parsed JSON payload."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise FfprobeError("ffprobe not found on PATH")

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise FfprobeError(f"ffprobe timed out after {timeout_sec}s") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise FfprobeError(
            f"ffprobe failed with exit code {completed.returncode}: {stderr or 'no stderr'}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FfprobeError("ffprobe returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise FfprobeError("ffprobe JSON root must be an object")
    return payload


def select_video_stream(ffprobe_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first video stream or raise if none exists."""
    streams = ffprobe_payload.get("streams")
    if not isinstance(streams, list):
        raise FfprobeError("ffprobe payload missing streams list")
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream
    raise FfprobeError("no video stream found in ffprobe output")


def extract_ffprobe_fields(ffprobe_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized fields from an ffprobe payload without inventing values."""
    video_stream = select_video_stream(ffprobe_payload)
    fmt = ffprobe_payload.get("format")
    if not isinstance(fmt, dict):
        fmt = {}

    duration_raw = fmt.get("duration")
    duration_sec: float | None
    try:
        duration_sec = float(duration_raw) if duration_raw is not None else None
        if duration_sec is not None and duration_sec <= 0:
            duration_sec = None
    except (TypeError, ValueError):
        duration_sec = None

    bit_rate_raw = fmt.get("bit_rate")
    bit_rate: int | None
    try:
        bit_rate = int(bit_rate_raw) if bit_rate_raw is not None else None
    except (TypeError, ValueError):
        bit_rate = None

    width = _positive_int(video_stream.get("width"))
    height = _positive_int(video_stream.get("height"))
    nb_frames = _positive_int(video_stream.get("nb_frames"))

    avg_frame_rate = parse_frame_rate(
        video_stream.get("avg_frame_rate")
        if isinstance(video_stream.get("avg_frame_rate"), str)
        else None
    )
    r_frame_rate = parse_frame_rate(
        video_stream.get("r_frame_rate")
        if isinstance(video_stream.get("r_frame_rate"), str)
        else None
    )

    return {
        "format_name": fmt.get("format_name") if isinstance(fmt.get("format_name"), str) else None,
        "duration_sec": duration_sec,
        "bit_rate": bit_rate,
        "codec_name": (
            video_stream.get("codec_name")
            if isinstance(video_stream.get("codec_name"), str)
            else None
        ),
        "width": width,
        "height": height,
        "avg_frame_rate": avg_frame_rate,
        "r_frame_rate": r_frame_rate,
        "nb_frames": nb_frames,
        "avg_frame_rate_raw": video_stream.get("avg_frame_rate"),
        "r_frame_rate_raw": video_stream.get("r_frame_rate"),
    }


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number
