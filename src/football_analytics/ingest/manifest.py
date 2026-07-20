"""Manifest assembly and atomic JSON writes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest(
    *,
    video_path: Path,
    size_bytes: int,
    sha256: str,
    ffprobe_fields: dict[str, Any],
    opencv_fields: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the video_manifest.json payload."""
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "path": str(video_path),
            "filename": video_path.name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        },
        "container": {
            "format_name": ffprobe_fields.get("format_name"),
            "duration_sec": ffprobe_fields.get("duration_sec"),
            "bit_rate": ffprobe_fields.get("bit_rate"),
        },
        "video_stream": {
            "codec_name": ffprobe_fields.get("codec_name"),
            "width": ffprobe_fields.get("width"),
            "height": ffprobe_fields.get("height"),
            "avg_frame_rate": ffprobe_fields.get("avg_frame_rate_raw"),
            "r_frame_rate": ffprobe_fields.get("r_frame_rate_raw"),
            "nb_frames": ffprobe_fields.get("nb_frames"),
        },
        "opencv": {
            "opened": opencv_fields.get("opened"),
            "first_frame_ok": opencv_fields.get("first_frame_ok"),
            "width": opencv_fields.get("width"),
            "height": opencv_fields.get("height"),
            "fps": opencv_fields.get("fps"),
            "frame_count": opencv_fields.get("frame_count"),
        },
        "resolved": {
            "width": resolved.get("width"),
            "height": resolved.get("height"),
            "fps": resolved.get("fps"),
            "frame_count": resolved.get("frame_count"),
            "duration_sec": resolved.get("duration_sec"),
            "notes": list(resolved.get("notes") or []),
        },
        "status": "ok",
        "errors": [],
    }


def write_json_atomic(path: Path, payload: dict[str, Any] | Any) -> None:
    """Write JSON via a temporary sibling file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
