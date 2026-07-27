"""Build a local interactive-review video proxy (no source overwrite)."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from football_analytics.reid.hil.common import sha256_file


def ensure_interactive_review_proxy(
    *,
    source_video: Path,
    source_video_sha256: str,
    output_path: Path,
    max_width: int = 960,
) -> dict:
    """Create H.264 proxy for HTML5 interactive review; never mutate source."""
    source_video = source_video.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual = sha256_file(source_video)
    if actual != source_video_sha256.lower():
        raise RuntimeError("source video SHA mismatch for interactive proxy")

    if not output_path.is_file():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            "-crf",
            "28",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not output_path.is_file():
            raise RuntimeError(f"interactive proxy ffmpeg failed: {proc.stderr[-500:]}")

    data = output_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "proxy_path": str(output_path),
        "proxy_sha256": sha256_file(output_path),
        "source_video_sha256": actual,
        "bytes": len(data),
        "data_url": f"data:video/mp4;base64,{b64}",
        "source_overwritten": False,
    }
