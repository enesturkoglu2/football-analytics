"""Orchestrate video ingest: validate, probe, checksum, write outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.ingest.checksum import sha256_file
from football_analytics.ingest.ffprobe import (
    FfprobeError,
    extract_ffprobe_fields,
    run_ffprobe,
)
from football_analytics.ingest.manifest import build_manifest, write_json_atomic
from football_analytics.ingest.opencv_meta import OpenCVMetaError, read_opencv_metadata
from football_analytics.ingest.validate import (
    ValidationError,
    check_output_collision,
    output_paths,
    resolve_metadata,
    validate_video_path,
)


class IngestError(RuntimeError):
    """Raised when the ingest pipeline fails."""


def run_ingest(
    video: str | Path,
    output_dir: str | Path = "outputs/ingest",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Validate a video and write video_manifest.json plus ffprobe.json.

    On failure, no success JSON files are written and IngestError is raised.
    """
    try:
        video_path = validate_video_path(Path(video))
        out_dir = Path(output_dir).expanduser().resolve()
        check_output_collision(out_dir, overwrite=overwrite)

        size_bytes = video_path.stat().st_size
        digest = sha256_file(video_path)

        ffprobe_payload = run_ffprobe(video_path)
        ffprobe_fields = extract_ffprobe_fields(ffprobe_payload)
        opencv_fields = read_opencv_metadata(video_path)
        resolved = resolve_metadata(
            ffprobe_fields=ffprobe_fields,
            opencv_fields=opencv_fields,
        )

        manifest = build_manifest(
            video_path=video_path,
            size_bytes=size_bytes,
            sha256=digest,
            ffprobe_fields=ffprobe_fields,
            opencv_fields=opencv_fields,
            resolved=resolved,
        )

        manifest_path, ffprobe_path = output_paths(out_dir)
        # Write ffprobe first, then manifest. Collision was already checked.
        write_json_atomic(ffprobe_path, ffprobe_payload)
        write_json_atomic(manifest_path, manifest)

        return {
            "manifest_path": str(manifest_path),
            "ffprobe_path": str(ffprobe_path),
            "manifest": manifest,
        }
    except (ValidationError, FfprobeError, OpenCVMetaError, OSError) as exc:
        raise IngestError(str(exc)) from exc
