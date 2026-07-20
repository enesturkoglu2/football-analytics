"""Input and output validation for video ingest."""

from __future__ import annotations

from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

MANIFEST_NAME = "video_manifest.json"
FFPROBE_NAME = "ffprobe.json"


class ValidationError(ValueError):
    """Raised when ingest inputs or outputs fail validation."""


def validate_video_path(video_path: Path) -> Path:
    """Validate that the video path exists, is non-empty, and has an allowed extension."""
    path = video_path.expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"video path does not exist: {path}")
    if not path.is_file():
        raise ValidationError(f"video path is not a regular file: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(
            f"unsupported video extension '{path.suffix}' (allowed: {allowed})"
        )
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValidationError(f"video file is empty: {path}")
    return path


def output_paths(output_dir: Path) -> tuple[Path, Path]:
    """Return (manifest_path, ffprobe_path) for an output directory."""
    directory = output_dir.expanduser().resolve()
    return directory / MANIFEST_NAME, directory / FFPROBE_NAME


def check_output_collision(output_dir: Path, *, overwrite: bool) -> None:
    """Fail if output JSON files exist and overwrite is not requested."""
    manifest_path, ffprobe_path = output_paths(output_dir)
    existing = [p.name for p in (manifest_path, ffprobe_path) if p.exists()]
    if existing and not overwrite:
        names = ", ".join(existing)
        raise ValidationError(
            f"output already exists ({names}) in {output_dir.resolve()}; "
            "re-run with --overwrite to replace"
        )


def resolve_metadata(
    *,
    ffprobe_fields: dict,
    opencv_fields: dict,
) -> dict:
    """Cross-check sources and build resolved fields without inventing values."""
    notes: list[str] = []
    errors: list[str] = []

    ff_width = ffprobe_fields.get("width")
    ff_height = ffprobe_fields.get("height")
    oc_width = opencv_fields.get("width")
    oc_height = opencv_fields.get("height")

    width = _resolve_dimension("width", ff_width, oc_width, notes, errors)
    height = _resolve_dimension("height", ff_height, oc_height, notes, errors)

    ff_fps = ffprobe_fields.get("avg_frame_rate")
    if ff_fps is None:
        ff_fps = ffprobe_fields.get("r_frame_rate")
        if ff_fps is not None:
            notes.append("using ffprobe r_frame_rate because avg_frame_rate is unavailable")
    oc_fps = opencv_fields.get("fps")
    fps = _resolve_fps(ff_fps, oc_fps, notes, errors)

    ff_frames = ffprobe_fields.get("nb_frames")
    oc_frames = opencv_fields.get("frame_count")
    frame_count = _resolve_optional_int("frame_count", ff_frames, oc_frames, notes)

    duration_sec = ffprobe_fields.get("duration_sec")
    if duration_sec is None:
        if frame_count is not None and fps is not None and fps > 0:
            duration_sec = frame_count / fps
            notes.append("duration_sec derived from resolved frame_count / fps")
        else:
            notes.append("duration_sec unavailable from ffprobe and not derivable")

    if width is None or height is None:
        errors.append("resolved width/height unavailable")
    if fps is None:
        errors.append("resolved fps unavailable from ffprobe and OpenCV")
    if duration_sec is None:
        errors.append("resolved duration_sec unavailable")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "notes": notes,
    }


def _resolve_dimension(
    name: str,
    ff_value: int | None,
    oc_value: int | None,
    notes: list[str],
    errors: list[str],
) -> int | None:
    if ff_value is not None and oc_value is not None:
        if ff_value != oc_value:
            errors.append(
                f"{name} mismatch: ffprobe={ff_value}, opencv={oc_value}"
            )
            return None
        return ff_value
    if ff_value is not None:
        notes.append(f"resolved {name} from ffprobe only")
        return ff_value
    if oc_value is not None:
        notes.append(f"resolved {name} from opencv only")
        return oc_value
    return None


def _resolve_fps(
    ff_fps: float | None,
    oc_fps: float | None,
    notes: list[str],
    errors: list[str],
) -> float | None:
    if ff_fps is not None and oc_fps is not None:
        # Allow small floating differences between sources.
        if abs(ff_fps - oc_fps) > 0.05:
            notes.append(
                f"fps sources differ (ffprobe={ff_fps}, opencv={oc_fps}); "
                "resolved fps left null"
            )
            return None
        return ff_fps
    if ff_fps is not None:
        notes.append("resolved fps from ffprobe only")
        return ff_fps
    if oc_fps is not None:
        notes.append("resolved fps from opencv only")
        return oc_fps
    return None


def _resolve_optional_int(
    name: str,
    ff_value: int | None,
    oc_value: int | None,
    notes: list[str],
) -> int | None:
    if ff_value is not None and oc_value is not None:
        if ff_value != oc_value:
            notes.append(
                f"{name} sources differ (ffprobe={ff_value}, opencv={oc_value}); "
                "resolved value left null"
            )
            return None
        return ff_value
    if ff_value is not None:
        notes.append(f"resolved {name} from ffprobe only")
        return ff_value
    if oc_value is not None:
        notes.append(f"resolved {name} from opencv only")
        return oc_value
    notes.append(f"{name} unavailable from ffprobe and OpenCV")
    return None
