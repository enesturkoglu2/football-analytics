"""Output path helpers, overwrite checks, and atomic finalize/cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

TRACKED_NAME = "tracked.mp4"
TRACKS_NAME = "tracks.jsonl"
SUMMARY_NAME = "tracking_summary.json"

# Codec-friendly temp suffix (OpenCV rejects "*.mp4.tmp").
TMP_TRACKED_NAME = "_tmp_tracked.mp4"
TMP_TRACKS_NAME = "_tmp_tracks.jsonl"
TMP_SUMMARY_NAME = "_tmp_tracking_summary.json"


class WritersError(RuntimeError):
    """Raised when tracking outputs cannot be prepared or finalized."""


def output_paths(output_dir: Path) -> dict[str, Path]:
    directory = output_dir.expanduser().resolve()
    return {
        "tracked": directory / TRACKED_NAME,
        "tracks": directory / TRACKS_NAME,
        "summary": directory / SUMMARY_NAME,
        "tmp_tracked": directory / TMP_TRACKED_NAME,
        "tmp_tracks": directory / TMP_TRACKS_NAME,
        "tmp_summary": directory / TMP_SUMMARY_NAME,
    }


def check_output_collision(output_dir: Path, *, overwrite: bool) -> None:
    paths = output_paths(output_dir)
    finals = [paths["tracked"], paths["tracks"], paths["summary"]]
    existing = [p.name for p in finals if p.exists()]
    if existing and not overwrite:
        names = ", ".join(existing)
        raise WritersError(
            f"output already exists ({names}) in {output_dir.resolve()}; "
            "re-run with --overwrite to replace"
        )


def cleanup_partial_outputs(output_dir: Path) -> None:
    """Remove temporary and known partial output files."""
    paths = output_paths(output_dir)
    for key in ("tmp_tracked", "tmp_tracks", "tmp_summary"):
        path = paths[key]
        if path.exists():
            path.unlink(missing_ok=True)
    directory = output_dir.expanduser().resolve()
    if directory.is_dir():
        for pattern in ("_tmp_*", "*.tmp"):
            for stray in directory.glob(pattern):
                if stray.is_file():
                    stray.unlink(missing_ok=True)


def write_jsonl_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def finalize_outputs(output_dir: Path) -> dict[str, Path]:
    """Atomically move temporary outputs to final names via os.replace."""
    paths = output_paths(output_dir)
    required_tmps = (
        ("tmp_tracked", "tracked"),
        ("tmp_tracks", "tracks"),
        ("tmp_summary", "summary"),
    )
    for tmp_key, _final_key in required_tmps:
        if not paths[tmp_key].is_file():
            raise WritersError(
                f"missing temporary output before finalize: {paths[tmp_key].name}"
            )

    for tmp_key, final_key in required_tmps:
        os.replace(paths[tmp_key], paths[final_key])

    return {
        "tracked": paths["tracked"],
        "tracks": paths["tracks"],
        "summary": paths["summary"],
    }
