"""Output path helpers, overwrite checks, and atomic finalize/cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

ANNOTATED_NAME = "annotated.mp4"
DETECTIONS_NAME = "detections.jsonl"
SUMMARY_NAME = "detection_summary.json"

# Temp names must keep a codec-friendly suffix (OpenCV rejects "*.mp4.tmp").
TMP_ANNOTATED_NAME = "_tmp_annotated.mp4"
TMP_DETECTIONS_NAME = "_tmp_detections.jsonl"
TMP_SUMMARY_NAME = "_tmp_detection_summary.json"


class WritersError(RuntimeError):
    """Raised when detection outputs cannot be prepared or finalized."""


def output_paths(output_dir: Path) -> dict[str, Path]:
    directory = output_dir.expanduser().resolve()
    return {
        "annotated": directory / ANNOTATED_NAME,
        "detections": directory / DETECTIONS_NAME,
        "summary": directory / SUMMARY_NAME,
        "tmp_annotated": directory / TMP_ANNOTATED_NAME,
        "tmp_detections": directory / TMP_DETECTIONS_NAME,
        "tmp_summary": directory / TMP_SUMMARY_NAME,
    }


def check_output_collision(output_dir: Path, *, overwrite: bool) -> None:
    paths = output_paths(output_dir)
    finals = [paths["annotated"], paths["detections"], paths["summary"]]
    existing = [p.name for p in finals if p.exists()]
    if existing and not overwrite:
        names = ", ".join(existing)
        raise WritersError(
            f"output already exists ({names}) in {output_dir.resolve()}; "
            "re-run with --overwrite to replace"
        )


def cleanup_partial_outputs(output_dir: Path) -> None:
    """Remove all temporary and known partial output files in the output directory."""
    paths = output_paths(output_dir)
    for key in ("tmp_annotated", "tmp_detections", "tmp_summary"):
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
        ("tmp_annotated", "annotated"),
        ("tmp_detections", "detections"),
        ("tmp_summary", "summary"),
    )
    for tmp_key, _final_key in required_tmps:
        if not paths[tmp_key].is_file():
            raise WritersError(f"missing temporary output before finalize: {paths[tmp_key].name}")

    for tmp_key, final_key in required_tmps:
        os.replace(paths[tmp_key], paths[final_key])

    return {
        "annotated": paths["annotated"],
        "detections": paths["detections"],
        "summary": paths["summary"],
    }
