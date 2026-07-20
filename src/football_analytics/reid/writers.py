"""Atomic output directory helpers for ReID crop extraction."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

MANIFEST_NAME = "crop_manifest.jsonl"
CROPS_DIRNAME = "crops"


class ReIDWritersError(RuntimeError):
    """Raised when ReID crop outputs cannot be prepared or finalized."""


def check_output_collision(output_dir: Path, *, overwrite: bool) -> None:
    directory = output_dir.expanduser().resolve()
    if directory.exists() and not overwrite:
        raise ReIDWritersError(
            f"output already exists: {directory}; re-run with --overwrite to replace"
        )


def create_temp_output_dir(output_dir: Path) -> Path:
    """Create a unique temporary directory beside the final output directory."""
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_crops_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise ReIDWritersError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    (tmp_dir / CROPS_DIRNAME).mkdir(parents=False, exist_ok=False)
    return tmp_dir


def write_manifest_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def validate_manifest_disk_consistency(
    output_dir: Path, rows: Sequence[dict[str, Any]]
) -> None:
    directory = output_dir.expanduser().resolve()
    jpeg_paths = sorted(p for p in directory.rglob("*.jpg") if p.is_file())
    if len(jpeg_paths) != len(rows):
        raise ReIDWritersError(
            f"manifest/JPEG count mismatch: manifest={len(rows)} jpeg={len(jpeg_paths)}"
        )

    crop_ids: set[str] = set()
    rel_paths: set[str] = set()
    for row in rows:
        crop_id = row["crop_id"]
        rel = row["crop_relative_path"]
        if crop_id in crop_ids:
            raise ReIDWritersError(f"duplicate crop_id in manifest: {crop_id}")
        if rel in rel_paths:
            raise ReIDWritersError(f"duplicate crop_relative_path in manifest: {rel}")
        crop_ids.add(crop_id)
        rel_paths.add(rel)
        absolute = directory / rel
        if not absolute.is_file():
            raise ReIDWritersError(f"manifest path missing on disk: {rel}")

    expected = {str((directory / row["crop_relative_path"]).resolve()) for row in rows}
    actual = {str(p.resolve()) for p in jpeg_paths}
    if expected != actual:
        raise ReIDWritersError("manifest JPEG set does not match files on disk")


def cleanup_dir(path: Path | None) -> None:
    if path is None:
        return
    directory = Path(path)
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


def finalize_output_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    """Promote temp_dir to final_dir with backup/restore semantics on overwrite."""
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise ReIDWritersError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise ReIDWritersError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_crops_{final_path.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final_path, backup_path)

        os.rename(temp_path, final_path)

        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=False)
            backup_path = None
    except Exception:
        # Best-effort restore of previous final directory.
        if backup_path is not None and backup_path.exists() and not final_path.exists():
            try:
                os.rename(backup_path, final_path)
                backup_path = None
            except OSError:
                pass
        raise

    # Ensure no leftover tmp/backup siblings with our prefixes remain for this name.
    parent = final_path.parent
    for stray in parent.glob(f"_tmp_reid_crops_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_crops_{final_path.name}_*"):
        cleanup_dir(stray)

    return final_path
