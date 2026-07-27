"""Deterministic review media cache (decode-only; no inference)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.common import sha256_bytes, sha256_file
from football_analytics.reid.hil_ui.package import ReviewPackageError, assert_path_not_writable_to_frozen

CACHE_MANIFEST_NAME = "media_cache_manifest.jsonl"
GENERATOR_ID = "hil_b_opencv_frame_decode_v1"


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ReviewPackageError("opencv required for media decode in UI env") from exc
    return cv2


def append_cache_record(cache_root: Path, record: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / CACHE_MANIFEST_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def decode_frame_to_cache(
    *,
    video_path: Path,
    video_sha256: str,
    frame_index: int,
    cache_root: Path,
    read_only_roots: list[Path],
    overlay_bboxes: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Decode one frame into cache_root; never write into frozen source roots."""
    assert_path_not_writable_to_frozen(cache_root, read_only_roots)
    cv2 = _require_cv2()
    cache_root.mkdir(parents=True, exist_ok=True)
    out_name = f"frame_{frame_index:06d}"
    if overlay_bboxes:
        out_name += "_overlay"
    out_path = cache_root / f"{out_name}.png"
    if out_path.is_file():
        digest = sha256_file(out_path)
        return {
            "path": str(out_path),
            "sha256": digest,
            "frame_index": frame_index,
            "cache_hit": True,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ReviewPackageError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ReviewPackageError(f"failed to decode frame {frame_index} from {video_path}")
        if overlay_bboxes:
            for xyxy in overlay_bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        ok_write, buf = cv2.imencode(".png", frame)
        if not ok_write:
            raise ReviewPackageError("png encode failed")
        data = buf.tobytes()
        out_path.write_bytes(data)
    finally:
        cap.release()

    digest = sha256_bytes(out_path.read_bytes())
    record = {
        "schema_version": "hil_b_media_cache_record_v1",
        "source_video_path": str(video_path),
        "source_sha256": video_sha256,
        "frame_index": frame_index,
        "bbox": overlay_bboxes,
        "generation_tool": GENERATOR_ID,
        "output_path": str(out_path),
        "output_sha256": digest,
    }
    append_cache_record(cache_root, record)
    return {
        "path": str(out_path),
        "sha256": digest,
        "frame_index": frame_index,
        "cache_hit": False,
        "record": record,
    }


def read_cache_manifest(cache_root: Path) -> list[dict[str, Any]]:
    path = cache_root / CACHE_MANIFEST_NAME
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
