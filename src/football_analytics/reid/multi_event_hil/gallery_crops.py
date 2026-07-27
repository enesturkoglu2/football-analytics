"""Extract match-specific enrollment crop *candidates* (not auto-gallery members)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.crop_extract import crop_frame_region, write_crop_jpeg
from football_analytics.reid.hil.common import sha256_file


def _clamp_bbox(bbox: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(width, int(np.ceil(x2)))
    bottom = min(height, int(np.ceil(y2)))
    if right <= left or bottom <= top:
        raise ValueError(f"empty clamped bbox from {bbox}")
    return [left, top, right, bottom]


def _pick_indices(n: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    # start / early-mid / mid / late-mid / end — diversity without inventing counts
    raw = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    out: list[int] = []
    seen: set[int] = set()
    for i in raw:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_enrollment_crop_candidates(
    *,
    video_path: Path,
    mapping_row: Mapping[str, Any],
    segment_id: str,
    output_dir: Path,
    video_sha256: str,
) -> dict[str, Any]:
    """Decode selected frames and write JPEG crop candidates for human gallery review."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = list(mapping_row.get("bbox_per_observation") or [])
    if not obs:
        raise RuntimeError("mapping row has no bbox_per_observation")

    idxs = _pick_indices(len(obs))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    candidates: list[dict[str, Any]] = []
    try:
        for rank, idx in enumerate(idxs, start=1):
            item = obs[idx]
            frame_index = int(item["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            bbox = _clamp_bbox(item["bbox_xyxy"], width, height)
            crop = crop_frame_region(frame, bbox)
            crop_id = f"mehil_crop_{segment_id}_{frame_index:06d}"
            rel = f"{crop_id}.jpg"
            abs_path = output_dir / rel
            write_crop_jpeg(abs_path, crop)
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            candidates.append(
                {
                    "crop_id": crop_id,
                    "segment_id": segment_id,
                    "raw_track_id": str(mapping_row["raw_external_track_id"]),
                    "external_candidate_code": mapping_row.get("external_candidate_code"),
                    "frame_index": frame_index,
                    "bbox_xyxy": bbox,
                    "crop_path": str(abs_path),
                    "crop_sha256": sha256_file(abs_path),
                    "view_hint": {
                        1: "start",
                        2: "early",
                        3: "middle",
                        4: "late",
                        5: "end",
                    }.get(rank, "sample"),
                    "scale_proxy_area": area,
                    "automatic_gallery_member": False,
                    "gallery_approved": False,
                    "provenance": {
                        "source": "enrollment_segment_observation",
                        "video_sha256": video_sha256,
                        "not_from_development_gallery": True,
                        "not_from_holdout": True,
                    },
                }
            )
    finally:
        cap.release()

    manifest = {
        "schema_version": "match_specific_enrollment_crop_candidates_v1",
        "segment_id": segment_id,
        "raw_track_id": str(mapping_row["raw_external_track_id"]),
        "video_sha256": video_sha256,
        "candidate_count": len(candidates),
        "note": (
            "Candidates only. Membership requires explicit gallery approval. "
            "Count is evidence-driven from track observations, not a fixed quota."
        ),
        "automatic_gallery_expansion": False,
        "candidates": candidates,
    }
    man_path = output_dir / "enrollment_crop_candidates.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    manifest["manifest_sha256"] = sha256_file(man_path)
    return manifest
