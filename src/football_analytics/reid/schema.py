"""Schema constants and validation for ReID crop manifests."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

CROP_MANIFEST_SCHEMA_VERSION = "reid_crop_manifest_v1"

REQUIRED_TRACK_FIELDS = (
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_xyxy",
)

MANIFEST_FIELDS = (
    "crop_id",
    "track_id",
    "frame_index",
    "timestamp_sec",
    "source_video",
    "bbox_xyxy",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "short_side",
    "detection_confidence",
    "quality_score",
    "crop_relative_path",
    "selection_rank",
    "schema_version",
)


class ReIDSchemaError(ValueError):
    """Raised when a ReID schema constraint is violated."""


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def require_finite(value: Any, *, field: str) -> float:
    if not is_finite_number(value):
        raise ReIDSchemaError(f"{field} must be a finite number, got {value!r}")
    return float(value)


def validate_bbox_xyxy(bbox: Any, *, field: str = "bbox_xyxy") -> list[float]:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        raise ReIDSchemaError(f"{field} must be a sequence of four numbers")
    if len(bbox) != 4:
        raise ReIDSchemaError(f"{field} must contain exactly four numbers")
    values = [require_finite(v, field=f"{field}[{i}]") for i, v in enumerate(bbox)]
    return values


def make_crop_id(*, track_id: int, frame_index: int, selection_rank: int) -> str:
    return f"track_{track_id}_frame_{frame_index}_rank_{selection_rank}"


def make_crop_relative_path(
    *, track_id: int, frame_index: int, selection_rank: int
) -> str:
    return f"crops/track_{track_id}/crop_{frame_index}_{selection_rank}.jpg"


def build_crop_manifest_row(
    *,
    track_id: int,
    frame_index: int,
    timestamp_sec: float,
    source_video: str,
    bbox_xyxy: Sequence[int],
    detection_confidence: float,
    quality_score: float,
    selection_rank: int,
) -> dict[str, Any]:
    if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id <= 0:
        raise ReIDSchemaError(f"track_id must be a positive int, got {track_id!r}")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ReIDSchemaError(f"frame_index must be a non-negative int, got {frame_index!r}")
    if not isinstance(selection_rank, int) or isinstance(selection_rank, bool) or selection_rank < 1:
        raise ReIDSchemaError(
            f"selection_rank must be an int >= 1, got {selection_rank!r}"
        )

    left, top, right, bottom = [int(v) for v in bbox_xyxy]
    if right <= left or bottom <= top:
        raise ReIDSchemaError(
            f"bbox_xyxy must have positive size, got {[left, top, right, bottom]}"
        )

    width = right - left
    height = bottom - top
    area = float(width * height)
    short_side = float(min(width, height))
    conf = require_finite(detection_confidence, field="detection_confidence")
    quality = require_finite(quality_score, field="quality_score")
    ts = require_finite(timestamp_sec, field="timestamp_sec")

    row = {
        "crop_id": make_crop_id(
            track_id=track_id, frame_index=frame_index, selection_rank=selection_rank
        ),
        "track_id": track_id,
        "frame_index": frame_index,
        "timestamp_sec": ts,
        "source_video": str(source_video),
        "bbox_xyxy": [left, top, right, bottom],
        "bbox_width": float(width),
        "bbox_height": float(height),
        "bbox_area": area,
        "short_side": short_side,
        "detection_confidence": conf,
        "quality_score": quality,
        "crop_relative_path": make_crop_relative_path(
            track_id=track_id, frame_index=frame_index, selection_rank=selection_rank
        ),
        "selection_rank": selection_rank,
        "schema_version": CROP_MANIFEST_SCHEMA_VERSION,
    }
    validate_manifest_row(row)
    return row


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    missing = [key for key in MANIFEST_FIELDS if key not in row]
    if missing:
        raise ReIDSchemaError(f"manifest row missing fields: {missing}")
    if row["schema_version"] != CROP_MANIFEST_SCHEMA_VERSION:
        raise ReIDSchemaError(
            f"unexpected schema_version {row['schema_version']!r}; "
            f"expected {CROP_MANIFEST_SCHEMA_VERSION!r}"
        )
    for key in (
        "timestamp_sec",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "short_side",
        "detection_confidence",
        "quality_score",
    ):
        require_finite(row[key], field=key)
    validate_bbox_xyxy(row["bbox_xyxy"])
    if float(row["bbox_width"]) <= 0 or float(row["bbox_height"]) <= 0:
        raise ReIDSchemaError("bbox_width and bbox_height must be positive")
