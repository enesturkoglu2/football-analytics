"""ReID crop quality and tracking-bbox contamination measurements (Stage 5A2A).

Produces measurement-only signals. Does not apply quality thresholds, delete
crops, modify embeddings, or run linking / identity fusion.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.crop_select import clamp_bbox_xyxy, float_bbox_to_int_crop
from football_analytics.reid.embedding import resolve_crop_path
from football_analytics.reid.schema import (
    REQUIRED_TRACK_FIELDS,
    ReIDSchemaError,
    is_finite_number,
    require_finite,
    validate_bbox_xyxy,
    validate_manifest_row,
)
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

CROP_QUALITY_SCHEMA = "reid_crop_quality_signal_v1"
TRACK_QUALITY_SCHEMA = "reid_track_quality_summary_v1"
QUALITY_SUMMARY_SCHEMA = "reid_quality_summary_v1"

CROP_QUALITY_NAME = "crop_quality_signals.jsonl"
TRACK_QUALITY_NAME = "track_quality_summary.jsonl"
QUALITY_SUMMARY_NAME = "quality_summary.json"

DARK_PIXEL_MAX = 20
BRIGHT_PIXEL_MIN = 235


class QualityError(RuntimeError):
    """Raised when crop quality analysis inputs or outputs are invalid."""


def _reject_non_finite_json(value: str) -> None:
    raise QualityError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise QualityError(f"{field} must be finite, got {value!r}")
    return number


def _ensure_ratio(value: float, *, field: str) -> float:
    number = _ensure_finite_float(value, field=field)
    if number < 0.0 or number > 1.0:
        raise QualityError(f"{field} must be in [0, 1], got {number}")
    return number


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QualityError(f"{field} must be a positive int, got {value!r}")
    return value


def _require_nonneg_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise QualityError(f"{field} must be a non-negative int, got {value!r}")
    return value


def _percentile_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p5": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray([_ensure_finite_float(v, field="stat") for v in values], dtype=np.float64)
    qs = {
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }
    for key, number in qs.items():
        if not math.isfinite(number):
            raise QualityError(f"non-finite percentile {key}")
    return qs


def create_temp_quality_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_quality_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise QualityError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_quality_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise QualityError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise QualityError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_quality_{final_path.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final_path, backup_path)

        os.rename(temp_path, final_path)

        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=False)
            backup_path = None
    except Exception:
        if backup_path is not None and backup_path.exists() and not final_path.exists():
            try:
                os.rename(backup_path, final_path)
                backup_path = None
            except OSError:
                pass
        raise

    parent = final_path.parent
    for stray in parent.glob(f"_tmp_reid_quality_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_quality_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def load_crop_manifest_for_quality(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise QualityError(f"crop manifest not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualityError(f"could not read crop manifest: {path}: {exc}") from exc
    if not text.strip():
        raise QualityError("crop manifest is empty")

    rows: list[dict[str, Any]] = []
    crop_ids: set[str] = set()
    rel_paths: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_json)
        except QualityError:
            raise
        except json.JSONDecodeError as exc:
            raise QualityError(
                f"invalid JSON on crop manifest line {line_no}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise QualityError(f"manifest line {line_no} must be a JSON object")
        try:
            validate_manifest_row(payload)
        except ReIDSchemaError as exc:
            raise QualityError(f"manifest line {line_no}: {exc}") from exc

        track_id = _require_positive_int(payload["track_id"], field="track_id")
        frame_index = _require_nonneg_int(payload["frame_index"], field="frame_index")
        selection_rank = _require_positive_int(
            payload["selection_rank"], field="selection_rank"
        )
        crop_id = payload["crop_id"]
        rel = payload["crop_relative_path"]
        if not isinstance(crop_id, str) or not crop_id:
            raise QualityError(f"manifest line {line_no}: crop_id must be non-empty str")
        if crop_id in crop_ids:
            raise QualityError(f"duplicate crop_id in manifest: {crop_id}")
        if not isinstance(rel, str) or not rel:
            raise QualityError(
                f"manifest line {line_no}: crop_relative_path must be non-empty str"
            )
        if rel in rel_paths:
            raise QualityError(f"duplicate crop_relative_path in manifest: {rel}")

        bbox = payload["bbox_xyxy"]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
            raise QualityError(
                f"manifest line {line_no}: bbox_xyxy values must be ints"
            )
        left, top, right, bottom = [int(v) for v in bbox]
        if right <= left or bottom <= top:
            raise QualityError(f"manifest line {line_no}: invalid bbox size")
        width = right - left
        height = bottom - top
        if float(payload["bbox_width"]) != float(width):
            raise QualityError(f"manifest line {line_no}: bbox_width mismatch")
        if float(payload["bbox_height"]) != float(height):
            raise QualityError(f"manifest line {line_no}: bbox_height mismatch")
        if float(payload["bbox_area"]) != float(width * height):
            raise QualityError(f"manifest line {line_no}: bbox_area mismatch")

        crop_ids.add(crop_id)
        rel_paths.add(rel)
        rows.append(
            {
                **payload,
                "track_id": track_id,
                "frame_index": frame_index,
                "selection_rank": selection_rank,
                "bbox_xyxy": [left, top, right, bottom],
            }
        )

    if not rows:
        raise QualityError("crop manifest is empty")
    return rows


def load_person_track_observations(
    tracks_path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(tracks_path).expanduser().resolve()
    if not path.is_file():
        raise QualityError(f"tracks JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text, parse_constant=_reject_non_finite_json)
            except QualityError:
                raise
            except json.JSONDecodeError as exc:
                raise QualityError(
                    f"invalid JSON on tracks line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise QualityError(f"tracks line {line_no} must be a JSON object")
            missing = [key for key in REQUIRED_TRACK_FIELDS if key not in obj]
            if missing:
                raise QualityError(
                    f"tracks line {line_no} missing fields: {missing}"
                )

            track_id = obj.get("track_id")
            frame_index = obj.get("frame_index")
            if (
                not isinstance(track_id, int)
                or isinstance(track_id, bool)
                or track_id <= 0
            ):
                raise QualityError(
                    f"tracks line {line_no}: invalid track_id {track_id!r}"
                )
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index < 0
            ):
                raise QualityError(
                    f"tracks line {line_no}: invalid frame_index {frame_index!r}"
                )

            if obj.get("class_id") != 0 or obj.get("class_name") != "person":
                continue

            try:
                bbox = validate_bbox_xyxy(obj["bbox_xyxy"])
            except ReIDSchemaError as exc:
                raise QualityError(f"tracks line {line_no}: {exc}") from exc
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise QualityError(
                    f"tracks line {line_no}: bbox must have positive area"
                )
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if not math.isfinite(area) or area <= 0:
                raise QualityError(f"tracks line {line_no}: invalid bbox area")

            key = (track_id, frame_index)
            if key in seen_keys:
                raise QualityError(
                    f"duplicate track_id+frame_index observation: {key}"
                )
            seen_keys.add(key)
            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "bbox_xyxy": bbox,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": require_finite(
                        obj["confidence"], field="confidence"
                    ),
                    "timestamp_sec": require_finite(
                        obj["timestamp_sec"], field="timestamp_sec"
                    ),
                }
            )
    if not rows:
        raise QualityError("tracks JSONL has no valid person observations")
    return rows


def infer_frame_size(
    *,
    observations: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Infer frame size from observation/manifest extents (no video input)."""
    max_right = 0
    max_bottom = 0
    for row in observations:
        x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
        max_right = max(max_right, int(math.ceil(x2)), int(math.floor(x1)) + 1)
        max_bottom = max(max_bottom, int(math.ceil(y2)), int(math.floor(y1)) + 1)
    for row in manifest_rows:
        left, top, right, bottom = [int(v) for v in row["bbox_xyxy"]]
        max_right = max(max_right, right)
        max_bottom = max(max_bottom, bottom)
    if max_right <= 0 or max_bottom <= 0:
        raise QualityError("could not infer positive frame dimensions")
    return int(max_right), int(max_bottom)


def index_observations_by_frame(
    observations: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(dict(row))
    return dict(by_frame)


def verify_target_observation(
    *,
    manifest_row: Mapping[str, Any],
    frame_observations: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    track_id = int(manifest_row["track_id"])
    frame_index = int(manifest_row["frame_index"])
    matches = [
        row
        for row in frame_observations
        if int(row["track_id"]) == track_id and int(row["frame_index"]) == frame_index
    ]
    if not matches:
        raise QualityError(
            f"missing target observation for track_id={track_id} "
            f"frame_index={frame_index}"
        )
    if len(matches) != 1:
        raise QualityError(
            f"duplicate target observation for track_id={track_id} "
            f"frame_index={frame_index}"
        )
    obs = matches[0]
    clamped = clamp_bbox_xyxy(
        obs["bbox_xyxy"], video_width=frame_width, video_height=frame_height
    )
    int_bbox = float_bbox_to_int_crop(
        clamped, video_width=frame_width, video_height=frame_height
    )
    expected = [int(v) for v in manifest_row["bbox_xyxy"]]
    if int_bbox != expected:
        raise QualityError(
            f"stale/tampered input: track_id={track_id} frame_index={frame_index} "
            f"manifest bbox {expected} != converted observation bbox {int_bbox}"
        )
    return obs


def compute_image_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        raise QualityError("decoded crop image is empty")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise QualityError("decoded crop must be a BGR image")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        raise QualityError("grayscale histogram is empty")
    probs = hist / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    dark = float(np.mean(gray <= DARK_PIXEL_MAX))
    bright = float(np.mean(gray >= BRIGHT_PIXEL_MIN))
    return {
        "grayscale_mean": _ensure_finite_float(mean, field="grayscale_mean"),
        "grayscale_std": _ensure_finite_float(std, field="grayscale_std"),
        "laplacian_variance": _ensure_finite_float(lap_var, field="laplacian_variance"),
        "grayscale_entropy_bits": _ensure_finite_float(
            entropy, field="grayscale_entropy_bits"
        ),
        "dark_pixel_ratio": _ensure_ratio(dark, field="dark_pixel_ratio"),
        "bright_pixel_ratio": _ensure_ratio(bright, field="bright_pixel_ratio"),
    }


def compute_edge_contacts(
    bbox_xyxy: Sequence[int], *, frame_width: int, frame_height: int
) -> dict[str, Any]:
    left, top, right, bottom = [int(v) for v in bbox_xyxy]
    touches_left = left == 0
    touches_top = top == 0
    touches_right = right == frame_width
    touches_bottom = bottom == frame_height
    count = int(touches_left) + int(touches_top) + int(touches_right) + int(touches_bottom)
    return {
        "touches_left_edge": touches_left,
        "touches_top_edge": touches_top,
        "touches_right_edge": touches_right,
        "touches_bottom_edge": touches_bottom,
        "frame_edge_contact_count": count,
    }


def _intersection_area(a: Sequence[int], b: Sequence[int]) -> float:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = float((a[2] - a[0]) * (a[3] - a[1]))
    area_b = float((b[2] - b[0]) * (b[3] - b[1]))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _center_inside(other: Sequence[int], target: Sequence[int]) -> bool:
    cx = 0.5 * (float(other[0]) + float(other[2]))
    cy = 0.5 * (float(other[1]) + float(other[3]))
    return (
        float(target[0]) <= cx < float(target[2])
        and float(target[1]) <= cy < float(target[3])
    )


def compute_tracking_bbox_contamination(
    *,
    target_bbox: Sequence[int],
    target_track_id: int,
    frame_observations: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    target = [int(v) for v in target_bbox]
    target_area = float((target[2] - target[0]) * (target[3] - target[1]))
    if target_area <= 0:
        raise QualityError("target crop area must be positive")

    others: list[list[int]] = []
    for row in frame_observations:
        if int(row["track_id"]) == int(target_track_id):
            continue
        clamped = clamp_bbox_xyxy(
            row["bbox_xyxy"], video_width=frame_width, video_height=frame_height
        )
        try:
            other_int = float_bbox_to_int_crop(
                clamped, video_width=frame_width, video_height=frame_height
            )
        except Exception as exc:  # noqa: BLE001 - convert selection errors
            raise QualityError(
                f"could not convert other-person bbox for contamination: {exc}"
            ) from exc
        others.append(other_int)

    overlap_boxes: list[list[int]] = []
    coverages: list[float] = []
    ious: list[float] = []
    center_inside = 0
    for other in others:
        inter = _intersection_area(target, other)
        if inter > 0:
            overlap_boxes.append(other)
            coverages.append(float(inter / target_area))
            ious.append(_iou(target, other))
        if _center_inside(other, target):
            center_inside += 1

    max_coverage = max(coverages) if coverages else 0.0
    max_iou = max(ious) if ious else 0.0

    # Exact union of intersections inside the target crop via boolean mask.
    tw = target[2] - target[0]
    th = target[3] - target[1]
    if overlap_boxes:
        mask = np.zeros((th, tw), dtype=bool)
        for other in overlap_boxes:
            x1 = max(target[0], other[0])
            y1 = max(target[1], other[1])
            x2 = min(target[2], other[2])
            y2 = min(target[3], other[3])
            if x2 > x1 and y2 > y1:
                mask[y1 - target[1] : y2 - target[1], x1 - target[0] : x2 - target[0]] = True
        union_area = float(np.count_nonzero(mask))
        union_coverage = float(union_area / target_area)
    else:
        union_coverage = 0.0

    return {
        "other_person_observation_count_in_frame": len(others),
        "other_person_overlap_count": len(overlap_boxes),
        "other_person_center_inside_count": int(center_inside),
        "max_other_person_crop_coverage": _ensure_ratio(
            max_coverage, field="max_other_person_crop_coverage"
        ),
        "union_other_person_crop_coverage": _ensure_ratio(
            union_coverage, field="union_other_person_crop_coverage"
        ),
        "max_other_person_iou": _ensure_ratio(max_iou, field="max_other_person_iou"),
    }


def analyze_one_crop(
    *,
    manifest_row: Mapping[str, Any],
    manifest_dir: Path,
    frame_observations: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    crop_path = resolve_crop_path(manifest_dir, manifest_row["crop_relative_path"])
    # Re-bind EmbeddingError from resolve into QualityError for CLI clarity.
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise QualityError(f"could not decode crop JPEG: {crop_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise QualityError(f"decoded crop has non-positive size: {crop_path}")

    left, top, right, bottom = [int(v) for v in manifest_row["bbox_xyxy"]]
    expected_w = right - left
    expected_h = bottom - top
    if width != expected_w or height != expected_h:
        raise QualityError(
            f"decoded size {width}x{height} != manifest bbox size "
            f"{expected_w}x{expected_h} for {manifest_row['crop_id']}"
        )

    verify_target_observation(
        manifest_row=manifest_row,
        frame_observations=frame_observations,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    crop_area = float(width * height)
    frame_area = float(frame_width * frame_height)
    metrics = compute_image_metrics(image)
    edges = compute_edge_contacts(
        [left, top, right, bottom],
        frame_width=frame_width,
        frame_height=frame_height,
    )
    contamination = compute_tracking_bbox_contamination(
        target_bbox=[left, top, right, bottom],
        target_track_id=int(manifest_row["track_id"]),
        frame_observations=frame_observations,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    return {
        "crop_id": manifest_row["crop_id"],
        "track_id": int(manifest_row["track_id"]),
        "frame_index": int(manifest_row["frame_index"]),
        "selection_rank": int(manifest_row["selection_rank"]),
        "crop_relative_path": manifest_row["crop_relative_path"],
        "target_observation_verified": True,
        "crop_width": int(width),
        "crop_height": int(height),
        "crop_area": crop_area,
        "aspect_ratio": _ensure_finite_float(width / height, field="aspect_ratio"),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "bbox_area_frame_ratio": _ensure_ratio(
            crop_area / frame_area, field="bbox_area_frame_ratio"
        ),
        **edges,
        **metrics,
        **contamination,
        "quality_threshold": None,
        "contamination_threshold": None,
        "quality_decision": "measurement_only",
        "automatic_exclusion_applied": False,
        "contamination_source": "tracking_bbox_overlap",
        "schema_version": CROP_QUALITY_SCHEMA,
    }


def build_track_quality_summaries(
    crop_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_track: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_rows:
        by_track[int(row["track_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for track_id in sorted(by_track.keys()):
        rows = by_track[track_id]
        laps = [float(r["laplacian_variance"]) for r in rows]
        means = [float(r["grayscale_mean"]) for r in rows]
        stds = [float(r["grayscale_std"]) for r in rows]
        ents = [float(r["grayscale_entropy_bits"]) for r in rows]
        unions = [float(r["union_other_person_crop_coverage"]) for r in rows]
        max_covs = [float(r["max_other_person_crop_coverage"]) for r in rows]
        crop_ids = [str(r["crop_id"]) for r in rows]
        if len(crop_ids) != len(rows):
            raise QualityError("crop_ids length mismatch")
        summaries.append(
            {
                "track_id": int(track_id),
                "crop_count": len(rows),
                "crop_ids": crop_ids,
                "laplacian_variance_min": float(min(laps)),
                "laplacian_variance_median": float(np.median(np.asarray(laps))),
                "laplacian_variance_max": float(max(laps)),
                "grayscale_mean_mean": float(np.mean(np.asarray(means))),
                "grayscale_std_mean": float(np.mean(np.asarray(stds))),
                "grayscale_entropy_bits_mean": float(np.mean(np.asarray(ents))),
                "frame_edge_contact_crop_count": sum(
                    1 for r in rows if int(r["frame_edge_contact_count"]) > 0
                ),
                "other_person_overlap_crop_count": sum(
                    1 for r in rows if int(r["other_person_overlap_count"]) > 0
                ),
                "other_person_center_inside_crop_count": sum(
                    1 for r in rows if int(r["other_person_center_inside_count"]) > 0
                ),
                "max_other_person_crop_coverage": float(max(max_covs)),
                "mean_union_other_person_crop_coverage": float(
                    np.mean(np.asarray(unions))
                ),
                "automatic_excluded_crop_count": 0,
                "quality_threshold": None,
                "contamination_threshold": None,
                "schema_version": TRACK_QUALITY_SCHEMA,
            }
        )
    return summaries


def build_quality_summary(
    *,
    crop_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    crop_manifest: Path,
    tracks_jsonl: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    laps = [float(r["laplacian_variance"]) for r in crop_rows]
    ents = [float(r["grayscale_entropy_bits"]) for r in crop_rows]
    unions = [float(r["union_other_person_crop_coverage"]) for r in crop_rows]
    summary = {
        "status": "ok",
        "crop_count": len(crop_rows),
        "track_count": len(track_rows),
        "source_crop_manifest": str(crop_manifest.expanduser().resolve()),
        "source_tracks_jsonl": str(tracks_jsonl.expanduser().resolve()),
        "crops_touching_any_frame_edge": sum(
            1 for r in crop_rows if int(r["frame_edge_contact_count"]) > 0
        ),
        "crops_with_other_person_overlap": sum(
            1 for r in crop_rows if int(r["other_person_overlap_count"]) > 0
        ),
        "crops_with_other_person_center_inside": sum(
            1 for r in crop_rows if int(r["other_person_center_inside_count"]) > 0
        ),
        "laplacian_variance": _percentile_stats(laps),
        "grayscale_entropy_bits": _percentile_stats(ents),
        "union_other_person_crop_coverage": _percentile_stats(unions),
        "quality_threshold": None,
        "contamination_threshold": None,
        "composite_quality_score_created": False,
        "automatic_exclusion_performed": False,
        "embedding_aggregation_modified": False,
        "crop_files_modified": False,
        "contamination_source": "tracking_bbox_overlap",
        "contamination_limitations": (
            "tracking observations can miss untracked or undetected people"
        ),
        "elapsed_sec": _ensure_finite_float(elapsed_sec, field="elapsed_sec"),
        "schema_version": QUALITY_SUMMARY_SCHEMA,
    }
    return summary


def run_analyze_reid_crop_quality(
    *,
    crop_manifest: str | Path,
    tracks: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(crop_manifest).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    manifest_dir = manifest_path.parent

    check_output_collision(final_dir, overwrite=overwrite)

    # Map EmbeddingError from resolve_crop_path into QualityError for callers.
    from football_analytics.reid.embedding import EmbeddingError

    temp_dir: Path | None = None
    try:
        manifest_rows = load_crop_manifest_for_quality(manifest_path)
        observations = load_person_track_observations(tracks_path)
        frame_width, frame_height = infer_frame_size(
            observations=observations, manifest_rows=manifest_rows
        )
        by_frame = index_observations_by_frame(observations)

        crop_signals: list[dict[str, Any]] = []
        for row in manifest_rows:
            frame_index = int(row["frame_index"])
            frame_obs = by_frame.get(frame_index, [])
            try:
                signal = analyze_one_crop(
                    manifest_row=row,
                    manifest_dir=manifest_dir,
                    frame_observations=frame_obs,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            except EmbeddingError as exc:
                raise QualityError(str(exc)) from exc
            crop_signals.append(signal)

        track_summaries = build_track_quality_summaries(crop_signals)
        elapsed = time.perf_counter() - started
        summary = build_quality_summary(
            crop_rows=crop_signals,
            track_rows=track_summaries,
            crop_manifest=manifest_path,
            tracks_jsonl=tracks_path,
            elapsed_sec=elapsed,
        )

        temp_dir = create_temp_quality_dir(final_dir)
        write_manifest_jsonl(temp_dir / CROP_QUALITY_NAME, crop_signals)
        write_manifest_jsonl(temp_dir / TRACK_QUALITY_NAME, track_summaries)
        (temp_dir / QUALITY_SUMMARY_NAME).write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        finalized = finalize_quality_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "crop_quality_path": str(finalized / CROP_QUALITY_NAME),
        "track_quality_path": str(finalized / TRACK_QUALITY_NAME),
        "summary_path": str(finalized / QUALITY_SUMMARY_NAME),
        "crop_count": summary["crop_count"],
        "track_count": summary["track_count"],
        "crops_touching_any_frame_edge": summary["crops_touching_any_frame_edge"],
        "crops_with_other_person_overlap": summary["crops_with_other_person_overlap"],
        "crops_with_other_person_center_inside": summary[
            "crops_with_other_person_center_inside"
        ],
        "laplacian_variance": summary["laplacian_variance"],
        "union_other_person_crop_coverage": summary[
            "union_other_person_crop_coverage"
        ],
        "quality_threshold": None,
        "contamination_threshold": None,
        "automatic_exclusion_performed": False,
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
    }
