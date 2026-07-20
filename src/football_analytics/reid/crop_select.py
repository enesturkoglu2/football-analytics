"""Load crop-selection config, filter observations, and select crops per track."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from football_analytics.reid.schema import (
    REQUIRED_TRACK_FIELDS,
    ReIDSchemaError,
    build_crop_manifest_row,
    is_finite_number,
    require_finite,
    validate_bbox_xyxy,
)

DEFAULT_CROP_CONFIG = "configs/reid/crop_selection_stage4b.yaml"
EXPECTED_ORDERING = (
    "quality_score_desc",
    "confidence_desc",
    "frame_index_asc",
)


class CropSelectError(RuntimeError):
    """Raised when crop selection inputs or config are invalid."""


def load_crop_selection_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise CropSelectError(f"crop selection config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CropSelectError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CropSelectError(f"crop selection config must be a mapping: {config_path}")
    return validate_crop_selection_config(payload, source=str(config_path))


def validate_crop_selection_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if payload.get("schema_version") != "reid_crop_selection_v1":
        raise CropSelectError(
            f"{source}: schema_version must be 'reid_crop_selection_v1', "
            f"got {payload.get('schema_version')!r}"
        )

    filters = payload.get("filters")
    selection = payload.get("selection")
    if not isinstance(filters, Mapping):
        raise CropSelectError(f"{source}: filters must be a mapping")
    if not isinstance(selection, Mapping):
        raise CropSelectError(f"{source}: selection must be a mapping")

    min_bbox_area = _require_nonnegative_float(filters, "min_bbox_area", source=source)
    min_short_side = _require_nonnegative_float(filters, "min_short_side", source=source)
    min_confidence = _require_nonnegative_float(filters, "min_confidence", source=source)
    if min_confidence > 1.0:
        raise CropSelectError(f"{source}: filters.min_confidence must be <= 1.0")

    max_crops = selection.get("max_crops_per_track")
    if not isinstance(max_crops, int) or isinstance(max_crops, bool) or max_crops < 1:
        raise CropSelectError(
            f"{source}: selection.max_crops_per_track must be an int >= 1"
        )

    min_gap = selection.get("min_frame_gap_within_track")
    if not isinstance(min_gap, int) or isinstance(min_gap, bool) or min_gap < 1:
        raise CropSelectError(
            f"{source}: selection.min_frame_gap_within_track must be an int >= 1"
        )

    if selection.get("quality_score") != "area_times_confidence":
        raise CropSelectError(
            f"{source}: selection.quality_score must be 'area_times_confidence'"
        )
    if selection.get("greedy_frame_gap") is not True:
        raise CropSelectError(f"{source}: selection.greedy_frame_gap must be true")

    ordering = selection.get("ordering")
    if list(ordering) != list(EXPECTED_ORDERING):
        raise CropSelectError(
            f"{source}: selection.ordering must be {list(EXPECTED_ORDERING)}, "
            f"got {ordering!r}"
        )

    return {
        "schema_version": "reid_crop_selection_v1",
        "profile_name": payload.get("profile_name"),
        "filters": {
            "min_bbox_area": min_bbox_area,
            "min_short_side": min_short_side,
            "min_confidence": min_confidence,
            "require_positive_bbox": bool(filters.get("require_positive_bbox", True)),
            "clamp_bbox_to_video": bool(filters.get("clamp_bbox_to_video", True)),
        },
        "selection": {
            "quality_score": "area_times_confidence",
            "max_crops_per_track": max_crops,
            "min_frame_gap_within_track": min_gap,
            "ordering": list(EXPECTED_ORDERING),
            "greedy_frame_gap": True,
        },
        "embedding_eligibility": dict(payload.get("embedding_eligibility") or {}),
        "source": source,
    }


def _require_nonnegative_float(
    mapping: Mapping[str, Any], key: str, *, source: str
) -> float:
    if key not in mapping:
        raise CropSelectError(f"{source}: missing filters.{key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CropSelectError(f"{source}: filters.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise CropSelectError(f"{source}: filters.{key} must be a finite number >= 0")
    return number


def clamp_bbox_xyxy(
    bbox_xyxy: Sequence[float], *, video_width: int, video_height: int
) -> list[float]:
    if video_width <= 0 or video_height <= 0:
        raise CropSelectError("video dimensions must be positive")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = min(max(x1, 0.0), float(video_width))
    x2 = min(max(x2, 0.0), float(video_width))
    y1 = min(max(y1, 0.0), float(video_height))
    y2 = min(max(y2, 0.0), float(video_height))
    return [x1, y1, x2, y2]


def float_bbox_to_int_crop(
    bbox_xyxy: Sequence[float], *, video_width: int, video_height: int
) -> list[int]:
    """Deterministic floor/ceil integer crop box, clamped to video bounds."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    left = int(math.floor(x1))
    top = int(math.floor(y1))
    right = int(math.ceil(x2))
    bottom = int(math.ceil(y2))

    left = min(max(left, 0), video_width)
    right = min(max(right, 0), video_width)
    top = min(max(top, 0), video_height)
    bottom = min(max(bottom, 0), video_height)

    if right <= left or bottom <= top:
        raise CropSelectError(
            f"empty integer crop after clamp: {[left, top, right, bottom]}"
        )
    return [left, top, right, bottom]


def load_track_observations(tracks_path: str | Path) -> list[dict[str, Any]]:
    path = Path(tracks_path).expanduser().resolve()
    if not path.is_file():
        raise CropSelectError(f"tracks JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CropSelectError(
                    f"invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise CropSelectError(
                    f"tracks JSONL line {line_no} must be an object"
                )
            missing = [key for key in REQUIRED_TRACK_FIELDS if key not in obj]
            if missing:
                raise CropSelectError(
                    f"tracks JSONL line {line_no} missing fields: {missing}"
                )
            rows.append(obj)
    return rows


def filter_candidate_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    video_width: int,
    video_height: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    filters = config["filters"]
    reasons = defaultdict(int)
    kept: list[dict[str, Any]] = []

    for raw in observations:
        track_id = raw.get("track_id")
        if track_id is None:
            reasons["null_track_id"] += 1
            continue
        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id <= 0:
            reasons["invalid_track_id"] += 1
            continue

        if raw.get("class_id") != 0 or raw.get("class_name") != "person":
            reasons["non_person_class"] += 1
            continue

        try:
            frame_index = raw["frame_index"]
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index < 0
            ):
                raise ReIDSchemaError("bad frame_index")
            timestamp_sec = require_finite(raw["timestamp_sec"], field="timestamp_sec")
            confidence = require_finite(raw["confidence"], field="confidence")
            bbox = validate_bbox_xyxy(raw["bbox_xyxy"])
        except ReIDSchemaError:
            reasons["non_finite_or_invalid_fields"] += 1
            continue

        x1, y1, x2, y2 = bbox
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            reasons["nonpositive_bbox"] += 1
            continue

        clamped = clamp_bbox_xyxy(
            bbox, video_width=video_width, video_height=video_height
        )
        cw = clamped[2] - clamped[0]
        ch = clamped[3] - clamped[1]
        if cw <= 0 or ch <= 0:
            reasons["empty_after_clamp"] += 1
            continue

        area = cw * ch
        short_side = min(cw, ch)
        if area < float(filters["min_bbox_area"]):
            reasons["area_below_min"] += 1
            continue
        if short_side < float(filters["min_short_side"]):
            reasons["short_side_below_min"] += 1
            continue
        if confidence < float(filters["min_confidence"]):
            reasons["confidence_below_min"] += 1
            continue

        try:
            int_bbox = float_bbox_to_int_crop(
                clamped, video_width=video_width, video_height=video_height
            )
        except CropSelectError:
            reasons["empty_integer_crop"] += 1
            continue

        quality_score = area * confidence
        kept.append(
            {
                "track_id": track_id,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "detection_confidence": confidence,
                "float_bbox_xyxy": clamped,
                "bbox_xyxy": int_bbox,
                "bbox_area": area,
                "short_side": short_side,
                "quality_score": quality_score,
            }
        )

    return kept, dict(reasons)


def select_crops_for_tracks(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    source_video: str,
    track_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    selection = config["selection"]
    max_crops = int(selection["max_crops_per_track"])
    min_gap = int(selection["min_frame_gap_within_track"])

    allowed: set[int] | None = None
    if track_ids is not None:
        allowed = set(int(t) for t in track_ids)

    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        tid = int(cand["track_id"])
        if allowed is not None and tid not in allowed:
            continue
        by_track[tid].append(dict(cand))

    selected_rows: list[dict[str, Any]] = []
    for track_id in sorted(by_track.keys()):
        items = by_track[track_id]
        items.sort(
            key=lambda item: (
                -float(item["quality_score"]),
                -float(item["detection_confidence"]),
                int(item["frame_index"]),
            )
        )

        chosen: list[dict[str, Any]] = []
        chosen_frames: list[int] = []
        for item in items:
            frame_index = int(item["frame_index"])
            if frame_index in chosen_frames:
                continue
            if any(abs(frame_index - prev) < min_gap for prev in chosen_frames):
                continue
            chosen.append(item)
            chosen_frames.append(frame_index)
            if len(chosen) >= max_crops:
                break

        for rank, item in enumerate(chosen, start=1):
            selected_rows.append(
                build_crop_manifest_row(
                    track_id=track_id,
                    frame_index=int(item["frame_index"]),
                    timestamp_sec=float(item["timestamp_sec"]),
                    source_video=source_video,
                    bbox_xyxy=item["bbox_xyxy"],
                    detection_confidence=float(item["detection_confidence"]),
                    quality_score=float(item["quality_score"]),
                    selection_rank=rank,
                )
            )

    selected_rows.sort(key=lambda row: (int(row["track_id"]), int(row["selection_rank"])))
    return selected_rows


def select_crops_from_tracks_file(
    *,
    tracks_path: str | Path,
    config: Mapping[str, Any] | str | Path,
    video_width: int,
    video_height: int,
    source_video: str,
    track_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        config = load_crop_selection_config(config)

    observations = load_track_observations(tracks_path)
    all_track_ids = {
        int(row["track_id"])
        for row in observations
        if isinstance(row.get("track_id"), int) and not isinstance(row.get("track_id"), bool)
    }

    if track_ids is not None:
        requested = [int(t) for t in track_ids]
        missing = sorted(set(requested) - all_track_ids)
        if missing:
            raise CropSelectError(
                f"requested track_id(s) not found in tracks JSONL: {missing}"
            )

    candidates, filter_reasons = filter_candidate_observations(
        observations,
        config=config,
        video_width=video_width,
        video_height=video_height,
    )
    selected = select_crops_for_tracks(
        candidates,
        config=config,
        source_video=source_video,
        track_ids=track_ids,
    )

    if track_ids is not None:
        produced = {int(row["track_id"]) for row in selected}
        empty = sorted(set(int(t) for t in track_ids) - produced)
        if empty:
            raise CropSelectError(
                f"requested track_id(s) produced no crops after filters: {empty}"
            )

    tracks_examined = (
        set(int(t) for t in track_ids) if track_ids is not None else all_track_ids
    )
    return {
        "observations_read": len(observations),
        "tracks_examined": len(tracks_examined),
        "eligible_observations": len(candidates),
        "tracks_with_crops": len({int(r["track_id"]) for r in selected}),
        "crops_selected": len(selected),
        "filter_reasons": filter_reasons,
        "selected": selected,
        "config": config,
    }
