"""Stage 5C-A jersey visibility/readability measurement baseline.

The analyzer reads existing selected-crop provenance only. It does not open
video, extract or modify crops, run recognition, classify readability, or
create a composite score.
"""

from __future__ import annotations

import hashlib
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
import yaml

from football_analytics.reid.segments import canonical_observation_json
from football_analytics.reid.writers import check_output_collision, cleanup_dir

CONFIG_SCHEMA = "reid_jersey_visibility_config_v1"
CROP_SIGNAL_SCHEMA = "reid_jersey_visibility_crop_signal_v1"
SEGMENT_SUMMARY_SCHEMA = "reid_jersey_visibility_segment_summary_v1"
SUMMARY_SCHEMA = "reid_jersey_visibility_summary_v1"

CROP_SIGNALS_NAME = "jersey_visibility_crop_signals.jsonl"
SEGMENT_SUMMARY_NAME = "jersey_visibility_segment_summary.jsonl"
SUMMARY_NAME = "jersey_visibility_summary.json"
OUTPUT_NAMES = (CROP_SIGNALS_NAME, SEGMENT_SUMMARY_NAME, SUMMARY_NAME)

SEGMENT_FILES = (
    "track_segments.jsonl",
    "segment_observations.jsonl",
    "unassigned_observations.jsonl",
    "segment_view_summary.json",
)
REGRESSION_FILES = (
    "segment_crop_manifest.jsonl",
    "segment_embedding_index.jsonl",
    "baseline_to_segment_replacement.jsonl",
    "segmented_reid_regression_summary.json",
)
SEGMENT_KINDS = {
    "manual_split_segment",
    "no_split_control",
    "preserved_full_track",
}
REPRESENTATION_SOURCES = {
    "recomputed_manual_segment",
    "reused_baseline_raw_track_embedding",
    "no_baseline_embedding",
}


class JerseyVisibilityError(RuntimeError):
    """Raised when Stage 5C-A inputs, measurements, or outputs are invalid."""


def _reject_non_finite(value: str) -> None:
    raise JerseyVisibilityError(f"NaN/Infinity forbidden in JSON: {value}")


def _require_bool(
    value: Any, *, field: str, expected: bool | None = None
) -> bool:
    if not isinstance(value, bool):
        raise JerseyVisibilityError(f"{field} must be bool, got {value!r}")
    if expected is not None and value is not expected:
        raise JerseyVisibilityError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise JerseyVisibilityError(
            f"{field} must be int >= {minimum}, got {value!r}"
        )
    return value


def _finite(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise JerseyVisibilityError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise JerseyVisibilityError(f"{field} must be finite, got {value!r}")
    return result


def _ratio(value: Any, *, field: str) -> float:
    result = _finite(value, field=field)
    if result < 0.0 or result > 1.0:
        raise JerseyVisibilityError(f"{field} must be in [0, 1]")
    return result


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JerseyVisibilityError(f"JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite)
        except json.JSONDecodeError as exc:
            raise JerseyVisibilityError(
                f"invalid JSON on {path.name} line {line_no}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise JerseyVisibilityError(
                f"{path.name} line {line_no} must be an object"
            )
        rows.append(row)
    if not rows and not allow_empty:
        raise JerseyVisibilityError(f"JSONL is empty: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JerseyVisibilityError(f"JSON not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite
        )
    except json.JSONDecodeError as exc:
        raise JerseyVisibilityError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JerseyVisibilityError(f"{path} must contain an object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise JerseyVisibilityError(f"{key} must be a mapping")
    return value


def validate_jersey_visibility_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JerseyVisibilityError(f"{source}: config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise JerseyVisibilityError(f"{source}: invalid schema_version")
    if payload.get("stage_status") != "implementation_measurement_baseline":
        raise JerseyVisibilityError(f"{source}: invalid stage_status")

    input_cfg = _mapping(payload, "input")
    if input_cfg.get("entity_id") != "segment_id":
        raise JerseyVisibilityError(f"{source}: input.entity_id must be segment_id")
    expected_input = {
        "use_existing_selected_crops_only": True,
        "allow_video_input": False,
        "allow_new_crop_extraction": False,
        "include_recomputed_manual_segment_crops": True,
        "include_reused_baseline_crop_provenance": True,
        "include_retired_mixed_parent_entities": False,
        "ambiguous_observations_allowed": False,
        "unassigned_observations_allowed": False,
    }
    for key, expected in expected_input.items():
        _require_bool(
            input_cfg.get(key),
            field=f"{source}.input.{key}",
            expected=expected,
        )

    roi = _mapping(payload, "number_search_roi")
    coords = {
        key: _finite(roi.get(key), field=f"{source}.number_search_roi.{key}")
        for key in (
            "x_min_normalized",
            "x_max_normalized",
            "y_min_normalized",
            "y_max_normalized",
        )
    }
    if not (
        0.0 <= coords["x_min_normalized"] < coords["x_max_normalized"] <= 1.0
        and 0.0
        <= coords["y_min_normalized"]
        < coords["y_max_normalized"]
        <= 1.0
    ):
        raise JerseyVisibilityError(f"{source}: invalid normalized ROI range")
    if roi.get("coordinate_rounding") != "floor_min_ceil_max_then_clamp":
        raise JerseyVisibilityError(f"{source}: invalid ROI coordinate_rounding")

    measurements = _mapping(payload, "measurements")
    enabled_metrics = (
        "crop_dimensions",
        "roi_dimensions",
        "grayscale_mean",
        "grayscale_std",
        "laplacian_variance",
        "tenengrad_mean",
        "edge_density",
        "entropy",
        "local_contrast",
        "roi_other_person_union_coverage",
        "roi_other_person_center_inside_count",
        "full_crop_other_person_union_coverage",
    )
    for key in enabled_metrics:
        _require_bool(
            measurements.get(key),
            field=f"{source}.measurements.{key}",
            expected=True,
        )
    edge = _mapping(measurements, "edge_density_definition")
    if edge.get("method") != "canny_nonzero_fraction":
        raise JerseyVisibilityError(f"{source}: unsupported edge-density method")
    edge_low = _finite(edge.get("low_threshold"), field="edge low_threshold")
    edge_high = _finite(edge.get("high_threshold"), field="edge high_threshold")
    if not 0 <= edge_low < edge_high:
        raise JerseyVisibilityError(f"{source}: invalid Canny thresholds")
    local = _mapping(measurements, "local_contrast_definition")
    if local.get("method") != "mean_absolute_difference_from_gaussian_3x3":
        raise JerseyVisibilityError(f"{source}: unsupported local-contrast method")

    ranking = _mapping(payload, "ranking")
    _require_bool(
        ranking.get("independent_metrics_only"),
        field=f"{source}.ranking.independent_metrics_only",
        expected=True,
    )
    _require_bool(
        ranking.get("composite_score_enabled"),
        field=f"{source}.ranking.composite_score_enabled",
        expected=False,
    )
    for key in (
        "automatic_visibility_threshold",
        "automatic_readability_threshold",
    ):
        if ranking.get(key) is not None:
            raise JerseyVisibilityError(f"{source}.ranking.{key} must be null")
    _require_bool(
        ranking.get("sharpness_ranking_size_stratified"),
        field=f"{source}.ranking.sharpness_ranking_size_stratified",
        expected=True,
    )
    raw_strata = ranking.get("roi_height_strata_pixels")
    if (
        not isinstance(raw_strata, list)
        or len(raw_strata) < 2
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in raw_strata
        )
        or raw_strata != sorted(set(raw_strata))
        or raw_strata[0] != 0
    ):
        raise JerseyVisibilityError(f"{source}: invalid roi_height_strata_pixels")

    recognition = _mapping(payload, "recognition")
    for key in (
        "ocr_enabled",
        "recognizer_enabled",
        "checkpoint_required",
        "jersey_number_candidate_enabled",
        "automatic_jersey_assignment_enabled",
    ):
        _require_bool(
            recognition.get(key),
            field=f"{source}.recognition.{key}",
            expected=False,
        )

    manual = _mapping(payload, "manual_review")
    for key in (
        "number_visible",
        "number_readable",
        "back_facing",
        "digit_count",
        "jersey_number",
        "notes",
    ):
        if manual.get(key) is not None:
            raise JerseyVisibilityError(f"{source}.manual_review.{key} must be null")

    safety = _mapping(payload, "safety")
    expected_safety = {
        "source_crops_immutable": True,
        "segment_view_immutable": True,
        "regression_artifacts_immutable": True,
        "identity_ground_truth_available": False,
        "accuracy_claim_allowed": False,
        "team_assignment_enabled": False,
        "global_id_rewrite_enabled": False,
    }
    for key, expected in expected_safety.items():
        _require_bool(
            safety.get(key),
            field=f"{source}.safety.{key}",
            expected=expected,
        )

    return {
        **dict(payload),
        "number_search_roi": dict(roi),
        "measurements": dict(measurements),
        "ranking": {**dict(ranking), "roi_height_strata_pixels": list(raw_strata)},
        "source": source,
    }


def load_jersey_visibility_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise JerseyVisibilityError(f"config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JerseyVisibilityError(f"invalid YAML in {config_path}: {exc}") from exc
    return validate_jersey_visibility_config(payload, source=str(config_path))


def compute_upper_torso_number_search_roi(
    width: int, height: int, roi_config: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    width = _require_int(width, field="crop width", minimum=1)
    height = _require_int(height, field="crop height", minimum=1)
    x0 = max(
        0,
        min(
            width,
            math.floor(width * float(roi_config["x_min_normalized"])),
        ),
    )
    x1 = max(
        0,
        min(
            width,
            math.ceil(width * float(roi_config["x_max_normalized"])),
        ),
    )
    y0 = max(
        0,
        min(
            height,
            math.floor(height * float(roi_config["y_min_normalized"])),
        ),
    )
    y1 = max(
        0,
        min(
            height,
            math.ceil(height * float(roi_config["y_max_normalized"])),
        ),
    )
    if x1 <= x0 or y1 <= y0:
        raise JerseyVisibilityError("upper_torso_number_search_roi has no area")
    return x0, y0, x1, y1


def compute_image_measurements(
    image_bgr: np.ndarray, *, edge_low: float = 100, edge_high: float = 200
) -> dict[str, float]:
    if (
        image_bgr is None
        or image_bgr.size == 0
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise JerseyVisibilityError("image must be a non-empty BGR array")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray64 = gray.astype(np.float64)
    sobel_x = cv2.Sobel(gray64, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray64, cv2.CV_64F, 0, 1, ksize=3)
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / float(gray.size)
    blurred = cv2.GaussianBlur(gray64, (3, 3), 0)
    result = {
        "grayscale_mean": float(np.mean(gray64)),
        "grayscale_std": float(np.std(gray64)),
        "laplacian_variance": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
        "tenengrad_mean": float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y)),
        "edge_density": float(
            np.mean(cv2.Canny(gray, float(edge_low), float(edge_high)) > 0)
        ),
        "entropy": float(-np.sum(probabilities * np.log2(probabilities))),
        "local_contrast": float(np.mean(np.abs(gray64 - blurred))),
    }
    return {key: _finite(value, field=key) for key, value in result.items()}


def _int_bbox(values: Sequence[Any], *, field: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise JerseyVisibilityError(f"{field} must be a four-value sequence")
    if len(values) != 4:
        raise JerseyVisibilityError(f"{field} must contain four values")
    nums = [_finite(value, field=field) for value in values]
    result = [
        math.floor(nums[0]),
        math.floor(nums[1]),
        math.ceil(nums[2]),
        math.ceil(nums[3]),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise JerseyVisibilityError(f"{field} must have positive area")
    return result


def compute_other_person_contamination(
    *,
    target_frame_bbox: Sequence[int],
    roi_local_bbox: Sequence[int],
    frame_observations: Sequence[Mapping[str, Any]],
    own_source_observation_sha256: str | None,
    own_raw_track_id: int,
) -> dict[str, Any]:
    target = _int_bbox(target_frame_bbox, field="target_frame_bbox")
    roi_local = _int_bbox(roi_local_bbox, field="roi_local_bbox")
    roi_frame = [
        target[0] + roi_local[0],
        target[1] + roi_local[1],
        target[0] + roi_local[2],
        target[1] + roi_local[3],
    ]

    def coverage(region: Sequence[int]) -> tuple[float, int]:
        width = int(region[2] - region[0])
        height = int(region[3] - region[1])
        mask = np.zeros((height, width), dtype=np.uint8)
        centers = 0
        for row in frame_observations:
            digest = row.get("source_observation_sha256")
            if own_source_observation_sha256 and digest == own_source_observation_sha256:
                continue
            nested = row["source_observation"]
            if (
                own_source_observation_sha256 is None
                and int(row["raw_track_id"]) == int(own_raw_track_id)
            ):
                continue
            other = _int_bbox(nested["bbox_xyxy"], field="other bbox")
            x0 = max(int(region[0]), other[0])
            y0 = max(int(region[1]), other[1])
            x1 = min(int(region[2]), other[2])
            y1 = min(int(region[3]), other[3])
            if x1 > x0 and y1 > y0:
                mask[
                    y0 - int(region[1]) : y1 - int(region[1]),
                    x0 - int(region[0]) : x1 - int(region[0]),
                ] = 1
            cx = 0.5 * (other[0] + other[2])
            cy = 0.5 * (other[1] + other[3])
            if region[0] <= cx < region[2] and region[1] <= cy < region[3]:
                centers += 1
        return float(np.mean(mask)), centers

    full_coverage, _ = coverage(target)
    roi_coverage, roi_centers = coverage(roi_frame)
    return {
        "full_crop_other_person_union_coverage": _ratio(
            full_coverage, field="full crop contamination"
        ),
        "roi_other_person_union_coverage": _ratio(
            roi_coverage, field="ROI contamination"
        ),
        "roi_other_person_center_inside_count": int(roi_centers),
    }


def _safe_crop_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise JerseyVisibilityError("crop_relative_path must be non-empty str")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise JerseyVisibilityError(f"unsafe crop_relative_path: {relative!r}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise JerseyVisibilityError(f"crop path escapes source root: {relative}") from exc
    if not candidate.is_file():
        raise JerseyVisibilityError(f"selected crop file missing: {candidate}")
    return candidate


def load_segment_view_inputs(segment_view_dir: str | Path) -> dict[str, Any]:
    root = Path(segment_view_dir).expanduser().resolve()
    if not root.is_dir():
        raise JerseyVisibilityError(f"segment-view directory not found: {root}")
    paths = {name: root / name for name in SEGMENT_FILES}
    segments = _load_jsonl(paths["track_segments.jsonl"])
    assigned = _load_jsonl(paths["segment_observations.jsonl"])
    unassigned = _load_jsonl(
        paths["unassigned_observations.jsonl"], allow_empty=True
    )
    summary = _load_json(paths["segment_view_summary.json"])

    by_id: dict[str, dict[str, Any]] = {}
    for row in segments:
        if row.get("schema_version") != "reid_track_segment_v1":
            raise JerseyVisibilityError("track segment schema mismatch")
        sid = row.get("segment_id")
        kind = row.get("segment_kind")
        if not isinstance(sid, str) or not sid or sid in by_id:
            raise JerseyVisibilityError(f"invalid or duplicate segment_id: {sid!r}")
        if kind not in SEGMENT_KINDS:
            raise JerseyVisibilityError(f"invalid segment_kind: {kind!r}")
        _require_int(row.get("raw_track_id"), field=f"{sid}.raw_track_id", minimum=1)
        by_id[sid] = row

    seen_hashes: set[str] = set()
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_segment_frame: dict[tuple[str, int], dict[str, Any]] = {}
    for row in assigned:
        sid = row.get("segment_id")
        if sid not in by_id:
            raise JerseyVisibilityError(f"unknown assigned segment: {sid!r}")
        nested = row.get("source_observation")
        if not isinstance(nested, Mapping):
            raise JerseyVisibilityError("source_observation must be an object")
        digest = row.get("source_observation_sha256")
        actual = hashlib.sha256(
            canonical_observation_json(nested).encode("utf-8")
        ).hexdigest()
        if digest != actual:
            raise JerseyVisibilityError(
                f"source observation SHA mismatch for segment {sid}"
            )
        if digest in seen_hashes:
            raise JerseyVisibilityError(f"duplicate observation provenance: {digest}")
        seen_hashes.add(str(digest))
        frame = _require_int(row.get("frame_index"), field="frame_index")
        raw_id = _require_int(row.get("raw_track_id"), field="raw_track_id", minimum=1)
        if int(nested.get("frame_index", -1)) != frame:
            raise JerseyVisibilityError("nested source frame mismatch")
        if int(nested.get("track_id", -1)) != raw_id:
            raise JerseyVisibilityError("nested source track mismatch")
        key = (str(sid), frame)
        if key in by_segment_frame:
            raise JerseyVisibilityError(f"duplicate segment/frame observation: {key}")
        normalized = dict(row)
        by_segment_frame[key] = normalized
        by_frame[frame].append(normalized)

    unassigned_hashes: set[str] = set()
    for row in unassigned:
        if "segment_id" in row:
            raise JerseyVisibilityError("unassigned observation has segment_id")
        digest = str(row.get("source_observation_sha256"))
        if digest in seen_hashes or digest in unassigned_hashes:
            raise JerseyVisibilityError("assigned/unassigned provenance collision")
        unassigned_hashes.add(digest)

    if summary.get("status") != "ok":
        raise JerseyVisibilityError("segment view summary status must be ok")
    return {
        "root": root,
        "segments": segments,
        "by_id": by_id,
        "assigned": assigned,
        "by_frame": dict(by_frame),
        "by_segment_frame": by_segment_frame,
        "unassigned": unassigned,
        "summary": summary,
        "paths": paths,
    }


def build_crop_provenance_plan(
    *,
    segment_view: Mapping[str, Any],
    segmented_regression_dir: str | Path,
    baseline_run_dir: str | Path,
) -> dict[str, Any]:
    regression_root = Path(segmented_regression_dir).expanduser().resolve()
    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    if not regression_root.is_dir():
        raise JerseyVisibilityError(
            f"segmented regression directory not found: {regression_root}"
        )
    if not baseline_root.is_dir():
        raise JerseyVisibilityError(f"baseline run directory not found: {baseline_root}")
    regression_paths = {name: regression_root / name for name in REGRESSION_FILES}
    manual_crops = _load_jsonl(regression_paths["segment_crop_manifest.jsonl"])
    entities = _load_jsonl(regression_paths["segment_embedding_index.jsonl"])
    replacements = _load_jsonl(
        regression_paths["baseline_to_segment_replacement.jsonl"]
    )
    regression_summary = _load_json(
        regression_paths["segmented_reid_regression_summary.json"]
    )
    baseline_manifest_path = baseline_root / "crops" / "crop_manifest.jsonl"
    baseline_crops = _load_jsonl(baseline_manifest_path)

    segments = segment_view["by_id"]
    entity_by_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        sid = entity.get("segment_id")
        if sid not in segments:
            raise JerseyVisibilityError(f"embedding index references unknown segment {sid}")
        if sid in entity_by_id:
            raise JerseyVisibilityError(f"duplicate embedding segment {sid}")
        if int(entity.get("raw_track_id", -1)) != int(segments[sid]["raw_track_id"]):
            raise JerseyVisibilityError(f"embedding parent mismatch for {sid}")
        if entity.get("representation_source") not in REPRESENTATION_SOURCES:
            raise JerseyVisibilityError(f"invalid representation source for {sid}")
        entity_by_id[str(sid)] = entity
    if set(entity_by_id) != set(segments):
        raise JerseyVisibilityError("embedding index must cover every derived segment")

    baseline_by_id: dict[str, dict[str, Any]] = {}
    for row in baseline_crops:
        crop_id = row.get("crop_id")
        if not isinstance(crop_id, str) or crop_id in baseline_by_id:
            raise JerseyVisibilityError(f"invalid/duplicate baseline crop_id {crop_id!r}")
        baseline_by_id[crop_id] = row

    crop_plan: list[dict[str, Any]] = []
    assigned_crop_ids: set[str] = set()
    manual_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manual_crops:
        sid = row.get("segment_id")
        if sid not in segments:
            raise JerseyVisibilityError(f"manual crop references unknown segment {sid}")
        segment = segments[sid]
        entity = entity_by_id[sid]
        if segment["segment_kind"] != "manual_split_segment":
            raise JerseyVisibilityError(f"manual crop assigned to non-manual segment {sid}")
        if row.get("representation_source") != "recomputed_manual_segment":
            raise JerseyVisibilityError(f"manual crop representation mismatch for {sid}")
        if row.get("ambiguous_observation_used") is not False:
            raise JerseyVisibilityError(f"ambiguous observation used by crop {row.get('crop_id')}")
        crop_id = str(row.get("crop_id"))
        if crop_id in assigned_crop_ids:
            raise JerseyVisibilityError(f"duplicate crop-to-segment assignment: {crop_id}")
        assigned_crop_ids.add(crop_id)
        if int(row.get("raw_track_id", -1)) != int(segment["raw_track_id"]):
            raise JerseyVisibilityError(f"manual crop parent mismatch for {crop_id}")
        frame = _require_int(row.get("frame_index"), field=f"{crop_id}.frame_index")
        observation = segment_view["by_segment_frame"].get((str(sid), frame))
        if observation is None:
            raise JerseyVisibilityError(f"manual crop source observation missing: {crop_id}")
        if row.get("source_observation_sha256") != observation.get(
            "source_observation_sha256"
        ):
            raise JerseyVisibilityError(f"source SHA mismatch for crop {crop_id}")
        if int(row.get("source_observation_row_index", -1)) != int(
            observation.get("source_row_index", -2)
        ):
            raise JerseyVisibilityError(f"source row mismatch for crop {crop_id}")
        path = _safe_crop_path(regression_root, row.get("crop_relative_path"))
        planned = {
            **dict(row),
            "segment": segment,
            "entity": entity,
            "crop_source_kind": "recomputed_manual_segment",
            "source_root": regression_root,
            "source_crop_path": path,
            "source_observation": observation,
        }
        manual_by_segment[str(sid)].append(planned)
        crop_plan.append(planned)

    for sid, entity in entity_by_id.items():
        source = entity["representation_source"]
        segment = segments[sid]
        listed_ids = list(entity.get("crop_ids") or [])
        if source == "recomputed_manual_segment":
            actual_ids = [row["crop_id"] for row in manual_by_segment.get(sid, [])]
            if actual_ids != listed_ids:
                raise JerseyVisibilityError(f"manual crop ID list mismatch for {sid}")
            continue
        if source != "reused_baseline_raw_track_embedding":
            if listed_ids:
                raise JerseyVisibilityError(f"no-provenance segment has crop IDs: {sid}")
            continue
        if entity.get("parent_mixed_embedding_retired") is not False:
            raise JerseyVisibilityError(f"retired parent reused for {sid}")
        for crop_id in listed_ids:
            if crop_id in assigned_crop_ids:
                raise JerseyVisibilityError(
                    f"duplicate crop-to-segment assignment: {crop_id}"
                )
            row = baseline_by_id.get(str(crop_id))
            if row is None:
                raise JerseyVisibilityError(f"baseline crop ID missing: {crop_id}")
            if int(row.get("track_id", -1)) != int(segment["raw_track_id"]):
                raise JerseyVisibilityError(f"baseline crop parent mismatch for {crop_id}")
            assigned_crop_ids.add(str(crop_id))
            frame = _require_int(row.get("frame_index"), field=f"{crop_id}.frame_index")
            observation = segment_view["by_segment_frame"].get((sid, frame))
            path = _safe_crop_path(baseline_root, row.get("crop_relative_path"))
            crop_plan.append(
                {
                    **dict(row),
                    "segment_id": sid,
                    "raw_track_id": int(segment["raw_track_id"]),
                    "segment_kind": segment["segment_kind"],
                    "segment_index": segment.get("segment_index"),
                    "representation_source": source,
                    "segment": segment,
                    "entity": entity,
                    "crop_source_kind": "reused_baseline_selected_crop",
                    "source_root": baseline_root,
                    "source_crop_path": path,
                    "source_observation": observation,
                }
            )

    crop_plan.sort(
        key=lambda row: (
            str(row["segment_id"]),
            int(row["selection_rank"]),
            int(row["frame_index"]),
            str(row["crop_id"]),
        )
    )
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in crop_plan:
        by_segment[str(row["segment_id"])].append(row)

    return {
        "crops": crop_plan,
        "crops_by_segment": dict(by_segment),
        "entities": entities,
        "entity_by_id": entity_by_id,
        "replacements": replacements,
        "regression_summary": regression_summary,
        "regression_paths": regression_paths,
        "baseline_manifest_path": baseline_manifest_path,
    }


def _size_stratum(height: int, boundaries: Sequence[int]) -> str:
    for low, high in zip(boundaries, boundaries[1:]):
        if low <= height < high:
            return f"{low}_{high - 1}"
    raise JerseyVisibilityError(f"ROI height {height} outside configured strata")


def measure_crop(
    crop: Mapping[str, Any],
    *,
    frame_observations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(crop["source_crop_path"])
    expected_sha = crop.get("source_crop_sha256") or crop.get("crop_sha256")
    actual_sha = sha256_file(path)
    if expected_sha is not None and expected_sha != actual_sha:
        raise JerseyVisibilityError(f"source crop SHA mismatch: {crop['crop_id']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise JerseyVisibilityError(f"could not decode selected crop: {path}")
    crop_height, crop_width = image.shape[:2]
    bbox = _int_bbox(crop["bbox_xyxy"], field=f"{crop['crop_id']}.bbox_xyxy")
    if crop_width != bbox[2] - bbox[0] or crop_height != bbox[3] - bbox[1]:
        raise JerseyVisibilityError(
            f"crop dimensions do not match provenance bbox: {crop['crop_id']}"
        )
    roi_box = compute_upper_torso_number_search_roi(
        crop_width, crop_height, config["number_search_roi"]
    )
    x0, y0, x1, y1 = roi_box
    roi_image = image[y0:y1, x0:x1]
    edge_cfg = config["measurements"]["edge_density_definition"]
    metrics = compute_image_measurements(
        roi_image,
        edge_low=float(edge_cfg["low_threshold"]),
        edge_high=float(edge_cfg["high_threshold"]),
    )
    observation = crop.get("source_observation")
    contamination = compute_other_person_contamination(
        target_frame_bbox=bbox,
        roi_local_bbox=roi_box,
        frame_observations=frame_observations,
        own_source_observation_sha256=(
            str(observation["source_observation_sha256"])
            if observation is not None
            else None
        ),
        own_raw_track_id=int(crop["raw_track_id"]),
    )
    signal = {
        "schema_version": CROP_SIGNAL_SCHEMA,
        "segment_id": str(crop["segment_id"]),
        "raw_track_id": int(crop["raw_track_id"]),
        "segment_kind": str(crop["segment_kind"]),
        "segment_index": crop.get("segment_index"),
        "representation_source": str(crop["representation_source"]),
        "crop_source_kind": str(crop["crop_source_kind"]),
        "crop_id": str(crop["crop_id"]),
        "selection_rank": int(crop["selection_rank"]),
        "frame_index": int(crop["frame_index"]),
        "source_crop_path": str(path),
        "source_crop_relative_path": str(crop["crop_relative_path"]),
        "source_crop_sha256": actual_sha,
        "source_observation_row_index": (
            int(observation["source_row_index"]) if observation is not None else None
        ),
        "source_observation_sha256": (
            str(observation["source_observation_sha256"])
            if observation is not None
            else None
        ),
        "source_observation_provenance_available": observation is not None,
        "parent_mixed_embedding_retired": bool(
            crop["entity"]["parent_mixed_embedding_retired"]
        ),
        "crop_width_px": int(crop_width),
        "crop_height_px": int(crop_height),
        "crop_area_px": int(crop_width * crop_height),
        "roi_x_min": x0,
        "roi_y_min": y0,
        "roi_x_max": x1,
        "roi_y_max": y1,
        "roi_width_px": x1 - x0,
        "roi_height_px": y1 - y0,
        "roi_area_px": (x1 - x0) * (y1 - y0),
        "roi_semantics": "upper_torso_number_search_roi",
        **metrics,
        **contamination,
        "manual_number_visible": None,
        "manual_number_readable": None,
        "manual_back_facing": None,
        "manual_digit_count": None,
        "manual_jersey_number": None,
        "manual_notes": None,
    }
    signal["sharpness_size_stratum"] = _size_stratum(
        int(signal["roi_height_px"]),
        config["ranking"]["roi_height_strata_pixels"],
    )
    return signal


def assign_independent_ranks(rows: list[dict[str, Any]]) -> None:
    tie = lambda row: (  # noqa: E731 - compact deterministic key
        str(row["segment_id"]),
        int(row["frame_index"]),
        str(row["crop_id"]),
    )
    rank_specs = (
        ("roi_height_px", "roi_height_global_rank", True),
        ("roi_area_px", "roi_area_global_rank", True),
        ("local_contrast", "local_contrast_global_rank", True),
        ("edge_density", "edge_density_global_rank", True),
        ("entropy", "entropy_global_rank", True),
        (
            "roi_other_person_union_coverage",
            "contamination_low_global_rank",
            False,
        ),
    )
    for metric, field, descending in rank_specs:
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row[metric]) if descending else float(row[metric]),
                *tie(row),
            ),
        )
        for rank, row in enumerate(ordered, 1):
            row[field] = rank

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["sharpness_size_stratum"])].append(row)
    for stratum_rows in by_stratum.values():
        for metric, field in (
            ("laplacian_variance", "laplacian_rank_within_size_stratum"),
            ("tenengrad_mean", "tenengrad_rank_within_size_stratum"),
        ):
            ordered = sorted(
                stratum_rows,
                key=lambda row: (-float(row[metric]), *tie(row)),
            )
            for rank, row in enumerate(ordered, 1):
                row[field] = rank


def _distribution(values: Sequence[Any]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    array = np.asarray([_finite(v, field="distribution value") for v in values])
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _quantiles(values: Sequence[Any]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    array = np.asarray([_finite(v, field="quantile value") for v in values])
    return {
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def _best_crop(
    rows: Sequence[Mapping[str, Any]], metric: str, *, highest: bool
) -> str | None:
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[metric]) if highest else float(row[metric]),
            str(row["segment_id"]),
            int(row["frame_index"]),
            str(row["crop_id"]),
        ),
    )
    return str(ordered[0]["crop_id"])


def build_segment_summaries(
    *,
    segments: Sequence[Mapping[str, Any]],
    crop_signals: Sequence[Mapping[str, Any]],
    entity_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_segment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_signals:
        by_segment[str(row["segment_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for segment in sorted(
        segments, key=lambda row: (int(row["raw_track_id"]), str(row["segment_id"]))
    ):
        sid = str(segment["segment_id"])
        rows = by_segment.get(sid, [])
        entity = entity_by_id[sid]
        strata = sorted({str(row["sharpness_size_stratum"]) for row in rows})
        summaries.append(
            {
                "schema_version": SEGMENT_SUMMARY_SCHEMA,
                "segment_id": sid,
                "raw_track_id": int(segment["raw_track_id"]),
                "segment_kind": segment["segment_kind"],
                "segment_index": segment.get("segment_index"),
                "representation_status": entity["representation_status"],
                "measurement_status": (
                    "measured_selected_crops"
                    if rows
                    else "no_selected_crop_provenance"
                ),
                "selected_crop_count": len(rows),
                "measured_crop_count": len(rows),
                "no_crop_reason": None if rows else "no_selected_crop_provenance",
                "first_selected_frame": (
                    min(int(row["frame_index"]) for row in rows) if rows else None
                ),
                "last_selected_frame": (
                    max(int(row["frame_index"]) for row in rows) if rows else None
                ),
                "roi_height_px": _distribution(
                    [row["roi_height_px"] for row in rows]
                ),
                "laplacian_variance": _distribution(
                    [row["laplacian_variance"] for row in rows]
                ),
                "tenengrad_mean": _distribution(
                    [row["tenengrad_mean"] for row in rows]
                ),
                "local_contrast": _distribution(
                    [row["local_contrast"] for row in rows]
                ),
                "entropy": _distribution([row["entropy"] for row in rows]),
                "roi_other_person_union_coverage": _distribution(
                    [row["roi_other_person_union_coverage"] for row in rows]
                ),
                "largest_roi_crop_id": _best_crop(
                    rows, "roi_area_px", highest=True
                ),
                "highest_contrast_crop_id": _best_crop(
                    rows, "local_contrast", highest=True
                ),
                "highest_entropy_crop_id": _best_crop(
                    rows, "entropy", highest=True
                ),
                "lowest_contamination_crop_id": _best_crop(
                    rows, "roi_other_person_union_coverage", highest=False
                ),
                "sharpest_crop_id_per_size_stratum": {
                    stratum: _best_crop(
                        [
                            row
                            for row in rows
                            if row["sharpness_size_stratum"] == stratum
                        ],
                        "laplacian_variance",
                        highest=True,
                    )
                    for stratum in strata
                },
                "manual_any_number_visible": None,
                "manual_any_number_readable": None,
                "manual_consensus_jersey_number": None,
                "manual_review_status": None,
                "manual_notes": None,
                "ocr_attempted": False,
                "recognizer_attempted": False,
                "jersey_number_candidate": None,
                "automatic_assignment": False,
            }
        )
    return summaries


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_global_summary(
    *,
    segment_view: Mapping[str, Any],
    plan: Mapping[str, Any],
    crop_signals: Sequence[Mapping[str, Any]],
    segment_summaries: Sequence[Mapping[str, Any]],
    config_path: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    measured = [row for row in segment_summaries if row["measured_crop_count"] > 0]
    segments = segment_view["segments"]
    kinds = {
        kind: [row for row in segments if row["segment_kind"] == kind]
        for kind in SEGMENT_KINDS
    }
    measured_ids = {row["segment_id"] for row in measured}
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_signals:
        by_stratum[str(row["sharpness_size_stratum"])].append(row)

    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "ok",
        "source_artifacts": {
            "segment_view": {
                name: _artifact_record(path)
                for name, path in segment_view["paths"].items()
            },
            "segmented_regression": {
                name: _artifact_record(path)
                for name, path in plan["regression_paths"].items()
            },
            "baseline_crop_manifest": _artifact_record(
                plan["baseline_manifest_path"]
            ),
            "config": _artifact_record(config_path),
        },
        "total_derived_segment_count": len(segments),
        "measured_segment_count": len(measured),
        "no_selected_crop_segment_count": len(segments) - len(measured),
        "total_selected_crop_count": len(crop_signals),
        "recomputed_manual_crop_count": sum(
            row["crop_source_kind"] == "recomputed_manual_segment"
            for row in crop_signals
        ),
        "reused_baseline_crop_count": sum(
            row["crop_source_kind"] == "reused_baseline_selected_crop"
            for row in crop_signals
        ),
        "manual_segment_count": len(kinds["manual_split_segment"]),
        "measured_manual_segment_count": sum(
            row["segment_id"] in measured_ids
            for row in kinds["manual_split_segment"]
        ),
        "control_segment_count": len(kinds["no_split_control"]),
        "measured_control_segment_count": sum(
            row["segment_id"] in measured_ids
            for row in kinds["no_split_control"]
        ),
        "preserved_full_segment_count": len(kinds["preserved_full_track"]),
        "measured_preserved_full_segment_count": sum(
            row["segment_id"] in measured_ids
            for row in kinds["preserved_full_track"]
        ),
        "metric_distributions": {
            key: _quantiles([row[key] for row in crop_signals])
            for key in (
                "crop_width_px",
                "crop_height_px",
                "roi_width_px",
                "roi_height_px",
                "roi_area_px",
                "local_contrast",
                "entropy",
                "edge_density",
                "full_crop_other_person_union_coverage",
                "roi_other_person_union_coverage",
            )
        },
        "sharpness_quantiles_by_roi_height_stratum": {
            stratum: {
                "laplacian_variance": _quantiles(
                    [row["laplacian_variance"] for row in rows]
                ),
                "tenengrad_mean": _quantiles(
                    [row["tenengrad_mean"] for row in rows]
                ),
            }
            for stratum, rows in sorted(by_stratum.items())
        },
        "safety": {
            "video_opened": False,
            "new_crop_extraction_performed": False,
            "crop_files_modified": False,
            "source_artifacts_mutated": False,
            "ambiguous_observation_used": False,
            "unassigned_observation_used": False,
            "unassigned_observations_in_contamination_context": False,
            "retired_mixed_parent_entity_used": False,
            "OCR_performed": False,
            "recognizer_performed": False,
            "checkpoint_loaded": False,
            "jersey_number_candidate_generated": False,
            "automatic_jersey_assignment_performed": False,
            "team_assignment_performed": False,
            "global_id_rewrite_performed": False,
            "identity_ground_truth_available": False,
            "accuracy_claimed": False,
        },
        "semantics_and_limitations": [
            "upper-torso ROI is only a number-search region",
            "no automatic back-facing determination is performed",
            "no visibility or readability classification is performed",
            "no global sharpness threshold is selected",
            "size-stratified sharpness ranks are audit aids",
            "bbox contamination is not pixel segmentation",
            "selected crops are not every observation",
            "no selected crop does not mean unreadable",
            "pass-through track purity is unproven",
            "zero bbox contamination does not prove crop purity",
            "unassigned observations are excluded even from contamination context",
            "sample.mp4 is not a product constant",
        ],
        "elapsed_sec": _finite(elapsed_sec, field="elapsed_sec"),
    }


def create_temp_jersey_visibility_dir(output_dir: Path) -> Path:
    final = output_dir.expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.parent / (
        f"_tmp_reid_jersey_visibility_{final.name}_{uuid.uuid4().hex[:8]}"
    )
    temp.mkdir(parents=False, exist_ok=False)
    return temp


def finalize_jersey_visibility_dir(
    *, temp_dir: Path, final_dir: Path, overwrite: bool
) -> Path:
    temp = temp_dir.expanduser().resolve()
    final = final_dir.expanduser().resolve()
    if not temp.is_dir():
        raise JerseyVisibilityError(f"temporary output missing: {temp}")
    backup: Path | None = None
    try:
        if final.exists():
            if not overwrite:
                raise JerseyVisibilityError(f"output already exists: {final}")
            backup = final.with_name(
                f"_backup_reid_jersey_visibility_{final.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final, backup)
        os.rename(temp, final)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            try:
                os.rename(backup, final)
                backup = None
            except OSError:
                pass
        raise
    for prefix in (
        "_tmp_reid_jersey_visibility_",
        "_backup_reid_jersey_visibility_",
    ):
        for stray in final.parent.glob(f"{prefix}{final.name}_*"):
            cleanup_dir(stray)
    return final


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            handle.write("\n")


def validate_temp_outputs(temp_dir: Path) -> None:
    actual = sorted(path.name for path in temp_dir.iterdir())
    if actual != sorted(OUTPUT_NAMES):
        raise JerseyVisibilityError(f"unexpected output files: {actual}")
    crop_rows = _load_jsonl(temp_dir / CROP_SIGNALS_NAME, allow_empty=True)
    segment_rows = _load_jsonl(temp_dir / SEGMENT_SUMMARY_NAME)
    summary = _load_json(temp_dir / SUMMARY_NAME)
    if any(row.get("schema_version") != CROP_SIGNAL_SCHEMA for row in crop_rows):
        raise JerseyVisibilityError("crop signal output schema mismatch")
    if any(
        row.get("schema_version") != SEGMENT_SUMMARY_SCHEMA for row in segment_rows
    ):
        raise JerseyVisibilityError("segment summary output schema mismatch")
    if summary.get("schema_version") != SUMMARY_SCHEMA or summary.get("status") != "ok":
        raise JerseyVisibilityError("global summary output schema/status mismatch")
    if len(crop_rows) != int(summary["total_selected_crop_count"]):
        raise JerseyVisibilityError("written crop count mismatch")
    if len(segment_rows) != int(summary["total_derived_segment_count"]):
        raise JerseyVisibilityError("written segment count mismatch")
    for name in OUTPUT_NAMES:
        if not (temp_dir / name).read_bytes().endswith(b"\n"):
            raise JerseyVisibilityError(f"{name} lacks final newline")


def run_analyze_jersey_visibility(
    *,
    segment_view_dir: str | Path,
    segmented_regression_dir: str | Path,
    baseline_run_dir: str | Path,
    config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    final_dir = Path(output_dir).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    try:
        check_output_collision(final_dir, overwrite=overwrite)
    except Exception as exc:
        raise JerseyVisibilityError(str(exc)) from exc

    validated_config = load_jersey_visibility_config(config_path)
    segment_view = load_segment_view_inputs(segment_view_dir)
    plan = build_crop_provenance_plan(
        segment_view=segment_view,
        segmented_regression_dir=segmented_regression_dir,
        baseline_run_dir=baseline_run_dir,
    )

    signals: list[dict[str, Any]] = []
    for crop in plan["crops"]:
        frame = int(crop["frame_index"])
        signals.append(
            measure_crop(
                crop,
                frame_observations=segment_view["by_frame"].get(frame, []),
                config=validated_config,
            )
        )
    assign_independent_ranks(signals)
    signals.sort(
        key=lambda row: (
            str(row["segment_id"]),
            int(row["selection_rank"]),
            int(row["frame_index"]),
            str(row["crop_id"]),
        )
    )
    segment_summaries = build_segment_summaries(
        segments=segment_view["segments"],
        crop_signals=signals,
        entity_by_id=plan["entity_by_id"],
    )
    summary = build_global_summary(
        segment_view=segment_view,
        plan=plan,
        crop_signals=signals,
        segment_summaries=segment_summaries,
        config_path=config_path,
        elapsed_sec=time.perf_counter() - started,
    )

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_jersey_visibility_dir(final_dir)
        _write_jsonl(temp_dir / CROP_SIGNALS_NAME, signals)
        _write_jsonl(temp_dir / SEGMENT_SUMMARY_NAME, segment_summaries)
        (temp_dir / SUMMARY_NAME).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_temp_outputs(temp_dir)
        finalized = finalize_jersey_visibility_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "crop_signals_path": str(finalized / CROP_SIGNALS_NAME),
        "segment_summary_path": str(finalized / SEGMENT_SUMMARY_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "total_derived_segment_count": summary["total_derived_segment_count"],
        "measured_segment_count": summary["measured_segment_count"],
        "no_selected_crop_segment_count": summary[
            "no_selected_crop_segment_count"
        ],
        "total_selected_crop_count": summary["total_selected_crop_count"],
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
    }
