"""ReID torso-oriented coarse team/kit descriptors (Stage 5B1A).

Measurement-only: deterministic torso ROI, color descriptors, equal-weight
track aggregation. No team assignment, clustering, linking, or crop mutation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from football_analytics.reid.embedding import EmbeddingError, resolve_crop_path
from football_analytics.reid.schema import ReIDSchemaError, validate_manifest_row
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

CROP_KIT_SCHEMA = "reid_crop_kit_descriptor_v1"
TRACK_KIT_SCHEMA = "reid_track_kit_descriptor_v1"
KIT_SUMMARY_SCHEMA = "reid_kit_descriptor_summary_v1"
CONFIG_SCHEMA = "reid_kit_descriptor_config_v1"

CROP_KIT_NAME = "crop_kit_descriptors.jsonl"
TRACK_KIT_NAME = "track_kit_descriptors.jsonl"
KIT_SUMMARY_NAME = "kit_descriptor_summary.json"

CHROMATIC_FAMILY_NAMES = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
)
ACHROMATIC_FAMILY_NAMES = ("black", "gray", "white")
DEFAULT_FAMILY_ORDER = ACHROMATIC_FAMILY_NAMES + CHROMATIC_FAMILY_NAMES


class KitError(RuntimeError):
    """Raised when kit-descriptor inputs, config, or outputs are invalid."""


def _reject_non_finite_json(value: str) -> None:
    raise KitError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise KitError(f"{field} must be finite, got {value!r}")
    return number


def _ensure_ratio(value: float, *, field: str) -> float:
    number = _ensure_finite_float(value, field=field)
    if number < 0.0 or number > 1.0:
        raise KitError(f"{field} must be in [0, 1], got {number}")
    return number


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise KitError(f"{field} must be a positive int, got {value!r}")
    return value


def _require_nonneg_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise KitError(f"{field} must be a non-negative int, got {value!r}")
    return value


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise KitError(f"{field} must be a bool, got {value!r}")
    if expected is not None and value is not expected:
        raise KitError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_fraction(mapping: Mapping[str, Any], key: str, *, source: str) -> float:
    if key not in mapping:
        raise KitError(f"{source}: missing {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KitError(f"{source}: {key} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise KitError(f"{source}: {key} must be finite in [0, 1], got {value!r}")
    return number


def _require_positive_bin_count(
    mapping: Mapping[str, Any], key: str, *, source: str
) -> int:
    if key not in mapping:
        raise KitError(f"{source}: missing {key}")
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise KitError(f"{source}: {key} must be an int >= 1, got {value!r}")
    return value


def _require_int_in_range(
    value: Any, *, field: str, low: int, high: int
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise KitError(f"{field} must be an int, got {value!r}")
    if value < low or value > high:
        raise KitError(f"{field} must be in [{low}, {high}], got {value}")
    return value


def _require_range_pair(
    mapping: Mapping[str, Any], key: str, *, source: str, expected: list[int]
) -> list[int]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise KitError(f"{source}: {key} must be a two-int sequence")
    if len(value) != 2:
        raise KitError(f"{source}: {key} must have length 2")
    pair = [
        _require_int_in_range(value[0], field=f"{source}.{key}[0]", low=0, high=10_000),
        _require_int_in_range(value[1], field=f"{source}.{key}[1]", low=0, high=10_000),
    ]
    if pair != expected:
        raise KitError(f"{source}: {key} must be {expected}, got {pair}")
    return pair


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
            "mean": None,
        }
    arr = np.asarray(
        [_ensure_finite_float(v, field="stat") for v in values], dtype=np.float64
    )
    qs = {
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }
    for key, number in qs.items():
        if not math.isfinite(number):
            raise KitError(f"non-finite percentile {key}")
    return qs


def _validate_hue_family_coverage(
    chromatic_ranges: Mapping[str, Any], *, source: str
) -> dict[str, list[list[int]]]:
    if set(chromatic_ranges.keys()) != set(CHROMATIC_FAMILY_NAMES):
        raise KitError(
            f"{source}: chromatic_hue_ranges keys must be exactly "
            f"{list(CHROMATIC_FAMILY_NAMES)}, got {sorted(chromatic_ranges)}"
        )
    coverage = np.full(180, -1, dtype=np.int32)
    name_to_index = {name: i for i, name in enumerate(CHROMATIC_FAMILY_NAMES)}
    normalized: dict[str, list[list[int]]] = {}
    for family in CHROMATIC_FAMILY_NAMES:
        ranges = chromatic_ranges[family]
        if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)):
            raise KitError(f"{source}: chromatic_hue_ranges.{family} must be a list")
        if not ranges:
            raise KitError(f"{source}: chromatic_hue_ranges.{family} must not be empty")
        parsed: list[list[int]] = []
        for idx, item in enumerate(ranges):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                raise KitError(
                    f"{source}: chromatic_hue_ranges.{family}[{idx}] must be [lo, hi]"
                )
            if len(item) != 2:
                raise KitError(
                    f"{source}: chromatic_hue_ranges.{family}[{idx}] must have length 2"
                )
            lo = _require_int_in_range(
                item[0],
                field=f"{source}.chromatic_hue_ranges.{family}[{idx}][0]",
                low=0,
                high=179,
            )
            hi = _require_int_in_range(
                item[1],
                field=f"{source}.chromatic_hue_ranges.{family}[{idx}][1]",
                low=0,
                high=179,
            )
            if hi < lo:
                raise KitError(
                    f"{source}: chromatic_hue_ranges.{family}[{idx}] hi < lo"
                )
            for hue in range(lo, hi + 1):
                if coverage[hue] >= 0:
                    overlap = CHROMATIC_FAMILY_NAMES[int(coverage[hue])]
                    raise KitError(
                        f"{source}: hue {hue} overlaps families "
                        f"{overlap!r} and {family!r}"
                    )
                coverage[hue] = name_to_index[family]
            parsed.append([lo, hi])
        normalized[family] = parsed

    missing = [int(h) for h in range(180) if coverage[h] < 0]
    if missing:
        preview = missing[:12]
        raise KitError(
            f"{source}: chromatic hue coverage gap for values {preview}"
            f"{'...' if len(missing) > 12 else ''}"
        )
    return normalized


def validate_kit_descriptor_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise KitError(
            f"{source}: schema_version must be {CONFIG_SCHEMA!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("stage_status") != "measurement_baseline":
        raise KitError(
            f"{source}: stage_status must be 'measurement_baseline', "
            f"got {payload.get('stage_status')!r}"
        )

    for flag in (
        "automatic_team_assignment_enabled",
        "forced_two_team_clustering_enabled",
        "automatic_link_enabled",
        "automatic_reject_enabled",
    ):
        _require_bool(payload.get(flag), field=f"{source}.{flag}", expected=False)

    if payload.get("kit_similarity_threshold") is not None:
        raise KitError(
            f"{source}: kit_similarity_threshold must be null in this measurement gate"
        )

    region = payload.get("source_region")
    if not isinstance(region, Mapping):
        raise KitError(f"{source}: source_region must be a mapping")
    if region.get("name") != "normalized_center_torso":
        raise KitError(
            f"{source}: source_region.name must be 'normalized_center_torso'"
        )
    x_min = _require_fraction(region, "x_min_fraction", source=f"{source}.source_region")
    x_max = _require_fraction(region, "x_max_fraction", source=f"{source}.source_region")
    y_min = _require_fraction(region, "y_min_fraction", source=f"{source}.source_region")
    y_max = _require_fraction(region, "y_max_fraction", source=f"{source}.source_region")
    if x_min >= x_max:
        raise KitError(f"{source}: x_min_fraction must be < x_max_fraction")
    if y_min >= y_max:
        raise KitError(f"{source}: y_min_fraction must be < y_max_fraction")
    for flag in ("resize_enabled", "segmentation_enabled", "background_removal_enabled"):
        _require_bool(region.get(flag), field=f"{source}.source_region.{flag}", expected=False)

    quality_usage = payload.get("quality_usage")
    if not isinstance(quality_usage, Mapping):
        raise KitError(f"{source}: quality_usage must be a mapping")
    _require_bool(
        quality_usage.get("quality_signals_required"),
        field=f"{source}.quality_usage.quality_signals_required",
        expected=True,
    )
    _require_bool(
        quality_usage.get("exclusion_enabled"),
        field=f"{source}.quality_usage.exclusion_enabled",
        expected=False,
    )
    _require_bool(
        quality_usage.get("weighting_enabled"),
        field=f"{source}.quality_usage.weighting_enabled",
        expected=False,
    )
    _require_bool(
        quality_usage.get("copied_for_audit_only"),
        field=f"{source}.quality_usage.copied_for_audit_only",
        expected=True,
    )

    histograms = payload.get("histograms")
    if not isinstance(histograms, Mapping):
        raise KitError(f"{source}: histograms must be a mapping")
    hue_bins = _require_positive_bin_count(
        histograms, "hue_bin_count", source=f"{source}.histograms"
    )
    sat_bins = _require_positive_bin_count(
        histograms, "saturation_bin_count", source=f"{source}.histograms"
    )
    val_bins = _require_positive_bin_count(
        histograms, "value_bin_count", source=f"{source}.histograms"
    )
    if hue_bins != 18 or sat_bins != 8 or val_bins != 8:
        raise KitError(
            f"{source}: histogram bin counts must be hue=18, saturation=8, value=8"
        )
    hue_range = _require_range_pair(
        histograms, "hue_range_opencv", source=f"{source}.histograms", expected=[0, 180]
    )
    sat_range = _require_range_pair(
        histograms, "saturation_range", source=f"{source}.histograms", expected=[0, 256]
    )
    val_range = _require_range_pair(
        histograms, "value_range", source=f"{source}.histograms", expected=[0, 256]
    )

    chromatic = payload.get("chromatic_hue")
    if not isinstance(chromatic, Mapping):
        raise KitError(f"{source}: chromatic_hue must be a mapping")
    min_sat = chromatic.get("minimum_saturation")
    min_val = chromatic.get("minimum_value")
    if not isinstance(min_sat, int) or isinstance(min_sat, bool) or not (0 <= min_sat <= 255):
        raise KitError(f"{source}: chromatic_hue.minimum_saturation must be int in [0,255]")
    if not isinstance(min_val, int) or isinstance(min_val, bool) or not (0 <= min_val <= 255):
        raise KitError(f"{source}: chromatic_hue.minimum_value must be int in [0,255]")
    if chromatic.get("empty_histogram_behavior") != "all_zero":
        raise KitError(
            f"{source}: chromatic_hue.empty_histogram_behavior must be 'all_zero'"
        )

    families = payload.get("coarse_color_families")
    if not isinstance(families, Mapping):
        raise KitError(f"{source}: coarse_color_families must be a mapping")
    _require_bool(
        families.get("enabled"),
        field=f"{source}.coarse_color_families.enabled",
        expected=True,
    )
    achromatic_sat_max = families.get("achromatic_saturation_max")
    black_value_max = families.get("black_value_max")
    white_value_min = families.get("white_value_min")
    high_sat_min = families.get("high_saturation_min")
    for name, value in (
        ("achromatic_saturation_max", achromatic_sat_max),
        ("black_value_max", black_value_max),
        ("white_value_min", white_value_min),
        ("high_saturation_min", high_sat_min),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value <= 255):
            raise KitError(
                f"{source}: coarse_color_families.{name} must be int in [0,255]"
            )
    if black_value_max >= white_value_min:
        raise KitError(
            f"{source}: black_value_max must be < white_value_min"
        )

    family_order = families.get("family_order")
    if list(family_order) != list(DEFAULT_FAMILY_ORDER):
        raise KitError(
            f"{source}: coarse_color_families.family_order must be "
            f"{list(DEFAULT_FAMILY_ORDER)}, got {family_order!r}"
        )
    if len(set(family_order)) != len(family_order):
        raise KitError(f"{source}: duplicate family name in family_order")

    hue_ranges = families.get("chromatic_hue_ranges")
    if not isinstance(hue_ranges, Mapping):
        raise KitError(f"{source}: coarse_color_families.chromatic_hue_ranges required")
    normalized_ranges = _validate_hue_family_coverage(
        hue_ranges, source=f"{source}.coarse_color_families"
    )

    aggregation = payload.get("track_aggregation")
    if not isinstance(aggregation, Mapping):
        raise KitError(f"{source}: track_aggregation must be a mapping")
    if aggregation.get("method") != "equal_weight_mean":
        raise KitError(f"{source}: track_aggregation.method must be equal_weight_mean")
    _require_bool(
        aggregation.get("quality_weighting_enabled"),
        field=f"{source}.track_aggregation.quality_weighting_enabled",
        expected=False,
    )
    _require_bool(
        aggregation.get("crop_exclusion_enabled"),
        field=f"{source}.track_aggregation.crop_exclusion_enabled",
        expected=False,
    )
    _require_bool(
        aggregation.get("preserve_crop_order"),
        field=f"{source}.track_aggregation.preserve_crop_order",
        expected=True,
    )

    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise KitError(f"{source}: policy must be a mapping")
    _require_bool(
        policy.get("same_kit_is_identity_proof"),
        field=f"{source}.policy.same_kit_is_identity_proof",
        expected=False,
    )
    _require_bool(
        policy.get("different_kit_hard_reject_enabled"),
        field=f"{source}.policy.different_kit_hard_reject_enabled",
        expected=False,
    )
    if policy.get("usage") != "measurement_audit_and_future_ranking":
        raise KitError(
            f"{source}: policy.usage must be measurement_audit_and_future_ranking"
        )
    _require_bool(
        policy.get("unknown_outlier_preserved"),
        field=f"{source}.policy.unknown_outlier_preserved",
        expected=True,
    )

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise KitError(f"{source}: evaluation must be a mapping")
    _require_bool(
        evaluation.get("ground_truth_team_labels_available"),
        field=f"{source}.evaluation.ground_truth_team_labels_available",
        expected=False,
    )
    _require_bool(
        evaluation.get("accuracy_claim_allowed"),
        field=f"{source}.evaluation.accuracy_claim_allowed",
        expected=False,
    )

    # Build hue lookup table once.
    hue_family_table = np.empty(180, dtype=object)
    for family, ranges in normalized_ranges.items():
        for lo, hi in ranges:
            hue_family_table[lo : hi + 1] = family

    return {
        "schema_version": CONFIG_SCHEMA,
        "stage_status": "measurement_baseline",
        "automatic_team_assignment_enabled": False,
        "forced_two_team_clustering_enabled": False,
        "automatic_link_enabled": False,
        "automatic_reject_enabled": False,
        "kit_similarity_threshold": None,
        "source_region": {
            "name": "normalized_center_torso",
            "x_min_fraction": x_min,
            "x_max_fraction": x_max,
            "y_min_fraction": y_min,
            "y_max_fraction": y_max,
            "resize_enabled": False,
            "segmentation_enabled": False,
            "background_removal_enabled": False,
        },
        "quality_usage": {
            "quality_signals_required": True,
            "exclusion_enabled": False,
            "weighting_enabled": False,
            "copied_for_audit_only": True,
        },
        "histograms": {
            "hue_bin_count": hue_bins,
            "saturation_bin_count": sat_bins,
            "value_bin_count": val_bins,
            "hue_range_opencv": hue_range,
            "saturation_range": sat_range,
            "value_range": val_range,
        },
        "chromatic_hue": {
            "minimum_saturation": int(min_sat),
            "minimum_value": int(min_val),
            "empty_histogram_behavior": "all_zero",
        },
        "coarse_color_families": {
            "enabled": True,
            "achromatic_saturation_max": int(achromatic_sat_max),
            "black_value_max": int(black_value_max),
            "white_value_min": int(white_value_min),
            "high_saturation_min": int(high_sat_min),
            "family_order": list(DEFAULT_FAMILY_ORDER),
            "chromatic_hue_ranges": normalized_ranges,
            "hue_family_table": hue_family_table,
        },
        "track_aggregation": {
            "method": "equal_weight_mean",
            "quality_weighting_enabled": False,
            "crop_exclusion_enabled": False,
            "preserve_crop_order": True,
        },
        "policy": {
            "same_kit_is_identity_proof": False,
            "different_kit_hard_reject_enabled": False,
            "usage": "measurement_audit_and_future_ranking",
            "unknown_outlier_preserved": True,
        },
        "evaluation": {
            "ground_truth_team_labels_available": False,
            "accuracy_claim_allowed": False,
        },
        "source": source,
    }


def load_kit_descriptor_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise KitError(f"kit descriptor config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise KitError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise KitError(f"kit descriptor config must be a mapping: {config_path}")
    validated = validate_kit_descriptor_config(payload, source=str(config_path))
    validated["source_path"] = str(config_path)
    return validated


def create_temp_kit_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_kit_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise KitError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_kit_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise KitError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise KitError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_kit_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_kit_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_kit_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def load_crop_manifest_for_kit(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise KitError(f"crop manifest not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KitError(f"could not read crop manifest: {path}: {exc}") from exc
    if not text.strip():
        raise KitError("crop manifest is empty")

    rows: list[dict[str, Any]] = []
    crop_ids: set[str] = set()
    rel_paths: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_json)
        except KitError:
            raise
        except json.JSONDecodeError as exc:
            raise KitError(f"invalid JSON on crop manifest line {line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise KitError(f"manifest line {line_no} must be a JSON object")
        try:
            validate_manifest_row(payload)
        except ReIDSchemaError as exc:
            raise KitError(f"manifest line {line_no}: {exc}") from exc

        track_id = _require_positive_int(payload["track_id"], field="track_id")
        frame_index = _require_nonneg_int(payload["frame_index"], field="frame_index")
        selection_rank = _require_positive_int(
            payload["selection_rank"], field="selection_rank"
        )
        crop_id = payload["crop_id"]
        rel = payload["crop_relative_path"]
        if not isinstance(crop_id, str) or not crop_id:
            raise KitError(f"manifest line {line_no}: crop_id must be non-empty str")
        if crop_id in crop_ids:
            raise KitError(f"duplicate crop_id in manifest: {crop_id}")
        if not isinstance(rel, str) or not rel:
            raise KitError(
                f"manifest line {line_no}: crop_relative_path must be non-empty str"
            )
        if rel in rel_paths:
            raise KitError(f"duplicate crop_relative_path in manifest: {rel}")

        bbox = payload["bbox_xyxy"]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
            raise KitError(f"manifest line {line_no}: bbox_xyxy values must be ints")
        left, top, right, bottom = [int(v) for v in bbox]
        if right <= left or bottom <= top:
            raise KitError(f"manifest line {line_no}: invalid bbox size")
        width = right - left
        height = bottom - top
        if float(payload["bbox_width"]) != float(width):
            raise KitError(f"manifest line {line_no}: bbox_width mismatch")
        if float(payload["bbox_height"]) != float(height):
            raise KitError(f"manifest line {line_no}: bbox_height mismatch")
        if float(payload["bbox_area"]) != float(width * height):
            raise KitError(f"manifest line {line_no}: bbox_area mismatch")

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
        raise KitError("crop manifest is empty")
    return rows


def load_quality_signals_for_kit(
    quality_path: str | Path, *, manifest_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    path = Path(quality_path).expanduser().resolve()
    if not path.is_file():
        raise KitError(f"quality signals not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KitError(f"could not read quality signals: {path}: {exc}") from exc
    if not text.strip():
        raise KitError("quality signals file is empty")

    rows: list[dict[str, Any]] = []
    crop_ids: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_json)
        except KitError:
            raise
        except json.JSONDecodeError as exc:
            raise KitError(
                f"invalid JSON on quality signals line {line_no}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise KitError(f"quality line {line_no} must be a JSON object")

        for key in (
            "crop_id",
            "track_id",
            "frame_index",
            "selection_rank",
            "crop_relative_path",
            "laplacian_variance",
            "union_other_person_crop_coverage",
            "frame_edge_contact_count",
            "quality_decision",
            "automatic_exclusion_applied",
            "quality_threshold",
            "contamination_threshold",
        ):
            if key not in payload:
                raise KitError(f"quality line {line_no}: missing {key}")

        crop_id = payload["crop_id"]
        if not isinstance(crop_id, str) or not crop_id:
            raise KitError(f"quality line {line_no}: crop_id must be non-empty str")
        if crop_id in crop_ids:
            raise KitError(f"duplicate crop_id in quality signals: {crop_id}")
        crop_ids.add(crop_id)

        track_id = _require_positive_int(payload["track_id"], field="track_id")
        frame_index = _require_nonneg_int(payload["frame_index"], field="frame_index")
        selection_rank = _require_positive_int(
            payload["selection_rank"], field="selection_rank"
        )
        rel = payload["crop_relative_path"]
        if not isinstance(rel, str) or not rel:
            raise KitError(
                f"quality line {line_no}: crop_relative_path must be non-empty str"
            )

        if payload.get("quality_decision") != "measurement_only":
            raise KitError(
                f"quality line {line_no}: quality_decision must be measurement_only"
            )
        _require_bool(
            payload.get("automatic_exclusion_applied"),
            field=f"quality line {line_no}: automatic_exclusion_applied",
            expected=False,
        )
        if payload.get("quality_threshold") is not None:
            raise KitError(
                f"quality line {line_no}: quality_threshold must be null"
            )
        if payload.get("contamination_threshold") is not None:
            raise KitError(
                f"quality line {line_no}: contamination_threshold must be null"
            )

        lap = _ensure_finite_float(
            float(payload["laplacian_variance"]), field="laplacian_variance"
        )
        union = _ensure_ratio(
            float(payload["union_other_person_crop_coverage"]),
            field="union_other_person_crop_coverage",
        )
        edge = _require_nonneg_int(
            payload["frame_edge_contact_count"], field="frame_edge_contact_count"
        )

        rows.append(
            {
                "crop_id": crop_id,
                "track_id": track_id,
                "frame_index": frame_index,
                "selection_rank": selection_rank,
                "crop_relative_path": rel,
                "laplacian_variance": lap,
                "union_other_person_crop_coverage": union,
                "frame_edge_contact_count": edge,
                "quality_decision": "measurement_only",
                "automatic_exclusion_applied": False,
                "quality_threshold": None,
                "contamination_threshold": None,
            }
        )

    if len(rows) != len(manifest_rows):
        raise KitError(
            f"quality row count {len(rows)} != crop manifest count {len(manifest_rows)}"
        )

    for idx, (manifest_row, quality_row) in enumerate(zip(manifest_rows, rows)):
        for key in (
            "crop_id",
            "track_id",
            "frame_index",
            "selection_rank",
            "crop_relative_path",
        ):
            if manifest_row[key] != quality_row[key]:
                raise KitError(
                    f"quality/manifest mismatch at index {idx} field {key}: "
                    f"manifest={manifest_row[key]!r} quality={quality_row[key]!r}"
                )
    return rows


def compute_torso_roi(
    *,
    width: int,
    height: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise KitError(f"invalid crop size {width}x{height}")
    region = config["source_region"]
    x0 = int(math.floor(width * float(region["x_min_fraction"])))
    x1 = int(math.ceil(width * float(region["x_max_fraction"])))
    y0 = int(math.floor(height * float(region["y_min_fraction"])))
    y1 = int(math.ceil(height * float(region["y_max_fraction"])))

    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    if x1 <= x0:
        x0 = max(0, min(x0, width - 1))
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y0 = max(0, min(y0, height - 1))
        y1 = min(height, y0 + 1)

    torso_width = x1 - x0
    torso_height = y1 - y0
    if torso_width < 1 or torso_height < 1:
        raise KitError(f"torso ROI has non-positive size: {torso_width}x{torso_height}")

    torso_area = float(torso_width * torso_height)
    crop_area = float(width * height)
    return {
        "torso_x0": int(x0),
        "torso_y0": int(y0),
        "torso_x1": int(x1),
        "torso_y1": int(y1),
        "torso_width": int(torso_width),
        "torso_height": int(torso_height),
        "torso_area": torso_area,
        "torso_area_crop_ratio": _ensure_ratio(
            torso_area / crop_area, field="torso_area_crop_ratio"
        ),
        "torso_region_valid": True,
    }


def _l1_normalize(hist: np.ndarray, *, field: str) -> list[float]:
    total = float(np.sum(hist))
    if total <= 0.0:
        raise KitError(f"{field} histogram has non-positive sum")
    out = (hist.astype(np.float64) / total).tolist()
    for value in out:
        _ensure_ratio(float(value), field=field)
    return [float(v) for v in out]


def _channel_stats(channel: np.ndarray) -> tuple[float, float]:
    flat = channel.reshape(-1).astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    return (
        _ensure_finite_float(mean, field="channel_mean"),
        _ensure_finite_float(std, field="channel_std"),
    )


def _top_color_families(
    fractions: Mapping[str, float], *, family_order: Sequence[str], limit: int = 3
) -> list[dict[str, Any]]:
    order_index = {name: i for i, name in enumerate(family_order)}
    ranked = sorted(
        family_order,
        key=lambda name: (-float(fractions[name]), order_index[name]),
    )
    tops: list[dict[str, Any]] = []
    for name in ranked[:limit]:
        tops.append(
            {
                "family": name,
                "fraction": _ensure_ratio(
                    float(fractions[name]), field=f"top_family.{name}"
                ),
            }
        )
    return tops


def compute_torso_kit_metrics(
    image_bgr: np.ndarray, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise KitError("crop image must be HxWx3 BGR")
    height, width = image_bgr.shape[:2]
    roi_geom = compute_torso_roi(width=width, height=height, config=config)
    x0 = int(roi_geom["torso_x0"])
    y0 = int(roi_geom["torso_y0"])
    x1 = int(roi_geom["torso_x1"])
    y1 = int(roi_geom["torso_y1"])
    torso = image_bgr[y0:y1, x0:x1]
    if torso.size == 0:
        raise KitError("torso ROI is empty")

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    h_mean, h_std = _channel_stats(h_ch)
    s_mean, s_std = _channel_stats(s_ch)
    v_mean, v_std = _channel_stats(v_ch)
    l_mean, l_std = _channel_stats(lab[:, :, 0])
    a_mean, a_std = _channel_stats(lab[:, :, 1])
    b_mean, b_std = _channel_stats(lab[:, :, 2])

    hist_cfg = config["histograms"]
    chromatic_cfg = config["chromatic_hue"]
    family_cfg = config["coarse_color_families"]

    chromatic_mask = (
        (s_ch >= int(chromatic_cfg["minimum_saturation"]))
        & (v_ch >= int(chromatic_cfg["minimum_value"]))
    )
    torso_pixel_count = int(torso.shape[0] * torso.shape[1])
    chromatic_pixel_count = int(np.count_nonzero(chromatic_mask))
    chromatic_pixel_ratio = _ensure_ratio(
        float(chromatic_pixel_count) / float(torso_pixel_count),
        field="chromatic_pixel_ratio",
    )
    achromatic_pixel_ratio = _ensure_ratio(
        1.0 - chromatic_pixel_ratio, field="achromatic_pixel_ratio"
    )

    hue_bins = int(hist_cfg["hue_bin_count"])
    sat_bins = int(hist_cfg["saturation_bin_count"])
    val_bins = int(hist_cfg["value_bin_count"])

    if chromatic_pixel_count > 0:
        hue_hist = cv2.calcHist(
            [hsv],
            [0],
            chromatic_mask.astype(np.uint8),
            [hue_bins],
            hist_cfg["hue_range_opencv"],
        ).reshape(-1)
        hue_histogram = _l1_normalize(hue_hist, field="hue_histogram_chromatic")
    else:
        hue_histogram = [0.0] * hue_bins

    sat_hist = cv2.calcHist(
        [hsv], [1], None, [sat_bins], hist_cfg["saturation_range"]
    ).reshape(-1)
    val_hist = cv2.calcHist(
        [hsv], [2], None, [val_bins], hist_cfg["value_range"]
    ).reshape(-1)
    saturation_histogram = _l1_normalize(sat_hist, field="saturation_histogram")
    value_histogram = _l1_normalize(val_hist, field="value_histogram")

    dark_max = int(family_cfg["black_value_max"])
    bright_min = int(family_cfg["white_value_min"])
    low_sat_max = int(family_cfg["achromatic_saturation_max"])
    high_sat_min = int(family_cfg["high_saturation_min"])

    dark_pixel_ratio = _ensure_ratio(
        float(np.count_nonzero(v_ch <= dark_max)) / float(torso_pixel_count),
        field="dark_pixel_ratio",
    )
    bright_pixel_ratio = _ensure_ratio(
        float(np.count_nonzero(v_ch >= bright_min)) / float(torso_pixel_count),
        field="bright_pixel_ratio",
    )
    low_saturation_pixel_ratio = _ensure_ratio(
        float(np.count_nonzero(s_ch <= low_sat_max)) / float(torso_pixel_count),
        field="low_saturation_pixel_ratio",
    )
    high_saturation_pixel_ratio = _ensure_ratio(
        float(np.count_nonzero(s_ch >= high_sat_min)) / float(torso_pixel_count),
        field="high_saturation_pixel_ratio",
    )

    hue_table = family_cfg["hue_family_table"]
    family_counts = {name: 0 for name in family_cfg["family_order"]}
    h_flat = h_ch.reshape(-1)
    s_flat = s_ch.reshape(-1)
    v_flat = v_ch.reshape(-1)
    for hue, sat, val in zip(h_flat.tolist(), s_flat.tolist(), v_flat.tolist()):
        if int(sat) <= low_sat_max:
            if int(val) <= dark_max:
                family_counts["black"] += 1
            elif int(val) >= bright_min:
                family_counts["white"] += 1
            else:
                family_counts["gray"] += 1
        else:
            family = str(hue_table[int(hue)])
            family_counts[family] += 1

    counted = sum(family_counts.values())
    if counted != torso_pixel_count:
        raise KitError(
            f"color family assignment count {counted} != torso pixels {torso_pixel_count}"
        )

    color_family_fractions = {
        name: _ensure_ratio(
            float(family_counts[name]) / float(torso_pixel_count),
            field=f"color_family_fractions.{name}",
        )
        for name in family_cfg["family_order"]
    }
    fraction_sum = sum(color_family_fractions.values())
    if abs(fraction_sum - 1.0) > 1e-6:
        raise KitError(f"color family fractions sum {fraction_sum} != 1")

    top = _top_color_families(
        color_family_fractions, family_order=family_cfg["family_order"], limit=3
    )
    dominant = top[0]

    return {
        **roi_geom,
        "torso_pixel_count": torso_pixel_count,
        "chromatic_pixel_count": chromatic_pixel_count,
        "chromatic_pixel_ratio": chromatic_pixel_ratio,
        "achromatic_pixel_ratio": achromatic_pixel_ratio,
        "hsv_mean": {"h": h_mean, "s": s_mean, "v": v_mean},
        "hsv_std": {"h": h_std, "s": s_std, "v": v_std},
        "lab_mean": {"l": l_mean, "a": a_mean, "b": b_mean},
        "lab_std": {"l": l_std, "a": a_std, "b": b_std},
        "lab_encoding": "opencv_uint8_bgr2lab",
        "hue_histogram_chromatic": hue_histogram,
        "saturation_histogram": saturation_histogram,
        "value_histogram": value_histogram,
        "dark_pixel_ratio": dark_pixel_ratio,
        "bright_pixel_ratio": bright_pixel_ratio,
        "low_saturation_pixel_ratio": low_saturation_pixel_ratio,
        "high_saturation_pixel_ratio": high_saturation_pixel_ratio,
        "color_family_fractions": color_family_fractions,
        "dominant_color_family": dominant["family"],
        "dominant_color_family_fraction": dominant["fraction"],
        "top_color_families": top,
    }


def analyze_one_crop_kit(
    *,
    manifest_row: Mapping[str, Any],
    quality_row: Mapping[str, Any],
    manifest_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        crop_path = resolve_crop_path(manifest_dir, manifest_row["crop_relative_path"])
    except EmbeddingError as exc:
        raise KitError(str(exc)) from exc

    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise KitError(f"could not decode crop JPEG: {crop_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise KitError(f"decoded crop has non-positive size: {crop_path}")

    left, top, right, bottom = [int(v) for v in manifest_row["bbox_xyxy"]]
    expected_w = right - left
    expected_h = bottom - top
    if width != expected_w or height != expected_h:
        raise KitError(
            f"decoded size {width}x{height} != manifest bbox size "
            f"{expected_w}x{expected_h} for {manifest_row['crop_id']}"
        )

    metrics = compute_torso_kit_metrics(image, config=config)
    return {
        "crop_id": manifest_row["crop_id"],
        "track_id": int(manifest_row["track_id"]),
        "frame_index": int(manifest_row["frame_index"]),
        "selection_rank": int(manifest_row["selection_rank"]),
        "crop_relative_path": manifest_row["crop_relative_path"],
        **metrics,
        "quality_signal_joined": True,
        "laplacian_variance": float(quality_row["laplacian_variance"]),
        "union_other_person_crop_coverage": float(
            quality_row["union_other_person_crop_coverage"]
        ),
        "frame_edge_contact_count": int(quality_row["frame_edge_contact_count"]),
        "quality_weight_applied": False,
        "quality_exclusion_applied": False,
        "team_assignment": None,
        "kit_similarity_threshold": None,
        "automatic_link_applied": False,
        "automatic_reject_applied": False,
        "descriptor_usage": "measurement_only",
        "schema_version": CROP_KIT_SCHEMA,
    }


def build_track_kit_descriptors(
    crop_rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_track: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_rows:
        by_track[int(row["track_id"])].append(row)

    family_order = list(config["coarse_color_families"]["family_order"])
    track_rows: list[dict[str, Any]] = []
    for track_id in sorted(by_track):
        crops = by_track[track_id]
        crop_count = len(crops)
        crop_ids = [str(c["crop_id"]) for c in crops]

        mean_hue = np.mean(
            [np.asarray(c["hue_histogram_chromatic"], dtype=np.float64) for c in crops],
            axis=0,
        )
        mean_sat = np.mean(
            [np.asarray(c["saturation_histogram"], dtype=np.float64) for c in crops],
            axis=0,
        )
        mean_val = np.mean(
            [np.asarray(c["value_histogram"], dtype=np.float64) for c in crops],
            axis=0,
        )
        mean_family = {
            name: float(
                np.mean([float(c["color_family_fractions"][name]) for c in crops])
            )
            for name in family_order
        }
        for name, value in mean_family.items():
            _ensure_ratio(value, field=f"mean_color_family_fractions.{name}")

        mean_sat_list = [float(v) for v in mean_sat.tolist()]
        mean_val_list = [float(v) for v in mean_val.tolist()]
        mean_hue_list = [float(v) for v in mean_hue.tolist()]
        for value in mean_hue_list + mean_sat_list + mean_val_list:
            _ensure_finite_float(value, field="track_histogram")
            if value < 0.0 or value > 1.0:
                raise KitError(f"track histogram bin out of [0,1]: {value}")
        if abs(sum(mean_sat_list) - 1.0) > 1e-6:
            raise KitError("mean saturation histogram does not sum to ~1")
        if abs(sum(mean_val_list) - 1.0) > 1e-6:
            raise KitError("mean value histogram does not sum to ~1")
        if abs(sum(mean_family.values()) - 1.0) > 1e-6:
            raise KitError("mean color family fractions do not sum to ~1")

        top = _top_color_families(mean_family, family_order=family_order, limit=3)
        chrom_ratios = [float(c["chromatic_pixel_ratio"]) for c in crops]
        hsv_across = {
            "h": float(np.mean([float(c["hsv_mean"]["h"]) for c in crops])),
            "s": float(np.mean([float(c["hsv_mean"]["s"]) for c in crops])),
            "v": float(np.mean([float(c["hsv_mean"]["v"]) for c in crops])),
        }
        lab_across = {
            "l": float(np.mean([float(c["lab_mean"]["l"]) for c in crops])),
            "a": float(np.mean([float(c["lab_mean"]["a"]) for c in crops])),
            "b": float(np.mean([float(c["lab_mean"]["b"]) for c in crops])),
        }
        for key, value in {**hsv_across, **lab_across}.items():
            _ensure_finite_float(value, field=f"across_crops.{key}")

        track_rows.append(
            {
                "track_id": int(track_id),
                "crop_count": crop_count,
                "crop_ids": crop_ids,
                "mean_hue_histogram_chromatic": mean_hue_list,
                "mean_saturation_histogram": mean_sat_list,
                "mean_value_histogram": mean_val_list,
                "mean_color_family_fractions": mean_family,
                "chromatic_pixel_ratio_mean": float(np.mean(chrom_ratios)),
                "chromatic_pixel_ratio_median": float(np.median(chrom_ratios)),
                "dark_pixel_ratio_mean": float(
                    np.mean([float(c["dark_pixel_ratio"]) for c in crops])
                ),
                "bright_pixel_ratio_mean": float(
                    np.mean([float(c["bright_pixel_ratio"]) for c in crops])
                ),
                "low_saturation_pixel_ratio_mean": float(
                    np.mean([float(c["low_saturation_pixel_ratio"]) for c in crops])
                ),
                "high_saturation_pixel_ratio_mean": float(
                    np.mean([float(c["high_saturation_pixel_ratio"]) for c in crops])
                ),
                "hsv_mean_across_crops": hsv_across,
                "lab_mean_across_crops": lab_across,
                "dominant_track_color_family": top[0]["family"],
                "dominant_track_color_family_fraction": top[0]["fraction"],
                "top_track_color_families": top,
                "mean_laplacian_variance": float(
                    np.mean([float(c["laplacian_variance"]) for c in crops])
                ),
                "max_union_other_person_crop_coverage": float(
                    max(float(c["union_other_person_crop_coverage"]) for c in crops)
                ),
                "frame_edge_contact_crop_count": int(
                    sum(1 for c in crops if int(c["frame_edge_contact_count"]) > 0)
                ),
                "quality_weighting_applied": False,
                "excluded_crop_count": 0,
                "aggregation_method": "equal_weight_mean",
                "team_assignment": None,
                "forced_two_team_clustering_applied": False,
                "kit_similarity_threshold": None,
                "automatic_link_applied": False,
                "automatic_reject_applied": False,
                "schema_version": TRACK_KIT_SCHEMA,
            }
        )
    return track_rows


def build_kit_summary(
    *,
    crop_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    crop_manifest: Path,
    quality_signals: Path,
    config: Mapping[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    family_order = list(config["coarse_color_families"]["family_order"])
    crop_dom = Counter(str(r["dominant_color_family"]) for r in crop_rows)
    track_dom = Counter(str(r["dominant_track_color_family"]) for r in track_rows)
    crop_dom_hist = {name: int(crop_dom.get(name, 0)) for name in family_order}
    track_dom_hist = {name: int(track_dom.get(name, 0)) for name in family_order}

    region = config["source_region"]
    hist = config["histograms"]
    families = config["coarse_color_families"]
    chromatic = config["chromatic_hue"]

    return {
        "status": "ok",
        "crop_count": len(crop_rows),
        "track_count": len(track_rows),
        "source_crop_manifest": str(crop_manifest.expanduser().resolve()),
        "source_quality_signals": str(quality_signals.expanduser().resolve()),
        "source_config": str(config.get("source_path") or config.get("source")),
        "torso_region_name": region["name"],
        "torso_fraction_bounds": {
            "x_min_fraction": region["x_min_fraction"],
            "x_max_fraction": region["x_max_fraction"],
            "y_min_fraction": region["y_min_fraction"],
            "y_max_fraction": region["y_max_fraction"],
        },
        "resize_performed": False,
        "segmentation_performed": False,
        "background_removal_performed": False,
        "crop_descriptor_schema": CROP_KIT_SCHEMA,
        "track_descriptor_schema": TRACK_KIT_SCHEMA,
        "histogram_definitions": {
            "hue_bin_count": hist["hue_bin_count"],
            "saturation_bin_count": hist["saturation_bin_count"],
            "value_bin_count": hist["value_bin_count"],
            "hue_range_opencv": hist["hue_range_opencv"],
            "saturation_range": hist["saturation_range"],
            "value_range": hist["value_range"],
            "chromatic_hue_minimum_saturation": chromatic["minimum_saturation"],
            "chromatic_hue_minimum_value": chromatic["minimum_value"],
            "mean_hue_histogram_policy": (
                "equal_weight_mean_of_crop_histograms_including_all_zero;"
                "not_renormalized"
            ),
            "lab_encoding": "opencv_uint8_bgr2lab",
        },
        "color_family_definitions": {
            "family_order": family_order,
            "achromatic_saturation_max": families["achromatic_saturation_max"],
            "black_value_max": families["black_value_max"],
            "white_value_min": families["white_value_min"],
            "high_saturation_min": families["high_saturation_min"],
            "chromatic_hue_ranges": families["chromatic_hue_ranges"],
            "note": (
                "color family bins are descriptor labels only; "
                "not team or identity thresholds"
            ),
        },
        "chromatic_pixel_ratio": _percentile_stats(
            [float(r["chromatic_pixel_ratio"]) for r in crop_rows]
        ),
        "dark_pixel_ratio": _percentile_stats(
            [float(r["dark_pixel_ratio"]) for r in crop_rows]
        ),
        "bright_pixel_ratio": _percentile_stats(
            [float(r["bright_pixel_ratio"]) for r in crop_rows]
        ),
        "low_saturation_pixel_ratio": _percentile_stats(
            [float(r["low_saturation_pixel_ratio"]) for r in crop_rows]
        ),
        "high_saturation_pixel_ratio": _percentile_stats(
            [float(r["high_saturation_pixel_ratio"]) for r in crop_rows]
        ),
        "dominant_color_family_fraction": _percentile_stats(
            [float(r["dominant_color_family_fraction"]) for r in crop_rows]
        ),
        "crop_dominant_color_family_histogram": crop_dom_hist,
        "track_dominant_color_family_histogram": track_dom_hist,
        "automatic_team_assignment_performed": False,
        "forced_two_team_clustering_performed": False,
        "kit_similarity_threshold": None,
        "crop_exclusion_performed": False,
        "quality_weighting_performed": False,
        "automatic_link_performed": False,
        "automatic_reject_performed": False,
        "same_kit_is_identity_proof": False,
        "different_kit_hard_reject_enabled": False,
        "sample_specific_colors_hardcoded": False,
        "ground_truth_team_labels_available": False,
        "accuracy_claimed": False,
        "elapsed_sec": _ensure_finite_float(elapsed_sec, field="elapsed_sec"),
        "schema_version": KIT_SUMMARY_SCHEMA,
    }


def run_analyze_reid_kit_descriptors(
    *,
    crop_manifest: str | Path,
    quality_signals: str | Path,
    config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(crop_manifest).expanduser().resolve()
    quality_path = Path(quality_signals).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    manifest_dir = manifest_path.parent

    check_output_collision(final_dir, overwrite=overwrite)
    validated_config = load_kit_descriptor_config(config_path)

    temp_dir: Path | None = None
    try:
        manifest_rows = load_crop_manifest_for_kit(manifest_path)
        quality_rows = load_quality_signals_for_kit(
            quality_path, manifest_rows=manifest_rows
        )

        crop_descriptors: list[dict[str, Any]] = []
        for manifest_row, quality_row in zip(manifest_rows, quality_rows):
            crop_descriptors.append(
                analyze_one_crop_kit(
                    manifest_row=manifest_row,
                    quality_row=quality_row,
                    manifest_dir=manifest_dir,
                    config=validated_config,
                )
            )

        track_descriptors = build_track_kit_descriptors(
            crop_descriptors, config=validated_config
        )
        elapsed = time.perf_counter() - started
        summary = build_kit_summary(
            crop_rows=crop_descriptors,
            track_rows=track_descriptors,
            crop_manifest=manifest_path,
            quality_signals=quality_path,
            config=validated_config,
            elapsed_sec=elapsed,
        )

        temp_dir = create_temp_kit_dir(final_dir)
        write_manifest_jsonl(temp_dir / CROP_KIT_NAME, crop_descriptors)
        write_manifest_jsonl(temp_dir / TRACK_KIT_NAME, track_descriptors)
        (temp_dir / KIT_SUMMARY_NAME).write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        finalized = finalize_kit_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "crop_kit_path": str(finalized / CROP_KIT_NAME),
        "track_kit_path": str(finalized / TRACK_KIT_NAME),
        "summary_path": str(finalized / KIT_SUMMARY_NAME),
        "crop_count": summary["crop_count"],
        "track_count": summary["track_count"],
        "torso_region_name": summary["torso_region_name"],
        "torso_fraction_bounds": summary["torso_fraction_bounds"],
        "crop_dominant_color_family_histogram": summary[
            "crop_dominant_color_family_histogram"
        ],
        "track_dominant_color_family_histogram": summary[
            "track_dominant_color_family_histogram"
        ],
        "chromatic_pixel_ratio": summary["chromatic_pixel_ratio"],
        "quality_weighting_performed": False,
        "crop_exclusion_performed": False,
        "automatic_team_assignment_performed": False,
        "forced_two_team_clustering_performed": False,
        "kit_similarity_threshold": None,
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
    }
