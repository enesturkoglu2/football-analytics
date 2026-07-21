"""Stage 5C-A2b deterministic jersey visual review panel builder.

Reads existing Stage 5C-A2a measurement artifacts only. It renders
display-only review panels (original crop + nearest-neighbor ROI display
zoom) and a blank manual annotation template. It never opens video, extracts
or modifies crops, enhances images, runs OCR/recognition, or produces any
automatic visibility/readability/back-facing/jersey-number decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from football_analytics.reid.writers import check_output_collision, cleanup_dir

CONFIG_SCHEMA = "reid_jersey_review_panel_config_v1"
REVIEW_ITEM_SCHEMA = "reid_jersey_review_item_v1"
GROUP_MEMBERSHIP_SCHEMA = "reid_jersey_review_group_membership_v1"
PANEL_INDEX_SCHEMA = "reid_jersey_review_panel_index_v1"
SUMMARY_SCHEMA = "reid_jersey_review_summary_v1"

MEASUREMENT_CROP_SIGNALS_NAME = "jersey_visibility_crop_signals.jsonl"
MEASUREMENT_SEGMENT_SUMMARY_NAME = "jersey_visibility_segment_summary.jsonl"
MEASUREMENT_SUMMARY_NAME = "jersey_visibility_summary.json"

REVIEW_ITEMS_NAME = "jersey_review_items.jsonl"
GROUP_MEMBERSHIPS_NAME = "jersey_review_group_memberships.jsonl"
PANEL_INDEX_NAME = "jersey_review_panel_index.jsonl"
ANNOTATION_TEMPLATE_NAME = "jersey_review_annotations_template.csv"
SUMMARY_NAME = "jersey_review_summary.json"
PANELS_DIRNAME = "panels"
OUTPUT_NAMES = (
    REVIEW_ITEMS_NAME,
    GROUP_MEMBERSHIPS_NAME,
    PANEL_INDEX_NAME,
    ANNOTATION_TEMPLATE_NAME,
    SUMMARY_NAME,
)

SEGMENT_KINDS = {
    "manual_split_segment",
    "no_split_control",
    "preserved_full_track",
}
CROP_SOURCE_KINDS = {
    "recomputed_manual_segment",
    "reused_baseline_selected_crop",
}
REPRESENTATION_SOURCES = {
    "recomputed_manual_segment",
    "reused_baseline_raw_track_embedding",
}

MANUAL_FIELDS = (
    "manual_crop_valid",
    "manual_back_facing",
    "manual_number_visible",
    "manual_number_readable",
    "manual_digit_count",
    "manual_jersey_number",
    "manual_contamination_affects_number_region",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

ANNOTATION_COLUMNS = (
    "review_item_id",
    "review_index",
    "segment_id",
    "raw_track_id",
    "crop_id",
    "frame_index",
    "master_panel_path",
    "master_panel_page",
    "master_panel_tile_index",
    "group_memberships",
) + MANUAL_FIELDS

GLOBAL_RANK_FIELDS = (
    "roi_height_global_rank",
    "roi_area_global_rank",
    "local_contrast_global_rank",
    "edge_density_global_rank",
    "entropy_global_rank",
    "contamination_low_global_rank",
)

# Fixed group registry: (name, kind). Metric groups carry the global rank
# field driving selection and the reported metric field.
FIXED_GROUPS = (
    ("all_selected_crops", "master"),
    ("all_manual_segment_crops", "master"),
    ("roi_height_top", "metric"),
    ("roi_height_bottom", "metric"),
    ("local_contrast_top", "metric"),
    ("local_contrast_bottom", "metric"),
    ("entropy_top", "metric"),
    ("entropy_bottom", "metric"),
    ("roi_contamination_low", "metric"),
    ("roi_contamination_high", "metric"),
)
METRIC_GROUP_SPECS = {
    "roi_height_top": ("roi_height_global_rank", "roi_height_px", False),
    "roi_height_bottom": ("roi_height_global_rank", "roi_height_px", True),
    "local_contrast_top": ("local_contrast_global_rank", "local_contrast", False),
    "local_contrast_bottom": ("local_contrast_global_rank", "local_contrast", True),
    "entropy_top": ("entropy_global_rank", "entropy", False),
    "entropy_bottom": ("entropy_global_rank", "entropy", True),
    "roi_contamination_low": (
        "contamination_low_global_rank",
        "roi_other_person_union_coverage",
        False,
    ),
    "roi_contamination_high": (
        "contamination_low_global_rank",
        "roi_other_person_union_coverage",
        True,
    ),
}
SHARPNESS_VARIANTS = (
    ("laplacian_top", "laplacian_rank_within_size_stratum", "laplacian_variance", False),
    ("laplacian_bottom", "laplacian_rank_within_size_stratum", "laplacian_variance", True),
    ("tenengrad_top", "tenengrad_rank_within_size_stratum", "tenengrad_mean", False),
    ("tenengrad_bottom", "tenengrad_rank_within_size_stratum", "tenengrad_mean", True),
)

_STRATUM_NAME_RE = re.compile(r"^[0-9]+_[0-9]+$")
_SEGMENT_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Fixed deterministic panel geometry (display only).
TILE_WIDTH = 340
TILE_HEIGHT = 420
TILE_PADDING = 6
TEXT_AREA_HEIGHT = 108
ORIGINAL_AREA_HEIGHT = 178
ZOOM_LABEL_HEIGHT = 16
CANVAS_BACKGROUND = 28
TILE_BACKGROUND = 46
EMPTY_TILE_BACKGROUND = 36
TEXT_COLOR = (235, 235, 235)
ROI_RECT_COLOR = (0, 220, 255)


class JerseyReviewError(RuntimeError):
    """Raised when Stage 5C-A2b inputs, rendering, or outputs are invalid."""


def _reject_non_finite(value: str) -> None:
    raise JerseyReviewError(f"NaN/Infinity forbidden in JSON: {value}")


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise JerseyReviewError(f"{field} must be bool, got {value!r}")
    if expected is not None and value is not expected:
        raise JerseyReviewError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise JerseyReviewError(f"{field} must be int >= {minimum}, got {value!r}")
    return value


def _require_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise JerseyReviewError(f"{field} must be a non-empty string")
    return value


def _finite(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise JerseyReviewError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise JerseyReviewError(f"{field} must be finite, got {value!r}")
    return result


def _ratio(value: Any, *, field: str) -> float:
    result = _finite(value, field=field)
    if result < 0.0 or result > 1.0:
        raise JerseyReviewError(f"{field} must be in [0, 1]")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JerseyReviewError(f"JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite)
        except json.JSONDecodeError as exc:
            raise JerseyReviewError(
                f"invalid JSON on {path.name} line {line_no}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise JerseyReviewError(f"{path.name} line {line_no} must be an object")
        rows.append(row)
    if not rows and not allow_empty:
        raise JerseyReviewError(f"JSONL is empty: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JerseyReviewError(f"JSON not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite
        )
    except json.JSONDecodeError as exc:
        raise JerseyReviewError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JerseyReviewError(f"{path} must contain an object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise JerseyReviewError(f"{key} must be a mapping")
    return value


def validate_jersey_review_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JerseyReviewError(f"{source}: config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise JerseyReviewError(f"{source}: invalid schema_version")
    if payload.get("stage_status") != "implementation_manual_review_panel_baseline":
        raise JerseyReviewError(f"{source}: invalid stage_status")

    input_cfg = _mapping(payload, "input")
    _require_bool(
        input_cfg.get("require_measurement_status_ok"),
        field=f"{source}.input.require_measurement_status_ok",
        expected=True,
    )
    if input_cfg.get("require_crop_signal_schema") != (
        "reid_jersey_visibility_crop_signal_v1"
    ):
        raise JerseyReviewError(f"{source}: invalid require_crop_signal_schema")
    if input_cfg.get("require_segment_summary_schema") != (
        "reid_jersey_visibility_segment_summary_v1"
    ):
        raise JerseyReviewError(f"{source}: invalid require_segment_summary_schema")
    for key, expected in (
        ("use_existing_crop_signals_only", True),
        ("allow_video_input", False),
        ("allow_new_crop_extraction", False),
        ("require_source_crop_sha_match", True),
    ):
        _require_bool(
            input_cfg.get(key), field=f"{source}.input.{key}", expected=expected
        )

    universe = _mapping(payload, "review_universe")
    for key, expected in (
        ("include_all_selected_crops", True),
        ("include_no_selected_crop_segments", False),
        ("deduplicate_review_items", True),
        ("preserve_all_group_memberships", True),
    ):
        _require_bool(
            universe.get(key),
            field=f"{source}.review_universe.{key}",
            expected=expected,
        )

    groups = _mapping(payload, "groups")
    master_all = _mapping(groups, "all_selected_crops")
    _require_bool(
        master_all.get("enabled"),
        field=f"{source}.groups.all_selected_crops.enabled",
        expected=True,
    )
    manual_group = _mapping(groups, "all_manual_segment_crops")
    _require_bool(
        manual_group.get("enabled"),
        field=f"{source}.groups.all_manual_segment_crops.enabled",
    )
    for name in METRIC_GROUP_SPECS:
        group = _mapping(groups, name)
        enabled = _require_bool(
            group.get("enabled"), field=f"{source}.groups.{name}.enabled"
        )
        if enabled:
            _require_int(
                group.get("count"), field=f"{source}.groups.{name}.count", minimum=1
            )
    sharp = _mapping(groups, "sharpness_by_roi_height_stratum")
    sharp_enabled = _require_bool(
        sharp.get("enabled"),
        field=f"{source}.groups.sharpness_by_roi_height_stratum.enabled",
    )
    if sharp_enabled:
        for key in (
            "laplacian_top_count",
            "laplacian_bottom_count",
            "tenengrad_top_count",
            "tenengrad_bottom_count",
        ):
            _require_int(
                sharp.get(key),
                field=f"{source}.groups.sharpness_by_roi_height_stratum.{key}",
                minimum=1,
            )
    critical = _mapping(groups, "critical_segments")
    critical_enabled = _require_bool(
        critical.get("enabled"), field=f"{source}.groups.critical_segments.enabled"
    )
    if critical_enabled:
        segment_ids = critical.get("segment_ids")
        if not isinstance(segment_ids, Sequence) or isinstance(segment_ids, str):
            raise JerseyReviewError(f"{source}: critical segment_ids must be a list")
        if not segment_ids:
            raise JerseyReviewError(f"{source}: critical segment_ids must not be empty")
        seen: set[str] = set()
        for sid in segment_ids:
            if not isinstance(sid, str) or not _SEGMENT_ID_RE.fullmatch(sid):
                raise JerseyReviewError(
                    f"{source}: invalid critical segment id {sid!r}"
                )
            if sid in seen:
                raise JerseyReviewError(
                    f"{source}: duplicate critical segment id {sid}"
                )
            seen.add(sid)

    panel = _mapping(payload, "panel")
    _require_int(panel.get("columns"), field=f"{source}.panel.columns", minimum=1)
    _require_int(panel.get("rows"), field=f"{source}.panel.rows", minimum=1)
    if panel.get("output_format") != "png":
        raise JerseyReviewError(f"{source}: panel.output_format must be png")
    if panel.get("roi_zoom_interpolation") != "nearest":
        raise JerseyReviewError(
            f"{source}: panel.roi_zoom_interpolation must be nearest"
        )
    for key, expected in (
        ("original_crop_display", True),
        ("draw_number_search_roi", True),
        ("roi_zoom_display", True),
        ("allow_visual_enhancement", False),
        ("preserve_aspect_ratio", True),
        ("annotate_actual_dimensions", True),
        ("annotate_segment_id", True),
        ("annotate_crop_id", True),
        ("annotate_frame_index", True),
        ("annotate_crop_source_kind", True),
        ("annotate_roi_height", True),
        ("annotate_roi_contamination", True),
        ("annotate_group_metric", True),
        ("annotate_predicted_number", False),
        ("annotate_visibility_class", False),
    ):
        _require_bool(
            panel.get(key), field=f"{source}.panel.{key}", expected=expected
        )

    manual = _mapping(payload, "manual_review")
    _require_bool(
        manual.get("emit_blank_annotation_template"),
        field=f"{source}.manual_review.emit_blank_annotation_template",
        expected=True,
    )
    _require_bool(
        manual.get("initial_values_are_blank"),
        field=f"{source}.manual_review.initial_values_are_blank",
        expected=True,
    )
    for key in (
        "allowed_crop_valid_values",
        "allowed_back_facing_values",
        "allowed_number_visible_values",
        "allowed_number_readable_values",
        "allowed_digit_count_values",
    ):
        values = manual.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, str)
            or not values
            or len(set(values)) != len(values)
            or not all(isinstance(v, str) and v for v in values)
        ):
            raise JerseyReviewError(
                f"{source}: manual_review.{key} must be unique non-empty strings"
            )

    recognition = _mapping(payload, "recognition")
    for key in (
        "OCR_enabled",
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

    safety = _mapping(payload, "safety")
    for key, expected in (
        ("source_crops_immutable", True),
        ("measurement_artifacts_immutable", True),
        ("panels_are_display_only", True),
        ("panel_images_are_not_recognition_input", True),
        ("identity_ground_truth_available", False),
        ("accuracy_claim_allowed", False),
        ("team_assignment_enabled", False),
        ("global_id_rewrite_enabled", False),
    ):
        _require_bool(
            safety.get(key), field=f"{source}.safety.{key}", expected=expected
        )

    return dict(payload)


def load_jersey_review_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise JerseyReviewError(f"config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JerseyReviewError(f"invalid YAML in {config_path}: {exc}") from exc
    return validate_jersey_review_config(payload, source=str(config_path))


def _validate_crop_signal_row(
    row: Mapping[str, Any], *, expected_schema: str, index: int
) -> None:
    field = f"crop_signals row {index}"
    if row.get("schema_version") != expected_schema:
        raise JerseyReviewError(f"{field}: schema mismatch")
    _require_str(row.get("crop_id"), field=f"{field}.crop_id")
    _require_str(row.get("segment_id"), field=f"{field}.segment_id")
    _require_int(row.get("raw_track_id"), field=f"{field}.raw_track_id", minimum=1)
    if row.get("segment_kind") not in SEGMENT_KINDS:
        raise JerseyReviewError(f"{field}: invalid segment_kind")
    if row.get("crop_source_kind") not in CROP_SOURCE_KINDS:
        raise JerseyReviewError(f"{field}: invalid crop_source_kind")
    if row.get("representation_source") not in REPRESENTATION_SOURCES:
        raise JerseyReviewError(f"{field}: invalid representation_source")
    _require_int(row.get("frame_index"), field=f"{field}.frame_index")
    _require_int(row.get("selection_rank"), field=f"{field}.selection_rank", minimum=1)
    _require_str(row.get("source_crop_path"), field=f"{field}.source_crop_path")
    sha = _require_str(row.get("source_crop_sha256"), field=f"{field}.source_crop_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise JerseyReviewError(f"{field}: invalid source_crop_sha256")
    width = _require_int(row.get("crop_width_px"), field=f"{field}.crop_width_px", minimum=1)
    height = _require_int(
        row.get("crop_height_px"), field=f"{field}.crop_height_px", minimum=1
    )
    x_min = _require_int(row.get("roi_x_min"), field=f"{field}.roi_x_min")
    y_min = _require_int(row.get("roi_y_min"), field=f"{field}.roi_y_min")
    x_max = _require_int(row.get("roi_x_max"), field=f"{field}.roi_x_max", minimum=1)
    y_max = _require_int(row.get("roi_y_max"), field=f"{field}.roi_y_max", minimum=1)
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise JerseyReviewError(f"{field}: ROI outside crop bounds")
    if row.get("roi_width_px") != x_max - x_min or row.get("roi_height_px") != (
        y_max - y_min
    ):
        raise JerseyReviewError(f"{field}: ROI dimensions inconsistent")
    for metric in (
        "laplacian_variance",
        "tenengrad_mean",
        "local_contrast",
        "entropy",
        "edge_density",
        "grayscale_mean",
        "grayscale_std",
    ):
        _finite(row.get(metric), field=f"{field}.{metric}")
    _ratio(
        row.get("roi_other_person_union_coverage"),
        field=f"{field}.roi_other_person_union_coverage",
    )
    _ratio(
        row.get("full_crop_other_person_union_coverage"),
        field=f"{field}.full_crop_other_person_union_coverage",
    )
    _require_int(
        row.get("roi_other_person_center_inside_count"),
        field=f"{field}.roi_other_person_center_inside_count",
    )
    stratum = _require_str(
        row.get("sharpness_size_stratum"), field=f"{field}.sharpness_size_stratum"
    )
    if not _STRATUM_NAME_RE.fullmatch(stratum):
        raise JerseyReviewError(f"{field}: invalid sharpness_size_stratum {stratum!r}")
    for rank_field in GLOBAL_RANK_FIELDS + (
        "laplacian_rank_within_size_stratum",
        "tenengrad_rank_within_size_stratum",
    ):
        _require_int(row.get(rank_field), field=f"{field}.{rank_field}", minimum=1)


def load_measurement_inputs(
    *,
    measurement_dir: str | Path,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(measurement_dir).expanduser().resolve()
    if not root.is_dir():
        raise JerseyReviewError(f"measurement directory not found: {root}")
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise JerseyReviewError(f"project root not found: {project}")

    crop_rows = _load_jsonl(root / MEASUREMENT_CROP_SIGNALS_NAME)
    segment_rows = _load_jsonl(root / MEASUREMENT_SEGMENT_SUMMARY_NAME)
    summary = _load_json(root / MEASUREMENT_SUMMARY_NAME)

    input_cfg = config["input"]
    if input_cfg["require_measurement_status_ok"] and summary.get("status") != "ok":
        raise JerseyReviewError("measurement summary status is not ok")
    crop_schema = input_cfg["require_crop_signal_schema"]
    segment_schema = input_cfg["require_segment_summary_schema"]

    segment_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(segment_rows):
        if row.get("schema_version") != segment_schema:
            raise JerseyReviewError(f"segment summary row {index}: schema mismatch")
        sid = _require_str(row.get("segment_id"), field=f"segment row {index}.segment_id")
        if sid in segment_by_id:
            raise JerseyReviewError(f"duplicate segment summary id: {sid}")
        segment_by_id[sid] = row

    crop_ids: set[str] = set()
    for index, row in enumerate(crop_rows):
        _validate_crop_signal_row(row, expected_schema=crop_schema, index=index)
        crop_id = str(row["crop_id"])
        if crop_id in crop_ids:
            raise JerseyReviewError(f"duplicate crop signal id: {crop_id}")
        crop_ids.add(crop_id)
        sid = str(row["segment_id"])
        segment = segment_by_id.get(sid)
        if segment is None:
            raise JerseyReviewError(f"crop {crop_id} references unknown segment {sid}")
        if segment.get("measurement_status") != "measured_selected_crops":
            raise JerseyReviewError(
                f"crop {crop_id} belongs to non-measured segment {sid}"
            )

    total = len(crop_rows)
    if int(summary.get("total_selected_crop_count", -1)) != total:
        raise JerseyReviewError("summary total_selected_crop_count mismatch")
    if int(summary.get("total_derived_segment_count", -1)) != len(segment_rows):
        raise JerseyReviewError("summary total_derived_segment_count mismatch")
    measured = sum(
        1
        for row in segment_rows
        if row.get("measurement_status") == "measured_selected_crops"
    )
    no_crop = sum(
        1
        for row in segment_rows
        if row.get("measurement_status") == "no_selected_crop_provenance"
    )
    if measured + no_crop != len(segment_rows):
        raise JerseyReviewError("segment summary contains unknown measurement_status")
    if int(summary.get("measured_segment_count", -1)) != measured:
        raise JerseyReviewError("summary measured_segment_count mismatch")
    if int(summary.get("no_selected_crop_segment_count", -1)) != no_crop:
        raise JerseyReviewError("summary no_selected_crop_segment_count mismatch")
    selected_total = sum(
        _require_int(
            row.get("selected_crop_count"),
            field=f"segment {row.get('segment_id')}.selected_crop_count",
        )
        for row in segment_rows
    )
    if selected_total != total:
        raise JerseyReviewError("segment selected_crop_count sum mismatch")

    for rank_field in GLOBAL_RANK_FIELDS:
        ranks = sorted(int(row[rank_field]) for row in crop_rows)
        if ranks != list(range(1, total + 1)):
            raise JerseyReviewError(f"{rank_field} is not a permutation of 1..N")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in crop_rows:
        by_stratum[str(row["sharpness_size_stratum"])].append(row)
    for stratum, rows in by_stratum.items():
        for rank_field in (
            "laplacian_rank_within_size_stratum",
            "tenengrad_rank_within_size_stratum",
        ):
            ranks = sorted(int(row[rank_field]) for row in rows)
            if ranks != list(range(1, len(rows) + 1)):
                raise JerseyReviewError(
                    f"{rank_field} invalid within stratum {stratum}"
                )

    require_sha = bool(input_cfg["require_source_crop_sha_match"])
    images: dict[str, np.ndarray] = {}
    for row in crop_rows:
        crop_id = str(row["crop_id"])
        path = Path(str(row["source_crop_path"])).expanduser().resolve()
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise JerseyReviewError(
                f"source crop escapes project root: {path}"
            ) from exc
        if not path.is_file():
            raise JerseyReviewError(f"source crop missing: {path}")
        if require_sha and sha256_file(path) != row["source_crop_sha256"]:
            raise JerseyReviewError(f"source crop SHA mismatch: {crop_id}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise JerseyReviewError(f"could not decode source crop: {path}")
        height, width = image.shape[:2]
        if width != int(row["crop_width_px"]) or height != int(row["crop_height_px"]):
            raise JerseyReviewError(f"source crop dimensions mismatch: {crop_id}")
        images[crop_id] = image

    return {
        "measurement_dir": root,
        "project_root": project,
        "crop_rows": crop_rows,
        "segment_rows": segment_rows,
        "segment_by_id": segment_by_id,
        "summary": summary,
        "images": images,
    }


def _canonical_sort_key(row: Mapping[str, Any]) -> tuple:
    return (
        int(row["raw_track_id"]),
        str(row["segment_id"]),
        int(row["frame_index"]),
        int(row["selection_rank"]),
        str(row["crop_id"]),
    )


def build_canonical_review_items(
    crop_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(crop_rows, key=_canonical_sort_key)
    items: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        item = {
            "schema_version": REVIEW_ITEM_SCHEMA,
            "review_item_id": f"review_{row['crop_id']}",
            "review_index": index,
            "segment_id": str(row["segment_id"]),
            "raw_track_id": int(row["raw_track_id"]),
            "segment_kind": str(row["segment_kind"]),
            "crop_id": str(row["crop_id"]),
            "frame_index": int(row["frame_index"]),
            "selection_rank": int(row["selection_rank"]),
            "representation_source": str(row["representation_source"]),
            "crop_source_kind": str(row["crop_source_kind"]),
            "source_crop_path": str(row["source_crop_path"]),
            "source_crop_sha256": str(row["source_crop_sha256"]),
            "crop_width_px": int(row["crop_width_px"]),
            "crop_height_px": int(row["crop_height_px"]),
            "roi_x_min": int(row["roi_x_min"]),
            "roi_y_min": int(row["roi_y_min"]),
            "roi_x_max": int(row["roi_x_max"]),
            "roi_y_max": int(row["roi_y_max"]),
            "roi_width_px": int(row["roi_width_px"]),
            "roi_height_px": int(row["roi_height_px"]),
            "laplacian_variance": float(row["laplacian_variance"]),
            "tenengrad_mean": float(row["tenengrad_mean"]),
            "local_contrast": float(row["local_contrast"]),
            "entropy": float(row["entropy"]),
            "edge_density": float(row["edge_density"]),
            "roi_other_person_union_coverage": float(
                row["roi_other_person_union_coverage"]
            ),
            "roi_other_person_center_inside_count": int(
                row["roi_other_person_center_inside_count"]
            ),
            "sharpness_size_stratum": str(row["sharpness_size_stratum"]),
            **{
                rank_field: int(row[rank_field])
                for rank_field in GLOBAL_RANK_FIELDS
                + (
                    "laplacian_rank_within_size_stratum",
                    "tenengrad_rank_within_size_stratum",
                )
            },
            "group_memberships": [],
            "master_panel_path": None,
            "master_panel_page": None,
            "master_panel_tile_index": None,
        }
        for field in MANUAL_FIELDS:
            item[field] = None
        items.append(item)
    return items


def _group_dirname(order: int, name: str) -> str:
    fixed_names = {group_name for group_name, _ in FIXED_GROUPS}
    if name in fixed_names:
        return f"{order:02d}_{name}"
    if name == "critical_segments" or name.startswith("sharpness_"):
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise JerseyReviewError(f"unsafe group name: {name}")
        return name
    raise JerseyReviewError(f"unknown group name: {name}")


def build_review_groups(
    *,
    items: Sequence[Mapping[str, Any]],
    segment_by_id: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    groups_cfg = config["groups"]
    by_crop_id = {str(item["crop_id"]): item for item in items}
    canonical_order = {str(item["crop_id"]): int(item["review_index"]) for item in items}

    def ordered_by_rank(
        source: Sequence[Mapping[str, Any]],
        rank_field: str,
        *,
        reverse: bool,
        count: int,
    ) -> list[Mapping[str, Any]]:
        ordered = sorted(
            source,
            key=lambda row: (
                -int(row[rank_field]) if reverse else int(row[rank_field]),
                canonical_order[str(row["crop_id"])],
            ),
        )
        return list(ordered[:count])

    groups: list[dict[str, Any]] = []
    order = 0

    def add_group(
        name: str,
        selected: Sequence[Mapping[str, Any]],
        *,
        requested: int | None,
        metric_name: str | None,
        rank_source: str | None,
        stratum: str | None = None,
        empty_reason: str | None = None,
        notes: list[str] | None = None,
    ) -> None:
        nonlocal order
        crop_ids = [str(row["crop_id"]) for row in selected]
        if len(set(crop_ids)) != len(crop_ids):
            raise JerseyReviewError(f"duplicate crop within group {name}")
        groups.append(
            {
                "group_name": name,
                "group_order": order,
                "group_dirname": _group_dirname(order, name),
                "requested_count": requested,
                "items": [by_crop_id[cid] for cid in crop_ids],
                "metric_name": metric_name,
                "rank_source": rank_source,
                "sharpness_size_stratum": stratum,
                "empty_reason": empty_reason,
                "notes": notes or [],
            }
        )
        order += 1

    add_group(
        "all_selected_crops",
        list(items),
        requested=None,
        metric_name=None,
        rank_source="canonical_review_order",
    )

    if groups_cfg["all_manual_segment_crops"]["enabled"]:
        manual_items = [
            item for item in items if item["segment_kind"] == "manual_split_segment"
        ]
        add_group(
            "all_manual_segment_crops",
            manual_items,
            requested=None,
            metric_name=None,
            rank_source="canonical_review_order",
            empty_reason=None if manual_items else "no_manual_segment_crops",
        )

    for name, (rank_field, metric_name, reverse) in METRIC_GROUP_SPECS.items():
        group_cfg = groups_cfg[name]
        if not group_cfg["enabled"]:
            continue
        count = int(group_cfg["count"])
        selected = ordered_by_rank(items, rank_field, reverse=reverse, count=count)
        notes = []
        if len(selected) < count:
            notes.append(f"requested {count} but only {len(selected)} available")
        add_group(
            name,
            selected,
            requested=count,
            metric_name=metric_name,
            rank_source=rank_field,
            notes=notes,
        )

    sharp_cfg = groups_cfg["sharpness_by_roi_height_stratum"]
    empty_strata: list[str] = []
    if sharp_cfg["enabled"]:
        by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            by_stratum[str(item["sharpness_size_stratum"])].append(item)

        def stratum_key(name: str) -> tuple[int, int]:
            low, high = name.split("_", 1)
            return (int(low), int(high))

        for stratum in sorted(by_stratum, key=stratum_key):
            stratum_items = by_stratum[stratum]
            for suffix, rank_field, metric_name, reverse in SHARPNESS_VARIANTS:
                count = int(sharp_cfg[f"{suffix}_count"])
                selected = ordered_by_rank(
                    stratum_items, rank_field, reverse=reverse, count=count
                )
                notes = []
                if len(selected) < count:
                    notes.append(
                        f"requested {count} but only {len(selected)} available"
                    )
                add_group(
                    f"sharpness_{stratum}_{suffix}",
                    selected,
                    requested=count,
                    metric_name=metric_name,
                    rank_source=rank_field,
                    stratum=stratum,
                    notes=notes,
                )

    critical_cfg = groups_cfg["critical_segments"]
    critical_no_crop: list[str] = []
    if critical_cfg["enabled"]:
        wanted = [str(sid) for sid in critical_cfg["segment_ids"]]
        for sid in wanted:
            segment = segment_by_id.get(sid)
            if segment is None:
                raise JerseyReviewError(f"critical segment not found: {sid}")
            if segment.get("measurement_status") != "measured_selected_crops":
                critical_no_crop.append(sid)
        wanted_set = set(wanted)
        selected = [item for item in items if item["segment_id"] in wanted_set]
        add_group(
            "critical_segments",
            selected,
            requested=None,
            metric_name=None,
            rank_source="canonical_review_order",
            empty_reason=None if selected else "critical_segments_have_no_crops",
            notes=(
                [f"no-crop critical segments: {sorted(critical_no_crop)}"]
                if critical_no_crop
                else []
            ),
        )

    return {
        "groups": groups,
        "critical_no_crop_segments": sorted(critical_no_crop),
        "empty_strata": empty_strata,
    }


def _fit_display(
    image: np.ndarray, box_width: int, box_height: int
) -> tuple[np.ndarray, float]:
    """Aspect-preserving nearest-neighbor display resize (display only)."""
    height, width = image.shape[:2]
    scale = min(box_width / width, box_height / height)
    new_width = max(1, int(math.floor(width * scale)))
    new_height = max(1, int(math.floor(height * scale)))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    return resized, new_width / width


def _put_line(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        canvas,
        text[:44],
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def render_tile(
    item: Mapping[str, Any],
    image: np.ndarray,
    *,
    group: Mapping[str, Any],
) -> np.ndarray:
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), TILE_BACKGROUND, dtype=np.uint8)
    inner_width = TILE_WIDTH - 2 * TILE_PADDING
    x0 = TILE_PADDING
    line_y = 14
    metric_name = group.get("metric_name")
    if metric_name:
        metric_text = f"{metric_name}={float(item[metric_name]):.4g}"
    else:
        metric_text = (
            f"roi_h={item['roi_height_px']} lc={item['local_contrast']:.3g} "
            f"cont={item['roi_other_person_union_coverage']:.3g}"
        )
    lines = (
        f"#{item['review_index']} {item['segment_id']}",
        f"{item['crop_id']}",
        f"frame={item['frame_index']} src={item['crop_source_kind'][:20]}",
        f"crop={item['crop_width_px']}x{item['crop_height_px']} "
        f"roi={item['roi_width_px']}x{item['roi_height_px']}",
        metric_text,
        f"roi_contamination={item['roi_other_person_union_coverage']:.4g}",
    )
    for text in lines:
        _put_line(tile, text, x0, line_y)
        line_y += 15

    original_top = TEXT_AREA_HEIGHT
    display, scale = _fit_display(image, inner_width, ORIGINAL_AREA_HEIGHT)
    disp_h, disp_w = display.shape[:2]
    ox = x0 + (inner_width - disp_w) // 2
    oy = original_top + (ORIGINAL_AREA_HEIGHT - disp_h) // 2
    tile[oy : oy + disp_h, ox : ox + disp_w] = display
    rx0 = ox + int(round(item["roi_x_min"] * scale))
    ry0 = oy + int(round(item["roi_y_min"] * scale))
    rx1 = ox + int(round(item["roi_x_max"] * scale))
    ry1 = oy + int(round(item["roi_y_max"] * scale))
    cv2.rectangle(tile, (rx0, ry0), (max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)),
                  ROI_RECT_COLOR, 1)

    zoom_label_y = original_top + ORIGINAL_AREA_HEIGHT + 12
    _put_line(tile, "ROI DISPLAY ZOOM (not recognition input)", x0, zoom_label_y)
    zoom_top = zoom_label_y + 6
    zoom_height = TILE_HEIGHT - zoom_top - TILE_PADDING
    roi_pixels = image[
        item["roi_y_min"] : item["roi_y_max"],
        item["roi_x_min"] : item["roi_x_max"],
    ]
    zoom, _ = _fit_display(roi_pixels, inner_width, zoom_height)
    zh, zw = zoom.shape[:2]
    zx = x0 + (inner_width - zw) // 2
    zy = zoom_top + (zoom_height - zh) // 2
    tile[zy : zy + zh, zx : zx + zw] = zoom
    return tile


def _write_panel_png(image: np.ndarray, path: Path, *, temp_root: Path) -> None:
    resolved = path.resolve()
    if resolved.suffix != ".png":
        raise JerseyReviewError(f"panel output must be PNG: {resolved}")
    try:
        resolved.relative_to(temp_root.resolve())
    except ValueError as exc:
        raise JerseyReviewError(
            f"panel write outside temporary output: {resolved}"
        ) from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved), image):
        raise JerseyReviewError(f"failed to write panel PNG: {resolved}")


def render_group_panels(
    *,
    group: Mapping[str, Any],
    images: Mapping[str, np.ndarray],
    temp_root: Path,
    columns: int,
    rows: int,
) -> list[dict[str, Any]]:
    per_page = columns * rows
    items = list(group["items"])
    pages: list[dict[str, Any]] = []
    if not items:
        return pages
    page_count = (len(items) + per_page - 1) // per_page
    for page_number in range(1, page_count + 1):
        chunk = items[(page_number - 1) * per_page : page_number * per_page]
        canvas = np.full(
            (rows * TILE_HEIGHT, columns * TILE_WIDTH, 3),
            CANVAS_BACKGROUND,
            dtype=np.uint8,
        )
        tile_ids: list[str] = []
        for tile_index in range(per_page):
            row_i, col_i = divmod(tile_index, columns)
            y0 = row_i * TILE_HEIGHT
            x0 = col_i * TILE_WIDTH
            if tile_index < len(chunk):
                item = chunk[tile_index]
                tile = render_tile(
                    item, images[str(item["crop_id"])], group=group
                )
                tile_ids.append(str(item["review_item_id"]))
            else:
                tile = np.full(
                    (TILE_HEIGHT, TILE_WIDTH, 3),
                    EMPTY_TILE_BACKGROUND,
                    dtype=np.uint8,
                )
            canvas[y0 : y0 + TILE_HEIGHT, x0 : x0 + TILE_WIDTH] = tile
        relative = (
            f"{PANELS_DIRNAME}/{group['group_dirname']}/page_{page_number:03d}.png"
        )
        absolute = temp_root / relative
        _write_panel_png(canvas, absolute, temp_root=temp_root)
        pages.append(
            {
                "panel_path": relative,
                "page_number": page_number,
                "items": chunk,
                "review_item_ids": tile_ids,
            }
        )
    return pages


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            handle.write("\n")


def _write_annotation_template(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(ANNOTATION_COLUMNS)
        for item in items:
            row = [
                item["review_item_id"],
                item["review_index"],
                item["segment_id"],
                item["raw_track_id"],
                item["crop_id"],
                item["frame_index"],
                item["master_panel_path"],
                item["master_panel_page"],
                item["master_panel_tile_index"],
                json.dumps(item["group_memberships"], ensure_ascii=False),
            ]
            row.extend("" for _ in MANUAL_FIELDS)
            writer.writerow(row)


def validate_temp_review_outputs(
    *,
    temp_dir: Path,
    items: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    panel_index: Sequence[Mapping[str, Any]],
    columns: int,
    rows: int,
) -> None:
    actual = sorted(path.name for path in temp_dir.iterdir())
    if actual != sorted(list(OUTPUT_NAMES) + [PANELS_DIRNAME]):
        raise JerseyReviewError(f"unexpected output entries: {actual}")
    for entry in (temp_dir / PANELS_DIRNAME).rglob("*"):
        if entry.is_file() and entry.suffix != ".png":
            raise JerseyReviewError(f"non-PNG file under panels/: {entry}")

    written_items = _load_jsonl(temp_dir / REVIEW_ITEMS_NAME)
    if len(written_items) != len(items):
        raise JerseyReviewError("written review item count mismatch")
    ids = [str(row["review_item_id"]) for row in written_items]
    if len(set(ids)) != len(ids):
        raise JerseyReviewError("review item IDs not unique")
    indexes = [int(row["review_index"]) for row in written_items]
    if sorted(indexes) != list(range(1, len(items) + 1)):
        raise JerseyReviewError("review index not contiguous 1..N")
    for row in written_items:
        if row.get("schema_version") != REVIEW_ITEM_SCHEMA:
            raise JerseyReviewError("review item schema mismatch")
        for field in MANUAL_FIELDS:
            if row.get(field) is not None:
                raise JerseyReviewError(f"manual field not blank: {field}")
        if "all_selected_crops" not in row.get("group_memberships", []):
            raise JerseyReviewError("item missing master group membership")
        if not row.get("master_panel_path"):
            raise JerseyReviewError("item missing master panel placement")

    written_memberships = _load_jsonl(temp_dir / GROUP_MEMBERSHIPS_NAME)
    if len(written_memberships) != len(memberships):
        raise JerseyReviewError("membership row count mismatch")
    item_ids = set(ids)
    for row in written_memberships:
        if row.get("schema_version") != GROUP_MEMBERSHIP_SCHEMA:
            raise JerseyReviewError("membership schema mismatch")
        if str(row.get("review_item_id")) not in item_ids:
            raise JerseyReviewError("membership references unknown review item")

    written_index = _load_jsonl(temp_dir / PANEL_INDEX_NAME)
    if len(written_index) != len(panel_index):
        raise JerseyReviewError("panel index row count mismatch")
    for row in written_index:
        if row.get("schema_version") != PANEL_INDEX_SCHEMA:
            raise JerseyReviewError("panel index schema mismatch")
        panel_path = temp_dir / str(row["panel_path"])
        if not panel_path.is_file():
            raise JerseyReviewError(f"panel file missing: {row['panel_path']}")
        if sha256_file(panel_path) != row["panel_sha256"]:
            raise JerseyReviewError(f"panel SHA mismatch: {row['panel_path']}")
        if panel_path.stat().st_size != int(row["panel_byte_size"]):
            raise JerseyReviewError(f"panel byte size mismatch: {row['panel_path']}")
        decoded = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise JerseyReviewError(f"panel PNG not decodable: {row['panel_path']}")
        if decoded.shape[:2] != (rows * TILE_HEIGHT, columns * TILE_WIDTH):
            raise JerseyReviewError(f"panel dimensions unexpected: {row['panel_path']}")
        if len(row["review_item_ids"]) != int(row["tile_count"]):
            raise JerseyReviewError("panel tile count mismatch")
        for rid in row["review_item_ids"]:
            if str(rid) not in item_ids:
                raise JerseyReviewError("panel references unknown review item")

    csv_text = (temp_dir / ANNOTATION_TEMPLATE_NAME).read_text(encoding="utf-8")
    if not csv_text.endswith("\n"):
        raise JerseyReviewError("annotation template lacks final newline")
    reader = list(csv.reader(csv_text.splitlines()))
    if reader[0] != list(ANNOTATION_COLUMNS):
        raise JerseyReviewError("annotation template header mismatch")
    if len(reader) - 1 != len(items):
        raise JerseyReviewError("annotation template row count mismatch")
    manual_start = len(ANNOTATION_COLUMNS) - len(MANUAL_FIELDS)
    for row in reader[1:]:
        if any(value != "" for value in row[manual_start:]):
            raise JerseyReviewError("annotation template manual fields not blank")

    summary = _load_json(temp_dir / SUMMARY_NAME)
    if summary.get("schema_version") != SUMMARY_SCHEMA or summary.get("status") != "ok":
        raise JerseyReviewError("review summary schema/status mismatch")
    for name in OUTPUT_NAMES:
        if not (temp_dir / name).read_bytes().endswith(b"\n"):
            raise JerseyReviewError(f"{name} lacks final newline")


def create_temp_jersey_review_dir(output_dir: Path) -> Path:
    final = output_dir.expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.parent / (
        f"_tmp_reid_jersey_review_{final.name}_{uuid.uuid4().hex[:8]}"
    )
    temp.mkdir(parents=False, exist_ok=False)
    return temp


def finalize_jersey_review_dir(
    *, temp_dir: Path, final_dir: Path, overwrite: bool
) -> Path:
    temp = temp_dir.expanduser().resolve()
    final = final_dir.expanduser().resolve()
    if not temp.is_dir():
        raise JerseyReviewError(f"temporary output missing: {temp}")
    backup: Path | None = None
    try:
        if final.exists():
            if not overwrite:
                raise JerseyReviewError(f"output already exists: {final}")
            backup = final.with_name(
                f"_backup_reid_jersey_review_{final.name}_{uuid.uuid4().hex[:8]}"
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
    for prefix in ("_tmp_reid_jersey_review_", "_backup_reid_jersey_review_"):
        for stray in final.parent.glob(f"{prefix}{final.name}_*"):
            cleanup_dir(stray)
    return final


def _artifact_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def run_build_jersey_review_panels(
    *,
    measurement_dir: str | Path,
    project_root: str | Path,
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
        raise JerseyReviewError(str(exc)) from exc

    validated_config = load_jersey_review_config(config_path)
    inputs = load_measurement_inputs(
        measurement_dir=measurement_dir,
        project_root=project_root,
        config=validated_config,
    )
    for row in inputs["crop_rows"]:
        crop_path = Path(str(row["source_crop_path"])).resolve()
        try:
            final_dir.relative_to(crop_path.parent)
        except ValueError:
            continue
        raise JerseyReviewError("output directory would overlap source crops")

    items = build_canonical_review_items(inputs["crop_rows"])
    grouping = build_review_groups(
        items=items,
        segment_by_id=inputs["segment_by_id"],
        config=validated_config,
    )
    groups = grouping["groups"]
    columns = int(validated_config["panel"]["columns"])
    rows = int(validated_config["panel"]["rows"])

    item_by_id = {str(item["review_item_id"]): item for item in items}
    source_shas_before = {
        str(row["crop_id"]): str(row["source_crop_sha256"])
        for row in inputs["crop_rows"]
    }

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_jersey_review_dir(final_dir)
        (temp_dir / PANELS_DIRNAME).mkdir()

        memberships: list[dict[str, Any]] = []
        panel_index: list[dict[str, Any]] = []
        group_summaries: list[dict[str, Any]] = []
        for group in groups:
            pages = render_group_panels(
                group=group,
                images=inputs["images"],
                temp_root=temp_dir,
                columns=columns,
                rows=rows,
            )
            placement: dict[str, tuple[str, int, int]] = {}
            for page in pages:
                for tile_index, rid in enumerate(page["review_item_ids"]):
                    placement[rid] = (
                        page["panel_path"],
                        page["page_number"],
                        tile_index,
                    )
                panel_path = temp_dir / page["panel_path"]
                panel_index.append(
                    {
                        "schema_version": PANEL_INDEX_SCHEMA,
                        "group_name": group["group_name"],
                        "panel_path": page["panel_path"],
                        "page_number": page["page_number"],
                        "tile_count": len(page["review_item_ids"]),
                        "review_item_ids": page["review_item_ids"],
                        "source_crop_sha256_values": [
                            item_by_id[rid]["source_crop_sha256"]
                            for rid in page["review_item_ids"]
                        ],
                        "panel_sha256": sha256_file(panel_path),
                        "panel_byte_size": panel_path.stat().st_size,
                    }
                )
            for rank, item in enumerate(group["items"], start=1):
                rid = str(item["review_item_id"])
                item["group_memberships"].append(group["group_name"])
                panel_path, page_number, tile_index = placement[rid]
                if group["group_name"] == "all_selected_crops":
                    item["master_panel_path"] = panel_path
                    item["master_panel_page"] = page_number
                    item["master_panel_tile_index"] = tile_index
                metric_name = group["metric_name"]
                memberships.append(
                    {
                        "schema_version": GROUP_MEMBERSHIP_SCHEMA,
                        "group_name": group["group_name"],
                        "group_order": group["group_order"],
                        "group_item_rank": rank,
                        "review_item_id": rid,
                        "segment_id": item["segment_id"],
                        "crop_id": item["crop_id"],
                        "metric_name": metric_name,
                        "metric_value": (
                            float(item[metric_name]) if metric_name else None
                        ),
                        "sharpness_size_stratum": group["sharpness_size_stratum"],
                        "panel_path": panel_path,
                        "panel_page": page_number,
                        "panel_tile_index": tile_index,
                    }
                )
            group_summaries.append(
                {
                    "group_name": group["group_name"],
                    "group_order": group["group_order"],
                    "requested_count": group["requested_count"],
                    "actual_count": len(group["items"]),
                    "panel_page_count": len(pages),
                    "empty_reason": group["empty_reason"],
                    "duplicate_item_count_within_group": 0,
                    "metric_name": group["metric_name"],
                    "rank_source": group["rank_source"],
                    "sharpness_size_stratum": group["sharpness_size_stratum"],
                    "notes": group["notes"],
                }
            )

        memberships.sort(
            key=lambda row: (
                int(row["group_order"]),
                int(row["group_item_rank"]),
                str(row["review_item_id"]),
            )
        )
        panel_index.sort(
            key=lambda row: (str(row["panel_path"]), int(row["page_number"]))
        )

        for crop_id, sha in source_shas_before.items():
            row = next(
                r for r in inputs["crop_rows"] if str(r["crop_id"]) == crop_id
            )
            if sha256_file(Path(str(row["source_crop_path"]))) != sha:
                raise JerseyReviewError(f"source crop changed during run: {crop_id}")

        total_pngs = sum(1 for _ in (temp_dir / PANELS_DIRNAME).rglob("*.png"))
        empty_groups = [g for g in group_summaries if g["actual_count"] == 0]
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "ok",
            "source_provenance": {
                "measurement_crop_signals": _artifact_provenance(
                    inputs["measurement_dir"] / MEASUREMENT_CROP_SIGNALS_NAME
                ),
                "measurement_segment_summary": _artifact_provenance(
                    inputs["measurement_dir"] / MEASUREMENT_SEGMENT_SUMMARY_NAME
                ),
                "measurement_summary": _artifact_provenance(
                    inputs["measurement_dir"] / MEASUREMENT_SUMMARY_NAME
                ),
                "config": _artifact_provenance(config_path),
                "project_root": str(inputs["project_root"]),
            },
            "counts": {
                "crop_signal_count": len(inputs["crop_rows"]),
                "canonical_review_item_count": len(items),
                "master_panel_item_count": len(groups[0]["items"]),
                "manual_segment_crop_count": sum(
                    1
                    for item in items
                    if item["segment_kind"] == "manual_split_segment"
                ),
                "group_count": len(groups),
                "empty_group_count": len(empty_groups),
                "group_membership_count": len(memberships),
                "total_panel_page_count": len(panel_index),
                "total_panel_png_count": total_pngs,
                "annotation_template_row_count": len(items),
            },
            "groups": group_summaries,
            "critical_no_crop_segments": grouping["critical_no_crop_segments"],
            "panel_geometry": {
                "columns": columns,
                "rows": rows,
                "tile_width_px": TILE_WIDTH,
                "tile_height_px": TILE_HEIGHT,
                "page_width_px": columns * TILE_WIDTH,
                "page_height_px": rows * TILE_HEIGHT,
                "annotation_group_memberships_encoding": "json_array_string",
                "tile_index_base": 0,
            },
            "safety": {
                "video_opened": False,
                "new_crop_extraction_performed": False,
                "source_crop_modified": False,
                "measurement_artifact_modified": False,
                "OCR_performed": False,
                "recognizer_performed": False,
                "checkpoint_loaded": False,
                "image_enhancement_performed": False,
                "automatic_visibility_classification": False,
                "automatic_readability_classification": False,
                "back_facing_classification": False,
                "jersey_number_candidate_generated": False,
                "automatic_jersey_assignment": False,
                "team_assignment": False,
                "global_id_rewrite": False,
                "identity_ground_truth_available": False,
                "accuracy_claimed": False,
            },
            "semantics_and_limitations": [
                "panels are display-only review artifacts",
                "ROI zoom is nearest-neighbor display enlargement",
                "ROI zoom is not recognition input",
                "upper-torso ROI does not prove number visibility",
                "sharpness/contrast/entropy do not prove readability",
                "contamination zero does not prove crop purity",
                "manual annotation fields are blank",
                "same crop may belong to multiple groups",
                "selected crops are not all observations",
                "no-selected-crop segments are excluded from visual review",
                "sample.mp4 is not a product constant",
            ],
            "elapsed_sec": time.perf_counter() - started,
        }

        _write_jsonl(temp_dir / REVIEW_ITEMS_NAME, items)
        _write_jsonl(temp_dir / GROUP_MEMBERSHIPS_NAME, memberships)
        _write_jsonl(temp_dir / PANEL_INDEX_NAME, panel_index)
        _write_annotation_template(temp_dir / ANNOTATION_TEMPLATE_NAME, items)
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
        validate_temp_review_outputs(
            temp_dir=temp_dir,
            items=items,
            memberships=memberships,
            panel_index=panel_index,
            columns=columns,
            rows=rows,
        )
        finalized = finalize_jersey_review_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "review_items_path": str(finalized / REVIEW_ITEMS_NAME),
        "group_memberships_path": str(finalized / GROUP_MEMBERSHIPS_NAME),
        "panel_index_path": str(finalized / PANEL_INDEX_NAME),
        "annotation_template_path": str(finalized / ANNOTATION_TEMPLATE_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "canonical_review_item_count": summary["counts"][
            "canonical_review_item_count"
        ],
        "group_count": summary["counts"]["group_count"],
        "total_panel_page_count": summary["counts"]["total_panel_page_count"],
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
    }
