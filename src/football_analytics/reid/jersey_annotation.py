"""Stage 5C-A2b3 manual jersey annotation validation.

Validates a filled or blank jersey review annotation CSV against the
immutable Stage 5C-A2b2 review artifacts. It never runs OCR, recognition,
identity/team assignment, gallery update, or image enhancement, and never
modifies review items, panels, or source crops.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from football_analytics.reid.jersey_review import (
    ANNOTATION_COLUMNS,
    ANNOTATION_TEMPLATE_NAME,
    REVIEW_ITEMS_NAME,
    SUMMARY_NAME,
)

CONFIG_SCHEMA = "reid_jersey_manual_review_config_v1"
REPORT_SCHEMA = "reid_jersey_annotation_validation_report_v1"
REVIEW_ITEM_SCHEMA = "reid_jersey_review_item_v1"
REVIEW_SUMMARY_SCHEMA = "reid_jersey_review_summary_v1"

PROVENANCE_COLUMNS = (
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
)

DECISION_FIELDS = (
    "manual_crop_valid",
    "manual_back_facing",
    "manual_number_visible",
    "manual_number_readable",
    "manual_digit_count",
    "manual_jersey_number",
    "manual_contamination_affects_number_region",
    "manual_notes",
)

CORE_REVIEWED_FIELDS = (
    "manual_crop_valid",
    "manual_back_facing",
    "manual_number_visible",
    "manual_number_readable",
    "manual_digit_count",
    "manual_contamination_affects_number_region",
)

ISO8601_AWARE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


class JerseyAnnotationError(RuntimeError):
    """Raised when Stage 5C-A2b3 annotation inputs or reports are invalid."""


def _reject_non_finite(value: str) -> None:
    raise JerseyAnnotationError(f"NaN/Infinity forbidden in JSON: {value}")


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise JerseyAnnotationError(f"{field} must be bool, got {value!r}")
    if expected is not None and value is not expected:
        raise JerseyAnnotationError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise JerseyAnnotationError(f"{field} must be a non-empty string")
    return value


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise JerseyAnnotationError(f"{key} must be a mapping")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JerseyAnnotationError(f"JSON not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite
        )
    except json.JSONDecodeError as exc:
        raise JerseyAnnotationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JerseyAnnotationError(f"{path} must contain an object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JerseyAnnotationError(f"JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite)
        except json.JSONDecodeError as exc:
            raise JerseyAnnotationError(
                f"invalid JSON on {path.name} line {line_no}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise JerseyAnnotationError(f"{path.name} line {line_no} must be an object")
        rows.append(row)
    if not rows:
        raise JerseyAnnotationError(f"JSONL is empty: {path}")
    return rows


def _unique_string_list(values: Any, *, field: str) -> list[str]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or not all(isinstance(v, str) and v for v in values)
        or len(set(values)) != len(values)
    ):
        raise JerseyAnnotationError(
            f"{field} must be a non-empty list of unique non-empty strings"
        )
    return list(values)


def validate_jersey_manual_review_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JerseyAnnotationError(f"{source}: config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise JerseyAnnotationError(f"{source}: invalid schema_version")
    if payload.get("stage_status") != "implementation_manual_annotation_protocol":
        raise JerseyAnnotationError(f"{source}: invalid stage_status")

    input_cfg = _mapping(payload, "input")
    _require_bool(
        input_cfg.get("require_review_summary_status_ok"),
        field=f"{source}.input.require_review_summary_status_ok",
        expected=True,
    )
    if input_cfg.get("require_review_item_schema") != REVIEW_ITEM_SCHEMA:
        raise JerseyAnnotationError(f"{source}: invalid require_review_item_schema")
    for key, expected in (
        ("require_exact_template_identity", True),
        ("allow_missing_review_rows", False),
        ("allow_extra_review_rows", False),
        ("allow_duplicate_review_items", False),
        ("require_source_review_item_match", True),
    ):
        _require_bool(
            input_cfg.get(key), field=f"{source}.input.{key}", expected=expected
        )

    review_state = _mapping(payload, "review_state")
    if review_state.get("blank_row_means") != "unreviewed":
        raise JerseyAnnotationError(f"{source}: blank_row_means must be unreviewed")
    for key, expected in (
        ("uncertain_is_explicit_decision", True),
        ("reviewed_row_requires_reviewer", True),
        ("reviewed_row_requires_reviewed_at", True),
    ):
        _require_bool(
            review_state.get(key),
            field=f"{source}.review_state.{key}",
            expected=expected,
        )

    allowed = _mapping(payload, "allowed_values")
    expected_allowed = {
        "manual_crop_valid": ["valid", "invalid", "uncertain"],
        "manual_back_facing": ["yes", "no", "uncertain"],
        "manual_number_visible": ["yes", "no", "uncertain"],
        "manual_number_readable": ["yes", "no", "uncertain"],
        "manual_digit_count": ["0", "1", "2", "uncertain"],
        "manual_contamination_affects_number_region": ["yes", "no", "uncertain"],
    }
    for key, expected in expected_allowed.items():
        values = _unique_string_list(
            allowed.get(key), field=f"{source}.allowed_values.{key}"
        )
        if values != expected:
            raise JerseyAnnotationError(
                f"{source}: allowed_values.{key} must be {expected}"
            )

    jersey = _mapping(payload, "jersey_number")
    if jersey.get("storage_type") != "string":
        raise JerseyAnnotationError(f"{source}: jersey_number.storage_type must be string")
    for key, expected in (
        ("allow_one_digit", True),
        ("allow_two_digits", True),
        ("preserve_leading_zero", True),
        ("allow_non_digit_characters", False),
        ("allow_more_than_two_digits", False),
        ("blank_when_not_readable", True),
    ):
        _require_bool(
            jersey.get(key), field=f"{source}.jersey_number.{key}", expected=expected
        )

    timestamp = _mapping(payload, "timestamp")
    if timestamp.get("format") != "iso8601":
        raise JerseyAnnotationError(f"{source}: timestamp.format must be iso8601")
    _require_bool(
        timestamp.get("timezone_required"),
        field=f"{source}.timestamp.timezone_required",
        expected=True,
    )

    validation = _mapping(payload, "validation")
    for key, expected in (
        ("allow_partial_dataset_review", True),
        ("unreviewed_rows_allowed", True),
        ("reject_partially_filled_reviewed_rows", True),
        ("reject_identity_labels", True),
        ("reject_automatic_predictions", True),
    ):
        _require_bool(
            validation.get(key),
            field=f"{source}.validation.{key}",
            expected=expected,
        )

    safety = _mapping(payload, "safety")
    for key, expected in (
        ("review_items_immutable", True),
        ("panels_immutable", True),
        ("source_crops_immutable", True),
        ("annotation_does_not_update_gallery", True),
        ("annotation_does_not_assign_identity", True),
        ("annotation_does_not_assign_team", True),
        ("identity_ground_truth_available", False),
        ("accuracy_claim_allowed", False),
    ):
        _require_bool(
            safety.get(key), field=f"{source}.safety.{key}", expected=expected
        )

    return dict(payload)


def load_jersey_manual_review_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise JerseyAnnotationError(f"config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JerseyAnnotationError(f"invalid YAML in {config_path}: {exc}") from exc
    return validate_jersey_manual_review_config(payload, source=str(config_path))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JerseyAnnotationError(f"CSV cells must be strings, got {value!r}")
    return value.strip()


def _is_blank(value: str) -> bool:
    return value == ""


def _parse_iso8601_aware(value: str) -> datetime:
    if not ISO8601_AWARE_RE.fullmatch(value):
        raise ValueError("timestamp must be ISO-8601 with timezone")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp timezone required")
    return parsed


def _row_error(
    *,
    code: str,
    message: str,
    review_item_id: str | None = None,
    review_index: int | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "review_item_id": review_item_id,
        "review_index": review_index,
        "field": field,
    }


def _validate_reviewed_row(
    row: Mapping[str, str],
    *,
    review_item_id: str,
    review_index: int,
    allowed: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def err(code: str, message: str, field: str | None = None) -> None:
        errors.append(
            _row_error(
                code=code,
                message=message,
                review_item_id=review_item_id,
                review_index=review_index,
                field=field,
            )
        )

    crop_valid = row["manual_crop_valid"]
    back_facing = row["manual_back_facing"]
    visible = row["manual_number_visible"]
    readable = row["manual_number_readable"]
    digit_count = row["manual_digit_count"]
    jersey = row["manual_jersey_number"]
    contamination = row["manual_contamination_affects_number_region"]
    notes = row["manual_notes"]
    reviewer = row["reviewer"]
    reviewed_at = row["reviewed_at"]

    for field, value in (
        ("manual_crop_valid", crop_valid),
        ("manual_back_facing", back_facing),
        ("manual_number_visible", visible),
        ("manual_number_readable", readable),
        ("manual_digit_count", digit_count),
        ("manual_contamination_affects_number_region", contamination),
    ):
        if not _is_blank(value) and value not in allowed[field]:
            err("invalid_allowed_value", f"{field}={value!r} not allowed", field)

    if _is_blank(reviewer):
        err("missing_reviewer", "reviewed row requires reviewer", "reviewer")
    if _is_blank(reviewed_at):
        err(
            "missing_reviewed_at",
            "reviewed row requires reviewed_at",
            "reviewed_at",
        )
    elif not _is_blank(reviewed_at):
        try:
            _parse_iso8601_aware(reviewed_at)
        except ValueError:
            err(
                "invalid_reviewed_at",
                "reviewed_at must be ISO-8601 with timezone",
                "reviewed_at",
            )

    if crop_valid == "invalid":
        if not _is_blank(back_facing) and back_facing != "uncertain":
            err(
                "invalid_crop_back_facing",
                "invalid crop allows only blank or uncertain back_facing",
                "manual_back_facing",
            )
        for field in (
            "manual_number_visible",
            "manual_number_readable",
            "manual_digit_count",
            "manual_jersey_number",
        ):
            if not _is_blank(row[field]):
                err(
                    "invalid_crop_field_filled",
                    f"invalid crop requires blank {field}",
                    field,
                )
        if not _is_blank(contamination) and contamination != "uncertain":
            err(
                "invalid_crop_contamination",
                "invalid crop allows only blank or uncertain contamination",
                "manual_contamination_affects_number_region",
            )
        return errors

    if crop_valid not in ("valid", "uncertain"):
        if _is_blank(crop_valid):
            err(
                "partial_reviewed_row",
                "reviewed row requires manual_crop_valid",
                "manual_crop_valid",
            )
        return errors

    for field in CORE_REVIEWED_FIELDS:
        if _is_blank(row[field]):
            err(
                "partial_reviewed_row",
                f"reviewed row requires {field}",
                field,
            )
    if any(_is_blank(row[field]) for field in CORE_REVIEWED_FIELDS):
        return errors

    if visible == "no":
        if readable != "no":
            err(
                "visible_no_readable_mismatch",
                "visible=no requires readable=no",
                "manual_number_readable",
            )
        if digit_count != "0":
            err(
                "visible_no_digit_count_mismatch",
                "visible=no requires digit_count=0",
                "manual_digit_count",
            )
        if not _is_blank(jersey):
            err(
                "visible_no_jersey_present",
                "visible=no requires blank jersey_number",
                "manual_jersey_number",
            )

    if visible == "uncertain":
        if readable == "yes":
            err(
                "visible_uncertain_readable_yes",
                "visible=uncertain forbids readable=yes",
                "manual_number_readable",
            )
        if not _is_blank(jersey):
            err(
                "visible_uncertain_jersey_present",
                "visible=uncertain requires blank jersey_number",
                "manual_jersey_number",
            )
        if digit_count not in ("0", "uncertain"):
            err(
                "visible_uncertain_digit_count",
                "visible=uncertain allows digit_count 0 or uncertain",
                "manual_digit_count",
            )

    if readable == "yes":
        if visible != "yes":
            err(
                "readable_yes_requires_visible_yes",
                "readable=yes requires visible=yes",
                "manual_number_visible",
            )
        if digit_count not in ("1", "2"):
            err(
                "readable_yes_digit_count",
                "readable=yes requires digit_count 1 or 2",
                "manual_digit_count",
            )
        if _is_blank(jersey):
            err(
                "readable_yes_missing_jersey",
                "readable=yes requires jersey_number",
                "manual_jersey_number",
            )
        elif not re.fullmatch(r"[0-9]{1,2}", jersey):
            err(
                "invalid_jersey_number",
                "jersey_number must be one or two ASCII digits",
                "manual_jersey_number",
            )
        elif digit_count in ("1", "2") and len(jersey) != int(digit_count):
            err(
                "jersey_digit_count_mismatch",
                "jersey_number length must equal digit_count",
                "manual_jersey_number",
            )

    if readable == "no":
        if not _is_blank(jersey):
            err(
                "readable_no_jersey_present",
                "readable=no requires blank jersey_number",
                "manual_jersey_number",
            )
        if visible == "yes" and digit_count not in ("1", "2", "uncertain"):
            err(
                "readable_no_visible_yes_digit_count",
                "visible=yes/readable=no allows digit_count 1, 2, or uncertain",
                "manual_digit_count",
            )
        if visible == "no" and digit_count != "0":
            err(
                "readable_no_visible_no_digit_count",
                "visible=no/readable=no requires digit_count=0",
                "manual_digit_count",
            )

    if readable == "uncertain":
        if not _is_blank(jersey):
            err(
                "readable_uncertain_jersey_present",
                "readable=uncertain requires blank jersey_number",
                "manual_jersey_number",
            )
        if visible not in ("yes", "uncertain"):
            err(
                "readable_uncertain_visible",
                "readable=uncertain requires visible yes or uncertain",
                "manual_number_visible",
            )
        if digit_count not in ("1", "2", "uncertain"):
            err(
                "readable_uncertain_digit_count",
                "readable=uncertain allows digit_count 1, 2, or uncertain",
                "manual_digit_count",
            )

    if not _is_blank(jersey) and readable != "yes":
        err(
            "jersey_without_readable_yes",
            "jersey_number requires readable=yes",
            "manual_jersey_number",
        )

    if not _is_blank(notes) and len(notes) > 500:
        err("notes_too_long", "manual_notes exceeds 500 characters", "manual_notes")

    return errors


def load_review_dir(review_dir: str | Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(review_dir).expanduser().resolve()
    if not root.is_dir():
        raise JerseyAnnotationError(f"review directory not found: {root}")
    items_path = root / REVIEW_ITEMS_NAME
    summary_path = root / SUMMARY_NAME
    template_path = root / ANNOTATION_TEMPLATE_NAME
    items = _load_jsonl(items_path)
    summary = _load_json(summary_path)
    if config["input"]["require_review_summary_status_ok"] and summary.get("status") != "ok":
        raise JerseyAnnotationError("review summary status is not ok")
    if summary.get("schema_version") != REVIEW_SUMMARY_SCHEMA:
        raise JerseyAnnotationError("review summary schema mismatch")
    expected_schema = config["input"]["require_review_item_schema"]
    for index, row in enumerate(items):
        if row.get("schema_version") != expected_schema:
            raise JerseyAnnotationError(f"review item {index}: schema mismatch")
        _require_str(row.get("review_item_id"), field=f"review item {index}.review_item_id")
    if int(summary.get("counts", {}).get("canonical_review_item_count", -1)) != len(items):
        raise JerseyAnnotationError("summary canonical_review_item_count mismatch")
    if not template_path.is_file():
        raise JerseyAnnotationError(f"annotation template not found: {template_path}")
    return {
        "review_dir": root,
        "items": items,
        "summary": summary,
        "items_path": items_path,
        "summary_path": summary_path,
        "template_path": template_path,
    }


def _load_annotation_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise JerseyAnnotationError(f"annotation CSV not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise JerseyAnnotationError("annotation CSV lacks final newline")
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if not rows:
        raise JerseyAnnotationError("annotation CSV is empty")
    header = rows[0]
    if header != list(ANNOTATION_COLUMNS):
        raise JerseyAnnotationError("annotation CSV header mismatch")
    parsed: list[dict[str, str]] = []
    for line_no, values in enumerate(rows[1:], start=2):
        if len(values) != len(ANNOTATION_COLUMNS):
            raise JerseyAnnotationError(
                f"annotation CSV line {line_no}: expected "
                f"{len(ANNOTATION_COLUMNS)} columns, got {len(values)}"
            )
        parsed.append(
            {column: _cell(value) for column, value in zip(ANNOTATION_COLUMNS, values)}
        )
    return parsed


def _canonical_csv_row(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "review_item_id": str(item["review_item_id"]),
        "review_index": str(int(item["review_index"])),
        "segment_id": str(item["segment_id"]),
        "raw_track_id": str(int(item["raw_track_id"])),
        "crop_id": str(item["crop_id"]),
        "frame_index": str(int(item["frame_index"])),
        "master_panel_path": str(item["master_panel_path"]),
        "master_panel_page": str(int(item["master_panel_page"])),
        "master_panel_tile_index": str(int(item["master_panel_tile_index"])),
        "group_memberships": json.dumps(
            item["group_memberships"], ensure_ascii=False
        ),
    }


def validate_jersey_review_annotations(
    *,
    review_dir: str | Path,
    annotations_csv: str | Path,
    config: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not isinstance(config, Mapping):
        validated_config = load_jersey_manual_review_config(config)
        config_path = Path(config).expanduser().resolve()
    else:
        validated_config = validate_jersey_manual_review_config(config)
        config_path = None

    review = load_review_dir(review_dir, config=validated_config)
    csv_path = Path(annotations_csv).expanduser().resolve()
    annotation_rows = _load_annotation_csv(csv_path)
    expected_rows = [_canonical_csv_row(item) for item in review["items"]]
    allowed = validated_config["allowed_values"]

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    expected_count = len(expected_rows)
    actual_count = len(annotation_rows)
    if actual_count != expected_count:
        errors.append(
            _row_error(
                code="row_count_mismatch",
                message=(
                    f"expected {expected_count} annotation rows, got {actual_count}"
                ),
            )
        )

    expected_ids = [row["review_item_id"] for row in expected_rows]
    actual_ids = [row["review_item_id"] for row in annotation_rows]
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    missing_ids = sorted(expected_set - actual_set)
    extra_ids = sorted(actual_set - expected_set)
    duplicate_ids = sorted(
        rid for rid, count in {
            rid: actual_ids.count(rid) for rid in actual_set
        }.items()
        if count > 1
    )
    for rid in missing_ids:
        errors.append(
            _row_error(
                code="missing_review_item",
                message=f"missing review_item_id {rid}",
                review_item_id=rid,
            )
        )
    for rid in extra_ids:
        errors.append(
            _row_error(
                code="extra_review_item",
                message=f"extra review_item_id {rid}",
                review_item_id=rid,
            )
        )
    for rid in duplicate_ids:
        errors.append(
            _row_error(
                code="duplicate_review_item",
                message=f"duplicate review_item_id {rid}",
                review_item_id=rid,
            )
        )

    provenance_mismatch = 0
    compare_n = min(len(expected_rows), len(annotation_rows))
    for index in range(compare_n):
        expected = expected_rows[index]
        actual = annotation_rows[index]
        review_index = index + 1
        review_item_id = actual.get("review_item_id") or expected["review_item_id"]
        for column in PROVENANCE_COLUMNS:
            if actual.get(column) != expected[column]:
                provenance_mismatch += 1
                errors.append(
                    _row_error(
                        code="provenance_mismatch",
                        message=(
                            f"row {review_index} provenance field {column} "
                            "does not match canonical review item/template"
                        ),
                        review_item_id=review_item_id,
                        review_index=review_index,
                        field=column,
                    )
                )
                break
        if actual_ids[:compare_n] != expected_ids[:compare_n] and index < compare_n:
            # order mismatch is already covered by per-row provenance when IDs differ;
            # if IDs match but order differs earlier, the first mismatch catches it.
            pass

    if actual_ids != expected_ids and not missing_ids and not extra_ids and not duplicate_ids:
        errors.append(
            _row_error(
                code="row_order_mismatch",
                message="annotation CSV row order must match the blank template",
            )
        )

    unreviewed = 0
    reviewed = 0
    counts = {
        "valid_crop_count": 0,
        "invalid_crop_count": 0,
        "uncertain_crop_count": 0,
        "number_visible_yes_count": 0,
        "number_visible_no_count": 0,
        "number_visible_uncertain_count": 0,
        "number_readable_yes_count": 0,
        "number_readable_no_count": 0,
        "number_readable_uncertain_count": 0,
        "readable_number_count": 0,
        "contamination_yes_count": 0,
        "contamination_no_count": 0,
        "contamination_uncertain_count": 0,
    }
    reviewers: set[str] = set()

    for index, row in enumerate(annotation_rows):
        review_index = index + 1
        review_item_id = row["review_item_id"]
        decision_filled = any(not _is_blank(row[field]) for field in DECISION_FIELDS)
        meta_filled = any(not _is_blank(row[field]) for field in ("reviewer", "reviewed_at"))

        if not decision_filled and not meta_filled:
            unreviewed += 1
            continue

        if meta_filled and not decision_filled:
            errors.append(
                _row_error(
                    code="metadata_without_decision",
                    message="reviewer/reviewed_at require at least one manual decision",
                    review_item_id=review_item_id,
                    review_index=review_index,
                )
            )
            continue

        reviewed += 1
        row_errors = _validate_reviewed_row(
            row,
            review_item_id=review_item_id,
            review_index=review_index,
            allowed=allowed,
        )
        errors.extend(row_errors)
        if row_errors:
            continue

        crop_valid = row["manual_crop_valid"]
        if crop_valid == "valid":
            counts["valid_crop_count"] += 1
        elif crop_valid == "invalid":
            counts["invalid_crop_count"] += 1
        elif crop_valid == "uncertain":
            counts["uncertain_crop_count"] += 1

        visible = row["manual_number_visible"]
        if visible == "yes":
            counts["number_visible_yes_count"] += 1
        elif visible == "no":
            counts["number_visible_no_count"] += 1
        elif visible == "uncertain":
            counts["number_visible_uncertain_count"] += 1

        readable = row["manual_number_readable"]
        if readable == "yes":
            counts["number_readable_yes_count"] += 1
            counts["readable_number_count"] += 1
        elif readable == "no":
            counts["number_readable_no_count"] += 1
        elif readable == "uncertain":
            counts["number_readable_uncertain_count"] += 1

        contamination = row["manual_contamination_affects_number_region"]
        if contamination == "yes":
            counts["contamination_yes_count"] += 1
        elif contamination == "no":
            counts["contamination_no_count"] += 1
        elif contamination == "uncertain":
            counts["contamination_uncertain_count"] += 1

        if not _is_blank(row["reviewer"]):
            reviewers.add(row["reviewer"])

    errors.sort(
        key=lambda row: (
            row.get("review_index") is None,
            row.get("review_index") or 0,
            row.get("code") or "",
            row.get("field") or "",
            row.get("message") or "",
        )
    )

    status = "valid" if not errors else "invalid"
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "source_provenance": {
            "review_dir": str(review["review_dir"]),
            "review_summary": {
                "path": str(review["summary_path"]),
                "sha256": sha256_file(review["summary_path"]),
                "byte_size": review["summary_path"].stat().st_size,
            },
            "review_items": {
                "path": str(review["items_path"]),
                "sha256": sha256_file(review["items_path"]),
                "byte_size": review["items_path"].stat().st_size,
            },
            "annotation_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
                "byte_size": csv_path.stat().st_size,
            },
            "config": (
                {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                    "byte_size": config_path.stat().st_size,
                }
                if config_path is not None
                else None
            ),
        },
        "counts": {
            "expected_row_count": expected_count,
            "actual_row_count": actual_count,
            "unreviewed_row_count": unreviewed,
            "reviewed_row_count": reviewed,
            **counts,
            "distinct_reviewer_count": len(reviewers),
        },
        "validation": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "duplicate_id_count": len(duplicate_ids),
            "missing_id_count": len(missing_ids),
            "extra_id_count": len(extra_ids),
            "provenance_mismatch_count": provenance_mismatch,
        },
        "safety": {
            "OCR_performed": False,
            "recognizer_performed": False,
            "checkpoint_loaded": False,
            "gallery_updated": False,
            "identity_assigned": False,
            "team_assigned": False,
            "source_review_modified": False,
            "panel_modified": False,
            "source_crop_modified": False,
            "identity_ground_truth_available": False,
            "accuracy_claimed": False,
        },
        "semantics_and_limitations": [
            "blank manual fields mean unreviewed, not uncertain",
            "uncertain is an explicit human decision",
            "manual jersey labels are not physical player identity ground truth",
            "readable number alone is not global identity",
            "annotation does not update gallery or assign team/identity",
            "panels and source crops remain immutable",
            "selected crops are not all observations",
        ],
        "elapsed_sec": time.perf_counter() - started,
    }
    return report


def create_temp_annotation_report_path(report_path: Path) -> Path:
    final = report_path.expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.parent / (
        f"_tmp_reid_jersey_annotation_report_{final.name}_{uuid.uuid4().hex[:8]}"
    )
    return temp


def finalize_annotation_report(
    *, temp_path: Path, final_path: Path, overwrite: bool
) -> Path:
    temp = temp_path.expanduser().resolve()
    final = final_path.expanduser().resolve()
    if not temp.is_file():
        raise JerseyAnnotationError(f"temporary report missing: {temp}")
    backup: Path | None = None
    try:
        if final.exists():
            if not overwrite:
                raise JerseyAnnotationError(f"report already exists: {final}")
            backup = final.with_name(
                f"_backup_reid_jersey_annotation_report_{final.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final, backup)
        os.rename(temp, final)
        if backup is not None:
            backup.unlink()
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            try:
                os.rename(backup, final)
                backup = None
            except OSError:
                pass
        raise
    for stray in final.parent.glob(
        f"_tmp_reid_jersey_annotation_report_{final.name}_*"
    ):
        if stray.is_file():
            stray.unlink(missing_ok=True)
    for stray in final.parent.glob(
        f"_backup_reid_jersey_annotation_report_{final.name}_*"
    ):
        if stray.is_file():
            stray.unlink(missing_ok=True)
    return final


def run_validate_jersey_review_annotations(
    *,
    review_dir: str | Path,
    annotations_csv: str | Path,
    config: str | Path,
    report_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    final_path = Path(report_path).expanduser().resolve()
    if final_path.exists():
        if final_path.is_dir():
            raise JerseyAnnotationError(f"report path is a directory: {final_path}")
        if not overwrite:
            raise JerseyAnnotationError(
                f"report already exists: {final_path}; re-run with --overwrite"
            )

    report = validate_jersey_review_annotations(
        review_dir=review_dir,
        annotations_csv=annotations_csv,
        config=config,
    )
    temp_path: Path | None = None
    try:
        temp_path = create_temp_annotation_report_path(final_path)
        temp_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not temp_path.read_bytes().endswith(b"\n"):
            raise JerseyAnnotationError("temporary report lacks final newline")
        reloaded = _load_json(temp_path)
        if reloaded.get("schema_version") != REPORT_SCHEMA:
            raise JerseyAnnotationError("written report schema mismatch")
        finalized = finalize_annotation_report(
            temp_path=temp_path, final_path=final_path, overwrite=overwrite
        )
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    return {
        "status": report["status"],
        "report_path": str(finalized),
        "error_count": report["validation"]["error_count"],
        "reviewed_row_count": report["counts"]["reviewed_row_count"],
        "unreviewed_row_count": report["counts"]["unreviewed_row_count"],
        "elapsed_sec": report["elapsed_sec"],
        "report": report,
    }
