"""ReID raw-track purity / within-track kit-change audit (Stage 5B3A).

Measurement-only: chronological adjacent crop transitions, descriptor
distances, torso-region tracking-bbox contamination context, and
manual-review ranking. No splits, labels, thresholds, or global-ID rewrite.
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

import numpy as np
import yaml

from football_analytics.reid.crop_select import clamp_bbox_xyxy, float_bbox_to_int_crop
from football_analytics.reid.kit import (
    DEFAULT_FAMILY_ORDER,
    KitError,
    build_track_kit_descriptors,
    load_crop_manifest_for_kit,
    load_kit_descriptor_config,
    load_quality_signals_for_kit,
)
from football_analytics.reid.quality import (
    QualityError,
    compute_tracking_bbox_contamination,
    index_observations_by_frame,
    infer_frame_size,
    load_person_track_observations,
    verify_target_observation,
)
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

TRANSITION_SCHEMA = "reid_track_transition_audit_v1"
TRACK_PURITY_SCHEMA = "reid_track_purity_summary_v1"
AUDIT_SUMMARY_SCHEMA = "reid_track_purity_audit_summary_v1"
CONFIG_SCHEMA = "reid_track_purity_audit_config_v1"

TRANSITION_NAME = "track_transition_audit.jsonl"
TRACK_PURITY_NAME = "track_purity_summary.jsonl"
AUDIT_SUMMARY_NAME = "track_purity_audit_summary.json"

EXPECTED_RANK_METRICS = (
    "color_family_l1",
    "lab_mean_distance_normalized",
    "hue_histogram_l1",
    "saturation_histogram_l1",
    "value_histogram_l1",
)
CHRONOLOGY_ORDER = ("frame_index", "selection_rank", "crop_id")
LAB_MAX_DEFAULT = math.sqrt(3.0 * 255.0 * 255.0)


class PurityError(RuntimeError):
    """Raised when track-purity audit inputs or outputs are invalid."""


def _reject_non_finite_json(value: str) -> None:
    raise PurityError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise PurityError(f"{field} must be finite, got {value!r}")
    return number


def _ensure_ratio(value: float, *, field: str) -> float:
    number = _ensure_finite_float(value, field=field)
    if number < 0.0 or number > 1.0:
        raise PurityError(f"{field} must be in [0, 1], got {number}")
    return number


def _ensure_l1(value: float, *, field: str, high: float = 2.0) -> float:
    number = _ensure_finite_float(value, field=field)
    if number < 0.0 or number > high + 1e-9:
        raise PurityError(f"{field} must be in [0, {high}], got {number}")
    return float(min(max(number, 0.0), high))


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise PurityError(f"{field} must be a bool, got {value!r}")
    if expected is not None and value is not expected:
        raise PurityError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PurityError(f"{field} must be a positive int, got {value!r}")
    return value


def _require_nonneg_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PurityError(f"{field} must be a non-negative int, got {value!r}")
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
            raise PurityError(f"non-finite percentile {key}")
    return qs


def _metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    arr = np.asarray(
        [_ensure_finite_float(v, field="metric") for v in values], dtype=np.float64
    )
    out = {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }
    for key, number in out.items():
        if not math.isfinite(number):
            raise PurityError(f"non-finite metric summary {key}")
    return out


def validate_track_purity_audit_config(
    payload: Mapping[str, Any], *, source: str = "<config>"
) -> dict[str, Any]:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise PurityError(
            f"{source}: schema_version must be {CONFIG_SCHEMA!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("stage_status") != "measurement_baseline":
        raise PurityError(
            f"{source}: stage_status must be 'measurement_baseline', "
            f"got {payload.get('stage_status')!r}"
        )

    for flag in (
        "automatic_track_split_enabled",
        "automatic_track_delete_enabled",
        "automatic_global_id_rewrite_enabled",
        "automatic_team_assignment_enabled",
        "automatic_link_enabled",
        "automatic_reject_enabled",
        "composite_change_score_enabled",
    ):
        _require_bool(payload.get(flag), field=f"{source}.{flag}", expected=False)
    _require_bool(
        payload.get("manual_review_required"),
        field=f"{source}.manual_review_required",
        expected=True,
    )
    if payload.get("change_threshold") is not None:
        raise PurityError(f"{source}: change_threshold must be null")
    if payload.get("split_threshold") is not None:
        raise PurityError(f"{source}: split_threshold must be null")

    chronology = payload.get("chronology")
    if not isinstance(chronology, Mapping):
        raise PurityError(f"{source}: chronology must be a mapping")
    if list(chronology.get("order_by")) != list(CHRONOLOGY_ORDER):
        raise PurityError(
            f"{source}: chronology.order_by must be {list(CHRONOLOGY_ORDER)}"
        )
    _require_bool(
        chronology.get("duplicate_frame_within_track_allowed"),
        field=f"{source}.chronology.duplicate_frame_within_track_allowed",
        expected=False,
    )
    _require_bool(
        chronology.get("compare_adjacent_crops_only"),
        field=f"{source}.chronology.compare_adjacent_crops_only",
        expected=True,
    )
    _require_bool(
        chronology.get("preserve_original_manifest_order_for_provenance"),
        field=f"{source}.chronology.preserve_original_manifest_order_for_provenance",
        expected=True,
    )

    metrics = payload.get("descriptor_metrics")
    if not isinstance(metrics, Mapping):
        raise PurityError(f"{source}: descriptor_metrics must be a mapping")
    for key in (
        "color_family_l1_enabled",
        "hue_histogram_l1_enabled",
        "saturation_histogram_l1_enabled",
        "value_histogram_l1_enabled",
        "lab_mean_distance_enabled",
        "dominant_family_change_enabled",
        "chromatic_ratio_difference_enabled",
        "achromatic_ratio_difference_enabled",
    ):
        _require_bool(metrics.get(key), field=f"{source}.descriptor_metrics.{key}", expected=True)

    norms = payload.get("distance_normalization")
    if not isinstance(norms, Mapping):
        raise PurityError(f"{source}: distance_normalization must be a mapping")
    for key, expected in (
        ("histogram_l1_expected_range", [0.0, 2.0]),
        ("color_family_l1_expected_range", [0.0, 2.0]),
    ):
        value = norms.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise PurityError(f"{source}: distance_normalization.{key} invalid")
        pair = [float(value[0]), float(value[1])]
        if pair != expected:
            raise PurityError(
                f"{source}: distance_normalization.{key} must be {expected}"
            )
    _require_bool(
        norms.get("lab_mean_euclidean_normalized"),
        field=f"{source}.distance_normalization.lab_mean_euclidean_normalized",
        expected=True,
    )
    lab_max = norms.get("lab_uint8_max_distance")
    if not isinstance(lab_max, (int, float)) or isinstance(lab_max, bool):
        raise PurityError(f"{source}: lab_uint8_max_distance must be a number")
    lab_max_f = float(lab_max)
    if not math.isfinite(lab_max_f) or abs(lab_max_f - LAB_MAX_DEFAULT) > 1e-6:
        raise PurityError(
            f"{source}: lab_uint8_max_distance must be {LAB_MAX_DEFAULT}, got {lab_max!r}"
        )

    torso = payload.get("torso_tracking_overlap")
    if not isinstance(torso, Mapping):
        raise PurityError(f"{source}: torso_tracking_overlap must be a mapping")
    _require_bool(torso.get("enabled"), field=f"{source}.torso_tracking_overlap.enabled", expected=True)
    if torso.get("usage") != "audit_context_only":
        raise PurityError(f"{source}: torso_tracking_overlap.usage must be audit_context_only")
    for flag in ("hard_reject_enabled", "automatic_split_enabled", "detects_untracked_people", "visible_pixel_fraction"):
        _require_bool(
            torso.get(flag),
            field=f"{source}.torso_tracking_overlap.{flag}",
            expected=False,
        )
    if torso.get("coverage_threshold") is not None:
        raise PurityError(f"{source}: torso_tracking_overlap.coverage_threshold must be null")

    quality = payload.get("quality_context")
    if not isinstance(quality, Mapping):
        raise PurityError(f"{source}: quality_context must be a mapping")
    _require_bool(quality.get("copied_for_audit_only"), field=f"{source}.quality_context.copied_for_audit_only", expected=True)
    _require_bool(quality.get("quality_exclusion_enabled"), field=f"{source}.quality_context.quality_exclusion_enabled", expected=False)
    _require_bool(quality.get("quality_weighting_enabled"), field=f"{source}.quality_context.quality_weighting_enabled", expected=False)

    ranking = payload.get("ranking")
    if not isinstance(ranking, Mapping):
        raise PurityError(f"{source}: ranking must be a mapping")
    _require_bool(ranking.get("composite_rank_enabled"), field=f"{source}.ranking.composite_rank_enabled", expected=False)
    ranks = ranking.get("independent_metric_ranks")
    if list(ranks) != list(EXPECTED_RANK_METRICS):
        raise PurityError(
            f"{source}: ranking.independent_metric_ranks must be "
            f"{list(EXPECTED_RANK_METRICS)}, got {ranks!r}"
        )
    if len(set(ranks)) != len(ranks):
        raise PurityError(f"{source}: duplicate independent rank metric")
    _require_bool(
        ranking.get("dominant_family_change_is_decision"),
        field=f"{source}.ranking.dominant_family_change_is_decision",
        expected=False,
    )

    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise PurityError(f"{source}: policy must be a mapping")
    _require_bool(policy.get("raw_track_is_atomic_identity_guarantee"), field=f"{source}.policy.raw_track_is_atomic_identity_guarantee", expected=False)
    _require_bool(policy.get("within_track_change_is_identity_switch_proof"), field=f"{source}.policy.within_track_change_is_identity_switch_proof", expected=False)
    _require_bool(policy.get("no_detected_change_is_purity_proof"), field=f"{source}.policy.no_detected_change_is_purity_proof", expected=False)
    _require_bool(policy.get("selected_crop_audit_is_full_track_segmentation"), field=f"{source}.policy.selected_crop_audit_is_full_track_segmentation", expected=False)
    _require_bool(policy.get("raw_track_ids_preserved"), field=f"{source}.policy.raw_track_ids_preserved", expected=True)

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise PurityError(f"{source}: evaluation must be a mapping")
    for key in (
        "identity_switch_ground_truth_available",
        "track_purity_ground_truth_available",
        "accuracy_claim_allowed",
    ):
        _require_bool(evaluation.get(key), field=f"{source}.evaluation.{key}", expected=False)

    return {
        "schema_version": CONFIG_SCHEMA,
        "stage_status": "measurement_baseline",
        "automatic_track_split_enabled": False,
        "automatic_track_delete_enabled": False,
        "automatic_global_id_rewrite_enabled": False,
        "automatic_team_assignment_enabled": False,
        "automatic_link_enabled": False,
        "automatic_reject_enabled": False,
        "change_threshold": None,
        "split_threshold": None,
        "composite_change_score_enabled": False,
        "manual_review_required": True,
        "chronology": {
            "order_by": list(CHRONOLOGY_ORDER),
            "duplicate_frame_within_track_allowed": False,
            "compare_adjacent_crops_only": True,
            "preserve_original_manifest_order_for_provenance": True,
        },
        "descriptor_metrics": {k: True for k in metrics},
        "distance_normalization": {
            "histogram_l1_expected_range": [0.0, 2.0],
            "color_family_l1_expected_range": [0.0, 2.0],
            "lab_mean_euclidean_normalized": True,
            "lab_uint8_max_distance": LAB_MAX_DEFAULT,
        },
        "torso_tracking_overlap": {
            "enabled": True,
            "usage": "audit_context_only",
            "hard_reject_enabled": False,
            "automatic_split_enabled": False,
            "coverage_threshold": None,
            "detects_untracked_people": False,
            "visible_pixel_fraction": False,
        },
        "quality_context": {
            "copied_for_audit_only": True,
            "quality_exclusion_enabled": False,
            "quality_weighting_enabled": False,
        },
        "ranking": {
            "composite_rank_enabled": False,
            "independent_metric_ranks": list(EXPECTED_RANK_METRICS),
            "dominant_family_change_is_decision": False,
        },
        "policy": {
            "raw_track_is_atomic_identity_guarantee": False,
            "within_track_change_is_identity_switch_proof": False,
            "no_detected_change_is_purity_proof": False,
            "selected_crop_audit_is_full_track_segmentation": False,
            "raw_track_ids_preserved": True,
        },
        "evaluation": {
            "identity_switch_ground_truth_available": False,
            "track_purity_ground_truth_available": False,
            "accuracy_claim_allowed": False,
        },
        "source": source,
    }


def load_track_purity_audit_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise PurityError(f"track purity audit config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PurityError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PurityError(f"track purity audit config must be a mapping: {config_path}")
    validated = validate_track_purity_audit_config(payload, source=str(config_path))
    validated["source_path"] = str(config_path)
    return validated


def create_temp_purity_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_purity_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise PurityError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_purity_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise PurityError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise PurityError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_purity_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_purity_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_purity_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def _validate_relative_path(rel: str, *, manifest_dir: Path) -> None:
    if not isinstance(rel, str) or not rel.strip():
        raise PurityError(f"crop_relative_path must be non-empty string, got {rel!r}")
    normalized = rel.replace("\\", "/")
    if normalized.startswith("/") or Path(normalized).is_absolute():
        raise PurityError(f"absolute crop_relative_path is not allowed: {rel!r}")
    if any(part == ".." for part in Path(normalized).parts):
        raise PurityError(f"crop_relative_path must not contain '..': {rel!r}")
    root = manifest_dir.expanduser().resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PurityError(f"crop path escapes manifest directory: {rel!r}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PurityError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PurityError(f"file is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_json)
        except PurityError:
            raise
        except json.JSONDecodeError as exc:
            raise PurityError(f"invalid JSON on {path} line {line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PurityError(f"{path} line {line_no} must be a JSON object")
        rows.append(payload)
    if not rows:
        raise PurityError(f"file is empty: {path}")
    return rows


def _approx_equal(a: float, b: float, *, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_crop_kit_for_purity(
    path: str | Path, *, manifest_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = _load_jsonl(Path(path).expanduser().resolve())
    if len(rows) != len(manifest_rows):
        raise PurityError(
            f"crop kit row count {len(rows)} != manifest count {len(manifest_rows)}"
        )
    crop_ids: set[str] = set()
    for idx, (kit_row, manifest_row) in enumerate(zip(rows, manifest_rows)):
        for key in (
            "crop_id",
            "track_id",
            "frame_index",
            "selection_rank",
            "crop_relative_path",
        ):
            if kit_row.get(key) != manifest_row[key]:
                raise PurityError(
                    f"crop kit/manifest mismatch at index {idx} field {key}: "
                    f"kit={kit_row.get(key)!r} manifest={manifest_row[key]!r}"
                )
        crop_id = kit_row["crop_id"]
        if crop_id in crop_ids:
            raise PurityError(f"duplicate crop_id in crop kit: {crop_id}")
        crop_ids.add(crop_id)
        if kit_row.get("descriptor_usage") != "measurement_only":
            raise PurityError(f"crop kit {crop_id}: descriptor_usage must be measurement_only")
        if kit_row.get("team_assignment") is not None:
            raise PurityError(f"crop kit {crop_id}: team_assignment must be null")
        if kit_row.get("kit_similarity_threshold") is not None:
            raise PurityError(f"crop kit {crop_id}: kit_similarity_threshold must be null")
        for flag in (
            "automatic_link_applied",
            "automatic_reject_applied",
            "quality_weight_applied",
            "quality_exclusion_applied",
        ):
            _require_bool(kit_row.get(flag), field=f"crop kit {crop_id}.{flag}", expected=False)
        _require_bool(kit_row.get("torso_region_valid"), field=f"crop kit {crop_id}.torso_region_valid", expected=True)
        for hist_name, length in (
            ("hue_histogram_chromatic", 18),
            ("saturation_histogram", 8),
            ("value_histogram", 8),
        ):
            hist = kit_row.get(hist_name)
            if not isinstance(hist, list) or len(hist) != length:
                raise PurityError(f"crop kit {crop_id}: {hist_name} length must be {length}")
            for value in hist:
                _ensure_finite_float(float(value), field=hist_name)
                if float(value) < 0.0 or float(value) > 1.0:
                    raise PurityError(f"crop kit {crop_id}: {hist_name} bin out of [0,1]")
        fractions = kit_row.get("color_family_fractions")
        if not isinstance(fractions, Mapping):
            raise PurityError(f"crop kit {crop_id}: color_family_fractions required")
        for fam in DEFAULT_FAMILY_ORDER:
            if fam not in fractions:
                raise PurityError(f"crop kit {crop_id}: missing family {fam}")
            _ensure_ratio(float(fractions[fam]), field=f"family.{fam}")
        if abs(sum(float(fractions[f]) for f in DEFAULT_FAMILY_ORDER) - 1.0) > 1e-5:
            raise PurityError(f"crop kit {crop_id}: family fractions must sum to ~1")
        for key in ("torso_x0", "torso_y0", "torso_x1", "torso_y1"):
            _require_nonneg_int(kit_row.get(key), field=f"crop kit {crop_id}.{key}")
        for key in ("torso_width", "torso_height"):
            _require_positive_int(kit_row.get(key), field=f"crop kit {crop_id}.{key}")
        for key in ("chromatic_pixel_ratio", "achromatic_pixel_ratio"):
            _ensure_ratio(float(kit_row[key]), field=key)
        lab = kit_row.get("lab_mean")
        if not isinstance(lab, Mapping) or not all(k in lab for k in ("l", "a", "b")):
            raise PurityError(f"crop kit {crop_id}: lab_mean required")
        for k in ("l", "a", "b"):
            _ensure_finite_float(float(lab[k]), field=f"lab_mean.{k}")
    return rows


def load_track_kit_for_purity(
    path: str | Path,
    *,
    crop_kit_rows: Sequence[Mapping[str, Any]],
    kit_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _load_jsonl(Path(path).expanduser().resolve())
    recomputed = build_track_kit_descriptors(crop_kit_rows, config=kit_config)
    if len(rows) != len(recomputed):
        raise PurityError(
            f"track kit count {len(rows)} != recomputed {len(recomputed)}"
        )
    by_id = {int(r["track_id"]): r for r in rows}
    if len(by_id) != len(rows):
        raise PurityError("duplicate track_id in track kit descriptors")
    for expected in recomputed:
        tid = int(expected["track_id"])
        actual = by_id.get(tid)
        if actual is None:
            raise PurityError(f"missing track kit row for track_id={tid}")
        if int(actual["crop_count"]) != int(expected["crop_count"]):
            raise PurityError(f"track {tid}: crop_count mismatch")
        if list(actual["crop_ids"]) != list(expected["crop_ids"]):
            raise PurityError(f"track {tid}: crop_ids mismatch")
        for hist_key in (
            "mean_hue_histogram_chromatic",
            "mean_saturation_histogram",
            "mean_value_histogram",
        ):
            a = [float(v) for v in actual[hist_key]]
            b = [float(v) for v in expected[hist_key]]
            if len(a) != len(b) or any(not _approx_equal(x, y, tol=1e-5) for x, y in zip(a, b)):
                raise PurityError(f"track {tid}: {hist_key} mismatch vs recomputed")
        for fam in DEFAULT_FAMILY_ORDER:
            if not _approx_equal(
                float(actual["mean_color_family_fractions"][fam]),
                float(expected["mean_color_family_fractions"][fam]),
                tol=1e-5,
            ):
                raise PurityError(f"track {tid}: mean family {fam} mismatch")
        if actual.get("dominant_track_color_family") != expected["dominant_track_color_family"]:
            raise PurityError(f"track {tid}: dominant track family mismatch")
        if not _approx_equal(
            float(actual["dominant_track_color_family_fraction"]),
            float(expected["dominant_track_color_family_fraction"]),
            tol=1e-5,
        ):
            raise PurityError(f"track {tid}: dominant track fraction mismatch")
        _require_bool(actual.get("quality_weighting_applied"), field=f"track {tid}.quality_weighting_applied", expected=False)
        if int(actual.get("excluded_crop_count", -1)) != 0:
            raise PurityError(f"track {tid}: excluded_crop_count must be 0")
        if actual.get("team_assignment") is not None:
            raise PurityError(f"track {tid}: team_assignment must be null")
        _require_bool(actual.get("forced_two_team_clustering_applied"), field=f"track {tid}.forced_two_team_clustering_applied", expected=False)
        for flag in ("automatic_link_applied", "automatic_reject_applied"):
            _require_bool(actual.get(flag), field=f"track {tid}.{flag}", expected=False)
    return [by_id[int(r["track_id"])] for r in recomputed]


def l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise PurityError("L1 vectors must have equal length")
    return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)))


def lab_mean_distance(
    a: Mapping[str, float], b: Mapping[str, float], *, lab_max: float
) -> tuple[float, float]:
    raw = math.sqrt(
        (float(a["l"]) - float(b["l"])) ** 2
        + (float(a["a"]) - float(b["a"])) ** 2
        + (float(a["b"]) - float(b["b"])) ** 2
    )
    raw = _ensure_finite_float(raw, field="lab_mean_distance_raw")
    if lab_max <= 0:
        raise PurityError("lab_uint8_max_distance must be positive")
    normalized = _ensure_ratio(raw / lab_max, field="lab_mean_distance_normalized")
    return raw, normalized


def chronological_sort_key(crop: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(crop["frame_index"]),
        int(crop["selection_rank"]),
        str(crop["crop_id"]),
    )


def build_chronological_tracks(
    joined_crops: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, crop in enumerate(joined_crops):
        row = dict(crop)
        row["manifest_position"] = idx
        by_track[int(crop["track_id"])].append(row)

    ordered: dict[int, list[dict[str, Any]]] = {}
    for track_id in sorted(by_track):
        crops = list(by_track[track_id])
        frames = [int(c["frame_index"]) for c in crops]
        if len(frames) != len(set(frames)):
            raise PurityError(
                f"duplicate frame_index within track_id={track_id}: {frames}"
            )
        crops.sort(key=chronological_sort_key)
        for i, crop in enumerate(crops, start=1):
            crop["chronology_index"] = i
        ordered[track_id] = crops
    return ordered


def compute_frame_torso_bbox(
    *,
    manifest_bbox: Sequence[int],
    torso_x0: int,
    torso_y0: int,
    torso_x1: int,
    torso_y1: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    mx0, my0, _, _ = [int(v) for v in manifest_bbox]
    x0 = mx0 + int(torso_x0)
    y0 = my0 + int(torso_y0)
    x1 = mx0 + int(torso_x1)
    y1 = my0 + int(torso_y1)
    x0 = max(0, min(x0, frame_width))
    x1 = max(0, min(x1, frame_width))
    y0 = max(0, min(y0, frame_height))
    y1 = max(0, min(y1, frame_height))
    if x1 <= x0 or y1 <= y0:
        raise PurityError(
            f"empty torso frame bbox after clamp: {[x0, y0, x1, y1]}"
        )
    return [x0, y0, x1, y1]


def compute_torso_tracking_contamination(
    *,
    crop_row: Mapping[str, Any],
    frame_observations: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    torso_bbox = compute_frame_torso_bbox(
        manifest_bbox=crop_row["bbox_xyxy"],
        torso_x0=int(crop_row["torso_x0"]),
        torso_y0=int(crop_row["torso_y0"]),
        torso_x1=int(crop_row["torso_x1"]),
        torso_y1=int(crop_row["torso_y1"]),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    try:
        result = compute_tracking_bbox_contamination(
            target_bbox=torso_bbox,
            target_track_id=int(crop_row["track_id"]),
            frame_observations=frame_observations,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    except QualityError as exc:
        raise PurityError(str(exc)) from exc
    return {
        "frame_torso_bbox_xyxy": torso_bbox,
        "torso_other_person_observation_count_in_frame": result[
            "other_person_observation_count_in_frame"
        ],
        "torso_other_person_overlap_count": result["other_person_overlap_count"],
        "torso_other_person_center_inside_count": result[
            "other_person_center_inside_count"
        ],
        "torso_max_other_person_coverage": result["max_other_person_crop_coverage"],
        "torso_union_other_person_coverage": result["union_other_person_crop_coverage"],
        "torso_max_other_person_iou": result["max_other_person_iou"],
    }


def build_transition(
    *,
    track_id: int,
    transition_index: int,
    crop_from: Mapping[str, Any],
    crop_to: Mapping[str, Any],
    lab_max: float,
) -> dict[str, Any]:
    family_from = [float(crop_from["color_family_fractions"][f]) for f in DEFAULT_FAMILY_ORDER]
    family_to = [float(crop_to["color_family_fractions"][f]) for f in DEFAULT_FAMILY_ORDER]
    color_family_l1 = _ensure_l1(l1_distance(family_from, family_to), field="color_family_l1")
    hue_l1 = _ensure_l1(
        l1_distance(crop_from["hue_histogram_chromatic"], crop_to["hue_histogram_chromatic"]),
        field="hue_histogram_l1",
    )
    sat_l1 = _ensure_l1(
        l1_distance(crop_from["saturation_histogram"], crop_to["saturation_histogram"]),
        field="saturation_histogram_l1",
    )
    val_l1 = _ensure_l1(
        l1_distance(crop_from["value_histogram"], crop_to["value_histogram"]),
        field="value_histogram_l1",
    )
    lab_raw, lab_norm = lab_mean_distance(
        crop_from["lab_mean"], crop_to["lab_mean"], lab_max=lab_max
    )
    chrom_diff = _ensure_ratio(
        abs(float(crop_from["chromatic_pixel_ratio"]) - float(crop_to["chromatic_pixel_ratio"])),
        field="chromatic_pixel_ratio_abs_diff",
    )
    achrom_diff = _ensure_ratio(
        abs(
            float(crop_from["achromatic_pixel_ratio"])
            - float(crop_to["achromatic_pixel_ratio"])
        ),
        field="achromatic_pixel_ratio_abs_diff",
    )
    transition_id = (
        f"track_{track_id}__{crop_from['crop_id']}__{crop_to['crop_id']}"
    )
    return {
        "transition_id": transition_id,
        "track_id": int(track_id),
        "transition_index": int(transition_index),
        "crop_id_from": crop_from["crop_id"],
        "crop_id_to": crop_to["crop_id"],
        "frame_index_from": int(crop_from["frame_index"]),
        "frame_index_to": int(crop_to["frame_index"]),
        "frame_gap": int(crop_to["frame_index"]) - int(crop_from["frame_index"]),
        "selection_rank_from": int(crop_from["selection_rank"]),
        "selection_rank_to": int(crop_to["selection_rank"]),
        "manifest_position_from": int(crop_from["manifest_position"]),
        "manifest_position_to": int(crop_to["manifest_position"]),
        "dominant_family_from": crop_from["dominant_color_family"],
        "dominant_family_to": crop_to["dominant_color_family"],
        "dominant_family_changed": (
            crop_from["dominant_color_family"] != crop_to["dominant_color_family"]
        ),
        "dominant_fraction_from": float(crop_from["dominant_color_family_fraction"]),
        "dominant_fraction_to": float(crop_to["dominant_color_family_fraction"]),
        "color_family_l1": color_family_l1,
        "hue_histogram_l1": hue_l1,
        "saturation_histogram_l1": sat_l1,
        "value_histogram_l1": val_l1,
        "lab_mean_distance_raw": lab_raw,
        "lab_mean_distance_normalized": lab_norm,
        "chromatic_pixel_ratio_abs_diff": chrom_diff,
        "achromatic_pixel_ratio_abs_diff": achrom_diff,
        "laplacian_variance_from": float(crop_from["laplacian_variance"]),
        "laplacian_variance_to": float(crop_to["laplacian_variance"]),
        "minimum_laplacian_variance": float(
            min(float(crop_from["laplacian_variance"]), float(crop_to["laplacian_variance"]))
        ),
        "crop_union_contamination_from": float(
            crop_from["union_other_person_crop_coverage"]
        ),
        "crop_union_contamination_to": float(crop_to["union_other_person_crop_coverage"]),
        "maximum_crop_union_contamination": float(
            max(
                float(crop_from["union_other_person_crop_coverage"]),
                float(crop_to["union_other_person_crop_coverage"]),
            )
        ),
        "frame_edge_contact_count_from": int(crop_from["frame_edge_contact_count"]),
        "frame_edge_contact_count_to": int(crop_to["frame_edge_contact_count"]),
        "torso_union_contamination_from": float(
            crop_from["torso_union_other_person_coverage"]
        ),
        "torso_union_contamination_to": float(crop_to["torso_union_other_person_coverage"]),
        "maximum_torso_union_contamination": float(
            max(
                float(crop_from["torso_union_other_person_coverage"]),
                float(crop_to["torso_union_other_person_coverage"]),
            )
        ),
        "torso_overlap_count_from": int(crop_from["torso_other_person_overlap_count"]),
        "torso_overlap_count_to": int(crop_to["torso_other_person_overlap_count"]),
        "torso_center_inside_count_from": int(
            crop_from["torso_other_person_center_inside_count"]
        ),
        "torso_center_inside_count_to": int(
            crop_to["torso_other_person_center_inside_count"]
        ),
        "change_threshold": None,
        "split_threshold": None,
        "composite_change_score": None,
        "automatic_change_decision": None,
        "automatic_split_applied": False,
        "team_assignment_applied": False,
        "global_id_rewrite_applied": False,
        "audit_usage": "measurement_and_manual_review_ranking",
        "schema_version": TRANSITION_SCHEMA,
    }


def assign_independent_ranks(transitions: list[dict[str, Any]]) -> None:
    metric_to_rank_field = {
        "color_family_l1": "rank_by_color_family_l1",
        "lab_mean_distance_normalized": "rank_by_lab_mean_distance",
        "hue_histogram_l1": "rank_by_hue_histogram_l1",
        "saturation_histogram_l1": "rank_by_saturation_histogram_l1",
        "value_histogram_l1": "rank_by_value_histogram_l1",
    }
    for metric, field in metric_to_rank_field.items():
        ordered = sorted(
            transitions,
            key=lambda t: (
                -float(t[metric]),
                0 if t["dominant_family_changed"] else 1,
                int(t["track_id"]),
                int(t["transition_index"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row[field] = rank

    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in transitions:
        by_track[int(row["track_id"])].append(row)
    for track_id in by_track:
        for metric, field in (
            ("color_family_l1", "within_track_rank_by_color_family_l1"),
            ("lab_mean_distance_normalized", "within_track_rank_by_lab_mean_distance"),
        ):
            ordered = sorted(
                by_track[track_id],
                key=lambda t: (-float(t[metric]), int(t["transition_index"])),
            )
            for rank, row in enumerate(ordered, start=1):
                row[field] = rank


def _top_transition(
    transitions: Sequence[Mapping[str, Any]], *, metric: str
) -> dict[str, Any] | None:
    if not transitions:
        return None
    best = max(
        transitions,
        key=lambda t: (
            float(t[metric]),
            1 if t["dominant_family_changed"] else 0,
            -int(t["transition_index"]),
        ),
    )
    return {
        "transition_id": best["transition_id"],
        "transition_index": best["transition_index"],
        "crop_id_from": best["crop_id_from"],
        "crop_id_to": best["crop_id_to"],
        "frame_index_from": best["frame_index_from"],
        "frame_index_to": best["frame_index_to"],
        "metric_value": float(best[metric]),
    }


def build_track_purity_summaries(
    *,
    chronological_tracks: Mapping[int, Sequence[Mapping[str, Any]]],
    transitions_by_track: Mapping[int, Sequence[Mapping[str, Any]]],
    manifest_order_by_track: Mapping[int, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track_id in sorted(chronological_tracks):
        crops = list(chronological_tracks[track_id])
        transitions = list(transitions_by_track.get(track_id, []))
        families = [str(c["dominant_color_family"]) for c in crops]
        change_count = sum(
            1 for a, b in zip(families, families[1:]) if a != b
        )
        metric_names = (
            "color_family_l1",
            "hue_histogram_l1",
            "saturation_histogram_l1",
            "value_histogram_l1",
            "lab_mean_distance_normalized",
            "chromatic_pixel_ratio_abs_diff",
        )
        metric_stats = {
            name: _metric_summary([float(t[name]) for t in transitions])
            for name in metric_names
        }
        rows.append(
            {
                "track_id": int(track_id),
                "crop_count": len(crops),
                "transition_count": len(transitions),
                "crop_ids_manifest_order": list(manifest_order_by_track[track_id]),
                "crop_ids_chronological_order": [c["crop_id"] for c in crops],
                "frame_indices_chronological_order": [
                    int(c["frame_index"]) for c in crops
                ],
                "dominant_crop_families_chronological": families,
                "unique_dominant_family_count": len(set(families)),
                "dominant_family_change_count": int(change_count),
                "color_family_l1": metric_stats["color_family_l1"],
                "hue_histogram_l1": metric_stats["hue_histogram_l1"],
                "saturation_histogram_l1": metric_stats["saturation_histogram_l1"],
                "value_histogram_l1": metric_stats["value_histogram_l1"],
                "lab_mean_distance_normalized": metric_stats[
                    "lab_mean_distance_normalized"
                ],
                "chromatic_pixel_ratio_abs_diff": metric_stats[
                    "chromatic_pixel_ratio_abs_diff"
                ],
                "top_transition_by_color_family_l1": _top_transition(
                    transitions, metric="color_family_l1"
                ),
                "top_transition_by_lab_mean_distance": _top_transition(
                    transitions, metric="lab_mean_distance_normalized"
                ),
                "top_transition_by_hue_histogram_l1": _top_transition(
                    transitions, metric="hue_histogram_l1"
                ),
                "minimum_crop_laplacian_variance": float(
                    min(float(c["laplacian_variance"]) for c in crops)
                ),
                "maximum_crop_union_contamination": float(
                    max(float(c["union_other_person_crop_coverage"]) for c in crops)
                ),
                "maximum_torso_union_contamination": float(
                    max(float(c["torso_union_other_person_coverage"]) for c in crops)
                ),
                "frame_edge_contact_crop_count": int(
                    sum(1 for c in crops if int(c["frame_edge_contact_count"]) > 0)
                ),
                "single_crop_track": len(crops) == 1,
                "purity_label": None,
                "change_threshold": None,
                "split_threshold": None,
                "automatic_split_applied": False,
                "automatic_delete_applied": False,
                "global_id_rewrite_applied": False,
                "team_assignment": None,
                "selected_crop_audit_only": True,
                "schema_version": TRACK_PURITY_SCHEMA,
            }
        )
    return rows


def build_purity_audit_summary(
    *,
    joined_crops: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, str],
    audit_config: Mapping[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    multi = sum(1 for t in track_rows if int(t["crop_count"]) > 1)
    single = sum(1 for t in track_rows if int(t["crop_count"]) == 1)
    tracks_with_change = sum(
        1 for t in track_rows if int(t["dominant_family_change_count"]) > 0
    )
    tracks_without_change = len(track_rows) - tracks_with_change
    transitions_changed = sum(1 for t in transitions if t["dominant_family_changed"])

    crop_overlap = [
        float(c["union_other_person_crop_coverage"]) > 0 for c in joined_crops
    ]
    torso_overlap = [
        float(c["torso_union_other_person_coverage"]) > 0 for c in joined_crops
    ]
    both_pos = sum(1 for a, b in zip(crop_overlap, torso_overlap) if a and b)
    crop_only = sum(1 for a, b in zip(crop_overlap, torso_overlap) if a and not b)
    torso_only = sum(1 for a, b in zip(crop_overlap, torso_overlap) if (not a) and b)
    neither = sum(1 for a, b in zip(crop_overlap, torso_overlap) if (not a) and (not b))

    metric_dists = {}
    for name in (
        "color_family_l1",
        "hue_histogram_l1",
        "saturation_histogram_l1",
        "value_histogram_l1",
        "lab_mean_distance_normalized",
        "chromatic_pixel_ratio_abs_diff",
        "achromatic_pixel_ratio_abs_diff",
    ):
        metric_dists[name] = _percentile_stats([float(t[name]) for t in transitions])

    return {
        "status": "ok",
        "crop_count": len(joined_crops),
        "track_count": len(track_rows),
        "multi_crop_track_count": multi,
        "single_crop_track_count": single,
        "transition_count": len(transitions),
        "source_crop_manifest": source_paths["crop_manifest"],
        "source_quality_signals": source_paths["quality_signals"],
        "source_crop_kit_descriptors": source_paths["crop_kit_descriptors"],
        "source_track_kit_descriptors": source_paths["track_kit_descriptors"],
        "source_tracks": source_paths["tracks"],
        "source_kit_config": source_paths["kit_config"],
        "source_audit_config": source_paths["audit_config"],
        "chronology_order": list(CHRONOLOGY_ORDER),
        "compare_adjacent_crops_only": True,
        "selected_crop_audit_only": True,
        "full_track_segmentation_performed": False,
        "tracks_with_dominant_family_change": tracks_with_change,
        "tracks_without_dominant_family_change": tracks_without_change,
        "transitions_with_dominant_family_change": transitions_changed,
        "color_family_l1": metric_dists["color_family_l1"],
        "hue_histogram_l1": metric_dists["hue_histogram_l1"],
        "saturation_histogram_l1": metric_dists["saturation_histogram_l1"],
        "value_histogram_l1": metric_dists["value_histogram_l1"],
        "lab_mean_distance_normalized": metric_dists["lab_mean_distance_normalized"],
        "chromatic_pixel_ratio_abs_diff": metric_dists["chromatic_pixel_ratio_abs_diff"],
        "achromatic_pixel_ratio_abs_diff": metric_dists[
            "achromatic_pixel_ratio_abs_diff"
        ],
        "crops_with_torso_other_person_overlap": sum(
            1 for c in joined_crops if int(c["torso_other_person_overlap_count"]) > 0
        ),
        "crops_with_torso_other_person_center_inside": sum(
            1
            for c in joined_crops
            if int(c["torso_other_person_center_inside_count"]) > 0
        ),
        "torso_union_coverage": _percentile_stats(
            [float(c["torso_union_other_person_coverage"]) for c in joined_crops]
        ),
        "crop_overlap_gt0_torso_overlap_eq0_count": crop_only,
        "crop_overlap_eq0_torso_overlap_gt0_count": torso_only,
        "crop_and_torso_overlap_gt0_count": both_pos,
        "crop_and_torso_overlap_eq0_count": neither,
        "change_threshold": None,
        "split_threshold": None,
        "composite_change_score_created": False,
        "automatic_track_split_performed": False,
        "automatic_track_delete_performed": False,
        "automatic_global_id_rewrite_performed": False,
        "automatic_team_assignment_performed": False,
        "automatic_link_performed": False,
        "automatic_reject_performed": False,
        "raw_track_is_atomic_identity_guarantee": False,
        "within_track_change_is_identity_switch_proof": False,
        "no_detected_change_is_purity_proof": False,
        "identity_switch_ground_truth_available": False,
        "track_purity_ground_truth_available": False,
        "accuracy_claimed": False,
        "limitations": [
            "selected crops may miss changes between sampled frames",
            "tracking-bbox overlap cannot detect untracked people",
            (
                "descriptor changes can come from pose, lighting, occlusion or "
                "background rather than identity change"
            ),
            "low descriptor change does not prove track purity",
        ],
        "elapsed_sec": _ensure_finite_float(elapsed_sec, field="elapsed_sec"),
        "schema_version": AUDIT_SUMMARY_SCHEMA,
        "audit_config_stage_status": audit_config["stage_status"],
    }


def run_analyze_reid_track_purity(
    *,
    crop_manifest: str | Path,
    quality_signals: str | Path,
    crop_kit_descriptors: str | Path,
    track_kit_descriptors: str | Path,
    tracks: str | Path,
    kit_config: str | Path,
    audit_config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(crop_manifest).expanduser().resolve()
    quality_path = Path(quality_signals).expanduser().resolve()
    crop_kit_path = Path(crop_kit_descriptors).expanduser().resolve()
    track_kit_path = Path(track_kit_descriptors).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    kit_config_path = Path(kit_config).expanduser().resolve()
    audit_config_path = Path(audit_config).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    manifest_dir = manifest_path.parent

    check_output_collision(final_dir, overwrite=overwrite)
    validated_audit = load_track_purity_audit_config(audit_config_path)
    try:
        validated_kit = load_kit_descriptor_config(kit_config_path)
    except KitError as exc:
        raise PurityError(str(exc)) from exc

    temp_dir: Path | None = None
    try:
        try:
            manifest_rows = load_crop_manifest_for_kit(manifest_path)
            quality_rows = load_quality_signals_for_kit(
                quality_path, manifest_rows=manifest_rows
            )
        except KitError as exc:
            raise PurityError(str(exc)) from exc

        # Extra duplicate track+frame guard on manifest
        seen_tf: set[tuple[int, int]] = set()
        for row in manifest_rows:
            key = (int(row["track_id"]), int(row["frame_index"]))
            if key in seen_tf:
                raise PurityError(f"duplicate track_id+frame_index in manifest: {key}")
            seen_tf.add(key)
            _validate_relative_path(row["crop_relative_path"], manifest_dir=manifest_dir)

        crop_kit_rows = load_crop_kit_for_purity(
            crop_kit_path, manifest_rows=manifest_rows
        )
        load_track_kit_for_purity(
            track_kit_path, crop_kit_rows=crop_kit_rows, kit_config=validated_kit
        )

        try:
            observations = load_person_track_observations(tracks_path)
            frame_width, frame_height = infer_frame_size(
                observations=observations, manifest_rows=manifest_rows
            )
            by_frame = index_observations_by_frame(observations)
        except QualityError as exc:
            raise PurityError(str(exc)) from exc

        joined: list[dict[str, Any]] = []
        for manifest_row, quality_row, kit_row in zip(
            manifest_rows, quality_rows, crop_kit_rows
        ):
            frame_index = int(manifest_row["frame_index"])
            frame_obs = by_frame.get(frame_index, [])
            try:
                verify_target_observation(
                    manifest_row=manifest_row,
                    frame_observations=frame_obs,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            except QualityError as exc:
                raise PurityError(str(exc)) from exc

            merged = {
                **kit_row,
                "bbox_xyxy": list(manifest_row["bbox_xyxy"]),
                "laplacian_variance": float(quality_row["laplacian_variance"]),
                "union_other_person_crop_coverage": float(
                    quality_row["union_other_person_crop_coverage"]
                ),
                "frame_edge_contact_count": int(quality_row["frame_edge_contact_count"]),
            }
            torso = compute_torso_tracking_contamination(
                crop_row=merged,
                frame_observations=frame_obs,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            merged.update(torso)
            joined.append(merged)

        chronological = build_chronological_tracks(joined)
        manifest_order_by_track: dict[int, list[str]] = defaultdict(list)
        for row in manifest_rows:
            manifest_order_by_track[int(row["track_id"])].append(str(row["crop_id"]))

        transitions: list[dict[str, Any]] = []
        transitions_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
        lab_max = float(
            validated_audit["distance_normalization"]["lab_uint8_max_distance"]
        )
        for track_id, crops in chronological.items():
            for i in range(len(crops) - 1):
                transition = build_transition(
                    track_id=track_id,
                    transition_index=i + 1,
                    crop_from=crops[i],
                    crop_to=crops[i + 1],
                    lab_max=lab_max,
                )
                transitions.append(transition)
                transitions_by_track[track_id].append(transition)

        assign_independent_ranks(transitions)
        transitions.sort(key=lambda t: (int(t["track_id"]), int(t["transition_index"])))

        track_summaries = build_track_purity_summaries(
            chronological_tracks=chronological,
            transitions_by_track=transitions_by_track,
            manifest_order_by_track=manifest_order_by_track,
        )
        elapsed = time.perf_counter() - started
        summary = build_purity_audit_summary(
            joined_crops=joined,
            track_rows=track_summaries,
            transitions=transitions,
            source_paths={
                "crop_manifest": str(manifest_path),
                "quality_signals": str(quality_path),
                "crop_kit_descriptors": str(crop_kit_path),
                "track_kit_descriptors": str(track_kit_path),
                "tracks": str(tracks_path),
                "kit_config": str(kit_config_path),
                "audit_config": str(audit_config_path),
            },
            audit_config=validated_audit,
            elapsed_sec=elapsed,
        )

        temp_dir = create_temp_purity_dir(final_dir)
        write_manifest_jsonl(temp_dir / TRANSITION_NAME, transitions)
        write_manifest_jsonl(temp_dir / TRACK_PURITY_NAME, track_summaries)
        (temp_dir / AUDIT_SUMMARY_NAME).write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        finalized = finalize_purity_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "transition_path": str(finalized / TRANSITION_NAME),
        "track_purity_path": str(finalized / TRACK_PURITY_NAME),
        "summary_path": str(finalized / AUDIT_SUMMARY_NAME),
        "crop_count": summary["crop_count"],
        "track_count": summary["track_count"],
        "multi_crop_track_count": summary["multi_crop_track_count"],
        "single_crop_track_count": summary["single_crop_track_count"],
        "transition_count": summary["transition_count"],
        "tracks_with_dominant_family_change": summary[
            "tracks_with_dominant_family_change"
        ],
        "transitions_with_dominant_family_change": summary[
            "transitions_with_dominant_family_change"
        ],
        "crops_with_torso_other_person_overlap": summary[
            "crops_with_torso_other_person_overlap"
        ],
        "color_family_l1": summary["color_family_l1"],
        "lab_mean_distance_normalized": summary["lab_mean_distance_normalized"],
        "change_threshold": None,
        "split_threshold": None,
        "automatic_track_split_performed": False,
        "automatic_global_id_rewrite_performed": False,
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
    }
