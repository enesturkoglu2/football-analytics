#!/usr/bin/env python
"""Stage 5C-C3F-A label-blind / PARSeq-blind independent holdout design.

Builds primary/reserve review packages from unreviewed canonical jersey
crops without using manual labels or PARSeq/MMOCR predictions for
selection. Pre-registers a discovery-derived validation candidate cut
without selecting a deployment threshold.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMA = "reid_jersey_parseq_holdout_config_v1"
PREREG_SCHEMA = "reid_jersey_parseq_holdout_preregistration_v1"
SUMMARY_SCHEMA = "reid_jersey_parseq_holdout_design_summary_v1"
MANIFEST_SCHEMA = "reid_jersey_parseq_holdout_design_manifest_v1"
UNIVERSE_SCHEMA = "reid_jersey_parseq_holdout_source_universe_v1"
POOL_SCHEMA = "reid_jersey_parseq_holdout_independent_pool_v1"
SELECTED_SCHEMA = "reid_jersey_parseq_holdout_selected_item_v1"

STRATA = (
    "high_signal_candidate",
    "mid_signal_candidate",
    "safety_candidate",
)

RANK_FIELDS = (
    "roi_height_global_rank",
    "local_contrast_global_rank",
    "contamination_low_global_rank",
    "laplacian_rank_within_size_stratum",
    "roi_area_global_rank",
    "entropy_global_rank",
)

PROHIBITED_SELECTION_PREFIXES = ("manual_",)
PROHIBITED_SELECTION_SUBSTRINGS = (
    "parseq",
    "mmocr",
    "prediction",
    "sequence_confidence",
    "accepted_prediction",
    "exact_match",
    "wrong_number",
)

FINAL_TOP_LEVEL = (
    "holdout_source_universe.jsonl",
    "holdout_exclusion_report.json",
    "holdout_independent_pool.jsonl",
    "holdout_primary_manifest.jsonl",
    "holdout_reserve_manifest.jsonl",
    "holdout_annotation_template.csv",
    "holdout_review_instructions.md",
    "holdout_preregistration.json",
    "holdout_design_summary.json",
    "holdout_design_manifest.json",
)

IMAGE_DIRS = (
    "primary_contact_sheets",
    "reserve_contact_sheets",
    "primary_item_overviews",
    "reserve_item_overviews",
)

TEMPLATE_COLUMNS = (
    "holdout_item_id",
    "batch",
    "batch_order",
    "review_item_id",
    "stratum",
    "crop_id",
    "segment_id",
    "raw_track_id",
    "frame_index",
    "source_crop_path",
    "source_crop_sha256",
    "overview_path",
    "contact_sheet_path",
    "manual_crop_valid",
    "manual_number_visible",
    "manual_number_readable",
    "manual_jersey_number",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

MANUAL_TEMPLATE_FIELDS = (
    "manual_crop_valid",
    "manual_number_visible",
    "manual_number_readable",
    "manual_jersey_number",
    "manual_notes",
    "reviewer",
    "reviewed_at",
)

ALLOWED_TERNARY = frozenset({"yes", "no", "uncertain"})


class HoldoutDesignError(RuntimeError):
    """Contract failure for Stage 5C-C3F-A holdout design."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise HoldoutDesignError(f"{path}:{line_no} not an object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise HoldoutDesignError("unexpected config schema_version")
    if not config.get("offline_required"):
        raise HoldoutDesignError("offline_required must be true")
    if not config.get("threshold_selection_forbidden"):
        raise HoldoutDesignError("threshold_selection_forbidden must be true")
    return config


def _finite_number(value: Any, label: str) -> float:
    if value is None:
        raise HoldoutDesignError(f"missing {label}")
    number = float(value)
    if not math.isfinite(number):
        raise HoldoutDesignError(f"non-finite {label}: {value}")
    return number


def invert_one_based_rank_to_unit(rank: Any, n: int) -> float:
    """Map one-based ranks to [0, 1] with lower rank = better = higher unit."""
    if n < 1:
        raise HoldoutDesignError("rank universe size must be >= 1")
    rank_i = int(rank)
    if rank_i < 1 or rank_i > n:
        raise HoldoutDesignError(f"rank {rank_i} outside 1..{n}")
    if n == 1:
        return 1.0
    return float(n - rank_i) / float(n - 1)


def composite_rank_score(
    row: Mapping[str, Any],
    *,
    fields: Sequence[str] = RANK_FIELDS,
    universe_size: int,
) -> float:
    assert_no_prohibited_selection_fields(row)
    units = [
        invert_one_based_rank_to_unit(row[field], universe_size)
        for field in fields
    ]
    return float(sum(units) / len(units))


def assert_no_prohibited_selection_fields(row: Mapping[str, Any]) -> None:
    for key in row:
        lower = str(key).lower()
        if any(lower.startswith(prefix) for prefix in PROHIBITED_SELECTION_PREFIXES):
            # Presence of blank manual_* on source rows is allowed; using them
            # for ranking is forbidden. Callers must not pass them into score.
            continue
        if any(token in lower for token in PROHIBITED_SELECTION_SUBSTRINGS):
            raise HoldoutDesignError(
                f"prohibited selection field present in ranking context: {key}"
            )


def ranking_feature_vector(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only model-independent rank fields used for selection."""
    out: dict[str, Any] = {}
    for field in RANK_FIELDS:
        if field not in row:
            raise HoldoutDesignError(f"missing ranking field: {field}")
        value = row[field]
        if isinstance(value, float) and not math.isfinite(value):
            raise HoldoutDesignError(f"NaN/Inf ranking field: {field}")
        out[field] = int(value)
    return out


def assign_signal_strata(scored_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Tercile strata by composite score (desc). Names are not ground truth."""
    if not scored_rows:
        return []
    ordered = sorted(
        scored_rows,
        key=lambda row: (-float(row["composite_score"]), str(row["review_item_id"])),
    )
    n = len(ordered)
    base = n // 3
    rem = n % 3
    high_n = base + (1 if rem > 0 else 0)
    mid_n = base + (1 if rem > 1 else 0)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        item = dict(row)
        if index < high_n:
            item["signal_stratum"] = "high_signal_candidate"
        elif index < high_n + mid_n:
            item["signal_stratum"] = "mid_signal_candidate"
        else:
            item["signal_stratum"] = "safety_candidate"
        item["stratum_is_ground_truth"] = False
        out.append(item)
    return out


def average_hash_8x8(image_bgr: np.ndarray) -> np.ndarray:
    """Deterministic 8x8 average hash; grayscale + INTER_AREA resize."""
    if image_bgr is None or image_bgr.size == 0:
        raise HoldoutDesignError("empty image for average hash")
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(small))
    bits = (small.astype(np.float64) >= mean).astype(np.uint8).reshape(64)
    return bits


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        raise HoldoutDesignError("hash shape mismatch")
    return int(np.sum(a.astype(np.uint8) != b.astype(np.uint8)))


def build_exclusion_sets(discovery_rows: Sequence[Mapping[str, Any]]) -> dict[str, set[Any]]:
    return {
        "review_item_id": {str(r["review_item_id"]) for r in discovery_rows},
        "source_crop_sha256": {str(r["source_crop_sha256"]) for r in discovery_rows},
        "crop_id": {str(r["crop_id"]) for r in discovery_rows},
        "segment_id": {str(r["segment_id"]) for r in discovery_rows},
        "raw_track_id": {int(r["raw_track_id"]) for r in discovery_rows},
    }


def hard_exclusion_reason(
    row: Mapping[str, Any],
    *,
    reviewed_ids: set[str],
    discovery: Mapping[str, set[Any]],
) -> Optional[str]:
    rid = str(row["review_item_id"])
    if rid in reviewed_ids:
        return "reviewed_pilot"
    if rid in discovery["review_item_id"]:
        return "discovery_review_item_id"
    if str(row["source_crop_sha256"]) in discovery["source_crop_sha256"]:
        return "discovery_source_crop_sha256"
    if str(row["crop_id"]) in discovery["crop_id"]:
        return "discovery_crop_id"
    if str(row["segment_id"]) in discovery["segment_id"]:
        return "discovery_segment_id"
    if int(row["raw_track_id"]) in discovery["raw_track_id"]:
        return "discovery_raw_track_id"
    return None


class DiversityState:
    def __init__(
        self,
        *,
        max_per_segment: int = 1,
        max_per_raw_track: int = 2,
        min_frame_gap: int = 60,
    ) -> None:
        self.max_per_segment = max_per_segment
        self.max_per_raw_track = max_per_raw_track
        self.min_frame_gap = min_frame_gap
        self.segment_counts: dict[str, int] = defaultdict(int)
        self.track_counts: dict[int, int] = defaultdict(int)
        self.track_frames: dict[int, list[int]] = defaultdict(list)
        self.selected_ids: set[str] = set()
        self.selected_shas: set[str] = set()

    def accepts(self, row: Mapping[str, Any]) -> bool:
        rid = str(row["review_item_id"])
        if rid in self.selected_ids:
            return False
        sha = str(row["source_crop_sha256"])
        if sha in self.selected_shas:
            return False
        segment = str(row["segment_id"])
        if self.segment_counts[segment] >= self.max_per_segment:
            return False
        track = int(row["raw_track_id"])
        if self.track_counts[track] >= self.max_per_raw_track:
            return False
        frame = int(row["frame_index"])
        for prior in self.track_frames[track]:
            if abs(frame - prior) < self.min_frame_gap:
                return False
        return True

    def add(self, row: Mapping[str, Any]) -> None:
        self.selected_ids.add(str(row["review_item_id"]))
        self.selected_shas.add(str(row["source_crop_sha256"]))
        self.segment_counts[str(row["segment_id"])] += 1
        track = int(row["raw_track_id"])
        self.track_counts[track] += 1
        self.track_frames[track].append(int(row["frame_index"]))


def select_with_diversity(
    candidates: Sequence[Mapping[str, Any]],
    quota: int,
    state: DiversityState,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["composite_score"]), str(row["review_item_id"])),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        if len(selected) >= quota:
            break
        if not state.accepts(row):
            continue
        item = dict(row)
        selected.append(item)
        state.add(item)
    return selected


def select_primary_and_reserve(
    stratified_rows: Sequence[Mapping[str, Any]],
    *,
    combined_quotas: Mapping[str, int],
    primary_quotas: Mapping[str, int],
    reserve_quotas: Mapping[str, int],
    diversity: Optional[DiversityState] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select combined quotas first, then split to primary/reserve.

    Avoids sequential primary-then-reserve starvation across strata.
    """
    by_stratum: dict[str, list[Mapping[str, Any]]] = {name: [] for name in STRATA}
    for row in stratified_rows:
        stratum = str(row["signal_stratum"])
        if stratum not in by_stratum:
            raise HoldoutDesignError(f"unknown stratum: {stratum}")
        by_stratum[stratum].append(row)

    state = diversity or DiversityState()
    combined_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for stratum in STRATA:
        need = int(combined_quotas[stratum])
        picked = select_with_diversity(by_stratum[stratum], need, state)
        if len(picked) < need:
            raise HoldoutDesignError(
                f"BLOCKED_STRATUM_CAPACITY stratum={stratum} "
                f"need={need} got={len(picked)}"
            )
        expected_split = int(primary_quotas[stratum]) + int(reserve_quotas[stratum])
        if expected_split != need:
            raise HoldoutDesignError(
                f"quota mismatch for {stratum}: combined={need} "
                f"primary+reserve={expected_split}"
            )
        combined_by_stratum[stratum] = picked

    primary: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    primary_order = 0
    reserve_order = 0
    for stratum in STRATA:
        picked = combined_by_stratum[stratum]
        p_n = int(primary_quotas[stratum])
        for index, row in enumerate(picked):
            item = dict(row)
            item["stratum"] = stratum
            if index < p_n:
                primary_order += 1
                item["batch"] = "primary"
                item["batch_order"] = primary_order
                item["holdout_item_id"] = f"holdout_primary_{primary_order:03d}"
                primary.append(item)
            else:
                reserve_order += 1
                item["batch"] = "reserve"
                item["batch_order"] = reserve_order
                item["holdout_item_id"] = f"holdout_reserve_{reserve_order:03d}"
                reserve.append(item)
    return primary, reserve


def derive_validation_candidate_cut(
    operating_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Lowest confidence_cut among perfect-safe non-sentinel ops; none selected."""
    if not operating_points:
        raise HoldoutDesignError("empty operating points")
    for row in operating_points:
        if row.get("selected") is not False:
            raise HoldoutDesignError("operating point selected must be false")
        for key in ("exact_retained", "wrong_positive_retained", "negative_retained"):
            _finite_number(row.get(key), key)

    perfect = [
        row
        for row in operating_points
        if int(row["exact_retained"]) > 0
        and int(row["wrong_positive_retained"]) == 0
        and int(row["negative_retained"]) == 0
        and row.get("sentinel_type") is None
    ]
    if not perfect:
        raise HoldoutDesignError("no perfect-safe non-sentinel operating points")
    cuts = sorted(float(row["confidence_cut"]) for row in perfect)
    primary_cut = cuts[0]
    secondary = [c for c in cuts if c != primary_cut]
    return {
        "validation_candidate_cut": primary_cut,
        "secondary_descriptive_cuts": secondary,
        "perfect_safe_non_sentinel_count": len(perfect),
        "derivation_rule": (
            "lowest_confidence_cut_among_exact_gt0_wrong_eq0_neg_eq0_sentinel_null"
        ),
        "all_operating_points_selected_false": True,
        "deployment_threshold_selected": False,
        "threshold_selected_for_production": False,
        "candidate_derived_only_from_c3e_discovery_set": True,
        "selected_before_holdout_annotation_and_inference": True,
        "not_a_deployment_threshold": True,
        "not_calibrated_probability": True,
        "requires_independent_validation": True,
    }


def classify_holdout_decision(
    *,
    readable_positive: int,
    negative_or_safety: int,
    accepted_wrong_positive: int,
    accepted_negative: int,
    accepted_exact: int,
    min_readable: int = 12,
    min_negative: int = 24,
) -> str:
    """Pre-registered decision rules for a future holdout evaluation gate."""
    if readable_positive < min_readable or negative_or_safety < min_negative:
        return "BLOCKED_INSUFFICIENT_LABELED_HOLDOUT"
    if accepted_wrong_positive > 0 or accepted_negative > 0:
        return "FAIL_INDEPENDENT_GATE_SAFETY"
    if accepted_exact >= 2:
        return "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL"
    return "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT"


def validate_holdout_annotation_values(
    *,
    manual_crop_valid: str,
    manual_number_visible: str,
    manual_number_readable: str,
    manual_jersey_number: str,
) -> list[str]:
    """Blank-template validation rules for holdout manual fields."""
    errors: list[str] = []
    for field, value in (
        ("manual_crop_valid", manual_crop_valid),
        ("manual_number_visible", manual_number_visible),
        ("manual_number_readable", manual_number_readable),
    ):
        if value and value not in ALLOWED_TERNARY:
            errors.append(f"{field} invalid")
    jersey = (manual_jersey_number or "").strip()
    readable = manual_number_readable
    if readable == "yes":
        if not jersey:
            errors.append("readable=yes requires jersey_number")
        elif not jersey.isdigit() or not (1 <= len(jersey) <= 2):
            errors.append("jersey_number must be 1-2 ASCII digits")
        if manual_number_visible != "yes":
            errors.append("readable=yes requires visible=yes")
    elif readable in ("no", "uncertain"):
        if jersey:
            errors.append("readable!=yes requires blank jersey_number")
    elif jersey:
        errors.append("jersey_number requires readable=yes")
    return errors


def _git_output(args: Sequence[str]) -> str:
    env = os.environ.copy()
    git_dir = _PROJECT_ROOT / ".git"
    if git_dir.exists():
        env["GIT_DIR"] = str(git_dir)
        env["GIT_WORK_TREE"] = str(_PROJECT_ROOT)
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=env,
    )
    return result.stdout


def project_git_head() -> str:
    return _git_output(["rev-parse", "HEAD"]).strip()


def project_git_status() -> str:
    return _git_output(["status", "--short", "--untracked-files=all"])


def _verify_pinned(path: Path, expected_sha: str, expected_rows: int, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HoldoutDesignError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise HoldoutDesignError(
            f"{label} sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    rows = load_jsonl(path)
    if len(rows) != expected_rows:
        raise HoldoutDesignError(
            f"{label} row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    return rows


def _resolve_input(project_root: Path, relative: str) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def build_source_universe(
    canonical: Sequence[Mapping[str, Any]],
    *,
    reviewed_ids: set[str],
    kit_by_crop: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ids = [str(r["review_item_id"]) for r in canonical]
    if len(ids) != len(set(ids)):
        raise HoldoutDesignError("duplicate review_item_id in canonical")
    universe: list[dict[str, Any]] = []
    for row in canonical:
        crop_id = str(row["crop_id"])
        kit = kit_by_crop.get(crop_id, {})
        ranks = ranking_feature_vector(row)
        item = {
            "schema_version": UNIVERSE_SCHEMA,
            "review_item_id": str(row["review_item_id"]),
            "review_index": int(row.get("review_index") or 0),
            "crop_id": crop_id,
            "segment_id": str(row["segment_id"]),
            "raw_track_id": int(row["raw_track_id"]),
            "frame_index": int(row["frame_index"]),
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
            "pilot_reviewed": str(row["review_item_id"]) in reviewed_ids,
            "kit_family": kit.get("kit_family"),
            "kit_side_label": kit.get("kit_side_label") or kit.get("predicted_side"),
            **ranks,
        }
        universe.append(item)
    return universe


def apply_hard_exclusions(
    universe: Sequence[Mapping[str, Any]],
    *,
    reviewed_ids: set[str],
    discovery: Mapping[str, set[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    for row in universe:
        if row.get("pilot_reviewed"):
            reasons["reviewed_pilot"] += 1
            continue
        reason = hard_exclusion_reason(
            row, reviewed_ids=reviewed_ids, discovery=discovery
        )
        if reason is not None:
            reasons[reason] += 1
            continue
        kept.append(dict(row))
    report = {
        "reviewed_excluded": int(reasons.get("reviewed_pilot", 0)),
        "discovery_review_item_id": int(reasons.get("discovery_review_item_id", 0)),
        "discovery_source_crop_sha256": int(
            reasons.get("discovery_source_crop_sha256", 0)
        ),
        "discovery_crop_id": int(reasons.get("discovery_crop_id", 0)),
        "discovery_segment_id": int(reasons.get("discovery_segment_id", 0)),
        "discovery_raw_track_id": int(reasons.get("discovery_raw_track_id", 0)),
        "hard_exclusion_reason_counts": dict(sorted(reasons.items())),
        "remaining_after_hard_exclusion": len(kept),
    }
    return kept, report


def apply_near_duplicate_exclusion(
    candidates: Sequence[Mapping[str, Any]],
    *,
    discovery_rows: Sequence[Mapping[str, Any]],
    project_root: Path,
    max_hamming: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery_hashes: list[tuple[str, np.ndarray]] = []
    for row in discovery_rows:
        path = _resolve_input(project_root, str(row["source_crop_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise HoldoutDesignError(f"cannot decode discovery crop: {path}")
        discovery_hashes.append((str(row["review_item_id"]), average_hash_8x8(image)))

    kept: list[dict[str, Any]] = []
    excluded = 0
    pairs: list[dict[str, Any]] = []
    for row in candidates:
        path = _resolve_input(project_root, str(row["source_crop_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise HoldoutDesignError(f"cannot decode candidate crop: {path}")
        cand_hash = average_hash_8x8(image)
        hit = None
        best = 65
        for disc_id, disc_hash in discovery_hashes:
            dist = hamming_distance(cand_hash, disc_hash)
            if dist < best:
                best = dist
            if dist <= max_hamming:
                hit = disc_id
                break
        item = dict(row)
        item["average_hash_8x8"] = "".join(str(int(b)) for b in cand_hash.tolist())
        item["min_hamming_to_discovery"] = int(best)
        if hit is not None:
            excluded += 1
            pairs.append(
                {
                    "review_item_id": str(row["review_item_id"]),
                    "near_duplicate_of_discovery_review_item_id": hit,
                    "hamming_distance": int(best),
                }
            )
            continue
        kept.append(item)
    report = {
        "algorithm": "average_hash_8x8_grayscale_inter_area",
        "max_hamming_distance": max_hamming,
        "near_duplicate_excluded": excluded,
        "near_duplicate_pairs": pairs,
        "remaining_after_near_duplicate": len(kept),
    }
    return kept, report


def score_independent_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical_universe_size: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        features = ranking_feature_vector(row)
        # Guard: ranking context must not include prohibited prediction fields.
        assert_no_prohibited_selection_fields(features)
        score = composite_rank_score(
            features, universe_size=canonical_universe_size
        )
        item = dict(row)
        item["composite_score"] = score
        item["schema_version"] = POOL_SCHEMA
        scored.append(item)
    return assign_signal_strata(scored)


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)


def render_item_overview(
    item: Mapping[str, Any],
    image: np.ndarray,
) -> np.ndarray:
    canvas = np.full((480, 360, 3), 32, dtype=np.uint8)
    lines = [
        str(item["review_item_id"]),
        f"stratum={item['stratum']}",
        f"frame={item['frame_index']}",
        f"crop={item['crop_width_px']}x{item['crop_height_px']}",
        f"roi={item['roi_width_px']}x{item['roi_height_px']}",
        "NO PREDICTION / NO LABEL",
    ]
    y = 18
    for text in lines:
        cv2.putText(
            canvas,
            text[:48],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 16
    display = _fit_display(image, 340, 200)
    dh, dw = display.shape[:2]
    ox = (360 - dw) // 2
    oy = 110
    canvas[oy : oy + dh, ox : ox + dw] = display
    scale = dw / max(int(item["crop_width_px"]), 1)
    rx0 = ox + int(round(int(item["roi_x_min"]) * scale))
    ry0 = oy + int(round(int(item["roi_y_min"]) * scale))
    rx1 = ox + int(round(int(item["roi_x_max"]) * scale))
    ry1 = oy + int(round(int(item["roi_y_max"]) * scale))
    cv2.rectangle(
        canvas,
        (rx0, ry0),
        (max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)),
        (0, 220, 255),
        1,
    )
    roi = image[
        int(item["roi_y_min"]) : int(item["roi_y_max"]),
        int(item["roi_x_min"]) : int(item["roi_x_max"]),
    ]
    if roi.size:
        zoom = _fit_display(roi, 340, 140)
        zh, zw = zoom.shape[:2]
        zx = (360 - zw) // 2
        zy = 330
        canvas[zy : zy + zh, zx : zx + zw] = zoom
    return canvas


def render_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    images: Mapping[str, np.ndarray],
    *,
    max_items: int = 16,
) -> np.ndarray:
    if len(items) > max_items:
        raise HoldoutDesignError("contact sheet exceeds max items")
    cols = 4
    rows = int(math.ceil(len(items) / cols)) if items else 1
    tile_w, tile_h = 280, 360
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 24, dtype=np.uint8)
    for index, item in enumerate(items):
        r, c = divmod(index, cols)
        image = images[str(item["crop_id"])]
        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        meta = [
            str(item["review_item_id"])[:40],
            str(item["stratum"]),
            f"f={item['frame_index']}",
            f"{item['crop_width_px']}x{item['crop_height_px']}",
            f"roi {item['roi_width_px']}x{item['roi_height_px']}",
        ]
        y = 14
        for text in meta:
            cv2.putText(
                tile,
                text,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 14
        display = _fit_display(image, tile_w - 12, 160)
        dh, dw = display.shape[:2]
        ox = (tile_w - dw) // 2
        oy = 90
        tile[oy : oy + dh, ox : ox + dw] = display
        scale = dw / max(int(item["crop_width_px"]), 1)
        rx0 = ox + int(round(int(item["roi_x_min"]) * scale))
        ry0 = oy + int(round(int(item["roi_y_min"]) * scale))
        rx1 = ox + int(round(int(item["roi_x_max"]) * scale))
        ry1 = oy + int(round(int(item["roi_y_max"]) * scale))
        cv2.rectangle(
            tile,
            (rx0, ry0),
            (max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)),
            (0, 220, 255),
            1,
        )
        roi = image[
            int(item["roi_y_min"]) : int(item["roi_y_max"]),
            int(item["roi_x_min"]) : int(item["roi_x_max"]),
        ]
        if roi.size:
            zoom = _fit_display(roi, tile_w - 12, 100)
            zh, zw = zoom.shape[:2]
            zx = (tile_w - zw) // 2
            zy = 260
            tile[zy : zy + zh, zx : zx + zw] = zoom
        y0 = r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise HoldoutDesignError(f"failed to write png: {path}")


def build_review_instructions_tr() -> str:
    return """# Stage 5C-C3F-A Holdout İnceleme Talimatları

Bu paket, C3D discovery setinden bağımsız seçilmiş adayları içerir.

## Temel kurallar

- Stratum (`high_signal_candidate` / `mid_signal_candidate` /
  `safety_candidate`) ground-truth değildir. Yüksek sinyal okunabilir
  pozitif demek değildir; safety gerçek negatif demek değildir.
- PARSeq prediction veya confidence review sırasında gösterilmez ve
  kullanılmamalıdır.
- Forma numarası net biçimde okunmuyorsa `manual_number_readable=no`
  veya `uncertain` kullanın.
- Tahmin ederek jersey number yazmayın.
- Tek ve çift haneli numaraları göründüğü exact string olarak yazın.
  `09` ile `9` aynı kabul edilmez.
- Human review bitmeden model inference çalıştırılmayacaktır.
- Önce primary batch incelenir.
- Reserve yalnız primary minimum readable-positive hedefini
  karşılamazsa, önceden sabitlenen deterministic sırayla açılır.
- Bu annotation bir deployment accuracy benchmark değildir.

## Alanlar

- `manual_crop_valid`: yes / no / uncertain
- `manual_number_visible`: yes / no / uncertain
- `manual_number_readable`: yes / no / uncertain
- `manual_jersey_number`: boş veya `0`–`99` (1–2 digit exact string)

`readable=yes` ise jersey number zorunludur. `readable=no` veya
`uncertain` ise jersey number boş kalmalıdır.
"""


def build_annotation_template_rows(
    primary: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in list(primary) + list(reserve):
        row = {col: "" for col in TEMPLATE_COLUMNS}
        row.update(
            {
                "holdout_item_id": str(item["holdout_item_id"]),
                "batch": str(item["batch"]),
                "batch_order": str(item["batch_order"]),
                "review_item_id": str(item["review_item_id"]),
                "stratum": str(item["stratum"]),
                "crop_id": str(item["crop_id"]),
                "segment_id": str(item["segment_id"]),
                "raw_track_id": str(item["raw_track_id"]),
                "frame_index": str(item["frame_index"]),
                "source_crop_path": str(item["source_crop_path"]),
                "source_crop_sha256": str(item["source_crop_sha256"]),
                "overview_path": str(item.get("overview_path") or ""),
                "contact_sheet_path": str(item.get("contact_sheet_path") or ""),
            }
        )
        rows.append(row)
    if len(rows) != 96:
        raise HoldoutDesignError(f"annotation template must have 96 rows, got {len(rows)}")
    return rows


def write_annotation_template(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in TEMPLATE_COLUMNS})


def _artifact_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        meta["row_count"] = sum(1 for line in path.open() if line.strip())
    elif path.suffix == ".csv":
        meta["row_count"] = max(0, sum(1 for _ in path.open()) - 1)
    elif path.suffix == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise HoldoutDesignError(f"cannot read png meta: {path}")
        meta["height"] = int(image.shape[0])
        meta["width"] = int(image.shape[1])
    return meta


def run_holdout_design(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    git_before: str,
    git_head: str,
) -> dict[str, Any]:
    inputs = config["inputs"]
    canonical = _verify_pinned(
        _resolve_input(project_root, inputs["canonical_review_items"]["path"]),
        inputs["canonical_review_items"]["sha256"],
        int(inputs["canonical_review_items"]["expected_rows"]),
        "canonical_review_items",
    )
    pilot = _verify_pinned(
        _resolve_input(project_root, inputs["pilot_reviewed_items"]["path"]),
        inputs["pilot_reviewed_items"]["sha256"],
        int(inputs["pilot_reviewed_items"]["expected_rows"]),
        "pilot_reviewed_items",
    )
    visibility = _verify_pinned(
        _resolve_input(project_root, inputs["visibility_crop_signals"]["path"]),
        inputs["visibility_crop_signals"]["sha256"],
        int(inputs["visibility_crop_signals"]["expected_rows"]),
        "visibility_crop_signals",
    )
    discovery = _verify_pinned(
        _resolve_input(project_root, inputs["discovery_inference_manifest"]["path"]),
        inputs["discovery_inference_manifest"]["sha256"],
        int(inputs["discovery_inference_manifest"]["expected_rows"]),
        "discovery_inference_manifest",
    )
    operating_points = _verify_pinned(
        _resolve_input(project_root, inputs["c3e_operating_points"]["path"]),
        inputs["c3e_operating_points"]["sha256"],
        int(inputs["c3e_operating_points"]["expected_rows"]),
        "c3e_operating_points",
    )
    kit_rows = _verify_pinned(
        _resolve_input(project_root, inputs["kit_descriptors"]["path"]),
        inputs["kit_descriptors"]["sha256"],
        int(inputs["kit_descriptors"]["expected_rows"]),
        "kit_descriptors",
    )

    # Visibility is pinned for provenance; ranks already live on canonical items.
    vis_by_crop = {str(r["crop_id"]): r for r in visibility}
    for row in canonical:
        crop_id = str(row["crop_id"])
        if crop_id not in vis_by_crop:
            raise HoldoutDesignError(f"canonical crop missing visibility: {crop_id}")

    reviewed_ids = {str(r["review_item_id"]) for r in pilot}
    if len(reviewed_ids) != 78:
        raise HoldoutDesignError("reviewed pilot id count != 78")
    discovery_sets = build_exclusion_sets(discovery)
    kit_by_crop = {str(r["crop_id"]): r for r in kit_rows}

    universe = build_source_universe(
        canonical, reviewed_ids=reviewed_ids, kit_by_crop=kit_by_crop
    )
    if len(universe) != 474:
        raise HoldoutDesignError("canonical universe must be 474")

    after_hard, hard_report = apply_hard_exclusions(
        universe, reviewed_ids=reviewed_ids, discovery=discovery_sets
    )
    if len(after_hard) == 0:
        raise HoldoutDesignError("BLOCKED_INSUFFICIENT_INDEPENDENT_POOL")

    max_hamming = int(config["near_duplicate"]["max_hamming_distance"])
    after_nd, nd_report = apply_near_duplicate_exclusion(
        after_hard,
        discovery_rows=discovery,
        project_root=project_root,
        max_hamming=max_hamming,
    )
    if len(after_nd) == 0:
        raise HoldoutDesignError("BLOCKED_INSUFFICIENT_INDEPENDENT_POOL")

    pool = score_independent_pool(after_nd, canonical_universe_size=len(canonical))
    stratum_counts = {name: 0 for name in STRATA}
    for row in pool:
        stratum_counts[str(row["signal_stratum"])] += 1

    diversity_cfg = config["diversity"]
    state = DiversityState(
        max_per_segment=int(diversity_cfg["max_per_segment"]),
        max_per_raw_track=int(diversity_cfg["max_per_raw_track"]),
        min_frame_gap=int(diversity_cfg["min_frame_gap_same_raw_track"]),
    )
    selection = config["selection"]
    primary, reserve = select_primary_and_reserve(
        pool,
        combined_quotas=selection["combined_quotas"],
        primary_quotas=selection["primary_quotas"],
        reserve_quotas=selection["reserve_quotas"],
        diversity=state,
    )

    # Overlap / leakage checks
    primary_ids = {str(r["review_item_id"]) for r in primary}
    reserve_ids = {str(r["review_item_id"]) for r in reserve}
    if primary_ids & reserve_ids:
        raise HoldoutDesignError("primary/reserve overlap")
    if primary_ids & reviewed_ids or reserve_ids & reviewed_ids:
        raise HoldoutDesignError("selected overlaps reviewed pilot")
    if primary_ids & discovery_sets["review_item_id"]:
        raise HoldoutDesignError("primary overlaps discovery")
    if reserve_ids & discovery_sets["review_item_id"]:
        raise HoldoutDesignError("reserve overlaps discovery")

    cut_info = derive_validation_candidate_cut(operating_points)
    mins = config["annotation_minimums"]
    decision_rules = {
        "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL": {
            "readable_positive_min": int(mins["readable_positive"]),
            "negative_or_safety_min": int(mins["negative_or_safety"]),
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_exact_min": 2,
        },
        "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT": {
            "readable_positive_min": int(mins["readable_positive"]),
            "negative_or_safety_min": int(mins["negative_or_safety"]),
            "accepted_wrong_positive": 0,
            "accepted_negative": 0,
            "accepted_exact_max_exclusive": 2,
        },
        "FAIL_INDEPENDENT_GATE_SAFETY": {
            "accepted_wrong_positive_gt": 0,
            "or_accepted_negative_gt": 0,
        },
        "BLOCKED_INSUFFICIENT_LABELED_HOLDOUT": {
            "readable_positive_lt": int(mins["readable_positive"]),
            "or_negative_or_safety_lt": int(mins["negative_or_safety"]),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in IMAGE_DIRS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    for item in primary + reserve:
        path = _resolve_input(project_root, str(item["source_crop_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise HoldoutDesignError(f"cannot decode selected crop: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != str(item["source_crop_sha256"]):
            raise HoldoutDesignError(f"crop sha mismatch: {path}")
        images[str(item["crop_id"])] = image

    max_sheet = int(selection["contact_sheet_max_items"])

    def _assign_visuals(batch_items: list[dict[str, Any]], batch: str) -> None:
        overview_dir = output_dir / f"{batch}_item_overviews"
        sheet_dir = output_dir / f"{batch}_contact_sheets"
        for item in batch_items:
            overview_name = f"{item['holdout_item_id']}.png"
            overview_path = overview_dir / overview_name
            write_png(
                overview_path,
                render_item_overview(item, images[str(item["crop_id"])]),
            )
            item["overview_path"] = str(overview_path.relative_to(project_root))
        for sheet_index, start in enumerate(range(0, len(batch_items), max_sheet), start=1):
            chunk = batch_items[start : start + max_sheet]
            sheet_name = f"{batch}_contact_sheet_{sheet_index:02d}.png"
            sheet_path = sheet_dir / sheet_name
            write_png(sheet_path, render_contact_sheet(chunk, images, max_items=max_sheet))
            rel = str(sheet_path.relative_to(project_root))
            for item in chunk:
                item["contact_sheet_path"] = rel

    _assign_visuals(primary, "primary")
    _assign_visuals(reserve, "reserve")

    def _public_selected(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SELECTED_SCHEMA,
            "holdout_item_id": row["holdout_item_id"],
            "batch": row["batch"],
            "batch_order": row["batch_order"],
            "review_item_id": row["review_item_id"],
            "stratum": row["stratum"],
            "stratum_is_ground_truth": False,
            "crop_id": row["crop_id"],
            "segment_id": row["segment_id"],
            "raw_track_id": row["raw_track_id"],
            "frame_index": row["frame_index"],
            "source_crop_path": row["source_crop_path"],
            "source_crop_sha256": row["source_crop_sha256"],
            "crop_width_px": row["crop_width_px"],
            "crop_height_px": row["crop_height_px"],
            "roi_x_min": row["roi_x_min"],
            "roi_y_min": row["roi_y_min"],
            "roi_x_max": row["roi_x_max"],
            "roi_y_max": row["roi_y_max"],
            "roi_width_px": row["roi_width_px"],
            "roi_height_px": row["roi_height_px"],
            "composite_score": row["composite_score"],
            "overview_path": row["overview_path"],
            "contact_sheet_path": row["contact_sheet_path"],
            "kit_family": row.get("kit_family"),
            **{field: row[field] for field in RANK_FIELDS},
        }

    primary_public = [_public_selected(r) for r in primary]
    reserve_public = [_public_selected(r) for r in reserve]

    write_jsonl(output_dir / "holdout_source_universe.jsonl", universe)
    exclusion_report = {
        "schema_version": "reid_jersey_parseq_holdout_exclusion_report_v1",
        "hard_exclusion": hard_report,
        "near_duplicate": nd_report,
        "independent_pool_count": len(pool),
        "discovery_input_count": len(discovery),
        "reviewed_pilot_count": len(reviewed_ids),
    }
    write_json(output_dir / "holdout_exclusion_report.json", exclusion_report)
    write_jsonl(output_dir / "holdout_independent_pool.jsonl", pool)
    write_jsonl(output_dir / "holdout_primary_manifest.jsonl", primary_public)
    write_jsonl(output_dir / "holdout_reserve_manifest.jsonl", reserve_public)

    template_rows = build_annotation_template_rows(primary_public, reserve_public)
    write_annotation_template(output_dir / "holdout_annotation_template.csv", template_rows)
    (output_dir / "holdout_review_instructions.md").write_text(
        build_review_instructions_tr(), encoding="utf-8"
    )

    prereg = {
        "schema_version": PREREG_SCHEMA,
        "stage": "5C-C3F-A",
        "discovery_source": {
            "path": inputs["discovery_inference_manifest"]["path"],
            "sha256": inputs["discovery_inference_manifest"]["sha256"],
            "rows": 46,
        },
        "c3e_operating_points_source": {
            "path": inputs["c3e_operating_points"]["path"],
            "sha256": inputs["c3e_operating_points"]["sha256"],
            "rows": 48,
        },
        **cut_info,
        "annotation_minimums": dict(mins),
        "decision_rules": decision_rules,
        "deployment_threshold_selected": False,
        "holdout_predictions_seen": False,
        "holdout_labels_seen_at_preregistration": False,
        "legibility_model_used": False,
        "next_evaluation_gate": config["preregistration"]["next_evaluation_gate"],
        "next_annotation_gate": config["preregistration"]["next_annotation_gate"],
    }
    write_json(output_dir / "holdout_preregistration.json", prereg)

    primary_by_stratum = {
        name: sum(1 for r in primary if r["stratum"] == name) for name in STRATA
    }
    reserve_by_stratum = {
        name: sum(1 for r in reserve if r["stratum"] == name) for name in STRATA
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "holdout_review_package_generated_unreviewed",
        "universe_counts": {
            "canonical": 474,
            "reviewed_pilot": 78,
            "unreviewed": 396,
            "discovery": 46,
        },
        "exclusion": exclusion_report,
        "independent_pool_count": len(pool),
        "stratum_pool_counts": stratum_counts,
        "primary_counts": {"total": len(primary), "by_stratum": primary_by_stratum},
        "reserve_counts": {"total": len(reserve), "by_stratum": reserve_by_stratum},
        "diversity_metrics": {
            "unique_segments_selected": len(
                {str(r["segment_id"]) for r in primary + reserve}
            ),
            "unique_raw_tracks_selected": len(
                {int(r["raw_track_id"]) for r in primary + reserve}
            ),
            "max_per_segment": int(diversity_cfg["max_per_segment"]),
            "max_per_raw_track": int(diversity_cfg["max_per_raw_track"]),
            "min_frame_gap_same_raw_track": int(
                diversity_cfg["min_frame_gap_same_raw_track"]
            ),
        },
        "overlap_checks": {
            "primary_reserve_overlap": 0,
            "selected_reviewed_overlap": 0,
            "selected_discovery_overlap": 0,
        },
        "review_status": "unreviewed",
        "preregistration_status": "validation_candidate_cut_registered",
        "validation_candidate_cut": cut_info["validation_candidate_cut"],
        "interpretation_limits": [
            "strata are not ground truth",
            "validation_candidate_cut is not a deployment threshold",
            "cut derived only from C3E discovery operating points",
            "holdout labels and predictions not seen at design time",
        ],
        "next_gate": config["preregistration"]["next_annotation_gate"],
        "safety_flags": {
            "model_initialized": False,
            "checkpoint_loaded": False,
            "inference_performed": False,
            "prediction_accessed_for_selection": False,
            "confidence_accessed_for_selection": False,
            "threshold_selected_for_deployment": False,
            "manual_labels_generated": False,
            "dataset_downloaded": False,
            "identity_assigned": False,
        },
    }
    write_json(output_dir / "holdout_design_summary.json", summary)

    # Manifest written last; does not include its own sha.
    artifacts: dict[str, Any] = {}
    for name in FINAL_TOP_LEVEL:
        if name == "holdout_design_manifest.json":
            continue
        artifacts[name] = _artifact_meta(output_dir / name)
    image_artifacts: dict[str, list[dict[str, Any]]] = {}
    for dirname in IMAGE_DIRS:
        image_artifacts[dirname] = [
            _artifact_meta(path) for path in sorted((output_dir / dirname).glob("*.png"))
        ]

    feature_contract = [
        {
            "field": field,
            "source_artifact": inputs["canonical_review_items"]["path"],
            "semantic_purpose": "model_independent_visibility_or_quality_rank",
            "direction": "lower_rank_better",
            "normalization": "invert_one_based_rank_to_unit",
            "ranking_weight": 1.0 / len(RANK_FIELDS),
        }
        for field in RANK_FIELDS
    ]
    git_after = project_git_status()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        "project_git_head": git_head,
        "source_artifacts": {
            key: {
                "path": inputs[key]["path"],
                "sha256": inputs[key]["sha256"],
                "expected_rows": inputs[key]["expected_rows"],
            }
            for key in inputs
        },
        "selection_config": {
            "combined_quotas": dict(selection["combined_quotas"]),
            "primary_quotas": dict(selection["primary_quotas"]),
            "reserve_quotas": dict(selection["reserve_quotas"]),
            "diversity": dict(diversity_cfg),
            "near_duplicate": dict(config["near_duplicate"]),
        },
        "used_feature_contract": feature_contract,
        "excluded_field_contract": {
            "prohibited_prefixes": list(PROHIBITED_SELECTION_PREFIXES),
            "prohibited_substrings": list(PROHIBITED_SELECTION_SUBSTRINGS),
            "manual_labels_not_used_for_selection": True,
            "parseq_predictions_not_used_for_selection": True,
            "parseq_confidence_not_used_for_selection": True,
        },
        "artifacts": artifacts,
        "image_artifacts": image_artifacts,
        "git_status_before": git_before,
        "git_status_after": git_after,
        "source_immutability": True,
        "no_model_no_network": True,
        "network_policy": config["output"]["network_policy"],
        "atomic_finalization": True,
        "temp_cleanup": True,
        "safety_flags": summary["safety_flags"],
    }
    write_json(output_dir / "holdout_design_manifest.json", manifest)

    top_level = sorted(p.name for p in output_dir.iterdir())
    expected_top = sorted(list(FINAL_TOP_LEVEL) + list(IMAGE_DIRS))
    if top_level != expected_top:
        raise HoldoutDesignError(
            f"unexpected top-level artifacts: {top_level} != {expected_top}"
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            _PROJECT_ROOT / "configs/reid/jersey_parseq_holdout_stage5c_c3f_a.yaml"
        ),
    )
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    config = load_config(args.config)
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise HoldoutDesignError(f"final output exists (no overwrite): {final_dir}")

    # Capture git BEFORE mutating HOME / proxy env.
    git_before = project_git_status()
    git_head = project_git_head()

    unique = f"{int(time.time())}_{os.getpid()}"
    temp_parent = project_root / "outputs" / "reid" / "full_stage4b"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_parent / f"_tmp_jersey_parseq_holdout_design_{unique}"
    work_home = Path(tempfile.mkdtemp(prefix="c3f_a_home_"))
    original_env = {
        key: os.environ.get(key)
        for key in (
            "HOME",
            "XDG_CACHE_HOME",
            "TORCH_HOME",
            "TMPDIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        )
    }
    try:
        os.environ.update(
            {
                "HOME": str(work_home),
                "XDG_CACHE_HOME": str(work_home / "cache"),
                "TORCH_HOME": str(work_home / "torch"),
                "TMPDIR": str(work_home / "tmp"),
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
            }
        )
        (work_home / "cache").mkdir()
        (work_home / "torch").mkdir()
        (work_home / "tmp").mkdir()

        summary = run_holdout_design(
            config,
            project_root=project_root,
            output_dir=temp_dir,
            git_before=git_before,
            git_head=git_head,
        )
        os.replace(temp_dir, final_dir)
        print(
            "C3F_A_OK "
            f"status={summary['status']} "
            f"pool={summary['independent_pool_count']} "
            f"primary={summary['primary_counts']['total']} "
            f"reserve={summary['reserve_counts']['total']} "
            f"cut={summary['validation_candidate_cut']} "
            f"output={final_dir}",
            flush=True,
        )
        return 0
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(work_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
