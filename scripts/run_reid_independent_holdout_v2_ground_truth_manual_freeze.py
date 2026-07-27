#!/usr/bin/env python3
"""Stage 5D-F3L — Freeze human holdout v2 ground-truth decisions for target_001.

Freezes exact human decisions for H2_GT_REVIEW_000001..000141 from the F3K
review package. Does not load gallery/OSNet embeddings, compute
similarity/rank/metrics, select thresholds, assign identities, or mutate
the gallery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_independent_holdout_v2_ground_truth_manual_freeze_config_v1"
STATUS = (
    "COMPLETED_STAGE5D_F3L_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_FROZEN"
)
READINESS = "TARGET_001_INDEPENDENT_HOLDOUT_V2_READY_FOR_FROZEN_QUERY_EMBEDDING_AND_SCORING"
NEXT_GATE = (
    "STAGE5D-F3M_TARGET_001_NEW_INDEPENDENT_HOLDOUT_OSNET_QUERY_EMBEDDING_AND_"
    "FROZEN_TARGET_DISTRACTOR_SCORING"
)
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
COMPONENT_POLICY = "ONE_FROZEN_SEGMENT_PER_COMPONENT_NO_CROSS_TRACK_LINK_EVIDENCE"
REVIEWED_N = 141
INELIGIBLE_N = 102
UNIVERSE_N = 243

ALLOWED_DIRTY = {
    "scripts/run_reid_independent_holdout_v2_ground_truth_manual_freeze.py",
    "configs/reid/independent_holdout_v2_ground_truth_manual_freeze_stage5d_target_001.yaml",
    "tests/test_reid_independent_holdout_v2_ground_truth_manual_freeze.py",
    "docs/setup/stage5d-target-independent-holdout-ground-truth-manual-review-and-freeze.md",
}

_POSITIVE_SUFFIXES = (
    10,
    30,
    61,
    90,
    94,
    104,
    124,
    129,
    135,
    136,
)
_SAME_TEAM_NEGATIVE_SUFFIXES = (
    3,
    5,
    7,
    8,
    13,
    14,
    15,
    16,
    23,
    24,
    27,
    28,
    32,
    34,
    37,
    38,
    40,
    42,
    43,
    46,
    49,
    50,
    51,
    52,
    53,
    54,
    57,
    58,
    59,
    60,
    63,
    64,
    65,
    68,
    69,
    72,
    73,
    75,
    77,
    79,
    83,
    84,
    87,
    89,
    99,
    100,
    111,
    113,
    115,
    117,
    119,
    120,
    131,
    139,
    140,
)
_OTHER_TEAM_NEGATIVE_SUFFIXES = (
    1,
    2,
    6,
    9,
    11,
    12,
    17,
    18,
    19,
    20,
    25,
    26,
    29,
    31,
    33,
    36,
    39,
    41,
    45,
    47,
    55,
    56,
    71,
    78,
    85,
    86,
    88,
    92,
    93,
    95,
    96,
    97,
    101,
    102,
    103,
    105,
    106,
    107,
    108,
    109,
    110,
    112,
    118,
    121,
    122,
    125,
    126,
    134,
    138,
    141,
)
_NON_PLAYER_SUFFIXES = (4, 21, 22, 66, 98)
_INVALID_SUFFIXES = (62, 67, 91, 114, 116, 133, 137)
_AMBIGUOUS_SUFFIXES = (
    35,
    44,
    48,
    70,
    74,
    76,
    80,
    81,
    82,
    123,
    127,
    128,
    130,
    132,
)


def _review_id(suffix: int) -> str:
    return f"H2_GT_REVIEW_{suffix:06d}"


POSITIVE_IDS = tuple(_review_id(s) for s in _POSITIVE_SUFFIXES)
SAME_TEAM_NEGATIVE_IDS = tuple(_review_id(s) for s in _SAME_TEAM_NEGATIVE_SUFFIXES)
OTHER_TEAM_NEGATIVE_IDS = tuple(_review_id(s) for s in _OTHER_TEAM_NEGATIVE_SUFFIXES)
NON_PLAYER_IDS = tuple(_review_id(s) for s in _NON_PLAYER_SUFFIXES)
INVALID_IDS = tuple(_review_id(s) for s in _INVALID_SUFFIXES)
AMBIGUOUS_IDS = tuple(_review_id(s) for s in _AMBIGUOUS_SUFFIXES)

DECISION_CSV_FIELDS = (
    "review_item_id",
    "target_id",
    "segment_id",
    "raw_track_code",
    "tracker_native_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "observation_count",
    "primary_representative_frame",
    "representative_crop_path",
    "representative_crop_sha256",
    "ground_truth_component_id",
    "manual_ground_truth_decision",
    "manual_target_present",
    "manual_same_target_as_target_001",
    "manual_same_team_as_target",
    "manual_visible_jersey_number",
    "jersey_number_provenance",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_track_impurity_observed",
    "clean_positive",
    "clean_negative",
    "clean_same_team_negative",
    "metric_inclusion",
    "gallery_member",
    "enrollment_eligible",
    "gallery_growth_eligible",
    "query_score_eligibility",
    "exclusion_from_reid_query_reason",
    "automatic_negative",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "review_basis",
    "assistant_quality_triage_used",
    "automated_identity_used",
    "automated_team_classifier_used",
    "automated_OCR_used",
    "similarity_used",
    "ranking_used",
    "gallery_match_used",
)


class GroundTruthFreezeError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listing_sha(root: Path) -> tuple[int, str]:
    files: list[tuple[str, int, str]] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            path = Path(dp) / fn
            rel = str(path.relative_to(root)).replace("\\", "/")
            files.append((rel, path.stat().st_size, sha256_file(path)))
    files.sort()
    blob = "\n".join(f"{a}\t{b}\t{c}" for a, b, c in files).encode()
    return len(files), hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GroundTruthFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise GroundTruthFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GroundTruthFreezeError(f"path traversal rejected: {rel}")


def segment_numeric(segment_id: str) -> int:
    return int(str(segment_id).rsplit("_", 1)[-1])


def review_suffix(review_item_id: str) -> int:
    return int(str(review_item_id).rsplit("_", 1)[-1])


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GroundTruthFreezeError("BLOCKED_STAGE5D_F3L_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GroundTruthFreezeError("BLOCKED_STAGE5D_F3L_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GroundTruthFreezeError(
                    "BLOCKED_STAGE5D_F3L_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    if not sidecar.is_file() or not manifest.is_file() or not listing.is_file():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    if not listing.read_text(encoding="utf-8").strip():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH listing"
        )
    return actual


def validate_decision_sets() -> None:
    all_ids = {f"H2_GT_REVIEW_{i:06d}" for i in range(1, REVIEWED_N + 1)}
    groups = {
        "positive": set(POSITIVE_IDS),
        "same_team_negative": set(SAME_TEAM_NEGATIVE_IDS),
        "other_team_negative": set(OTHER_TEAM_NEGATIVE_IDS),
        "non_player": set(NON_PLAYER_IDS),
        "invalid": set(INVALID_IDS),
        "ambiguous": set(AMBIGUOUS_IDS),
    }
    expected_lens = {
        "positive": 10,
        "same_team_negative": 55,
        "other_team_negative": 50,
        "non_player": 5,
        "invalid": 7,
        "ambiguous": 14,
    }
    for name, ids in groups.items():
        if len(ids) != expected_lens[name]:
            raise GroundTruthFreezeError(f"decision set size {name}")
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = groups[a] & groups[b]
            if inter:
                raise GroundTruthFreezeError(f"overlap {a}/{b}: {sorted(inter)}")
    union = set().union(*groups.values())
    if union != all_ids:
        missing = sorted(all_ids - union)
        extra = sorted(union - all_ids)
        raise GroundTruthFreezeError(f"coverage missing={missing} extra={extra}")


def _human_review_base(config: Mapping[str, Any], reviewed_at: str) -> dict[str, Any]:
    hr = config["human_review"]
    return {
        "reviewer": hr["reviewer"],
        "final_approver": hr["final_approver"],
        "reviewed_at": reviewed_at,
        "review_basis": hr["review_basis"],
        "assistant_quality_triage_used": True,
        "automated_identity_used": False,
        "automated_team_classifier_used": False,
        "automated_OCR_used": False,
        "similarity_used": False,
        "ranking_used": False,
        "gallery_match_used": False,
    }


def build_decision_for_review_id(review_item_id: str) -> dict[str, Any]:
    suffix = review_suffix(review_item_id)
    rid = _review_id(suffix)
    if rid in POSITIVE_IDS:
        return {
            "manual_ground_truth_decision": "target_occurrence_yes",
            "manual_target_present": "yes",
            "manual_same_target_as_target_001": "yes",
            "manual_same_team_as_target": "yes",
            "manual_visible_jersey_number": "5",
            "jersey_number_provenance": "human_visual_review_by_Furkan",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "yes",
            "manual_single_person": "yes",
            "manual_identity_continuity_observed": "yes",
            "manual_track_impurity_observed": "no",
            "clean_positive": True,
            "clean_negative": False,
            "clean_same_team_negative": False,
            "metric_inclusion": True,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": True,
            "exclusion_from_reid_query_reason": "",
            "automatic_negative": False,
        }
    if rid in SAME_TEAM_NEGATIVE_IDS:
        return {
            "manual_ground_truth_decision": "target_occurrence_no",
            "manual_target_present": "no",
            "manual_same_target_as_target_001": "no",
            "manual_same_team_as_target": "yes",
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "manual_identity_continuity_observed": "yes",
            "manual_track_impurity_observed": "no",
            "clean_positive": False,
            "clean_negative": True,
            "clean_same_team_negative": True,
            "metric_inclusion": True,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": True,
            "exclusion_from_reid_query_reason": "",
            "automatic_negative": False,
        }
    if rid in OTHER_TEAM_NEGATIVE_IDS:
        return {
            "manual_ground_truth_decision": "target_occurrence_no",
            "manual_target_present": "no",
            "manual_same_target_as_target_001": "no",
            "manual_same_team_as_target": "no",
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "manual_identity_continuity_observed": "yes",
            "manual_track_impurity_observed": "no",
            "clean_positive": False,
            "clean_negative": True,
            "clean_same_team_negative": False,
            "metric_inclusion": True,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": True,
            "exclusion_from_reid_query_reason": "",
            "automatic_negative": False,
        }
    if rid in NON_PLAYER_IDS:
        return {
            "manual_ground_truth_decision": "non_player",
            "manual_target_present": "no",
            "manual_same_target_as_target_001": "no",
            "manual_same_team_as_target": "uncertain",
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "manual_identity_continuity_observed": "yes",
            "manual_track_impurity_observed": "no",
            "clean_positive": False,
            "clean_negative": True,
            "clean_same_team_negative": False,
            "metric_inclusion": True,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": False,
            "exclusion_from_reid_query_reason": "human_reviewed_non_player",
            "automatic_negative": False,
        }
    if rid in INVALID_IDS:
        return {
            "manual_ground_truth_decision": "invalid",
            "manual_target_present": "uncertain",
            "manual_same_target_as_target_001": "uncertain",
            "manual_same_team_as_target": "uncertain",
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_crop_valid": "no",
            "manual_target_dominant": "uncertain",
            "manual_single_person": "uncertain",
            "manual_identity_continuity_observed": "uncertain",
            "manual_track_impurity_observed": "uncertain",
            "clean_positive": False,
            "clean_negative": False,
            "clean_same_team_negative": False,
            "metric_inclusion": False,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": False,
            "exclusion_from_reid_query_reason": "",
            "automatic_negative": False,
        }
    if rid in AMBIGUOUS_IDS:
        return {
            "manual_ground_truth_decision": "multi_person_ambiguous",
            "manual_target_present": "uncertain",
            "manual_same_target_as_target_001": "uncertain",
            "manual_same_team_as_target": "uncertain",
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_crop_valid": "uncertain",
            "manual_target_dominant": "uncertain",
            "manual_single_person": "no",
            "manual_identity_continuity_observed": "uncertain",
            "manual_track_impurity_observed": "yes",
            "clean_positive": False,
            "clean_negative": False,
            "clean_same_team_negative": False,
            "metric_inclusion": False,
            "gallery_member": False,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "query_score_eligibility": False,
            "exclusion_from_reid_query_reason": "",
            "automatic_negative": False,
        }
    raise GroundTruthFreezeError(f"unknown review_item_id {review_item_id}")


def reviewed_component_id(review_item_id: str) -> str:
    return f"H2_GT_COMPONENT_{review_suffix(review_item_id):06d}"


def ineligible_component_id(segment_id: str) -> str:
    return f"H2_GT_COMPONENT_INELIG_{segment_numeric(segment_id):06d}"


def build_access_audit() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3l_access_audit_v1",
        "sample_video_read": False,
        "external_video_read": False,
        "holdout_video_read": False,
        "target_gallery_read_count": 0,
        "distractor_gallery_read_count": 0,
        "gallery_embedding_read_count": 0,
        "embedding_read_count": 0,
        "similarity_row_read_count": 0,
        "rank_row_read_count": 0,
        "score_row_read_count": 0,
        "metric_computation_count": 0,
        "detection_inference_passes": 0,
        "tracking_rerun_count": 0,
        "segmentation_rerun_count": 0,
        "ocr_calls": 0,
        "team_classifier_calls": 0,
        "identity_inference_calls": 0,
        "osnet_model_loads": 0,
        "yolo_model_loads": 0,
        "gallery_mutations": 0,
        "threshold_candidates": 0,
        "identity_assignments": 0,
        "crop_png_bytes_read": 0,
        "contact_sheet_bytes_read": 0,
        "video_bytes_read": 0,
    }


def validate_f3k(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    exp = config["stage5d_f3k_package"]
    root = project_root / exp["path"]
    if not root.is_dir():
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH missing"
        )
    summary = load_json(root / "stage5d_f3k_summary.json")
    if summary.get("final_status") != exp["expected_final_status"]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("readiness") != exp["expected_readiness"]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH readiness"
        )
    checks = {
        "complete_universe": exp["expected_complete_universe"],
        "review_eligible": exp["expected_review_eligible"],
        "review_ineligible": exp["expected_review_ineligible"],
        "review_item_count": exp["expected_review_items"],
        "representative_crop_count": exp["expected_crops"],
        "contact_sheets": exp["expected_contact_sheets"],
        "review_videos": exp["expected_videos"],
        "manual_ground_truth_decisions": exp["expected_manual_decisions"],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GroundTruthFreezeError(
                f"BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("sheet_distribution") != exp["expected_sheet_distribution"]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH sheet_distribution"
        )
    if summary.get("video_distribution") != exp["expected_video_distribution"]:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH video_distribution"
        )
    snapshot_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    inventory = load_jsonl(
        root / "inventory" / "target_001_holdout_v2_gt_review_item_inventory.jsonl"
    )
    if len(inventory) != REVIEWED_N:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH inventory"
        )
    inv_ids = [r["review_item_id"] for r in inventory]
    expected_ids = [f"H2_GT_REVIEW_{i:06d}" for i in range(1, REVIEWED_N + 1)]
    if inv_ids != expected_ids:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH inventory_order"
        )
    if len(set(inv_ids)) != REVIEWED_N:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH inventory_dup"
        )
    mapping = load_json(
        root / "inventory" / "target_001_holdout_v2_gt_review_item_mapping.json"
    )
    mapping_items = mapping.get("items") or []
    if len(mapping_items) != REVIEWED_N:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH mapping"
        )
    tpl = root / "templates" / "target_001_holdout_v2_ground_truth_review_template.csv"
    with tpl.open(encoding="utf-8", newline="") as handle:
        blank_rows = list(csv.DictReader(handle))
    if len(blank_rows) != REVIEWED_N:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH template_rows"
        )
    tpl_ids = [r["review_item_id"] for r in blank_rows]
    if tpl_ids != expected_ids:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH template_ids"
        )
    if any(r.get("manual_ground_truth_decision") for r in blank_rows):
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH template_filled"
        )
    ineligible = load_jsonl(
        root / "exclusions" / "target_001_holdout_v2_review_ineligible_segment_inventory.jsonl"
    )
    if len(ineligible) != INELIGIBLE_N:
        raise GroundTruthFreezeError(
            "BLOCKED_STAGE5D_F3L_REVIEW_PACKAGE_CONTRACT_MISMATCH ineligible"
        )
    return {
        "root": root,
        "summary": summary,
        "inventory": inventory,
        "mapping_items": mapping_items,
        "ineligible": ineligible,
        "snapshot_sha256": snapshot_sha,
        "package_sha256": sha256_file(root / "stage5d_f3k_manifest.json"),
    }


def validate_f3j(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_f3j_package"]
    root = project_root / block["path"]
    if not root.is_dir():
        raise GroundTruthFreezeError("F3J package missing")
    summary = load_json(root / "stage5d_f3j_summary.json")
    if summary.get("final_status") != block["expected_final_status"]:
        raise GroundTruthFreezeError("F3J status mismatch")
    checks = {
        "segment_count": block["expected_segments"],
        "raw_track_count": block["expected_raw_tracks"],
        "pass_through_segment_count": block["expected_pass_through"],
        "split_segment_count": block["expected_split"],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GroundTruthFreezeError(f"F3J {key} mismatch")
    return {"root": root, "summary": summary}


def validate_f3h_metric_metadata(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_f3h_package"]
    metric_path = project_root / block["path"] / block["metric_contract_relpath"]
    gt_path = project_root / block["path"] / block["ground_truth_policy_relpath"]
    if not metric_path.is_file() or not gt_path.is_file():
        raise GroundTruthFreezeError("F3H metadata missing")
    metric = load_json(metric_path)
    minimum_support = metric.get("minimum_support") or {}
    required = {
        "segment_clean_positive_ge": 5,
        "segment_clean_negative_ge": 20,
        "segment_clean_same_team_negative_ge": 10,
        "component_clean_positive_ge": 3,
        "component_clean_negative_ge": 10,
    }
    for key, expected in required.items():
        if int(minimum_support.get(key) or 0) != expected:
            raise GroundTruthFreezeError(f"F3H {key} mismatch")
    return {
        "metric_contract_sha256": sha256_file(metric_path),
        "ground_truth_policy_sha256": sha256_file(gt_path),
        "minimum_support": {k: int(minimum_support[k]) for k in required},
    }


def build_component_mapping(
    frozen_rows: Sequence[Mapping[str, Any]],
    ineligible_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for row in frozen_rows:
        components.append(
            {
                "ground_truth_component_id": row["ground_truth_component_id"],
                "segment_id": row["segment_id"],
                "raw_track_code": row["raw_track_code"],
                "review_item_id": row["review_item_id"],
                "component_policy": COMPONENT_POLICY,
                "segment_count": 1,
                "cross_track_link_evidence": False,
                "review_eligible": True,
                "ground_truth_decision": row["manual_ground_truth_decision"],
                "metric_inclusion": bool(row["metric_inclusion"]),
                "query_score_eligibility": bool(row["query_score_eligibility"]),
            }
        )
    for row in ineligible_rows:
        sid = str(row["segment_id"])
        components.append(
            {
                "ground_truth_component_id": ineligible_component_id(sid),
                "segment_id": sid,
                "raw_track_code": row["raw_track_code"],
                "review_item_id": None,
                "component_policy": COMPONENT_POLICY,
                "segment_count": 1,
                "cross_track_link_evidence": False,
                "review_eligible": False,
                "ground_truth_decision": "unreviewed_ineligible",
                "metric_inclusion": False,
                "query_score_eligibility": False,
            }
        )
    comp_ids = [c["ground_truth_component_id"] for c in components]
    if len(comp_ids) != len(set(comp_ids)):
        raise GroundTruthFreezeError("duplicate component ids")
    if len(components) != UNIVERSE_N:
        raise GroundTruthFreezeError("component count mismatch")
    seg_ids = [c["segment_id"] for c in components]
    if len(seg_ids) != len(set(seg_ids)):
        raise GroundTruthFreezeError("duplicate segment in components")
    summary = {
        "schema_version": "reid_stage5d_f3l_ground_truth_component_summary_v1",
        "component_policy": COMPONENT_POLICY,
        "total_components": UNIVERSE_N,
        "reviewed_components": REVIEWED_N,
        "ineligible_components": INELIGIBLE_N,
        "clean_positive_components": 10,
        "clean_negative_components": 110,
        "clean_same_team_negative_components": 55,
        "metric_excluded_reviewed_components": 21,
        "conflict_count": 0,
        "cross_track_merge_count": 0,
        "one_segment_per_component": True,
        "documented_cross_track_identity_links": 0,
        "component_score_aggregation_note": (
            "one_segment_per_component_so_max_equals_median"
        ),
    }
    return components, summary


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3l_gt_freeze_{final_dir.name}_{token}"
    if tmp.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F3L_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F3L_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise GroundTruthFreezeError("FAILED_STAGE5D_F3L_ATOMIC_OUTPUT final_exists")

    validate_decision_sets()
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )

    f3k = validate_f3k(project_root, config)
    f3j = validate_f3j(project_root, config)
    f3h = validate_f3h_metric_metadata(project_root, config)

    f3i_cfg = config.get("stage5d_f3i_package")
    if f3i_cfg:
        f3i_root = project_root / f3i_cfg["path"]
        if f3i_root.is_dir():
            f3i_summary = load_json(f3i_root / "stage5d_f3i_summary.json")
            if f3i_summary.get("final_status") != f3i_cfg["expected_final_status"]:
                raise GroundTruthFreezeError("F3I status mismatch")

    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    human_base = _human_review_base(config, reviewed_at)

    inv_by_id = {r["review_item_id"]: r for r in f3k["inventory"]}
    map_by_id = {r["review_item_id"]: r for r in f3k["mapping_items"]}

    frozen_rows: list[dict[str, Any]] = []
    for i in range(1, REVIEWED_N + 1):
        review_item_id = f"H2_GT_REVIEW_{i:06d}"
        inv = inv_by_id[review_item_id]
        mp = map_by_id[review_item_id]
        crop_rel = str(mp["representative_crop_path"])
        assert_no_path_traversal(crop_rel)
        crop_path = f3k["root"] / crop_rel
        if not crop_path.is_file():
            raise GroundTruthFreezeError(f"missing crop path {crop_rel}")
        crop_path.stat()  # existence/size only; no PNG byte read

        decision = build_decision_for_review_id(review_item_id)
        frozen_rows.append(
            {
                "review_item_id": review_item_id,
                "target_id": TARGET_ID,
                "segment_id": inv["segment_id"],
                "raw_track_code": inv["raw_track_code"],
                "tracker_native_id": int(inv["tracker_native_id"]),
                "start_frame": int(inv["start_frame"]),
                "end_frame": int(inv["end_frame"]),
                "start_time": inv["start_time"],
                "end_time": inv["end_time"],
                "observation_count": int(inv["observation_count"]),
                "primary_representative_frame": int(mp["representative_frame"]),
                "representative_crop_path": crop_rel,
                "representative_crop_sha256": mp["representative_crop_sha256"],
                "ground_truth_component_id": reviewed_component_id(review_item_id),
                **decision,
                **human_base,
            }
        )

    occ = Counter(r["manual_ground_truth_decision"] for r in frozen_rows)
    exp = config["expected_distribution"]
    expected_occ = {
        "target_occurrence_yes": exp["target_occurrence_yes"],
        "target_occurrence_no": exp["target_occurrence_no"],
        "non_player": exp["non_player"],
        "invalid": exp["invalid"],
        "multi_person_ambiguous": exp["multi_person_ambiguous"],
        "uncertain": exp["uncertain"],
    }
    for key, count in expected_occ.items():
        if int(occ.get(key, 0)) != int(count):
            raise GroundTruthFreezeError(f"distribution mismatch {key}")

    clean_pos = [r for r in frozen_rows if r["clean_positive"]]
    clean_neg = [r for r in frozen_rows if r["clean_negative"]]
    clean_stn = [r for r in frozen_rows if r["clean_same_team_negative"]]
    metric_excl = [r for r in frozen_rows if not r["metric_inclusion"]]
    query_eligible = [r for r in frozen_rows if r["query_score_eligibility"]]

    if len(clean_pos) != exp["clean_positive_segments"]:
        raise GroundTruthFreezeError("clean positive count")
    if len(clean_neg) != exp["clean_negative_segments"]:
        raise GroundTruthFreezeError("clean negative count")
    if len(clean_stn) != exp["clean_same_team_negative_segments"]:
        raise GroundTruthFreezeError("clean same-team negative count")
    if len(metric_excl) != exp["reviewed_metric_excluded"]:
        raise GroundTruthFreezeError("metric exclusion count")

    pos_jerseys = {
        r["manual_visible_jersey_number"]
        for r in clean_pos
        if r["manual_ground_truth_decision"] == "target_occurrence_yes"
    }
    if pos_jerseys != {"5"}:
        raise GroundTruthFreezeError("positive jersey policy")

    components, component_summary = build_component_mapping(
        frozen_rows, f3k["ineligible"]
    )

    coverage = {
        "schema_version": "reid_stage5d_f3l_complete_segment_ground_truth_coverage_v1",
        "reviewed_eligible": REVIEWED_N,
        "ineligible": INELIGIBLE_N,
        "complete": UNIVERSE_N,
        "silent_drop": 0,
        "duplicate": 0,
    }

    access_audit = build_access_audit()

    tmp = create_temp_root(final_dir)
    try:
        freeze_dir = tmp / "manual_freeze"
        comp_dir = tmp / "components"
        val_dir = tmp / "validation"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (freeze_dir, comp_dir, val_dir, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / Path(config_path).name)

        decisions_csv = (
            freeze_dir / "target_001_holdout_v2_ground_truth_decisions_frozen.csv"
        )
        with decisions_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DECISION_CSV_FIELDS))
            writer.writeheader()
            for row in frozen_rows:
                out = {k: row.get(k, "") for k in DECISION_CSV_FIELDS}
                for bool_key in (
                    "clean_positive",
                    "clean_negative",
                    "clean_same_team_negative",
                    "metric_inclusion",
                    "gallery_member",
                    "enrollment_eligible",
                    "gallery_growth_eligible",
                    "query_score_eligibility",
                    "assistant_quality_triage_used",
                    "automated_identity_used",
                    "automated_team_classifier_used",
                    "automated_OCR_used",
                    "similarity_used",
                    "ranking_used",
                    "gallery_match_used",
                    "automatic_negative",
                ):
                    out[bool_key] = bool(row[bool_key])
                writer.writerow(out)

        freeze_payload = {
            "schema_version": "reid_independent_holdout_ground_truth_freeze_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "readiness": READINESS,
            "f3k_package_manifest_sha256": f3k["package_sha256"],
            "f3k_snapshot_sha256": f3k["snapshot_sha256"],
            "f3j_manifest_path": config["stage5d_f3j_package"]["path"],
            "f3h_metric_contract_sha256": f3h["metric_contract_sha256"],
            "f3h_ground_truth_policy_sha256": f3h["ground_truth_policy_sha256"],
            "decision_lists": {
                "target_occurrence_yes": list(POSITIVE_IDS),
                "same_team_target_occurrence_no": list(SAME_TEAM_NEGATIVE_IDS),
                "other_team_target_occurrence_no": list(OTHER_TEAM_NEGATIVE_IDS),
                "non_player": list(NON_PLAYER_IDS),
                "invalid": list(INVALID_IDS),
                "multi_person_ambiguous": list(AMBIGUOUS_IDS),
            },
            "decision_distribution": {
                "reviewed_total": REVIEWED_N,
                "target_occurrence_yes": exp["target_occurrence_yes"],
                "target_occurrence_no": exp["target_occurrence_no"],
                "same_team_target_occurrence_no": exp["same_team_target_occurrence_no"],
                "other_team_target_occurrence_no": exp["other_team_target_occurrence_no"],
                "non_player": exp["non_player"],
                "invalid": exp["invalid"],
                "multi_person_ambiguous": exp["multi_person_ambiguous"],
                "uncertain": exp["uncertain"],
            },
            "metric_distribution": {
                "clean_positive_segments": exp["clean_positive_segments"],
                "clean_negative_segments": exp["clean_negative_segments"],
                "clean_same_team_negative_segments": exp["clean_same_team_negative_segments"],
                "reviewed_metric_excluded": exp["reviewed_metric_excluded"],
                "query_score_eligible": len(query_eligible),
            },
            "component_policy": COMPONENT_POLICY,
            "component_summary": component_summary,
            "coverage": coverage,
            "human_reviewer": config["human_review"]["reviewer"],
            "final_approver": config["human_review"]["final_approver"],
            "reviewed_at": reviewed_at,
            "manual_decisions_frozen": True,
            "similarity_observed_before_freeze": False,
            "gallery_vectors_read": False,
            "holdout_embedding_vectors_read": False,
            "threshold_selected": False,
            "gallery_mutation": False,
            "identity_assignments": 0,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            freeze_dir / "target_001_holdout_v2_ground_truth_decisions_frozen.json",
            freeze_payload,
        )

        write_jsonl(
            freeze_dir / "target_001_holdout_v2_clean_positive_inventory.jsonl",
            [r for r in frozen_rows if r["clean_positive"]],
        )
        write_jsonl(
            freeze_dir / "target_001_holdout_v2_clean_negative_inventory.jsonl",
            [r for r in frozen_rows if r["clean_negative"]],
        )
        write_jsonl(
            freeze_dir / "target_001_holdout_v2_clean_same_team_negative_inventory.jsonl",
            [r for r in frozen_rows if r["clean_same_team_negative"]],
        )
        write_jsonl(
            freeze_dir / "target_001_holdout_v2_reviewed_metric_exclusion_inventory.jsonl",
            [r for r in frozen_rows if not r["metric_inclusion"]],
        )
        write_json(
            freeze_dir / "target_001_holdout_v2_complete_segment_ground_truth_coverage.json",
            coverage,
        )

        write_jsonl(
            comp_dir / "target_001_holdout_v2_ground_truth_component_mapping.jsonl",
            components,
        )
        write_json(
            comp_dir / "target_001_holdout_v2_ground_truth_component_summary.json",
            component_summary,
        )

        validation = {
            "schema_version": "reid_stage5d_f3l_ground_truth_freeze_validation_v1",
            "missing_review_ids": 0,
            "extra_review_ids": 0,
            "duplicate_review_ids": 0,
            "reviewed_total": REVIEWED_N,
            "decision_distribution": freeze_payload["decision_distribution"],
            "metric_distribution": freeze_payload["metric_distribution"],
            "positive_ids": list(POSITIVE_IDS),
            "same_team_negative_ids": list(SAME_TEAM_NEGATIVE_IDS),
            "other_team_negative_ids": list(OTHER_TEAM_NEGATIVE_IDS),
            "non_player_ids": list(NON_PLAYER_IDS),
            "invalid_ids": list(INVALID_IDS),
            "ambiguous_ids": list(AMBIGUOUS_IDS),
            "component_policy": COMPONENT_POLICY,
            "component_conflict_count": 0,
            "coverage": coverage,
            "access_audit": access_audit,
            "minimum_support_audit": {
                "segment_clean_positive_ge": 5,
                "segment_clean_positive_observed": 10,
                "segment_clean_positive_pass": True,
                "segment_clean_negative_ge": 20,
                "segment_clean_negative_observed": 110,
                "segment_clean_negative_pass": True,
                "segment_clean_same_team_negative_ge": 10,
                "segment_clean_same_team_negative_observed": 55,
                "segment_clean_same_team_negative_pass": True,
                "component_clean_positive_ge": 3,
                "component_clean_positive_observed": 10,
                "component_clean_positive_pass": True,
                "component_clean_negative_ge": 10,
                "component_clean_negative_observed": 110,
                "component_clean_negative_pass": True,
                "note": "ground_truth_count_support_only_not_scoring_success",
            },
            "f3h_minimum_support": f3h["minimum_support"],
            "all_checks_passed": True,
        }
        write_json(
            val_dir / "target_001_holdout_v2_ground_truth_freeze_validation.json",
            validation,
        )

        write_json(runtime / "target_001_f3l_access_audit.json", access_audit)

        contract = {
            "schema_version": "reid_stage5d_f3l_ground_truth_manual_freeze_contract_v1",
            "target_id": TARGET_ID,
            "readiness": READINESS,
            "reviewed_total": REVIEWED_N,
            "clean_positive_segments": exp["clean_positive_segments"],
            "clean_negative_segments": exp["clean_negative_segments"],
            "clean_same_team_negative_segments": exp["clean_same_team_negative_segments"],
            "reviewed_metric_excluded": exp["reviewed_metric_excluded"],
            "unreviewed_ineligible": exp["unreviewed_ineligible"],
            "complete_universe": UNIVERSE_N,
            "manual_decisions_frozen": True,
            "component_policy": COMPONENT_POLICY,
            "component_conflict_count": 0,
            "clean_positive_components": 10,
            "clean_negative_components": 110,
            "clean_same_team_negative_components": 55,
            "similarity_observed_before_freeze": False,
            "gallery_vectors_read": False,
            "holdout_embedding_vectors_read": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "holdout_enrollment_forbidden": True,
            "gallery_growth_forbidden": True,
            "threshold_calibration_forbidden": True,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(tmp / "stage5d_f3l_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f3l_ground_truth_manual_freeze_summary_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reviewed_total": REVIEWED_N,
            "target_occurrence_yes": exp["target_occurrence_yes"],
            "target_occurrence_no": exp["target_occurrence_no"],
            "same_team_target_occurrence_no": exp["same_team_target_occurrence_no"],
            "other_team_target_occurrence_no": exp["other_team_target_occurrence_no"],
            "non_player": exp["non_player"],
            "invalid": exp["invalid"],
            "multi_person_ambiguous": exp["multi_person_ambiguous"],
            "uncertain": exp["uncertain"],
            "clean_positive_segments": exp["clean_positive_segments"],
            "clean_negative_segments": exp["clean_negative_segments"],
            "clean_same_team_negative_segments": exp["clean_same_team_negative_segments"],
            "reviewed_metric_excluded": exp["reviewed_metric_excluded"],
            "unreviewed_ineligible": exp["unreviewed_ineligible"],
            "complete_universe": UNIVERSE_N,
            "component_policy": COMPONENT_POLICY,
            "component_conflict_count": 0,
            "clean_positive_components": 10,
            "clean_negative_components": 110,
            "clean_same_team_negative_components": 55,
            "metric_excluded_reviewed_components": 21,
            "positive_exact_ids": list(POSITIVE_IDS),
            "f3k_snapshot_sha256": f3k["snapshot_sha256"],
            "f3j_segment_count": f3j["summary"]["segment_count"],
            "similarity_rows": 0,
            "ranking_rows": 0,
            "metric_results": 0,
            "embeddings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "gallery_vectors_read": False,
            "holdout_embedding_vectors_read": False,
            "holdout_enrollment_forbidden": True,
            "gallery_growth_forbidden": True,
            "threshold_calibration_forbidden": True,
            "minimum_support_audit_pass": True,
            "reviewer": config["human_review"]["reviewer"],
            "final_approver": config["human_review"]["final_approver"],
            "network_used": False,
            "access_audit": access_audit,
        }
        write_json(tmp / "stage5d_f3l_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")):
            raise GroundTruthFreezeError("artifact budget violated")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f3l_ground_truth_manual_freeze_manifest_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "f3k_snapshot_sha256": f3k["snapshot_sha256"],
            "reviewed_total": REVIEWED_N,
            "complete_universe": UNIVERSE_N,
            "generated_at": reviewed_at,
        }
        write_json(tmp / "stage5d_f3l_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3l_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3l_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f3l_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze human holdout v2 ground-truth decisions for target_001."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to independent_holdout_v2_ground_truth_manual_freeze_stage5d_target_001.yaml",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root (defaults to repository root inferred from script location)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else None
    try:
        summary = run(Path(args.config), project_root)
    except GroundTruthFreezeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"readiness={summary['readiness']} "
        f"reviewed={summary['reviewed_total']} "
        f"pos={summary['clean_positive_segments']} "
        f"neg={summary['clean_negative_segments']} "
        f"stn={summary['clean_same_team_negative_segments']} "
        f"excl={summary['reviewed_metric_excluded']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
