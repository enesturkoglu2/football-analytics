#!/usr/bin/env python3
"""Stage 5D-F3H — Target–distractor scoring contract and holdout design.

Preregisters primary/secondary scoring formulas, GT/metric/outcome policies,
and new independent holdout requirements. Does not score queries, read sample
artifacts, select thresholds, assign identity, mutate galleries, or embed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_distractor_scoring_contract_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3H_TARGET_001_TARGET_DISTRACTOR_SCORING_AND_HOLDOUT_DESIGN_READY"
)
NEXT_GATE = "STAGE5D-F3I_TARGET_001_NEW_INDEPENDENT_HOLDOUT_INGESTION_AND_PREFLIGHT"
PRIMARY_FORMULA = "TARGET_DISTRACTOR_MAX_MARGIN"
ALLOWED_DIRTY = {
    "scripts/run_reid_target_distractor_scoring_contract_and_holdout_design.py",
    "configs/reid/target_distractor_scoring_contract_stage5d_target_001.yaml",
    "tests/test_reid_target_distractor_scoring_contract_and_holdout_design.py",
    "docs/setup/stage5d-target-target-distractor-scoring-contract-and-new-holdout-design.md",
}
GT_VOCAB = (
    "target_occurrence_yes",
    "target_occurrence_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)


class ScoringDesignError(RuntimeError):
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
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ScoringDesignError("unexpected config schema")
    if not config.get("offline_required"):
        raise ScoringDesignError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise ScoringDesignError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ScoringDesignError("BLOCKED_STAGE5D_F3H_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ScoringDesignError("BLOCKED_STAGE5D_F3H_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise ScoringDesignError(
                    "BLOCKED_STAGE5D_F3H_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise ScoringDesignError("BLOCKED_STAGE5D_F3H_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_f3g(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["stage5d_f3g_package"]
    root = project_root / cfg["path"]
    summary = load_json(root / "stage5d_f3g_summary.json")
    contract = load_json(root / "stage5d_f3g_contract.json")
    manifest = load_json(root / "stage5d_f3g_manifest.json")
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH status"
        )
    if summary.get("readiness") != cfg["expected_readiness"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH readiness"
        )
    checks = {
        "reused_target_members": cfg["expected_reused_target_members"],
        "new_target_members": cfg["expected_new_target_members"],
        "target_gallery_v2_members": cfg["expected_target_members"],
        "distractor_members": cfg["expected_distractor_members"],
        "target_centroid": 1,
        "target_medoid": 1,
        "distractor_centroid": 1,
        "distractor_medoid": 1,
        "identity_assignments": 0,
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise ScoringDesignError(
                f"BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH {key}"
            )
    if summary.get("target_medoid_member_id") != cfg["expected_target_medoid_id"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH target_medoid"
        )
    if summary.get("distractor_medoid_member_id") != cfg["expected_distractor_medoid_id"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH distractor_medoid"
        )
    if summary.get("gallery_v1_mutation") is not False:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH gallery_v1_mutation"
        )
    if summary.get("threshold_selected") is not False:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH threshold"
        )
    if summary.get("sample_video_read") is not False:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH sample"
        )
    det = summary.get("two_pass_determinism") or {}
    if det.get("exact_match") is not True or det.get("deterministic") is not True:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH determinism"
        )
    if float(det.get("overall_max_absolute_difference", 1.0)) != 0.0:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH max_abs"
        )

    t_man = load_json(
        root / "target_gallery_v2" / "target_001_gallery_v2_embedding_manifest.json"
    )
    d_man = load_json(
        root
        / "distractor_gallery_v1"
        / "target_001_same_team_distractor_embedding_manifest.json"
    )
    if list(t_man.get("shape") or []) != list(cfg["expected_target_shape"]):
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH target_shape"
        )
    if list(d_man.get("shape") or []) != list(cfg["expected_distractor_shape"]):
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH distractor_shape"
        )
    if t_man.get("dtype") != "float32" or d_man.get("dtype") != "float32":
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH dtype"
        )
    if t_man.get("sha256") != cfg["expected_target_embedding_sha256"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH target_sha"
        )
    if d_man.get("sha256") != cfg["expected_distractor_embedding_sha256"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH distractor_sha"
        )

    # Integrity check via file SHA only — no array load / scoring inspection.
    t_npy = root / "target_gallery_v2" / "target_001_gallery_v2_individual_embeddings.npy"
    d_npy = (
        root
        / "distractor_gallery_v1"
        / "target_001_same_team_distractor_individual_embeddings.npy"
    )
    if sha256_file(t_npy) != t_man["sha256"] or sha256_file(d_npy) != d_man["sha256"]:
        raise ScoringDesignError(
            "BLOCKED_STAGE5D_F3H_GALLERY_CONTRACT_MISMATCH npy_sha_alignment"
        )
    if cfg.get("npy_content_inspection_forbidden") is not True:
        raise ScoringDesignError("npy content inspection must remain forbidden")

    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "manifest": manifest,
        "target_manifest": t_man,
        "distractor_manifest": d_man,
        "snapshot_sha256": snap,
    }


def validate_upstream_designs(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    f3b = load_json(
        project_root
        / config["stage5d_f3b_package"]["path"]
        / "stage5d_f3b_summary.json"
    )
    f1 = load_json(
        project_root
        / config["stage5d_f1_package"]["path"]
        / "stage5d_f1_summary.json"
    )
    if f3b.get("final_status") != config["stage5d_f3b_package"]["expected_final_status"]:
        raise ScoringDesignError("F3B status mismatch")
    if f1.get("final_status") != config["stage5d_f1_package"]["expected_final_status"]:
        raise ScoringDesignError("F1 status mismatch")
    return {"f3b": f3b, "f1": f1}


def build_primary_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_target_distractor_primary_scoring_contract_v1",
        "preregistered": True,
        "frozen_before_any_query_scoring": True,
        "formula_id": PRIMARY_FORMULA,
        "cosine_definition": {
            "vectors_must_be_l2_normalized": True,
            "cosine": "dot_product",
        },
        "query_embedding_contract": {
            "dimension": 512,
            "dtype": "float32",
            "finite": True,
            "non_zero": True,
            "l2_normalized": True,
            "preprocessing": "canonical_existing_osnet_bgr_rgb_256x128_imagenet",
        },
        "scoreable_query_requires": [
            "valid_512d_embedding",
            "verified_source_lineage",
            "suitable_for_identity_purity_review",
        ],
        "unscoreable_query_policy": {
            "not_automatic_negative": True,
            "score_forced_zero_forbidden": True,
            "forced_metric_denominator_forbidden": True,
            "report_in_separate_inventory": True,
            "reasons": [
                "missing_embedding",
                "nan_inf_zero",
                "invalid_or_dirty_geometry",
                "human_gt_invalid_or_non_player",
            ],
        },
        "T_max": {
            "definition": "max cosine against 13 target gallery-v2 individual members",
            "target_top_k": 1,
            "aggregation": "max",
            "member_weighting": False,
        },
        "D_max": {
            "definition": "max cosine against 23 same-team distractor individual members",
            "distractor_top_k": 1,
            "aggregation": "max",
            "member_weighting": False,
        },
        "S_primary": {
            "formula": "T_max(q) - D_max(q)",
            "direction": "higher_means_more_target_like_than_distractor_like",
            "subtraction_order": "target_minus_distractor",
        },
        "forbidden_in_score": [
            "target_member_weighting",
            "distractor_member_weighting",
            "member_removal",
            "jersey_metadata",
            "ocr",
            "team_classifier",
            "temporal_frame_count",
            "sample_coefficient_optimization",
        ],
        "tie_break": [
            "primary_score_descending",
            "T_max_descending",
            "D_max_ascending",
            "query_stable_id_ascending",
        ],
        "exact_float_tie_break": "query_stable_id_ascending",
        "gallery_mutation_forbidden": True,
        "sample_based_formula_selection_forbidden": True,
    }


def build_secondary_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_target_distractor_secondary_scoring_contract_v1",
        "preregistered": True,
        "diagnostic_only": True,
        "cannot_replace_primary_on_same_holdout": True,
        "formula_promotion_requires_new_preregistration_and_new_holdout": True,
        "primary_remains": PRIMARY_FORMULA,
        "formulas": {
            "S_top3_margin": {
                "T_top3_mean": "mean of top-3 target cosines among 13",
                "D_top3_mean": "mean of top-3 distractor cosines among 23",
                "formula": "T_top3_mean(q) - D_top3_mean(q)",
                "k": 3,
                "k_frozen": True,
                "k_may_not_change_from_sample_or_holdout_results": True,
            },
            "S_target_centroid_margin": {
                "formula": "cos(q, target_centroid) - D_max(q)"
            },
            "S_target_medoid_margin": {
                "formula": "cos(q, target_medoid) - D_max(q)"
            },
            "S_mean_margin": {
                "formula": (
                    "mean_cosine(q, 13 target individuals) - "
                    "mean_cosine(q, 23 distractor individuals)"
                )
            },
        },
        "raw_diagnostics": [
            "T_max",
            "D_max",
            "T_top3_mean",
            "D_top3_mean",
            "target_centroid_cosine",
            "target_medoid_cosine",
            "distractor_centroid_cosine",
            "distractor_medoid_cosine",
        ],
    }


def build_tie_break_aggregation_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_target_distractor_tie_break_and_aggregation_contract_v1",
        "query_level_tie_break": [
            "primary_score_descending",
            "T_max_descending",
            "D_max_ascending",
            "query_stable_id_ascending",
        ],
        "component_grouping_keys_frozen_before_holdout_scoring": [
            "raw_track_lineage",
            "documented_link_component",
            "exact_crop_sha_component",
            "source_observation_continuity",
            "overlapping_temporal_window_component",
            "manually_identified_track_impurity_component",
        ],
        "temporal_overlap_alone_insufficient_to_merge_distinct_players": True,
        "conflicting_component_policy": {
            "definition": "same component contains clean positive and clean negative GT",
            "exclude_from_component_level_official_metrics": True,
            "preserve_segment_level_items_with_own_gt": True,
            "report_conflict_separately": True,
        },
        "component_primary_aggregation": "maximum_segment_primary_score",
        "component_secondary_aggregation": "median_segment_primary_score",
        "component_tie_break": [
            "maximum_primary_descending",
            "median_primary_descending",
            "component_stable_id_ascending",
        ],
    }


def build_holdout_requirements(config: Mapping[str, Any], pending: bool) -> dict[str, Any]:
    hold = config["new_independent_holdout"]
    return {
        "schema_version": "reid_target_001_new_independent_holdout_requirements_v1",
        "new_independent_holdout_required": True,
        "holdout_input_pending": pending,
        "designed_path": hold["designed_path"],
        "forbidden_as_holdout": [
            "external_enrollment_video",
            "sample.mp4",
            "reencoded_copy_of_external_or_sample",
            "source_video_of_enrollment_crops",
            "source_video_of_gallery_v2_or_distractor_crops",
        ],
        "forbidden_source_paths": list(hold["forbidden_sources"]),
        "preference_order": [
            "different_match_or_recording_session",
            "same_player_same_kit_different_camera_or_condition",
            "same_match_truly_independent_non_overlapping_third_recording",
        ],
        "ingestion_required_checks": [
            "bytes",
            "sha256",
            "codec",
            "width_height",
            "fps",
            "frame_count",
            "duration",
            "audio_presence",
            "exact_duplicate_audit",
            "frame_fingerprint_overlap_audit",
            "verified_overlap_interval_audit",
            "gallery_crop_source_overlap_audit",
            "sample_external_independence_declaration",
        ],
        "holdout_not_for_gallery_or_threshold_development": True,
        "holdout_only_for_frozen_scoring_contract_evaluation": True,
        "sample_after_refinement_not_independent_revalidation": True,
        "prior_gallery_v1_sample_outcome_immutable_fact": config[
            "prior_gallery_v1_independent_outcome"
        ],
        "missing_file_is_not_blocker_in_f3h": True,
        "exact_next_work_order": [
            "provide_new_holdout_file",
            "ingestion_and_overlap_preflight",
            "label_blind_detection_tracking_segmentation_universe",
            "human_gt_review_package_without_similarity_or_rank",
            "freeze_human_gt_decisions",
            "score_once_with_frozen_gallery_and_scoring_contract",
            "immutable_report",
            "formula_or_gallery_change_requires_new_holdout",
        ],
    }


def build_gt_policy() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_new_holdout_ground_truth_policy_v1",
        "label_blind_before_similarity_or_ranking": True,
        "allowed_human_vocabulary": list(GT_VOCAB),
        "clean_positive": ["target_occurrence_yes"],
        "clean_negative": ["target_occurrence_no", "non_player"],
        "metric_excluded": ["uncertain", "invalid", "multi_person_ambiguous"],
        "unreviewed_is_not_negative": True,
        "jersey_number_human_metadata_only": True,
        "automated_ocr_not_identity_evidence": True,
        "gallery_similarity_or_rank_not_shown_on_review_sheets": True,
        "same_team_distractor_examples_must_be_adequately_represented": True,
        "target_present_multi_person_contaminated": {
            "not_positive_metric_input": True,
            "not_negative": True,
            "separate_diagnostic_cohort": True,
        },
    }


def build_metric_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_new_holdout_metric_contract_v1",
        "minimum_support": {
            "segment_clean_positive_ge": 5,
            "segment_clean_negative_ge": 20,
            "segment_clean_same_team_negative_ge": 10,
            "component_clean_positive_ge": 3,
            "component_clean_negative_ge": 10,
            "insufficient_outcome": "INDEPENDENT_HOLDOUT_INSUFFICIENT_GROUND_TRUTH",
        },
        "auroc_auprc_official_requires": {
            "clean_positive_ge": 5,
            "clean_negative_ge": 20,
        },
        "same_team_negative_auroc_auprc_requires": {
            "clean_positive_ge": 5,
            "clean_same_team_negative_ge": 10,
        },
        "segment_primary_metrics": [
            "positive_ranks",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "MRR",
            "AP",
            "AUROC_if_supported",
            "AUPRC_if_supported",
            "positive_score_min_median_mean_max",
            "negative_score_min_median_mean_max",
            "same_team_negative_score_summary",
            "min_positive_minus_max_negative_margin",
            "positive_median_minus_negative_median",
            "positive_median_minus_same_team_negative_median",
            "positive_overlap_negative_count",
            "same_team_false_positive_cohort",
        ],
        "component_primary_metrics": [
            "positive_component_ranks",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "MRR",
            "AP",
            "AUROC_if_supported",
            "AUPRC_if_supported",
            "component_separation_margin",
            "conflict_component_count",
        ],
        "secondary_diagnostic_metrics": [
            "Recall@5",
            "Recall@10",
            "AP",
            "AUROC_if_supported",
            "separation_margin",
        ],
        "secondary_results_diagnostic_only": True,
        "recall_at_k_ceiling_policy": {
            "if_positive_count_gt_k_report_mathematical_ceiling": True,
            "unreachable_absolute_criterion_forbidden": True,
        },
    }


def build_outcome_rules() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_new_holdout_outcome_rules_v1",
        "outcomes": {
            "INDEPENDENT_TARGET_DISTRACTOR_STRONG_SIGNAL": {
                "requires_minimum_support": True,
                "segment_AP_ge": 0.80,
                "component_AP_ge": 0.80,
                "segment_AUROC_ge": 0.90,
                "same_team_negative_AUROC_ge": 0.85,
                "segment_min_positive_minus_max_negative_margin_gt": 0,
                "component_margin_gt": 0,
            },
            "INDEPENDENT_TARGET_DISTRACTOR_PROMISING_BUT_OVERLAPPING": {
                "positives_near_top": True,
                "ap_auroc_meaningful": True,
                "any_of": [
                    "segment_or_component_margin_le_0",
                    "same_team_confusion_persists",
                ],
            },
            "INDEPENDENT_TARGET_DISTRACTOR_WEAK": {
                "low_ap_or_inconsistent_top_ranks_or_weak_margin_separation": True
            },
            "INDEPENDENT_HOLDOUT_INSUFFICIENT_GROUND_TRUTH": {
                "minimum_support_not_met": True
            },
        },
        "outcomes_are_not": [
            "deployment_approval",
            "threshold",
            "automatic_identity_permission",
            "automatic_gallery_growth_permission",
        ],
        "threshold_and_abstention": {
            "threshold_selected": False,
            "threshold_candidate_count": 0,
            "operating_point_selection_count": 0,
            "identity_assignments": 0,
            "future_threshold_requires_separate_calibration_set": True,
            "using_holdout_for_threshold_selection_burns_test_set": True,
            "abstention_design_only_no_numeric_threshold": True,
            "abstention_triggers_design": [
                "low_target_score",
                "small_target_distractor_margin",
                "multi_person_or_track_impurity",
                "no_embedding",
                "component_conflict",
            ],
        },
    }


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3h_scoring_design_{final_dir.name}_{token}"
    if tmp.exists():
        raise ScoringDesignError("tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ScoringDesignError("final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise ScoringDesignError("final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise ScoringDesignError("sample decode_forbidden required")
    if config["evaluation_source"].get("score_row_read_forbidden") is not True:
        raise ScoringDesignError("sample score_row_read_forbidden required")
    if config["osnet_checkpoint"].get("new_embedding_forbidden") is not True:
        raise ScoringDesignError("new_embedding_forbidden required")

    f3g = validate_f3g(project_root, config)
    upstream = validate_upstream_designs(project_root, config)

    holdout_path = project_root / config["new_independent_holdout"]["designed_path"]
    holdout_pending = not holdout_path.is_file()
    # Do not open sample/external/holdout content.
    sample_path = project_root / config["evaluation_source"]["path"]
    external_path = project_root / config["external_enrollment_source"]["path"]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = create_temp_root(final_dir)
    try:
        scoring = tmp / "scoring"
        evaluation = tmp / "evaluation"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (scoring, evaluation, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        primary = build_primary_contract()
        secondary = build_secondary_contract()
        aggregation = build_tie_break_aggregation_contract()
        holdout_req = build_holdout_requirements(config, holdout_pending)
        gt_policy = build_gt_policy()
        metric_contract = build_metric_contract()
        outcome_rules = build_outcome_rules()

        write_json(
            scoring / "target_001_target_distractor_primary_scoring_contract.json",
            primary,
        )
        write_json(
            scoring / "target_001_target_distractor_secondary_scoring_contract.json",
            secondary,
        )
        write_json(
            scoring
            / "target_001_target_distractor_tie_break_and_aggregation_contract.json",
            aggregation,
        )
        write_json(
            evaluation / "target_001_new_independent_holdout_requirements.json",
            holdout_req,
        )
        write_json(
            evaluation / "target_001_new_holdout_ground_truth_policy.json", gt_policy
        )
        write_json(
            evaluation / "target_001_new_holdout_metric_contract.json", metric_contract
        )
        write_json(
            evaluation / "target_001_new_holdout_outcome_rules.json", outcome_rules
        )

        access_audit = {
            "schema_version": "reid_stage5d_f3h_access_audit_v1",
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "sample_rank_row_read_count": 0,
            "external_video_read": False,
            "query_embedding_read_count": 0,
            "score_computation_count": 0,
            "ranking_row_count": 0,
            "metric_result_count": 0,
            "gallery_npy_content_inspected_for_scoring": False,
            "gallery_npy_sha_verified_only": True,
            "new_embeddings": 0,
            "new_detection": False,
            "new_tracking": False,
            "network_download": 0,
            "sample_path_exists_but_unread": sample_path.is_file(),
            "external_path_exists_but_unread": external_path.is_file(),
            "holdout_path_exists": holdout_path.is_file(),
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "automatic_gallery_growth": False,
            "prior_gallery_v1_independent_outcome_recorded": config[
                "prior_gallery_v1_independent_outcome"
            ],
            "refinement_sample_not_independent_revalidation": True,
        }
        write_json(runtime / "target_001_f3h_access_audit.json", access_audit)

        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise ScoringDesignError("media/npy artifacts forbidden in F3H")

        file_count, files_sha = listing_sha(tmp)
        contract = {
            "schema_version": "reid_stage5d_f3h_scoring_and_holdout_design_contract_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "primary_formula": PRIMARY_FORMULA,
            "primary_preregistered": True,
            "secondary_preregistered": True,
            "target_gallery_v2_members": 13,
            "distractor_gallery_v1_members": 23,
            "query_scoring_rows": 0,
            "rankings": 0,
            "metrics": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "new_embeddings": 0,
            "sample_reads": 0,
            "new_independent_holdout_required": True,
            "holdout_input_pending": holdout_pending,
            "generated_at": generated_at,
            "project_head": head,
        }
        write_json(tmp / "stage5d_f3h_contract.json", contract)
        summary = {
            "schema_version": "reid_stage5d_f3h_scoring_and_holdout_design_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "target_gallery_v2_members": 13,
            "distractor_gallery_v1_members": 23,
            "primary_formula": PRIMARY_FORMULA,
            "primary_target_top_k": 1,
            "primary_distractor_top_k": 1,
            "secondary_top_k": 3,
            "primary_preregistered": True,
            "secondary_preregistered": True,
            "query_scoring_rows": 0,
            "rankings": 0,
            "metrics": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "new_embeddings": 0,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "sample_rank_row_read_count": 0,
            "external_video_read": False,
            "query_embedding_read_count": 0,
            "score_computation_count": 0,
            "new_independent_holdout_required": True,
            "holdout_input_pending": holdout_pending,
            "holdout_designed_path": config["new_independent_holdout"]["designed_path"],
            "prior_gallery_v1_independent_outcome": config[
                "prior_gallery_v1_independent_outcome"
            ],
            "refinement_sample_not_independent_revalidation": True,
            "f3g_snapshot_sha256": f3g["snapshot_sha256"],
            "target_embedding_manifest_sha256": f3g["target_manifest"]["sha256"],
            "distractor_embedding_manifest_sha256": f3g["distractor_manifest"][
                "sha256"
            ],
            "f3b_final_status": upstream["f3b"]["final_status"],
            "f1_final_status": upstream["f1"]["final_status"],
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3h_summary.json", summary)
        write_json(
            tmp / "stage5d_f3h_manifest.json",
            {
                "schema_version": "reid_stage5d_f3h_scoring_and_holdout_design_manifest_v1",
                "final_status": FINAL_STATUS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "npy": 0,
                    "png": 0,
                    "mp4": 0,
                    "score_rows": 0,
                    "rankings": 0,
                    "metrics": 0,
                    "threshold_artifacts": 0,
                    "identity_assignments": 0,
                    "new_embeddings": 0,
                },
                "generated_at": generated_at,
            },
        )
        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_json(final_dir / "stage5d_f3h_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_distractor_scoring_contract_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "exact_next_gate": summary["exact_next_gate"],
                "primary_formula": summary["primary_formula"],
                "holdout_input_pending": summary["holdout_input_pending"],
                "query_scoring_rows": summary["query_scoring_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
