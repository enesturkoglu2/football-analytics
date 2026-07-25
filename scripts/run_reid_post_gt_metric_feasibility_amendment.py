#!/usr/bin/env python3
"""Stage 5D-F2B — Amend retrieval metric feasibility before scoring.

Derives reachable Recall@K ceilings and an amended strong-signal outcome
from frozen ground-truth counts only. Does not load gallery/sample
embedding vectors, compute similarity/rank/metrics, mutate ground truth
or gallery, or select thresholds.
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
from typing import Any, Mapping

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_post_gt_metric_feasibility_amendment_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F2B_TARGET_001_METRIC_CONTRACT_AMENDED_READY_FOR_SCORING"
)
NEXT_GATE = (
    "STAGE5D-F3_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_SCORING_AND_EVALUATION"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_post_gt_metric_feasibility_amendment.py",
    "configs/reid/post_gt_metric_feasibility_amendment_stage5d_target_001.yaml",
    "tests/test_reid_post_gt_metric_feasibility_amendment.py",
    "docs/setup/stage5d-target-post-ground-truth-metric-feasibility-amendment.md",
}

POSITIVE_IDS = (
    "SAMPLE_EVAL_003",
    "SAMPLE_EVAL_024",
    "SAMPLE_EVAL_028",
    "SAMPLE_EVAL_042",
    "SAMPLE_EVAL_046",
    "SAMPLE_EVAL_069",
    "SAMPLE_EVAL_100",
    "SAMPLE_EVAL_102",
)

CONFLICTING_COMPONENTS = (
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_005",
        "positive_members": ["SAMPLE_EVAL_003"],
        "negative_members": ["SAMPLE_EVAL_004"],
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_018",
        "positive_members": ["SAMPLE_EVAL_024"],
        "negative_members": ["SAMPLE_EVAL_025"],
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_032",
        "positive_members": ["SAMPLE_EVAL_042"],
        "negative_members": ["SAMPLE_EVAL_041"],
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_055",
        "positive_members": ["SAMPLE_EVAL_069"],
        "negative_members": ["SAMPLE_EVAL_068"],
    },
)

SEGMENT_POS = 8
SEGMENT_NEG = 110
SEGMENT_EXCL = 32
ELIGIBLE_SEGMENTS = 118
COMP_POS = 4
COMP_NEG = 95
COMP_EXCL = 26
COMP_CONFLICT = 4
METRIC_COMPONENTS = 99

PRIMARY_SCORE = "max_individual_cosine"
SECONDARY_SCORES = (
    "top3_mean_individual_cosine",
    "centroid_cosine",
    "medoid_cosine",
    "mean_individual_cosine",
)


class MetricFeasibilityError(RuntimeError):
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MetricFeasibilityError("unexpected config schema")
    if not config.get("offline_required"):
        raise MetricFeasibilityError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise MetricFeasibilityError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise MetricFeasibilityError("BLOCKED_STAGE5D_F2B_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GIT_CONTRACT_MISMATCH origin"
        )
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise MetricFeasibilityError(
                    "BLOCKED_STAGE5D_F2B_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GIT_CONTRACT_MISMATCH message"
        )
    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "data/enrollment_clips/target_001_external_enrollment_v1.mp4",
        ],
        cwd=project_root,
        text=True,
    ).strip()
    if tracked:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_snapshot_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def recall_ceiling(positives: int, k: int) -> float:
    if positives <= 0:
        return 0.0
    return min(k, positives) / positives


def segment_recall_feasibility() -> dict[str, Any]:
    ceilings = {
        "Recall@1": recall_ceiling(SEGMENT_POS, 1),
        "Recall@3": recall_ceiling(SEGMENT_POS, 3),
        "Recall@5": recall_ceiling(SEGMENT_POS, 5),
        "Recall@10": recall_ceiling(SEGMENT_POS, 10),
    }
    return {
        "schema_version": "reid_target_001_recall_at_k_feasibility_v1",
        "amendment_basis": "frozen_ground_truth_counts_only",
        "similarity_observed": False,
        "ranks_observed": False,
        "segment_positives": SEGMENT_POS,
        "segment_negatives": SEGMENT_NEG,
        "segment_excluded": SEGMENT_EXCL,
        "definition": (
            "Recall@K = count of clean target-positive segments in top-K / "
            f"{SEGMENT_POS}"
        ),
        "mathematical_ceilings": ceilings,
        "original_strong_signal_segment_Recall@5": 1.0,
        "segment_Recall@5_equals_1_0_mathematically_attainable": False,
        "original_strong_signal_criterion_feasible": False,
        "infeasibility_reason": "positive_count_exceeds_k",
        "component_positives": COMP_POS,
        "component_definition": (
            "component Recall@K = count of clean positive components in top-K / "
            f"{COMP_POS}"
        ),
        "component_mathematical_ceilings": {
            "Recall@1": recall_ceiling(COMP_POS, 1),
            "Recall@3": recall_ceiling(COMP_POS, 3),
            "Recall@5": recall_ceiling(COMP_POS, 5),
            "Recall@10": recall_ceiling(COMP_POS, 10),
        },
        "component_Recall@5_equals_1_0_mathematically_attainable": True,
        "exact_positive_ids": list(POSITIVE_IDS),
    }


def original_outcome_reference(f1_metric: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = f1_metric["descriptive_outcomes"]
    strong = outcomes["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"]
    return {
        "schema_version": "reid_target_001_original_outcome_contract_reference_v1",
        "source": "stage5d_f1_metric_design",
        "primary_retrieval_score": PRIMARY_SCORE,
        "secondary_diagnostic_scores": list(SECONDARY_SCORES),
        "segment_level_primary_metrics": list(
            f1_metric["segment_level_primary_metrics"]
        ),
        "independent_component_level_primary_metrics": list(
            f1_metric["independent_component_level_primary_metrics"]
        ),
        "descriptive_outcomes": outcomes,
        "original_strong_signal": strong,
        "original_scoring_formulas_unchanged": True,
        "only_amendment_target": (
            "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL.Recall@5 unreachable segment "
            "requirement (1.0 with 8 positives)"
        ),
    }


def amended_outcome_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_amended_retrieval_outcome_contract_v1",
        "amended_before_scoring": True,
        "amendment_basis": "frozen_ground_truth_counts_only",
        "similarity_observed": False,
        "ranks_observed": False,
        "original_scoring_formulas_unchanged": True,
        "primary_retrieval_score_unchanged": PRIMARY_SCORE,
        "secondary_diagnostic_scores_unchanged": list(SECONDARY_SCORES),
        "outcome_names_preserved": [
            "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL",
            "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING",
            "INDEPENDENT_RETRIEVAL_WEAK",
            "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH",
        ],
        "insufficient_ground_truth_expected": False,
        "frozen_support": {
            "clean_positive_segment_count": SEGMENT_POS,
            "clean_negative_segment_count": SEGMENT_NEG,
            "clean_positive_component_count": COMP_POS,
            "clean_negative_component_count": COMP_NEG,
        },
        "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL": {
            "support": {
                "clean_positive_segment_count_ge": 2,
                "clean_negative_segment_count_ge": 20,
                "clean_positive_component_count_ge": 2,
                "clean_negative_component_count_ge": 20,
            },
            "ranking": {
                "segment_Recall@5": 0.625,
                "segment_Recall@5_note": (
                    "Reachable maximum with 8 positives; all of the first five "
                    "ranks must be target-positive."
                ),
                "segment_Recall@10": 1.0,
                "segment_Recall@10_note": (
                    "All eight clean target-positive segments must appear in "
                    "the first ten ranks."
                ),
                "component_Recall@5": 1.0,
                "component_Recall@5_note": (
                    "All four clean target-positive components must appear in "
                    "the first five component ranks."
                ),
            },
            "quality": {
                "segment_AP_ge": 0.80,
                "component_AP_ge": 0.80,
                "min_positive_segment_score_gt_max_negative_segment_score": True,
                "min_positive_component_score_gt_max_negative_component_score": True,
            },
        },
        "other_outcomes_unchanged": {
            "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING": {
                "positives_near_top": True,
                "positive_negative_overlap_or_margin_le_0": True,
                "threshold_not_independently_selectable": True,
            },
            "INDEPENDENT_RETRIEVAL_WEAK": {
                "positives_not_consistently_top_ranked": True,
                "Recall@5_or_AP_far_below_strong_signal_mins": True,
            },
            "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH": {
                "positive_scoreable_segments_lt": 2,
                "negative_scoreable_segments_lt": 20,
                "positive_independent_components_lt": 2,
            },
        },
        "f3_ranking_universe": {
            "full_diagnostic_scoring_universe": {
                "items": 150,
                "note": (
                    "All scoreable SAMPLE_EVAL items may receive primary and "
                    "secondary scores; excluded item scores are diagnostic only "
                    "and do not receive metric labels."
                ),
            },
            "primary_segment_metric_ranking": {
                "eligible_segments": ELIGIBLE_SEGMENTS,
                "positives": SEGMENT_POS,
                "negatives": SEGMENT_NEG,
                "uncertain_and_ambiguous_excluded": True,
            },
            "component_metric_ranking": {
                "clean_components": METRIC_COMPONENTS,
                "positives": COMP_POS,
                "negatives": COMP_NEG,
                "excluded_components": COMP_EXCL,
                "conflicting_components": COMP_CONFLICT,
                "excluded_and_conflicting_out_of_ranking": True,
            },
            "tie_break": [
                "score_descending",
                "SAMPLE_EVAL_or_SAMPLE_COMPONENT_code_ascending",
            ],
            "nondeterministic_ordering_forbidden": True,
        },
    }


def conflict_policy() -> dict[str, Any]:
    retained_segments: list[str] = []
    for row in CONFLICTING_COMPONENTS:
        retained_segments.extend(row["positive_members"])
        retained_segments.extend(row["negative_members"])
    return {
        "schema_version": "reid_target_001_component_conflict_metric_policy_v1",
        "conflicting_component_count": COMP_CONFLICT,
        "conflicting_components": list(CONFLICTING_COMPONENTS),
        "policy": {
            "excluded_from_component_level_metrics": True,
            "component_score_label_not_created": True,
            "not_counted_as_positive_or_negative_component": True,
            "not_in_component_AP_MRR_Recall_universe": True,
            "clean_segment_members_retain_segment_metric_eligibility": True,
            "ground_truth_decisions_unchanged": True,
            "similarity_not_used_to_resolve_conflict": True,
        },
        "retained_segment_level_member_codes": sorted(set(retained_segments)),
        "component_metric_universe": {
            "clean_positive_components": COMP_POS,
            "clean_negative_components": COMP_NEG,
            "metric_component_total": METRIC_COMPONENTS,
            "excluded_components": COMP_EXCL,
            "conflicting_components": COMP_CONFLICT,
        },
    }


def validate_f2a(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f2a_freeze"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    if not root.is_dir():
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH missing_f2a"
        )
    summary = load_json(root / "stage5d_f2a_summary.json")
    cfg = config["stage5d_f2a_freeze"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH status"
        )
    expected = {
        "reviewed_total": cfg["expected_reviewed_total"],
        "clean_positive_metric_items": cfg["expected_segment_positives"],
        "clean_negative_metric_items": cfg["expected_segment_negatives"],
        "excluded_metric_items": cfg["expected_segment_excluded"],
        "eligible_total": cfg["expected_eligible_total"],
        "similarity_rows": cfg["expected_similarity_rows"],
        "ranking_rows": cfg["expected_ranking_rows"],
        "gallery_members": cfg["expected_gallery_members"],
    }
    for key, want in expected.items():
        if summary.get(key) != want:
            raise MetricFeasibilityError(
                f"BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH {key}"
            )
    if summary.get("threshold_selected") is not False:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH threshold"
        )
    if summary.get("gallery_vectors_read") is not False:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH gallery_vectors"
        )
    if summary.get("sample_embedding_vectors_read") is not False:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH sample_vectors"
        )
    dist = summary["component_label_distribution"]
    if dist["positive_component_count"] != cfg["expected_positive_components"]:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH cpos"
        )
    if dist["negative_component_count"] != cfg["expected_negative_components"]:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH cneg"
        )
    if dist["excluded_component_count"] != cfg["expected_excluded_components"]:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH cexcl"
        )
    if dist["conflicting_component_count"] != cfg["expected_conflicting_components"]:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH cconflict"
        )
    if summary.get("positive_exact_ids") != list(POSITIVE_IDS):
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH positive_ids"
        )
    if summary.get("target_occurrence_yes") != 8:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH yes"
        )
    if summary.get("target_occurrence_no") != 103:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH no"
        )
    if summary.get("non_player") != 7:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH non_player"
        )
    if summary.get("uncertain") != 8:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH uncertain"
        )
    if summary.get("multi_person_ambiguous") != 24:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH ambiguous"
        )
    if summary.get("invalid") != 0:
        raise MetricFeasibilityError(
            "BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH invalid"
        )

    live_conflicts = {
        row["evaluation_component_id"]: row for row in dist["conflict_details"]
    }
    for expected_row in CONFLICTING_COMPONENTS:
        cid = expected_row["evaluation_component_id"]
        if cid not in live_conflicts:
            raise MetricFeasibilityError(
                f"BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH missing_{cid}"
            )
        live = live_conflicts[cid]
        if live["positive_members"] != expected_row["positive_members"]:
            raise MetricFeasibilityError(
                f"BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH pos_{cid}"
            )
        if live["negative_members"] != expected_row["negative_members"]:
            raise MetricFeasibilityError(
                f"BLOCKED_STAGE5D_F2B_GROUND_TRUTH_CONTRACT_MISMATCH neg_{cid}"
            )

    snap_sha = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    freeze = load_json(
        root / "ground_truth_freeze" / "target_001_sample_ground_truth_freeze.json"
    )
    return {
        "summary": summary,
        "freeze": freeze,
        "snapshot_sha256": snap_sha,
        "root": root,
        "root_listing_sha256": listing_sha(root)[1],
    }


def validate_f1(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f1_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f1_summary.json")
    if summary.get("final_status") != config["stage5d_f1_package"][
        "expected_final_status"
    ]:
        raise MetricFeasibilityError("F1 status mismatch")
    if summary.get("primary_retrieval_score") != PRIMARY_SCORE:
        raise MetricFeasibilityError("F1 primary score mismatch")
    scoring = load_json(
        root / "scoring_design" / "target_001_retrieval_scoring_contract.json"
    )
    metric = load_json(
        root / "metric_design" / "target_001_retrieval_metric_contract.json"
    )
    if scoring["primary_retrieval_score"]["name"] != PRIMARY_SCORE:
        raise MetricFeasibilityError("scoring primary mismatch")
    sec = [row["name"] for row in scoring["secondary_diagnostic_scores"]]
    if sec != list(SECONDARY_SCORES):
        raise MetricFeasibilityError("secondary scores mismatch")
    strong = metric["descriptive_outcomes"]["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"]
    if strong.get("Recall@5") != 1.0:
        raise MetricFeasibilityError("original Recall@5 expected 1.0")
    return {
        "summary": summary,
        "scoring": scoring,
        "metric": metric,
        "root": root,
        "root_listing_sha256": listing_sha(root)[1],
    }


def validate_gallery_meta(project_root: Path, config: Mapping[str, Any]) -> None:
    rel = config["gallery_v1"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    members = summary.get("individual_gallery_members", summary.get("gallery_members"))
    if int(members if members is not None else -1) != int(
        config["gallery_v1"]["expected_members"]
    ):
        raise MetricFeasibilityError("gallery members mismatch")
    if config["gallery_v1"].get("npy_load_forbidden") is not True:
        raise MetricFeasibilityError("npy_load_forbidden required")
    # Do not open any .npy under gallery.
    for npy in root.rglob("*.npy"):
        _ = npy  # path existence only; never load arrays
        break


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f2b_metric_amendment_{final_dir.name}_{token}"
    if tmp.exists():
        raise MetricFeasibilityError(f"temp root exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise MetricFeasibilityError(f"final root already exists: {final_dir}")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise MetricFeasibilityError(f"final root already exists: {final_dir}")

    f2a = validate_f2a(project_root, config)
    f1 = validate_f1(project_root, config)
    validate_gallery_meta(project_root, config)

    f2a_listing_before = f2a["root_listing_sha256"]
    f1_listing_before = f1["root_listing_sha256"]

    feasibility = segment_recall_feasibility()
    original_ref = original_outcome_reference(f1["metric"])
    amended = amended_outcome_contract()
    conflicts = conflict_policy()

    generated_at = datetime.now(timezone.utc).isoformat()
    tmp = create_temp_root(final_dir)
    try:
        mf = tmp / "metric_feasibility"
        mf.mkdir(parents=True, exist_ok=True)
        write_json(mf / "target_001_recall_at_k_feasibility.json", feasibility)
        write_json(
            mf / "target_001_original_outcome_contract_reference.json", original_ref
        )
        write_json(
            mf / "target_001_amended_retrieval_outcome_contract.json", amended
        )
        write_json(
            mf / "target_001_component_conflict_metric_policy.json", conflicts
        )

        contract = {
            "schema_version": "reid_stage5d_f2b_metric_feasibility_amendment_contract_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "amendment_basis": "frozen_ground_truth_counts_only",
            "similarity_observed": False,
            "ranks_observed": False,
            "gallery_vectors_read": False,
            "sample_vectors_read": False,
            "ground_truth_mutated": False,
            "gallery_mutated": False,
            "threshold_selected": False,
            "identity_assignment": False,
            "original_scoring_formulas_unchanged": True,
            "amended_before_scoring": True,
            "reviewed_ground_truth": 150,
            "segment_positives": SEGMENT_POS,
            "segment_negatives": SEGMENT_NEG,
            "segment_excluded": SEGMENT_EXCL,
            "clean_positive_components": COMP_POS,
            "clean_negative_components": COMP_NEG,
            "excluded_components": COMP_EXCL,
            "conflicting_components": COMP_CONFLICT,
            "similarity_rows": 0,
            "ranking_rows": 0,
            "metric_result_rows": 0,
            "gallery_members": 7,
            "f2a_snapshot_sha256": f2a["snapshot_sha256"],
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f2b_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f2b_metric_feasibility_amendment_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reviewed_ground_truth": 150,
            "segment_positives": SEGMENT_POS,
            "segment_negatives": SEGMENT_NEG,
            "segment_excluded": SEGMENT_EXCL,
            "eligible_segment_total": ELIGIBLE_SEGMENTS,
            "clean_positive_components": COMP_POS,
            "clean_negative_components": COMP_NEG,
            "excluded_components": COMP_EXCL,
            "conflicting_components": COMP_CONFLICT,
            "metric_component_total": METRIC_COMPONENTS,
            "positive_exact_ids": list(POSITIVE_IDS),
            "segment_recall_ceilings": feasibility["mathematical_ceilings"],
            "component_recall_ceilings": feasibility[
                "component_mathematical_ceilings"
            ],
            "original_segment_Recall@5_1_0_attainable": False,
            "original_strong_signal_feasible": False,
            "infeasibility_reason": "positive_count_exceeds_k",
            "amended_strong_signal": amended["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"],
            "insufficient_ground_truth_expected": False,
            "similarity_rows": 0,
            "ranking_rows": 0,
            "metric_result_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_members": 7,
            "gallery_unchanged": True,
            "gallery_vectors_read": False,
            "sample_embedding_vectors_read": False,
            "original_scoring_formulas_unchanged": True,
            "amended_before_scoring": True,
            "amendment_basis": "frozen_ground_truth_counts_only",
            "f2a_snapshot_sha256": f2a["snapshot_sha256"],
            "sample_sha256": config["evaluation_source"]["expected_sha256"],
            "external_source_sha256": config["external_enrollment_source"][
                "expected_sha256"
            ],
            "network_used": False,
            "package_environment_changed": False,
            "artifact_budget": {
                "csv": 0,
                "npy": 0,
                "png": 0,
                "mp4": 0,
                "similarity_rows": 0,
                "ranking_rows": 0,
                "metric_result_rows": 0,
            },
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f2b_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")) or list(tmp.rglob("*.csv")):
            raise MetricFeasibilityError("artifact budget violated")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f2b_metric_feasibility_amendment_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "f2a_snapshot_sha256": f2a["snapshot_sha256"],
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f2b_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2b_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f2b_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Upstream immutability
    if listing_sha(f2a["root"])[1] != f2a_listing_before:
        raise MetricFeasibilityError("F2A root mutated")
    if listing_sha(f1["root"])[1] != f1_listing_before:
        raise MetricFeasibilityError("F1 root mutated")

    return load_json(final_dir / "stage5d_f2b_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Amend target_001 retrieval metric feasibility before scoring."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to post_gt_metric_feasibility_amendment_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except MetricFeasibilityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"seg_pos={summary['segment_positives']} "
        f"seg_neg={summary['segment_negatives']} "
        f"Recall@5_ceiling={summary['segment_recall_ceilings']['Recall@5']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
