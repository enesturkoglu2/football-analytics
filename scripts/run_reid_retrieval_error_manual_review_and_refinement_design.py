#!/usr/bin/env python3
"""Stage 5D-F3B — Freeze human retrieval-error findings and refinement design.

Freezes human review of F3A diagnostics and an external-only refinement
plan. Does not recompute scores, mutate gallery/GT, create embeddings,
select thresholds, or assign identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
CONFIG_SCHEMA = "reid_retrieval_error_manual_review_and_refinement_design_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = "COMPLETED_STAGE5D_F3B_TARGET_001_REFINEMENT_DESIGN_READY"
NEXT_GATE = (
    "STAGE5D-F3C_TARGET_001_EXTERNAL_ONLY_HARD_NEGATIVE_AND_VIEW_ANCHOR_REVIEW_PACKAGE"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_retrieval_error_manual_review_and_refinement_design.py",
    "configs/reid/retrieval_error_manual_review_and_refinement_design_stage5d_target_001.yaml",
    "tests/test_reid_retrieval_error_manual_review_and_refinement_design.py",
    "docs/setup/stage5d-target-retrieval-error-manual-review-and-refinement-design.md",
}

HIGH_POSITIVES = ("SAMPLE_EVAL_100", "SAMPLE_EVAL_003", "SAMPLE_EVAL_028")
VISIBLE_JERSEY = {
    "SAMPLE_EVAL_111": "20",
    "SAMPLE_EVAL_090": "7",
    "SAMPLE_EVAL_130": "9",
    "SAMPLE_EVAL_095": "30",
    "SAMPLE_EVAL_127": "3",
    "SAMPLE_EVAL_085": "14",
    "SAMPLE_EVAL_080": "14",
    "SAMPLE_EVAL_052": "14",
    "SAMPLE_EVAL_128": "25",
    "SAMPLE_EVAL_043": "7",
    "SAMPLE_EVAL_092": "20",
    "SAMPLE_EVAL_121": "30",
}
CONFLICT_FINDINGS = (
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_005",
        "positive": "SAMPLE_EVAL_003",
        "negative": "SAMPLE_EVAL_004",
        "note": "yellow target and white number 8 are different players",
        "cause": "component_grouping_overmerge_candidate",
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_018",
        "positive": "SAMPLE_EVAL_024",
        "negative": "SAMPLE_EVAL_025",
        "note": "yellow target and white player are different players",
        "cause": "component_grouping_overmerge_candidate",
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_032",
        "positive": "SAMPLE_EVAL_042",
        "negative": "SAMPLE_EVAL_041",
        "note": "yellow target and white player are different players",
        "cause": "component_grouping_overmerge_candidate",
    },
    {
        "evaluation_component_id": "SAMPLE_COMPONENT_055",
        "positive": "SAMPLE_EVAL_069",
        "negative": "SAMPLE_EVAL_068",
        "note": "yellow target and white player are different players",
        "cause": "component_grouping_overmerge_candidate",
    },
)

FINDINGS_CSV_FIELDS = (
    "item_type",
    "sample_eval_code",
    "evaluation_component_id",
    "cohort",
    "frozen_ground_truth_decision",
    "visible_jersey_number",
    "manual_finding",
    "hypothesis",
    "root_cause",
    "ground_truth_change",
    "gallery_change",
    "hypothesis_only",
    "removal_authorized",
    "reviewer",
    "final_approver",
)


class RefinementDesignError(RuntimeError):
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
        raise RefinementDesignError("unexpected config schema")
    if not config.get("offline_required"):
        raise RefinementDesignError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise RefinementDesignError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise RefinementDesignError("BLOCKED_STAGE5D_F3B_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise RefinementDesignError("BLOCKED_STAGE5D_F3B_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise RefinementDesignError(
                    "BLOCKED_STAGE5D_F3B_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not manifest.is_file():
        manifest = snapshot_path.with_name(
            snapshot_path.name.replace(".tar.gz", "_snapshot_manifest.json")
        )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_f3a(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3a_diagnostics"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3a_summary.json")
    cfg = config["stage5d_f3a_diagnostics"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH status"
        )
    if summary.get("official_f3_descriptive_outcome") != cfg["expected_f3_outcome"]:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH outcome"
        )
    checks = {
        "high_ranked_positives": 3,
        "low_ranked_positives": 5,
        "top_false_positive_cohort": 24,
        "overlap_negative_cohort": 26,
        "excluded_review_cohort": 12,
        "conflict_components": 4,
        "contact_sheets": 5,
        "diagnostic_videos": 2,
        "identity_assignments": 0,
    }
    for key, want in checks.items():
        if summary.get(key) != want:
            raise RefinementDesignError(
                f"BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH {key}"
            )
    if summary.get("gallery_mutation") is not False:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH gallery_mutation"
        )
    if summary.get("score_recompute") is not False:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH score_recompute"
        )
    if summary.get("threshold_selected") is not False:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH threshold"
        )
    fp = load_jsonl(root / "analysis" / "target_001_top_false_positive_analysis.jsonl")
    if len(fp) != 24:
        raise RefinementDesignError(
            "BLOCKED_STAGE5D_F3B_DIAGNOSTIC_CONTRACT_MISMATCH fp_count"
        )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "top_false_positives": fp,
        "snapshot_sha256": snap,
        "listing_sha256": listing_sha(root)[1],
    }


def validate_f3(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3_evaluation"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3_summary.json")
    cfg = config["stage5d_f3_evaluation"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise RefinementDesignError("F3 status mismatch")
    seg = summary["segment_metrics"]
    if not math.isclose(
        float(seg["Average_Precision"]),
        float(cfg["expected_segment_ap"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RefinementDesignError("F3 AP changed")
    if not math.isclose(
        float(seg["separation_margin"]),
        float(cfg["expected_segment_margin"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RefinementDesignError("F3 margin changed")
    if summary.get("gallery_members") != 7:
        raise RefinementDesignError("gallery members changed")
    return {"root": root, "summary": summary, "listing_sha256": listing_sha(root)[1]}


def build_manual_findings(top_fp: list[dict[str, Any]]) -> dict[str, Any]:
    high = []
    for code in HIGH_POSITIVES:
        high.append(
            {
                "sample_eval_code": code,
                "cohort": "high_ranked_positive",
                "target_identity_confirmed": True,
                "crop_target_dominant": True,
                "gallery_compatible_appearance": True,
                "relatively_clean_target_view": True,
                "ground_truth_change": False,
                "gallery_change": False,
                "hypothesis_only": True,
            }
        )
    low = [
        {
            "sample_eval_code": "SAMPLE_EVAL_046",
            "cohort": "low_ranked_positive",
            "target_crop_valid": True,
            "target_identity_confirmed": True,
            "view": "rear_or_rear_oblique",
            "hypothesis": "view_specific_confusion_and_same_uniform_overlap",
            "tracking_identity_switch_proven": False,
            "ground_truth_change": False,
            "gallery_change": False,
            "hypothesis_only": True,
        },
        {
            "sample_eval_code": "SAMPLE_EVAL_102",
            "cohort": "low_ranked_positive",
            "target_crop_valid": True,
            "target_identity_confirmed": True,
            "view": "front_or_front_oblique",
            "hypothesis": "front_view_gallery_support_gap_candidate",
            "tracking_identity_switch_proven": False,
            "ground_truth_change": False,
            "gallery_change": False,
            "hypothesis_only": True,
        },
        {
            "sample_eval_code": "SAMPLE_EVAL_024",
            "cohort": "low_ranked_positive",
            "target_crop_valid": True,
            "visible_jersey_5": True,
            "nearby_white_player_present": True,
            "hypothesis": "nearby_player_context_and_overlap_candidate",
            "tracking_identity_switch_proven": False,
            "ground_truth_change": False,
            "gallery_change": False,
            "hypothesis_only": True,
        },
        {
            "sample_eval_code": "SAMPLE_EVAL_042",
            "cohort": "low_ranked_positive",
            "target_crop_valid": True,
            "front_view": True,
            "small_scale_candidate": True,
            "hypothesis": "front_view_and_small_scale_gallery_gap_candidate",
            "ground_truth_change": False,
            "gallery_change": False,
            "hypothesis_only": True,
        },
        {
            "sample_eval_code": "SAMPLE_EVAL_069",
            "cohort": "low_ranked_positive",
            "target_crop_valid": True,
            "very_small_scale": True,
            "blur_candidate": True,
            "low_observation_support": True,
            "hypothesis": "small_scale_and_blur_candidate",
            "ground_truth_change": False,
            "gallery_change": False,
            "hypothesis_only": True,
        },
    ]

    fp_rows = []
    top_codes = [r["sample_eval_code"] for r in top_fp]
    for row in top_fp:
        code = row["sample_eval_code"]
        if code in VISIBLE_JERSEY:
            fp_rows.append(
                {
                    "sample_eval_code": code,
                    "official_rank": row.get("official_rank"),
                    "official_primary_score": row.get("official_primary_score"),
                    "frozen_target_negative_preserved": True,
                    "visible_jersey_number": VISIBLE_JERSEY[code],
                    "same_team_same_uniform_distractor": True,
                    "is_target_001": False,
                    "false_positive_root_cause": "same_uniform_confusion_candidate",
                    "automatic_ocr_used": False,
                    "ground_truth_change": False,
                }
            )
        else:
            fp_rows.append(
                {
                    "sample_eval_code": code,
                    "official_rank": row.get("official_rank"),
                    "official_primary_score": row.get("official_primary_score"),
                    "frozen_target_negative_preserved": True,
                    "visible_jersey_number": "unknown",
                    "primary_visual_cause": (
                        "same_uniform_and_similar_view_confusion_candidate"
                    ),
                    "is_target_001": False,
                    "automatic_ocr_used": False,
                    "ground_truth_change": False,
                }
            )

    for code in VISIBLE_JERSEY:
        if code not in top_codes:
            raise RefinementDesignError(f"visible jersey code not in top24: {code}")

    conflicts = []
    for row in CONFLICT_FINDINGS:
        conflicts.append(
            {
                **row,
                "human_label_conflict": False,
                "target_identity_error": False,
                "component_overmerge": True,
                "different_team_players_merged": True,
                "official_component_exclusion_remains_valid": True,
                "grouping_contract_refinement_required": True,
                "grouping_artifacts_changed": False,
            }
        )

    anchors = [
        {
            "anchor_id": "target_001_ext_anchor_006",
            "role": "retain_critical_anchor_candidate",
            "positive_best_match_count": 5,
            "supports_front_front_oblique_positives": True,
            "leave_one_out_ap_materially_decreases": True,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_008",
            "role": "review_for_overgeneralization",
            "best_all": 77,
            "best_negative": 63,
            "best_positive": 2,
            "broad_same_uniform_matcher_candidate": True,
            "still_positive_supporting": True,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_001",
            "role": "review_for_confusion",
            "positive_best_match": 0,
            "negative_best_match": 15,
            "leave_one_out_ap_improves": True,
            "rear_view_same_uniform_confusion_candidate": True,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_014",
            "role": "review_for_confusion",
            "positive_best_match": 0,
            "negative_best_match": 4,
            "leave_one_out_ap_improves": True,
            "internal_consistency_diagnostic_flag_exists": True,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_004",
            "role": "internal_medoid_not_discriminative",
            "positive_best_match": 0,
            "negative_best_match": 5,
            "internal_representativeness_ne_external_discrimination": True,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_003",
            "role": "neutral_or_limited_support",
            "automatic_action": False,
            "removal_authorized": False,
        },
        {
            "anchor_id": "target_001_ext_anchor_011",
            "role": "limited_positive_support",
            "positive_best_match": 1,
            "automatic_action": False,
            "removal_authorized": False,
        },
    ]

    return {
        "schema_version": "reid_target_001_retrieval_error_manual_findings_v1",
        "reviewer": "Furkan",
        "final_approver": "Furkan",
        "reviewed_diagnostic_findings_frozen": True,
        "high_ranked_positives": high,
        "low_ranked_positives": low,
        "top_false_positives_exact_codes": top_codes,
        "false_positive_findings": fp_rows,
        "false_positive_common_human_finding": {
            "predominantly_same_yellow_team": True,
            "yellow_kit_and_black_sleeves_dominate_appearance_similarity": True,
            "generic_osnet_does_not_reliably_separate_jersey_identity": True,
            "target_nontarget_score_overlap_visually_plausible": True,
            "ground_truth_change_not_authorized": True,
        },
        "conflicting_component_findings": conflicts,
        "component_grouping_refinement_design_principles": [
            "temporal_overlap_alone_must_not_connect_identities",
            "simultaneous_spatially_distinct_players_must_not_merge",
            "different_team_evidence_may_veto_component_linking_when_reliable",
            "shared_raw_track_documented_link_must_be_audited_for_identity_switch",
            "component_connection_must_preserve_observation_level_identity_continuity",
            "future_component_policy_must_be_preregistered_before_new_validation",
        ],
        "gallery_anchor_manual_findings": anchors,
        "root_cause_summary": {
            "primary_causes": [
                "same_uniform_confusion",
                "low_resolution_and_blur",
                "view_coverage_gap",
                "max_over_gallery_overgeneralization",
                "component_grouping_overmerge",
            ],
            "not_proven": [
                "gallery_source_leakage",
                "wrong_target_enrollment",
                "osnet_nondeterminism",
                "embedding_corruption",
                "universal_tracking_identity_switch",
                "ground_truth_failure",
            ],
        },
    }


def build_csv_rows(findings: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in findings["high_ranked_positives"]:
        rows.append(
            {
                "item_type": "positive",
                "sample_eval_code": item["sample_eval_code"],
                "evaluation_component_id": "",
                "cohort": item["cohort"],
                "frozen_ground_truth_decision": "target_occurrence_yes",
                "visible_jersey_number": "",
                "manual_finding": "relatively_clean_target_view",
                "hypothesis": "",
                "root_cause": "",
                "ground_truth_change": "false",
                "gallery_change": "false",
                "hypothesis_only": "true",
                "removal_authorized": "",
                "reviewer": "Furkan",
                "final_approver": "Furkan",
            }
        )
    for item in findings["low_ranked_positives"]:
        rows.append(
            {
                "item_type": "positive",
                "sample_eval_code": item["sample_eval_code"],
                "evaluation_component_id": "",
                "cohort": item["cohort"],
                "frozen_ground_truth_decision": "target_occurrence_yes",
                "visible_jersey_number": "5"
                if item.get("visible_jersey_5")
                else "",
                "manual_finding": "target_crop_valid",
                "hypothesis": item["hypothesis"],
                "root_cause": "",
                "ground_truth_change": "false",
                "gallery_change": "false",
                "hypothesis_only": "true",
                "removal_authorized": "",
                "reviewer": "Furkan",
                "final_approver": "Furkan",
            }
        )
    for item in findings["false_positive_findings"]:
        rows.append(
            {
                "item_type": "false_positive",
                "sample_eval_code": item["sample_eval_code"],
                "evaluation_component_id": "",
                "cohort": "top_false_positive",
                "frozen_ground_truth_decision": "target_occurrence_no_or_non_player",
                "visible_jersey_number": item.get("visible_jersey_number", "unknown"),
                "manual_finding": "same_team_distractor",
                "hypothesis": "",
                "root_cause": item.get("false_positive_root_cause")
                or item.get("primary_visual_cause"),
                "ground_truth_change": "false",
                "gallery_change": "false",
                "hypothesis_only": "true",
                "removal_authorized": "",
                "reviewer": "Furkan",
                "final_approver": "Furkan",
            }
        )
    for item in findings["conflicting_component_findings"]:
        rows.append(
            {
                "item_type": "conflicting_component",
                "sample_eval_code": f"{item['positive']}|{item['negative']}",
                "evaluation_component_id": item["evaluation_component_id"],
                "cohort": "conflicting_component",
                "frozen_ground_truth_decision": "unchanged",
                "visible_jersey_number": "",
                "manual_finding": item["note"],
                "hypothesis": "",
                "root_cause": item["cause"],
                "ground_truth_change": "false",
                "gallery_change": "false",
                "hypothesis_only": "true",
                "removal_authorized": "",
                "reviewer": "Furkan",
                "final_approver": "Furkan",
            }
        )
    for item in findings["gallery_anchor_manual_findings"]:
        rows.append(
            {
                "item_type": "gallery_anchor",
                "sample_eval_code": "",
                "evaluation_component_id": item["anchor_id"],
                "cohort": "gallery_anchor",
                "frozen_ground_truth_decision": "",
                "visible_jersey_number": "",
                "manual_finding": item["role"],
                "hypothesis": "",
                "root_cause": "",
                "ground_truth_change": "false",
                "gallery_change": "false",
                "hypothesis_only": "true",
                "removal_authorized": "false",
                "reviewer": "Furkan",
                "final_approver": "Furkan",
            }
        )
    return rows


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3b_refinement_design_{final_dir.name}_{token}"
    if tmp.exists():
        raise RefinementDesignError(f"temp exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise RefinementDesignError(f"final root already exists: {final_dir}")
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
        raise RefinementDesignError(f"final root already exists: {final_dir}")

    f3a = validate_f3a(project_root, config)
    f3 = validate_f3(project_root, config)
    gallery_root = project_root / config["gallery_v1"]["path"]
    assert_no_path_traversal(config["gallery_v1"]["path"])
    if config["gallery_v1"].get("npy_load_forbidden") is not True:
        raise RefinementDesignError("npy_load_forbidden required")
    gallery_listing = listing_sha(gallery_root)[1]
    f3a_listing = f3a["listing_sha256"]
    f3_listing = f3["listing_sha256"]

    findings = build_manual_findings(f3a["top_false_positives"])
    generated_at = datetime.now(timezone.utc).isoformat()
    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "manual_review",
            "refinement_design",
            "validation_policy",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, tmp / "effective_configs" / Path(config_path).name)

        write_json(
            tmp / "manual_review" / "target_001_retrieval_error_manual_findings.json",
            findings,
        )
        csv_path = (
            tmp / "manual_review" / "target_001_retrieval_error_manual_findings.csv"
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FINDINGS_CSV_FIELDS))
            writer.writeheader()
            for row in build_csv_rows(findings):
                writer.writerow(row)

        write_json(
            tmp
            / "refinement_design"
            / "target_001_external_only_refinement_plan.json",
            {
                "schema_version": "reid_target_001_external_only_refinement_plan_v1",
                "applied_in_f3b": False,
                "priorities": [
                    {
                        "priority": 1,
                        "item": "external_only_same_team_hard_negative_review_package",
                        "action_authorized": False,
                    },
                    {
                        "priority": 2,
                        "item": "external_target_gallery_front_side_clean_anchor_expansion",
                        "action_authorized": False,
                    },
                    {
                        "priority": 3,
                        "item": "component_grouping_overmerge_correction_design",
                        "action_authorized": False,
                    },
                    {
                        "priority": 4,
                        "item": "target_versus_distractor_margin_scoring_contract",
                        "action_authorized": False,
                    },
                    {
                        "priority": 5,
                        "item": "view_aware_and_multi_crop_tracklet_aggregation_research",
                        "action_authorized": False,
                    },
                ],
                "branches": {
                    "A_target_gallery_view_expansion": {
                        "source": "external_enrollment_or_new_enrollment_only_video",
                        "desired_views": [
                            "clean_front",
                            "clean_front_oblique",
                            "clean_left_right_side",
                            "clean_rear_rear_oblique",
                        ],
                        "requirements": [
                            "multiple_scale_ranges",
                            "no_overlap",
                            "single_person",
                            "full_body",
                            "human_target_confirmation",
                        ],
                        "existing_seven_anchors_auto_changed": False,
                    },
                    "B_same_team_hard_negative_gallery": {
                        "source": "external_only",
                        "candidate_numbers": ["3", "7", "9", "14", "20", "25", "30"],
                        "hard_negative_is_not_target_gallery_member": True,
                        "hard_negative_is_not_target_identity": True,
                        "human_verified_non_target_required": True,
                        "automated_ocr_is_not_ground_truth": True,
                        "sample_crop_forbidden": True,
                    },
                    "C_target_versus_distractor_scoring": {
                        "candidate_formula": (
                            "target_distractor_margin = max_cosine_to_target_gallery "
                            "- max_cosine_to_human_verified_same_team_distractor_gallery"
                        ),
                        "applied_in_f3b": False,
                        "not_optimized_on_sample_f3": True,
                        "not_a_threshold": True,
                        "must_freeze_before_new_holdout": True,
                    },
                    "D_view_aware_representation": {
                        "candidate_representations": [
                            "front_prototype",
                            "side_prototype",
                            "rear_prototype",
                            "per_view_target_gallery",
                            "per_view_same_team_distractor_gallery",
                        ],
                        "view_selection_requires_human_or_validated_classifier": True,
                        "sample_result_weight_selection_forbidden": True,
                    },
                    "E_multi_crop_tracklet_aggregation": {
                        "candidates": [
                            "bounded_clean_observations",
                            "temporal_diversity",
                            "quality_gating",
                            "max_median_trimmed_aggregation",
                        ],
                        "design_on_external_refinement_source": True,
                        "freeze_before_new_independent_validation": True,
                    },
                    "F_component_purity": {
                        "future_grouping_signals": [
                            "raw_track_continuity",
                            "team_consistency",
                            "temporal_exclusivity",
                            "spatial_continuity",
                            "documented_link_audit",
                            "identity_switch_veto",
                        ]
                    },
                },
            },
        )
        write_json(
            tmp / "refinement_design" / "target_001_hard_negative_gallery_design.json",
            {
                "schema_version": "reid_target_001_hard_negative_gallery_design_v1",
                "source": "external_only",
                "sample_crops_forbidden": True,
                "candidate_visible_numbers_from_error_review": [
                    "3",
                    "7",
                    "9",
                    "14",
                    "20",
                    "25",
                    "30",
                ],
                "human_verified_non_target_required": True,
                "automated_ocr_is_not_ground_truth": True,
                "applied_in_f3b": False,
            },
        )
        write_json(
            tmp / "refinement_design" / "target_001_view_aware_gallery_design.json",
            {
                "schema_version": "reid_target_001_view_aware_gallery_design_v1",
                "existing_anchor_auto_mutation_forbidden": True,
                "desired_external_target_views": [
                    "front",
                    "front_oblique",
                    "left_side",
                    "right_side",
                    "rear",
                    "rear_oblique",
                ],
                "view_aware_prototypes_candidate": True,
                "applied_in_f3b": False,
            },
        )
        write_json(
            tmp
            / "refinement_design"
            / "target_001_component_purity_refinement_design.json",
            {
                "schema_version": "reid_target_001_component_purity_refinement_design_v1",
                "official_component_exclusion_remains_valid": True,
                "grouping_artifacts_changed_in_f3b": False,
                "overmerge_confirmed_components": [
                    c["evaluation_component_id"] for c in CONFLICT_FINDINGS
                ],
                "future_principles": findings[
                    "component_grouping_refinement_design_principles"
                ],
            },
        )
        write_json(
            tmp
            / "refinement_design"
            / "target_001_new_independent_holdout_requirements.json",
            {
                "schema_version": "reid_target_001_new_independent_holdout_requirements_v1",
                "new_independent_holdout_required": True,
                "select_after_gallery_v2_and_scoring_contract_frozen": True,
                "must_include_target_positive_and_same_team_distractor_support": True,
                "enrollment_verified_overlap_must_be_zero": True,
                "separate_from_threshold_calibration_source": True,
                "same_sample_improvement_not_independent_validation": True,
            },
        )

        write_json(
            tmp / "validation_policy" / "target_001_anti_overfit_policy.json",
            {
                "schema_version": "reid_target_001_anti_overfit_policy_v1",
                "sample_mp4_used_for_error_analysis_only": True,
                "sample_crops_forbidden_for_gallery_enrollment": True,
                "sample_negatives_forbidden_for_hard_negative_training": True,
                "sample_scores_forbidden_for_formula_weight_optimization": True,
                "gallery_v2_on_same_sample_is_development_diagnostic_only": True,
                "same_sample_improvement_cannot_be_called_independent_validation": True,
                "new_independent_holdout_required": True,
            },
        )
        write_json(
            tmp / "runtime" / "target_001_f3b_runtime_contract.json",
            {
                "schema_version": "reid_target_001_f3b_runtime_contract_v1",
                "refinement_applied": False,
                "score_recompute": False,
                "gallery_mutation": False,
                "new_embeddings": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "png": 0,
                "mp4": 0,
                "npy": 0,
            },
        )

        contract = {
            "schema_version": "reid_stage5d_f3b_refinement_design_contract_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "official_f3_outcome_unchanged": True,
            "official_f3_descriptive_outcome": (
                "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING"
            ),
            "reviewed_diagnostic_findings_frozen": True,
            "gallery_members": 7,
            "gallery_mutation": False,
            "new_embeddings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "external_only_refinement_design_ready": True,
            "refinement_applied": False,
            "new_independent_holdout_required": True,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3b_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f3b_refinement_design_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "official_f3_outcome_unchanged": True,
            "official_f3_descriptive_outcome": (
                "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING"
            ),
            "official_segment_ap": f3["summary"]["segment_metrics"]["Average_Precision"],
            "official_segment_margin": f3["summary"]["segment_metrics"][
                "separation_margin"
            ],
            "reviewed_diagnostic_findings_frozen": True,
            "high_ranked_positives": 3,
            "low_ranked_positives": 5,
            "top_false_positives": 24,
            "visible_jersey_false_positives": len(VISIBLE_JERSEY),
            "conflicting_components_overmerge": 4,
            "gallery_members": 7,
            "gallery_mutation": False,
            "new_embeddings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "score_recompute": False,
            "external_only_refinement_design_ready": True,
            "refinement_applied": False,
            "new_independent_holdout_required": True,
            "primary_root_causes": findings["root_cause_summary"]["primary_causes"],
            "f3a_snapshot_sha256": f3a["snapshot_sha256"],
            "sample_sha256": config["evaluation_source"]["expected_sha256"],
            "network_used": False,
            "package_environment_changed": False,
            "artifact_budget": {"csv": 1, "png": 0, "mp4": 0, "npy": 0},
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3b_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")):
            raise RefinementDesignError("artifact budget violated")
        if len(list(tmp.rglob("*.csv"))) != 1:
            raise RefinementDesignError("csv budget != 1")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f3b_refinement_design_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3b_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3b_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3b_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    if listing_sha(f3a["root"])[1] != f3a_listing:
        raise RefinementDesignError("F3A mutated")
    if listing_sha(f3["root"])[1] != f3_listing:
        raise RefinementDesignError("F3 mutated")
    if listing_sha(gallery_root)[1] != gallery_listing:
        raise RefinementDesignError("gallery mutated")

    return load_json(final_dir / "stage5d_f3b_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze target_001 retrieval-error findings and refinement design."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to retrieval_error_manual_review_and_refinement_design_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except RefinementDesignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"fp={summary['top_false_positives']} "
        f"visible_jersey={summary['visible_jersey_false_positives']} "
        f"overmerge={summary['conflicting_components_overmerge']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
