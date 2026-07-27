#!/usr/bin/env python3
"""Stage 5D-F3N — Holdout v2 frozen-result audit and visual error-analysis package.

Read-only visualization of F3M scores/ranks/metrics. No rescoring, embeddings,
gallery mutation, threshold, identity, automatic root-cause, or deletion.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMA = "reid_independent_holdout_v2_result_audit_config_v1"
TARGET_ID = "target_001"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3N_TARGET_001_INDEPENDENT_HOLDOUT_RESULT_AUDIT_PACKAGE_READY"
)
READINESS = "TARGET_001_HOLDOUT_V2_ERROR_ANALYSIS_READY_FOR_HUMAN_INTERPRETATION"
NEXT_GATE = (
    "STAGE5D-F3O_TARGET_001_INDEPENDENT_HOLDOUT_MANUAL_ERROR_INTERPRETATION_AND_BASELINE_CLOSURE"
)
FFMPEG_BIN = "/usr/bin/ffmpeg"

ALLOWED_DIRTY = {
    "scripts/run_reid_independent_holdout_v2_result_audit_and_error_analysis.py",
    "configs/reid/independent_holdout_v2_result_audit_stage5d_target_001.yaml",
    "tests/test_reid_independent_holdout_v2_result_audit_and_error_analysis.py",
    "docs/setup/stage5d-target-independent-holdout-result-audit-and-error-analysis-package.md",
}

POSITIVE_IDS = (
    "H2_GT_REVIEW_000010",
    "H2_GT_REVIEW_000030",
    "H2_GT_REVIEW_000061",
    "H2_GT_REVIEW_000090",
    "H2_GT_REVIEW_000094",
    "H2_GT_REVIEW_000104",
    "H2_GT_REVIEW_000124",
    "H2_GT_REVIEW_000129",
    "H2_GT_REVIEW_000135",
    "H2_GT_REVIEW_000136",
)

MANUAL_TEMPLATE_FIELDS = [
    "audit_item_id",
    "cohort",
    "query_id",
    "segment_id",
    "primary_rank",
    "frozen_gt_decision",
    "same_team_cohort",
    "T_max",
    "T_max_member_id",
    "D_max",
    "D_max_member_id",
    "S_primary",
    "query_crop_path",
    "best_target_crop_path",
    "best_distractor_crop_path",
    "manual_query_crop_quality",
    "manual_query_blur_observed",
    "manual_query_partial_body",
    "manual_query_occlusion",
    "manual_multi_person_contamination",
    "manual_track_impurity_observed",
    "manual_view_category",
    "manual_scale_category",
    "manual_same_uniform_confusion",
    "manual_target_gallery_view_gap",
    "manual_distractor_over_penalty_suspected",
    "manual_best_target_match_visually_plausible",
    "manual_best_distractor_match_visually_plausible",
    "manual_ground_truth_recheck_required",
    "manual_primary_failure_category",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
]

BLANK_MANUAL_FIELDS = [
    f for f in MANUAL_TEMPLATE_FIELDS
    if f.startswith("manual_") or f in {"reviewer", "final_approver", "reviewed_at"}
]

ALLOWED_FAILURE_CATEGORIES = (
    "query_blur_or_low_resolution",
    "rear_or_side_view_gap",
    "partial_body_or_edge_clipping",
    "occlusion_or_multi_person_contamination",
    "track_impurity_or_identity_switch",
    "same_uniform_confusion",
    "target_gallery_view_gap",
    "distractor_gallery_over_penalty",
    "misleading_target_member_match",
    "misleading_distractor_member_match",
    "ground_truth_recheck_required",
    "no_clear_visual_cause",
    "multiple_contributing_causes",
    "not_applicable_successful_positive",
    "other",
)


class ResultAuditError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assert_no_path_traversal(value: str) -> None:
    if ".." in value.replace("\\", "/").split("/"):
        raise ResultAuditError(f"path traversal rejected: {value}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != CONFIG_SCHEMA:
        raise ResultAuditError("config schema mismatch")
    return cfg


def run_git(args: Sequence[str], cwd: Path = _PROJECT_ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def validate_git_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = run_git(["rev-parse", "HEAD"])
    origin = run_git(["rev-parse", "origin/main"])
    msg = run_git(["log", "-1", "--format=%s"])
    status = run_git(["status", "--porcelain"])
    if branch != "main":
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_GIT_CONTRACT_MISMATCH branch")
    if head != cfg["project_head_expected"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_GIT_CONTRACT_MISMATCH HEAD")
    if head != origin:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_GIT_CONTRACT_MISMATCH origin")
    if msg != cfg["project_head_message_expected"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_GIT_CONTRACT_MISMATCH message")
    dirty = []
    for ln in status.splitlines():
        if not ln.strip():
            continue
        path = ln[3:] if len(ln) >= 3 else ln
        path = path.strip()
        if path not in ALLOWED_DIRTY:
            raise ResultAuditError(f"BLOCKED_STAGE5D_F3N_GIT_CONTRACT_MISMATCH dirty:{path}")
        dirty.append(path)
    return {"branch": branch, "head": head, "origin_main": origin, "message": msg, "dirty": dirty}


def validate_snapshot(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResultAuditError(f"snapshot missing: {path}")
    actual = sha256_file(path)
    side = Path(str(path) + ".sha256").read_text(encoding="utf-8").split()[0]
    man_path = path.with_name(path.name.replace(".tar.gz", "_manifest.json"))
    man = read_json(man_path)
    man_sha = man.get("sha256") or man.get("archive_sha256")
    if not (actual == side == man_sha == expected_sha):
        raise ResultAuditError(f"snapshot sha mismatch {path.name}")
    listing = path.with_name(path.name.replace(".tar.gz", "_listing.txt"))
    return {"path": str(path), "sha256": actual, "listing_exists": listing.is_file()}


def nearly(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def validate_f3m(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3m_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3m_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH status")
    if summary.get("readiness") != pack["expected_readiness"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH readiness")
    if summary.get("performance_outcome") != pack["expected_recorded_outcome"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH outcome")

    seal = read_json(root / "pre_execution" / "target_001_holdout_v2_pre_ground_truth_join_execution_seal.json")
    if seal.get("gt_labels_read_so_far") != 0:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH seal_gt")

    # Verify sealed SHAs still match files
    checks = {
        "projection": (
            root / "pre_execution" / "target_001_holdout_v2_scoreable_query_input_projection.jsonl",
            seal["projection_sha256"],
        ),
        "query_emb": (
            root / "query_embeddings" / "target_001_holdout_v2_query_embeddings.npy",
            seal["query_embedding_sha256"],
        ),
        "target_cos": (
            root / "scoring" / "target_001_holdout_v2_target_cosine.npy",
            seal["target_similarity_sha256"],
        ),
        "distractor_cos": (
            root / "scoring" / "target_001_holdout_v2_distractor_cosine.npy",
            seal["distractor_similarity_sha256"],
        ),
        "primary": (
            root / "scoring" / "target_001_holdout_v2_primary_scores_label_blind.jsonl",
            seal["primary_score_sha256"],
        ),
        "secondary": (
            root / "scoring" / "target_001_holdout_v2_secondary_scores_label_blind.jsonl",
            seal["secondary_score_sha256"],
        ),
        "ranking": (
            root / "ranking" / "target_001_holdout_v2_primary_ranking_label_blind.jsonl",
            seal["ranking_sha256"],
        ),
    }
    for name, (path, expected) in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ResultAuditError(
                f"BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH {name}_sha"
            )

    Q = np.load(checks["query_emb"][0])
    Ct = np.load(checks["target_cos"][0])
    Cd = np.load(checks["distractor_cos"][0])
    if Q.shape != (115, 512) or Ct.shape != (115, 13) or Cd.shape != (115, 23):
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH shapes")

    primary = read_jsonl(checks["primary"][0])
    ranking = read_jsonl(checks["ranking"][0])
    secondary = read_jsonl(checks["secondary"][0])
    joined = read_jsonl(root / "evaluation" / "target_001_holdout_v2_scored_ground_truth_join.jsonl")
    if len(primary) != 115 or len(ranking) != 115 or len(joined) != 115:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH rows")

    det = read_json(root / "query_embeddings" / "target_001_holdout_v2_query_embedding_determinism.json")
    if not det.get("exact_match") or float(det.get("overall_max_absolute_difference", 1)) != 0.0:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH determinism")

    seg = read_json(root / "evaluation" / "target_001_holdout_v2_segment_primary_metrics.json")
    comp = read_json(root / "evaluation" / "target_001_holdout_v2_component_primary_metrics.json")
    same = read_json(root / "evaluation" / "target_001_holdout_v2_same_team_specific_metrics.json")
    outcome = read_json(root / "evaluation" / "target_001_holdout_v2_frozen_outcome_evaluation.json")
    secondary_metrics = read_json(
        root / "evaluation" / "target_001_holdout_v2_secondary_diagnostic_metrics.json"
    )
    exp = pack["expected_metrics"]
    metric_ok = (
        nearly(seg["Recall@1"], exp["Recall@1"])
        and nearly(seg["Recall@3"], exp["Recall@3"])
        and nearly(seg["Recall@5"], exp["Recall@5"])
        and nearly(seg["Recall@10"], exp["Recall@10"])
        and nearly(seg["MRR"], exp["MRR"])
        and nearly(seg["Average_Precision"], exp["AP"])
        and nearly(seg["AUROC"], exp["AUROC"])
        and nearly(seg["AUPRC"], exp["AUPRC"])
        and nearly(seg["min_positive_minus_max_negative_margin"], exp["segment_margin"])
        and nearly(comp["separation_margin_min_pos_minus_max_neg"], exp["component_margin"])
        and nearly(same["AUROC"], exp["same_team_AUROC"])
        and nearly(same["Average_Precision"], exp["same_team_AP"])
        and nearly(
            seg["positive_median_minus_negative_median"],
            exp["positive_median_minus_negative_median"],
            tol=1e-6,
        )
        and nearly(
            seg["positive_median_minus_same_team_negative_median"],
            exp["positive_median_minus_same_team_negative_median"],
            tol=1e-6,
        )
    )
    if not metric_ok:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH metrics")
    if summary.get("positive_ranks") != pack["expected_positive_ranks"]:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH ranks")
    if summary.get("complete_universe") != 243 or summary.get("scoreable") != 115:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH counts")
    if (
        summary.get("positive") != 10
        or summary.get("negative_player") != 105
        or summary.get("same_team_negative") != 55
        or summary.get("other_team_negative") != 50
        or summary.get("excluded") != 128
    ):
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH cohorts")
    if summary.get("threshold_selected") is not False or summary.get("identity_assignments") != 0:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH policy")
    if summary.get("gallery_mutation") is not False or summary.get("holdout_enrollment") is not False:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH gallery")

    pos_audit = read_jsonl(root / "diagnostics" / "target_001_holdout_v2_positive_score_audit.jsonl")
    stfp = read_jsonl(root / "diagnostics" / "target_001_holdout_v2_top_same_team_negative_audit.jsonl")
    otfp = read_jsonl(root / "diagnostics" / "target_001_holdout_v2_top_other_team_negative_audit.jsonl")
    if len(pos_audit) != 10 or len(stfp) != 20 or len(otfp) != 20:
        raise ResultAuditError("BLOCKED_STAGE5D_F3N_FROZEN_RESULT_CONTRACT_MISMATCH diag")

    joined_sha = sha256_file(root / "evaluation" / "target_001_holdout_v2_scored_ground_truth_join.jsonl")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": root,
        "summary": summary,
        "seal": seal,
        "primary": primary,
        "ranking": ranking,
        "secondary": secondary,
        "joined": joined,
        "joined_sha256": joined_sha,
        "seg": seg,
        "comp": comp,
        "same": same,
        "outcome": outcome,
        "secondary_metrics": secondary_metrics,
        "pos_audit": pos_audit,
        "stfp": stfp,
        "otfp": otfp,
        "metric_shas": {
            "segment": sha256_file(root / "evaluation" / "target_001_holdout_v2_segment_primary_metrics.json"),
            "component": sha256_file(root / "evaluation" / "target_001_holdout_v2_component_primary_metrics.json"),
            "same_team": sha256_file(root / "evaluation" / "target_001_holdout_v2_same_team_specific_metrics.json"),
            "outcome": sha256_file(root / "evaluation" / "target_001_holdout_v2_frozen_outcome_evaluation.json"),
            "secondary": sha256_file(
                root / "evaluation" / "target_001_holdout_v2_secondary_diagnostic_metrics.json"
            ),
            "joined": joined_sha,
        },
        "snapshot": snap,
        "scores_recomputed": False,
        "embeddings_generated": 0,
    }


def validate_f3l(cfg: Mapping[str, Any], f3m_joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pack = cfg["stage5d_f3l_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3l_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise ResultAuditError("F3L status mismatch")
    if summary.get("component_policy") != pack["expected_component_policy"]:
        raise ResultAuditError("F3L component policy mismatch")
    decisions = {}
    with open(root / "manual_freeze" / "target_001_holdout_v2_ground_truth_decisions_frozen.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            decisions[row["review_item_id"]] = row
    if len(decisions) != 141:
        raise ResultAuditError("F3L reviewed != 141")
    pos = [rid for rid, r in decisions.items() if r["manual_ground_truth_decision"] == "target_occurrence_yes"]
    if sorted(pos) != sorted(POSITIVE_IDS):
        raise ResultAuditError("F3L positive IDs mismatch")
    for row in f3m_joined:
        qid = row["stable_query_id"]
        d = decisions[qid]
        if d["manual_ground_truth_decision"] != row["manual_ground_truth_decision"]:
            raise ResultAuditError(f"F3M/F3L GT mismatch {qid}")
        if row["binary_clean_player_label"] == 1 and d["manual_ground_truth_decision"] != "target_occurrence_yes":
            raise ResultAuditError(f"positive label mismatch {qid}")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {"root": root, "summary": summary, "decisions": decisions, "snapshot": snap}


def validate_f3k(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3k_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3k_summary.json")
    if summary.get("representative_crop_count") != pack["expected_representative_crop_count"]:
        raise ResultAuditError("F3K crop count mismatch")
    manifest = read_jsonl(root / "crops" / "target_001_holdout_v2_gt_review_crop_manifest.jsonl")
    if len(manifest) != pack["expected_crop_manifest_rows"]:
        raise ResultAuditError("F3K manifest rows mismatch")
    by_id = {}
    for row in manifest:
        assert_no_path_traversal(row["crop_path"])
        path = root / row["crop_path"]
        if not path.is_file():
            raise ResultAuditError(f"F3K crop missing {row['crop_path']}")
        if sha256_file(path) != row["crop_sha256"]:
            raise ResultAuditError(f"F3K crop sha mismatch {row['review_item_id']}")
        if row["holdout_sha256"] != pack["expected_holdout_sha256"]:
            raise ResultAuditError("F3K holdout sha mismatch")
        by_id[row["review_item_id"]] = {**row, "crop_path_absolute": str(path)}
    inv = read_jsonl(root / "inventory" / "target_001_holdout_v2_gt_review_item_inventory.jsonl")
    inv_by_id = {r["review_item_id"]: r for r in inv}
    videos = list((root / "videos").glob("*.mp4")) if (root / "videos").is_dir() else []
    video_count = summary.get("review_videos", summary.get("diagnostic_video_count", len(videos)))
    if len(videos) != pack["expected_diagnostic_video_count"] or video_count != pack[
        "expected_diagnostic_video_count"
    ]:
        raise ResultAuditError(f"F3K video count mismatch got files={len(videos)} summary={video_count}")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": root,
        "summary": summary,
        "crops_by_id": by_id,
        "inventory_by_id": inv_by_id,
        "snapshot": snap,
        "crop_shas": {k: v["crop_sha256"] for k, v in by_id.items()},
    }


def validate_f3g(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3g_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3g_summary.json")
    t_path = root / "target_gallery_v2" / "target_001_gallery_v2_individual_embeddings.npy"
    d_path = root / "distractor_gallery_v1" / "target_001_same_team_distractor_individual_embeddings.npy"
    t_sha = sha256_file(t_path)
    d_sha = sha256_file(d_path)
    if t_sha != pack["expected_target_embedding_sha256"]:
        raise ResultAuditError("F3G target sha mismatch")
    if d_sha != pack["expected_distractor_embedding_sha256"]:
        raise ResultAuditError("F3G distractor sha mismatch")
    t_inv = read_jsonl(root / "target_gallery_v2" / "target_001_gallery_v2_member_inventory.jsonl")
    d_inv = read_jsonl(root / "distractor_gallery_v1" / "target_001_same_team_distractor_member_inventory.jsonl")
    if len(t_inv) != 13 or len(d_inv) != 23:
        raise ResultAuditError("F3G member count mismatch")
    package_roots = [_PROJECT_ROOT / p for p in cfg["gallery_crop_package_roots"]]
    members: dict[str, dict[str, Any]] = {}

    def resolve_member(row: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
        rel = row["crop_path"]
        assert_no_path_traversal(rel)
        found = None
        for prow in package_roots:
            cand = prow / rel
            if cand.is_file():
                found = cand
                break
        if found is None:
            raise ResultAuditError(f"gallery crop missing: {row['member_id']} {rel}")
        actual = sha256_file(found)
        if actual != row["crop_sha256"]:
            raise ResultAuditError(f"gallery crop sha mismatch: {row['member_id']}")
        if kind == "target":
            member_type = (
                "frozen_anchor_v1"
                if row.get("gallery_v1_member") or row.get("reused_from_gallery_v1")
                else "external_refinement_anchor_v2"
            )
        else:
            member_type = "same_team_distractor_v1"
        return {
            "member_id": row["member_id"],
            "crop_path": rel,
            "crop_path_absolute": str(found),
            "crop_sha256": actual,
            "member_type": member_type,
            "human_visible_jersey_number": row.get("human_visible_jersey_number"),
            "source_occurrence_code": row.get("source_occurrence_code") or row.get("source_external_code"),
            "kind": kind,
        }

    for row in t_inv:
        members[row["member_id"]] = resolve_member(row, kind="target")
    for row in d_inv:
        members[row["member_id"]] = resolve_member(row, kind="distractor")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": root,
        "summary": summary,
        "target_ids": [r["member_id"] for r in t_inv],
        "distractor_ids": [r["member_id"] for r in d_inv],
        "members": members,
        "shas": {"target": t_sha, "distractor": d_sha},
        "snapshot": snap,
        "holdout_member_count": 0,
    }


def load_f3j_observations(cfg: Mapping[str, Any], segment_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    root = _PROJECT_ROOT / cfg["stage5d_f3j_package"]["path"]
    path = root / cfg["stage5d_f3j_package"]["observations_rel"]
    by_seg: dict[str, list[dict[str, Any]]] = {sid: [] for sid in segment_ids}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = row["segment_id"]
            if sid in by_seg:
                by_seg[sid].append(row)
    for sid, rows in by_seg.items():
        if not rows:
            raise ResultAuditError(f"F3J observations missing for {sid}")
        rows.sort(key=lambda r: int(r["frame_index"]))
    return by_seg


def build_audit_items(
    *,
    f3m: Mapping[str, Any],
    f3l: Mapping[str, Any],
    f3k: Mapping[str, Any],
    f3g: Mapping[str, Any],
) -> list[dict[str, Any]]:
    joined_by_id = {r["stable_query_id"]: r for r in f3m["joined"]}
    ranking_by_id = {r["stable_query_id"]: r for r in f3m["ranking"]}
    secondary_by_id = {r["stable_query_id"]: r for r in f3m["secondary"]}
    primary_by_id = {r["stable_query_id"]: r for r in f3m["primary"]}

    def enrich(qid: str, cohort: str, audit_id: str) -> dict[str, Any]:
        j = joined_by_id[qid]
        rk = ranking_by_id[qid]
        sec = secondary_by_id[qid]
        prim = primary_by_id[qid]
        crop = f3k["crops_by_id"][qid]
        inv = f3k["inventory_by_id"][qid]
        dec = f3l["decisions"][qid]
        # frozen score fields must match across artifacts
        if float(j["S_primary"]) != float(rk["S_primary"]) or int(j["rank"]) != int(rk["rank"]):
            raise ResultAuditError(f"rank/score mismatch {qid}")
        if float(j["T_max"]) != float(prim["T_max"]) or float(j["D_max"]) != float(prim["D_max"]):
            raise ResultAuditError(f"T/D mismatch {qid}")
        t_member = j.get("T_max_member_id") or j.get("best_target_member")
        d_member = j.get("D_max_member_id") or j.get("best_distractor_member")
        if t_member not in f3g["members"] or d_member not in f3g["members"]:
            raise ResultAuditError(f"gallery member missing for {qid}")
        t_info = f3g["members"][t_member]
        d_info = f3g["members"][d_member]
        same_team = dec.get("manual_same_team_as_target") == "yes"
        if cohort == "positive":
            if dec["manual_ground_truth_decision"] != "target_occurrence_yes":
                raise ResultAuditError(f"positive cohort GT mismatch {qid}")
        elif cohort == "same_team_fp":
            if dec["manual_ground_truth_decision"] != "target_occurrence_no" or not same_team:
                raise ResultAuditError(f"same-team FP GT mismatch {qid}")
        elif cohort == "other_team_fp":
            if dec["manual_ground_truth_decision"] != "target_occurrence_no" or same_team:
                raise ResultAuditError(f"other-team FP GT mismatch {qid}")
        return {
            "audit_item_id": audit_id,
            "cohort": cohort,
            "query_id": qid,
            "stable_query_id": qid,
            "review_item_id": qid,
            "segment_id": j["segment_id"],
            "component_id": j["component_id"],
            "primary_rank": int(j["rank"]),
            "frozen_gt_decision": dec["manual_ground_truth_decision"],
            "same_team_cohort": same_team,
            "representative_crop_path": crop["crop_path"],
            "representative_crop_path_absolute": crop["crop_path_absolute"],
            "representative_crop_sha256": crop["crop_sha256"],
            "source_frame_index": crop["source_frame_index"],
            "source_bbox_xyxy": crop["source_bbox_xyxy"],
            "canonical_crop_bbox_xyxy": crop["canonical_crop_bbox_xyxy"],
            "crop_width": crop["crop_width"],
            "crop_height": crop["crop_height"],
            "T_max": float(j["T_max"]),
            "T_max_member_id": t_member,
            "D_max": float(j["D_max"]),
            "D_max_member_id": d_member,
            "S_primary": float(j["S_primary"]),
            "S_top3_margin": float(sec["S_top3_margin"]),
            "S_target_centroid_margin": float(sec["S_target_centroid_margin"]),
            "S_target_medoid_margin": float(sec["S_target_medoid_margin"]),
            "S_mean_margin": float(sec["S_mean_margin"]),
            "best_target_crop_path": t_info["crop_path"],
            "best_target_crop_path_absolute": t_info["crop_path_absolute"],
            "best_target_crop_sha256": t_info["crop_sha256"],
            "best_target_member_type": t_info["member_type"],
            "best_distractor_crop_path": d_info["crop_path"],
            "best_distractor_crop_path_absolute": d_info["crop_path_absolute"],
            "best_distractor_crop_sha256": d_info["crop_sha256"],
            "best_distractor_jersey": d_info.get("human_visible_jersey_number"),
            "observation_count": inv["observation_count"],
            "start_frame": inv["start_frame"],
            "end_frame": inv["end_frame"],
            "start_time": inv["start_time"],
            "end_time": inv["end_time"],
            "raw_track_code": inv["raw_track_code"],
            "context_start_frame": inv["context_start_frame"],
            "context_middle_frame": inv["context_middle_frame"],
            "context_end_frame": inv["context_end_frame"],
            "selected_representative_frame": inv["selected_representative_frame"],
            "automatic_diagnosis_absent": True,
            "human_diagnosis_pending": True,
        }

    # Positive: rank ascending, then query ID
    pos_rows = sorted(
        f3m["pos_audit"],
        key=lambda r: (int(r["rank"]), str(r["stable_query_id"])),
    )
    if len(pos_rows) != 10:
        raise ResultAuditError("positive audit != 10")
    if set(r["stable_query_id"] for r in pos_rows) != set(POSITIVE_IDS):
        raise ResultAuditError("positive audit IDs mismatch")

    # Same-team / other-team: exact F3M diagnostic order (already top-20 by score)
    stfp_rows = list(f3m["stfp"])
    otfp_rows = list(f3m["otfp"])
    if len(stfp_rows) != 20 or len(otfp_rows) != 20:
        raise ResultAuditError("FP audit sizes")

    items: list[dict[str, Any]] = []
    for i, row in enumerate(pos_rows, start=1):
        items.append(enrich(row["stable_query_id"], "positive", f"F3N_POS_{i:03d}"))
    for i, row in enumerate(stfp_rows, start=1):
        items.append(enrich(row["stable_query_id"], "same_team_fp", f"F3N_STFP_{i:03d}"))
    for i, row in enumerate(otfp_rows, start=1):
        items.append(enrich(row["stable_query_id"], "other_team_fp", f"F3N_OTFP_{i:03d}"))

    ids = [it["query_id"] for it in items]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ResultAuditError("audit cohort duplicate/missing")
    return items


def compute_clip_window(
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
    total_frames: int,
    fps: float,
    pre_pad_sec: float,
    post_pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
) -> tuple[int, int]:
    pre = int(round(pre_pad_sec * fps))
    post = int(round(post_pad_sec * fps))
    min_frames = max(1, int(round(min_clip_sec * fps)))
    max_frames = max(min_frames, int(round(max_clip_sec * fps)))
    clip_start = max(0, int(start_frame) - pre)
    clip_end = min(total_frames - 1, int(end_frame) + post)
    span = clip_end - clip_start + 1
    if span > max_frames:
        half = max_frames // 2
        clip_start = max(0, int(representative_frame) - half)
        clip_end = min(total_frames - 1, clip_start + max_frames - 1)
        if clip_end - clip_start + 1 < max_frames:
            clip_start = max(0, clip_end - max_frames + 1)
    span = clip_end - clip_start + 1
    if span < min_frames:
        need = min_frames - span
        before = need // 2
        after = need - before
        clip_start = max(0, clip_start - before)
        clip_end = min(total_frames - 1, clip_end + after)
    return clip_start, clip_end


def decode_frames(video_path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    needed = sorted(set(int(i) for i in frame_indices))
    if not needed:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ResultAuditError("holdout open failed")
    out: dict[int, np.ndarray] = {}
    try:
        for fi in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ResultAuditError(f"failed to decode frame {fi}")
            out[fi] = frame
    finally:
        cap.release()
    return out


def _fit_bgr(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_watermark(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    cv2.putText(frame, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


def annotate_context(frame: np.ndarray, bbox: Sequence[float], role: str, frame_index: int) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
    cv2.putText(out, f"{role} f={frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(out, f"{role} f={frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def render_audit_tile(
    item: Mapping[str, Any],
    *,
    query_bgr: np.ndarray,
    target_bgr: np.ndarray,
    distractor_bgr: np.ndarray,
    contexts: Sequence[tuple[str, np.ndarray]],
    tile_w: int,
    tile_h: int,
    watermark: str,
) -> np.ndarray:
    tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
    header = [
        f"{item['audit_item_id']}  {item['cohort']}",
        f"query={item['query_id']}  seg={item['segment_id']}",
        f"GT={item['frozen_gt_decision']}  same_team={item['same_team_cohort']}",
        f"rank={item['primary_rank']}  S={item['S_primary']:.4f}  Tmax={item['T_max']:.4f}  Dmax={item['D_max']:.4f}",
        f"bestT={item['T_max_member_id']}",
        f"bestD={item['D_max_member_id']}",
    ]
    y = 22
    for text in header:
        cv2.putText(tile, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        y += 16

    panel_w = (tile_w - 40) // 3
    panel_h = int(tile_h * 0.42)
    panels = [
        ("QUERY", query_bgr),
        (f"TARGET {item['T_max']:.3f}", target_bgr),
        (f"DISTRACTOR {item['D_max']:.3f}", distractor_bgr),
    ]
    for i, (label, img) in enumerate(panels):
        disp = _fit_bgr(img, panel_w - 8, panel_h - 24)
        dh, dw = disp.shape[:2]
        ox = 12 + i * (panel_w + 8) + (panel_w - dw) // 2
        oy = 120
        tile[oy : oy + dh, ox : ox + dw] = disp
        cv2.putText(
            tile, label, (12 + i * (panel_w + 8), oy - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 120), 1, cv2.LINE_AA,
        )

    ctx_y = 120 + panel_h + 10
    ctx_slot_w = (tile_w - 48) // 3
    ctx_slot_h = tile_h - ctx_y - 36
    for i, (role, img) in enumerate(contexts[:3]):
        disp = _fit_bgr(img, ctx_slot_w - 8, ctx_slot_h - 8)
        dh, dw = disp.shape[:2]
        cx = 16 + i * ctx_slot_w + (ctx_slot_w - dw) // 2
        cy = ctx_y + (ctx_slot_h - dh) // 2
        tile[cy : cy + dh, cx : cx + dw] = disp
        cv2.putText(
            tile, role, (16 + i * ctx_slot_w + 4, ctx_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1, cv2.LINE_AA,
        )
    draw_watermark(tile, watermark)
    return tile


def render_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    title: str,
    min_width: int,
    cols: int,
    watermark: str,
) -> np.ndarray:
    n = len(items)
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w = max(1800, int(math.ceil(min_width / cols)))
    tile_h = 980
    header_h = 90
    width = max(min_width, cols * tile_w)
    height = header_h + rows_n * tile_h
    sheet = np.full((height, width, 3), 14, dtype=np.uint8)
    for i, text in enumerate([
        title,
        "FROZEN INDEPENDENT RESULT AUDIT — NO RESCORING — NO GALLERY CHANGE",
        f"items={n}",
    ]):
        cv2.putText(sheet, text, (16, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    for i, item in enumerate(items):
        r, c = divmod(i, cols)
        tile = render_audit_tile(
            item,
            query_bgr=item["query_bgr"],
            target_bgr=item["target_bgr"],
            distractor_bgr=item["distractor_bgr"],
            contexts=item["context_panels"],
            tile_w=tile_w,
            tile_h=tile_h,
            watermark=watermark,
        )
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


class FfmpegRawWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: float, crf: int, preset: str) -> None:
        if not Path(FFMPEG_BIN).is_file():
            raise ResultAuditError("ffmpeg missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frame_count = 0
        cmd = [
            FFMPEG_BIN, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-map_metadata", "-1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(crf), "-preset", preset,
            "-x264-params", "bframes=0:keyint=30:scenecut=0", str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.proc.stdin is None:
            raise ResultAuditError("ffmpeg stdin unavailable")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ResultAuditError("ffmpeg frame size mismatch")
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
            raise ResultAuditError(f"ffmpeg failed rc={rc}: {err[:500]}")


def draw_title_card(width: int, height: int, item: Mapping[str, Any]) -> np.ndarray:
    card = np.full((height, width, 3), 24, dtype=np.uint8)
    lines = [
        str(item["audit_item_id"]),
        str(item["query_id"]),
        f"cohort={item['cohort']}  GT={item['frozen_gt_decision']}",
        f"rank={item['primary_rank']}  S={item['S_primary']:.4f}",
        f"Tmax={item['T_max']:.4f} ({item['T_max_member_id']})",
        f"Dmax={item['D_max']:.4f} ({item['D_max_member_id']})",
        "HUMAN ROOT-CAUSE PENDING",
    ]
    y = 70
    for text in lines:
        cv2.putText(card, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        y += 40
    return card


def annotate_video_frame(
    frame: np.ndarray,
    *,
    item: Mapping[str, Any],
    frame_index: int,
    bbox: Optional[Sequence[float]],
    watermark: str,
) -> np.ndarray:
    out = frame.copy()
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
    lines = [
        f"{item['audit_item_id']}  {item['query_id']}",
        f"GT={item['frozen_gt_decision']}  rank={item['primary_rank']}",
        f"Tmax={item['T_max']:.4f}  Dmax={item['D_max']:.4f}  S={item['S_primary']:.4f}",
        f"bestT={item['T_max_member_id']}",
        f"bestD={item['D_max_member_id']}",
        f"frame {frame_index}",
    ]
    y = 28
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22
    draw_watermark(out, watermark)
    return out


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3n_result_audit_{final_dir.name}_{token}"
    if tmp.exists():
        raise ResultAuditError(f"temp exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ResultAuditError(f"final root already exists: {final_dir}")
    os.rename(tmp, final_dir)


def list_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file())


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    final_dir = _PROJECT_ROOT / cfg["output"]["final_dir"]
    if final_dir.exists():
        raise ResultAuditError(f"final root already exists: {final_dir}")

    access = {
        "holdout_mp4_decode_frames": 0,
        "sample_video_reads": 0,
        "external_enrollment_video_reads": 0,
        "detection_inference": 0,
        "tracking": 0,
        "segmentation": 0,
        "crop_generation": 0,
        "crop_copies": 0,
        "new_embeddings": 0,
        "similarity_rows_recomputed": 0,
        "score_rows_recomputed": 0,
        "metric_rows_recomputed": 0,
        "ocr_calls": 0,
        "team_classifier_calls": 0,
        "threshold_selection": 0,
        "identity_assignments": 0,
        "gallery_mutation": 0,
        "holdout_enrollment": 0,
        "hard_negative_mining": 0,
        "automatic_root_cause_decisions": 0,
        "upstream_deletions": 0,
        "deleted_files": 0,
        "query_crop_reads": 0,
        "gallery_crop_reads": 0,
    }

    git_info = validate_git_contract(cfg)
    f3m = validate_f3m(cfg)
    f3l = validate_f3l(cfg, f3m["joined"])
    f3k = validate_f3k(cfg)
    f3g = validate_f3g(cfg)

    # holdout SHA before
    holdout = _PROJECT_ROOT / cfg["holdout_source"]["path"]
    holdout_sha_before = sha256_file(holdout)
    if holdout_sha_before != cfg["holdout_source"]["expected_sha256"]:
        raise ResultAuditError("holdout sha mismatch")

    items = build_audit_items(f3m=f3m, f3l=f3l, f3k=f3k, f3g=f3g)
    segment_ids = {it["segment_id"] for it in items}
    obs_by_seg = load_f3j_observations(cfg, segment_ids)

    # Probe video size
    cap = cv2.VideoCapture(str(holdout))
    if not cap.isOpened():
        raise ResultAuditError("holdout open failed")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    vcfg = cfg["diagnostic_videos"]
    needed_frames: set[int] = set()
    for it in items:
        for key in ("context_start_frame", "context_middle_frame", "context_end_frame", "selected_representative_frame"):
            needed_frames.add(int(it[key]))
        clip_s, clip_e = compute_clip_window(
            start_frame=int(it["start_frame"]),
            end_frame=int(it["end_frame"]),
            representative_frame=int(it["selected_representative_frame"]),
            total_frames=total_frames,
            fps=fps,
            pre_pad_sec=float(vcfg["pre_pad_sec"]),
            post_pad_sec=float(vcfg["post_pad_sec"]),
            min_clip_sec=float(vcfg["min_clip_sec"]),
            max_clip_sec=float(vcfg["max_clip_sec"]),
        )
        it["clip_start_frame"] = clip_s
        it["clip_end_frame"] = clip_e
        for fi in range(clip_s, clip_e + 1):
            needed_frames.add(fi)

    frame_cache = decode_frames(holdout, sorted(needed_frames))
    access["holdout_mp4_decode_frames"] = len(frame_cache)

    # Attach images (read existing crops; no copies written as standalone crop files)
    for it in items:
        q = cv2.imread(it["representative_crop_path_absolute"], cv2.IMREAD_COLOR)
        t = cv2.imread(it["best_target_crop_path_absolute"], cv2.IMREAD_COLOR)
        d = cv2.imread(it["best_distractor_crop_path_absolute"], cv2.IMREAD_COLOR)
        if q is None or t is None or d is None:
            raise ResultAuditError(f"crop decode failed {it['query_id']}")
        access["query_crop_reads"] += 1
        access["gallery_crop_reads"] += 2
        it["query_bgr"] = q
        it["target_bgr"] = t
        it["distractor_bgr"] = d
        bbox_by_frame = {int(o["frame_index"]): o["bbox_xyxy"] for o in obs_by_seg[it["segment_id"]]}
        it["bbox_by_frame"] = bbox_by_frame
        contexts = []
        for role, key in (("START", "context_start_frame"), ("MIDDLE", "context_middle_frame"), ("END", "context_end_frame")):
            fi = int(it[key])
            frame = frame_cache[fi]
            bbox = bbox_by_frame.get(fi)
            if bbox is None:
                # nearest observation frame
                obs_frames = sorted(bbox_by_frame)
                nearest = min(obs_frames, key=lambda x: (abs(x - fi), x))
                fi = nearest
                frame = frame_cache.get(fi) or decode_frames(holdout, [fi])[fi]
                access["holdout_mp4_decode_frames"] = len(set(list(frame_cache) + [fi]))
                frame_cache[fi] = frame
                bbox = bbox_by_frame[fi]
            contexts.append((role, annotate_context(frame, bbox, role, fi)))
        it["context_panels"] = contexts

    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "inventory",
            "review_packages/target_001_holdout_v2_frozen_result_error_analysis",
            "videos",
            "templates",
            "audit",
            "governance",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=True)

        inv_rows = []
        for it in items:
            inv_rows.append({
                k: it[k]
                for k in it
                if k not in {"query_bgr", "target_bgr", "distractor_bgr", "context_panels", "bbox_by_frame"}
            })
        write_jsonl(tmp / "inventory" / "target_001_holdout_v2_f3n_error_audit_item_inventory.jsonl", inv_rows)
        mapping = {
            "positive": [it["audit_item_id"] for it in items if it["cohort"] == "positive"],
            "same_team_fp": [it["audit_item_id"] for it in items if it["cohort"] == "same_team_fp"],
            "other_team_fp": [it["audit_item_id"] for it in items if it["cohort"] == "other_team_fp"],
            "total": 50,
            "duplicate_across_cohorts": 0,
            "automatic_diagnosis_absent": True,
            "human_diagnosis_pending": True,
        }
        write_json(tmp / "inventory" / "target_001_holdout_v2_f3n_error_audit_mapping.json", mapping)

        sheet_dir = tmp / "review_packages" / "target_001_holdout_v2_frozen_result_error_analysis"
        watermark = cfg["diagnostic_videos"]["panel_watermark"]
        min_w = int(cfg["contact_sheets"]["min_width_px"])
        cols = int(cfg["contact_sheets"]["grid_cols"])

        cohorts_for_sheets = [
            ("positive", [it for it in items if it["cohort"] == "positive"],
             ["target_001_holdout_v2_positive_audit_sheet.png"]),
            ("same_team_fp", [it for it in items if it["cohort"] == "same_team_fp"],
             ["target_001_holdout_v2_same_team_fp_audit_sheet_01.png",
              "target_001_holdout_v2_same_team_fp_audit_sheet_02.png"]),
            ("other_team_fp", [it for it in items if it["cohort"] == "other_team_fp"],
             ["target_001_holdout_v2_other_team_fp_audit_sheet_01.png",
              "target_001_holdout_v2_other_team_fp_audit_sheet_02.png"]),
        ]
        sheet_paths = []
        for cohort_name, cohort_items, names in cohorts_for_sheets:
            per = 10
            for si, name in enumerate(names):
                chunk = cohort_items[si * per:(si + 1) * per]
                if len(chunk) != per:
                    raise ResultAuditError(f"sheet chunk size {cohort_name} {si}")
                sheet = render_contact_sheet(
                    chunk,
                    title=f"target_001 holdout v2 — {cohort_name} audit sheet {si+1:02d}",
                    min_width=min_w,
                    cols=cols,
                    watermark=watermark,
                )
                if sheet.shape[1] < min_w:
                    raise ResultAuditError("sheet width < 3600")
                out_path = sheet_dir / name
                ok = cv2.imwrite(str(out_path), sheet)
                if not ok:
                    raise ResultAuditError(f"sheet write failed {name}")
                sheet_paths.append(f"review_packages/target_001_holdout_v2_frozen_result_error_analysis/{name}")

        # Videos
        video_specs = [
            ("positive", [it for it in items if it["cohort"] == "positive"],
             "target_001_holdout_v2_positive_result_audit.mp4"),
            ("same_team_fp", [it for it in items if it["cohort"] == "same_team_fp"],
             "target_001_holdout_v2_same_team_fp_result_audit.mp4"),
            ("other_team_fp", [it for it in items if it["cohort"] == "other_team_fp"],
             "target_001_holdout_v2_other_team_fp_result_audit.mp4"),
        ]
        # positives already rank-sorted; FP cohorts: sort by primary rank ascending for video
        video_paths = []
        title_frames = max(1, int(round(float(vcfg["title_card_sec"]) * float(vcfg["fps"]))))
        for cohort_name, cohort_items, name in video_specs:
            ordered = sorted(cohort_items, key=lambda r: (int(r["primary_rank"]), str(r["query_id"])))
            expected_n = 10 if cohort_name == "positive" else 20
            if len(ordered) != expected_n:
                raise ResultAuditError(f"video cohort size {cohort_name}")
            out_path = tmp / "videos" / name
            writer = FfmpegRawWriter(
                out_path,
                width=width,
                height=height,
                fps=float(vcfg["fps"]),
                crf=int(vcfg["crf"]),
                preset=str(vcfg["preset"]),
            )
            try:
                for it in ordered:
                    card = draw_title_card(width, height, it)
                    for _ in range(title_frames):
                        writer.write(card)
                    for fi in range(int(it["clip_start_frame"]), int(it["clip_end_frame"]) + 1):
                        frame = frame_cache[fi]
                        bbox = it["bbox_by_frame"].get(fi)
                        annotated = annotate_video_frame(
                            frame,
                            item=it,
                            frame_index=fi,
                            bbox=bbox,
                            watermark=vcfg["watermark"],
                        )
                        writer.write(annotated)
            finally:
                writer.close()
            video_paths.append(f"videos/{name}")

        # Manual template
        template_path = tmp / "templates" / "target_001_holdout_v2_error_analysis_manual_review_template.csv"
        with template_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANUAL_TEMPLATE_FIELDS)
            writer.writeheader()
            for it in items:
                row = {f: "" for f in MANUAL_TEMPLATE_FIELDS}
                row.update({
                    "audit_item_id": it["audit_item_id"],
                    "cohort": it["cohort"],
                    "query_id": it["query_id"],
                    "segment_id": it["segment_id"],
                    "primary_rank": it["primary_rank"],
                    "frozen_gt_decision": it["frozen_gt_decision"],
                    "same_team_cohort": it["same_team_cohort"],
                    "T_max": it["T_max"],
                    "T_max_member_id": it["T_max_member_id"],
                    "D_max": it["D_max"],
                    "D_max_member_id": it["D_max_member_id"],
                    "S_primary": it["S_primary"],
                    "query_crop_path": it["representative_crop_path"],
                    "best_target_crop_path": it["best_target_crop_path"],
                    "best_distractor_crop_path": it["best_distractor_crop_path"],
                })
                for f in BLANK_MANUAL_FIELDS:
                    if row[f] != "":
                        raise ResultAuditError(f"manual field not blank: {f}")
                writer.writerow(row)

        # Quantitative audit (no recompute)
        ranking = sorted(f3m["joined"], key=lambda r: int(r["rank"]))
        def cohort_of(row: Mapping[str, Any]) -> str:
            if row["binary_clean_player_label"] == 1:
                return "positive"
            if row.get("same_team_negative_cohort"):
                return "same_team_fp"
            return "other_team_fp"

        def dist_for(k: int) -> dict[str, int]:
            c = Counter(cohort_of(r) for r in ranking if int(r["rank"]) <= k)
            return {
                "positive": int(c.get("positive", 0)),
                "same_team_fp": int(c.get("same_team_fp", 0)),
                "other_team_fp": int(c.get("other_team_fp", 0)),
            }

        pos_ranks = [int(r["rank"]) for r in ranking if r["binary_clean_player_label"] == 1]
        quant = {
            "metrics_recomputed": False,
            "positive_ranks": pos_ranks,
            "primary_segment_metrics": {
                "Recall@1": f3m["seg"]["Recall@1"],
                "Recall@3": f3m["seg"]["Recall@3"],
                "Recall@5": f3m["seg"]["Recall@5"],
                "Recall@10": f3m["seg"]["Recall@10"],
                "MRR": f3m["seg"]["MRR"],
                "AP": f3m["seg"]["Average_Precision"],
                "AUROC": f3m["seg"]["AUROC"],
                "AUPRC": f3m["seg"]["AUPRC"],
                "margin": f3m["seg"]["min_positive_minus_max_negative_margin"],
                "positive_median_minus_negative_median": f3m["seg"]["positive_median_minus_negative_median"],
                "positive_median_minus_same_team_negative_median": f3m["seg"][
                    "positive_median_minus_same_team_negative_median"
                ],
            },
            "primary_component_metrics": {
                "AP": f3m["comp"]["Average_Precision"],
                "AUROC": f3m["comp"]["AUROC"],
                "margin": f3m["comp"]["separation_margin_min_pos_minus_max_neg"],
            },
            "same_team_metrics": {
                "AUROC": f3m["same"]["AUROC"],
                "AP": f3m["same"]["Average_Precision"],
                "AUPRC": f3m["same"]["AUPRC"],
                "margin": f3m["same"]["min_positive_minus_max_same_team_negative_margin"],
            },
            "secondary_metrics": f3m["secondary_metrics"],
            "strong_signal_checks": f3m["outcome"]["strong_signal_checks"],
            "rank_1_to_10_cohort_distribution": dist_for(10),
            "rank_1_to_20_cohort_distribution": dist_for(20),
            "best_positive_rank": min(pos_ranks),
            "worst_positive_rank": max(pos_ranks),
            "median_positive_rank": float(np.median(pos_ranks)),
            "frozen_metric_source_shas": f3m["metric_shas"],
            "frozen_outcome": f3m["summary"]["performance_outcome"],
        }
        write_json(tmp / "audit" / "target_001_holdout_v2_frozen_result_quantitative_audit.json", quant)

        # Integrity
        holdout_sha_after = sha256_file(holdout)
        gallery_after = {
            "target": sha256_file(
                f3g["root"] / "target_gallery_v2" / "target_001_gallery_v2_individual_embeddings.npy"
            ),
            "distractor": sha256_file(
                f3g["root"] / "distractor_gallery_v1" / "target_001_same_team_distractor_individual_embeddings.npy"
            ),
        }
        # sample F3K crop SHA recheck for audit queries
        crop_ok = all(
            sha256_file(it["representative_crop_path_absolute"]) == it["representative_crop_sha256"]
            for it in items
        )
        integrity = {
            "score_rank_sha_unchanged": True,
            "primary_score_sha256": f3m["seal"]["primary_score_sha256"],
            "ranking_sha256": f3m["seal"]["ranking_sha256"],
            "gt_join_sha_unchanged": True,
            "gt_join_sha256": f3m["joined_sha256"],
            "gallery_sha_unchanged": gallery_after == f3g["shas"],
            "gallery_shas_before": f3g["shas"],
            "gallery_shas_after": gallery_after,
            "f3k_crop_sha_unchanged": crop_ok,
            "holdout_sha_before": holdout_sha_before,
            "holdout_sha_after": holdout_sha_after,
            "holdout_sha_unchanged": holdout_sha_before == holdout_sha_after,
            "no_post_result_mutation": True,
            "scores_recomputed": False,
            "embeddings_generated": 0,
        }
        if not integrity["gallery_sha_unchanged"] or not integrity["holdout_sha_unchanged"] or not crop_ok:
            raise ResultAuditError("immutability failure")
        write_json(tmp / "audit" / "target_001_holdout_v2_frozen_result_contract_integrity_audit.json", integrity)

        # Governance retirement
        retirement = {
            "holdout_id": "target_001_independent_holdout_v2",
            "independent_evaluation_completed": True,
            "frozen_evaluation_result_exists": True,
            "post_evaluation_error_analysis_performed": True,
            "future_independent_test_eligible": False,
            "future_threshold_calibration_eligible": False,
            "future_gallery_evaluation_eligible": False,
            "future_method_selection_eligible": True,
            "future_error_analysis_eligible": True,
            "future_regression_diagnostic_eligible": True,
            "enrollment_eligible": False,
            "gallery_growth_eligible": False,
            "hard_negative_mining_eligible": False,
            "fine_tuning_eligible": False,
            "new_method_requires_new_holdout": True,
            "source_video_deleted": False,
            "source_video_modified": False,
        }
        write_json(tmp / "governance" / "target_001_holdout_v2_post_evaluation_retirement.json", retirement)

        cleanup = {
            "cleanup_executed": False,
            "deleted_files": 0,
            "deletion_approvals_all_false": True,
            "categories": {
                "must_keep": {
                    "description": "Frozen contracts, GT, gallery embeddings, F3M scores, source videos, checkpoints",
                    "delete_approved": False,
                },
                "archive_candidate": {
                    "description": "Large rebuild snapshots already on NTFS recovery path",
                    "delete_approved": False,
                },
                "reproducible_large_artifact_candidate": {
                    "description": "Contact sheets/videos reproducible from frozen lineage",
                    "delete_approved": False,
                },
                "possible_safe_delete_after_archive_verification": {
                    "description": "Temp roots only after verified archive; none deleted in F3N",
                    "delete_approved": False,
                },
                "never_delete_without_explicit_approval": {
                    "description": "Holdout MP4, enrollment MP4, gallery NPY, F3L freeze, F3M scores",
                    "delete_approved": False,
                },
            },
            "note": "F3N performs no deletion or cleanup. Future dry-run retention only with explicit approval.",
        }
        write_json(tmp / "governance" / "target_001_stage5d_cleanup_deferred_inventory.json", cleanup)

        write_json(tmp / "runtime" / "target_001_f3n_access_audit.json", access)
        write_json(
            tmp / "runtime" / "target_001_f3n_immutability_audit.json",
            {
                **integrity,
                "threshold_selected": False,
                "identity_assignments": 0,
                "gallery_mutation": False,
                "manual_root_cause_decisions": 0,
                "upstream_deletions": 0,
            },
        )
        write_json(
            tmp / "runtime" / "target_001_f3n_video_contract.json",
            {
                "fps": vcfg["fps"],
                "codec": "libx264",
                "pix_fmt": "yuv420p",
                "audio": False,
                "map_metadata": -1,
                "bbox_interpolation": False,
                "title_card_sec": vcfg["title_card_sec"],
                "max_clip_sec": vcfg["max_clip_sec"],
                "watermark": vcfg["watermark"],
            },
        )
        shutil.copy2(config_path, tmp / "effective_configs" / config_path.name)

        # Artifact budget
        pngs = list((tmp / "review_packages").rglob("*.png"))
        mp4s = list((tmp / "videos").rglob("*.mp4"))
        npys = list(tmp.rglob("*.npy"))
        if len(pngs) != 5 or len(mp4s) != 3 or npys:
            raise ResultAuditError(f"artifact budget fail png={len(pngs)} mp4={len(mp4s)} npy={len(npys)}")

        summary = {
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "target_id": TARGET_ID,
            "project_head": git_info["head"],
            "frozen_f3m_outcome": f3m["summary"]["performance_outcome"],
            "scoreable_count": 115,
            "positive_count": 10,
            "negative_count": 105,
            "positive_audit_items": 10,
            "same_team_fp_audit_items": 20,
            "other_team_fp_audit_items": 20,
            "total_audit_items": 50,
            "contact_sheets": 5,
            "diagnostic_videos": 3,
            "sheet_paths": sheet_paths,
            "video_paths": video_paths,
            "manual_root_cause_decisions": 0,
            "scores_recomputed": False,
            "embeddings_generated": 0,
            "gallery_mutation": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "holdout_retired_from_future_independent_testing": True,
            "cleanup_deferred": True,
            "deleted_files": 0,
            "f3m_snapshot_sha256": f3m["snapshot"]["sha256"],
            "f3l_snapshot_sha256": f3l["snapshot"]["sha256"],
            "f3k_snapshot_sha256": f3k["snapshot"]["sha256"],
            "f3g_snapshot_sha256": f3g["snapshot"]["sha256"],
            "created_at": utc_now(),
        }
        contract = {
            "schema_version": "stage5d_f3n_contract_v1",
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "scores_recomputed": False,
            "embeddings_generated": 0,
            "gallery_mutation": False,
            "manual_root_cause_decisions": 0,
            "deleted_files": 0,
            "allowed_failure_categories": list(ALLOWED_FAILURE_CATEGORIES),
        }
        files = list_files(tmp)
        manifest = {
            "schema_version": "stage5d_f3n_manifest_v1",
            "file_count": len(files),
            "files": files,
            "png_count": 5,
            "mp4_count": 3,
            "npy_count": 0,
            "listing_sha256": hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest(),
        }
        write_json(tmp / "stage5d_f3n_summary.json", summary)
        write_json(tmp / "stage5d_f3n_contract.json", contract)
        write_json(tmp / "stage5d_f3n_manifest.json", manifest)

        atomic_publish(tmp, final_dir)
        summary["final_dir"] = str(final_dir)
        return summary
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/reid/independent_holdout_v2_result_audit_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(json.dumps({
        "final_status": summary["final_status"],
        "audit_items": summary["total_audit_items"],
        "sheets": summary["contact_sheets"],
        "videos": summary["diagnostic_videos"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
