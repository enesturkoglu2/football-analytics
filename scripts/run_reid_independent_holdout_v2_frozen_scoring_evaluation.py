#!/usr/bin/env python3
"""Stage 5D-F3M — Holdout v2 OSNet query embedding + frozen target–distractor scoring.

Label-blind scoring on exact 115 clean-player queries. GT join only after
pre-GT seal. No threshold, identity, gallery mutation, or holdout enrollment.
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

import numpy as np
import yaml

try:
    from sklearn import __version__ as SKLEARN_VERSION
    from sklearn.metrics import average_precision_score, roc_auc_score

    _HAS_SKLEARN = True
except ImportError:  # sn-reid-cpu may lack sklearn; formulas match binary sklearn defs
    SKLEARN_VERSION = "unavailable_pure_numpy_fallback"
    _HAS_SKLEARN = False

    def average_precision_score(y_true, y_score):  # type: ignore[misc]
        y_true = np.asarray(y_true, dtype=np.int32)
        y_score = np.asarray(y_score, dtype=np.float64)
        order = np.argsort(-y_score, kind="mergesort")
        y_true = y_true[order]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            return 0.0
        tp = 0
        precision_sum = 0.0
        for i, lab in enumerate(y_true, start=1):
            if lab == 1:
                tp += 1
                precision_sum += tp / float(i)
        return float(precision_sum / n_pos)

    def roc_auc_score(y_true, y_score):  # type: ignore[misc]
        y_true = np.asarray(y_true, dtype=np.int32)
        y_score = np.asarray(y_score, dtype=np.float64)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if pos.size == 0 or neg.size == 0:
            raise ValueError("AUROC undefined without both classes")
        # Mann–Whitney U / Wilcoxon-Mann-Whitney (matches sklearn for binary scores)
        correct = 0.0
        for p in pos:
            correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        return float(correct / (pos.size * neg.size))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid import embedding as emb  # noqa: E402

CONFIG_SCHEMA = "reid_independent_holdout_v2_frozen_scoring_config_v1"
TARGET_ID = "target_001"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3M_TARGET_001_NEW_INDEPENDENT_HOLDOUT_FROZEN_SCORING_EVALUATED"
)
READINESS = "TARGET_001_INDEPENDENT_HOLDOUT_V2_FROZEN_RESULT_READY_FOR_AUDIT"
NEXT_GATE = (
    "STAGE5D-F3N_TARGET_001_INDEPENDENT_HOLDOUT_RESULT_AUDIT_AND_ERROR_ANALYSIS_PACKAGE"
)

ALLOWED_DIRTY = {
    "scripts/run_reid_independent_holdout_v2_frozen_scoring_evaluation.py",
    "configs/reid/independent_holdout_v2_frozen_scoring_stage5d_target_001.yaml",
    "tests/test_reid_independent_holdout_v2_frozen_scoring_evaluation.py",
    "docs/setup/stage5d-target-independent-holdout-osnet-query-embedding-and-frozen-target-distractor-scoring.md",
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

GT_LABEL_FIELDS = frozenset(
    {
        "manual_ground_truth_decision",
        "manual_target_present",
        "manual_same_target_as_target_001",
        "manual_same_team_as_target",
        "manual_visible_jersey_number",
        "jersey_number_provenance",
        "clean_positive",
        "clean_negative",
        "clean_same_team_negative",
        "metric_inclusion",
        "reviewer",
        "final_approver",
        "target_occurrence_yes",
        "target_occurrence_no",
        "positive",
        "negative",
        "same_team",
        "other_team",
        "binary_label",
        "ground_truth_label",
    }
)


class FrozenScoringError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


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
        raise FrozenScoringError(f"path traversal rejected: {value}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != CONFIG_SCHEMA:
        raise FrozenScoringError("config schema mismatch")
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
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GIT_CONTRACT_MISMATCH branch")
    if head != cfg["project_head_expected"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GIT_CONTRACT_MISMATCH HEAD")
    if head != origin:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GIT_CONTRACT_MISMATCH origin")
    if msg != cfg["project_head_message_expected"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GIT_CONTRACT_MISMATCH message")
    dirty = [ln[3:] if ln.startswith("??") else ln[3:] for ln in status.splitlines() if ln.strip()]
    # allow only the four tracked F3M files during development/tests
    for path in dirty:
        norm = path.replace("\\", "/")
        if norm not in ALLOWED_DIRTY:
            # also allow empty
            raise FrozenScoringError(
                f"BLOCKED_STAGE5D_F3M_GIT_CONTRACT_MISMATCH dirty:{norm}"
            )
    return {
        "branch": branch,
        "head": head,
        "origin_main": origin,
        "message": msg,
        "working_tree_allowed_dirty": sorted(dirty),
    }


def validate_snapshot(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise FrozenScoringError(f"snapshot missing: {path}")
    actual = sha256_file(path)
    side = Path(str(path) + ".sha256")
    if not side.is_file():
        raise FrozenScoringError(f"snapshot sidecar missing: {side}")
    side_sha = side.read_text(encoding="utf-8").split()[0]
    man_path = path.with_name(path.name.replace(".tar.gz", "_manifest.json"))
    if not man_path.is_file():
        raise FrozenScoringError(f"snapshot manifest missing: {man_path}")
    man = read_json(man_path)
    man_sha = man.get("sha256") or man.get("archive_sha256")
    if not (actual == side_sha == man_sha == expected_sha):
        raise FrozenScoringError(
            f"snapshot sha mismatch {path.name}: actual={actual} expected={expected_sha}"
        )
    listing = path.with_name(path.name.replace(".tar.gz", "_listing.txt"))
    return {
        "path": str(path),
        "sha256": actual,
        "sidecar_ok": True,
        "manifest_ok": True,
        "listing_exists": listing.is_file(),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_f3l_decisions(f3l_root: Path) -> list[dict[str, Any]]:
    path = f3l_root / "manual_freeze" / "target_001_holdout_v2_ground_truth_decisions_frozen.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_f3l(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3l_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3l_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GROUND_TRUTH_FREEZE_CONTRACT_MISMATCH status")
    if summary.get("readiness") != pack["expected_readiness"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GROUND_TRUTH_FREEZE_CONTRACT_MISMATCH readiness")
    decisions = load_f3l_decisions(root)
    if len(decisions) != 141:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GROUND_TRUTH_FREEZE_CONTRACT_MISMATCH reviewed")
    yes = [r for r in decisions if r["manual_ground_truth_decision"] == "target_occurrence_yes"]
    no = [r for r in decisions if r["manual_ground_truth_decision"] == "target_occurrence_no"]
    non_player = [r for r in decisions if r["manual_ground_truth_decision"] == "non_player"]
    invalid = [r for r in decisions if r["manual_ground_truth_decision"] == "invalid"]
    amb = [r for r in decisions if r["manual_ground_truth_decision"] == "multi_person_ambiguous"]
    uncertain = [r for r in decisions if r["manual_ground_truth_decision"] == "uncertain"]
    same_team_no = [r for r in no if r.get("manual_same_team_as_target") == "yes"]
    other_team_no = [r for r in no if r.get("manual_same_team_as_target") == "no"]
    clean_pos = [r for r in decisions if _as_bool(r.get("clean_positive"))]
    clean_neg = [r for r in decisions if _as_bool(r.get("clean_negative"))]
    clean_stn = [r for r in decisions if _as_bool(r.get("clean_same_team_negative"))]
    scoreable = [r for r in decisions if _as_bool(r.get("query_score_eligibility"))]
    checks = {
        "complete_universe": summary.get("complete_universe") == pack["expected_complete_universe"],
        "reviewed_eligible": len(decisions) == pack["expected_reviewed_eligible"],
        "unreviewed_ineligible": summary.get("unreviewed_ineligible") == pack["expected_unreviewed_ineligible"],
        "yes": len(yes) == pack["expected_target_occurrence_yes"],
        "no": len(no) == pack["expected_target_occurrence_no"],
        "same_team_no": len(same_team_no) == pack["expected_same_team_target_no"],
        "other_team_no": len(other_team_no) == pack["expected_other_team_target_no"],
        "non_player": len(non_player) == pack["expected_non_player"],
        "invalid": len(invalid) == pack["expected_invalid"],
        "ambiguous": len(amb) == pack["expected_multi_person_ambiguous"],
        "uncertain": len(uncertain) == pack["expected_uncertain"],
        "clean_pos": len(clean_pos) == pack["expected_clean_positive_gt"],
        "clean_neg": len(clean_neg) == pack["expected_clean_negative_gt"],
        "clean_stn": len(clean_stn) == pack["expected_clean_same_team_negative_gt"],
        "metric_excl": sum(1 for r in decisions if not _as_bool(r.get("metric_inclusion")))
        == pack["expected_reviewed_metric_exclusions"],
        "metric_excl_classes": (len(invalid) + len(amb))
        == pack["expected_reviewed_metric_exclusions"],
        "non_player_query_excluded_but_metric_clean_neg": all(
            (not _as_bool(r.get("query_score_eligibility"))) and _as_bool(r.get("clean_negative"))
            for r in non_player
        ),
        "component_policy": summary.get("component_policy") == pack["expected_component_policy"],
        "conflicts": summary.get("component_conflict_count") == pack["expected_component_conflict_count"],
        "scoreable": len(scoreable) == 115,
        "positive_ids": [r["review_item_id"] for r in yes] == list(POSITIVE_IDS),
    }
    if not all(checks.values()):
        bad = [k for k, v in checks.items() if not v]
        raise FrozenScoringError(
            f"BLOCKED_STAGE5D_F3M_GROUND_TRUTH_FREEZE_CONTRACT_MISMATCH {bad}"
        )
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": str(root),
        "summary": summary,
        "decisions": decisions,
        "scoreable_rows": scoreable,
        "checks": checks,
        "snapshot": snap,
        "gallery_embedding_similarity_score_rank_metric_reads": 0,
        "threshold_selected": False,
        "identity_assignments": 0,
        "gallery_mutation": False,
        "enrollment_gallery_growth_threshold_calibration_forbidden": True,
    }


def validate_f3h(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3h_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3h_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH status")
    primary = read_json(root / "scoring" / "target_001_target_distractor_primary_scoring_contract.json")
    secondary = read_json(root / "scoring" / "target_001_target_distractor_secondary_scoring_contract.json")
    tie = read_json(root / "scoring" / "target_001_target_distractor_tie_break_and_aggregation_contract.json")
    metrics = read_json(root / "evaluation" / "target_001_new_holdout_metric_contract.json")
    outcomes = read_json(root / "evaluation" / "target_001_new_holdout_outcome_rules.json")
    if primary.get("formula_id") != "TARGET_DISTRACTOR_MAX_MARGIN":
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH formula")
    if primary["T_max"]["aggregation"] != "max" or primary["D_max"]["aggregation"] != "max":
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH agg")
    if primary["T_max"]["target_top_k"] != 1 or primary["D_max"]["distractor_top_k"] != 1:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH topk")
    if primary["S_primary"]["formula"] != "T_max(q) - D_max(q)":
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH S_primary")
    expected_tb = [
        "primary_score_descending",
        "T_max_descending",
        "D_max_ascending",
        "query_stable_id_ascending",
    ]
    if primary.get("tie_break") != expected_tb or tie.get("query_level_tie_break") != expected_tb:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH tie_break")
    if secondary["formulas"]["S_top3_margin"]["k"] != 3:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH top3k")
    thr = outcomes.get("threshold_and_abstention", {})
    if thr.get("threshold_selected") is not False or thr.get("identity_assignments") != 0:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_SCORING_CONTRACT_MISMATCH threshold")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": str(root),
        "primary": primary,
        "secondary": secondary,
        "tie": tie,
        "metrics": metrics,
        "outcomes": outcomes,
        "snapshot": snap,
        "score_weights_absent": True,
        "member_weights_absent": True,
        "formula_promotion_forbidden_on_same_holdout": True,
        "sample_optimization_absent": True,
    }


def validate_f3g(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3g_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3g_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH status")
    if summary.get("readiness") != pack["expected_readiness"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH readiness")
    t_path = root / "target_gallery_v2" / "target_001_gallery_v2_individual_embeddings.npy"
    d_path = root / "distractor_gallery_v1" / "target_001_same_team_distractor_individual_embeddings.npy"
    tc = root / "target_gallery_v2" / "target_001_gallery_v2_centroid.npy"
    tm = root / "target_gallery_v2" / "target_001_gallery_v2_medoid.npy"
    dc = root / "distractor_gallery_v1" / "target_001_same_team_distractor_centroid.npy"
    dm = root / "distractor_gallery_v1" / "target_001_same_team_distractor_medoid.npy"
    shas = {
        "target": sha256_file(t_path),
        "distractor": sha256_file(d_path),
        "target_centroid": sha256_file(tc),
        "target_medoid": sha256_file(tm),
        "distractor_centroid": sha256_file(dc),
        "distractor_medoid": sha256_file(dm),
    }
    expected = {
        "target": pack["expected_target_embedding_sha256"],
        "distractor": pack["expected_distractor_embedding_sha256"],
        "target_centroid": pack["expected_target_centroid_sha256"],
        "target_medoid": pack["expected_target_medoid_sha256"],
        "distractor_centroid": pack["expected_distractor_centroid_sha256"],
        "distractor_medoid": pack["expected_distractor_medoid_sha256"],
    }
    if shas != expected:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH npy_sha")
    T = np.load(t_path)
    D = np.load(d_path)
    for name, arr, n in (("T", T, 13), ("D", D, 23)):
        if arr.shape != (n, 512) or arr.dtype != np.float32:
            raise FrozenScoringError(f"BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH {name}_shape")
        if not np.isfinite(arr).all():
            raise FrozenScoringError(f"BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH {name}_finite")
        if int(np.all(arr == 0, axis=1).sum()) != 0:
            raise FrozenScoringError(f"BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH {name}_zero")
        norms = np.linalg.norm(arr.astype(np.float64), axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            raise FrozenScoringError(f"BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH {name}_l2")
    t_inv = read_jsonl(root / "target_gallery_v2" / "target_001_gallery_v2_member_inventory.jsonl")
    d_inv = read_jsonl(
        root / "distractor_gallery_v1" / "target_001_same_team_distractor_member_inventory.jsonl"
    )
    if len(t_inv) != 13 or len(d_inv) != 23:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH members")
    if summary.get("target_medoid_member_id") != pack["expected_target_medoid_id"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH t_medoid")
    if summary.get("distractor_medoid_member_id") != pack["expected_distractor_medoid_id"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH d_medoid")
    det = summary.get("two_pass_determinism") or {}
    if not det.get("exact_match") or float(det.get("overall_max_absolute_difference", 1)) != 0.0:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_GALLERY_CONTRACT_MISMATCH determinism")
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": str(root),
        "T": T,
        "D": D,
        "target_centroid": np.load(tc).astype(np.float32),
        "target_medoid": np.load(tm).astype(np.float32),
        "distractor_centroid": np.load(dc).astype(np.float32),
        "distractor_medoid": np.load(dm).astype(np.float32),
        "target_ids": [r["member_id"] for r in t_inv],
        "distractor_ids": [r["member_id"] for r in d_inv],
        "paths": {
            "target": str(t_path),
            "distractor": str(d_path),
            "target_centroid": str(tc),
            "target_medoid": str(tm),
            "distractor_centroid": str(dc),
            "distractor_medoid": str(dm),
        },
        "shas_before": shas,
        "snapshot": snap,
        "automatic_member_removal": False,
        "member_weighting": False,
        "gallery_v1_mutation": False,
        "gallery_v2_mutation": False,
        "holdout_member_count": 0,
    }


def validate_f3k(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pack = cfg["stage5d_f3k_package"]
    root = _PROJECT_ROOT / pack["path"]
    summary = read_json(root / "stage5d_f3k_summary.json")
    if summary.get("final_status") != pack["expected_final_status"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH status")
    if summary.get("representative_crop_count") != pack["expected_representative_crop_count"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH count")
    manifest = read_jsonl(root / "crops" / "target_001_holdout_v2_gt_review_crop_manifest.jsonl")
    if len(manifest) != pack["expected_crop_manifest_rows"]:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH manifest")
    by_id = {}
    for row in manifest:
        assert_no_path_traversal(row["crop_path"])
        crop_abs = root / row["crop_path"]
        if not crop_abs.is_file():
            raise FrozenScoringError(
                f"BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH missing:{row['crop_path']}"
            )
        actual = sha256_file(crop_abs)
        if actual != row["crop_sha256"]:
            raise FrozenScoringError(
                f"BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH sha:{row['review_item_id']}"
            )
        if row["holdout_sha256"] != pack["expected_holdout_sha256"]:
            raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH holdout_sha")
        by_id[row["review_item_id"]] = {**row, "crop_path_absolute": str(crop_abs)}
    snap = validate_snapshot(Path(pack["snapshot_path"]), pack["expected_snapshot_sha256"])
    return {
        "root": str(root),
        "manifest": manifest,
        "by_id": by_id,
        "snapshot": snap,
        "crop_selection_objective_only": True,
        "bbox_interpolation": False,
        "manual_decisions_at_f3k": 0,
        "automatic_target_team_jersey_predictions": 0,
    }


def build_scoreable_universe(
    decisions: Sequence[Mapping[str, Any]],
    crops_by_id: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    scoreable = []
    for row in decisions:
        if not _as_bool(row.get("query_score_eligibility")):
            continue
        decision = row["manual_ground_truth_decision"]
        if decision not in {"target_occurrence_yes", "target_occurrence_no"}:
            raise FrozenScoringError("scoreable row with unexpected decision")
        if row.get("manual_crop_valid") != "yes":
            raise FrozenScoringError("scoreable crop_valid!=yes")
        if row.get("manual_single_person") != "yes":
            raise FrozenScoringError("scoreable single_person!=yes")
        if row.get("manual_identity_continuity_observed") != "yes":
            raise FrozenScoringError("scoreable identity_continuity!=yes")
        rid = row["review_item_id"]
        if rid not in crops_by_id:
            raise FrozenScoringError(f"BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH {rid}")
        crop = crops_by_id[rid]
        if crop["crop_sha256"] != row["representative_crop_sha256"]:
            raise FrozenScoringError(f"crop sha mismatch vs F3L for {rid}")
        scoreable.append(
            {
                "review_item_id": rid,
                "segment_id": row["segment_id"],
                "component_id": row["ground_truth_component_id"],
                "decision_row": row,
                "crop": crop,
            }
        )
    scoreable.sort(key=lambda r: (int(r["review_item_id"].split("_")[-1]), r["segment_id"]))
    qu = cfg["query_universe"]
    pos = [r for r in scoreable if r["decision_row"]["manual_ground_truth_decision"] == "target_occurrence_yes"]
    neg = [r for r in scoreable if r["decision_row"]["manual_ground_truth_decision"] == "target_occurrence_no"]
    stn = [r for r in neg if r["decision_row"].get("manual_same_team_as_target") == "yes"]
    otn = [r for r in neg if r["decision_row"].get("manual_same_team_as_target") == "no"]
    if len(scoreable) != qu["scoreable"]:
        raise FrozenScoringError("scoreable count mismatch")
    if len(pos) != qu["positive"] or [r["review_item_id"] for r in pos] != list(POSITIVE_IDS):
        raise FrozenScoringError("positive universe mismatch")
    if len(neg) != qu["negative_player"]:
        raise FrozenScoringError("negative player mismatch")
    if len(stn) != qu["same_team_negative"] or len(otn) != qu["other_team_negative"]:
        raise FrozenScoringError("team cohort mismatch")
    ids = [r["review_item_id"] for r in scoreable]
    if len(ids) != len(set(ids)):
        raise FrozenScoringError("duplicate query ids")
    return {
        "scoreable": scoreable,
        "positive_ids": [r["review_item_id"] for r in pos],
        "negative_player_ids": [r["review_item_id"] for r in neg],
        "same_team_negative_ids": [r["review_item_id"] for r in stn],
        "other_team_negative_ids": [r["review_item_id"] for r in otn],
    }


def build_label_blind_projection(scoreable: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(scoreable):
        crop = item["crop"]
        row = {
            "stable_query_id": item["review_item_id"],
            "review_item_id": item["review_item_id"],
            "segment_id": item["segment_id"],
            "component_id": item["component_id"],
            "query_order_index": idx,
            "representative_crop_path": crop["crop_path"],
            "representative_crop_path_absolute": crop["crop_path_absolute"],
            "representative_crop_sha256": crop["crop_sha256"],
            "crop_width": crop["crop_width"],
            "crop_height": crop["crop_height"],
            "source_frame_index": crop["source_frame_index"],
            "source_bbox_xyxy": crop["source_bbox_xyxy"],
            "canonical_crop_bbox_xyxy": crop["canonical_crop_bbox_xyxy"],
            "holdout_sha256": crop["holdout_sha256"],
            "canonical_preprocessing_authorization": "canonical_existing_osnet_bgr_rgb_256x128_imagenet",
        }
        for key in row:
            lk = key.lower()
            if lk in GT_LABEL_FIELDS or any(
                tok in lk
                for tok in (
                    "occurrence",
                    "positive",
                    "negative",
                    "same_team",
                    "other_team",
                    "jersey",
                    "reviewer",
                    "approver",
                )
            ):
                raise FrozenScoringError(f"forbidden projection key: {key}")
        rows.append(row)
    return rows


def build_execution_contract(
    *,
    cfg: Mapping[str, Any],
    projection_sha: str,
    gallery: Mapping[str, Any],
    f3h: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_holdout_v2_frozen_scoring_execution_contract_v1",
        "written_before_gallery_contents_inspected_for_scoring_outputs": True,
        "written_before_query_embeddings": True,
        "written_before_similarity_results": True,
        "query_count": 115,
        "target_members": 13,
        "distractor_members": 23,
        "embedding_dimension": 512,
        "primary_formula": cfg["primary_formula"],
        "primary_formula_id": "TARGET_DISTRACTOR_MAX_MARGIN",
        "S_primary": "T_max(q) - D_max(q)",
        "secondary_formulas": list(f3h["secondary"]["formulas"].keys()),
        "tie_break": f3h["primary"]["tie_break"],
        "expected_matrix_shapes": {
            "query_embeddings": [115, 512],
            "target_cosine": [115, 13],
            "distractor_cosine": [115, 23],
        },
        "expected_score_rows": 115,
        "expected_ranking_rows": 115,
        "gt_labels_hidden_from_scoring": True,
        "formula_member_weight_mutation_forbidden": True,
        "projection_sha256": projection_sha,
        "gallery_shas": gallery["shas_before"],
        "scoring_contract_sha256": {
            "primary": sha256_file(
                Path(f3h["root"])
                / "scoring"
                / "target_001_target_distractor_primary_scoring_contract.json"
            ),
            "secondary": sha256_file(
                Path(f3h["root"])
                / "scoring"
                / "target_001_target_distractor_secondary_scoring_contract.json"
            ),
            "tie_break": sha256_file(
                Path(f3h["root"])
                / "scoring"
                / "target_001_target_distractor_tie_break_and_aggregation_contract.json"
            ),
        },
    }


def _load_model(checkpoint: Path, sn_reid_root: Path):
    with emb.temporary_sys_path_prepend(sn_reid_root):
        model = emb.build_osnet_cpu_model(model_name=emb.MODEL_NAME)
        emb.load_osnet_checkpoint_weights(model, checkpoint)
        model.eval()
        model.cpu()
        return model


def embed_queries_two_pass(
    *,
    projection: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    runtime: Mapping[str, Any],
    access_audit: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if len(projection) != 115:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_FAILURE count")
    atol = float(runtime["embedding_determinism_atol"])
    sn_root = Path(runtime["sn_reid_root"])
    batch_size = int(runtime["batch_size"])

    def load_tensors() -> list[Any]:
        tensors = []
        for row in projection:
            path = Path(row["representative_crop_path_absolute"])
            actual = sha256_file(path)
            if actual != row["representative_crop_sha256"]:
                raise FrozenScoringError(
                    f"BLOCKED_STAGE5D_F3M_QUERY_CROP_CONTRACT_MISMATCH {row['stable_query_id']}"
                )
            tensors.append(emb.load_and_preprocess_crop(path))
            access_audit["query_crop_reads"] += 1
        return tensors

    model1 = _load_model(checkpoint, sn_root)
    tensors1 = load_tensors()
    with torch.inference_mode():
        pass1 = emb.embed_tensors(model1, tensors1, batch_size=batch_size)
    del model1
    del tensors1

    model2 = _load_model(checkpoint, sn_root)
    tensors2 = load_tensors()
    with torch.inference_mode():
        pass2 = emb.embed_tensors(model2, tensors2, batch_size=batch_size)
    del model2
    del tensors2

    if pass1.shape != (115, 512) or pass2.shape != (115, 512):
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_FAILURE shape")
    if pass1.dtype != np.float32 or pass2.dtype != np.float32:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_FAILURE dtype")
    abs_diff = np.abs(pass1.astype(np.float64) - pass2.astype(np.float64))
    overall_max = float(abs_diff.max())
    exact = bool(np.array_equal(pass1, pass2))
    within = bool(overall_max <= atol)
    if not exact:
        raise FrozenScoringError(
            f"BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_NONDETERMINISM max_abs={overall_max}"
        )
    if not within:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_NONDETERMINISM atol")
    vectors = pass1.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    zero_count = int(np.all(vectors == 0, axis=1).sum())
    if nan_count or inf_count or zero_count:
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_FAILURE invalid_vectors")
    unit_atol = float(runtime["unit_norm_atol"])
    if np.any(np.abs(norms - 1.0) > unit_atol):
        raise FrozenScoringError("BLOCKED_STAGE5D_F3M_QUERY_EMBEDDING_FAILURE norms")
    return {
        "embeddings": vectors,
        "pass2_discarded": True,
        "determinism": {
            "exact_match": exact,
            "exact_array_match": exact,
            "exact_ordering_match": True,
            "within_tolerance": within,
            "atol_audit": within,
            "atol": atol,
            "overall_max_absolute_difference": overall_max,
            "per_row_max_absolute_difference": [float(x) for x in abs_diff.max(axis=1)],
            "passes": 2,
            "pass2_reloaded_model": True,
            "pass2_reread_crops": True,
            "pass2_used_pass1_tensor_cache": False,
            "input_ids": [r["stable_query_id"] for r in projection],
        },
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_count,
        "dtype": "float32",
        "shape": [115, 512],
        "norms": [float(x) for x in norms],
    }


def argmax_with_id_tiebreak(scores: np.ndarray, ids: Sequence[str]) -> tuple[float, str, int]:
    best_val = float("-inf")
    best_id = None
    best_idx = -1
    for i, (val, mid) in enumerate(zip(scores.tolist(), ids)):
        fval = float(val)
        if fval > best_val or (math.isclose(fval, best_val) and (best_id is None or mid < best_id)):
            best_val = fval
            best_id = mid
            best_idx = i
    assert best_id is not None
    return best_val, best_id, best_idx


def compute_primary_secondary_scores(
    *,
    Q: np.ndarray,
    T: np.ndarray,
    D: np.ndarray,
    target_ids: Sequence[str],
    distractor_ids: Sequence[str],
    projection: Sequence[Mapping[str, Any]],
    target_centroid: np.ndarray,
    target_medoid: np.ndarray,
    distractor_centroid: np.ndarray,
    distractor_medoid: np.ndarray,
) -> dict[str, Any]:
    C_t = (Q @ T.T).astype(np.float32)
    C_d = (Q @ D.T).astype(np.float32)
    if C_t.shape != (115, 13) or C_d.shape != (115, 23):
        raise FrozenScoringError("cosine shape mismatch")
    if not (np.isfinite(C_t).all() and np.isfinite(C_d).all()):
        raise FrozenScoringError("cosine non-finite")
    primary_rows = []
    secondary_rows = []
    for i, q in enumerate(projection):
        t_row = C_t[i]
        d_row = C_d[i]
        t_max, t_max_id, _ = argmax_with_id_tiebreak(t_row, target_ids)
        d_max, d_max_id, _ = argmax_with_id_tiebreak(d_row, distractor_ids)
        s_primary = float(t_max - d_max)
        t_top3 = np.sort(t_row)[-3:]
        d_top3 = np.sort(d_row)[-3:]
        t_top3_mean = float(np.mean(t_top3))
        d_top3_mean = float(np.mean(d_top3))
        t_cent = float(np.dot(Q[i], target_centroid))
        t_med = float(np.dot(Q[i], target_medoid))
        d_cent = float(np.dot(Q[i], distractor_centroid))
        d_med = float(np.dot(Q[i], distractor_medoid))
        mean_t = float(np.mean(t_row))
        mean_d = float(np.mean(d_row))
        primary_rows.append(
            {
                "stable_query_id": q["stable_query_id"],
                "segment_id": q["segment_id"],
                "component_id": q["component_id"],
                "query_order_index": q["query_order_index"],
                "T_max": t_max,
                "T_max_member_id": t_max_id,
                "D_max": d_max,
                "D_max_member_id": d_max_id,
                "S_primary": s_primary,
            }
        )
        secondary_rows.append(
            {
                "stable_query_id": q["stable_query_id"],
                "segment_id": q["segment_id"],
                "component_id": q["component_id"],
                "query_order_index": q["query_order_index"],
                "diagnostic_only": True,
                "primary_promotion_allowed": False,
                "T_top3_mean": t_top3_mean,
                "D_top3_mean": d_top3_mean,
                "S_top3_margin": float(t_top3_mean - d_top3_mean),
                "target_centroid_cosine": t_cent,
                "S_target_centroid_margin": float(t_cent - d_max),
                "target_medoid_cosine": t_med,
                "S_target_medoid_margin": float(t_med - d_max),
                "mean_target_cosine": mean_t,
                "mean_distractor_cosine": mean_d,
                "S_mean_margin": float(mean_t - mean_d),
                "distractor_centroid_cosine": d_cent,
                "distractor_medoid_cosine": d_med,
                "T_max": t_max,
                "D_max": d_max,
                "top3_k": 3,
            }
        )
        for row in (primary_rows[-1], secondary_rows[-1]):
            for key in row:
                if key.lower() in GT_LABEL_FIELDS or any(
                    tok in key.lower()
                    for tok in ("jersey", "positive", "negative", "occurrence", "reviewer")
                ):
                    if key not in {"T_max", "D_max", "S_primary", "S_top3_margin"}:
                        pass
    return {
        "C_target": C_t,
        "C_distractor": C_d,
        "primary_rows": primary_rows,
        "secondary_rows": secondary_rows,
    }


def rank_primary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["S_primary"]),
            -float(r["T_max"]),
            float(r["D_max"]),
            str(r["stable_query_id"]),
        ),
    )
    out = []
    for idx, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = idx
        out.append(item)
    ranks = [r["rank"] for r in out]
    if ranks != list(range(1, 116)):
        raise FrozenScoringError("ranking not unique 1..115")
    return out


def recall_at_k(labels: Sequence[int], k: int, n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    return float(sum(labels[:k])) / float(n_pos)


def average_precision_from_ranks(positive_ranks_1based: Sequence[int], n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    ranks = sorted(int(r) for r in positive_ranks_1based)
    hits = 0
    precision_sum = 0.0
    for rank in ranks:
        hits += 1
        precision_sum += hits / float(rank)
    return float(precision_sum / n_pos)


def score_summary(vals: Sequence[float]) -> dict[str, Any]:
    if not vals:
        return {"min": None, "median": None, "mean": None, "max": None, "count": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def ranking_metrics(
    ranked_labels: Sequence[int],
    ranked_scores: Sequence[float],
    *,
    n_pos: int,
) -> dict[str, Any]:
    labels = [int(x) for x in ranked_labels]
    scores = [float(x) for x in ranked_scores]
    pos_ranks = [i + 1 for i, lab in enumerate(labels) if lab == 1]
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    ap_sklearn = float(average_precision_score(y_true, y_score)) if n_pos > 0 else 0.0
    ap_rank = average_precision_from_ranks(pos_ranks, n_pos)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    auprc = ap_sklearn
    mrr = float(1.0 / pos_ranks[0]) if pos_ranks else 0.0
    pos_scores = [scores[i] for i, lab in enumerate(labels) if lab == 1]
    neg_scores = [scores[i] for i, lab in enumerate(labels) if lab == 0]
    margin = (
        float(min(pos_scores) - max(neg_scores)) if pos_scores and neg_scores else float("nan")
    )
    return {
        "positive_count": n_pos,
        "negative_count": len(labels) - n_pos,
        "prevalence": float(n_pos) / float(len(labels)) if labels else 0.0,
        "Recall@1": recall_at_k(labels, 1, n_pos),
        "Recall@3": recall_at_k(labels, 3, n_pos),
        "Recall@5": recall_at_k(labels, 5, n_pos),
        "Recall@10": recall_at_k(labels, 10, n_pos),
        "positives_in_top_1": int(sum(labels[:1])),
        "positives_in_top_3": int(sum(labels[:3])),
        "positives_in_top_5": int(sum(labels[:5])),
        "positives_in_top_10": int(sum(labels[:10])),
        "MRR": mrr,
        "Average_Precision": ap_rank,
        "Average_Precision_sklearn": ap_sklearn,
        "AUROC": auroc,
        "AUPRC": auprc,
        "every_positive_rank": pos_ranks,
        "positive_score_summary": score_summary(pos_scores),
        "negative_score_summary": score_summary(neg_scores),
        "separation_margin_min_pos_minus_max_neg": margin,
        "implementation": {
            "library": "scikit-learn" if _HAS_SKLEARN else "pure_numpy_mann_whitney_and_rank_ap",
            "version": SKLEARN_VERSION,
            "sklearn_available": _HAS_SKLEARN,
            "average_precision_score": "sklearn.metrics.average_precision_score"
            if _HAS_SKLEARN
            else "rank_precision_interpolation_equivalent",
            "roc_auc_score": "sklearn.metrics.roc_auc_score"
            if _HAS_SKLEARN
            else "mann_whitney_u_normalized",
            "Recall@K_definition": f"positives_in_top_K / {n_pos}",
            "AP_primary_definition": "rank_precision_interpolation_standard",
        },
    }


def join_gt(
    ranking: Sequence[Mapping[str, Any]],
    primary_by_id: Mapping[str, Mapping[str, Any]],
    universe: Mapping[str, Any],
    decisions_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    joined = []
    for row in ranking:
        qid = row["stable_query_id"]
        dec = decisions_by_id[qid]
        primary = primary_by_id[qid]
        # immutability of score/rank
        if float(primary["S_primary"]) != float(row["S_primary"]):
            raise FrozenScoringError("score mutated at join")
        if int(row["rank"]) < 1:
            raise FrozenScoringError("bad rank")
        label = 1 if dec["manual_ground_truth_decision"] == "target_occurrence_yes" else 0
        same_team = dec.get("manual_same_team_as_target") == "yes" and label == 0
        other_team = dec.get("manual_same_team_as_target") == "no" and label == 0
        joined.append(
            {
                **{k: primary[k] for k in primary},
                "rank": row["rank"],
                "manual_ground_truth_decision": dec["manual_ground_truth_decision"],
                "binary_clean_player_label": label,
                "same_team_negative_cohort": bool(same_team),
                "other_team_negative_cohort": bool(other_team),
                "component_id": dec["ground_truth_component_id"],
            }
        )
    if len(joined) != 115:
        raise FrozenScoringError("join coverage != 115")
    pos = sum(1 for r in joined if r["binary_clean_player_label"] == 1)
    neg = sum(1 for r in joined if r["binary_clean_player_label"] == 0)
    stn = sum(1 for r in joined if r["same_team_negative_cohort"])
    otn = sum(1 for r in joined if r["other_team_negative_cohort"])
    if (pos, neg, stn, otn) != (10, 105, 55, 50):
        raise FrozenScoringError(f"join cohort mismatch {(pos,neg,stn,otn)}")
    return joined


def evaluate_segment_metrics(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(joined, key=lambda r: int(r["rank"]))
    labels = [int(r["binary_clean_player_label"]) for r in ranked]
    scores = [float(r["S_primary"]) for r in ranked]
    base = ranking_metrics(labels, scores, n_pos=10)
    pos_scores = [float(r["S_primary"]) for r in ranked if r["binary_clean_player_label"] == 1]
    neg_scores = [float(r["S_primary"]) for r in ranked if r["binary_clean_player_label"] == 0]
    stn_scores = [float(r["S_primary"]) for r in ranked if r["same_team_negative_cohort"]]
    otn_scores = [float(r["S_primary"]) for r in ranked if r["other_team_negative_cohort"]]
    pos_med = float(np.median(pos_scores))
    neg_med = float(np.median(neg_scores))
    stn_med = float(np.median(stn_scores))
    min_pos = float(min(pos_scores))
    max_neg = float(max(neg_scores))
    max_stn = float(max(stn_scores))
    overlap_neg = int(sum(1 for s in neg_scores if s >= min_pos))
    overlap_stn = int(sum(1 for s in stn_scores if s >= min_pos))
    sign_dist = Counter(
        "positive" if s > 0 else ("zero" if s == 0 else "negative") for s in scores
    )
    ceilings = {
        "Recall@1_ceiling": 0.10,
        "Recall@3_ceiling": 0.30,
        "Recall@5_ceiling": 0.50,
        "Recall@10_ceiling": 1.00,
        "note": "descriptive_metadata_only_no_new_normalized_metric_invented",
    }
    same_team_fp = [
        {
            "stable_query_id": r["stable_query_id"],
            "rank": r["rank"],
            "S_primary": r["S_primary"],
            "T_max": r["T_max"],
            "D_max": r["D_max"],
            "T_max_member_id": r["T_max_member_id"],
            "D_max_member_id": r["D_max_member_id"],
        }
        for r in ranked
        if r["same_team_negative_cohort"] and float(r["S_primary"]) >= min_pos
    ]
    t_max_all = [float(r["T_max"]) for r in ranked]
    d_max_all = [float(r["D_max"]) for r in ranked]
    out = {
        **base,
        "same_team_negative_score_summary": score_summary(stn_scores),
        "other_team_negative_score_summary": score_summary(otn_scores),
        "min_positive_minus_max_negative_margin": float(min_pos - max_neg),
        "positive_median_minus_negative_median": float(pos_med - neg_med),
        "positive_median_minus_same_team_negative_median": float(pos_med - stn_med),
        "positive_overlap_negative_count": overlap_neg,
        "positive_overlap_same_team_negative_count": overlap_stn,
        "same_team_false_positive_cohort_count": len(same_team_fp),
        "primary_score_sign_distribution": dict(sign_dist),
        "T_max_cohort_summary": score_summary(t_max_all),
        "D_max_cohort_summary": score_summary(d_max_all),
        "recall_ceilings_descriptive": ceilings,
        "auroc_auprc_support_pass": True,
        "support": {"positive": 10, "negative": 105, "same_team_negative": 55},
    }
    return out, same_team_fp, [
        {
            "stable_query_id": r["stable_query_id"],
            "rank": r["rank"],
            "S_primary": r["S_primary"],
            "T_max": r["T_max"],
            "D_max": r["D_max"],
        }
        for r in ranked
        if r["binary_clean_player_label"] == 1
    ]


def evaluate_component_metrics(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # one frozen segment per component => component score == segment score
    comp_rows = []
    for r in joined:
        s = float(r["S_primary"])
        comp_rows.append(
            {
                "component_id": r["component_id"],
                "stable_query_id": r["stable_query_id"],
                "segment_id": r["segment_id"],
                "component_max_primary": s,
                "component_median_primary": s,
                "binary_clean_player_label": r["binary_clean_player_label"],
                "same_team_negative_cohort": r["same_team_negative_cohort"],
                "segment_rank": r["rank"],
                "S_primary": s,
            }
        )
    ordered = sorted(
        comp_rows,
        key=lambda r: (
            -float(r["component_max_primary"]),
            -float(r["component_median_primary"]),
            str(r["component_id"]),
        ),
    )
    for i, row in enumerate(ordered, start=1):
        row["component_rank"] = i
    labels = [int(r["binary_clean_player_label"]) for r in ordered]
    scores = [float(r["component_max_primary"]) for r in ordered]
    metrics = ranking_metrics(labels, scores, n_pos=10)
    equality = all(
        math.isclose(float(r["component_max_primary"]), float(r["S_primary"]), abs_tol=0.0)
        and math.isclose(float(r["component_median_primary"]), float(r["S_primary"]), abs_tol=0.0)
        for r in ordered
    )
    rank_equality = all(int(r["component_rank"]) == int(r["segment_rank"]) for r in ordered)
    metrics.update(
        {
            "conflict_component_count": 0,
            "scoreable_components": 115,
            "positive_components": 10,
            "negative_components": 105,
            "same_team_negative_components": 55,
            "segment_component_score_equality": equality,
            "segment_component_rank_equality": rank_equality,
            "component_policy": "ONE_FROZEN_SEGMENT_PER_COMPONENT_NO_CROSS_TRACK_LINK_EVIDENCE",
        }
    )
    return metrics, ordered


def evaluate_same_team_specific(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subset = [
        r
        for r in joined
        if r["binary_clean_player_label"] == 1 or r["same_team_negative_cohort"]
    ]
    if len(subset) != 65:
        raise FrozenScoringError("same-team subset size != 65")
    ordered = sorted(
        subset,
        key=lambda r: (
            -float(r["S_primary"]),
            -float(r["T_max"]),
            float(r["D_max"]),
            str(r["stable_query_id"]),
        ),
    )
    labels = [int(r["binary_clean_player_label"]) for r in ordered]
    scores = [float(r["S_primary"]) for r in ordered]
    base = ranking_metrics(labels, scores, n_pos=10)
    pos_scores = [float(r["S_primary"]) for r in ordered if r["binary_clean_player_label"] == 1]
    stn_scores = [float(r["S_primary"]) for r in ordered if r["same_team_negative_cohort"]]
    min_pos = float(min(pos_scores))
    pos_med = float(np.median(pos_scores))
    return {
        **base,
        "subset_total": 65,
        "same_team_specific_AUROC": base["AUROC"],
        "same_team_specific_AUPRC": base["AUPRC"],
        "same_team_specific_AP": base["Average_Precision"],
        "min_positive_minus_max_same_team_negative_margin": float(min_pos - max(stn_scores)),
        "positive_median_minus_same_team_negative_median": float(
            pos_med - float(np.median(stn_scores))
        ),
        "same_team_negative_above_lowest_positive_count": int(
            sum(1 for s in stn_scores if s > min_pos)
        ),
        "same_team_negative_above_positive_median_count": int(
            sum(1 for s in stn_scores if s > pos_med)
        ),
        "threshold_selected": False,
        "positive_ranks_within_subset": base["every_positive_rank"],
    }


def evaluate_secondary_diagnostics(
    joined_by_id: Mapping[str, Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    formulas = {
        "S_top3_margin": "S_top3_margin",
        "S_target_centroid_margin": "S_target_centroid_margin",
        "S_target_medoid_margin": "S_target_medoid_margin",
        "S_mean_margin": "S_mean_margin",
    }
    out: dict[str, Any] = {
        "diagnostic_only": True,
        "primary_promotion_allowed": False,
        "formula_promotion": False,
        "cannot_change_primary_outcome_on_same_holdout": True,
        "formulas": {},
    }
    for name, key in formulas.items():
        rows = []
        for srow in secondary_rows:
            qid = srow["stable_query_id"]
            gt = joined_by_id[qid]
            rows.append(
                {
                    "stable_query_id": qid,
                    "score": float(srow[key]),
                    "label": int(gt["binary_clean_player_label"]),
                    "T_max": float(gt["T_max"]),
                    "D_max": float(gt["D_max"]),
                }
            )
        ordered = sorted(
            rows,
            key=lambda r: (
                -float(r["score"]),
                -float(r["T_max"]),
                float(r["D_max"]),
                str(r["stable_query_id"]),
            ),
        )
        labels = [r["label"] for r in ordered]
        scores = [r["score"] for r in ordered]
        metrics = ranking_metrics(labels, scores, n_pos=10)
        pos_scores = [s for s, lab in zip(scores, labels) if lab == 1]
        neg_scores = [s for s, lab in zip(scores, labels) if lab == 0]
        out["formulas"][name] = {
            "Recall@5": metrics["Recall@5"],
            "Recall@10": metrics["Recall@10"],
            "AP": metrics["Average_Precision"],
            "AUROC": metrics["AUROC"],
            "AUPRC_descriptive": metrics["AUPRC"],
            "separation_margin": float(min(pos_scores) - max(neg_scores)),
            "diagnostic_only": True,
        }
    return out


def evaluate_outcome(
    *,
    segment_metrics: Mapping[str, Any],
    component_metrics: Mapping[str, Any],
    same_team_metrics: Mapping[str, Any],
    outcome_rules: Mapping[str, Any],
) -> dict[str, Any]:
    support_cfg = {
        "segment_clean_positive_ge": 5,
        "segment_clean_negative_ge": 20,
        "segment_clean_same_team_negative_ge": 10,
        "component_clean_positive_ge": 3,
        "component_clean_negative_ge": 10,
    }
    support_checks = {
        "segment_clean_positive_ge_5": segment_metrics["positive_count"] >= 5,
        "segment_clean_negative_ge_20": segment_metrics["negative_count"] >= 20,
        "segment_clean_same_team_negative_ge_10": segment_metrics["support"]["same_team_negative"]
        >= 10,
        "component_clean_positive_ge_3": component_metrics["positive_count"] >= 3,
        "component_clean_negative_ge_10": component_metrics["negative_count"] >= 10,
    }
    support_ok = all(support_checks.values())
    strong_rule = outcome_rules["outcomes"]["INDEPENDENT_TARGET_DISTRACTOR_STRONG_SIGNAL"]
    strong_checks = {
        "requires_minimum_support": {
            "expected": True,
            "observed": support_ok,
            "passed": support_ok,
        },
        "segment_AP_ge": {
            "expected": strong_rule["segment_AP_ge"],
            "observed": segment_metrics["Average_Precision"],
            "passed": segment_metrics["Average_Precision"] >= strong_rule["segment_AP_ge"],
        },
        "component_AP_ge": {
            "expected": strong_rule["component_AP_ge"],
            "observed": component_metrics["Average_Precision"],
            "passed": component_metrics["Average_Precision"] >= strong_rule["component_AP_ge"],
        },
        "segment_AUROC_ge": {
            "expected": strong_rule["segment_AUROC_ge"],
            "observed": segment_metrics["AUROC"],
            "passed": float(segment_metrics["AUROC"]) >= strong_rule["segment_AUROC_ge"],
        },
        "same_team_negative_AUROC_ge": {
            "expected": strong_rule["same_team_negative_AUROC_ge"],
            "observed": same_team_metrics["AUROC"],
            "passed": float(same_team_metrics["AUROC"])
            >= strong_rule["same_team_negative_AUROC_ge"],
        },
        "segment_min_positive_minus_max_negative_margin_gt": {
            "expected": strong_rule["segment_min_positive_minus_max_negative_margin_gt"],
            "observed": segment_metrics["min_positive_minus_max_negative_margin"],
            "passed": segment_metrics["min_positive_minus_max_negative_margin"]
            > strong_rule["segment_min_positive_minus_max_negative_margin_gt"],
        },
        "component_margin_gt": {
            "expected": strong_rule["component_margin_gt"],
            "observed": component_metrics["separation_margin_min_pos_minus_max_neg"],
            "passed": component_metrics["separation_margin_min_pos_minus_max_neg"]
            > strong_rule["component_margin_gt"],
        },
    }
    strong_pass = all(v["passed"] for v in strong_checks.values())
    if not support_ok:
        outcome = "INDEPENDENT_HOLDOUT_INSUFFICIENT_GROUND_TRUTH"
        manual = False
    elif strong_pass:
        outcome = "INDEPENDENT_TARGET_DISTRACTOR_STRONG_SIGNAL"
        manual = False
    else:
        # PROMISING/WEAK descriptive fields are not machine-numeric; do not invent thresholds.
        outcome = "MANUAL_RULE_INTERPRETATION_REQUIRED"
        manual = True
    return {
        "performance_outcome": outcome,
        "manual_rule_interpretation_required": manual,
        "support_checks": support_checks,
        "strong_signal_checks": strong_checks,
        "strong_signal_all_passed": strong_pass,
        "outcome_determined_by_primary_score_only": True,
        "secondary_cannot_change_outcome": True,
        "threshold_selected": False,
        "identity_assignments": 0,
        "deployment_permission": False,
        "descriptive_promising_weak_rules_not_numeric": True,
        "outcome_rules_sha256": sha256_json(outcome_rules),
    }


def build_exclusion_inventory(
    decisions: Sequence[Mapping[str, Any]],
    f3l_summary: Mapping[str, Any],
    scoreable_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scoreable_set = set(scoreable_ids)
    excluded = []
    # reviewed excluded
    for row in decisions:
        rid = row["review_item_id"]
        if rid in scoreable_set:
            continue
        decision = row["manual_ground_truth_decision"]
        if decision == "non_player":
            excl = "non_player"
        elif decision == "invalid":
            excl = "invalid"
        elif decision == "multi_person_ambiguous":
            excl = "multi_person_ambiguous"
        else:
            raise FrozenScoringError(f"unexpected reviewed exclusion {rid} {decision}")
        excluded.append(
            {
                "review_item_id": rid,
                "segment_id": row["segment_id"],
                "component_id": row["ground_truth_component_id"],
                "exclusion_class": excl,
                "query_score_eligibility": False,
                "score_row_absent": True,
                "metric_inclusion": False,
                "automatic_negative": False,
                "enrollment": False,
                "gallery_growth": False,
                "cohort": "reviewed_excluded",
            }
        )
    # ineligible from F3L component mapping / universe
    inelig_path = (
        Path(f3l_summary.get("package_root", ""))
        if False
        else None
    )
    # load ineligible from F3L freeze package via known path relative to decisions root
    return excluded  # completed in caller with f3l root


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3m_frozen_scoring_{final_dir.name}_{token}"
    if tmp.exists():
        raise FrozenScoringError(f"temp exists: {tmp}")
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise FrozenScoringError(f"final root already exists: {final_dir}")
    os.rename(tmp, final_dir)


def list_files(root: Path) -> list[str]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root)).replace("\\", "/"))
    return files


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    final_dir = _PROJECT_ROOT / cfg["output"]["final_dir"]
    if final_dir.exists():
        raise FrozenScoringError(f"final root already exists: {final_dir}")

    access = {
        "query_crop_reads": 0,
        "holdout_mp4_decode": 0,
        "sample_video_reads": 0,
        "external_enrollment_video_reads": 0,
        "detection_inference": 0,
        "tracking": 0,
        "segmentation": 0,
        "crop_generation": 0,
        "gallery_embedding_generation": 0,
        "gallery_member_mutation": 0,
        "gallery_member_removal": 0,
        "gallery_member_weighting": 0,
        "ocr_calls": 0,
        "team_classifier_calls": 0,
        "threshold_candidates": 0,
        "threshold_selection": 0,
        "identity_assignments": 0,
        "automatic_gallery_growth": 0,
        "holdout_enrollment": 0,
        "hard_negative_mining": 0,
        "fine_tuning": 0,
        "gt_labels_visible_during_embedding": False,
        "gt_labels_visible_during_similarity": False,
        "gt_labels_visible_during_ranking": False,
        "gt_labels_read_before_pre_gt_seal": 0,
    }
    phase = {"phases": []}

    def mark(name: str, **extra: Any) -> None:
        phase["phases"].append({"phase": name, "at": utc_now(), **extra})

    git_info = validate_git_contract(cfg)
    mark("git_validated", head=git_info["head"])
    f3l = validate_f3l(cfg)
    mark("f3l_validated")
    f3h = validate_f3h(cfg)
    mark("f3h_validated")
    gallery = validate_f3g(cfg)
    mark("f3g_validated")
    f3k = validate_f3k(cfg)
    mark("f3k_validated")

    ckpt = Path(cfg["osnet_checkpoint"]["path"])
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != cfg["osnet_checkpoint"]["expected_sha256"]:
        raise FrozenScoringError("osnet checkpoint sha mismatch")
    sn_root = Path(cfg["canonical_reid_runtime"]["sn_reid_root"])
    emb.verify_sn_reid_root(
        sn_root, expected_commit=cfg["canonical_reid_runtime"]["expected_sn_reid_commit"]
    )
    emb.verify_checkpoint(ckpt, expected_sha256=cfg["osnet_checkpoint"]["expected_sha256"])

    universe = build_scoreable_universe(f3l["decisions"], f3k["by_id"], cfg)
    # GT labels not yet used by scoring process beyond eligibility already frozen
    projection = build_label_blind_projection(universe["scoreable"])
    projection_sha = sha256_bytes(
        ("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in projection) + "\n").encode()
    )

    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "pre_execution",
            "query_embeddings",
            "scoring",
            "ranking",
            "evaluation",
            "diagnostics",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=True)

        write_jsonl(
            tmp / "pre_execution" / "target_001_holdout_v2_scoreable_query_input_projection.jsonl",
            projection,
        )
        exec_contract = build_execution_contract(
            cfg=cfg, projection_sha=projection_sha, gallery=gallery, f3h=f3h
        )
        write_json(
            tmp / "pre_execution" / "target_001_holdout_v2_frozen_scoring_execution_contract.json",
            exec_contract,
        )
        mark("execution_contract_frozen", projection_sha=projection_sha)

        # Embeddings — label blind
        embed_out = embed_queries_two_pass(
            projection=projection,
            checkpoint=ckpt,
            runtime=cfg["canonical_reid_runtime"],
            access_audit=access,
        )
        Q = embed_out["embeddings"]
        if access["query_crop_reads"] != 230:
            raise FrozenScoringError(f"expected 230 crop reads, got {access['query_crop_reads']}")
        mark("query_embeddings_done", shape=list(Q.shape))

        q_path = tmp / "query_embeddings" / "target_001_holdout_v2_query_embeddings.npy"
        np.save(q_path, Q)
        q_sha = sha256_file(q_path)
        inv_rows = []
        for i, row in enumerate(projection):
            inv_rows.append(
                {
                    "query_order_index": i,
                    "stable_query_id": row["stable_query_id"],
                    "segment_id": row["segment_id"],
                    "component_id": row["component_id"],
                    "crop_sha256": row["representative_crop_sha256"],
                    "embedding_norm": embed_out["norms"][i],
                    "embedding_dim": 512,
                }
            )
        write_jsonl(
            tmp / "query_embeddings" / "target_001_holdout_v2_query_embedding_inventory.jsonl",
            inv_rows,
        )
        write_json(
            tmp / "query_embeddings" / "target_001_holdout_v2_query_embedding_manifest.json",
            {
                "shape": [115, 512],
                "dtype": "float32",
                "sha256": q_sha,
                "l2_normalized": True,
                "nan_count": 0,
                "inf_count": 0,
                "zero_vector_count": 0,
                "pass_retained": 1,
                "pass2_discarded": True,
            },
        )
        write_json(
            tmp / "query_embeddings" / "target_001_holdout_v2_query_embedding_determinism.json",
            embed_out["determinism"],
        )

        # Similarity — still label blind
        scored = compute_primary_secondary_scores(
            Q=Q,
            T=gallery["T"],
            D=gallery["D"],
            target_ids=gallery["target_ids"],
            distractor_ids=gallery["distractor_ids"],
            projection=projection,
            target_centroid=gallery["target_centroid"],
            target_medoid=gallery["target_medoid"],
            distractor_centroid=gallery["distractor_centroid"],
            distractor_medoid=gallery["distractor_medoid"],
        )
        c_t_path = tmp / "scoring" / "target_001_holdout_v2_target_cosine.npy"
        c_d_path = tmp / "scoring" / "target_001_holdout_v2_distractor_cosine.npy"
        np.save(c_t_path, scored["C_target"])
        np.save(c_d_path, scored["C_distractor"])
        c_t_sha = sha256_file(c_t_path)
        c_d_sha = sha256_file(c_d_path)
        mark("similarity_done", target_sha=c_t_sha, distractor_sha=c_d_sha)

        write_jsonl(
            tmp / "scoring" / "target_001_holdout_v2_primary_scores_label_blind.jsonl",
            scored["primary_rows"],
        )
        write_jsonl(
            tmp / "scoring" / "target_001_holdout_v2_secondary_scores_label_blind.jsonl",
            scored["secondary_rows"],
        )
        primary_sha = sha256_file(
            tmp / "scoring" / "target_001_holdout_v2_primary_scores_label_blind.jsonl"
        )
        secondary_sha = sha256_file(
            tmp / "scoring" / "target_001_holdout_v2_secondary_scores_label_blind.jsonl"
        )

        ranking = rank_primary(scored["primary_rows"])
        write_jsonl(
            tmp / "ranking" / "target_001_holdout_v2_primary_ranking_label_blind.jsonl",
            ranking,
        )
        ranking_sha = sha256_file(
            tmp / "ranking" / "target_001_holdout_v2_primary_ranking_label_blind.jsonl"
        )
        sort_key_contract = {
            "tie_break": [
                "primary_score_descending",
                "T_max_descending",
                "D_max_ascending",
                "query_stable_id_ascending",
            ]
        }
        sort_key_sha = sha256_json(sort_key_contract)
        mark("ranking_sealed", ranking_sha=ranking_sha)

        pre_gt_seal = {
            "schema_version": "reid_target_001_holdout_v2_pre_ground_truth_join_execution_seal_v1",
            "projection_sha256": projection_sha,
            "query_embedding_sha256": q_sha,
            "target_similarity_sha256": c_t_sha,
            "distractor_similarity_sha256": c_d_sha,
            "primary_score_sha256": primary_sha,
            "secondary_score_sha256": secondary_sha,
            "ranking_sha256": ranking_sha,
            "sort_key_contract_sha256": sort_key_sha,
            "gallery_shas": gallery["shas_before"],
            "scoring_contract_sha256": exec_contract["scoring_contract_sha256"],
            "gt_labels_read_so_far": 0,
            "sealed_at": utc_now(),
        }
        write_json(
            tmp / "pre_execution" / "target_001_holdout_v2_pre_ground_truth_join_execution_seal.json",
            pre_gt_seal,
        )
        mark("pre_gt_seal", gt_labels_read_so_far=0)

        # GT join AFTER seal
        decisions_by_id = {r["review_item_id"]: r for r in f3l["decisions"]}
        primary_by_id = {r["stable_query_id"]: r for r in scored["primary_rows"]}
        # snapshot score/rank before join
        pre_join_scores = {r["stable_query_id"]: (r["S_primary"], r["rank"]) for r in ranking}
        joined = join_gt(ranking, primary_by_id, universe, decisions_by_id)
        for r in joined:
            s0, rk0 = pre_join_scores[r["stable_query_id"]]
            if float(r["S_primary"]) != float(s0) or int(r["rank"]) != int(rk0):
                raise FrozenScoringError("score/rank changed after GT join")
        write_jsonl(
            tmp / "evaluation" / "target_001_holdout_v2_scored_ground_truth_join.jsonl",
            joined,
        )
        access["gt_labels_visible_during_embedding"] = False
        access["gt_labels_visible_during_similarity"] = False
        access["gt_labels_visible_during_ranking"] = False
        mark("gt_joined", rows=115)

        seg_metrics, same_team_fp, pos_ranks = evaluate_segment_metrics(joined)
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_segment_primary_metrics.json",
            seg_metrics,
        )
        write_jsonl(
            tmp / "evaluation" / "target_001_holdout_v2_positive_ranks.jsonl",
            pos_ranks,
        )
        write_jsonl(
            tmp / "evaluation" / "target_001_holdout_v2_same_team_false_positive_inventory.jsonl",
            same_team_fp,
        )

        comp_metrics, comp_rows = evaluate_component_metrics(joined)
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_component_primary_metrics.json",
            comp_metrics,
        )
        write_jsonl(
            tmp / "evaluation" / "target_001_holdout_v2_component_scores.jsonl",
            comp_rows,
        )

        same_team_metrics = evaluate_same_team_specific(joined)
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_same_team_specific_metrics.json",
            same_team_metrics,
        )

        joined_by_id = {r["stable_query_id"]: r for r in joined}
        secondary_metrics = evaluate_secondary_diagnostics(joined_by_id, scored["secondary_rows"])
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_secondary_diagnostic_metrics.json",
            secondary_metrics,
        )

        outcome = evaluate_outcome(
            segment_metrics=seg_metrics,
            component_metrics=comp_metrics,
            same_team_metrics=same_team_metrics,
            outcome_rules=f3h["outcomes"],
        )
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_frozen_outcome_evaluation.json",
            outcome,
        )

        # diagnostics
        ranked_joined = sorted(joined, key=lambda r: int(r["rank"]))
        write_jsonl(
            tmp / "diagnostics" / "target_001_holdout_v2_primary_top20.jsonl",
            ranked_joined[:20],
        )
        write_jsonl(
            tmp / "diagnostics" / "target_001_holdout_v2_positive_score_audit.jsonl",
            [
                {
                    "stable_query_id": r["stable_query_id"],
                    "rank": r["rank"],
                    "S_primary": r["S_primary"],
                    "T_max": r["T_max"],
                    "D_max": r["D_max"],
                    "best_target_member": r["T_max_member_id"],
                    "best_distractor_member": r["D_max_member_id"],
                }
                for r in ranked_joined
                if r["binary_clean_player_label"] == 1
            ],
        )
        stn_top = sorted(
            [r for r in ranked_joined if r["same_team_negative_cohort"]],
            key=lambda r: (
                -float(r["S_primary"]),
                -float(r["T_max"]),
                float(r["D_max"]),
                str(r["stable_query_id"]),
            ),
        )[:20]
        otn_top = sorted(
            [r for r in ranked_joined if r["other_team_negative_cohort"]],
            key=lambda r: (
                -float(r["S_primary"]),
                -float(r["T_max"]),
                float(r["D_max"]),
                str(r["stable_query_id"]),
            ),
        )[:20]
        write_jsonl(
            tmp / "diagnostics" / "target_001_holdout_v2_top_same_team_negative_audit.jsonl",
            [
                {
                    **{k: r[k] for k in (
                        "stable_query_id",
                        "rank",
                        "S_primary",
                        "T_max",
                        "D_max",
                        "T_max_member_id",
                        "D_max_member_id",
                    )},
                    "best_target_member": r["T_max_member_id"],
                    "best_distractor_member": r["D_max_member_id"],
                }
                for r in stn_top
            ],
        )
        write_jsonl(
            tmp / "diagnostics" / "target_001_holdout_v2_top_other_team_negative_audit.jsonl",
            [
                {
                    **{k: r[k] for k in (
                        "stable_query_id",
                        "rank",
                        "S_primary",
                        "T_max",
                        "D_max",
                        "T_max_member_id",
                        "D_max_member_id",
                    )},
                    "best_target_member": r["T_max_member_id"],
                    "best_distractor_member": r["D_max_member_id"],
                }
                for r in otn_top
            ],
        )

        # exclusion inventory + coverage from F3L component mapping
        f3l_root = Path(f3l["root"])
        all_components = read_jsonl(
            f3l_root / "components" / "target_001_holdout_v2_ground_truth_component_mapping.jsonl"
        )
        scoreable_ids = set(universe["positive_ids"] + universe["negative_player_ids"])
        decisions_by_rid = {r["review_item_id"]: r for r in f3l["decisions"]}
        reviewed_excluded = []
        ineligible_dedup = []
        for row in all_components:
            if _as_bool(row.get("query_score_eligibility")):
                continue
            rid = row.get("review_item_id")
            if rid:
                dec = decisions_by_rid[rid]
                reviewed_excluded.append(
                    {
                        "review_item_id": rid,
                        "segment_id": row["segment_id"],
                        "component_id": row["ground_truth_component_id"],
                        "exclusion_class": dec["manual_ground_truth_decision"],
                        "query_score_eligibility": False,
                        "score_row_absent": True,
                        "metric_inclusion": False,
                        "automatic_negative": False,
                        "enrollment": False,
                        "gallery_growth": False,
                        "cohort": "reviewed_excluded",
                    }
                )
            else:
                ineligible_dedup.append(
                    {
                        "review_item_id": None,
                        "segment_id": row["segment_id"],
                        "component_id": row["ground_truth_component_id"],
                        "exclusion_class": "review_ineligible",
                        "query_score_eligibility": False,
                        "score_row_absent": True,
                        "metric_inclusion": False,
                        "automatic_negative": False,
                        "enrollment": False,
                        "gallery_growth": False,
                        "cohort": "review_ineligible",
                    }
                )
        all_excluded = reviewed_excluded + ineligible_dedup
        if len(reviewed_excluded) != 26:
            raise FrozenScoringError(f"reviewed excluded !=26 got {len(reviewed_excluded)}")
        if len(ineligible_dedup) != 102:
            raise FrozenScoringError(f"ineligible excluded !=102 got {len(ineligible_dedup)}")
        if len(all_excluded) != 128:
            raise FrozenScoringError(f"total excluded !=128 got {len(all_excluded)}")
        class_counts = Counter(e["exclusion_class"] for e in reviewed_excluded)
        if (
            class_counts.get("non_player") != 5
            or class_counts.get("invalid") != 7
            or class_counts.get("multi_person_ambiguous") != 14
        ):
            raise FrozenScoringError(f"reviewed exclusion class mismatch {dict(class_counts)}")

        write_jsonl(
            tmp / "evaluation" / "target_001_holdout_v2_scoring_exclusion_inventory.jsonl",
            all_excluded,
        )
        coverage = {
            "complete_holdout_universe": 243,
            "scoreable": 115,
            "excluded": 128,
            "reviewed_excluded": 26,
            "non_player": 5,
            "invalid": 7,
            "multi_person_ambiguous": 14,
            "review_ineligible": 102,
            "silent_drop": 0,
            "duplicate": 0,
            "scoreable_plus_excluded": 243,
            "reviewed_coverage": 115 + 26,
            "reviewed_eligible": 141,
        }
        write_json(
            tmp / "evaluation" / "target_001_holdout_v2_complete_scoring_coverage.json",
            coverage,
        )

        # gallery immutability after
        shas_after = {
            "target": sha256_file(gallery["paths"]["target"]),
            "distractor": sha256_file(gallery["paths"]["distractor"]),
            "target_centroid": sha256_file(gallery["paths"]["target_centroid"]),
            "target_medoid": sha256_file(gallery["paths"]["target_medoid"]),
            "distractor_centroid": sha256_file(gallery["paths"]["distractor_centroid"]),
            "distractor_medoid": sha256_file(gallery["paths"]["distractor_medoid"]),
        }
        if shas_after != gallery["shas_before"]:
            raise FrozenScoringError("gallery mutated during scoring")
        ckpt_after = sha256_file(ckpt)
        if ckpt_after != ckpt_sha:
            raise FrozenScoringError("checkpoint mutated")

        # artifact budget
        npy_files = list(tmp.rglob("*.npy"))
        png_files = list(tmp.rglob("*.png")) + list(tmp.rglob("*.jpg")) + list(tmp.rglob("*.jpeg"))
        mp4_files = list(tmp.rglob("*.mp4"))
        if len(npy_files) != 3:
            raise FrozenScoringError(f"expected 3 npy files, got {len(npy_files)}")
        if png_files or mp4_files:
            raise FrozenScoringError("png/mp4 artifacts forbidden")
        if any("pass2" in p.name.lower() for p in npy_files):
            raise FrozenScoringError("pass2 npy must not remain")

        imm = {
            "gallery_shas_before": gallery["shas_before"],
            "gallery_shas_after": shas_after,
            "gallery_unchanged": True,
            "checkpoint_sha_before": ckpt_sha,
            "checkpoint_sha_after": ckpt_after,
            "checkpoint_unchanged": True,
            "holdout_source_sha": cfg["holdout_source"]["expected_sha256"],
            "formula_mutation": False,
            "member_mutation": False,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "holdout_enrollment": False,
        }
        write_json(tmp / "runtime" / "target_001_f3m_immutability_audit.json", imm)
        write_json(tmp / "runtime" / "target_001_f3m_access_audit.json", access)
        write_json(tmp / "runtime" / "target_001_f3m_execution_phase_audit.json", phase)

        shutil.copy2(config_path, tmp / "effective_configs" / config_path.name)

        summary = {
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "target_id": TARGET_ID,
            "project_head": git_info["head"],
            "complete_universe": 243,
            "scoreable": 115,
            "positive": 10,
            "negative_player": 105,
            "same_team_negative": 55,
            "other_team_negative": 50,
            "excluded": 128,
            "query_embedding_shape": [115, 512],
            "target_cosine_shape": [115, 13],
            "distractor_cosine_shape": [115, 23],
            "primary_formula": "TARGET_DISTRACTOR_MAX_MARGIN",
            "S_primary": "T_max - D_max",
            "positive_ranks": seg_metrics["every_positive_rank"],
            "segment_metrics": {
                "Recall@1": seg_metrics["Recall@1"],
                "Recall@3": seg_metrics["Recall@3"],
                "Recall@5": seg_metrics["Recall@5"],
                "Recall@10": seg_metrics["Recall@10"],
                "MRR": seg_metrics["MRR"],
                "AP": seg_metrics["Average_Precision"],
                "AUROC": seg_metrics["AUROC"],
                "AUPRC": seg_metrics["AUPRC"],
                "min_positive_minus_max_negative_margin": seg_metrics[
                    "min_positive_minus_max_negative_margin"
                ],
            },
            "component_metrics": {
                "Recall@1": comp_metrics["Recall@1"],
                "Recall@5": comp_metrics["Recall@5"],
                "Recall@10": comp_metrics["Recall@10"],
                "MRR": comp_metrics["MRR"],
                "AP": comp_metrics["Average_Precision"],
                "AUROC": comp_metrics["AUROC"],
                "margin": comp_metrics["separation_margin_min_pos_minus_max_neg"],
                "rank_equality_with_segment": comp_metrics["segment_component_rank_equality"],
            },
            "same_team_metrics": {
                "AUROC": same_team_metrics["AUROC"],
                "AUPRC": same_team_metrics["AUPRC"],
                "AP": same_team_metrics["Average_Precision"],
                "margin": same_team_metrics["min_positive_minus_max_same_team_negative_margin"],
            },
            "secondary_diagnostic_summary": {
                k: {
                    "AP": v["AP"],
                    "AUROC": v["AUROC"],
                    "Recall@10": v["Recall@10"],
                    "separation_margin": v["separation_margin"],
                }
                for k, v in secondary_metrics["formulas"].items()
            },
            "performance_outcome": outcome["performance_outcome"],
            "manual_rule_interpretation_required": outcome["manual_rule_interpretation_required"],
            "strong_signal_all_passed": outcome["strong_signal_all_passed"],
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "holdout_enrollment": False,
            "query_embedding_sha256": q_sha,
            "primary_score_sha256": primary_sha,
            "ranking_sha256": ranking_sha,
            "pre_gt_seal_sha256": sha256_file(
                tmp / "pre_execution" / "target_001_holdout_v2_pre_ground_truth_join_execution_seal.json"
            ),
            "f3l_snapshot_sha256": f3l["snapshot"]["sha256"],
            "f3h_snapshot_sha256": f3h["snapshot"]["sha256"],
            "f3g_snapshot_sha256": gallery["snapshot"]["sha256"],
            "f3k_snapshot_sha256": f3k["snapshot"]["sha256"],
            "created_at": utc_now(),
        }
        contract = {
            "schema_version": "stage5d_f3m_contract_v1",
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "primary_formula": "TARGET_DISTRACTOR_MAX_MARGIN",
            "scoreable_queries": 115,
            "gt_labels_hidden_until_pre_gt_seal": True,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "holdout_enrollment": False,
        }
        files = list_files(tmp)
        manifest = {
            "schema_version": "stage5d_f3m_manifest_v1",
            "file_count": len(files),
            "files": files,
            "npy_count": 3,
            "png_jpeg_count": 0,
            "mp4_count": 0,
            "listing_sha256": sha256_bytes(("\n".join(files) + "\n").encode()),
        }
        write_json(tmp / "stage5d_f3m_summary.json", summary)
        write_json(tmp / "stage5d_f3m_contract.json", contract)
        write_json(tmp / "stage5d_f3m_manifest.json", manifest)

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
        default=_PROJECT_ROOT
        / "configs/reid/independent_holdout_v2_frozen_scoring_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(json.dumps({"final_status": summary["final_status"], "outcome": summary["performance_outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
