#!/usr/bin/env python3
"""Stage 5D-F3C — external-only hard-negative and view-anchor review package.

Builds human review artifacts from existing external tracking/occurrence/anchor
lineage only. No sample.mp4 reads, no OSNet/OCR/similarity, no approvals,
and no gallery mutation.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.crop_select import (  # noqa: E402
    clamp_bbox_xyxy,
    float_bbox_to_int_crop,
)

CONFIG_SCHEMA = "reid_external_refinement_review_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FROZEN_CODES = ("EXT_004", "EXT_183", "EXT_198")
FINAL_STATUS = "COMPLETED_STAGE5D_F3C_TARGET_001_EXTERNAL_REFINEMENT_REVIEW_READY"
NEXT_GATE = "STAGE5D-F3D_TARGET_001_EXTERNAL_REFINEMENT_MANUAL_REVIEW_AND_FREEZE"
ALLOWED_DIRTY = {
    "scripts/run_reid_external_refinement_review_package.py",
    "configs/reid/external_refinement_review_stage5d_target_001.yaml",
    "tests/test_reid_external_refinement_review_package.py",
    "docs/setup/stage5d-target-external-hard-negative-and-view-anchor-review-package.md",
}

VIEW_TEMPLATE_FIELDS = (
    "target_view_candidate_id",
    "source_occurrence_code",
    "frame_index",
    "video_time",
    "crop_path",
    "crop_sha256",
    "source_bbox",
    "crop_bbox",
    "manual_anchor_expansion_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
OCC_TEMPLATE_FIELDS = (
    "external_candidate_code",
    "raw_track_id",
    "first_frame",
    "last_frame",
    "representative_frame",
    "observation_count",
    "representative_crop_path",
    "representative_crop_sha256",
    "manual_refinement_decision",
    "manual_same_target_as_target_001",
    "manual_same_team_as_target",
    "manual_visible_jersey_number",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_view_category",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_VIEW_DECISIONS = (
    "target_anchor_expansion_yes",
    "target_anchor_expansion_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
)
ALLOWED_OCC_DECISIONS = (
    "additional_target_occurrence_yes",
    "same_team_distractor_yes",
    "other_team_player",
    "non_player",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
)
ALLOWED_VIEW = (
    "front",
    "rear",
    "left_side",
    "right_side",
    "front_oblique",
    "rear_oblique",
    "unknown",
)
ALLOWED_SCALE = ("small", "medium", "large", "unknown")
MANUAL_BLANK_VIEW = (
    "manual_anchor_expansion_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
MANUAL_BLANK_OCC = (
    "manual_refinement_decision",
    "manual_same_target_as_target_001",
    "manual_same_team_as_target",
    "manual_visible_jersey_number",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_view_category",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)


class RefinementReviewError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        raise RefinementReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise RefinementReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise RefinementReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise RefinementReviewError("BLOCKED_STAGE5D_F3C_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise RefinementReviewError("BLOCKED_STAGE5D_F3C_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise RefinementReviewError(
                    "BLOCKED_STAGE5D_F3C_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def pad_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    fraction: float,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    px, py = fraction * bw, fraction * bh
    padded = [x1 - px, y1 - py, x2 + px, y2 + py]
    return clamp_bbox_xyxy(padded, video_width=width, video_height=height)


def hamming_hex(a: str, b: str) -> int:
    ba = bytes.fromhex(a)
    bb = bytes.fromhex(b)
    if len(ba) != len(bb):
        return 10**9
    return sum(bin(x ^ y).count("1") for x, y in zip(ba, bb))


def rank_key(item: Mapping[str, Any]) -> tuple:
    q = item["quality"]
    return (
        float(q["max_person_iou"]),
        float(q["edge_clipping_fraction"]),
        -float(q["bbox_area"]),
        -float(q["laplacian_variance"]),
        int(item["frame_index"]),
    )


def select_context_frames(
    obs_frames: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
) -> list[tuple[str, int]]:
    frames = sorted(set(int(f) for f in obs_frames))
    if not frames:
        raise RefinementReviewError("segment has no observations")

    def nearest(target: int) -> int:
        return min(frames, key=lambda f: (abs(f - target), f))

    chosen: list[tuple[str, int]] = []
    for role, target in (
        ("START", start_frame),
        ("REP", representative_frame),
        ("END", end_frame),
    ):
        fi = nearest(target)
        if any(existing == fi for _, existing in chosen):
            continue
        chosen.append((role, fi))
    return chosen


def _fit(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def ext_numeric(code: str) -> int:
    return int(code.split("_", 1)[1])


def validate_f3b(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3b_refinement_design"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3b_summary.json")
    contract = load_json(root / "stage5d_f3b_contract.json")
    plan = load_json(
        root / "refinement_design" / "target_001_external_only_refinement_plan.json"
    )
    policy = load_json(
        root / "validation_policy" / "target_001_anti_overfit_policy.json"
    )
    cfg = config["stage5d_f3b_refinement_design"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH status"
        )
    if summary.get("official_f3_descriptive_outcome") != cfg["expected_f3_outcome"]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH outcome"
        )
    if abs(float(summary["official_segment_ap"]) - float(cfg["expected_segment_ap"])) > 1e-12:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH ap"
        )
    if (
        abs(
            float(summary["official_segment_margin"])
            - float(cfg["expected_segment_margin"])
        )
        > 1e-12
    ):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH margin"
        )
    if "same_uniform_confusion" not in summary.get("primary_root_causes", []):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH root_cause"
        )
    if int(summary["gallery_members"]) != int(cfg["expected_gallery_members"]):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH gallery"
        )
    if summary.get("gallery_mutation") is not False:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH mutation"
        )
    if summary.get("threshold_selected") is not False:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH threshold"
        )
    if int(summary["identity_assignments"]) != 0:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH identity"
        )
    if summary.get("new_independent_holdout_required") is not True:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH holdout"
        )
    if int(plan["priorities"][0]["priority"]) != 1:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH priority1"
        )
    if "hard_negative" not in str(plan["priorities"][0].get("item", "")):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH priority1_item"
        )
    if int(plan["priorities"][1]["priority"]) != 2:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH priority2"
        )
    if policy.get("sample_crops_forbidden_for_gallery_enrollment") is not True:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH sample_enroll"
        )
    if policy.get("sample_negatives_forbidden_for_hard_negative_training") is not True:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_REFINEMENT_DESIGN_CONTRACT_MISMATCH sample_hn"
        )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "plan": plan,
        "policy": policy,
        "snapshot_sha256": snap,
    }


def validate_external_source(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rel = config["external_enrollment_source"]["path"]
    assert_no_path_traversal(rel)
    path = project_root / rel
    if not path.is_file() or path.is_symlink():
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH external_file"
        )
    exp = config["external_enrollment_source"]
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH external_bytes"
        )
    if sha256_file(path) != exp["expected_sha256"]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH external_sha"
        )
    if exp.get("enrollment_only") is not True:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH enrollment_only"
        )
    if int(exp["sample_overlap"]) != 0:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH sample_overlap"
        )
    if exp.get("future_evaluation_input") is not False:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH future_eval"
        )
    return {
        "path": path,
        "rel": rel,
        "sha256": exp["expected_sha256"],
        "bytes": int(exp["expected_bytes"]),
        "width": int(exp["expected_width"]),
        "height": int(exp["expected_height"]),
        "fps": float(exp["expected_fps"]),
        "frames": int(exp["expected_frames"]),
    }


def validate_b1eb(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_b1e_b_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_b1e_b_summary.json")
    exp = config["stage5d_b1e_b_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH b1eb_status"
        )
    checks = {
        "detection_total": "expected_detection_total",
        "detection_frames_with_boxes": "expected_detection_frames_with_boxes",
        "tracking_total_observations": "expected_tracking_observations",
        "raw_track_count": "expected_raw_track_count",
        "ext_candidate_count": "expected_ext_candidate_count",
        "review_eligible_candidate_count": "expected_review_eligible_count",
    }
    for got_k, exp_k in checks.items():
        if int(summary[got_k]) != int(exp[exp_k]):
            raise RefinementReviewError(
                f"BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH {got_k}"
            )
    if summary.get("two_replay_determinism") is not True:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH replay"
        )
    mapping = load_jsonl(
        root / "inventory" / "target_001_external_track_candidate_mapping.jsonl"
    )
    if len(mapping) != 248:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH mapping_count"
        )
    eligible = [r for r in mapping if r.get("review_eligible") is True]
    if len(eligible) != 138:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH eligible"
        )
    return {"root": root, "summary": summary, "mapping": mapping, "eligible": eligible}


def validate_b1ec(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_b1e_c_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_b1e_c_summary.json")
    if summary.get("final_status") != config["stage5d_b1e_c_package"][
        "expected_final_status"
    ]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH b1ec_status"
        )
    if int(summary["selected_positive_count"]) != 3:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH positives"
        )
    if tuple(summary["selected_external_candidate_codes"]) != FROZEN_CODES:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH codes"
        )
    expected = {
        "EXT_004": (11, 0, 186, 185),
        "EXT_183": (388, 456, 499, 44),
        "EXT_198": (450, 511, 783, 272),
    }
    for item in summary["observation_ranges"]:
        code = item["external_candidate_code"]
        got = (
            int(item["resolved_raw_track_id"]),
            int(item["first_frame"]),
            int(item["last_frame"]),
            int(item["observation_count"]),
        )
        if got != expected[code]:
            raise RefinementReviewError(
                f"BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH lineage {code}"
            )
    return {"root": root, "summary": summary}


def validate_b1ed_b1ee(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    d_rel = config["stage5d_b1e_d_package"]["path"]
    e_rel = config["stage5d_b1e_e_package"]["path"]
    assert_no_path_traversal(d_rel)
    assert_no_path_traversal(e_rel)
    d_root = project_root / d_rel
    e_root = project_root / e_rel
    quality_rows = load_jsonl(
        d_root / "inventory" / "target_001_external_tracklet_observation_quality.jsonl"
    )
    candidates = load_jsonl(
        d_root / "inventory" / "target_001_external_anchor_candidate_inventory.jsonl"
    )
    if len(quality_rows) != int(
        config["stage5d_b1e_d_package"]["expected_observation_quality_rows"]
    ):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH quality_rows"
        )
    if len(candidates) != int(config["stage5d_b1e_d_package"]["expected_candidate_count"]):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH candidates"
        )
    e_summary = load_json(e_root / "stage5d_b1e_e_summary.json")
    if e_summary.get("final_status") != config["stage5d_b1e_e_package"][
        "expected_final_status"
    ]:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH b1ee_status"
        )
    if int(e_summary["frozen_approved_anchors"]) != 7:
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH approved"
        )
    freeze = load_json(
        e_root / "anchor_freeze" / "target_001_external_anchor_freeze.json"
    )
    approved_ids = list(freeze["approved_exact_ids"])
    if approved_ids != list(config["approved_anchor_ids"]):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH approved_ids"
        )
    return {
        "d_root": d_root,
        "e_root": e_root,
        "quality_rows": quality_rows,
        "candidates": candidates,
        "approved_ids": approved_ids,
        "e_summary": e_summary,
        "freeze": freeze,
    }


def validate_gallery_v1(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["gallery_v1"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    if int(summary["individual_gallery_members"]) != int(
        config["gallery_v1"]["expected_members"]
    ):
        raise RefinementReviewError(
            "BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH gallery_members"
        )
    if config["gallery_v1"].get("npy_load_forbidden") is not True:
        raise RefinementReviewError("gallery npy_load_forbidden required")
    return {"root": root, "summary": summary, "members": 7}


def select_view_candidates(
    quality_rows: Sequence[Mapping[str, Any]],
    *,
    prior_frames: Mapping[str, Sequence[int]],
    approved_candidates: Sequence[Mapping[str, Any]],
    approved_ids: Sequence[str],
    max_per_occ: int,
    max_total: int,
    min_gap: int,
    dup_ham: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    approved = [c for c in approved_candidates if c["anchor_candidate_id"] in approved_ids]
    if len(approved) != 7:
        raise RefinementReviewError("approved candidate inventory incomplete")
    approved_dhashes = [c["quality"]["perceptual_dhash"] for c in approved]
    by_code: dict[str, list[dict[str, Any]]] = {c: [] for c in FROZEN_CODES}
    for row in quality_rows:
        code = row["source_occurrence_code"]
        if code not in by_code:
            continue
        by_code[code].append(dict(row))

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "schema_version": "reid_target_001_external_target_view_candidate_selection_audit_v1",
        "existing_observation_quality_only": True,
        "f3_item_score_based_selection": False,
        "sample_used_for_candidate_selection": False,
        "occurrences": {},
        "excluded_prior_candidate_frames": {
            k: list(v) for k, v in prior_frames.items()
        },
        "approved_anchor_near_duplicate_hamming_max": dup_ham,
    }

    for code in FROZEN_CODES:
        used = {int(f) for f in prior_frames[code]}
        pool: list[dict[str, Any]] = []
        suppressed_prior = 0
        suppressed_anchor_dup = 0
        for row in by_code[code]:
            if not row.get("hard_quality_pass"):
                continue
            if int(row["frame_index"]) in used:
                suppressed_prior += 1
                continue
            dh = row["quality"]["perceptual_dhash"]
            if any(hamming_hex(dh, ad) <= dup_ham for ad in approved_dhashes):
                suppressed_anchor_dup += 1
                continue
            pool.append(row)
        pool = sorted(pool, key=rank_key)
        chosen: list[dict[str, Any]] = []
        suppressed_gap = 0
        suppressed_self_dup = 0
        for cand in pool:
            if len(chosen) >= max_per_occ:
                break
            if any(abs(int(cand["frame_index"]) - int(s["frame_index"])) < min_gap for s in chosen):
                suppressed_gap += 1
                continue
            if any(
                hamming_hex(
                    cand["quality"]["perceptual_dhash"],
                    s["quality"]["perceptual_dhash"],
                )
                <= dup_ham
                for s in chosen
            ):
                suppressed_self_dup += 1
                continue
            chosen.append(cand)
        chosen = sorted(chosen, key=lambda x: int(x["frame_index"]))
        audit["occurrences"][code] = {
            "hard_quality_pool_after_filters": len(pool),
            "selected": len(chosen),
            "selected_frames": [int(c["frame_index"]) for c in chosen],
            "suppressed_prior_candidate_frames": suppressed_prior,
            "suppressed_approved_anchor_near_duplicate": suppressed_anchor_dup,
            "suppressed_min_frame_gap": suppressed_gap,
            "suppressed_self_near_duplicate": suppressed_self_dup,
        }
        selected.extend(chosen)

    if len(selected) > max_total:
        raise RefinementReviewError("max total view candidates exceeded")
    selected = sorted(
        selected,
        key=lambda x: (FROZEN_CODES.index(x["source_occurrence_code"]), int(x["frame_index"])),
    )
    for i, item in enumerate(selected, start=1):
        item["target_view_candidate_id"] = f"target_001_ext_view_candidate_{i:03d}"
    audit["selected_total"] = len(selected)
    audit["selected_ids"] = [s["target_view_candidate_id"] for s in selected]
    return selected, audit


def unreviewed_eligible(
    eligible: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        dict(r)
        for r in eligible
        if r["external_candidate_code"] not in FROZEN_CODES
    ]
    if len(rows) != 135:
        raise RefinementReviewError(
            f"BLOCKED_STAGE5D_F3C_EXTERNAL_UNIVERSE_MISMATCH unreviewed={len(rows)}"
        )

    def sort_key(r: Mapping[str, Any]) -> tuple:
        bbox = r["representative_bbox"]
        x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
        return (
            int(r["first_frame"]),
            x_center,
            ext_numeric(str(r["external_candidate_code"])),
        )

    return sorted(rows, key=sort_key)


def draw_readable_label(
    frame: np.ndarray,
    *,
    text: str,
    x: int,
    y: int,
    min_px: int = 22,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min_px / 30.0)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x0 = max(0, x)
    y0 = max(th + pad + 2, y)
    top = y0 - th - pad - 2
    if top < 0:
        top = min(frame.shape[0] - th - pad - 2, y + pad)
        y0 = top + th + pad
    cv2.rectangle(
        frame,
        (x0, top),
        (min(frame.shape[1] - 1, x0 + tw + 2 * pad), min(frame.shape[0] - 1, y0 + baseline)),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x0 + pad, y0 - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def write_review_mp4(
    path: Path,
    *,
    video_path: Path,
    start: int,
    end: int,
    frame_items: Mapping[int, Sequence[Mapping[str, Any]]],
    frozen_items: Mapping[int, Sequence[Mapping[str, Any]]],
    fps: float,
    watermark: str,
    frozen_label: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RefinementReviewError("cannot open external video for review mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RefinementReviewError("FAILED review mp4 writer")
    codes_seen: set[str] = set()
    frames_with = 0
    count = 0
    try:
        for fi in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RefinementReviewError(f"failed read {fi} for mp4")
            out = frame.copy()
            items = list(frame_items.get(fi, []))
            frozen = list(frozen_items.get(fi, []))
            if items or frozen:
                frames_with += 1
            for it in sorted(items, key=lambda x: str(x["external_candidate_code"])):
                x1, y1, x2, y2 = [int(round(v)) for v in it["bbox_xyxy"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), (40, 200, 255), 2)
                draw_readable_label(
                    out,
                    text=str(it["external_candidate_code"]),
                    x=x1,
                    y=y1,
                )
                codes_seen.add(str(it["external_candidate_code"]))
            for it in sorted(frozen, key=lambda x: str(x["external_candidate_code"])):
                x1, y1, x2, y2 = [int(round(v)) for v in it["bbox_xyxy"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 2)
                draw_readable_label(
                    out,
                    text=frozen_label,
                    x=x1,
                    y=y1,
                    min_px=18,
                )
            cv2.putText(
                out,
                watermark,
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                watermark,
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(out)
            count += 1
    finally:
        writer.release()
        cap.release()
    return {
        "frame_count": count,
        "fps": fps,
        "duration_sec": count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frame_coverage": {
            "start": start,
            "end": end,
            "frames_with_boxes": frames_with,
        },
        "candidate_code_coverage": sorted(codes_seen),
        "width": w,
        "height": h,
        "bbox_interpolation": False,
        "existing_observations_only": True,
    }


def render_view_sheet(
    candidates: Sequence[Mapping[str, Any]],
    *,
    occurrence_code: str,
) -> np.ndarray:
    n = len(candidates)
    cols = min(2, max(1, n))
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w, tile_h = 900, 640
    sheet = np.full((rows_n * tile_h + 48, cols * tile_w, 3), 18, dtype=np.uint8)
    cv2.putText(
        sheet,
        f"{occurrence_code} target-view candidates (not anchors)",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    if n == 0:
        cv2.putText(
            sheet,
            "no additional view candidates after exclusion/diversity",
            (12, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return sheet
    for i, cand in enumerate(candidates):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 30, dtype=np.uint8)
        crop = cand["crop_image"]
        ch, cw = crop.shape[:2]
        crop_disp = _fit(crop, tile_w - 24, 340)
        dh, dw = crop_disp.shape[:2]
        ox = (tile_w - dw) // 2
        tile[70 : 70 + dh, ox : ox + dw] = crop_disp
        ctx_y = 70 + dh + 12
        ctx_h = tile_h - ctx_y - 12
        ctx_w = (tile_w - 30) // max(1, len(cand["context_images"]))
        x = 10
        for role, img in cand["context_images"]:
            disp = _fit(img, ctx_w - 4, ctx_h - 4)
            cdh, cdw = disp.shape[:2]
            tile[ctx_y : ctx_y + cdh, x : x + cdw] = disp
            cv2.putText(
                tile,
                role,
                (x + 4, ctx_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
            x += ctx_w
        q = cand["quality"]
        lines = [
            str(cand["target_view_candidate_id"]),
            f"{occurrence_code} f={cand['frame_index']} t={cand['video_time']:.2f}s",
            f"{q['crop_width']}x{q['crop_height']} blur={q['laplacian_variance']:.1f}",
            f"iou={q['max_person_iou']:.2f} clip={q['edge_clipping_fraction']:.2f}",
            "NOT ANCHOR / NOT GALLERY / NEEDS HUMAN APPROVAL",
        ]
        y = 18
        for text in lines:
            cv2.putText(
                tile,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += 12
        y0 = 48 + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def render_occurrence_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    sheet_index: int,
) -> np.ndarray:
    cols = 3
    rows_n = int(math.ceil(len(items) / cols)) if items else 1
    tile_w, tile_h = 620, 520
    sheet = np.full((rows_n * tile_h + 48, cols * tile_w, 3), 16, dtype=np.uint8)
    cv2.putText(
        sheet,
        f"External occurrence review sheet {sheet_index:02d} (neutral; no auto labels)",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    for i, item in enumerate(items):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
        crop = item["crop_image"]
        crop_disp = _fit(crop, tile_w - 20, 240)
        dh, dw = crop_disp.shape[:2]
        ox = (tile_w - dw) // 2
        tile[78 : 78 + dh, ox : ox + dw] = crop_disp
        ctx_y = 78 + dh + 8
        ctx_h = tile_h - ctx_y - 10
        ctx_w = (tile_w - 24) // max(1, len(item["context_images"]))
        x = 8
        for role, img in item["context_images"]:
            disp = _fit(img, ctx_w - 4, ctx_h - 4)
            cdh, cdw = disp.shape[:2]
            tile[ctx_y : ctx_y + cdh, x : x + cdw] = disp
            cv2.putText(
                tile,
                role,
                (x + 2, ctx_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
            x += ctx_w
        lines = [
            str(item["external_candidate_code"]),
            f"f={item['first_frame']}-{item['last_frame']} t={item['first_time']:.2f}-{item['last_time']:.2f}s",
            f"obs={item['observation_count']} rep_f={item['representative_frame']}",
            "NO TARGET / DISTRACTOR / HARD-NEG SUGGESTION",
        ]
        y = 18
        for text in lines:
            cv2.putText(
                tile,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += 14
        y0 = 48 + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3c_external_refinement_review_contract_v1",
        "target_id": TARGET_ID,
        "external_only": True,
        "sample_read": False,
        "sample_candidate_use": False,
        "sample_gallery_optimization": False,
        "existing_detection_tracking_only": True,
        "no_new_detection": True,
        "no_new_tracking": True,
        "frozen_target_occurrence_count": 3,
        "existing_frozen_anchor_count": 7,
        "unreviewed_review_eligible_count": 135,
        "automated_team_classification": False,
        "automated_ocr": False,
        "osnet_similarity": False,
        "manual_approval_required": True,
        "candidate_is_not_anchor_until_freeze": True,
        "distractor_is_not_hard_negative_member_until_freeze": True,
        "gallery_mutation": False,
        "threshold": False,
        "identity_assignment": False,
        "new_embeddings": 0,
        "similarity_scoring": 0,
        "hard_negative_approvals": 0,
        "new_anchor_approvals": 0,
        "external_occurrence_manual_decisions": 0,
        "exact_next_gate": NEXT_GATE,
    }


def make_tmp(project_root: Path, final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3c_external_refinement_review_{final_dir.name}_{token}"
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise RefinementReviewError("final_exists")
    tmp.rename(final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise RefinementReviewError("final_exists")

    # Sample leakage guards — never open sample paths.
    sample_rel = config["evaluation_source"]["path"]
    sample_path = project_root / sample_rel
    runtime_audit = {
        "schema_version": "reid_stage5d_f3c_runtime_audit_v1",
        "sample_video_read": False,
        "sample_crop_read_count": 0,
        "sample_embedding_read_count": 0,
        "sample_used_for_candidate_selection": False,
        "sample_used_for_gallery_optimization": False,
        "f3_item_score_based_selection": False,
        "new_detection": False,
        "new_tracking": False,
        "osnet_inference": False,
        "ocr_inference": False,
        "similarity_scoring": 0,
        "new_embeddings": 0,
        "gallery_mutation": False,
        "network_download": 0,
        "sample_path_exists_but_unread": sample_path.is_file(),
    }
    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise RefinementReviewError("BLOCKED_STAGE5D_F3C_SAMPLE_LEAKAGE decode_policy")

    f3b = validate_f3b(project_root, config)
    external = validate_external_source(project_root, config)
    b1eb = validate_b1eb(project_root, config)
    b1ec = validate_b1ec(project_root, config)
    b1ed = validate_b1ed_b1ee(project_root, config)
    gallery = validate_gallery_v1(project_root, config)

    prior_frames = {
        k: [int(x) for x in v]
        for k, v in config["prior_anchor_candidate_frames"].items()
    }
    for code, frames in prior_frames.items():
        if code not in FROZEN_CODES:
            raise RefinementReviewError("unexpected prior frame code")
    if sum(len(v) for v in prior_frames.values()) != 15:
        raise RefinementReviewError("expected 15 prior candidate frames")

    selected, selection_audit = select_view_candidates(
        b1ed["quality_rows"],
        prior_frames=prior_frames,
        approved_candidates=b1ed["candidates"],
        approved_ids=config["approved_anchor_ids"],
        max_per_occ=int(config["selection"]["max_candidates_per_occurrence"]),
        max_total=int(config["selection"]["max_total_candidates"]),
        min_gap=int(config["selection"]["min_frame_gap"]),
        dup_ham=int(config["selection"]["near_duplicate_dhash_hamming_max"]),
    )
    unreviewed = unreviewed_eligible(b1eb["eligible"])

    width = external["width"]
    height = external["height"]
    fps = external["fps"]
    pad_frac = float(config["crop_extraction"]["padding_fraction"])
    if pad_frac > float(config["crop_extraction"]["max_padding_fraction"]):
        raise RefinementReviewError("padding exceeds max")

    mapping_by_code = {
        r["external_candidate_code"]: r for r in b1eb["mapping"]
    }
    tracks_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (b1eb["root"] / "tracking" / "tracks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tracks_by_frame[int(row["frame_index"])].append(row)

    # Build frame overlays for unreviewed + frozen reference.
    raw_to_code = {
        int(r["raw_external_track_id"]): r["external_candidate_code"]
        for r in b1eb["mapping"]
    }
    unreviewed_codes = {r["external_candidate_code"] for r in unreviewed}
    frozen_raw = {
        int(s["raw_track_id"]): s["external_candidate_code"]
        for s in config["frozen_target_occurrences"]
    }
    frame_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frozen_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fi, rows in tracks_by_frame.items():
        for row in rows:
            tid = int(row["raw_track_id"])
            code = raw_to_code.get(tid)
            if code in unreviewed_codes:
                frame_items[fi].append(
                    {
                        "external_candidate_code": code,
                        "bbox_xyxy": row["bbox_xyxy"],
                    }
                )
            elif tid in frozen_raw and config["review_video"].get(
                "draw_frozen_as_reference", True
            ):
                frozen_items[fi].append(
                    {
                        "external_candidate_code": frozen_raw[tid],
                        "bbox_xyxy": row["bbox_xyxy"],
                    }
                )

    # Needed frames for crops/context.
    needed_frames: set[int] = set()
    for cand in selected:
        code = cand["source_occurrence_code"]
        mapping = mapping_by_code[code]
        obs_frames = [int(f) for f in mapping["observation_frames"]]
        needed_frames.add(int(cand["frame_index"]))
        for _, fi in select_context_frames(
            obs_frames,
            start_frame=int(mapping["first_frame"]),
            end_frame=int(mapping["last_frame"]),
            representative_frame=int(cand["frame_index"]),
        ):
            needed_frames.add(fi)
    for row in unreviewed:
        obs_frames = [int(f) for f in row["observation_frames"]]
        needed_frames.add(int(row["representative_frame"]))
        for _, fi in select_context_frames(
            obs_frames,
            start_frame=int(row["first_frame"]),
            end_frame=int(row["last_frame"]),
            representative_frame=int(row["representative_frame"]),
        ):
            needed_frames.add(fi)

    cap = cv2.VideoCapture(str(external["path"]))
    if not cap.isOpened():
        raise RefinementReviewError("cannot open external enrollment video")
    frame_cache: dict[int, np.ndarray] = {}

    def read_frame(fi: int) -> np.ndarray:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RefinementReviewError(f"failed to read external frame {fi}")
        frame_cache[fi] = frame
        return frame

    try:
        for fi in sorted(needed_frames):
            read_frame(fi)

        # Enrich view candidates with crops/context.
        for cand in selected:
            code = cand["source_occurrence_code"]
            mapping = mapping_by_code[code]
            fi = int(cand["frame_index"])
            bbox = [float(v) for v in cand["original_bbox_xyxy"]]
            frame = read_frame(fi)
            padded = pad_bbox(bbox, width=width, height=height, fraction=pad_frac)
            crop_int = float_bbox_to_int_crop(
                padded, video_width=width, video_height=height
            )
            x1, y1, x2, y2 = crop_int
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise RefinementReviewError(f"empty crop for {cand['target_view_candidate_id']}")
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                raise RefinementReviewError("png encode failed")
            crop_bytes = buf.tobytes()
            cand["crop_image"] = crop
            cand["crop_sha256"] = sha256_bytes(crop_bytes)
            cand["crop_bytes"] = crop_bytes
            cand["padded_crop_bbox_xyxy"] = padded
            cand["source_bbox"] = bbox
            obs_frames = [int(f) for f in mapping["observation_frames"]]
            bbox_by_f = {
                int(b["frame_index"]): [float(v) for v in b["bbox_xyxy"]]
                for b in mapping["bbox_per_observation"]
            }
            contexts: list[tuple[str, np.ndarray]] = []
            for role, cfi in select_context_frames(
                obs_frames,
                start_frame=int(mapping["first_frame"]),
                end_frame=int(mapping["last_frame"]),
                representative_frame=fi,
            ):
                cframe = read_frame(cfi).copy()
                cb = bbox_by_f[cfi]
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in cb]
                cv2.rectangle(cframe, (cx1, cy1), (cx2, cy2), (0, 220, 255), 2)
                cv2.putText(
                    cframe,
                    f"{role} f={cfi}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                contexts.append((role, cframe))
            cand["context_images"] = contexts

        # Enrich unreviewed occurrence panels.
        for row in unreviewed:
            obs_frames = [int(f) for f in row["observation_frames"]]
            bbox_by_f = {
                int(b["frame_index"]): [float(v) for v in b["bbox_xyxy"]]
                for b in row["bbox_per_observation"]
            }
            rep_f = int(row["representative_frame"])
            rep_bbox = [float(v) for v in row["representative_bbox"]]
            # Prefer mapping bbox at rep frame if present.
            if rep_f in bbox_by_f:
                rep_bbox = bbox_by_f[rep_f]
            frame = read_frame(rep_f)
            padded = pad_bbox(rep_bbox, width=width, height=height, fraction=pad_frac)
            crop_int = float_bbox_to_int_crop(
                padded, video_width=width, video_height=height
            )
            x1, y1, x2, y2 = crop_int
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                # Fall back to clamped original bbox.
                clamped = clamp_bbox_xyxy(
                    rep_bbox, video_width=width, video_height=height
                )
                crop_int = float_bbox_to_int_crop(
                    clamped, video_width=width, video_height=height
                )
                x1, y1, x2, y2 = crop_int
                crop = frame[y1:y2, x1:x2].copy()
            row["crop_image"] = crop
            row["first_time"] = int(row["first_frame"]) / fps
            row["last_time"] = int(row["last_frame"]) / fps
            contexts = []
            for role, cfi in select_context_frames(
                obs_frames,
                start_frame=int(row["first_frame"]),
                end_frame=int(row["last_frame"]),
                representative_frame=rep_f,
            ):
                cframe = read_frame(cfi).copy()
                cb = bbox_by_f[cfi]
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in cb]
                cv2.rectangle(cframe, (cx1, cy1), (cx2, cy2), (40, 200, 255), 2)
                cv2.putText(
                    cframe,
                    f"{role} f={cfi}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                contexts.append((role, cframe))
            row["context_images"] = contexts
    finally:
        cap.release()

    tmp = make_tmp(project_root, final_dir)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        inv = tmp / "inventory"
        quality = tmp / "quality"
        review = tmp / "review"
        review_pkg = tmp / "review_packages" / "target_001_external_refinement_review"
        videos = tmp / "videos"
        templates = tmp / "templates"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        crops = review_pkg / "crops"
        for d in (inv, quality, review, review_pkg, videos, templates, runtime, cfg_dir, crops):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        # Write view candidate crops + inventory.
        view_inv_path = inv / "target_001_external_target_view_candidate_inventory.jsonl"
        with view_inv_path.open("w", encoding="utf-8") as handle:
            for cand in selected:
                code = cand["source_occurrence_code"]
                cid = cand["target_view_candidate_id"]
                rel_crop = f"review_packages/target_001_external_refinement_review/crops/{code}/{cid}.png"
                crop_path = tmp / rel_crop
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop_path.write_bytes(cand["crop_bytes"])
                if sha256_file(crop_path) != cand["crop_sha256"]:
                    raise RefinementReviewError("crop sha mismatch after write")
                rec = {
                    "target_view_candidate_id": cid,
                    "target_id": TARGET_ID,
                    "source_occurrence_code": code,
                    "raw_track_id": int(cand["raw_track_id"]),
                    "frame_index": int(cand["frame_index"]),
                    "video_time": float(cand["video_time"]),
                    "crop_path": rel_crop,
                    "crop_sha256": cand["crop_sha256"],
                    "source_bbox": cand["source_bbox"],
                    "crop_bbox": cand["padded_crop_bbox_xyxy"],
                    "quality": cand["quality"],
                    "hard_quality_pass": True,
                    "is_anchor": False,
                    "is_gallery_member": False,
                    "embedding_input": False,
                    "manual_approval_required": True,
                    "source_video_sha256": external["sha256"],
                    "selection_basis": "existing_observation_quality_diagnostics",
                    "f3_sample_score_used": False,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # View sheets by occurrence.
        by_occ: dict[str, list[dict[str, Any]]] = {c: [] for c in FROZEN_CODES}
        for cand in selected:
            by_occ[cand["source_occurrence_code"]].append(cand)
        for code in FROZEN_CODES:
            sheet = render_view_sheet(by_occ[code], occurrence_code=code)
            out = review_pkg / f"target_view_candidates_{code}.png"
            if not cv2.imwrite(str(out), sheet):
                raise RefinementReviewError(f"failed write {out.name}")

        # Blank view template.
        view_tpl = templates / "target_001_external_target_view_candidate_review_template.csv"
        with view_tpl.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=VIEW_TEMPLATE_FIELDS)
            writer.writeheader()
            for cand in selected:
                row = {k: "" for k in VIEW_TEMPLATE_FIELDS}
                row.update(
                    {
                        "target_view_candidate_id": cand["target_view_candidate_id"],
                        "source_occurrence_code": cand["source_occurrence_code"],
                        "frame_index": int(cand["frame_index"]),
                        "video_time": float(cand["video_time"]),
                        "crop_path": (
                            "review_packages/target_001_external_refinement_review/crops/"
                            f"{cand['source_occurrence_code']}/"
                            f"{cand['target_view_candidate_id']}.png"
                        ),
                        "crop_sha256": cand["crop_sha256"],
                        "source_bbox": json.dumps(cand["source_bbox"]),
                        "crop_bbox": json.dumps(cand["padded_crop_bbox_xyxy"]),
                    }
                )
                writer.writerow(row)

        # Unreviewed occurrence inventory + sheets.
        occ_inv_path = inv / "target_001_external_unreviewed_occurrence_inventory.jsonl"
        with occ_inv_path.open("w", encoding="utf-8") as handle:
            for row in unreviewed:
                rec = {
                    "external_candidate_code": row["external_candidate_code"],
                    "raw_track_id": int(row["raw_external_track_id"]),
                    "first_frame": int(row["first_frame"]),
                    "last_frame": int(row["last_frame"]),
                    "representative_frame": int(row["representative_frame"]),
                    "observation_count": int(row["observation_count"]),
                    "representative_bbox": row["representative_bbox"],
                    "source_video_sha256": row.get("source_video_sha256")
                    or external["sha256"],
                    "detection_lineage": row.get("detection_lineage"),
                    "bbox_per_observation_count": len(row["bbox_per_observation"]),
                    "review_eligible": True,
                    "frozen_target_occurrence": False,
                    "target_suggestion": None,
                    "hard_negative_suggestion": None,
                    "automated_team_label": None,
                    "automated_ocr": None,
                    "identity_prediction": None,
                    "representative_crop_path": "",
                    "representative_crop_sha256": "",
                    "source_representative_crop_copy": False,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        per_sheet = int(config["occurrence_sheets"]["items_per_sheet"])
        sheets: list[list[dict[str, Any]]] = []
        for i in range(0, len(unreviewed), per_sheet):
            sheets.append(unreviewed[i : i + per_sheet])
        if len(sheets) != int(config["occurrence_sheets"]["expected_sheet_count"]):
            raise RefinementReviewError("occurrence sheet count mismatch")
        if any(len(s) != per_sheet for s in sheets[:-1]):
            raise RefinementReviewError("full sheet item count mismatch")
        if len(sheets[-1]) != int(config["occurrence_sheets"]["expected_final_sheet_items"]):
            raise RefinementReviewError("final sheet item count mismatch")
        for idx, chunk in enumerate(sheets, start=1):
            sheet = render_occurrence_sheet(chunk, sheet_index=idx)
            out = review_pkg / f"external_occurrence_review_sheet_{idx:02d}.png"
            if not cv2.imwrite(str(out), sheet):
                raise RefinementReviewError(f"failed write {out.name}")

        # Blank occurrence template.
        occ_tpl = (
            templates / "target_001_external_occurrence_refinement_review_template.csv"
        )
        with occ_tpl.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OCC_TEMPLATE_FIELDS)
            writer.writeheader()
            for row in unreviewed:
                rec = {k: "" for k in OCC_TEMPLATE_FIELDS}
                rec.update(
                    {
                        "external_candidate_code": row["external_candidate_code"],
                        "raw_track_id": int(row["raw_external_track_id"]),
                        "first_frame": int(row["first_frame"]),
                        "last_frame": int(row["last_frame"]),
                        "representative_frame": int(row["representative_frame"]),
                        "observation_count": int(row["observation_count"]),
                        "representative_crop_path": "",
                        "representative_crop_sha256": "",
                    }
                )
                writer.writerow(rec)

        # Review video.
        mp4_path = videos / "target_001_external_unreviewed_occurrence_review.mp4"
        mp4_meta = write_review_mp4(
            mp4_path,
            video_path=external["path"],
            start=int(config["review_video"]["start_frame"]),
            end=int(config["review_video"]["end_frame"]),
            frame_items=frame_items,
            frozen_items=frozen_items,
            fps=fps,
            watermark=str(config["review_video"]["watermark"]),
            frozen_label=str(config["review_video"]["frozen_reference_label"]),
        )
        if set(mp4_meta["candidate_code_coverage"]) != unreviewed_codes:
            missing = unreviewed_codes - set(mp4_meta["candidate_code_coverage"])
            extra = set(mp4_meta["candidate_code_coverage"]) - unreviewed_codes
            raise RefinementReviewError(
                f"review video code coverage mismatch missing={sorted(missing)[:5]} "
                f"extra={sorted(extra)[:5]}"
            )
        if any(c in mp4_meta["candidate_code_coverage"] for c in FROZEN_CODES):
            raise RefinementReviewError("frozen codes must not appear as review codes")

        write_json(quality / "target_001_external_target_view_candidate_selection_audit.json", selection_audit)
        write_json(
            review / "target_001_external_refinement_review_contract.json",
            {
                **build_contract(),
                "generated_at": generated_at,
                "project_head": head,
                "external_source_sha256": external["sha256"],
                "f3b_snapshot_sha256": f3b["snapshot_sha256"],
                "target_view_candidate_count": len(selected),
                "target_view_candidate_distribution": {
                    c: len(by_occ[c]) for c in FROZEN_CODES
                },
                "occurrence_sheet_count": len(sheets),
                "occurrence_sheet_item_distribution": [len(s) for s in sheets],
                "review_video": {
                    "path": "videos/target_001_external_unreviewed_occurrence_review.mp4",
                    "sha256": mp4_meta["sha256"],
                    "frame_count": mp4_meta["frame_count"],
                },
            },
        )
        write_json(tmp / "stage5d_f3c_contract.json", {
            **build_contract(),
            "final_status": FINAL_STATUS,
            "generated_at": generated_at,
            "project_head": head,
        })

        file_count, files_sha = listing_sha(tmp)
        view_sheets = sorted(review_pkg.glob("target_view_candidates_EXT_*.png"))
        occ_sheets = sorted(review_pkg.glob("external_occurrence_review_sheet_*.png"))
        crop_files = sorted(crops.rglob("*.png"))
        summary = {
            "schema_version": "reid_stage5d_f3c_external_refinement_review_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "external_only": True,
            "frozen_target_occurrences": 3,
            "existing_frozen_target_anchors": 7,
            "unreviewed_review_eligible_external_tracks": 135,
            "external_occurrence_manual_decisions": 0,
            "new_anchor_approvals": 0,
            "hard_negative_approvals": 0,
            "new_embeddings": 0,
            "gallery_mutation": False,
            "gallery_members": gallery["members"],
            "similarity_scoring": 0,
            "threshold": False,
            "identity_assignments": 0,
            "target_view_candidate_count": len(selected),
            "target_view_candidate_distribution": {
                c: len(by_occ[c]) for c in FROZEN_CODES
            },
            "target_view_sheets": len(view_sheets),
            "occurrence_sheets": len(occ_sheets),
            "occurrence_sheet_item_distribution": [len(s) for s in sheets],
            "diagnostic_mp4": 1,
            "candidate_crop_copies": len(crop_files),
            "source_video_copy": 0,
            "source_representative_crop_copies": 0,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_used_for_candidate_selection": False,
            "sample_used_for_gallery_optimization": False,
            "automated_team_classification": False,
            "automated_ocr": False,
            "osnet_similarity": False,
            "f3b_final_status": f3b["summary"]["final_status"],
            "f3b_snapshot_sha256": f3b["snapshot_sha256"],
            "external_source_sha256": external["sha256"],
            "b1eb_detection_total": b1eb["summary"]["detection_total"],
            "b1eb_review_eligible": 138,
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3c_summary.json", summary)
        write_json(
            review / "target_001_external_refinement_review_manifest.json",
            {
                "schema_version": "reid_target_001_external_refinement_review_manifest_v1",
                "target_view_sheets": [p.name for p in view_sheets],
                "occurrence_sheets": [p.name for p in occ_sheets],
                "candidate_crops": [
                    str(p.relative_to(tmp)).replace("\\", "/") for p in crop_files
                ],
                "review_video": "videos/target_001_external_unreviewed_occurrence_review.mp4",
                "review_video_sha256": mp4_meta["sha256"],
                "templates": [
                    "templates/target_001_external_target_view_candidate_review_template.csv",
                    "templates/target_001_external_occurrence_refinement_review_template.csv",
                ],
            },
        )
        write_json(runtime / "stage5d_f3c_runtime_audit.json", runtime_audit)
        write_json(
            tmp / "stage5d_f3c_manifest.json",
            {
                "schema_version": "reid_stage5d_f3c_external_refinement_review_manifest_v1",
                "final_status": FINAL_STATUS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "target_view_sheets": 3,
                    "occurrence_sheets": 12,
                    "diagnostic_mp4": 1,
                    "candidate_crop_copies_max": 12,
                    "candidate_crop_copies": len(crop_files),
                    "source_video_copy": 0,
                    "source_representative_crop_copies": 0,
                    "new_embeddings": 0,
                    "similarity_rows": 0,
                    "hard_negative_approvals": 0,
                    "new_target_anchors": 0,
                    "gallery_mutation": 0,
                    "identity_assignments": 0,
                },
                "generated_at": generated_at,
            },
        )

        # Budget checks before publish.
        if len(view_sheets) != 3:
            raise RefinementReviewError("expected 3 target-view sheets")
        if len(occ_sheets) != 12:
            raise RefinementReviewError("expected 12 occurrence sheets")
        if len(crop_files) > 12:
            raise RefinementReviewError("crop copy budget exceeded")
        if len(crop_files) != len(selected):
            raise RefinementReviewError("crop copy count mismatch")
        if list(tmp.rglob("sample.mp4")):
            raise RefinementReviewError("BLOCKED_STAGE5D_F3C_SAMPLE_LEAKAGE sample_copy")
        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.npz")):
            raise RefinementReviewError("embedding artifacts forbidden")

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f3c_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_refinement_review_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(json.dumps({"final_status": summary["final_status"], "exact_next_gate": summary["exact_next_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
