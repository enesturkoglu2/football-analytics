#!/usr/bin/env python3
"""Stage 5D-F3D — freeze human external refinement decisions.

Freezes exact Furkan-approved target-view and occurrence decisions from
the F3C review package. No embeddings, gallery-v2, hard-negative members,
OCR/similarity inference, crop regeneration, or sample access.
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
from typing import Any, Mapping, Optional, Sequence

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_external_refinement_manual_freeze_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3D_TARGET_001_EXTERNAL_REFINEMENT_DECISIONS_FROZEN"
)
NEXT_GATE = (
    "STAGE5D-F3E_TARGET_001_EXTERNAL_REFINEMENT_CROP_QUALITY_AND_HARD_NEGATIVE_REVIEW_PACKAGE"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_external_refinement_manual_freeze.py",
    "configs/reid/external_refinement_manual_freeze_stage5d_target_001.yaml",
    "tests/test_reid_external_refinement_manual_freeze.py",
    "docs/setup/stage5d-target-external-refinement-manual-review-and-freeze.md",
}
EXISTING_FROZEN_OCCURRENCES = ("EXT_004", "EXT_183", "EXT_198")
EXISTING_ANCHOR_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)

VIEW_EXPANSION_YES = (
    "target_001_ext_view_candidate_001",
    "target_001_ext_view_candidate_002",
)
VIEW_EXPANSION_NO = (
    "target_001_ext_view_candidate_003",
    "target_001_ext_view_candidate_004",
    "target_001_ext_view_candidate_006",
    "target_001_ext_view_candidate_007",
    "target_001_ext_view_candidate_008",
    "target_001_ext_view_candidate_009",
    "target_001_ext_view_candidate_010",
)
VIEW_AMBIGUOUS = (
    "target_001_ext_view_candidate_005",
    "target_001_ext_view_candidate_011",
)
VIEW_ALL = VIEW_EXPANSION_YES + VIEW_EXPANSION_NO + VIEW_AMBIGUOUS

ADDITIONAL_TARGET = ("EXT_161",)
SAME_TEAM_DISTRACTORS = (
    "EXT_001",
    "EXT_003",
    "EXT_005",
    "EXT_007",
    "EXT_010",
    "EXT_017",
    "EXT_028",
    "EXT_029",
    "EXT_034",
    "EXT_036",
    "EXT_042",
    "EXT_044",
    "EXT_047",
    "EXT_050",
    "EXT_055",
    "EXT_057",
    "EXT_061",
    "EXT_136",
    "EXT_138",
    "EXT_140",
    "EXT_158",
    "EXT_167",
    "EXT_175",
    "EXT_178",
    "EXT_182",
    "EXT_184",
    "EXT_208",
    "EXT_212",
    "EXT_215",
    "EXT_217",
    "EXT_218",
    "EXT_230",
    "EXT_242",
    "EXT_245",
    "EXT_247",
)
OTHER_TEAM = (
    "EXT_002",
    "EXT_012",
    "EXT_013",
    "EXT_014",
    "EXT_019",
    "EXT_023",
    "EXT_025",
    "EXT_026",
    "EXT_030",
    "EXT_033",
    "EXT_035",
    "EXT_041",
    "EXT_051",
    "EXT_052",
    "EXT_056",
    "EXT_062",
    "EXT_067",
    "EXT_069",
    "EXT_075",
    "EXT_078",
    "EXT_079",
    "EXT_081",
    "EXT_082",
    "EXT_084",
    "EXT_100",
    "EXT_116",
    "EXT_129",
    "EXT_134",
    "EXT_137",
    "EXT_142",
    "EXT_168",
    "EXT_171",
    "EXT_196",
    "EXT_199",
    "EXT_200",
    "EXT_201",
    "EXT_203",
    "EXT_205",
    "EXT_210",
    "EXT_211",
    "EXT_216",
    "EXT_219",
    "EXT_222",
    "EXT_223",
    "EXT_225",
    "EXT_226",
    "EXT_227",
    "EXT_231",
    "EXT_235",
    "EXT_239",
    "EXT_246",
    "EXT_248",
)
NON_PLAYER = (
    "EXT_024",
    "EXT_031",
    "EXT_119",
    "EXT_125",
    "EXT_132",
    "EXT_193",
    "EXT_197",
    "EXT_202",
    "EXT_221",
)
UNCERTAIN = ("EXT_072", "EXT_073", "EXT_089", "EXT_224")
INVALID = ("EXT_027", "EXT_139", "EXT_145", "EXT_157", "EXT_190")
AMBIGUOUS = (
    "EXT_006",
    "EXT_008",
    "EXT_009",
    "EXT_011",
    "EXT_022",
    "EXT_032",
    "EXT_043",
    "EXT_049",
    "EXT_053",
    "EXT_059",
    "EXT_060",
    "EXT_066",
    "EXT_071",
    "EXT_085",
    "EXT_103",
    "EXT_117",
    "EXT_120",
    "EXT_122",
    "EXT_147",
    "EXT_163",
    "EXT_165",
    "EXT_166",
    "EXT_172",
    "EXT_204",
    "EXT_207",
    "EXT_213",
    "EXT_229",
    "EXT_232",
    "EXT_233",
)
DISTRACTOR_JERSEY = {
    "EXT_158": "30",
    "EXT_184": "14",
    "EXT_208": "17",
    "EXT_212": "25",
    "EXT_215": "30",
    "EXT_217": "14",
    "EXT_230": "30",
    "EXT_242": "9",
    "EXT_245": "3",
}
JERSEY_PROVENANCE = "human_verified_by_Furkan_not_automated_ocr"
TRANSCRIPTION_CORRECTIONS = (
    ("EXT_016", "EXT_019", "other_team_player"),
    ("EXT_048", "EXT_049", "multi_person_ambiguous"),
    ("EXT_065", "EXT_066", "multi_person_ambiguous"),
    ("EXT_080", "EXT_082", "other_team_player"),
    ("EXT_228", "EXT_226", "other_team_player"),
)

VIEW_CSV_FIELDS = (
    "target_view_candidate_id",
    "target_id",
    "source_occurrence_code",
    "raw_track_id",
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
    "decision_reason",
    "future_target_embedding_source_eligible",
    "current_gallery_member",
    "identity_negative",
    "hard_negative_eligible",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "automated_ocr_used",
    "automated_team_classifier_used",
    "similarity_used",
    "model_identity_used",
    "decisions_human_approved",
)
OCC_CSV_FIELDS = (
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
    "jersey_number_provenance",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_view_category",
    "manual_notes",
    "additional_target_occurrence_frozen",
    "same_team_distractor_source_frozen",
    "target_identity_negative",
    "future_target_crop_review_source_eligible",
    "future_hard_negative_crop_review_source_eligible",
    "hard_negative_gallery_member",
    "hard_negative_eligible",
    "same_team_hard_negative_eligible",
    "target_anchor_eligible",
    "target_anchor_member",
    "target_present_but_contaminated",
    "target_positive",
    "target_negative",
    "embedding_input",
    "current_gallery_member",
    "automatic_negative",
    "identity_assignment",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "automated_ocr_used",
    "automated_team_classifier_used",
    "similarity_used",
    "model_identity_used",
    "decisions_human_approved",
)


class RefinementFreezeError(RuntimeError):
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
        raise RefinementFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise RefinementFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise RefinementFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise RefinementFreezeError("BLOCKED_STAGE5D_F3D_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise RefinementFreezeError("BLOCKED_STAGE5D_F3D_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise RefinementFreezeError(
                    "BLOCKED_STAGE5D_F3D_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str, label: str) -> str:
    if not snapshot_path.is_file():
        raise RefinementFreezeError(
            f"BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH {label}_snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise RefinementFreezeError(
            f"BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH {label}_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise RefinementFreezeError(
            f"BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH {label}_sha"
        )
    return actual


def decision_code_set() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for code in ADDITIONAL_TARGET:
        mapping[code] = "additional_target_occurrence_yes"
    for code in SAME_TEAM_DISTRACTORS:
        mapping[code] = "same_team_distractor_yes"
    for code in OTHER_TEAM:
        mapping[code] = "other_team_player"
    for code in NON_PLAYER:
        mapping[code] = "non_player"
    for code in UNCERTAIN:
        mapping[code] = "uncertain"
    for code in INVALID:
        mapping[code] = "invalid"
    for code in AMBIGUOUS:
        mapping[code] = "multi_person_ambiguous"
    if len(mapping) != 135 or len(set(mapping)) != 135:
        raise RefinementFreezeError("decision code set size mismatch")
    counts = Counter(mapping.values())
    expected = {
        "additional_target_occurrence_yes": 1,
        "same_team_distractor_yes": 35,
        "other_team_player": 52,
        "non_player": 9,
        "uncertain": 4,
        "invalid": 5,
        "multi_person_ambiguous": 29,
    }
    if dict(counts) != expected:
        raise RefinementFreezeError(f"decision distribution mismatch {dict(counts)}")
    return mapping


def validate_f3c(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3c_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3c_summary.json")
    contract = load_json(root / "stage5d_f3c_contract.json")
    cfg = config["stage5d_f3c_package"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("external_source_sha256") != config["external_enrollment_source"][
        "expected_sha256"
    ]:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH external_sha"
        )
    checks = {
        "b1eb_review_eligible": cfg["expected_review_eligible_total"],
        "frozen_target_occurrences": cfg["expected_frozen_target_occurrences"],
        "unreviewed_review_eligible_external_tracks": cfg[
            "expected_unreviewed_occurrences"
        ],
        "occurrence_sheets": 12,
        "target_view_candidate_count": cfg["expected_target_view_candidates"],
        "target_view_sheets": 3,
        "external_occurrence_manual_decisions": 0,
        "new_embeddings": 0,
        "similarity_scoring": 0,
        "sample_crop_read_count": 0,
        "sample_embedding_read_count": 0,
    }
    for key, exp in checks.items():
        if summary.get(key) != exp:
            raise RefinementFreezeError(
                f"BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("occurrence_sheet_item_distribution") != [12] * 11 + [3]:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH sheet_dist"
        )
    if summary.get("target_view_candidate_distribution") != {
        "EXT_004": 4,
        "EXT_183": 3,
        "EXT_198": 4,
    }:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH view_dist"
        )
    if summary.get("gallery_mutation") is not False:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH gallery_mutation"
        )
    if summary.get("sample_video_read") is not False:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH sample_read"
        )
    view_inv = load_jsonl(
        root / "inventory" / "target_001_external_target_view_candidate_inventory.jsonl"
    )
    occ_inv = load_jsonl(
        root
        / "inventory"
        / "target_001_external_unreviewed_occurrence_inventory.jsonl"
    )
    if len(view_inv) != 11 or len(occ_inv) != 135:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH inventory_counts"
        )
    # Blank template checks.
    view_tpl = (
        root
        / "templates"
        / "target_001_external_target_view_candidate_review_template.csv"
    )
    occ_tpl = (
        root
        / "templates"
        / "target_001_external_occurrence_refinement_review_template.csv"
    )
    with view_tpl.open(encoding="utf-8") as handle:
        view_rows = list(csv.DictReader(handle))
    with occ_tpl.open(encoding="utf-8") as handle:
        occ_rows = list(csv.DictReader(handle))
    if len(view_rows) != 11 or len(occ_rows) != 135:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH template_rows"
        )
    for row in view_rows:
        if row.get("manual_anchor_expansion_decision"):
            raise RefinementFreezeError(
                "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH view_not_blank"
            )
    for row in occ_rows:
        if row.get("manual_refinement_decision"):
            raise RefinementFreezeError(
                "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH occ_not_blank"
            )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"], "f3c"
    )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "view_inv": view_inv,
        "occ_inv": occ_inv,
        "snapshot_sha256": snap,
    }


def validate_upstream(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    f3b_rel = config["stage5d_f3b_refinement_design"]["path"]
    assert_no_path_traversal(f3b_rel)
    f3b_root = project_root / f3b_rel
    f3b_summary = load_json(f3b_root / "stage5d_f3b_summary.json")
    if f3b_summary.get("final_status") != config["stage5d_f3b_refinement_design"][
        "expected_final_status"
    ]:
        raise RefinementFreezeError("F3B status mismatch")
    f3b_snap = resolve_snapshot_sha(
        Path(config["stage5d_f3b_refinement_design"]["snapshot_path"]),
        config["stage5d_f3b_refinement_design"]["expected_snapshot_sha256"],
        "f3b",
    )
    b1ec = load_json(
        project_root
        / config["stage5d_b1e_c_package"]["path"]
        / "stage5d_b1e_c_summary.json"
    )
    if int(b1ec["selected_positive_count"]) != 3:
        raise RefinementFreezeError("B1E-C positives mismatch")
    if tuple(b1ec["selected_external_candidate_codes"]) != EXISTING_FROZEN_OCCURRENCES:
        raise RefinementFreezeError("B1E-C codes mismatch")
    b1ee = load_json(
        project_root
        / config["stage5d_b1e_e_package"]["path"]
        / "stage5d_b1e_e_summary.json"
    )
    if int(b1ee["frozen_approved_anchors"]) != 7:
        raise RefinementFreezeError("B1E-E approved mismatch")
    gallery = load_json(
        project_root / config["gallery_v1"]["path"] / "stage5d_b1e_f_summary.json"
    )
    if int(gallery["individual_gallery_members"]) != 7:
        raise RefinementFreezeError("gallery members mismatch")
    td_rel = config["target_definition"]["path"]
    assert_no_path_traversal(td_rel)
    td = load_json(project_root / td_rel)
    if td.get("target_id") != TARGET_ID:
        raise RefinementFreezeError("target definition mismatch")
    return {
        "f3b_summary": f3b_summary,
        "f3b_snapshot_sha256": f3b_snap,
        "b1ec": b1ec,
        "b1ee": b1ee,
        "gallery": gallery,
        "target_definition": td,
        "target_definition_sha256": sha256_file(project_root / td_rel),
    }


def build_view_decisions(
    view_inv: Sequence[Mapping[str, Any]],
    *,
    reviewer: str,
    final_approver: str,
    reviewed_at: str,
) -> list[dict[str, Any]]:
    by_id = {r["target_view_candidate_id"]: r for r in view_inv}
    if set(by_id) != set(VIEW_ALL):
        raise RefinementFreezeError("target-view inventory ID mismatch")
    specs: dict[str, dict[str, Any]] = {}
    specs["target_001_ext_view_candidate_001"] = {
        "manual_anchor_expansion_decision": "target_anchor_expansion_yes",
        "manual_crop_valid": "yes",
        "manual_target_dominant": "yes",
        "manual_single_person": "yes",
        "manual_identity_confirmed": "yes",
        "manual_view_category": "left_side",
        "manual_scale_category": "small",
        "decision_reason": "approved_clean_side_view_expansion_candidate",
        "future_target_embedding_source_eligible": True,
        "identity_negative": False,
        "hard_negative_eligible": False,
    }
    specs["target_001_ext_view_candidate_002"] = {
        "manual_anchor_expansion_decision": "target_anchor_expansion_yes",
        "manual_crop_valid": "yes",
        "manual_target_dominant": "yes",
        "manual_single_person": "yes",
        "manual_identity_confirmed": "yes",
        "manual_view_category": "front_oblique",
        "manual_scale_category": "small",
        "decision_reason": "approved_clean_front_oblique_expansion_candidate",
        "future_target_embedding_source_eligible": True,
        "identity_negative": False,
        "hard_negative_eligible": False,
    }
    for cid in VIEW_EXPANSION_NO:
        specs[cid] = {
            "manual_anchor_expansion_decision": "target_anchor_expansion_no",
            "manual_crop_valid": "yes",
            "manual_target_dominant": "yes",
            "manual_single_person": "yes",
            "manual_identity_confirmed": "yes",
            "manual_view_category": "",
            "manual_scale_category": "",
            "decision_reason": "not_selected_redundant_valid_target_crop",
            "future_target_embedding_source_eligible": False,
            "identity_negative": False,
            "hard_negative_eligible": False,
        }
    specs["target_001_ext_view_candidate_005"] = {
        "manual_anchor_expansion_decision": "multi_person_ambiguous",
        "manual_crop_valid": "uncertain",
        "manual_target_dominant": "yes",
        "manual_single_person": "no",
        "manual_identity_confirmed": "yes",
        "manual_view_category": "",
        "manual_scale_category": "",
        "decision_reason": "excluded_additional_person_contamination",
        "future_target_embedding_source_eligible": False,
        "identity_negative": False,
        "hard_negative_eligible": False,
    }
    specs["target_001_ext_view_candidate_011"] = {
        "manual_anchor_expansion_decision": "multi_person_ambiguous",
        "manual_crop_valid": "uncertain",
        "manual_target_dominant": "yes",
        "manual_single_person": "no",
        "manual_identity_confirmed": "yes",
        "manual_view_category": "",
        "manual_scale_category": "",
        "decision_reason": "excluded_white_player_overlap",
        "future_target_embedding_source_eligible": False,
        "identity_negative": False,
        "hard_negative_eligible": False,
    }
    rows: list[dict[str, Any]] = []
    for cid in sorted(VIEW_ALL):
        src = by_id[cid]
        spec = specs[cid]
        rows.append(
            {
                "target_view_candidate_id": cid,
                "target_id": TARGET_ID,
                "source_occurrence_code": src["source_occurrence_code"],
                "raw_track_id": int(src["raw_track_id"]),
                "frame_index": int(src["frame_index"]),
                "video_time": float(src["video_time"]),
                "crop_path": src["crop_path"],
                "crop_sha256": src["crop_sha256"],
                "source_bbox": json.dumps(src["source_bbox"]),
                "crop_bbox": json.dumps(src["crop_bbox"]),
                "manual_quality_notes": "",
                "current_gallery_member": False,
                "reviewer": reviewer,
                "final_approver": final_approver,
                "reviewed_at": reviewed_at,
                "automated_ocr_used": False,
                "automated_team_classifier_used": False,
                "similarity_used": False,
                "model_identity_used": False,
                "decisions_human_approved": True,
                **spec,
            }
        )
    decisions = Counter(r["manual_anchor_expansion_decision"] for r in rows)
    if decisions != {
        "target_anchor_expansion_yes": 2,
        "target_anchor_expansion_no": 7,
        "multi_person_ambiguous": 2,
    }:
        raise RefinementFreezeError(f"view decision distribution mismatch {dict(decisions)}")
    return rows


def build_occurrence_decisions(
    occ_inv: Sequence[Mapping[str, Any]],
    *,
    decision_map: Mapping[str, str],
    reviewer: str,
    final_approver: str,
    reviewed_at: str,
) -> list[dict[str, Any]]:
    by_code = {r["external_candidate_code"]: r for r in occ_inv}
    inv_codes = set(by_code)
    dec_codes = set(decision_map)
    if inv_codes != dec_codes:
        raise RefinementFreezeError(
            "BLOCKED_STAGE5D_F3D_REVIEW_PACKAGE_CONTRACT_MISMATCH coverage "
            + json.dumps(
                {
                    "missing": sorted(inv_codes - dec_codes),
                    "extra": sorted(dec_codes - inv_codes),
                }
            )
        )
    rows: list[dict[str, Any]] = []
    for code in sorted(dec_codes, key=lambda c: int(c.split("_", 1)[1])):
        src = by_code[code]
        decision = decision_map[code]
        row: dict[str, Any] = {
            "external_candidate_code": code,
            "raw_track_id": int(src["raw_track_id"]),
            "first_frame": int(src["first_frame"]),
            "last_frame": int(src["last_frame"]),
            "representative_frame": int(src["representative_frame"]),
            "observation_count": int(src["observation_count"]),
            "representative_crop_path": src.get("representative_crop_path") or "",
            "representative_crop_sha256": src.get("representative_crop_sha256") or "",
            "manual_refinement_decision": decision,
            "manual_visible_jersey_number": "",
            "jersey_number_provenance": "",
            "manual_view_category": "",
            "manual_notes": "",
            "additional_target_occurrence_frozen": False,
            "same_team_distractor_source_frozen": False,
            "target_identity_negative": False,
            "future_target_crop_review_source_eligible": False,
            "future_hard_negative_crop_review_source_eligible": False,
            "hard_negative_gallery_member": False,
            "hard_negative_eligible": False,
            "same_team_hard_negative_eligible": False,
            "target_anchor_eligible": False,
            "target_anchor_member": False,
            "target_present_but_contaminated": False,
            "target_positive": False,
            "target_negative": False,
            "embedding_input": False,
            "current_gallery_member": False,
            "automatic_negative": False,
            "identity_assignment": False,
            "manual_identity_continuity_observed": "",
            "reviewer": reviewer,
            "final_approver": final_approver,
            "reviewed_at": reviewed_at,
            "automated_ocr_used": False,
            "automated_team_classifier_used": False,
            "similarity_used": False,
            "model_identity_used": False,
            "decisions_human_approved": True,
        }
        if decision == "additional_target_occurrence_yes":
            row.update(
                {
                    "manual_same_target_as_target_001": "yes",
                    "manual_same_team_as_target": "yes",
                    "manual_visible_jersey_number": "5",
                    "jersey_number_provenance": JERSEY_PROVENANCE,
                    "manual_crop_valid": "yes",
                    "manual_target_dominant": "yes",
                    "manual_single_person": "yes",
                    "manual_identity_continuity_observed": "yes",
                    "additional_target_occurrence_frozen": True,
                    "future_target_crop_review_source_eligible": True,
                    "target_positive": True,
                }
            )
        elif decision == "same_team_distractor_yes":
            jersey = DISTRACTOR_JERSEY.get(code, "")
            row.update(
                {
                    "manual_same_target_as_target_001": "no",
                    "manual_same_team_as_target": "yes",
                    "manual_visible_jersey_number": jersey,
                    "jersey_number_provenance": JERSEY_PROVENANCE if jersey else "",
                    "manual_crop_valid": "yes",
                    "manual_target_dominant": "no",
                    "manual_single_person": "yes",
                    "target_identity_negative": True,
                    "same_team_distractor_source_frozen": True,
                    "future_hard_negative_crop_review_source_eligible": True,
                    "target_negative": True,
                }
            )
        elif decision == "other_team_player":
            notes = ""
            if code == "EXT_248":
                notes = (
                    "human_note_white_team_number_5_is_not_target_001;"
                    "not_target_jersey_identity_evidence"
                )
            row.update(
                {
                    "manual_same_target_as_target_001": "no",
                    "manual_same_team_as_target": "no",
                    "manual_crop_valid": "yes",
                    "manual_target_dominant": "no",
                    "manual_single_person": "yes",
                    "manual_notes": notes,
                    "same_team_hard_negative_eligible": False,
                }
            )
        elif decision == "non_player":
            row.update(
                {
                    "manual_same_target_as_target_001": "no",
                    "manual_same_team_as_target": "no",
                    "manual_crop_valid": "yes",
                    "manual_target_dominant": "no",
                    "manual_single_person": "",
                    "manual_notes": "referee_or_non_player_human_track",
                }
            )
        elif decision == "uncertain":
            row.update(
                {
                    "manual_same_target_as_target_001": "uncertain",
                    "manual_same_team_as_target": "yes",
                    "manual_crop_valid": "yes",
                    "manual_target_dominant": "uncertain",
                    "manual_single_person": "uncertain",
                    "manual_identity_continuity_observed": "uncertain",
                    "target_positive": False,
                    "target_negative": False,
                }
            )
        elif decision == "invalid":
            row.update(
                {
                    "manual_same_target_as_target_001": "uncertain",
                    "manual_same_team_as_target": "uncertain",
                    "manual_crop_valid": "no",
                    "manual_target_dominant": "no",
                    "manual_single_person": "uncertain",
                }
            )
        elif decision == "multi_person_ambiguous":
            if code == "EXT_213":
                row.update(
                    {
                        "manual_same_target_as_target_001": "yes",
                        "manual_same_team_as_target": "yes",
                        "manual_visible_jersey_number": "5",
                        "jersey_number_provenance": JERSEY_PROVENANCE,
                        "manual_crop_valid": "uncertain",
                        "manual_target_dominant": "uncertain",
                        "manual_single_person": "no",
                        "manual_identity_continuity_observed": "uncertain",
                        "target_present_but_contaminated": True,
                        "additional_target_occurrence_frozen": False,
                        "manual_notes": (
                            "target_001_present_but_merged_with_white_player;"
                            "not_clean_occurrence_or_anchor"
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "manual_same_target_as_target_001": "no",
                        "manual_same_team_as_target": "",
                        "manual_crop_valid": "uncertain",
                        "manual_target_dominant": "uncertain",
                        "manual_single_person": "no",
                        "manual_identity_continuity_observed": "uncertain",
                    }
                )
        else:
            raise RefinementFreezeError(f"unknown decision {decision}")
        rows.append(row)
    counts = Counter(r["manual_refinement_decision"] for r in rows)
    expected = {
        "additional_target_occurrence_yes": 1,
        "same_team_distractor_yes": 35,
        "other_team_player": 52,
        "non_player": 9,
        "uncertain": 4,
        "invalid": 5,
        "multi_person_ambiguous": 29,
    }
    if dict(counts) != expected:
        raise RefinementFreezeError(f"occurrence distribution mismatch {dict(counts)}")
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            for k, v in list(out.items()):
                if isinstance(v, bool):
                    out[k] = "true" if v else "false"
            writer.writerow(out)


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3d_external_refinement_manual_freeze_contract_v1",
        "target_id": TARGET_ID,
        "decisions_frozen": True,
        "sample_read": False,
        "sample_used_for_decision": False,
        "similarity_used": False,
        "automated_ocr_used": False,
        "model_identity_used": False,
        "gallery_mutation": False,
        "hard_negative_gallery_built": False,
        "threshold_selected": False,
        "identity_assignment": False,
        "new_embeddings": 0,
        "similarity_rows": 0,
        "hard_negative_gallery_members": 0,
        "gallery_members": 7,
        "new_crop_copies": 0,
        "exact_next_gate": NEXT_GATE,
    }


def make_tmp(project_root: Path, final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3d_external_refinement_freeze_{final_dir.name}_{token}"
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise RefinementFreezeError("final_exists")
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
        raise RefinementFreezeError("final_exists")

    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise RefinementFreezeError("sample decode_forbidden required")
    sample_path = project_root / config["evaluation_source"]["path"]
    runtime_audit = {
        "schema_version": "reid_stage5d_f3d_runtime_audit_v1",
        "sample_video_read": False,
        "sample_crop_read_count": 0,
        "sample_embedding_read_count": 0,
        "sample_score_row_read_count": 0,
        "sample_used_for_decision": False,
        "sample_used_for_gallery_optimization": False,
        "new_detection": False,
        "new_tracking": False,
        "osnet_inference": False,
        "ocr_inference": False,
        "similarity_scoring": 0,
        "new_embeddings": 0,
        "new_crop_copies": 0,
        "gallery_mutation": False,
        "network_download": 0,
        "sample_path_exists_but_unread": sample_path.is_file(),
    }

    f3c = validate_f3c(project_root, config)
    upstream = validate_upstream(project_root, config)
    decision_map = decision_code_set()
    human = config["human_freeze"]
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    view_rows = build_view_decisions(
        f3c["view_inv"],
        reviewer=human["reviewer"],
        final_approver=human["final_approver"],
        reviewed_at=reviewed_at,
    )
    occ_rows = build_occurrence_decisions(
        f3c["occ_inv"],
        decision_map=decision_map,
        reviewer=human["reviewer"],
        final_approver=human["final_approver"],
        reviewed_at=reviewed_at,
    )

    tmp = make_tmp(project_root, final_dir)
    generated_at = reviewed_at
    try:
        freeze = tmp / "manual_freeze"
        validation = tmp / "validation"
        runtime = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (freeze, validation, runtime, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        write_csv(
            freeze / "target_001_external_target_view_decisions_frozen.csv",
            VIEW_CSV_FIELDS,
            view_rows,
        )
        write_csv(
            freeze / "target_001_external_occurrence_refinement_decisions_frozen.csv",
            OCC_CSV_FIELDS,
            occ_rows,
        )
        add_rows = [
            r for r in occ_rows if r["manual_refinement_decision"] == "additional_target_occurrence_yes"
        ]
        dist_rows = [
            r for r in occ_rows if r["manual_refinement_decision"] == "same_team_distractor_yes"
        ]
        exp_rows = [
            r
            for r in view_rows
            if r["manual_anchor_expansion_decision"] == "target_anchor_expansion_yes"
        ]
        if len(add_rows) != 1 or add_rows[0]["external_candidate_code"] != "EXT_161":
            raise RefinementFreezeError("EXT_161 freeze mismatch")
        if len(dist_rows) != 35:
            raise RefinementFreezeError("distractor freeze count mismatch")
        if [r["target_view_candidate_id"] for r in exp_rows] != list(VIEW_EXPANSION_YES):
            raise RefinementFreezeError("expansion freeze IDs mismatch")
        write_csv(
            freeze / "target_001_external_additional_target_occurrences_frozen.csv",
            OCC_CSV_FIELDS,
            add_rows,
        )
        write_csv(
            freeze / "target_001_external_same_team_distractor_sources_frozen.csv",
            OCC_CSV_FIELDS,
            dist_rows,
        )
        write_csv(
            freeze / "target_001_external_target_anchor_expansion_sources_frozen.csv",
            VIEW_CSV_FIELDS,
            exp_rows,
        )

        view_dist = dict(Counter(r["manual_anchor_expansion_decision"] for r in view_rows))
        occ_dist = dict(Counter(r["manual_refinement_decision"] for r in occ_rows))
        ext213 = next(r for r in occ_rows if r["external_candidate_code"] == "EXT_213")
        freeze_payload = {
            "schema_version": "reid_target_external_refinement_manual_freeze_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "target_definition_path": config["target_definition"]["path"],
            "target_definition_sha256": upstream["target_definition_sha256"],
            "external_source_path": config["external_enrollment_source"]["path"],
            "external_source_sha256": config["external_enrollment_source"][
                "expected_sha256"
            ],
            "f3b_path": config["stage5d_f3b_refinement_design"]["path"],
            "f3b_snapshot_sha256": upstream["f3b_snapshot_sha256"],
            "f3c_path": config["stage5d_f3c_package"]["path"],
            "f3c_snapshot_sha256": f3c["snapshot_sha256"],
            "existing_frozen_target_occurrences": list(EXISTING_FROZEN_OCCURRENCES),
            "new_frozen_target_occurrence": "EXT_161",
            "total_frozen_target_occurrences_after_f3d": 4,
            "existing_frozen_target_anchors": list(EXISTING_ANCHOR_IDS),
            "newly_approved_target_expansion_candidates": list(VIEW_EXPANSION_YES),
            "total_human_approved_target_anchor_sources_after_f3d": 9,
            "official_gallery_v1_members": 7,
            "same_team_distractor_source_occurrences": list(SAME_TEAM_DISTRACTORS),
            "hard_negative_gallery_members": 0,
            "target_view_decision_distribution": view_dist,
            "occurrence_decision_distribution": occ_dist,
            "target_view_decisions": {
                "expansion_yes": list(VIEW_EXPANSION_YES),
                "expansion_no": list(VIEW_EXPANSION_NO),
                "multi_person_ambiguous": list(VIEW_AMBIGUOUS),
            },
            "occurrence_decisions": {
                "additional_target_occurrence_yes": list(ADDITIONAL_TARGET),
                "same_team_distractor_yes": list(SAME_TEAM_DISTRACTORS),
                "other_team_player": list(OTHER_TEAM),
                "non_player": list(NON_PLAYER),
                "uncertain": list(UNCERTAIN),
                "invalid": list(INVALID),
                "multi_person_ambiguous": list(AMBIGUOUS),
            },
            "visible_jersey_human_metadata": {
                "EXT_161": "5",
                "EXT_213": "5",
                **DISTRACTOR_JERSEY,
            },
            "jersey_number_provenance": JERSEY_PROVENANCE,
            "special_ext_213_record": {
                "manual_refinement_decision": "multi_person_ambiguous",
                "manual_same_target_as_target_001": "yes",
                "target_present_but_contaminated": True,
                "additional_target_occurrence_frozen": False,
                "target_anchor_eligible": False,
                "hard_negative_eligible": False,
                "manual_visible_jersey_number": "5",
            },
            "transcription_corrections": [
                {"from": a, "to": b, "decision": c}
                for a, b, c in TRANSCRIPTION_CORRECTIONS
            ],
            "sample_access_read": False,
            "similarity_used": False,
            "ocr_used": False,
            "model_identity_used": False,
            "gallery_mutation": False,
            "hard_negative_gallery_built": False,
            "threshold_selected": False,
            "identity_assignment": False,
            "decisions_frozen": True,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "reviewed_at": reviewed_at,
            "project_head": head,
            "generated_at": generated_at,
            "ext213_snapshot": {
                k: ext213[k]
                for k in (
                    "external_candidate_code",
                    "manual_refinement_decision",
                    "manual_same_target_as_target_001",
                    "target_present_but_contaminated",
                    "manual_visible_jersey_number",
                )
            },
        }
        write_json(
            freeze / "target_001_external_refinement_manual_freeze.json", freeze_payload
        )
        write_json(
            freeze / "target_001_external_refinement_manual_freeze_contract.json",
            {
                **build_contract(),
                "generated_at": generated_at,
                "project_head": head,
                "f3c_snapshot_sha256": f3c["snapshot_sha256"],
            },
        )

        validation_payload = {
            "schema_version": "reid_target_001_external_refinement_manual_freeze_validation_v1",
            "inventory_code_set_equals_decision_code_set": True,
            "missing": 0,
            "extra": 0,
            "duplicate": 0,
            "unknown": 0,
            "reviewed_target_view": 11,
            "reviewed_external_occurrence": 135,
            "target_view_distribution": view_dist,
            "occurrence_distribution": occ_dist,
            "transcription_corrections_applied": 5,
            "uncertain_forced_negative": False,
            "invalid_forced_negative": False,
            "ambiguous_forced_negative": False,
            "distractor_is_hard_negative_member": False,
            "expansion_is_gallery_member": False,
            "gallery_members": 7,
            "sample_reads": 0,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
        }
        write_json(
            validation / "target_001_external_refinement_manual_freeze_validation.json",
            validation_payload,
        )
        write_json(runtime / "stage5d_f3d_runtime_audit.json", runtime_audit)

        file_count, files_sha = listing_sha(tmp)
        write_json(
            freeze / "target_001_external_refinement_manual_freeze_manifest.json",
            {
                "schema_version": "reid_target_001_external_refinement_manual_freeze_manifest_v1",
                "csv_files": [
                    "target_001_external_target_view_decisions_frozen.csv",
                    "target_001_external_occurrence_refinement_decisions_frozen.csv",
                    "target_001_external_additional_target_occurrences_frozen.csv",
                    "target_001_external_same_team_distractor_sources_frozen.csv",
                    "target_001_external_target_anchor_expansion_sources_frozen.csv",
                ],
                "json_files": [
                    "target_001_external_refinement_manual_freeze.json",
                    "target_001_external_refinement_manual_freeze_contract.json",
                    "target_001_external_refinement_manual_freeze_manifest.json",
                ],
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
            },
        )
        write_json(
            tmp / "stage5d_f3d_contract.json",
            {
                **build_contract(),
                "final_status": FINAL_STATUS,
                "generated_at": generated_at,
                "project_head": head,
            },
        )
        summary = {
            "schema_version": "reid_stage5d_f3d_external_refinement_manual_freeze_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reviewed_target_view": 11,
            "target_anchor_expansion_yes": 2,
            "target_anchor_expansion_no": 7,
            "target_view_multi_person_ambiguous": 2,
            "target_view_uncertain": 0,
            "target_view_invalid": 0,
            "reviewed_external_occurrence": 135,
            "additional_target_occurrence_yes": 1,
            "same_team_distractor_yes": 35,
            "other_team_player": 52,
            "non_player": 9,
            "uncertain": 4,
            "invalid": 5,
            "multi_person_ambiguous": 29,
            "existing_frozen_target_occurrences": 3,
            "new_frozen_target_occurrence": 1,
            "total_frozen_target_occurrences_after_f3d": 4,
            "existing_frozen_target_anchors": 7,
            "newly_approved_target_expansion_candidates": 2,
            "total_human_approved_target_anchor_sources_after_f3d": 9,
            "official_gallery_v1_members": 7,
            "same_team_distractor_source_occurrences": 35,
            "hard_negative_gallery_members": 0,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "sample_used_for_decision": False,
            "transcription_corrections_applied": 5,
            "inventory_coverage_missing": 0,
            "inventory_coverage_extra": 0,
            "approved_expansion_candidate_ids": list(VIEW_EXPANSION_YES),
            "additional_target_occurrence_code": "EXT_161",
            "f3c_snapshot_sha256": f3c["snapshot_sha256"],
            "f3b_snapshot_sha256": upstream["f3b_snapshot_sha256"],
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3d_summary.json", summary)
        write_json(
            tmp / "stage5d_f3d_manifest.json",
            {
                "schema_version": "reid_stage5d_f3d_external_refinement_manual_freeze_manifest_v1",
                "final_status": FINAL_STATUS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "new_crop_copies": 0,
                    "new_embeddings": 0,
                    "similarity_rows": 0,
                    "hard_negative_members": 0,
                    "gallery_members": 7,
                    "threshold": False,
                    "identity_assignments": 0,
                },
                "generated_at": generated_at,
            },
        )

        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.npz")):
            raise RefinementFreezeError("embedding artifacts forbidden")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise RefinementFreezeError("media artifacts forbidden in freeze")
        if list(tmp.rglob("sample.mp4")):
            raise RefinementFreezeError("sample leakage")

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f3d_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_refinement_manual_freeze_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "exact_next_gate": summary["exact_next_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
