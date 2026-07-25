#!/usr/bin/env python3
"""Stage 5D-B1E-E — freeze human external anchor decisions.

Approves exactly seven target_anchor_yes crops. target_anchor_no is
not_selected_redundant_valid_target_crop (not identity-negative).
No embeddings, gallery, OCR, crop copies, or new inference.
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
CONFIG_SCHEMA = "reid_external_anchor_manual_freeze_config_v1"
CANDIDATE_IDS = tuple(f"target_001_ext_anchor_{i:03d}" for i in range(1, 16))
APPROVED_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)
REDUNDANT_IDS = (
    "target_001_ext_anchor_002",
    "target_001_ext_anchor_005",
    "target_001_ext_anchor_009",
    "target_001_ext_anchor_012",
    "target_001_ext_anchor_015",
)
AMBIGUOUS_IDS = (
    "target_001_ext_anchor_007",
    "target_001_ext_anchor_013",
)
INVALID_IDS = ("target_001_ext_anchor_010",)
REDUNDANT_REASON = "not_selected_redundant_valid_target_crop"
FINAL_STATUS = "COMPLETED_STAGE5D_B1E_E_TARGET_001_EXTERNAL_ANCHORS_FROZEN"
NEXT_GATE = "STAGE5D-B1E-F_TARGET_001_FROZEN_ANCHOR_OSNET_EMBEDDING_AND_GALLERY_BUILD"

DECISION_CSV_FIELDS = (
    "anchor_candidate_id",
    "target_id",
    "source_occurrence_code",
    "raw_track_id",
    "frame_index",
    "video_time",
    "crop_path",
    "crop_sha256",
    "manual_anchor_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_quality_notes",
    "decision_reason",
    "is_frozen_approved_anchor",
    "is_identity_negative",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
APPROVED_CSV_FIELDS = (
    "frozen_anchor_id",
    "anchor_candidate_id",
    "target_id",
    "source_occurrence_code",
    "raw_track_id",
    "frame_index",
    "video_time",
    "crop_path",
    "crop_sha256",
    "original_bbox_xyxy",
    "padded_crop_bbox_xyxy",
    "crop_width",
    "crop_height",
    "manual_view_category",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "target_definition_sha256",
    "occurrence_freeze_sha256",
    "anchor_review_package_sha256",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "is_gallery_member",
    "embedding_generated",
)


class AnchorFreezeError(RuntimeError):
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
            rel = str(path.relative_to(root))
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
        raise AnchorFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise AnchorFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise AnchorFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise AnchorFreezeError("BLOCKED_STAGE5D_B1E_E_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise AnchorFreezeError("BLOCKED_STAGE5D_B1E_E_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_external_anchor_manual_freeze.py",
        "configs/reid/external_anchor_manual_freeze_stage5d_target_001.yaml",
        "tests/test_reid_external_anchor_manual_freeze.py",
        "docs/setup/stage5d-target-external-anchor-manual-review-and-freeze.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise AnchorFreezeError(
                    "BLOCKED_STAGE5D_B1E_E_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add target 001 external anchor review package":
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_GIT_CONTRACT_MISMATCH message"
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
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise AnchorFreezeError("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise AnchorFreezeError("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise AnchorFreezeError("external sha mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise AnchorFreezeError("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise AnchorFreezeError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise AnchorFreezeError("yolo sha mismatch")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise AnchorFreezeError("osnet sha mismatch")
    return {
        "path": config["external_enrollment_source"]["path"],
        "sha256": config["external_enrollment_source"]["expected_sha256"],
    }


def validate_target(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = project_root / config["target_definition"]["path"]
    td = load_json(path)
    if td.get("target_id") != "target_001":
        raise AnchorFreezeError("target_id mismatch")
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise AnchorFreezeError("target_alias mismatch")
    if not td.get("target_definition_frozen"):
        raise AnchorFreezeError("target_definition_frozen required")
    if int(td.get("human_verified_jersey_number")) != 5:
        raise AnchorFreezeError("jersey number mismatch")
    if (
        td.get("jersey_number_provenance")
        != "human_verified_by_user_not_automated_ocr"
    ):
        raise AnchorFreezeError("jersey provenance mismatch")
    if td.get("automated_jersey_used") is not False:
        raise AnchorFreezeError("automated_jersey_used must be false")
    return {"path": config["target_definition"]["path"], "sha256": sha256_file(path), "definition": td}


def validate_b1e_c(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_c_package"]["path"]
    summary = load_json(root / "stage5d_b1e_c_summary.json")
    if summary.get("final_status") != config["stage5d_b1e_c_package"][
        "expected_final_status"
    ]:
        raise AnchorFreezeError("B1E-C status mismatch")
    if tuple(summary.get("selected_external_candidate_codes") or ()) != (
        "EXT_004",
        "EXT_183",
        "EXT_198",
    ):
        raise AnchorFreezeError("B1E-C codes mismatch")
    man = root / "stage5d_b1e_c_manifest.json"
    return {
        "root": root,
        "summary": summary,
        "sha256": sha256_file(man) if man.is_file() else sha256_file(root / "stage5d_b1e_c_summary.json"),
    }


def validate_b1e_d(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_d_package"]["path"]
    summary = load_json(root / "stage5d_b1e_d_summary.json")
    manifest = load_json(root / "stage5d_b1e_d_manifest.json")
    exp = config["stage5d_b1e_d_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH target"
        )
    if list(summary.get("selected_occurrence_codes") or []) != [
        "EXT_004",
        "EXT_183",
        "EXT_198",
    ]:
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH occurrences"
        )
    if int(summary.get("source_observation_count")) != 501:
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH observations"
        )
    if int(summary.get("total_candidate_count")) != int(exp["expected_candidate_count"]):
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH candidates"
        )
    for code, count in exp["expected_per_occurrence"].items():
        if int(summary["candidate_counts_per_occurrence"][code]) != int(count):
            raise AnchorFreezeError(
                f"BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH {code}"
            )
    for key in (
        "approved_anchor_crops",
        "embeddings",
        "gallery_members",
        "prototypes",
        "identity_assignments",
        "manual_decisions",
    ):
        if int(summary.get(key) or 0) != 0:
            raise AnchorFreezeError(
                f"BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH {key}"
            )

    inv_path = root / "inventory" / "target_001_external_anchor_candidate_inventory.jsonl"
    inventory = [
        json.loads(line)
        for line in inv_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [r["anchor_candidate_id"] for r in inventory]
    if ids != list(CANDIDATE_IDS):
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH candidate_ids"
        )
    for row in inventory:
        crop = root / row["crop_path"]
        if not crop.is_file():
            raise AnchorFreezeError(f"missing crop {row['crop_path']}")
        if sha256_file(crop) != row["crop_sha256"]:
            raise AnchorFreezeError(f"crop sha mismatch {row['anchor_candidate_id']}")
        if row.get("manual_anchor_decision"):
            raise AnchorFreezeError("inventory prefilled decision")

    tpl = root / "templates" / "target_001_external_anchor_crop_review_template.csv"
    with tpl.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH template_rows"
        )
    if any(r.get("manual_anchor_decision") for r in rows):
        raise AnchorFreezeError(
            "BLOCKED_STAGE5D_B1E_E_REVIEW_PACKAGE_CONTRACT_MISMATCH template_prefilled"
        )

    snap = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    package_sha = str(manifest.get("listing_sha256") or sha256_file(inv_path))
    return {
        "root": root,
        "summary": summary,
        "manifest": manifest,
        "inventory": inventory,
        "snapshot_sha256": snap,
        "package_sha256": package_sha,
    }


def decision_reason(cid: str, decision: str, redundant_reason: str) -> str:
    if decision == "target_anchor_yes":
        return "approved_frozen_anchor"
    if decision == "target_anchor_no":
        if cid not in REDUNDANT_IDS:
            raise AnchorFreezeError(f"unexpected target_anchor_no for {cid}")
        return redundant_reason
    if decision == "multi_person_ambiguous":
        return "excluded_multi_person_ambiguous"
    if decision == "invalid":
        return "excluded_invalid_crop"
    raise AnchorFreezeError(f"unexpected decision {decision} for {cid}")


def validate_human_decisions(config: Mapping[str, Any]) -> dict[str, Any]:
    human = config["human_anchor_freeze"]
    decisions = human["decisions"]
    if set(decisions) != set(CANDIDATE_IDS):
        raise AnchorFreezeError("decision coverage must be exact 15 candidates")
    if tuple(human["approved_exact_ids"]) != APPROVED_IDS:
        raise AnchorFreezeError("approved_exact_ids mismatch")
    if human.get("redundant_nonselected_reason") != REDUNDANT_REASON:
        raise AnchorFreezeError("redundant reason mismatch")
    if human.get("automated_ocr_used") is not False:
        raise AnchorFreezeError("automated_ocr_used must be false")
    if human.get("similarity_used") is not False:
        raise AnchorFreezeError("similarity_used must be false")
    if human.get("model_identity_prediction_used") is not False:
        raise AnchorFreezeError("model_identity_prediction_used must be false")

    counts: Counter[str] = Counter()
    for cid in CANDIDATE_IDS:
        d = decisions[cid]
        decision = d["manual_anchor_decision"]
        counts[decision] += 1
        if cid in APPROVED_IDS:
            if decision != "target_anchor_yes":
                raise AnchorFreezeError(f"approved id must be yes: {cid}")
            for field in (
                "manual_crop_valid",
                "manual_target_dominant",
                "manual_single_person",
            ):
                if str(d[field]) != "yes":
                    raise AnchorFreezeError(f"{cid} {field} must be yes")
        if cid in REDUNDANT_IDS and decision != "target_anchor_no":
            raise AnchorFreezeError(f"redundant id must be no: {cid}")
        if cid in AMBIGUOUS_IDS and decision != "multi_person_ambiguous":
            raise AnchorFreezeError(f"ambiguous id mismatch: {cid}")
        if cid in INVALID_IDS and decision != "invalid":
            raise AnchorFreezeError(f"invalid id mismatch: {cid}")
    expected_counts = {
        "target_anchor_yes": 7,
        "target_anchor_no": 5,
        "multi_person_ambiguous": 2,
        "invalid": 1,
    }
    for key, value in expected_counts.items():
        if counts[key] != value:
            raise AnchorFreezeError(f"decision count mismatch {key}={counts[key]}")
    if counts.get("uncertain", 0) != 0 or counts.get("non_player", 0) != 0:
        raise AnchorFreezeError("unexpected uncertain/non_player decisions")
    return human


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1e_e_anchor_freeze_contract_v1",
        "target_id": "target_001",
        "reviewed_candidate_crops": 15,
        "frozen_approved_anchors": 7,
        "redundant_valid_non_selected": 5,
        "multi_person_ambiguous": 2,
        "invalid": 1,
        "target_anchor_no_is_identity_negative": False,
        "redundant_reason": REDUNDANT_REASON,
        "manual_decisions_frozen": True,
        "automated_approval": False,
        "osnet_used": False,
        "ocr_used": False,
        "similarity_used": False,
        "frozen_anchors_are_gallery_members": False,
        "embedding_generated": False,
        "automatic_gallery_growth": False,
        "unknown_identity_preserved": True,
        "crop_copies": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "exact_next_gate": NEXT_GATE,
    }


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or _PROJECT_ROOT
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise AnchorFreezeError("final_exists")

    assets = validate_assets(project_root, config)
    target = validate_target(project_root, config)
    b1ec = validate_b1e_c(project_root, config)
    b1ed = validate_b1e_d(project_root, config)
    human = validate_human_decisions(config)
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    decisions_cfg = human["decisions"]
    inv_by_id = {r["anchor_candidate_id"]: r for r in b1ed["inventory"]}

    decision_rows: list[dict[str, Any]] = []
    approved_rows: list[dict[str, Any]] = []
    for cid in CANDIDATE_IDS:
        cand = inv_by_id[cid]
        d = decisions_cfg[cid]
        decision = d["manual_anchor_decision"]
        reason = decision_reason(
            cid, decision, human["redundant_nonselected_reason"]
        )
        is_approved = decision == "target_anchor_yes"
        row = {
            "anchor_candidate_id": cid,
            "target_id": "target_001",
            "source_occurrence_code": cand["source_occurrence_code"],
            "raw_track_id": cand["raw_track_id"],
            "frame_index": cand["frame_index"],
            "video_time": cand["video_time"],
            "crop_path": cand["crop_path"],
            "crop_sha256": cand["crop_sha256"],
            "manual_anchor_decision": decision,
            "manual_crop_valid": d["manual_crop_valid"],
            "manual_target_dominant": d["manual_target_dominant"],
            "manual_single_person": d["manual_single_person"],
            "manual_identity_confirmed": human["manual_identity_confirmed"],
            "manual_view_category": d["manual_view_category"],
            "manual_quality_notes": " ".join(str(d["manual_quality_notes"]).split()),
            "decision_reason": reason,
            "is_frozen_approved_anchor": is_approved,
            "is_identity_negative": False,
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "reviewed_at": reviewed_at,
        }
        decision_rows.append(row)
        # Re-verify crop SHA without copying.
        crop_abs = b1ed["root"] / cand["crop_path"]
        if sha256_file(crop_abs) != cand["crop_sha256"]:
            raise AnchorFreezeError(f"crop mutated before freeze {cid}")
        if is_approved:
            approved_rows.append(
                {
                    "frozen_anchor_id": f"frozen_{cid}",
                    "anchor_candidate_id": cid,
                    "target_id": "target_001",
                    "source_occurrence_code": cand["source_occurrence_code"],
                    "raw_track_id": cand["raw_track_id"],
                    "frame_index": cand["frame_index"],
                    "video_time": cand["video_time"],
                    "crop_path": cand["crop_path"],
                    "crop_sha256": cand["crop_sha256"],
                    "original_bbox_xyxy": json.dumps(cand["original_bbox_xyxy"]),
                    "padded_crop_bbox_xyxy": json.dumps(cand["padded_crop_bbox_xyxy"]),
                    "crop_width": cand["crop_width"],
                    "crop_height": cand["crop_height"],
                    "manual_view_category": d["manual_view_category"],
                    "manual_crop_valid": d["manual_crop_valid"],
                    "manual_target_dominant": d["manual_target_dominant"],
                    "manual_single_person": d["manual_single_person"],
                    "manual_identity_confirmed": human["manual_identity_confirmed"],
                    "target_definition_sha256": target["sha256"],
                    "occurrence_freeze_sha256": b1ec["sha256"],
                    "anchor_review_package_sha256": b1ed["package_sha256"],
                    "reviewer": human["reviewer"],
                    "final_approver": human["final_approver"],
                    "reviewed_at": reviewed_at,
                    "is_gallery_member": False,
                    "embedding_generated": False,
                }
            )

    if len(approved_rows) != 7:
        raise AnchorFreezeError(f"expected 7 approved, got {len(approved_rows)}")
    approved_ids = [r["anchor_candidate_id"] for r in approved_rows]
    if approved_ids != list(APPROVED_IDS):
        raise AnchorFreezeError("approved id order/content mismatch")

    occ_dist = Counter(r["source_occurrence_code"] for r in approved_rows)
    if dict(occ_dist) != {"EXT_004": 4, "EXT_183": 1, "EXT_198": 2}:
        raise AnchorFreezeError(f"occurrence distribution mismatch {dict(occ_dist)}")
    view_dist = Counter(r["manual_view_category"] for r in approved_rows)
    expected_views = {
        "front": 1,
        "front_oblique": 1,
        "right_side": 1,
        "rear_oblique": 2,
        "rear": 2,
    }
    if dict(view_dist) != expected_views:
        raise AnchorFreezeError(f"view distribution mismatch {dict(view_dist)}")

    tmp = project_root / (
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_freeze_"
        + uuid.uuid4().hex
    )
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        freeze_dir = tmp / "anchor_freeze"
        val_dir = tmp / "validation"
        runtime_dir = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (freeze_dir, val_dir, runtime_dir, cfg_dir):
            d.mkdir(parents=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        with (
            freeze_dir / "target_001_external_anchor_review_decisions_frozen.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DECISION_CSV_FIELDS))
            writer.writeheader()
            for row in decision_rows:
                writer.writerow(row)

        with (
            freeze_dir / "target_001_external_approved_anchors_frozen.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(APPROVED_CSV_FIELDS))
            writer.writeheader()
            for row in approved_rows:
                writer.writerow(row)

        freeze_payload = {
            "schema_version": "reid_target_external_anchor_freeze_v1",
            "final_status": FINAL_STATUS,
            "target_id": "target_001",
            "target_definition_path": target["path"],
            "target_definition_sha256": target["sha256"],
            "target_alias": target["definition"]["target_alias"],
            "human_verified_jersey_number": 5,
            "external_source_path": assets["path"],
            "external_source_sha256": assets["sha256"],
            "occurrence_freeze_path": config["stage5d_b1e_c_package"]["path"],
            "occurrence_freeze_sha256": b1ec["sha256"],
            "anchor_review_package_path": config["stage5d_b1e_d_package"]["path"],
            "anchor_review_package_sha256": b1ed["package_sha256"],
            "reviewed_count": 15,
            "approved_count": 7,
            "redundant_valid_non_selected_count": 5,
            "multi_person_ambiguous_count": 2,
            "invalid_count": 1,
            "approved_exact_ids": list(APPROVED_IDS),
            "occurrence_distribution": dict(occ_dist),
            "view_distribution": dict(view_dist),
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "reviewed_at": reviewed_at,
            "manual_decisions_frozen": True,
            "automated_approval": False,
            "osnet_used": False,
            "ocr_used": False,
            "similarity_used": False,
            "model_identity_prediction_used": False,
            "frozen_anchors_are_gallery_members": False,
            "embedding_generated": False,
            "automatic_gallery_growth": False,
            "unknown_identity_preserved": True,
            "target_anchor_no_is_identity_negative": False,
            "redundant_nonselected_reason": REDUNDANT_REASON,
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            freeze_dir / "target_001_external_anchor_freeze.json", freeze_payload
        )

        contract = build_contract()
        write_json(
            freeze_dir / "target_001_external_anchor_freeze_contract.json", contract
        )
        write_json(
            freeze_dir / "target_001_external_anchor_freeze_manifest.json",
            {
                "schema_version": "reid_stage5d_b1e_e_anchor_freeze_manifest_v1",
                "decisions_csv": (
                    "anchor_freeze/target_001_external_anchor_review_decisions_frozen.csv"
                ),
                "approved_csv": (
                    "anchor_freeze/target_001_external_approved_anchors_frozen.csv"
                ),
                "freeze_json": "anchor_freeze/target_001_external_anchor_freeze.json",
                "reviewed_count": 15,
                "approved_count": 7,
                "crop_copies": 0,
                "png_count": 0,
                "mp4_count": 0,
            },
        )

        validation = {
            "schema_version": "reid_stage5d_b1e_e_anchor_decision_validation_v1",
            "candidate_ids": list(CANDIDATE_IDS),
            "approved_exact_ids": list(APPROVED_IDS),
            "redundant_ids": list(REDUNDANT_IDS),
            "ambiguous_ids": list(AMBIGUOUS_IDS),
            "invalid_ids": list(INVALID_IDS),
            "decision_counts": {
                "target_anchor_yes": 7,
                "target_anchor_no": 5,
                "multi_person_ambiguous": 2,
                "invalid": 1,
                "uncertain": 0,
                "non_player": 0,
            },
            "occurrence_distribution": dict(occ_dist),
            "view_distribution": dict(view_dist),
            "all_crop_shas_verified": True,
            "no_crop_copies": True,
            "target_anchor_no_not_identity_negative": True,
            "redundant_reason": REDUNDANT_REASON,
        }
        write_json(
            val_dir / "target_001_external_anchor_decision_validation.json",
            validation,
        )

        write_json(
            runtime_dir / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1e_e_runtime_v1",
                "started_at": reviewed_at,
                "project_head": head,
                "osnet_loaded": False,
                "yolo_loaded": False,
                "yolo_inference": 0,
                "bytetrack_inference": 0,
                "network_download": 0,
            },
        )

        summary = {
            "schema_version": "reid_stage5d_b1e_e_summary_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": "sarı takım 5 numaralı oyuncu",
            "external_source_sha256": assets["sha256"],
            "reviewed_candidate_crops": 15,
            "frozen_approved_anchors": 7,
            "redundant_valid_non_selected": 5,
            "multi_person_ambiguous": 2,
            "invalid": 1,
            "approved_exact_ids": list(APPROVED_IDS),
            "occurrence_distribution": dict(occ_dist),
            "view_distribution": dict(view_dist),
            "manual_decisions_frozen": True,
            "osnet_inference": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "embeddings": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "crop_copies": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "b1e_d_snapshot_sha256": b1ed["snapshot_sha256"],
            "exact_next_gate": NEXT_GATE,
        }
        write_json(tmp / "stage5d_b1e_e_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.pt")):
            raise AnchorFreezeError("artifact budget violated")

        n_files, listing = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_b1e_e_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "listing_file_count_before_manifest": n_files,
            "listing_sha256_before_manifest": listing,
            "reviewed_candidate_crops": 15,
            "frozen_approved_anchors": 7,
            "gallery_members": 0,
            "embeddings": 0,
            "crop_copies": 0,
        }
        write_json(tmp / "stage5d_b1e_e_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_b1e_e_manifest.json", manifest)

        os.replace(str(tmp), str(final_dir))
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "final_status": FINAL_STATUS,
        "reviewed": 15,
        "approved": 7,
        "approved_ids": list(APPROVED_IDS),
        "next": NEXT_GATE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5D-B1E-E external anchor manual freeze"
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
