#!/usr/bin/env python3
"""Stage 5D-F3F — freeze human external refinement crop decisions.

Freezes exact Furkan-approved EXT_161 target crops and hard-negative crop
decisions from the F3E review package. No embeddings, gallery mutation,
OCR/similarity, threshold, or identity assignment.
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
CONFIG_SCHEMA = "reid_external_refinement_crop_manual_freeze_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3F_TARGET_001_EXTERNAL_REFINEMENT_CROP_DECISIONS_FROZEN"
)
NEXT_GATE = (
    "STAGE5D-F3G_TARGET_001_APPROVED_CROP_OSNET_EMBEDDING_AND_GALLERY_V2_BUILD"
)
ALLOWED_DIRTY = {
    "scripts/run_reid_external_refinement_crop_manual_freeze.py",
    "configs/reid/external_refinement_crop_manual_freeze_stage5d_target_001.yaml",
    "tests/test_reid_external_refinement_crop_manual_freeze.py",
    "docs/setup/stage5d-target-external-refinement-crop-manual-review-and-freeze.md",
}

TARGET_APPROVED = (
    "target_001_ext_refine_target_candidate_001",
    "target_001_ext_refine_target_candidate_002",
    "target_001_ext_refine_target_candidate_003",
    "target_001_ext_refine_target_candidate_004",
)

# Sequence numbers 1..35 map to target_001_ext_hard_negative_candidate_NNN.
HN_YES = (2, 5, 6, 7, 8, 11, 12, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27, 29, 30, 31, 33, 34, 35)
HN_NO_WRONG_TEAM = (4, 13, 16, 17)
HN_INVALID = (1, 10, 23, 28, 32)
HN_AMBIGUOUS = (3, 9, 22)

REASON_WRONG_TEAM = "wrong_team_crop_or_track_impurity"
REASON_INVALID = "partial_body_or_edge_clipped_crop"
REASON_AMBIGUOUS = "additional_player_contamination"

TARGET_CSV_FIELDS = (
    "target_crop_candidate_id",
    "source_occurrence_code",
    "raw_track_id",
    "frame_index",
    "video_time",
    "crop_path",
    "crop_sha256",
    "source_bbox",
    "crop_bbox",
    "quality_pass",
    "blur",
    "max_person_iou",
    "edge_clipping",
    "manual_target_crop_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_confirmed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "is_frozen_approved_target_crop",
    "current_gallery_member",
    "embedding_input",
    "automatic_enrollment",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "automated_ocr_used",
    "similarity_used",
    "model_identity_used",
    "decisions_human_approved",
)
HN_CSV_FIELDS = (
    "hard_negative_candidate_id",
    "source_external_code",
    "raw_track_id",
    "selected_frame",
    "video_time",
    "crop_path",
    "crop_sha256",
    "source_bbox",
    "crop_bbox",
    "quality_pass",
    "quality_exception_review_only",
    "quality_exclusion_reasons",
    "blur",
    "max_person_iou",
    "edge_clipping",
    "frozen_source_decision",
    "human_visible_jersey_number",
    "manual_hard_negative_crop_decision",
    "manual_crop_valid",
    "manual_target_absent",
    "manual_same_team_confirmed",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_view_category",
    "manual_scale_category",
    "manual_quality_notes",
    "decision_reason",
    "is_frozen_approved_hard_negative_crop",
    "hard_negative_gallery_member",
    "embedding_input",
    "automatic_enrollment",
    "reviewer",
    "final_approver",
    "reviewed_at",
    "automated_ocr_used",
    "similarity_used",
    "model_identity_used",
    "decisions_human_approved",
)


class CropFreezeError(RuntimeError):
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
        raise CropFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise CropFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise CropFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise CropFreezeError("BLOCKED_STAGE5D_F3F_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise CropFreezeError("BLOCKED_STAGE5D_F3F_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise CropFreezeError(
                    "BLOCKED_STAGE5D_F3F_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise CropFreezeError("BLOCKED_STAGE5D_F3F_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def hn_id(seq: int) -> str:
    return f"target_001_ext_hard_negative_candidate_{seq:03d}"


def hn_decision_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for seq in HN_YES:
        out[hn_id(seq)] = {
            "manual_hard_negative_crop_decision": "hard_negative_crop_yes",
            "manual_crop_valid": "yes",
            "manual_target_absent": "yes",
            "manual_same_team_confirmed": "yes",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "decision_reason": "approved_clean_same_team_distractor_crop",
            "is_frozen_approved_hard_negative_crop": True,
        }
    for seq in HN_NO_WRONG_TEAM:
        out[hn_id(seq)] = {
            "manual_hard_negative_crop_decision": "hard_negative_crop_no",
            "manual_crop_valid": "yes",
            "manual_target_absent": "yes",
            "manual_same_team_confirmed": "no",
            "manual_target_dominant": "no",
            "manual_single_person": "yes",
            "decision_reason": REASON_WRONG_TEAM,
            "is_frozen_approved_hard_negative_crop": False,
        }
    for seq in HN_INVALID:
        out[hn_id(seq)] = {
            "manual_hard_negative_crop_decision": "invalid",
            "manual_crop_valid": "no",
            "manual_target_absent": "uncertain",
            "manual_same_team_confirmed": "uncertain",
            "manual_target_dominant": "no",
            "manual_single_person": "uncertain",
            "decision_reason": REASON_INVALID,
            "is_frozen_approved_hard_negative_crop": False,
        }
    for seq in HN_AMBIGUOUS:
        out[hn_id(seq)] = {
            "manual_hard_negative_crop_decision": "multi_person_ambiguous",
            "manual_crop_valid": "yes",
            "manual_target_absent": "yes",
            "manual_same_team_confirmed": "uncertain",
            "manual_target_dominant": "uncertain",
            "manual_single_person": "no",
            "decision_reason": REASON_AMBIGUOUS,
            "is_frozen_approved_hard_negative_crop": False,
        }
    if len(out) != 35:
        raise CropFreezeError(f"HN decision map size {len(out)}")
    return out


def validate_f3e(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rel = config["stage5d_f3e_package"]["path"]
    assert_no_path_traversal(rel)
    root = project_root / rel
    summary = load_json(root / "stage5d_f3e_summary.json")
    cfg = config["stage5d_f3e_package"]
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH status"
        )
    if int(summary["ext_161_target_crop_candidates"]) != int(
        cfg["expected_target_candidates"]
    ):
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH target_count"
        )
    if int(summary["distractor_crop_candidates"]) != int(
        cfg["expected_hard_negative_candidates"]
    ):
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH hn_count"
        )
    if summary.get("manual_crop_decisions") != 0:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH already_decided"
        )
    if summary.get("gallery_mutation") is not False:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH gallery_mutation"
        )
    if summary.get("sample_video_read") is not False:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH sample"
        )
    target_inv = load_jsonl(
        root / "inventory" / "target_001_EXT_161_target_crop_candidate_inventory.jsonl"
    )
    hn_inv = load_jsonl(
        root
        / "inventory"
        / "target_001_external_hard_negative_crop_candidate_inventory.jsonl"
    )
    target_ids = [r["target_crop_candidate_id"] for r in target_inv]
    hn_ids = [r["hard_negative_candidate_id"] for r in hn_inv]
    if target_ids != list(TARGET_APPROVED):
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH target_ids "
            + json.dumps({"expected": list(TARGET_APPROVED), "got": target_ids})
        )
    expected_hn = [hn_id(i) for i in range(1, 36)]
    if hn_ids != expected_hn:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH hn_ids "
            + json.dumps(
                {
                    "missing": sorted(set(expected_hn) - set(hn_ids)),
                    "extra": sorted(set(hn_ids) - set(expected_hn)),
                }
            )
        )
    # Blank templates must still be blank.
    with (
        root / "templates" / "target_001_EXT_161_target_crop_review_template.csv"
    ).open(encoding="utf-8") as handle:
        trows = list(csv.DictReader(handle))
    with (
        root
        / "templates"
        / "target_001_external_hard_negative_crop_review_template.csv"
    ).open(encoding="utf-8") as handle:
        hrows = list(csv.DictReader(handle))
    if len(trows) != 4 or len(hrows) != 35:
        raise CropFreezeError(
            "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH template_rows"
        )
    for row in trows:
        if row.get("manual_target_crop_decision"):
            raise CropFreezeError(
                "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH target_not_blank"
            )
    for row in hrows:
        if row.get("manual_hard_negative_crop_decision"):
            raise CropFreezeError(
                "BLOCKED_STAGE5D_F3F_CROP_REVIEW_CONTRACT_MISMATCH hn_not_blank"
            )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]), cfg["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "target_inv": target_inv,
        "hn_inv": hn_inv,
        "snapshot_sha256": snap,
    }


def validate_upstream(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    f3d = load_json(
        project_root
        / config["stage5d_f3d_package"]["path"]
        / "stage5d_f3d_summary.json"
    )
    if f3d.get("final_status") != config["stage5d_f3d_package"]["expected_final_status"]:
        raise CropFreezeError("F3D status mismatch")
    gallery = load_json(
        project_root / config["gallery_v1"]["path"] / "stage5d_b1e_f_summary.json"
    )
    if int(gallery["individual_gallery_members"]) != 7:
        raise CropFreezeError("gallery-v1 members mismatch")
    return {"f3d": f3d, "gallery": gallery}


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
                elif isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(out)


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3f_external_refinement_crop_manual_freeze_contract_v1",
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
        "approved_new_target_crops": 4,
        "approved_hard_negative_crops": 23,
        "target_crops_are_not_gallery_members_until_later_gate": True,
        "hard_negative_crops_are_not_gallery_members_until_later_gate": True,
        "exact_next_gate": NEXT_GATE,
    }


def make_tmp(project_root: Path, final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3f_crop_manual_freeze_{final_dir.name}_{token}"
    tmp.mkdir(parents=True, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise CropFreezeError("final_exists")
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
        raise CropFreezeError("final_exists")

    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise CropFreezeError("sample decode_forbidden required")
    sample_path = project_root / config["evaluation_source"]["path"]
    runtime_audit = {
        "schema_version": "reid_stage5d_f3f_runtime_audit_v1",
        "sample_video_read": False,
        "sample_crop_read_count": 0,
        "sample_embedding_read_count": 0,
        "sample_score_row_read_count": 0,
        "sample_rank_row_read_count": 0,
        "sample_used_for_decision": False,
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

    f3e = validate_f3e(project_root, config)
    upstream = validate_upstream(project_root, config)
    human = config["human_freeze"]
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hn_map = hn_decision_map()

    target_rows: list[dict[str, Any]] = []
    for src in f3e["target_inv"]:
        cid = src["target_crop_candidate_id"]
        if cid not in TARGET_APPROVED:
            raise CropFreezeError(f"unexpected target candidate {cid}")
        q = src["quality"]
        target_rows.append(
            {
                "target_crop_candidate_id": cid,
                "source_occurrence_code": src["source_occurrence_code"],
                "raw_track_id": int(src["raw_track_id"]),
                "frame_index": int(src["frame_index"]),
                "video_time": float(src["video_time"]),
                "crop_path": src["crop_path"],
                "crop_sha256": src["crop_sha256"],
                "source_bbox": src["source_bbox"],
                "crop_bbox": src["crop_bbox"],
                "quality_pass": src["hard_quality_pass"],
                "blur": q["laplacian_variance"],
                "max_person_iou": q["max_person_iou"],
                "edge_clipping": q["edge_clipping_fraction"],
                "manual_target_crop_decision": "target_crop_yes",
                "manual_crop_valid": "yes",
                "manual_target_dominant": "yes",
                "manual_single_person": "yes",
                "manual_identity_confirmed": "yes",
                "manual_view_category": "",
                "manual_scale_category": "",
                "manual_quality_notes": "approved_clean_ext_161_target_crop",
                "is_frozen_approved_target_crop": True,
                "current_gallery_member": False,
                "embedding_input": False,
                "automatic_enrollment": False,
                "reviewer": human["reviewer"],
                "final_approver": human["final_approver"],
                "reviewed_at": reviewed_at,
                "automated_ocr_used": False,
                "similarity_used": False,
                "model_identity_used": False,
                "decisions_human_approved": True,
            }
        )

    hn_rows: list[dict[str, Any]] = []
    for src in f3e["hn_inv"]:
        cid = src["hard_negative_candidate_id"]
        if cid not in hn_map:
            raise CropFreezeError(f"missing HN decision for {cid}")
        dec = hn_map[cid]
        q = src["quality"]
        hn_rows.append(
            {
                "hard_negative_candidate_id": cid,
                "source_external_code": src["source_external_code"],
                "raw_track_id": int(src["raw_track_id"]),
                "selected_frame": int(src["selected_frame"]),
                "video_time": float(src["video_time"]),
                "crop_path": src["crop_path"],
                "crop_sha256": src["crop_sha256"],
                "source_bbox": src["source_bbox"],
                "crop_bbox": src["crop_bbox"],
                "quality_pass": src["hard_quality_pass"],
                "quality_exception_review_only": src["quality_exception_review_only"],
                "quality_exclusion_reasons": src.get("quality_exclusion_reasons") or [],
                "blur": q["laplacian_variance"],
                "max_person_iou": q["max_person_iou"],
                "edge_clipping": q["edge_clipping_fraction"],
                "frozen_source_decision": src.get("frozen_source_decision")
                or "same_team_distractor_yes",
                "human_visible_jersey_number": src.get("human_visible_jersey_number")
                or "",
                "manual_identity_continuity_observed": "",
                "manual_view_category": "",
                "manual_scale_category": "",
                "manual_quality_notes": "",
                "hard_negative_gallery_member": False,
                "embedding_input": False,
                "automatic_enrollment": False,
                "reviewer": human["reviewer"],
                "final_approver": human["final_approver"],
                "reviewed_at": reviewed_at,
                "automated_ocr_used": False,
                "similarity_used": False,
                "model_identity_used": False,
                "decisions_human_approved": True,
                **dec,
            }
        )

    # Exact distribution checks.
    if len(target_rows) != 4:
        raise CropFreezeError("target reviewed count mismatch")
    if any(r["manual_target_crop_decision"] != "target_crop_yes" for r in target_rows):
        raise CropFreezeError("all target crops must be yes")
    hn_dist = Counter(r["manual_hard_negative_crop_decision"] for r in hn_rows)
    expected_hn_dist = {
        "hard_negative_crop_yes": 23,
        "hard_negative_crop_no": 4,
        "invalid": 5,
        "multi_person_ambiguous": 3,
    }
    if dict(hn_dist) != expected_hn_dist:
        raise CropFreezeError(f"HN distribution mismatch {dict(hn_dist)}")

    approved_hn_ids = [
        r["hard_negative_candidate_id"]
        for r in hn_rows
        if r["manual_hard_negative_crop_decision"] == "hard_negative_crop_yes"
    ]
    expected_yes_ids = [hn_id(i) for i in HN_YES]
    if approved_hn_ids != expected_yes_ids:
        raise CropFreezeError("approved HN ID set mismatch")

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
            freeze / "target_001_EXT_161_target_crop_decisions_frozen.csv",
            TARGET_CSV_FIELDS,
            target_rows,
        )
        write_csv(
            freeze / "target_001_external_hard_negative_crop_decisions_frozen.csv",
            HN_CSV_FIELDS,
            hn_rows,
        )
        write_csv(
            freeze / "target_001_EXT_161_approved_target_crops_frozen.csv",
            TARGET_CSV_FIELDS,
            target_rows,
        )
        write_csv(
            freeze / "target_001_external_approved_hard_negative_crops_frozen.csv",
            HN_CSV_FIELDS,
            [r for r in hn_rows if r["is_frozen_approved_hard_negative_crop"]],
        )

        freeze_payload = {
            "schema_version": "reid_target_external_refinement_crop_manual_freeze_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "f3e_path": config["stage5d_f3e_package"]["path"],
            "f3e_snapshot_sha256": f3e["snapshot_sha256"],
            "external_source_sha256": config["external_enrollment_source"][
                "expected_sha256"
            ],
            "reviewed_target_crop_candidates": 4,
            "approved_new_target_crops": 4,
            "approved_target_crop_ids": list(TARGET_APPROVED),
            "reviewed_hard_negative_crop_candidates": 35,
            "approved_hard_negative_crops": 23,
            "hard_negative_crop_no_wrong_team": 4,
            "invalid": 5,
            "multi_person_ambiguous": 3,
            "approved_hard_negative_crop_ids": expected_yes_ids,
            "wrong_team_rejected_ids": [hn_id(i) for i in HN_NO_WRONG_TEAM],
            "invalid_ids": [hn_id(i) for i in HN_INVALID],
            "multi_person_ambiguous_ids": [hn_id(i) for i in HN_AMBIGUOUS],
            "hard_negative_decision_distribution": expected_hn_dist,
            "official_gallery_v1_members": 7,
            "hard_negative_gallery_members": 0,
            "target_crops_current_gallery_members": 0,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "hard_negative_gallery_built": False,
            "sample_access_read": False,
            "similarity_used": False,
            "ocr_used": False,
            "model_identity_used": False,
            "decisions_frozen": True,
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "reviewed_at": reviewed_at,
            "project_head": head,
            "generated_at": generated_at,
        }
        write_json(
            freeze / "target_001_external_refinement_crop_manual_freeze.json",
            freeze_payload,
        )
        write_json(
            freeze / "target_001_external_refinement_crop_manual_freeze_contract.json",
            {
                **build_contract(),
                "generated_at": generated_at,
                "project_head": head,
                "f3e_snapshot_sha256": f3e["snapshot_sha256"],
            },
        )
        write_json(
            validation / "target_001_external_refinement_crop_manual_freeze_validation.json",
            {
                "schema_version": "reid_target_001_external_refinement_crop_manual_freeze_validation_v1",
                "target_inventory_equals_decision_set": True,
                "hn_inventory_equals_decision_set": True,
                "missing": 0,
                "extra": 0,
                "duplicate": 0,
                "unknown": 0,
                "reviewed_target": 4,
                "reviewed_hard_negative": 35,
                "target_all_yes": True,
                "hn_distribution": expected_hn_dist,
                "gallery_members": 7,
                "hard_negative_gallery_members": 0,
                "sample_reads": 0,
                "new_embeddings": 0,
                "similarity_rows": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "gallery_mutation": False,
            },
        )
        write_json(runtime / "stage5d_f3f_runtime_audit.json", runtime_audit)

        file_count, files_sha = listing_sha(tmp)
        write_json(
            freeze / "target_001_external_refinement_crop_manual_freeze_manifest.json",
            {
                "schema_version": "reid_target_001_external_refinement_crop_manual_freeze_manifest_v1",
                "csv_files": [
                    "target_001_EXT_161_target_crop_decisions_frozen.csv",
                    "target_001_external_hard_negative_crop_decisions_frozen.csv",
                    "target_001_EXT_161_approved_target_crops_frozen.csv",
                    "target_001_external_approved_hard_negative_crops_frozen.csv",
                ],
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
            },
        )
        write_json(
            tmp / "stage5d_f3f_contract.json",
            {
                **build_contract(),
                "final_status": FINAL_STATUS,
                "generated_at": generated_at,
                "project_head": head,
            },
        )
        summary = {
            "schema_version": "reid_stage5d_f3f_external_refinement_crop_manual_freeze_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reviewed_target_crop_candidates": 4,
            "approved_new_target_crops": 4,
            "approved_target_crop_ids": list(TARGET_APPROVED),
            "reviewed_hard_negative_crop_candidates": 35,
            "approved_hard_negative_crops": 23,
            "hard_negative_crop_no_wrong_team": 4,
            "invalid": 5,
            "multi_person_ambiguous": 3,
            "approved_hard_negative_crop_ids": expected_yes_ids,
            "wrong_team_rejected_ids": [hn_id(i) for i in HN_NO_WRONG_TEAM],
            "invalid_ids": [hn_id(i) for i in HN_INVALID],
            "multi_person_ambiguous_ids": [hn_id(i) for i in HN_AMBIGUOUS],
            "official_gallery_v1_members": 7,
            "hard_negative_gallery_members": 0,
            "target_crops_current_gallery_members": 0,
            "new_embeddings": 0,
            "similarity_rows": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "sample_score_row_read_count": 0,
            "sample_rank_row_read_count": 0,
            "sample_used_for_decision": False,
            "f3e_snapshot_sha256": f3e["snapshot_sha256"],
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3f_summary.json", summary)
        write_json(
            tmp / "stage5d_f3f_manifest.json",
            {
                "schema_version": "reid_stage5d_f3f_external_refinement_crop_manual_freeze_manifest_v1",
                "final_status": FINAL_STATUS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "new_crop_copies": 0,
                    "new_embeddings": 0,
                    "similarity_rows": 0,
                    "hard_negative_gallery_members": 0,
                    "gallery_members": 7,
                    "threshold": False,
                    "identity_assignments": 0,
                },
                "generated_at": generated_at,
            },
        )

        if list(tmp.rglob("*.npy")) or list(tmp.rglob("*.npz")):
            raise CropFreezeError("embedding artifacts forbidden")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise CropFreezeError("media artifacts forbidden in freeze")
        if list(tmp.rglob("sample.mp4")):
            raise CropFreezeError("sample leakage")

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_json(final_dir / "stage5d_f3f_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_refinement_crop_manual_freeze_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "exact_next_gate": summary["exact_next_gate"],
                "approved_new_target_crops": summary["approved_new_target_crops"],
                "approved_hard_negative_crops": summary["approved_hard_negative_crops"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
