#!/usr/bin/env python3
"""Stage 5D-F3G — Target gallery-v2 + same-team distractor gallery-v1.

Reuses gallery-v1 (7) immutably. Embeds only 6 approved new target crops and
23 approved hard-negative crops with canonical OSNet (two-pass). Diagnostic
centroid/medoid/pairwise/cross matrices only. No sample evaluation, threshold,
identity assignment, or gallery-v1 mutation.
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
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid import embedding as emb  # noqa: E402

CONFIG_SCHEMA = "reid_target_gallery_v2_and_distractor_gallery_config_v1"
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F3G_TARGET_001_GALLERY_V2_AND_DISTRACTOR_GALLERY_BUILT"
)
READINESS = (
    "TARGET_001_GALLERY_V2_AND_DISTRACTOR_GALLERY_READY_FOR_SCORING_DESIGN"
)
NEXT_GATE = (
    "STAGE5D-F3H_TARGET_001_TARGET_DISTRACTOR_SCORING_CONTRACT_AND_NEW_HOLDOUT_DESIGN"
)

ALLOWED_DIRTY = {
    "scripts/run_reid_target_gallery_v2_and_distractor_gallery_build.py",
    "configs/reid/target_gallery_v2_and_distractor_gallery_stage5d_target_001.yaml",
    "tests/test_reid_target_gallery_v2_and_distractor_gallery_build.py",
    "docs/setup/stage5d-target-approved-crop-osnet-embedding-and-gallery-v2-build.md",
}

GALLERY_V1_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)
EXPANSION_IDS = (
    "target_001_ext_view_candidate_001",
    "target_001_ext_view_candidate_002",
)
EXT161_IDS = (
    "target_001_ext_refine_target_candidate_001",
    "target_001_ext_refine_target_candidate_002",
    "target_001_ext_refine_target_candidate_003",
    "target_001_ext_refine_target_candidate_004",
)
NEW_TARGET_IDS = EXPANSION_IDS + EXT161_IDS
GALLERY_V2_IDS = GALLERY_V1_IDS + NEW_TARGET_IDS

HN_YES_SEQ = (
    2,
    5,
    6,
    7,
    8,
    11,
    12,
    14,
    15,
    18,
    19,
    20,
    21,
    24,
    25,
    26,
    27,
    29,
    30,
    31,
    33,
    34,
    35,
)
HN_YES_IDS = tuple(f"target_001_ext_hard_negative_candidate_{i:03d}" for i in HN_YES_SEQ)
HN_EXCLUDED_SEQ = (1, 3, 4, 9, 10, 13, 16, 17, 22, 23, 28, 32)
HN_EXCLUDED_IDS = tuple(
    f"target_001_ext_hard_negative_candidate_{i:03d}" for i in HN_EXCLUDED_SEQ
)

EXPECTED_NPY = 9


class GalleryV2Error(RuntimeError):
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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GalleryV2Error("unexpected config schema")
    if not config.get("offline_required"):
        raise GalleryV2Error("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GalleryV2Error(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GalleryV2Error(
                    "BLOCKED_STAGE5D_F3G_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str, block_code: str) -> str:
    if not snapshot_path.is_file():
        raise GalleryV2Error(f"{block_code} snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise GalleryV2Error(f"{block_code} snapshot_sidecar")
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(
        man_payload.get("sha256")
        or man_payload.get("archive_sha256")
        or ""
    )
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GalleryV2Error(f"{block_code} snapshot_sha")
    return actual


def resolve_crop(
    package_root: Path, rel: str, expected_sha: str, crop_read_log: list[str]
) -> dict[str, Any]:
    assert_no_path_traversal(rel)
    path = (package_root / rel).resolve()
    try:
        path.relative_to(package_root.resolve())
    except ValueError as exc:
        raise GalleryV2Error(f"crop escaped package: {rel}") from exc
    if path.is_symlink() or not path.is_file():
        raise GalleryV2Error(f"crop missing or symlink: {rel}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise GalleryV2Error(f"crop sha mismatch: {rel}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise GalleryV2Error(f"crop decode failed: {rel}")
    h, w = int(image.shape[0]), int(image.shape[1])
    crop_read_log.append(str(path))
    return {
        "crop_path_absolute": str(path),
        "crop_path": rel,
        "crop_sha256": actual,
        "height": h,
        "width": w,
    }


def validate_immutable_assets(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise GalleryV2Error("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise GalleryV2Error("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise GalleryV2Error("external sha mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise GalleryV2Error("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise GalleryV2Error("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise GalleryV2Error("yolo sha mismatch")
    ck = emb.verify_checkpoint(
        osnet, expected_sha256=config["osnet_checkpoint"]["expected_sha256"]
    )
    return {
        "external_sha256": config["external_enrollment_source"]["expected_sha256"],
        "sample_sha256": config["evaluation_source"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet": ck,
    }


def validate_f3f(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["stage5d_f3f_package"]
    root = project_root / cfg["path"]
    summary = load_json(root / "stage5d_f3f_summary.json")
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH status")
    checks = {
        "approved_new_target_crops": cfg["expected_approved_target_crops"],
        "approved_hard_negative_crops": cfg["expected_approved_hard_negative_crops"],
        "hard_negative_crop_no_wrong_team": cfg["expected_hard_negative_crop_no"],
        "invalid": cfg["expected_invalid"],
        "multi_person_ambiguous": cfg["expected_multi_person_ambiguous"],
        "official_gallery_v1_members": cfg["expected_gallery_members"],
        "hard_negative_gallery_members": cfg["expected_hard_negative_gallery_members"],
        "new_embeddings": 0,
        "similarity_rows": 0,
        "identity_assignments": 0,
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GalleryV2Error(
                f"BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("gallery_mutation") is not False:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH gallery_mutation"
        )
    if summary.get("sample_video_read") is not False:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH sample"
        )
    if list(summary.get("approved_target_crop_ids") or []) != list(EXT161_IDS):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH target_ids"
        )
    if list(summary.get("approved_hard_negative_crop_ids") or []) != list(HN_YES_IDS):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH hn_ids"
        )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]),
        cfg["expected_snapshot_sha256"],
        "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH",
    )
    with (
        root / "manual_freeze" / "target_001_EXT_161_approved_target_crops_frozen.csv"
    ).open(encoding="utf-8") as handle:
        target_rows = list(csv.DictReader(handle))
    with (
        root
        / "manual_freeze"
        / "target_001_external_approved_hard_negative_crops_frozen.csv"
    ).open(encoding="utf-8") as handle:
        hn_rows = list(csv.DictReader(handle))
    if len(target_rows) != 4 or len(hn_rows) != 23:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_CROP_FREEZE_CONTRACT_MISMATCH csv_counts"
        )
    return {
        "root": root,
        "summary": summary,
        "snapshot_sha256": snap,
        "target_rows": target_rows,
        "hn_rows": hn_rows,
    }


def validate_gallery_v1(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    cfg = config["gallery_v1"]
    root = project_root / cfg["path"]
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    if summary.get("final_status") != cfg["expected_final_status"]:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH status")
    if summary.get("readiness") != cfg["expected_readiness"]:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH readiness"
        )
    if int(summary["individual_gallery_members"]) != 7:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH members"
        )
    if summary.get("medoid_anchor_candidate_id") != cfg["expected_medoid_id"]:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH medoid"
        )
    if list(summary.get("approved_exact_ids") or []) != list(GALLERY_V1_IDS):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH ids"
        )
    emb_path = root / "embeddings" / "target_001_anchor_embeddings.npy"
    gal_path = root / "gallery" / "target_001_individual_gallery.npy"
    vectors = np.load(emb_path)
    gallery = np.load(gal_path)
    if vectors.shape != (7, 512) or vectors.dtype != np.float32:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH shape"
        )
    if not np.array_equal(vectors, gallery):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH emb_ne_gallery"
        )
    if not np.isfinite(vectors).all():
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH finite"
        )
    if int(np.all(vectors == 0, axis=1).sum()) != 0:
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH zero"
        )
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-4):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH l2"
        )
    sha_payload = load_json(
        root / "embeddings" / "target_001_anchor_embeddings_sha256.json"
    )
    actual_sha = sha256_bytes(emb_path.read_bytes())
    if not (
        actual_sha
        == sha_payload["sha256"]
        == cfg["expected_embedding_sha256"]
        == sha256_bytes(gal_path.read_bytes())
    ):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH emb_sha"
        )
    if list(sha_payload.get("row_order") or []) != list(GALLERY_V1_IDS):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH row_order"
        )
    members = [
        json.loads(line)
        for line in (
            root / "gallery" / "target_001_gallery_members.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    member_ids = [m["anchor_candidate_id"] for m in members]
    if member_ids != list(GALLERY_V1_IDS):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH member_order"
        )
    det = load_json(root / "runtime" / "determinism_audit.json")
    if float(det.get("overall_max_absolute_difference", 1.0)) != 0.0:
        # Accept exact recorded max; B1E-F stored 0.0 for exact match.
        if not det.get("exact_match", False):
            raise GalleryV2Error(
                "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH determinism"
            )
    snap = resolve_snapshot_sha(
        Path(cfg["snapshot_path"]),
        cfg["expected_snapshot_sha256"],
        "BLOCKED_STAGE5D_F3G_GALLERY_V1_CONTRACT_MISMATCH",
    )
    return {
        "root": root,
        "summary": summary,
        "vectors": vectors.astype(np.float32, copy=True),
        "members": members,
        "embedding_sha256": actual_sha,
        "snapshot_sha256": snap,
    }


def validate_expansion(
    project_root: Path,
    config: Mapping[str, Any],
    crop_read_log: list[str],
) -> list[dict[str, Any]]:
    f3d_root = project_root / config["stage5d_f3d_package"]["path"]
    f3c_root = project_root / config["stage5d_f3c_package"]["path"]
    f3d_summary = load_json(f3d_root / "stage5d_f3d_summary.json")
    if (
        f3d_summary.get("final_status")
        != config["stage5d_f3d_package"]["expected_final_status"]
    ):
        raise GalleryV2Error("F3D status mismatch")
    f3c_summary = load_json(f3c_root / "stage5d_f3c_summary.json")
    if (
        f3c_summary.get("final_status")
        != config["stage5d_f3c_package"]["expected_final_status"]
    ):
        raise GalleryV2Error("F3C status mismatch")
    with (
        f3d_root
        / "manual_freeze"
        / "target_001_external_target_anchor_expansion_sources_frozen.csv"
    ).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise GalleryV2Error("expansion count mismatch")
    out: list[dict[str, Any]] = []
    expected_views = {
        "target_001_ext_view_candidate_001": "left_side",
        "target_001_ext_view_candidate_002": "front_oblique",
    }
    for row, expected_id in zip(rows, EXPANSION_IDS, strict=True):
        cid = row["target_view_candidate_id"]
        if cid != expected_id:
            raise GalleryV2Error(f"expansion id order mismatch {cid}")
        if row["manual_anchor_expansion_decision"] != "target_anchor_expansion_yes":
            raise GalleryV2Error(f"expansion not approved {cid}")
        for key in (
            "manual_crop_valid",
            "manual_target_dominant",
            "manual_single_person",
            "manual_identity_confirmed",
        ):
            if row[key] != "yes":
                raise GalleryV2Error(f"expansion field {key} for {cid}")
        if row["current_gallery_member"] != "false":
            raise GalleryV2Error(f"expansion already gallery member {cid}")
        if row["source_occurrence_code"] != "EXT_004":
            raise GalleryV2Error(f"expansion source occ {cid}")
        if row["manual_view_category"] != expected_views[cid]:
            raise GalleryV2Error(f"expansion view mismatch {cid}")
        crop = resolve_crop(
            f3c_root, row["crop_path"], row["crop_sha256"], crop_read_log
        )
        out.append(
            {
                "member_id": cid,
                "source_kind": "approved_expansion",
                "source_occurrence_code": row["source_occurrence_code"],
                "raw_track_id": int(row["raw_track_id"]),
                "frame_index": int(row["frame_index"]),
                "video_time": float(row["video_time"]),
                "source_bbox": row["source_bbox"],
                "crop_bbox": row["crop_bbox"],
                "view_category": row["manual_view_category"],
                "human_visible_jersey_number": "",
                **crop,
            }
        )
    return out


def validate_new_target_crops(
    project_root: Path,
    config: Mapping[str, Any],
    f3f: Mapping[str, Any],
    crop_read_log: list[str],
) -> list[dict[str, Any]]:
    f3e_root = project_root / config["stage5d_f3e_package"]["path"]
    f3e_summary = load_json(f3e_root / "stage5d_f3e_summary.json")
    if (
        f3e_summary.get("final_status")
        != config["stage5d_f3e_package"]["expected_final_status"]
    ):
        raise GalleryV2Error("F3E status mismatch")
    out: list[dict[str, Any]] = []
    for row, expected_id in zip(f3f["target_rows"], EXT161_IDS, strict=True):
        cid = row["target_crop_candidate_id"]
        if cid != expected_id:
            raise GalleryV2Error(f"EXT161 id order mismatch {cid}")
        if row["manual_target_crop_decision"] != "target_crop_yes":
            raise GalleryV2Error(f"EXT161 not approved {cid}")
        for key in (
            "manual_crop_valid",
            "manual_target_dominant",
            "manual_single_person",
            "manual_identity_confirmed",
        ):
            if row[key] != "yes":
                raise GalleryV2Error(f"EXT161 field {key} for {cid}")
        if row["current_gallery_member"] != "false":
            raise GalleryV2Error(f"EXT161 already gallery member {cid}")
        if row["embedding_input"] != "false":
            raise GalleryV2Error(f"EXT161 embedding_input at freeze {cid}")
        crop = resolve_crop(
            f3e_root, row["crop_path"], row["crop_sha256"], crop_read_log
        )
        out.append(
            {
                "member_id": cid,
                "source_kind": "ext_161_approved_target_crop",
                "source_occurrence_code": row["source_occurrence_code"],
                "raw_track_id": int(row["raw_track_id"]),
                "frame_index": int(row["frame_index"]),
                "video_time": float(row["video_time"]),
                "source_bbox": row["source_bbox"],
                "crop_bbox": row["crop_bbox"],
                "view_category": row.get("manual_view_category") or "",
                "human_visible_jersey_number": "",
                **crop,
            }
        )
    return out


def validate_hn_crops(
    project_root: Path,
    config: Mapping[str, Any],
    f3f: Mapping[str, Any],
    crop_read_log: list[str],
) -> list[dict[str, Any]]:
    f3e_root = project_root / config["stage5d_f3e_package"]["path"]
    out: list[dict[str, Any]] = []
    for row, expected_id in zip(f3f["hn_rows"], HN_YES_IDS, strict=True):
        cid = row["hard_negative_candidate_id"]
        if cid != expected_id:
            raise GalleryV2Error(f"HN id order mismatch {cid}")
        if row["manual_hard_negative_crop_decision"] != "hard_negative_crop_yes":
            raise GalleryV2Error(f"HN not yes {cid}")
        if row["manual_crop_valid"] != "yes":
            raise GalleryV2Error(f"HN crop_valid {cid}")
        if row["manual_target_absent"] != "yes":
            raise GalleryV2Error(f"HN target_absent {cid}")
        if row["manual_same_team_confirmed"] != "yes":
            raise GalleryV2Error(f"HN same_team {cid}")
        if row["manual_single_person"] != "yes":
            raise GalleryV2Error(f"HN single_person {cid}")
        if row["hard_negative_gallery_member"] != "false":
            raise GalleryV2Error(f"HN already member {cid}")
        if row["embedding_input"] != "false":
            raise GalleryV2Error(f"HN embedding_input at freeze {cid}")
        crop = resolve_crop(
            f3e_root, row["crop_path"], row["crop_sha256"], crop_read_log
        )
        jersey = (row.get("human_visible_jersey_number") or "").strip()
        out.append(
            {
                "member_id": cid,
                "source_kind": "approved_hard_negative_crop",
                "source_external_code": row["source_external_code"],
                "raw_track_id": int(row["raw_track_id"]),
                "selected_frame": int(row["selected_frame"]),
                "video_time": float(row["video_time"]),
                "source_bbox": row["source_bbox"],
                "crop_bbox": row["crop_bbox"],
                "human_visible_jersey_number": jersey,
                "human_jersey_provenance": "human_f3d_f3f"
                if jersey
                else "unknown_unfilled",
                **crop,
            }
        )
    # Ensure excluded crops were never opened.
    for cid in HN_EXCLUDED_IDS:
        for path in crop_read_log:
            if cid in path:
                raise GalleryV2Error(f"excluded crop read: {cid}")
    return out


def resolve_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    rt = config["canonical_reid_runtime"]
    verified = emb.verify_sn_reid_root(
        rt["sn_reid_root"], expected_commit=rt["expected_sn_reid_commit"]
    )
    return {
        "sn_reid_root": str(verified["root"]),
        "sn_reid_commit": verified["commit"],
        "batch_size": int(rt["batch_size"]),
        "device": "cpu",
        "conda_env": rt["conda_env"],
        "embedding_determinism_atol": float(rt["embedding_determinism_atol"]),
        "unit_norm_atol": float(rt["unit_norm_atol"]),
    }


def build_preprocessing_contract(
    *,
    config: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime: Mapping[str, Any],
    f3f_snap: str,
    gallery_v1_snap: str,
    gallery_v1_emb_sha: str,
) -> dict[str, Any]:
    contract = {
        "schema_version": "reid_target_001_gallery_v2_preprocessing_contract_v1",
        "predictions_embeddings_not_inspected_before_contract_freeze": True,
        "existing_reused_vectors": 7,
        "new_target_crop_count": 6,
        "new_hard_negative_crop_count": 23,
        "total_new_inference_crops": 29,
        "expected_dimension": 512,
        "model_name": emb.MODEL_NAME,
        "embedding_dimension": emb.EMBEDDING_DIM,
        "rgb_conversion_policy": emb.PREPROCESSING["color_conversion"],
        "resize_height": emb.RESIZE_HEIGHT,
        "resize_width": emb.RESIZE_WIDTH,
        "interpolation": "PIL_Image_resize_default_bilinear_via_torchvision_Resize",
        "normalization_mean": list(emb.IMAGENET_MEAN),
        "normalization_std": list(emb.IMAGENET_STD),
        "tensor_layout": "NCHW_float32_cpu",
        "dtype": "float32",
        "batch_size": int(runtime["batch_size"]),
        "eval_mode": True,
        "gradients_disabled": True,
        "output_tensor_selection": "model_forward_full_embedding",
        "original_embedding_normalization_policy": "l2_normalized_via_embed_tensors",
        "checkpoint_path": config["osnet_checkpoint"]["path"],
        "checkpoint_sha256": assets["osnet"]["sha256"],
        "sn_reid_root": runtime["sn_reid_root"],
        "sn_reid_commit": runtime["sn_reid_commit"],
        "f3f_snapshot_sha256": f3f_snap,
        "gallery_v1_snapshot_sha256": gallery_v1_snap,
        "gallery_v1_embedding_sha256": gallery_v1_emb_sha,
        "gallery_v1_reused_immutable": True,
        "embedding_determinism_atol": float(runtime["embedding_determinism_atol"]),
        "unit_norm_atol": float(runtime["unit_norm_atol"]),
        "contract_source_paths": {
            "embedding_module": "src/football_analytics/reid/embedding.py",
            "b1e_f_preprocessing_reused": True,
        },
    }
    if contract["resize_height"] != 256 or contract["resize_width"] != 128:
        raise GalleryV2Error("preprocessing resize mismatch")
    if contract["expected_dimension"] != 512:
        raise GalleryV2Error("preprocessing dim mismatch")
    if contract["model_name"] != "osnet_x1_0":
        raise GalleryV2Error("preprocessing model mismatch")
    return contract


def _load_model(checkpoint: Path, sn_reid_root: Path):
    with emb.temporary_sys_path_prepend(sn_reid_root):
        model = emb.build_osnet_cpu_model(model_name=emb.MODEL_NAME)
        emb.load_osnet_checkpoint_weights(model, checkpoint)
        model.eval()
        model.cpu()
        return model


def embed_new_crops_two_pass(
    *,
    crops: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    if len(crops) != 29:
        raise GalleryV2Error(f"expected 29 new crops, got {len(crops)}")
    tensors = [
        emb.load_and_preprocess_crop(Path(row["crop_path_absolute"])) for row in crops
    ]
    atol = float(runtime["embedding_determinism_atol"])
    sn_root = Path(runtime["sn_reid_root"])
    batch_size = int(runtime["batch_size"])

    model1 = _load_model(checkpoint, sn_root)
    with torch.inference_mode():
        pass1 = emb.embed_tensors(model1, tensors, batch_size=batch_size)
    del model1

    model2 = _load_model(checkpoint, sn_root)
    with torch.inference_mode():
        pass2 = emb.embed_tensors(model2, tensors, batch_size=batch_size)
    del model2

    if pass1.shape != (29, 512) or pass2.shape != (29, 512):
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_EMBEDDING_NONDETERMINISM shape")
    abs_diff = np.abs(pass1.astype(np.float64) - pass2.astype(np.float64))
    overall_max = float(abs_diff.max())
    exact = bool(np.array_equal(pass1, pass2))
    within = bool(overall_max <= atol)
    if not (exact or within):
        raise GalleryV2Error(
            "BLOCKED_STAGE5D_F3G_EMBEDDING_NONDETERMINISM "
            f"max_abs={overall_max} atol={atol}"
        )
    vectors = pass1.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    zero_count = int(np.all(vectors == 0, axis=1).sum())
    if nan_count or inf_count or zero_count:
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_EMBEDDING_NONDETERMINISM vectors")
    unit_atol = float(runtime["unit_norm_atol"])
    if np.any(np.abs(norms - 1.0) > unit_atol):
        raise GalleryV2Error("BLOCKED_STAGE5D_F3G_EMBEDDING_NONDETERMINISM norms")
    return {
        "embeddings": vectors,
        "norms": norms,
        "determinism": {
            "exact_match": exact,
            "within_tolerance": within,
            "atol_audit": within,
            "comparison_policy": "exact_or_max_abs_le_atol",
            "atol": atol,
            "per_row_max_absolute_difference": [
                float(x) for x in abs_diff.max(axis=1)
            ],
            "overall_max_absolute_difference": overall_max,
            "passes": 2,
            "ordering_match": True,
            "deterministic": True,
            "input_ids": [c["member_id"] for c in crops],
        },
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_count,
        "dtype": "float32",
        "shape": [29, 512],
    }


def compute_centroid(gallery: np.ndarray) -> np.ndarray:
    mean = gallery.mean(axis=0).astype(np.float64)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0:
        raise GalleryV2Error("centroid invalid")
    return (mean / norm).astype(np.float32)


def compute_medoid(gallery: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    sims = gallery @ gallery.T
    n = gallery.shape[0]
    mean_dist = []
    for i in range(n):
        others = [1.0 - float(sims[i, j]) for j in range(n) if j != i]
        mean_dist.append(float(np.mean(others)))
    best = min(range(n), key=lambda i: (mean_dist[i], ids[i]))
    return {
        "index": best,
        "member_id": ids[best],
        "mean_cosine_distance_to_others": mean_dist[best],
        "vector": gallery[best].astype(np.float32, copy=True),
        "all_mean_distances": mean_dist,
        "tie_break": "member_id_ascending",
    }


def internal_diagnostics(
    gallery: np.ndarray,
    ids: Sequence[str],
    centroid: np.ndarray,
    medoid: Mapping[str, Any],
    outlier_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    sims = (gallery @ gallery.T).astype(np.float64)
    n = sims.shape[0]
    off = [float(sims[i, j]) for i in range(n) for j in range(n) if i != j]
    off_arr = np.asarray(off, dtype=np.float64)
    per_mean = []
    per_median = []
    for i in range(n):
        others = [float(sims[i, j]) for j in range(n) if j != i]
        per_mean.append(float(np.mean(others)))
        per_median.append(float(np.median(others)))
    median_mean = float(np.median(per_mean))
    mad = float(np.median([abs(x - median_mean) for x in per_mean]))
    mult = float(outlier_cfg.get("mad_multiplier", 3.0))
    threshold = median_mean - mult * mad if mad > 0 else None
    flagged = []
    for i, mean_sim in enumerate(per_mean):
        if threshold is not None and mean_sim < threshold:
            flagged.append(
                {
                    "member_id": ids[i],
                    "mean_similarity_to_others": mean_sim,
                    "flags": ["mean_similarity_below_median_mad"],
                    "removal_allowed": False,
                    "descriptive_only": True,
                }
            )
    near_dup = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sims[i, j])
            if val >= 0.999:
                near_dup.append(
                    {"a": ids[i], "b": ids[j], "cosine_similarity": val}
                )
    return {
        "pairwise": sims.astype(np.float32),
        "metrics": {
            "shape": [n, n],
            "diagonal_mean": float(np.mean(np.diag(sims))),
            "off_diagonal_min": float(np.min(off_arr)),
            "off_diagonal_median": float(np.median(off_arr)),
            "off_diagonal_mean": float(np.mean(off_arr)),
            "off_diagonal_max": float(np.max(off_arr)),
            "per_member_mean_similarity_to_others": {
                ids[i]: per_mean[i] for i in range(n)
            },
            "per_member_median_similarity_to_others": {
                ids[i]: per_median[i] for i in range(n)
            },
            "centroid_similarity_per_member": {
                ids[i]: float(np.dot(gallery[i], centroid)) for i in range(n)
            },
            "medoid_similarity_per_member": {
                ids[i]: float(np.dot(gallery[i], medoid["vector"])) for i in range(n)
            },
            "near_duplicate_pairs": near_dup,
            "outlier_diagnostics": {
                "method": "median_mad",
                "median_mean_similarity": median_mean,
                "mad": mad,
                "mad_multiplier": mult,
                "threshold_mean_similarity": threshold,
                "removal_allowed": False,
                "descriptive_only": True,
                "flagged": flagged,
            },
            "automatic_member_removal": False,
            "gallery_mutation_after_diagnostics": False,
            "threshold_selected": False,
            "identity_threshold": None,
        },
    }


def cross_diagnostics(
    target: np.ndarray,
    target_ids: Sequence[str],
    distractor: np.ndarray,
    distractor_ids: Sequence[str],
    distractor_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cross = (target @ distractor.T).astype(np.float64)
    flat = []
    for i, tid in enumerate(target_ids):
        for j, did in enumerate(distractor_ids):
            flat.append((float(cross[i, j]), tid, did, j))
    flat.sort(key=lambda x: (-x[0], x[1], x[2]))
    top20 = []
    for score, tid, did, j in flat[:20]:
        meta = distractor_meta[j]
        top20.append(
            {
                "target_member_id": tid,
                "distractor_member_id": did,
                "cosine_similarity": score,
                "distractor_source_external_code": meta["source_external_code"],
                "human_visible_jersey_number": meta["human_visible_jersey_number"],
                "human_jersey_provenance": meta["human_jersey_provenance"],
                "descriptive_internal_enrollment_diagnostic_only": True,
                "same_source_cross_similarity_not_independent": True,
            }
        )
    v1_ids = set(GALLERY_V1_IDS)
    v1_rows = [i for i, tid in enumerate(target_ids) if tid in v1_ids]
    v2_rows = [i for i, tid in enumerate(target_ids) if tid not in v1_ids]

    def _block_stats(rows: Sequence[int]) -> dict[str, float]:
        block = cross[list(rows), :]
        return {
            "min": float(block.min()),
            "median": float(np.median(block)),
            "mean": float(block.mean()),
            "max": float(block.max()),
        }

    return {
        "matrix": cross.astype(np.float32),
        "summary": {
            "schema_version": "reid_target_001_target_distractor_cross_similarity_v1",
            "shape": [13, 23],
            "global_min": float(cross.min()),
            "global_median": float(np.median(cross)),
            "global_mean": float(cross.mean()),
            "global_max": float(cross.max()),
            "per_target_max_distractor_similarity": {
                target_ids[i]: {
                    "max_similarity": float(cross[i].max()),
                    "distractor_member_id": distractor_ids[int(cross[i].argmax())],
                }
                for i in range(len(target_ids))
            },
            "per_distractor_max_target_similarity": {
                distractor_ids[j]: {
                    "max_similarity": float(cross[:, j].max()),
                    "target_member_id": target_ids[int(cross[:, j].argmax())],
                }
                for j in range(len(distractor_ids))
            },
            "old_v1_target_members_vs_distractors": _block_stats(v1_rows),
            "new_v2_target_members_vs_distractors": _block_stats(v2_rows),
            "descriptive_internal_enrollment_diagnostic_only": True,
            "same_source_cross_similarity_not_independent": True,
            "threshold": None,
            "threshold_selected": False,
            "identity_assignment": False,
            "score_formula_applied": False,
            "max_target_minus_max_distractor_forbidden_in_this_gate": True,
        },
        "top_pairs": top20,
    }


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f3g_gallery_v2_{final_dir.name}_{token}"
    if tmp.exists():
        raise GalleryV2Error("tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GalleryV2Error("final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    assert_no_path_traversal(config["output"]["final_dir"])
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise GalleryV2Error("final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    if config["evaluation_source"].get("decode_forbidden") is not True:
        raise GalleryV2Error("sample decode_forbidden required")
    if config["gallery_policy"].get("gallery_v1_mutation") is not False:
        raise GalleryV2Error("gallery_v1_mutation must be false")

    crop_read_log: list[str] = []
    assets = validate_immutable_assets(project_root, config)
    f3f = validate_f3f(project_root, config)
    gallery_v1 = validate_gallery_v1(project_root, config)
    expansion = validate_expansion(project_root, config, crop_read_log)
    ext161 = validate_new_target_crops(project_root, config, f3f, crop_read_log)
    hn_crops = validate_hn_crops(project_root, config, f3f, crop_read_log)
    new_target = expansion + ext161
    if [c["member_id"] for c in new_target] != list(NEW_TARGET_IDS):
        raise GalleryV2Error("new target ordering mismatch")
    if [c["member_id"] for c in hn_crops] != list(HN_YES_IDS):
        raise GalleryV2Error("hn ordering mismatch")
    inference_crops = new_target + hn_crops
    if len(inference_crops) != 29:
        raise GalleryV2Error("inference crop count")

    for cid in HN_EXCLUDED_IDS:
        for path in crop_read_log:
            if cid in Path(path).name:
                raise GalleryV2Error(f"excluded crop source read: {cid}")

    runtime = resolve_runtime(config)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = create_temp_root(final_dir)
    try:
        tg = tmp / "target_gallery_v2"
        dg = tmp / "distractor_gallery_v1"
        xd = tmp / "cross_diagnostics"
        rt = tmp / "runtime"
        cfg_dir = tmp / "effective_configs"
        for d in (tg, dg, xd, rt, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, cfg_dir / config_path.name)

        pre_contract = build_preprocessing_contract(
            config=config,
            assets=assets,
            runtime=runtime,
            f3f_snap=f3f["snapshot_sha256"],
            gallery_v1_snap=gallery_v1["snapshot_sha256"],
            gallery_v1_emb_sha=gallery_v1["embedding_sha256"],
        )
        write_json(
            rt / "target_001_gallery_v2_preprocessing_contract_pre_inference.json",
            pre_contract,
        )

        embed_result = embed_new_crops_two_pass(
            crops=inference_crops,
            checkpoint=Path(config["osnet_checkpoint"]["path"]),
            runtime=runtime,
        )
        write_json(
            rt / "target_001_gallery_v2_embedding_determinism.json",
            embed_result["determinism"],
        )

        new_vecs = embed_result["embeddings"]
        target_new = new_vecs[:6]
        distractor_vecs = new_vecs[6:]
        reused = gallery_v1["vectors"]
        if not np.array_equal(reused, gallery_v1["vectors"]):
            raise GalleryV2Error("gallery-v1 mutated during build")
        target_gallery = np.concatenate([reused, target_new], axis=0).astype(np.float32)
        if target_gallery.shape != (13, 512):
            raise GalleryV2Error("target gallery shape")
        if not np.array_equal(target_gallery[:7], reused):
            raise GalleryV2Error("reused float32 values changed")

        target_centroid = compute_centroid(target_gallery)
        target_medoid = compute_medoid(target_gallery, GALLERY_V2_IDS)
        target_diag = internal_diagnostics(
            target_gallery,
            GALLERY_V2_IDS,
            target_centroid,
            target_medoid,
            config["outlier_diagnostics"],
        )

        distractor_centroid = compute_centroid(distractor_vecs)
        distractor_medoid = compute_medoid(distractor_vecs, HN_YES_IDS)
        distractor_diag = internal_diagnostics(
            distractor_vecs,
            HN_YES_IDS,
            distractor_centroid,
            distractor_medoid,
            config["outlier_diagnostics"],
        )
        cross = cross_diagnostics(
            target_gallery, GALLERY_V2_IDS, distractor_vecs, HN_YES_IDS, hn_crops
        )

        # --- target gallery artifacts ---
        np.save(tg / "target_001_gallery_v2_individual_embeddings.npy", target_gallery)
        np.save(tg / "target_001_gallery_v2_centroid.npy", target_centroid)
        np.save(tg / "target_001_gallery_v2_medoid.npy", target_medoid["vector"])
        np.save(
            tg / "target_001_gallery_v2_pairwise_cosine.npy", target_diag["pairwise"]
        )
        target_members = []
        for i, mid in enumerate(GALLERY_V2_IDS):
            if i < 7:
                src = gallery_v1["members"][i]
                target_members.append(
                    {
                        "gallery_row_index": i,
                        "member_id": mid,
                        "target_id": TARGET_ID,
                        "gallery_member_type": "individual_frozen_anchor_v1_reused",
                        "human_approved": True,
                        "automatic_enrollment": False,
                        "gallery_v1_member": True,
                        "gallery_v2_member": True,
                        "source_external_only": True,
                        "source_occurrence_code": src["source_occurrence_code"],
                        "view_category": src.get("view_category", ""),
                        "crop_path": src.get("crop_path", ""),
                        "crop_sha256": src.get("crop_sha256", ""),
                        "reused_from_gallery_v1": True,
                        "embedding_recomputed": False,
                    }
                )
            else:
                src = new_target[i - 7]
                target_members.append(
                    {
                        "gallery_row_index": i,
                        "member_id": mid,
                        "target_id": TARGET_ID,
                        "gallery_member_type": "individual_external_refinement_anchor_v2",
                        "human_approved": True,
                        "automatic_enrollment": False,
                        "gallery_v1_member": False,
                        "gallery_v2_member": True,
                        "source_external_only": True,
                        "source_kind": src["source_kind"],
                        "source_occurrence_code": src["source_occurrence_code"],
                        "raw_track_id": src["raw_track_id"],
                        "frame_index": src["frame_index"],
                        "view_category": src.get("view_category", ""),
                        "crop_path": src["crop_path"],
                        "crop_sha256": src["crop_sha256"],
                        "reused_from_gallery_v1": False,
                        "embedding_recomputed": True,
                    }
                )
        write_jsonl(tg / "target_001_gallery_v2_member_inventory.jsonl", target_members)
        write_json(
            tg / "target_001_gallery_v2_lineage.json",
            {
                "schema_version": "reid_target_001_gallery_v2_lineage_v1",
                "reused_gallery_v1_ids": list(GALLERY_V1_IDS),
                "new_target_ids": list(NEW_TARGET_IDS),
                "gallery_v2_ids": list(GALLERY_V2_IDS),
                "gallery_v1_embedding_sha256": gallery_v1["embedding_sha256"],
                "gallery_v1_snapshot_sha256": gallery_v1["snapshot_sha256"],
                "f3f_snapshot_sha256": f3f["snapshot_sha256"],
                "external_source_sha256": assets["external_sha256"],
            },
        )
        target_emb_sha = sha256_bytes(
            (tg / "target_001_gallery_v2_individual_embeddings.npy").read_bytes()
        )
        write_json(
            tg / "target_001_gallery_v2_embedding_manifest.json",
            {
                "schema_version": "reid_target_001_gallery_v2_embedding_manifest_v1",
                "npy": "target_001_gallery_v2_individual_embeddings.npy",
                "sha256": target_emb_sha,
                "shape": [13, 512],
                "dtype": "float32",
                "row_order": list(GALLERY_V2_IDS),
                "l2_normalized": True,
                "reused_prefix_sha256": gallery_v1["embedding_sha256"],
                "reused_rows": 7,
                "new_rows": 6,
            },
        )
        write_json(
            tg / "target_001_gallery_v2_internal_diagnostics.json",
            {
                "schema_version": "reid_target_001_gallery_v2_internal_diagnostics_v1",
                "medoid_member_id": target_medoid["member_id"],
                "medoid_mean_cosine_distance": target_medoid[
                    "mean_cosine_distance_to_others"
                ],
                "medoid_tie_break": target_medoid["tie_break"],
                "medoid_all_mean_distances": {
                    GALLERY_V2_IDS[i]: target_medoid["all_mean_distances"][i]
                    for i in range(13)
                },
                **target_diag["metrics"],
            },
        )

        # --- distractor gallery ---
        np.save(
            dg / "target_001_same_team_distractor_individual_embeddings.npy",
            distractor_vecs,
        )
        np.save(dg / "target_001_same_team_distractor_centroid.npy", distractor_centroid)
        np.save(
            dg / "target_001_same_team_distractor_medoid.npy", distractor_medoid["vector"]
        )
        np.save(
            dg / "target_001_same_team_distractor_pairwise_cosine.npy",
            distractor_diag["pairwise"],
        )
        distractor_members = []
        for i, src in enumerate(hn_crops):
            distractor_members.append(
                {
                    "gallery_row_index": i,
                    "member_id": src["member_id"],
                    "target_id": TARGET_ID,
                    "identity_role": "same_team_human_verified_non_target",
                    "human_approved": True,
                    "target_identity_negative": True,
                    "automatic_enrollment": False,
                    "source_external_only": True,
                    "source_external_code": src["source_external_code"],
                    "raw_track_id": src["raw_track_id"],
                    "selected_frame": src["selected_frame"],
                    "crop_path": src["crop_path"],
                    "crop_sha256": src["crop_sha256"],
                    "human_visible_jersey_number": src["human_visible_jersey_number"],
                    "human_jersey_provenance": src["human_jersey_provenance"],
                }
            )
        write_jsonl(
            dg / "target_001_same_team_distractor_member_inventory.jsonl",
            distractor_members,
        )
        write_json(
            dg / "target_001_same_team_distractor_lineage.json",
            {
                "schema_version": "reid_target_001_same_team_distractor_lineage_v1",
                "member_ids": list(HN_YES_IDS),
                "f3f_snapshot_sha256": f3f["snapshot_sha256"],
                "external_source_sha256": assets["external_sha256"],
                "note": "23 members may span multiple real players; not a single identity prototype",
            },
        )
        dist_emb_sha = sha256_bytes(
            (
                dg / "target_001_same_team_distractor_individual_embeddings.npy"
            ).read_bytes()
        )
        write_json(
            dg / "target_001_same_team_distractor_embedding_manifest.json",
            {
                "schema_version": "reid_target_001_same_team_distractor_embedding_manifest_v1",
                "npy": "target_001_same_team_distractor_individual_embeddings.npy",
                "sha256": dist_emb_sha,
                "shape": [23, 512],
                "dtype": "float32",
                "row_order": list(HN_YES_IDS),
                "l2_normalized": True,
            },
        )
        write_json(
            dg / "target_001_same_team_distractor_internal_diagnostics.json",
            {
                "schema_version": "reid_target_001_same_team_distractor_internal_diagnostics_v1",
                "medoid_member_id": distractor_medoid["member_id"],
                "medoid_mean_cosine_distance": distractor_medoid[
                    "mean_cosine_distance_to_others"
                ],
                "medoid_tie_break": distractor_medoid["tie_break"],
                "medoid_all_mean_distances": {
                    HN_YES_IDS[i]: distractor_medoid["all_mean_distances"][i]
                    for i in range(23)
                },
                **distractor_diag["metrics"],
            },
        )
        jersey_groups: dict[str, list[str]] = {}
        for src in hn_crops:
            key = src["human_visible_jersey_number"] or "unknown"
            jersey_groups.setdefault(key, []).append(src["member_id"])
        write_json(
            dg / "target_001_same_team_distractor_human_jersey_inventory.json",
            {
                "schema_version": "reid_target_001_same_team_distractor_human_jersey_inventory_v1",
                "provenance_rule": "human_only_no_guessing",
                "groups": {
                    k: {"member_ids": v, "count": len(v)}
                    for k, v in sorted(jersey_groups.items())
                },
                "aggregation_prototypes_created": False,
                "unknown_identities_not_merged": True,
            },
        )

        # --- cross ---
        np.save(
            xd / "target_001_target_distractor_cross_cosine.npy", cross["matrix"]
        )
        write_json(
            xd / "target_001_target_distractor_cross_similarity_summary.json",
            cross["summary"],
        )
        write_jsonl(
            xd / "target_001_target_distractor_top_cross_pairs.jsonl",
            cross["top_pairs"],
        )

        access_audit = {
            "schema_version": "reid_stage5d_f3g_access_audit_v1",
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "f3_score_row_read_count": 0,
            "f3_rank_row_read_count": 0,
            "excluded_hard_negative_crop_read_count": 0,
            "excluded_hard_negative_ids": list(HN_EXCLUDED_IDS),
            "approved_crop_reads": len(crop_read_log),
            "new_detection": False,
            "new_tracking": False,
            "ocr_inference": False,
            "external_mp4_decode": False,
            "gallery_v1_mutation": False,
            "gallery_v1_embedding_recomputed": False,
            "network_download": 0,
            "crop_copies": 0,
            "checkpoint_copies": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "target_distractor_score_formula_applied": False,
        }
        write_json(rt / "target_001_gallery_v2_access_audit.json", access_audit)

        npy_files = list(tmp.rglob("*.npy"))
        if len(npy_files) != EXPECTED_NPY:
            raise GalleryV2Error(f"npy budget {len(npy_files)} != {EXPECTED_NPY}")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise GalleryV2Error("media artifacts forbidden")
        if list(tmp.rglob("*.pth")) or list(tmp.rglob("*.tar")):
            raise GalleryV2Error("checkpoint copies forbidden")

        # Confirm gallery-v1 unchanged on disk after build.
        g1_path = (
            project_root
            / config["gallery_v1"]["path"]
            / "embeddings"
            / "target_001_anchor_embeddings.npy"
        )
        if sha256_file(g1_path) != gallery_v1["embedding_sha256"]:
            raise GalleryV2Error("gallery-v1 mutated on disk")

        file_count, files_sha = listing_sha(tmp)
        contract = {
            "schema_version": "reid_stage5d_f3g_gallery_v2_contract_v1",
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "exact_next_gate": NEXT_GATE,
            "reused_target_members": 7,
            "new_target_members": 6,
            "target_gallery_v2_members": 13,
            "distractor_members": 23,
            "new_embeddings": 29,
            "embedding_dimension": 512,
            "target_centroid": 1,
            "target_medoid": 1,
            "distractor_centroid": 1,
            "distractor_medoid": 1,
            "target_pairwise_shape": [13, 13],
            "distractor_pairwise_shape": [23, 23],
            "cross_matrix_shape": [13, 23],
            "sample_reads": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "gallery_v1_mutation": False,
            "descriptive_internal_enrollment_diagnostic_only": True,
            "same_source_cross_similarity_not_independent": True,
            "generated_at": generated_at,
            "project_head": head,
        }
        write_json(tmp / "stage5d_f3g_contract.json", contract)
        summary = {
            "schema_version": "reid_stage5d_f3g_gallery_v2_summary_v1",
            "final_status": FINAL_STATUS,
            "readiness": READINESS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "reused_target_members": 7,
            "new_target_members": 6,
            "target_gallery_v2_members": 13,
            "target_gallery_v2_ids": list(GALLERY_V2_IDS),
            "distractor_members": 23,
            "distractor_member_ids": list(HN_YES_IDS),
            "new_embeddings": 29,
            "embedding_dimension": 512,
            "target_centroid": 1,
            "target_medoid": 1,
            "target_medoid_member_id": target_medoid["member_id"],
            "distractor_centroid": 1,
            "distractor_medoid": 1,
            "distractor_medoid_member_id": distractor_medoid["member_id"],
            "target_pairwise_shape": [13, 13],
            "distractor_pairwise_shape": [23, 23],
            "cross_matrix_shape": [13, 23],
            "target_internal_off_diagonal": {
                "min": target_diag["metrics"]["off_diagonal_min"],
                "median": target_diag["metrics"]["off_diagonal_median"],
                "mean": target_diag["metrics"]["off_diagonal_mean"],
                "max": target_diag["metrics"]["off_diagonal_max"],
            },
            "distractor_internal_off_diagonal": {
                "min": distractor_diag["metrics"]["off_diagonal_min"],
                "median": distractor_diag["metrics"]["off_diagonal_median"],
                "mean": distractor_diag["metrics"]["off_diagonal_mean"],
                "max": distractor_diag["metrics"]["off_diagonal_max"],
            },
            "cross_similarity_global": {
                "min": cross["summary"]["global_min"],
                "median": cross["summary"]["global_median"],
                "mean": cross["summary"]["global_mean"],
                "max": cross["summary"]["global_max"],
            },
            "two_pass_determinism": embed_result["determinism"],
            "gallery_v1_members": 7,
            "gallery_v1_embedding_sha256": gallery_v1["embedding_sha256"],
            "gallery_v1_mutation": False,
            "sample_video_read": False,
            "sample_crop_read_count": 0,
            "sample_embedding_read_count": 0,
            "f3_score_row_read_count": 0,
            "f3_rank_row_read_count": 0,
            "excluded_hard_negative_crop_read_count": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "osnet_checkpoint_sha256": assets["osnet"]["sha256"],
            "sn_reid_commit": runtime["sn_reid_commit"],
            "f3f_snapshot_sha256": f3f["snapshot_sha256"],
            "gallery_v1_snapshot_sha256": gallery_v1["snapshot_sha256"],
            "npy_count": EXPECTED_NPY,
            "output_file_count": file_count,
            "output_listing_sha256": files_sha,
            "generated_at": generated_at,
        }
        write_json(tmp / "stage5d_f3g_summary.json", summary)
        write_json(
            tmp / "stage5d_f3g_manifest.json",
            {
                "schema_version": "reid_stage5d_f3g_gallery_v2_manifest_v1",
                "final_status": FINAL_STATUS,
                "readiness": READINESS,
                "exact_next_gate": NEXT_GATE,
                "project_head": head,
                "output_file_count": file_count,
                "output_listing_sha256": files_sha,
                "artifact_budget": {
                    "npy": EXPECTED_NPY,
                    "png": 0,
                    "mp4": 0,
                    "crop_copies": 0,
                    "checkpoint_copies": 0,
                    "source_video_copies": 0,
                    "new_target_embeddings": 6,
                    "new_distractor_embeddings": 23,
                    "total_new_embeddings": 29,
                    "threshold_artifacts": 0,
                    "identity_assignments": 0,
                },
                "generated_at": generated_at,
            },
        )
        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_json(final_dir / "stage5d_f3g_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_gallery_v2_and_distractor_gallery_stage5d_target_001.yaml",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "readiness": summary["readiness"],
                "exact_next_gate": summary["exact_next_gate"],
                "target_gallery_v2_members": summary["target_gallery_v2_members"],
                "distractor_members": summary["distractor_members"],
                "new_embeddings": summary["new_embeddings"],
                "target_medoid_member_id": summary["target_medoid_member_id"],
                "distractor_medoid_member_id": summary["distractor_medoid_member_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
