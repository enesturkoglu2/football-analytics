#!/usr/bin/env python3
"""Stage 5D-B1E-F — Target 001 frozen anchor OSNet embedding and gallery.

Embeds only the seven human-approved external anchors with canonical OSNet.
Builds individual gallery + centroid + medoid and an internal cosine audit.
Does not run sample.mp4 retrieval, threshold selection, YOLO, ByteTrack, OCR,
or automatic gallery growth.
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
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid import embedding as emb  # noqa: E402

CONFIG_SCHEMA = "reid_target_gallery_build_config_v1"
APPROVED_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)
EXCLUDED_IDS = (
    "target_001_ext_anchor_002",
    "target_001_ext_anchor_005",
    "target_001_ext_anchor_007",
    "target_001_ext_anchor_009",
    "target_001_ext_anchor_010",
    "target_001_ext_anchor_012",
    "target_001_ext_anchor_013",
    "target_001_ext_anchor_015",
)
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
STATUS_READY = (
    "COMPLETED_STAGE5D_B1E_F_TARGET_001_GALLERY_BUILT_READY_FOR_VALIDATION"
)
STATUS_INTERNAL = (
    "COMPLETED_STAGE5D_B1E_F_TARGET_001_GALLERY_INTERNAL_REVIEW_REQUIRED"
)
NEXT_GATE_READY = (
    "STAGE5D-F1_TARGET_001_INDEPENDENT_SAMPLE_RETRIEVAL_VALIDATION_DESIGN"
)
NEXT_GATE_INTERNAL = "STAGE5D-B1E-F2_TARGET_001_GALLERY_INTERNAL_OUTLIER_REVIEW"
READINESS_A = "TARGET_001_GALLERY_BUILT_READY_FOR_INDEPENDENT_VALIDATION"
READINESS_B = "TARGET_001_GALLERY_BUILT_REQUIRES_INTERNAL_REVIEW"
READINESS_C = "BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY"

ALLOWED_DIRTY = {
    "scripts/run_reid_target_gallery_build.py",
    "configs/reid/target_gallery_build_stage5d_target_001.yaml",
    "tests/test_reid_target_gallery_build.py",
    "docs/setup/stage5d-target-frozen-anchor-osnet-gallery-build.md",
}


class GalleryBuildError(RuntimeError):
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
        raise GalleryBuildError("unexpected config schema")
    if not config.get("offline_required"):
        raise GalleryBuildError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GalleryBuildError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path.startswith("?? "):
                path = path[3:]
            # git status porcelain: XY PATH or XY PATH -> PATH2
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GalleryBuildError(
                    "BLOCKED_STAGE5D_B1E_F_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_GIT_CONTRACT_MISMATCH message")
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
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_immutable_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise GalleryBuildError("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise GalleryBuildError("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise GalleryBuildError("external sha mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise GalleryBuildError("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise GalleryBuildError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise GalleryBuildError("yolo sha mismatch")
    ck = emb.verify_checkpoint(
        osnet, expected_sha256=config["osnet_checkpoint"]["expected_sha256"]
    )
    return {
        "external_sha256": config["external_enrollment_source"]["expected_sha256"],
        "sample_sha256": config["evaluation_source"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet": ck,
    }


def validate_anchor_freeze(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    exp = config["stage5d_b1e_e_package"]
    root = project_root / exp["path"]
    summary = load_json(root / "stage5d_b1e_e_summary.json")
    contract = load_json(
        root / "anchor_freeze" / "target_001_external_anchor_freeze_contract.json"
    )
    freeze = load_json(root / "anchor_freeze" / "target_001_external_anchor_freeze.json")
    if summary.get("final_status") != exp["expected_final_status"]:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != exp["expected_target_id"]:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH target"
        )
    if summary.get("target_alias") != exp["expected_target_alias"]:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH alias"
        )
    checks = {
        "reviewed_candidate_crops": exp["expected_reviewed_candidates"],
        "frozen_approved_anchors": exp["expected_approved_anchors"],
        "redundant_valid_non_selected": exp["expected_redundant_valid_non_selected"],
        "multi_person_ambiguous": exp["expected_multi_person_ambiguous"],
        "invalid": exp["expected_invalid"],
        "embeddings": exp["expected_embeddings"],
        "gallery_members": exp["expected_gallery_members"],
        "prototypes": exp["expected_prototypes"],
        "identity_assignments": exp["expected_identity_assignments"],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GalleryBuildError(
                f"BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH {key}"
            )
    if list(summary.get("approved_exact_ids") or []) != list(APPROVED_IDS):
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH approved_ids"
        )
    occ = summary.get("occurrence_distribution") or {}
    for code, count in exp["expected_occurrence_distribution"].items():
        if int(occ.get(code) or 0) != int(count):
            raise GalleryBuildError(
                f"BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH occ_{code}"
            )
    views = summary.get("view_distribution") or {}
    for view, count in exp["expected_view_distribution"].items():
        if int(views.get(view) or 0) != int(count):
            raise GalleryBuildError(
                f"BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH view_{view}"
            )
    if contract.get("embedding_generated") is not False:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH embeddings"
        )
    if contract.get("frozen_anchors_are_gallery_members") is not False:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH gallery"
        )
    if contract.get("automatic_gallery_growth") is not False:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_ANCHOR_FREEZE_CONTRACT_MISMATCH auto_growth"
        )
    snapshot_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    approved_csv = root / "anchor_freeze" / "target_001_external_approved_anchors_frozen.csv"
    decisions_csv = (
        root / "anchor_freeze" / "target_001_external_anchor_review_decisions_frozen.csv"
    )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "freeze": freeze,
        "approved_csv": approved_csv,
        "decisions_csv": decisions_csv,
        "snapshot_sha256": snapshot_sha,
        "freeze_sha256": sha256_file(
            root / "anchor_freeze" / "target_001_external_anchor_freeze_manifest.json"
        ),
    }


def _parse_bbox(raw: str) -> list[float]:
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 4:
        raise GalleryBuildError("invalid bbox")
    return [float(x) for x in value]


def validate_approved_crops(
    project_root: Path,
    config: Mapping[str, Any],
    freeze_info: Mapping[str, Any],
    *,
    crop_read_log: list[str],
) -> list[dict[str, Any]]:
    b1ed = project_root / config["stage5d_b1e_d_package"]["path"]
    summary_d = load_json(b1ed / "stage5d_b1e_d_summary.json")
    if summary_d.get("final_status") != config["stage5d_b1e_d_package"][
        "expected_final_status"
    ]:
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY b1e_d")
    inv_path = b1ed / "inventory" / "target_001_external_anchor_candidate_inventory.jsonl"
    inventory = {
        row["anchor_candidate_id"]: row
        for row in (
            json.loads(line)
            for line in inv_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    decisions: dict[str, dict[str, str]] = {}
    with freeze_info["decisions_csv"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            decisions[row["anchor_candidate_id"]] = row

    approved: list[dict[str, Any]] = []
    with freeze_info["approved_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7:
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY count")
    if [r["anchor_candidate_id"] for r in rows] != list(APPROVED_IDS):
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY order")

    ext_sha = config["external_enrollment_source"]["expected_sha256"]
    for row in rows:
        cid = row["anchor_candidate_id"]
        assert_no_path_traversal(row["crop_path"])
        if row["target_id"] != TARGET_ID:
            raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY target")
        if row["source_occurrence_code"] not in {"EXT_004", "EXT_183", "EXT_198"}:
            raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY occ")
        inv = inventory.get(cid)
        if inv is None:
            raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY inv")
        crop_path = (b1ed / row["crop_path"]).resolve()
        try:
            crop_path.relative_to(b1ed.resolve())
        except ValueError as exc:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY escape"
            ) from exc
        if not crop_path.is_file() or crop_path.is_symlink():
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY not_regular"
            )
        crop_read_log.append(str(crop_path))
        actual_sha = sha256_file(crop_path)
        if actual_sha != row["crop_sha256"] or actual_sha != inv["crop_sha256"]:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY sha"
            )
        import cv2

        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY decode"
            )
        h, w = int(image.shape[0]), int(image.shape[1])
        if h != int(row["crop_height"]) or w != int(row["crop_width"]):
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY dims"
            )
        if h != int(inv["crop_height"]) or w != int(inv["crop_width"]):
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY inv_dims"
            )
        if _parse_bbox(row["original_bbox_xyxy"]) != [
            float(x) for x in inv["original_bbox_xyxy"]
        ]:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY bbox"
            )
        if _parse_bbox(row["padded_crop_bbox_xyxy"]) != [
            float(x) for x in inv["padded_crop_bbox_xyxy"]
        ]:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY crop_bbox"
            )
        if inv.get("source_video_sha256") != ext_sha:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY ext_sha"
            )
        dec = decisions.get(cid)
        if dec is None:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY decision"
            )
        for key, expected in (
            ("manual_anchor_decision", "target_anchor_yes"),
            ("manual_crop_valid", "yes"),
            ("manual_target_dominant", "yes"),
            ("manual_single_person", "yes"),
            ("manual_identity_confirmed", "yes"),
        ):
            if str(dec.get(key)) != expected:
                raise GalleryBuildError(
                    f"BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY {key}"
                )
            if key != "manual_anchor_decision" and str(row.get(key)) != expected:
                raise GalleryBuildError(
                    f"BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY approved_{key}"
                )
        if int(row["frame_index"]) < 0:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY frame"
            )
        approved.append(
            {
                "anchor_candidate_id": cid,
                "frozen_anchor_id": row["frozen_anchor_id"],
                "target_id": TARGET_ID,
                "source_occurrence_code": row["source_occurrence_code"],
                "raw_track_id": int(row["raw_track_id"]),
                "frame_index": int(row["frame_index"]),
                "video_time": float(row["video_time"]),
                "view_category": row["manual_view_category"],
                "crop_path_relative": row["crop_path"],
                "crop_path_absolute": str(crop_path),
                "crop_sha256": actual_sha,
                "crop_width": w,
                "crop_height": h,
                "original_bbox_xyxy": _parse_bbox(row["original_bbox_xyxy"]),
                "padded_crop_bbox_xyxy": _parse_bbox(row["padded_crop_bbox_xyxy"]),
                "target_definition_sha256": row["target_definition_sha256"],
                "occurrence_freeze_sha256": row["occurrence_freeze_sha256"],
                "anchor_review_package_sha256": row["anchor_review_package_sha256"],
            }
        )

    # Ensure excluded crops were never opened.
    excluded_paths = {
        str((b1ed / f"crops/{occ}/{cid}.png").resolve())
        for cid in EXCLUDED_IDS
        for occ in ("EXT_004", "EXT_183", "EXT_198")
        if (b1ed / f"crops").exists()
    }
    # Resolve actual excluded paths from inventory without reading image bytes.
    for cid in EXCLUDED_IDS:
        inv = inventory[cid]
        assert_no_path_traversal(inv["crop_path"])
        excluded_paths.add(str((b1ed / inv["crop_path"]).resolve()))
    for path in crop_read_log:
        if path in excluded_paths and Path(path).name.replace(".png", "") in EXCLUDED_IDS:
            # crop_read_log only gets approved paths; double-check
            stem = Path(path).stem
            if stem in EXCLUDED_IDS:
                raise GalleryBuildError(
                    "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY excluded_read"
                )
    return approved


def resolve_stage5d_a(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_a_preflight"]["path"]
    design_path = root / config["stage5d_a_preflight"]["gallery_design_contract"]
    design = load_json(design_path)
    if design.get("automatic_gallery_growth") is not False:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_PREPROCESSING_CONTRACT_MISMATCH auto_growth"
        )
    return {
        "root": str(root),
        "design_contract_path": str(design_path.relative_to(project_root)),
        "design_contract_sha256": sha256_file(design_path),
        "design": design,
    }


def build_preprocessing_contract(
    *,
    config: Mapping[str, Any],
    assets: Mapping[str, Any],
    freeze_info: Mapping[str, Any],
    stage5d_a: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    contract = {
        "schema_version": "reid_target_001_gallery_preprocessing_contract_v1",
        "predictions_embeddings_seen_at_freeze": False,
        "crop_count": 7,
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
        "scoring_view": "l2_normalized_embedding",
        "stored_source": "original_frozen_embedding",
        "original_is_l2_normalized": True,
        "checkpoint_path": config["osnet_checkpoint"]["path"],
        "checkpoint_sha256": assets["osnet"]["sha256"],
        "sn_reid_root": runtime["sn_reid_root"],
        "sn_reid_commit": runtime["sn_reid_commit"],
        "approved_anchor_freeze_sha256": freeze_info["freeze_sha256"],
        "b1e_e_snapshot_sha256": freeze_info["snapshot_sha256"],
        "stage5d_a_gallery_design_contract_sha256": stage5d_a["design_contract_sha256"],
        "contract_source_paths": {
            "embedding_module": "src/football_analytics/reid/embedding.py",
            "stage5d_a_design": stage5d_a["design_contract_path"],
            "b1e_e_freeze": str(freeze_info["root"]),
        },
        "embedding_determinism_atol": float(runtime["embedding_determinism_atol"]),
        "unit_norm_atol": float(runtime["unit_norm_atol"]),
    }
    # Hard validation against canonical module constants.
    if contract["resize_height"] != 256 or contract["resize_width"] != 128:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_PREPROCESSING_CONTRACT_MISMATCH resize"
        )
    if contract["expected_dimension"] != 512:
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_PREPROCESSING_CONTRACT_MISMATCH dim"
        )
    if contract["model_name"] != "osnet_x1_0":
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_PREPROCESSING_CONTRACT_MISMATCH model"
        )
    return contract


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


def _load_model(checkpoint: Path, sn_reid_root: Path):
    with emb.temporary_sys_path_prepend(sn_reid_root):
        model = emb.build_osnet_cpu_model(model_name=emb.MODEL_NAME)
        emb.load_osnet_checkpoint_weights(model, checkpoint)
        model.eval()
        model.cpu()
        return model


def embed_approved_two_pass(
    *,
    approved: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    runtime: Mapping[str, Any],
    crop_read_log: list[str],
) -> dict[str, Any]:
    import torch

    tensors: list[Any] = []
    for row in approved:
        path = Path(row["crop_path_absolute"])
        if path.stem in EXCLUDED_IDS:
            raise GalleryBuildError("excluded crop embed attempted")
        crop_read_log.append(str(path.resolve()))
        tensors.append(emb.load_and_preprocess_crop(path))

    atol = float(runtime["embedding_determinism_atol"])
    sn_root = Path(runtime["sn_reid_root"])
    batch_size = int(runtime["batch_size"])

    # Pass 1
    model1 = _load_model(checkpoint, sn_root)
    with torch.inference_mode():
        pass1 = emb.embed_tensors(model1, tensors, batch_size=batch_size)
    del model1

    # Pass 2 — reload model for independent pass
    model2 = _load_model(checkpoint, sn_root)
    with torch.inference_mode():
        pass2 = emb.embed_tensors(model2, tensors, batch_size=batch_size)
    del model2

    if pass1.shape != (7, 512) or pass2.shape != (7, 512):
        raise GalleryBuildError("BLOCKED_STAGE5D_B1E_F_EMBEDDING_NONDETERMINISTIC shape")
    abs_diff = np.abs(pass1.astype(np.float64) - pass2.astype(np.float64))
    per_anchor_max = abs_diff.max(axis=1)
    overall_max = float(abs_diff.max())
    exact = bool(np.array_equal(pass1, pass2))
    within = bool(overall_max <= atol)
    if not (exact or within):
        raise GalleryBuildError(
            "BLOCKED_STAGE5D_B1E_F_EMBEDDING_NONDETERMINISTIC "
            f"max_abs={overall_max} atol={atol}"
        )

    vectors = pass1.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    zero_count = int(np.all(vectors == 0, axis=1).sum())
    if nan_count or inf_count or zero_count:
        raise GalleryBuildError("BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY vectors")
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise GalleryBuildError("BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY norms")

    return {
        "embeddings": vectors,
        "norms": norms.astype(np.float64),
        "pass1": pass1,
        "pass2": pass2,
        "determinism": {
            "exact_match": exact,
            "within_tolerance": within,
            "comparison_policy": "exact_or_max_abs_le_atol",
            "atol": atol,
            "per_anchor_max_absolute_difference": [float(x) for x in per_anchor_max],
            "overall_max_absolute_difference": overall_max,
            "passes": 2,
            "deterministic": True,
        },
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_count,
        "dtype": "float32",
        "shape": [7, 512],
    }


def compute_centroid(gallery: np.ndarray) -> np.ndarray:
    mean = gallery.mean(axis=0).astype(np.float64)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0:
        raise GalleryBuildError("BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY centroid")
    return (mean / norm).astype(np.float32)


def compute_medoid(gallery: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    # Cosine distance = 1 - cosine similarity; for L2 rows sim = dot.
    sims = gallery @ gallery.T
    n = gallery.shape[0]
    mean_dist = []
    for i in range(n):
        others = [1.0 - float(sims[i, j]) for j in range(n) if j != i]
        mean_dist.append(float(np.mean(others)))
    best = min(range(n), key=lambda i: (mean_dist[i], ids[i]))
    return {
        "index": best,
        "anchor_candidate_id": ids[best],
        "mean_cosine_distance_to_others": mean_dist[best],
        "vector": gallery[best].astype(np.float32, copy=True),
        "all_mean_distances": mean_dist,
    }


def pairwise_audit(
    gallery: np.ndarray,
    ids: Sequence[str],
    approved: Sequence[Mapping[str, Any]],
    centroid: np.ndarray,
    medoid: Mapping[str, Any],
    outlier_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    sims = (gallery @ gallery.T).astype(np.float64)
    n = sims.shape[0]
    off = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off.append(float(sims[i, j]))
    off_arr = np.asarray(off, dtype=np.float64)
    per_mean = []
    per_min = []
    for i in range(n):
        others = [float(sims[i, j]) for j in range(n) if j != i]
        per_mean.append(float(np.mean(others)))
        per_min.append(float(np.min(others)))
    centroid_sims = [float(np.dot(gallery[i], centroid)) for i in range(n)]
    medoid_vec = medoid["vector"]
    medoid_sims = [float(np.dot(gallery[i], medoid_vec)) for i in range(n)]

    # nearest / farthest off-diagonal
    nearest = None
    farthest = None
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sims[i, j])
            pair = (ids[i], ids[j], val)
            if nearest is None or val > nearest[2]:
                nearest = pair
            if farthest is None or val < farthest[2]:
                farthest = pair

    same_occ = []
    cross_occ = []
    same_view = []
    cross_view = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sims[i, j])
            if approved[i]["source_occurrence_code"] == approved[j]["source_occurrence_code"]:
                same_occ.append(val)
            else:
                cross_occ.append(val)
            if approved[i]["view_category"] == approved[j]["view_category"]:
                same_view.append(val)
            else:
                cross_view.append(val)

    median_mean = float(np.median(per_mean))
    mad = float(np.median([abs(x - median_mean) for x in per_mean]))
    # Robust scale: MAD * 1.4826 approximates std; gate asks MAD-based descriptive.
    mult = float(outlier_cfg.get("mad_multiplier", 3.0))
    threshold = median_mean - mult * mad if mad > 0 else None
    outliers = []
    for i, mean_sim in enumerate(per_mean):
        flags = []
        if threshold is not None and mean_sim < threshold:
            flags.append("mean_similarity_below_median_mad")
        if mean_sim == min(per_mean) and centroid_sims[i] == min(centroid_sims):
            flags.append("lowest_mean_and_centroid_similarity")
        elif centroid_sims[i] == min(centroid_sims):
            flags.append("lowest_centroid_similarity")
        if flags:
            outliers.append(
                {
                    "anchor_candidate_id": ids[i],
                    "mean_similarity_to_others": mean_sim,
                    "centroid_similarity": centroid_sims[i],
                    "flags": flags,
                    "removal_allowed": False,
                    "descriptive_only": True,
                }
            )

    serious_concern = False
    # Heuristic for readiness B: very low off-diagonal max or many MAD outliers.
    if float(np.min(off_arr)) < 0.0 or len(outliers) >= 3:
        serious_concern = True

    return {
        "pairwise": sims.astype(np.float32),
        "metrics": {
            "off_diagonal_min": float(np.min(off_arr)),
            "off_diagonal_median": float(np.median(off_arr)),
            "off_diagonal_mean": float(np.mean(off_arr)),
            "off_diagonal_max": float(np.max(off_arr)),
            "diagonal_mean": float(np.mean(np.diag(sims))),
            "per_anchor_mean_similarity_to_others": {
                ids[i]: per_mean[i] for i in range(n)
            },
            "per_anchor_min_similarity_to_others": {
                ids[i]: per_min[i] for i in range(n)
            },
            "centroid_similarity_per_anchor": {
                ids[i]: centroid_sims[i] for i in range(n)
            },
            "medoid_similarity_per_anchor": {
                ids[i]: medoid_sims[i] for i in range(n)
            },
            "nearest_anchor_pair": {
                "a": nearest[0],
                "b": nearest[1],
                "cosine_similarity": nearest[2],
            },
            "farthest_anchor_pair": {
                "a": farthest[0],
                "b": farthest[1],
                "cosine_similarity": farthest[2],
            },
            "occurrence_pair_distribution": {
                "same_occurrence_count": len(same_occ),
                "cross_occurrence_count": len(cross_occ),
                "same_occurrence_mean": float(np.mean(same_occ)) if same_occ else None,
                "cross_occurrence_mean": float(np.mean(cross_occ)) if cross_occ else None,
            },
            "view_pair_diagnostics": {
                "same_view_count": len(same_view),
                "cross_view_count": len(cross_view),
                "same_view_mean": float(np.mean(same_view)) if same_view else None,
                "cross_view_mean": float(np.mean(cross_view)) if cross_view else None,
            },
            "outlier_diagnostics": {
                "method": "median_mad",
                "median_mean_similarity": median_mean,
                "mad": mad,
                "mad_multiplier": mult,
                "threshold_mean_similarity": threshold,
                "removal_allowed": False,
                "descriptive_only": True,
                "flagged": outliers,
            },
            "serious_internal_consistency_concern": serious_concern,
            "identity_threshold": None,
            "threshold_selected": False,
            "automatic_anchor_removal": False,
        },
    }


def decide_readiness(
    *,
    approved: Sequence[Mapping[str, Any]],
    embed_result: Mapping[str, Any],
    audit: Mapping[str, Any],
    centroid: np.ndarray,
    medoid: Mapping[str, Any],
) -> dict[str, Any]:
    occ = {r["source_occurrence_code"] for r in approved}
    views = {r["view_category"] for r in approved}
    tech_ok = (
        len(approved) == 7
        and embed_result["shape"] == [7, 512]
        and embed_result["nan_count"] == 0
        and embed_result["inf_count"] == 0
        and embed_result["zero_vector_count"] == 0
        and embed_result["determinism"]["deterministic"] is True
        and len(occ) >= 2
        and len(views) >= 3
        and np.isfinite(centroid).all()
        and float(np.linalg.norm(centroid)) > 0
        and medoid["anchor_candidate_id"] in APPROVED_IDS
    )
    if not tech_ok:
        return {
            "readiness": READINESS_C,
            "final_status": "BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY",
            "exact_next_gate": None,
            "technical_minimums_passed": False,
        }
    if audit["metrics"]["serious_internal_consistency_concern"]:
        return {
            "readiness": READINESS_B,
            "final_status": STATUS_INTERNAL,
            "exact_next_gate": NEXT_GATE_INTERNAL,
            "technical_minimums_passed": True,
        }
    return {
        "readiness": READINESS_A,
        "final_status": STATUS_READY,
        "exact_next_gate": NEXT_GATE_READY,
        "technical_minimums_passed": True,
    }


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_b1e_f_gallery_{final_dir.name}_{token}"
    if tmp.exists():
        raise GalleryBuildError("FAILED_STAGE5D_B1E_F_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GalleryBuildError("FAILED_STAGE5D_B1E_F_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise GalleryBuildError("FAILED_STAGE5D_B1E_F_ATOMIC_OUTPUT final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    if tuple(config["approved_exact_ids"]) != APPROVED_IDS:
        raise GalleryBuildError("approved_exact_ids config mismatch")
    if tuple(config["excluded_candidate_ids"]) != EXCLUDED_IDS:
        raise GalleryBuildError("excluded_candidate_ids config mismatch")

    assets = validate_immutable_assets(project_root, config)
    freeze_info = validate_anchor_freeze(project_root, config)
    crop_read_log: list[str] = []
    approved = validate_approved_crops(
        project_root, config, freeze_info, crop_read_log=crop_read_log
    )
    stage5d_a = resolve_stage5d_a(project_root, config)
    runtime = resolve_runtime(config)

    tmp = create_temp_root(final_dir)
    try:
        for sub in ("embeddings", "gallery", "audit", "runtime", "effective_configs"):
            (tmp / sub).mkdir(parents=True, exist_ok=False)

        pre_contract = build_preprocessing_contract(
            config=config,
            assets=assets,
            freeze_info=freeze_info,
            stage5d_a=stage5d_a,
            runtime=runtime,
        )
        pre_path = tmp / "runtime" / "target_001_gallery_preprocessing_contract_pre_inference.json"
        write_json(pre_path, pre_contract)
        pre_sha = sha256_file(pre_path)

        # Inference only after contract freeze.
        checkpoint = Path(config["osnet_checkpoint"]["path"])
        osnet_sha_before = assets["osnet"]["sha256"]
        embed_result = embed_approved_two_pass(
            approved=approved,
            checkpoint=checkpoint,
            runtime=runtime,
            crop_read_log=crop_read_log,
        )
        if sha256_file(checkpoint) != osnet_sha_before:
            raise GalleryBuildError("osnet checkpoint mutated")

        # Excluded crop reads must remain zero (image opens only for approved).
        excluded_stems_read = [
            Path(p).stem for p in crop_read_log if Path(p).stem in EXCLUDED_IDS
        ]
        if excluded_stems_read:
            raise GalleryBuildError(
                "BLOCKED_STAGE5D_B1E_F_APPROVED_CROP_INTEGRITY excluded_read"
            )
        approved_reads = [Path(p).stem for p in crop_read_log if Path(p).stem in APPROVED_IDS]
        # Each approved crop read during integrity + embedding (>=2 per id).
        if set(approved_reads) != set(APPROVED_IDS):
            raise GalleryBuildError("approved crop read set incomplete")

        vectors = embed_result["embeddings"]
        gallery = vectors.astype(np.float32, copy=True)  # already L2
        unit_atol = float(runtime["unit_norm_atol"])
        norms = np.linalg.norm(gallery, axis=1)
        if np.any(np.abs(norms - 1.0) > unit_atol):
            raise GalleryBuildError("BLOCKED_TARGET_001_GALLERY_BUILD_INTEGRITY unit_norm")

        centroid = compute_centroid(gallery)
        medoid = compute_medoid(gallery, list(APPROVED_IDS))
        audit = pairwise_audit(
            gallery,
            list(APPROVED_IDS),
            approved,
            centroid,
            medoid,
            config["outlier_diagnostics"],
        )
        decision = decide_readiness(
            approved=approved,
            embed_result=embed_result,
            audit=audit,
            centroid=centroid,
            medoid=medoid,
        )
        if decision["readiness"] == READINESS_C:
            raise GalleryBuildError(decision["final_status"])

        # Write embeddings
        emb_path = tmp / "embeddings" / "target_001_anchor_embeddings.npy"
        np.save(emb_path, vectors)
        meta_path = tmp / "embeddings" / "target_001_anchor_embedding_metadata.jsonl"
        with meta_path.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(approved):
                rec = {
                    "row_index": i,
                    "anchor_candidate_id": row["anchor_candidate_id"],
                    "frozen_anchor_id": row["frozen_anchor_id"],
                    "target_id": TARGET_ID,
                    "source_occurrence_code": row["source_occurrence_code"],
                    "view_category": row["view_category"],
                    "frame_index": row["frame_index"],
                    "video_time": row["video_time"],
                    "crop_path": row["crop_path_relative"],
                    "crop_sha256": row["crop_sha256"],
                    "embedding_dtype": embed_result["dtype"],
                    "embedding_shape": [512],
                    "embedding_l2_norm": float(embed_result["norms"][i]),
                    "checkpoint_sha256": osnet_sha_before,
                    "preprocessing_contract_sha256": pre_sha,
                    "inference_pass_determinism": {
                        "exact_match": embed_result["determinism"]["exact_match"],
                        "within_tolerance": embed_result["determinism"]["within_tolerance"],
                        "max_absolute_difference": embed_result["determinism"][
                            "per_anchor_max_absolute_difference"
                        ][i],
                        "atol": embed_result["determinism"]["atol"],
                    },
                    "human_approved": True,
                    "automatic_enrollment": False,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        emb_sha_doc = {
            "schema_version": "reid_target_001_anchor_embeddings_sha256_v1",
            "embeddings_npy": "embeddings/target_001_anchor_embeddings.npy",
            "sha256": sha256_file(emb_path),
            "shape": [7, 512],
            "dtype": "float32",
            "row_order": list(APPROVED_IDS),
            "l2_normalized": True,
        }
        write_json(tmp / "embeddings" / "target_001_anchor_embeddings_sha256.json", emb_sha_doc)

        # Gallery
        np.save(tmp / "gallery" / "target_001_individual_gallery.npy", gallery)
        np.save(tmp / "gallery" / "target_001_gallery_centroid.npy", centroid)
        np.save(tmp / "gallery" / "target_001_gallery_medoid.npy", medoid["vector"])
        members_path = tmp / "gallery" / "target_001_gallery_members.jsonl"
        with members_path.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(approved):
                handle.write(
                    json.dumps(
                        {
                            "gallery_row_index": i,
                            "gallery_member_type": "individual_frozen_anchor",
                            "target_id": TARGET_ID,
                            "anchor_candidate_id": row["anchor_candidate_id"],
                            "frozen_anchor_id": row["frozen_anchor_id"],
                            "source_occurrence_code": row["source_occurrence_code"],
                            "view_category": row["view_category"],
                            "crop_path": row["crop_path_relative"],
                            "crop_sha256": row["crop_sha256"],
                            "human_approved": True,
                            "automatic_enrollment": False,
                            "pseudo_label": False,
                            "embedding_row_index": i,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        prototypes = {
            "schema_version": "reid_target_001_gallery_prototypes_v1",
            "target_id": TARGET_ID,
            "individual_members": 7,
            "centroid_count": 1,
            "medoid_count": 1,
            "centroid": {
                "type": "arithmetic_mean_then_l2",
                "dimension": 512,
                "l2_norm": float(np.linalg.norm(centroid)),
                "finite": bool(np.isfinite(centroid).all()),
                "non_zero": bool(float(np.linalg.norm(centroid)) > 0),
            },
            "medoid": {
                "type": "min_mean_cosine_distance",
                "anchor_candidate_id": medoid["anchor_candidate_id"],
                "gallery_row_index": medoid["index"],
                "mean_cosine_distance_to_others": medoid[
                    "mean_cosine_distance_to_others"
                ],
                "tie_break": "approved_anchor_id_ascending",
                "dimension": 512,
            },
            "automatic_members": 0,
            "pseudo_label_members": 0,
            "threshold_selected": False,
            "automatic_gallery_growth": False,
        }
        write_json(tmp / "gallery" / "target_001_gallery_prototypes.json", prototypes)

        # Audit
        np.save(
            tmp / "audit" / "target_001_gallery_pairwise_cosine.npy",
            audit["pairwise"],
        )
        consistency = {
            "schema_version": "reid_target_001_gallery_internal_consistency_v1",
            "target_id": TARGET_ID,
            "member_ids": list(APPROVED_IDS),
            "is_identity_threshold": False,
            "threshold_selected": False,
            "automatic_anchor_removal": False,
            **audit["metrics"],
        }
        write_json(tmp / "audit" / "target_001_gallery_internal_consistency.json", consistency)

        # Runtime
        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "project_head": head,
                "device": "cpu",
                "offline_required": True,
                "network_used": False,
                "conda_env": runtime["conda_env"],
                "sn_reid_commit": runtime["sn_reid_commit"],
                "sample_mp4_decoded": False,
                "sample_mp4_inference": False,
                "yolo_loaded": False,
                "bytetrack_run": False,
                "ocr_run": False,
                "identity_assignments": 0,
                "threshold_selected": False,
                "automatic_gallery_growth": False,
                "approved_crop_source_reads": len(
                    [p for p in crop_read_log if Path(p).stem in APPROVED_IDS]
                ),
                "excluded_crop_source_reads": 0,
                "crop_copies": 0,
                "png_written": 0,
                "mp4_written": 0,
            },
        )
        write_json(
            tmp / "runtime" / "determinism_audit.json",
            embed_result["determinism"],
        )
        shutil.copy2(
            config_path,
            tmp / "effective_configs" / Path(config_path).name,
        )

        contract = {
            "schema_version": "reid_stage5d_b1e_f_gallery_build_contract_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "frozen_anchor_count": 7,
            "embedding_count": 7,
            "embedding_dimension": 512,
            "individual_gallery_members": 7,
            "centroid_count": 1,
            "medoid_count": 1,
            "automatic_members": 0,
            "pseudo_label_members": 0,
            "target_assignments_on_sample_mp4": 0,
            "threshold_selected": False,
            "automatic_gallery_growth": False,
            "approved_exact_ids": list(APPROVED_IDS),
            "excluded_candidate_ids": list(EXCLUDED_IDS),
            "excluded_crop_source_reads": 0,
            "crop_copies": 0,
            "readiness": decision["readiness"],
            "exact_next_gate": decision["exact_next_gate"],
            "osnet_checkpoint_sha256": osnet_sha_before,
            "preprocessing_contract_sha256": pre_sha,
            "two_pass_determinism": True,
            "deployment_readiness": False,
            "reid_success_proven": False,
        }
        write_json(tmp / "stage5d_b1e_f_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_b1e_f_gallery_build_summary_v1",
            "final_status": decision["final_status"],
            "readiness": decision["readiness"],
            "exact_next_gate": decision["exact_next_gate"],
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "frozen_anchor_count": 7,
            "embedding_count": 7,
            "embedding_dimension": 512,
            "embedding_dtype": "float32",
            "individual_gallery_members": 7,
            "centroid_count": 1,
            "medoid_count": 1,
            "medoid_anchor_candidate_id": medoid["anchor_candidate_id"],
            "automatic_members": 0,
            "target_assignments_on_sample_mp4": 0,
            "threshold_selected": False,
            "automatic_gallery_growth": False,
            "approved_exact_ids": list(APPROVED_IDS),
            "occurrence_coverage": sorted(
                {r["source_occurrence_code"] for r in approved}
            ),
            "view_coverage": sorted({r["view_category"] for r in approved}),
            "pairwise_off_diagonal": {
                "min": consistency["off_diagonal_min"],
                "median": consistency["off_diagonal_median"],
                "mean": consistency["off_diagonal_mean"],
                "max": consistency["off_diagonal_max"],
            },
            "outlier_flag_count": len(consistency["outlier_diagnostics"]["flagged"]),
            "two_pass_determinism": embed_result["determinism"],
            "embedding_norm_min": float(np.min(embed_result["norms"])),
            "embedding_norm_max": float(np.max(embed_result["norms"])),
            "osnet_checkpoint_sha256": osnet_sha_before,
            "b1e_e_snapshot_sha256": freeze_info["snapshot_sha256"],
            "external_source_sha256": assets["external_sha256"],
            "sample_sha256": assets["sample_sha256"],
            "sample_mp4_inference": False,
            "identity_assignments": 0,
            "network_used": False,
            "package_environment_changed": False,
        }
        write_json(tmp / "stage5d_b1e_f_summary.json", summary)

        npy_count = sum(1 for p in tmp.rglob("*.npy"))
        if npy_count != 5:
            raise GalleryBuildError(f"artifact budget npy={npy_count} expected 5")
        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")):
            raise GalleryBuildError("crop/media copies forbidden")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_b1e_f_gallery_build_manifest_v1",
            "final_status": decision["final_status"],
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "embeddings_sha256": emb_sha_doc["sha256"],
            "preprocessing_contract_sha256": pre_sha,
            "checkpoint_sha256": osnet_sha_before,
            "b1e_e_snapshot_sha256": freeze_info["snapshot_sha256"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(tmp / "stage5d_b1e_f_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_b1e_f_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_b1e_f_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1e_f_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build target_001 frozen OSNet gallery from 7 approved external anchors."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to target_gallery_build_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except GalleryBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"embeddings={summary['embedding_count']} "
        f"gallery={summary['individual_gallery_members']} "
        f"medoid={summary['medoid_anchor_candidate_id']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
