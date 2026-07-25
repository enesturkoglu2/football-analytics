#!/usr/bin/env python3
"""Stage 5D-F1 — Target 001 independent sample retrieval validation design.

Design/preflight only. Does not run sample OSNet inference, cosine scoring,
retrieval ranking, ground-truth fill, threshold selection, gallery mutation,
YOLO/ByteTrack/OCR, or identity assignment.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_independent_validation_design_config_v1"
APPROVED_IDS = (
    "target_001_ext_anchor_001",
    "target_001_ext_anchor_003",
    "target_001_ext_anchor_004",
    "target_001_ext_anchor_006",
    "target_001_ext_anchor_008",
    "target_001_ext_anchor_011",
    "target_001_ext_anchor_014",
)
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
FINAL_STATUS = (
    "COMPLETED_STAGE5D_F1_TARGET_001_INDEPENDENT_VALIDATION_DESIGN_READY"
)
NEXT_GATE = "STAGE5D-F2_TARGET_001_SAMPLE_GROUND_TRUTH_REVIEW_PACKAGE"
ALLOWED_DIRTY = {
    "scripts/run_reid_target_independent_validation_design.py",
    "configs/reid/target_independent_validation_design_stage5d_target_001.yaml",
    "tests/test_reid_target_independent_validation_design.py",
    "docs/setup/stage5d-target-independent-sample-retrieval-validation-design.md",
}


class ValidationDesignError(RuntimeError):
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
        raise ValidationDesignError("unexpected config schema")
    if not config.get("offline_required"):
        raise ValidationDesignError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise ValidationDesignError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ValidationDesignError("BLOCKED_STAGE5D_F1_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ValidationDesignError("BLOCKED_STAGE5D_F1_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise ValidationDesignError(
                    "BLOCKED_STAGE5D_F1_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise ValidationDesignError("BLOCKED_STAGE5D_F1_GIT_CONTRACT_MISMATCH message")
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
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    if not sidecar.is_file() or not manifest.is_file() or not listing.is_file():
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH snapshot_sha"
        )
    if not listing.read_text(encoding="utf-8").strip():
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH snapshot_listing"
        )
    return actual


def validate_immutable_sources(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    ext = project_root / config["external_enrollment_source"]["path"]
    sample = project_root / config["evaluation_source"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if not ext.is_file() or ext.is_symlink():
        raise ValidationDesignError("external must be regular file")
    if ext.stat().st_size != int(config["external_enrollment_source"]["expected_bytes"]):
        raise ValidationDesignError("external bytes mismatch")
    if sha256_file(ext) != config["external_enrollment_source"]["expected_sha256"]:
        raise ValidationDesignError("external sha mismatch")
    if sample.stat().st_size != int(config["evaluation_source"]["expected_bytes"]):
        raise ValidationDesignError("sample bytes mismatch")
    if sha256_file(sample) != config["evaluation_source"]["expected_sha256"]:
        raise ValidationDesignError("sample sha mismatch")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise ValidationDesignError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise ValidationDesignError("yolo sha mismatch")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise ValidationDesignError("osnet sha mismatch")
    return {
        "external_sha256": config["external_enrollment_source"]["expected_sha256"],
        "sample_sha256": config["evaluation_source"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet_sha256": config["osnet_checkpoint"]["expected_sha256"],
    }


def validate_gallery_v1(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    exp = config["gallery_v1"]
    root = project_root / exp["path"]
    summary = load_json(root / "stage5d_b1e_f_summary.json")
    contract = load_json(root / "stage5d_b1e_f_contract.json")
    if summary.get("final_status") != exp["expected_final_status"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH status"
        )
    if summary.get("readiness") != exp["expected_readiness"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH readiness"
        )
    if summary.get("target_id") != exp["expected_target_id"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH target"
        )
    if summary.get("target_alias") != exp["expected_target_alias"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH alias"
        )
    if int(summary.get("individual_gallery_members")) != int(
        exp["expected_individual_members"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH members"
        )
    if int(summary.get("centroid_count")) != int(exp["expected_centroid_count"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH centroid"
        )
    if int(summary.get("medoid_count")) != int(exp["expected_medoid_count"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH medoid_count"
        )
    if summary.get("medoid_anchor_candidate_id") != exp["expected_medoid_anchor"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH medoid"
        )
    if list(summary.get("approved_exact_ids") or []) != list(APPROVED_IDS):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH approved_ids"
        )
    if list(config["approved_exact_ids"]) != list(APPROVED_IDS):
        raise ValidationDesignError("config approved_exact_ids mismatch")

    emb = np.load(root / "embeddings" / "target_001_anchor_embeddings.npy")
    shape = list(exp["expected_embedding_shape"])
    if list(emb.shape) != shape or str(emb.dtype) != exp["expected_dtype"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH emb_shape"
        )
    if int(np.isnan(emb).sum()) or int(np.isinf(emb).sum()):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH nan_inf"
        )
    if int(np.all(emb == 0, axis=1).sum()):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH zero"
        )
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH l2"
        )
    det = summary.get("two_pass_determinism") or {}
    if det.get("deterministic") is not True:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH determinism"
        )
    if float(det.get("overall_max_absolute_difference", 1)) != float(
        exp["expected_max_absolute_difference"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH max_abs"
        )

    pair = summary.get("pairwise_off_diagonal") or {}
    tol = float(exp["pairwise_tolerance"])
    for key in ("min", "median", "mean", "max"):
        actual = float(pair[key])
        expected = float(exp["expected_pairwise_off_diagonal"][key])
        if abs(actual - expected) > tol:
            raise ValidationDesignError(
                f"BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH pairwise_{key}"
            )

    consistency = load_json(
        root / "audit" / "target_001_gallery_internal_consistency.json"
    )
    flagged = consistency.get("outlier_diagnostics", {}).get("flagged") or []
    if not flagged:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH outlier_missing"
        )
    flag0 = flagged[0]
    if flag0.get("anchor_candidate_id") != exp["diagnostic_outlier_anchor"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH outlier_id"
        )
    if exp["diagnostic_outlier_reason"] not in (flag0.get("flags") or []):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH outlier_reason"
        )
    if flag0.get("removal_allowed") is not False:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH removal"
        )
    if consistency.get("automatic_anchor_removal") is not False:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH auto_remove"
        )

    gallery = np.load(root / "gallery" / "target_001_individual_gallery.npy")
    centroid = np.load(root / "gallery" / "target_001_gallery_centroid.npy")
    medoid = np.load(root / "gallery" / "target_001_gallery_medoid.npy")
    if gallery.shape != emb.shape or not np.allclose(gallery, emb):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH gallery"
        )
    if centroid.shape != (512,) or medoid.shape != (512,):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH proto_shape"
        )
    if contract.get("automatic_gallery_growth") is not False:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH auto_growth"
        )
    if contract.get("threshold_selected") is not False:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH threshold"
        )

    members = [
        json.loads(line)
        for line in (root / "gallery" / "target_001_gallery_members.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(members) != 7:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH member_rows"
        )
    if [m["anchor_candidate_id"] for m in members] != list(APPROVED_IDS):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_GALLERY_CONTRACT_MISMATCH member_order"
        )

    snapshot_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "members": members,
        "embedding_sha256": sha256_file(
            root / "embeddings" / "target_001_anchor_embeddings.npy"
        ),
        "snapshot_sha256": snapshot_sha,
        "diagnostic_outlier": flag0,
    }


def validate_independence(
    project_root: Path,
    config: Mapping[str, Any],
    gallery_info: Mapping[str, Any],
) -> dict[str, Any]:
    b1ea = project_root / config["stage5d_b1e_a_package"]["path"]
    summary_a = load_json(b1ea / "stage5d_b1e_a_summary.json")
    overlap = load_json(
        b1ea / config["stage5d_b1e_a_package"]["overlap_audit_relpath"]
    )
    exp = config["stage5d_b1e_a_package"]
    if bool(summary_a.get("exact_file_duplicate")) != bool(
        exp["expected_exact_file_duplicate"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE exact_dup_summary"
        )
    if bool(overlap.get("exact_file_duplicate")) != bool(
        exp["expected_exact_file_duplicate"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE exact_dup"
        )
    if int(overlap.get("verified_overlapping_pair_count") or 0) != int(
        exp["expected_verified_overlapping_pair_count"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE pairs"
        )
    if int(summary_a.get("verified_overlap_interval_count") or 0) != int(
        exp["expected_verified_overlap_interval_count"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE intervals"
        )
    if overlap.get("source_overlap_decision") != exp["expected_source_overlap_decision"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE decision"
        )
    eval_src = summary_a.get("evaluation_source") or {}
    for key, cfg_key in (
        ("bytes", "expected_bytes"),
        ("frames", "expected_frames"),
        ("width", "expected_width"),
        ("height", "expected_height"),
    ):
        if int(eval_src.get(key)) != int(config["evaluation_source"][cfg_key]):
            raise ValidationDesignError(
                f"BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE sample_{key}"
            )
    if float(eval_src.get("fps")) != float(config["evaluation_source"]["expected_fps"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE sample_fps"
        )
    if abs(
        float(eval_src.get("format_duration_sec"))
        - float(config["evaluation_source"]["expected_duration_sec"])
    ) > 1e-6:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE sample_duration"
        )
    if eval_src.get("sha256") != config["evaluation_source"]["expected_sha256"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE sample_sha_meta"
        )

    gallery_crop_shas = {m["crop_sha256"] for m in gallery_info["members"]}
    if len(gallery_crop_shas) != 7:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE gallery_sha_count"
        )

    # Sample crop SHAs from baseline + segmented crop trees (no decode of sample.mp4).
    sample_crop_roots = [
        project_root / config["sample_baseline_crops"]["path"],
        project_root
        / config["sample_segmented_reid"]["path"]
        / "crops",
    ]
    sample_crop_shas: set[str] = set()
    sample_crop_files = 0
    for root in sample_crop_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file():
                sample_crop_files += 1
                sample_crop_shas.add(sha256_file(path))
    overlap_shas = gallery_crop_shas & sample_crop_shas
    if overlap_shas:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE crop_sha_overlap"
        )

    # Gallery members must not reference sample.mp4 as crop source.
    freeze_csv = (
        project_root
        / config["stage5d_b1e_e_package"]["path"]
        / "anchor_freeze"
        / "target_001_external_approved_anchors_frozen.csv"
    )
    with freeze_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # crop_path is relative under B1E-D external review package, not sample.
            if "sample.mp4" in row.get("crop_path", ""):
                raise ValidationDesignError(
                    "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE gallery_from_sample"
                )
            if row["source_occurrence_code"] not in {"EXT_004", "EXT_183", "EXT_198"}:
                raise ValidationDesignError(
                    "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE non_external_occ"
                )

    forbidden = config["forbidden_sample_seed_segment"]
    if any(forbidden in json.dumps(m) for m in gallery_info["members"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE sample_seed"
        )

    # Stage 5C exclusion contract: sample holdout/discovery must not be gallery.
    exclusion = load_json(
        project_root
        / config["stage5d_a_preflight"]["path"]
        / config["stage5d_a_preflight"]["exclusion_contract"]
    )
    if exclusion.get("holdout_primary_forbidden_for_gallery_enrollment") is not True:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_ENROLLMENT_EVALUATION_LEAKAGE exclusion"
        )

    return {
        "exact_file_duplicate": False,
        "verified_overlapping_frame_pairs": 0,
        "verified_overlap_intervals": 0,
        "enrollment_and_evaluation_independent": True,
        "gallery_crop_sha_count": len(gallery_crop_shas),
        "sample_crop_file_count": sample_crop_files,
        "sample_crop_sha_count": len(sample_crop_shas),
        "gallery_sample_crop_sha_overlap": 0,
        "sample_seed_in_gallery": False,
        "sample_mp4_is_gallery_crop_source": False,
        "stage5c_sample_batches_in_gallery": False,
        "b1e_a_overlap_decision": overlap.get("source_overlap_decision"),
        "evaluation_source_meta": eval_src,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_sample_universe(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    seg_cfg = config["sample_segmented_reid"]
    seg_root = project_root / seg_cfg["path"]
    emb_path = seg_root / seg_cfg["embeddings_npz"]
    idx_path = seg_root / seg_cfg["embedding_index"]
    if sha256_file(emb_path) != seg_cfg["expected_embeddings_sha256"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH emb_sha"
        )
    if sha256_file(idx_path) != seg_cfg["expected_index_sha256"]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH idx_sha"
        )

    preflight = load_json(
        project_root
        / config["stage5d_a_preflight"]["path"]
        / config["stage5d_a_preflight"]["embedding_preflight"]
    )
    if int(preflight["embedded_segment_count"]) != int(seg_cfg["expected_embedded"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH pre_embedded"
        )
    if int(preflight["no_embedding_segment_count"]) != int(
        seg_cfg["expected_no_embedding"]
    ):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH pre_noemb"
        )

    payload = np.load(emb_path)
    # NPZ key discovery
    key = "embeddings" if "embeddings" in payload.files else payload.files[0]
    vectors = payload[key]
    if list(vectors.shape) != [150, 512]:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH shape"
        )
    if vectors.dtype != np.float32:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH dtype"
        )
    if int(np.isnan(vectors).sum()) or int(np.isinf(vectors).sum()):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH nan_inf"
        )
    if int(np.all(vectors == 0, axis=1).sum()):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH zero"
        )

    index_rows = _load_jsonl(idx_path)
    if len(index_rows) != int(seg_cfg["expected_total_segments"]):
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH total"
        )
    embedded = [r for r in index_rows if r.get("embedding_available") is True]
    no_emb = [r for r in index_rows if r.get("embedding_available") is not True]
    if len(embedded) != 150 or len(no_emb) != 141:
        raise ValidationDesignError(
            "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH counts"
        )

    # Canonical preprocessing must match gallery OSNet contract.
    for row in embedded:
        prep = row.get("preprocessing") or {}
        if prep.get("color_conversion") != "bgr_to_rgb":
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH prep_rgb"
            )
        if list(prep.get("resize_hw") or []) != [256, 128]:
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH prep_hw"
            )
        if row.get("checkpoint_sha256") != config["osnet_checkpoint"]["expected_sha256"]:
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH ckpt"
            )
        if int(row.get("embedding_dimension") or 0) != 512:
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH dim"
            )

    track_segments = {
        r["segment_id"]: r
        for r in _load_jsonl(
            project_root
            / config["sample_segment_view"]["path"]
            / config["sample_segment_view"]["track_segments"]
        )
    }
    baseline_crops = {
        r["crop_id"]: r
        for r in _load_jsonl(
            project_root
            / config["sample_baseline_crops"]["path"]
            / config["sample_baseline_crops"]["crop_manifest"]
        )
    }
    segment_crops = {
        r["crop_id"]: r
        for r in _load_jsonl(seg_root / seg_cfg["segment_crop_manifest"])
    }
    crop_lookup = {**baseline_crops, **segment_crops}

    gid_map = {
        int(r["raw_track_id"]): r
        for r in _load_jsonl(
            project_root
            / config["documented_link_overlay"]["path"]
            / config["documented_link_overlay"]["global_id_map"]
        )
    }

    split_root = project_root / config["stage5c_clean_split"]["path"]
    split_membership: dict[str, str] = {}
    leakage_by_segment: dict[str, str] = {}
    near_dup_by_segment: dict[str, str] = {}
    for batch, rel in config["stage5c_clean_split"]["manifests"].items():
        for row in _load_jsonl(split_root / rel):
            sid = row.get("segment_id")
            if sid:
                split_membership[sid] = batch
                if row.get("leakage_group_id"):
                    leakage_by_segment[sid] = row["leakage_group_id"]
                if row.get("near_duplicate_cluster_id"):
                    near_dup_by_segment[sid] = row["near_duplicate_cluster_id"]
    for row in _load_jsonl(split_root / config["stage5c_clean_split"]["leakage_groups"]):
        for sid in row.get("segment_ids") or []:
            leakage_by_segment.setdefault(sid, row["leakage_group_id"])

    eligibility = {
        r["segment_id"]: r
        for r in load_json(
            project_root
            / config["stage5d_a_preflight"]["path"]
            / config["stage5d_a_preflight"]["source_eligibility_audit"]
        ).get("rows")
        or []
    }

    scoreable_rows: list[dict[str, Any]] = []
    for row in embedded:
        sid = row["segment_id"]
        assert_no_path_traversal(sid) if "/" in sid else None
        ts = track_segments.get(sid) or {}
        crop_ids = list(row.get("crop_ids") or [])
        if not crop_ids:
            raise ValidationDesignError(
                f"BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH no_crop {sid}"
            )
        rep_id = crop_ids[0]
        crop = crop_lookup.get(rep_id)
        if crop is None:
            raise ValidationDesignError(
                f"BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH crop_missing {rep_id}"
            )
        rel = crop["crop_relative_path"]
        assert_no_path_traversal(rel)
        if crop.get("segment_id"):
            crop_root = seg_root
        else:
            crop_root = project_root / config["sample_baseline_crops"]["path"]
        crop_abs = (crop_root / rel).resolve()
        try:
            crop_abs.relative_to(crop_root.resolve())
        except ValueError as exc:
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH crop_escape"
            ) from exc
        if not crop_abs.is_file():
            raise ValidationDesignError(
                f"BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH crop_file {rel}"
            )
        crop_sha = sha256_file(crop_abs)
        emb_row = int(row["embedding_row"])
        vec = vectors[emb_row]
        if int(row["embedding_dimension"]) != 512 or vec.shape != (512,):
            raise ValidationDesignError(
                "BLOCKED_STAGE5D_F1_SAMPLE_REID_UNIVERSE_MISMATCH vec"
            )
        raw_tid = int(row["raw_track_id"])
        gid = gid_map.get(raw_tid) or {}
        component_tracks = list(gid.get("component_member_track_ids") or [raw_tid])
        source_type = (
            "reused"
            if "reused" in str(row.get("representation_source") or "")
            else "recomputed"
            if "recomputed" in str(row.get("representation_source") or "")
            else "other"
        )
        elig = eligibility.get(sid) or {}
        scoreable_rows.append(
            {
                "segment_id": sid,
                "raw_track_id": raw_tid,
                "frame_range": [
                    int(ts.get("first_observation_frame", row.get("first_frame") or 0)),
                    int(ts.get("last_observation_frame", row.get("last_frame") or 0)),
                ],
                "observation_count": int(ts.get("observation_count") or 0),
                "representative_crop_id": rep_id,
                "representative_crop_path": rel,
                "representative_crop_sha256": crop_sha,
                "embedding_artifact_path": str(
                    Path(seg_cfg["path"]) / seg_cfg["embeddings_npz"]
                ),
                "embedding_artifact_sha256": seg_cfg["expected_embeddings_sha256"],
                "embedding_row": emb_row,
                "embedding_vector_sha256": row["embedding_sha256"],
                "embedding_shape": [512],
                "embedding_dtype": "float32",
                "embedding_finite": True,
                "embedding_non_zero": True,
                "source_type": source_type,
                "representation_source": row.get("representation_source"),
                "documented_link_component_id": (
                    f"doc_component_{int(gid.get('global_candidate_id', raw_tid))}"
                ),
                "documented_component_member_track_ids": component_tracks,
                "global_candidate_id": gid.get("global_candidate_id"),
                "stage5c_split_membership": split_membership.get(sid, "not_in_clean_split"),
                "leakage_group_id": leakage_by_segment.get(sid),
                "near_duplicate_cluster_id": near_dup_by_segment.get(sid),
                "exact_crop_sha_group": crop_sha,
                "exact_duplicate_embedding_group": row["embedding_sha256"],
                "temporal_source_component": (
                    f"raw_track_{raw_tid}_"
                    f"{int(ts.get('first_observation_frame', 0))}_"
                    f"{int(ts.get('last_observation_frame', 0))}"
                ),
                "source_observation_component": (
                    f"obs_component_track_{raw_tid}"
                ),
                "quality_score_representative": crop.get("quality_score"),
                "visibility_or_eligibility_status": elig.get("eligibility_status"),
                "scoreable": True,
                "embedding_recompute_planned": False,
                "manual_ground_truth_decision": None,
                "similarity_computed": False,
            }
        )

    no_embedding_inventory = [
        {
            "segment_id": r["segment_id"],
            "raw_track_id": int(r["raw_track_id"]),
            "scoreable": False,
            "embedding_available": False,
            "no_embedding_reason": r.get("no_embedding_reason"),
            "automatic_negative": False,
            "include_in_f2_ground_truth_review_package": False,
            "include_in_f2_unscoreable_inventory": True,
            "embedding_recompute_planned": False,
        }
        for r in no_emb
    ]

    summary = {
        "schema_version": "reid_target_001_sample_universe_summary_v1",
        "total_segment_units": 291,
        "scoreable_embedded_units": 150,
        "no_embedding_units": 141,
        "embedding_dimension": 512,
        "embedding_dtype": "float32",
        "nan_count": 0,
        "inf_count": 0,
        "zero_vector_count": 0,
        "embedding_recompute": False,
        "sample_similarity_rows": 0,
        "retrieval_rankings": 0,
        "manual_ground_truth_decisions": 0,
        "embeddings_npz_sha256": seg_cfg["expected_embeddings_sha256"],
        "embedding_index_sha256": seg_cfg["expected_index_sha256"],
        "source_type_distribution": {
            "reused": sum(1 for r in scoreable_rows if r["source_type"] == "reused"),
            "recomputed": sum(
                1 for r in scoreable_rows if r["source_type"] == "recomputed"
            ),
        },
        "no_embedding_policy": {
            "scoreable": False,
            "automatic_negative": False,
            "f2_ground_truth_review_included": False,
            "f2_unscoreable_inventory_included": True,
            "recompute_in_f1": False,
        },
    }
    return {
        "scoreable_rows": scoreable_rows,
        "no_embedding_inventory": no_embedding_inventory,
        "summary": summary,
        "vectors_immutable_sha256": seg_cfg["expected_embeddings_sha256"],
    }


def gallery_validation_contract(gallery_info: Mapping[str, Any], independence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_gallery_validation_contract_v1",
        "gallery_root": "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1",
        "final_status": gallery_info["summary"]["final_status"],
        "readiness": gallery_info["summary"]["readiness"],
        "target_id": TARGET_ID,
        "target_alias": TARGET_ALIAS,
        "individual_gallery_members": 7,
        "centroid_count": 1,
        "medoid_count": 1,
        "medoid_anchor_candidate_id": "target_001_ext_anchor_004",
        "approved_exact_ids": list(APPROVED_IDS),
        "embedding_shape": [7, 512],
        "dtype": "float32",
        "l2_normalized": True,
        "two_pass_deterministic": True,
        "max_absolute_difference": 0.0,
        "diagnostic_outlier": {
            "anchor_candidate_id": "target_001_ext_anchor_014",
            "reason": "lowest_mean_and_centroid_similarity",
            "removed": False,
            "removal_allowed": False,
        },
        "gallery_mutation_allowed": False,
        "automatic_gallery_growth": False,
        "snapshot_sha256": gallery_info["snapshot_sha256"],
        "independence": independence,
    }


def ground_truth_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_sample_ground_truth_contract_v1",
        "label_blind": True,
        "decisions_filled_in_f1": False,
        "similarity_scores_shown_to_reviewer": False,
        "human_before_similarity": True,
        "allowed_visible_fields": [
            "temporal_full_frame_context",
            "representative_full_body_crop",
            "segment_frame_time_range",
            "neutral_evaluation_candidate_code",
            "human_visible_jersey_number_in_natural_image",
        ],
        "forbidden_visible_fields": [
            "gallery_similarity",
            "centroid_similarity",
            "medoid_similarity",
            "osnet_score",
            "rank",
            "model_identity_prediction",
            "automated_ocr_parseq_prediction",
            "automated_jersey_confidence",
            "expected_target_suggestion",
            "previous_similarity_derived_candidate_labels",
        ],
        "decision_vocabulary": [
            "target_occurrence_yes",
            "target_occurrence_no",
            "uncertain",
            "invalid",
            "multi_person_ambiguous",
            "non_player",
        ],
        "tri_state_fields": [
            "manual_same_target_as_target_001",
            "manual_identity_continuity_observed",
            "manual_crop_valid",
            "manual_target_dominant",
            "manual_human_verified_number_seen",
        ],
        "tri_state_values": ["yes", "no", "uncertain"],
        "policy": {
            "only_human_target_occurrence_yes_is_positive": True,
            "only_human_target_occurrence_no_or_non_player_enter_true_negative_set": True,
            "unreviewed_is_not_negative": True,
            "uncertain_invalid_multi_person_excluded_from_metrics": True,
            "human_number_seen_is_auxiliary_evidence_only": True,
            "automated_jersey_is_not_ground_truth": True,
            "track_or_global_id_alone_is_not_true_identity": True,
        },
        "f2_coverage": {
            "scoreable_segments_for_ground_truth_review": 150,
            "no_embedding_segments_in_ground_truth_review": False,
            "no_embedding_segments_in_unscoreable_inventory": True,
            "no_embedding_automatic_negative": False,
            "neutral_codes": "SAMPLE_EVAL_001..SAMPLE_EVAL_N",
            "max_items_per_contact_sheet": 12,
            "contact_sheet_ordering": "similarity_blind_deterministic",
            "raw_track_global_id_hidden_on_visuals": True,
        },
        "manual_decisions_in_f1": 0,
    }


def scoring_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_retrieval_scoring_contract_v1",
        "executed_in_f1": False,
        "query_embeddings": {
            "source": "existing_sample_segment_embeddings",
            "count": 150,
            "dimension": 512,
            "l2_normalized": True,
            "recompute_forbidden": True,
        },
        "gallery": {
            "source": "gallery_v1_immutable",
            "individual_members": 7,
            "mutation_forbidden": True,
        },
        "primary_retrieval_score": {
            "name": "max_individual_cosine",
            "definition": (
                "maximum cosine similarity between the query embedding and the "
                "7 individual frozen gallery embeddings"
            ),
            "used_for_primary_ranking": True,
        },
        "secondary_diagnostic_scores": [
            {
                "name": "top3_mean_individual_cosine",
                "definition": "mean of the three highest individual cosine scores",
                "top_k_preregistered": 3,
                "top_k_frozen_before_seeing_sample_scores": True,
            },
            {
                "name": "centroid_cosine",
                "definition": "cosine with frozen gallery centroid",
            },
            {
                "name": "medoid_cosine",
                "definition": "cosine with frozen gallery medoid",
            },
            {
                "name": "mean_individual_cosine",
                "definition": "arithmetic mean of all 7 individual cosine scores",
            },
        ],
        "forbidden": [
            "change_formula_after_seeing_sample_scores",
            "remove_anchor_014_after_scoring",
            "promote_best_prototype_to_primary_after_the_fact",
            "add_team_or_jersey_score_to_cosine",
            "select_threshold",
            "assign_identity",
        ],
        "similarity_rows_in_f1": 0,
        "ranking_rows_in_f1": 0,
    }


def leakage_grouping_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_leakage_grouping_contract_v1",
        "primary_evaluation_unit": "sample_segmented_reid_unit",
        "reporting_levels": ["segment_level", "independent_component_level"],
        "grouping_keys": [
            "segment_id",
            "raw_track_id",
            "documented_link_component",
            "exact_crop_sha",
            "exact_duplicate_group",
            "near_duplicate_component",
            "overlapping_temporal_source_window",
            "source_observation_component",
        ],
        "component_score_aggregation": {
            "primary": "max_segment_retrieval_score_within_component",
            "secondary_diagnostic": "median_segment_retrieval_score_within_component",
        },
        "component_label_rules": {
            "positive_if_any_human_confirmed_target_positive": True,
            "negative_only_if_all_reviewed_valid_members_are_target_no": True,
            "mixed_or_uncertain_excluded_from_evaluation": True,
        },
        "computed_with_scores_in_f1": False,
        "lineage_design_only_in_f1": True,
    }


def metric_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_target_001_retrieval_metric_contract_v1",
        "executed_in_f1": False,
        "threshold_required": False,
        "segment_level_primary_metrics": [
            "positive_count",
            "negative_count",
            "excluded_uncertain_count",
            "every_positive_query_rank",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "Mean_Reciprocal_Rank",
            "Average_Precision",
            "best_positive_score",
            "worst_positive_score",
            "best_negative_score",
            "positive_negative_score_margin_min_pos_minus_max_neg",
        ],
        "independent_component_level_primary_metrics": [
            "positive_component_count",
            "negative_component_count",
            "target_positive_component_ranks",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "MRR",
            "AP",
            "minimum_positive_component_score",
            "maximum_negative_component_score",
            "separation_margin",
        ],
        "secondary_diagnostics": {
            "AUROC": {
                "support_gate": {"positive_ge": 2, "negative_ge": 20},
                "report_only_if_support_met": True,
            },
            "AUPRC": {
                "support_gate": {"positive_ge": 2, "negative_ge": 20},
                "report_only_if_support_met": True,
            },
            "other": [
                "score_distributions",
                "centroid_medoid_top3_score_comparisons",
                "source_type_reused_recomputed_diagnostics",
                "view_quality_diagnostics_if_ground_truth_allows",
            ],
        },
        "descriptive_outcomes": {
            "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL": {
                "min_positive_scoreable_segments": 2,
                "min_human_confirmed_negative_scoreable_segments": 20,
                "min_component_level_positives": 2,
                "Recall@5": 1.0,
                "component_Recall@5": 1.0,
                "AP_ge": 0.80,
                "component_AP_ge": 0.80,
                "min_positive_score_gt_max_negative_score": True,
            },
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
        "outcome_is_not": [
            "deployment_approval",
            "identity_assignment_permission",
            "threshold",
            "automatic_gallery_growth_permission",
        ],
        "threshold_policy": {
            "acceptance_threshold_selected": False,
            "calibrated_probability": False,
            "deployment_threshold": False,
            "automatic_identity_assignment": False,
            "gallery_mutation": False,
            "sample_used_only_for_independent_retrieval_ranking_evaluation": True,
            "future_threshold_requires_separate_calibration_source_or_frozen_protocol": True,
        },
    }


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f"_tmp_stage5d_f1_validation_design_{final_dir.name}_{token}"
    if tmp.exists():
        raise ValidationDesignError("FAILED_STAGE5D_F1_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ValidationDesignError("FAILED_STAGE5D_F1_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise ValidationDesignError("FAILED_STAGE5D_F1_ATOMIC_OUTPUT final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    assets = validate_immutable_sources(project_root, config)
    gallery_info = validate_gallery_v1(project_root, config)
    independence = validate_independence(project_root, config, gallery_info)
    universe = build_sample_universe(project_root, config)

    tmp = create_temp_root(final_dir)
    try:
        for sub in (
            "gallery_validation",
            "sample_universe",
            "ground_truth_design",
            "scoring_design",
            "metric_design",
            "runtime",
            "effective_configs",
        ):
            (tmp / sub).mkdir(parents=True, exist_ok=False)

        write_json(
            tmp / "gallery_validation" / "target_001_gallery_validation_contract.json",
            gallery_validation_contract(gallery_info, independence),
        )
        with (
            tmp / "sample_universe" / "target_001_sample_scoreable_universe.jsonl"
        ).open("w", encoding="utf-8") as handle:
            for row in universe["scoreable_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            tmp / "sample_universe" / "target_001_sample_universe_summary.json",
            universe["summary"],
        )
        with (
            tmp / "sample_universe" / "target_001_sample_no_embedding_inventory.jsonl"
        ).open("w", encoding="utf-8") as handle:
            for row in universe["no_embedding_inventory"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_json(
            tmp / "ground_truth_design" / "target_001_sample_ground_truth_contract.json",
            ground_truth_contract(),
        )
        write_json(
            tmp / "scoring_design" / "target_001_retrieval_scoring_contract.json",
            scoring_contract(),
        )
        write_json(
            tmp / "scoring_design" / "target_001_leakage_grouping_contract.json",
            leakage_grouping_contract(),
        )
        write_json(
            tmp / "metric_design" / "target_001_retrieval_metric_contract.json",
            metric_contract(),
        )

        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "project_head": head,
                "offline_required": True,
                "network_used": False,
                "sample_mp4_decoded": False,
                "sample_mp4_inference": False,
                "osnet_loaded": False,
                "yolo_loaded": False,
                "bytetrack_run": False,
                "ocr_run": False,
                "sample_similarity_rows": 0,
                "retrieval_rankings": 0,
                "manual_ground_truth_decisions": 0,
                "threshold_selected": False,
                "identity_assignments": 0,
                "automatic_gallery_growth": False,
                "gallery_mutation": False,
                "embedding_recompute": False,
                "new_embedding_rows": 0,
                "png_written": 0,
                "mp4_written": 0,
            },
        )
        shutil.copy2(config_path, tmp / "effective_configs" / Path(config_path).name)

        contract = {
            "schema_version": "reid_stage5d_f1_independent_validation_design_contract_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "gallery_members": 7,
            "sample_ground_truth_decisions": 0,
            "sample_similarity_rows": 0,
            "retrieval_rankings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "gallery_mutation": False,
            "embedding_recompute": False,
            "label_blind_ground_truth": True,
            "primary_score": "max_individual_cosine",
            "exact_next_gate": NEXT_GATE,
            "enrollment_evaluation_independent": True,
            "scoreable_sample_units": 150,
            "no_embedding_sample_units": 141,
        }
        write_json(tmp / "stage5d_f1_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f1_independent_validation_design_summary_v1",
            "final_status": FINAL_STATUS,
            "exact_next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "gallery_members": 7,
            "gallery_snapshot_sha256": gallery_info["snapshot_sha256"],
            "sample_ground_truth_decisions": 0,
            "sample_similarity_rows": 0,
            "retrieval_rankings": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "automatic_gallery_growth": False,
            "scoreable_sample_units": 150,
            "no_embedding_sample_units": 141,
            "sample_embedding_dimension": 512,
            "enrollment_evaluation_independent": True,
            "verified_overlapping_frame_pairs": 0,
            "primary_retrieval_score": "max_individual_cosine",
            "secondary_scores": [
                "top3_mean_individual_cosine",
                "centroid_cosine",
                "medoid_cosine",
                "mean_individual_cosine",
            ],
            "external_source_sha256": assets["external_sha256"],
            "sample_sha256": assets["sample_sha256"],
            "osnet_checkpoint_sha256": assets["osnet_sha256"],
            "sample_embeddings_sha256": universe["vectors_immutable_sha256"],
            "network_used": False,
            "package_environment_changed": False,
        }
        write_json(tmp / "stage5d_f1_summary.json", summary)

        if list(tmp.rglob("*.png")) or list(tmp.rglob("*.mp4")) or list(tmp.rglob("*.npy")):
            raise ValidationDesignError("artifact budget violated")

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f1_independent_validation_design_manifest_v1",
            "final_status": FINAL_STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "gallery_snapshot_sha256": gallery_info["snapshot_sha256"],
            "sample_embeddings_sha256": universe["vectors_immutable_sha256"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(tmp / "stage5d_f1_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f1_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f1_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f1_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5D-F1 independent sample retrieval validation design/preflight "
            "for target_001."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to target_independent_validation_design_stage5d_target_001.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config))
    except ValidationDesignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"gallery_members={summary['gallery_members']} "
        f"scoreable={summary['scoreable_sample_units']} "
        f"similarity_rows={summary['sample_similarity_rows']} "
        f"next_gate={summary['exact_next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
