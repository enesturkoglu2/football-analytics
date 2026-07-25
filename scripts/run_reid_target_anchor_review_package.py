#!/usr/bin/env python3
"""Stage 5D-B — freeze target_001 definition + label-blind anchor review package.

No gallery membership, similarity ranking, OCR inference, prototypes, or
identity assignment. Anchor selections are not final gallery membership.
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
CONFIG_SCHEMA = "reid_target_anchor_review_config_v1"
ALLOWED_MANUAL_DECISIONS = (
    "target_anchor_yes",
    "target_anchor_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
ANNOTATION_FIELDS = (
    "anchor_candidate_id",
    "target_id",
    "segment_id",
    "raw_track_id",
    "frame_index",
    "source_crop_path",
    "source_crop_sha256",
    "manual_anchor_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)


class AnchorPackageError(RuntimeError):
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
        raise AnchorPackageError("unexpected config schema")
    if not config.get("offline_required"):
        raise AnchorPackageError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    parts = Path(rel).parts
    if ".." in parts or rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        raise AnchorPackageError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed_paths = {
        "scripts/run_reid_target_anchor_review_package.py",
        "configs/reid/target_anchor_review_stage5d_target_001.yaml",
        "tests/test_reid_target_anchor_review_package.py",
        "docs/setup/stage5d-target-definition-and-anchor-review-package.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed_paths or line[:2] not in {
                "??",
                " M",
                "M ",
                "A ",
                "AM",
                "A",
            }:
                # tolerate exact "A path" / status variants
                if path in allowed_paths and line[:1] in {"?", "M", "A"}:
                    continue
                raise AnchorPackageError(
                    "BLOCKED_STAGE5D_B_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add Stage 5D target gallery preflight":
        raise AnchorPackageError("BLOCKED_STAGE5D_B_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path, config: Mapping[str, Any]) -> str:
    if not snapshot_path.is_file():
        raise AnchorPackageError("BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH snapshot_sidecar"
        )
    sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = load_json(manifest)
    man_sha = str(man.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (sidecar_sha == man_sha == actual):
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH snapshot_sha_mismatch"
        )
    prefix = str(config["stage5d_a_preflight"]["snapshot_sha256_prefix"])
    suffix = str(config["stage5d_a_preflight"]["snapshot_sha256_suffix"])
    if not (actual.startswith(prefix) and actual.endswith(suffix)):
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH snapshot_prefix_suffix"
        )
    return actual


def validate_stage5d_a_preflight(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5d_a_preflight"]["path"]
    summary = load_json(root / "stage5d_preflight_summary.json")
    emb = load_json(root / "stage5d_embedding_preflight.json")
    design = load_json(root / "stage5d_gallery_design_contract.json")
    tpl = load_json(root / "templates" / "target_definition_template.json")
    accepted = set(config["stage5d_a_preflight"]["accepted_final_statuses"])
    status = summary.get("final_status")
    if status not in accepted:
        raise AnchorPackageError(
            f"BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH status={status!r}"
        )
    checks = {
        "embedded_segments": int(summary.get("embedded_segments") or emb.get("embedded_segment_count")),
        "no_embedding_segments": int(
            summary.get("no_embedding_segments") or emb.get("no_embedding_segment_count")
        ),
        "nan_count": int(emb.get("nan_count")),
        "inf_count": int(emb.get("inf_count")),
        "zero_vector_count": int(emb.get("zero_vector_count")),
        "gallery_members": int(summary.get("gallery_members")),
        "prototypes": int(summary.get("prototypes")),
        "identity_assignments": int(summary.get("identity_assignments")),
        "similarity_ranking_rows": int(summary.get("similarity_ranking_rows") or 0),
    }
    expected = {
        "embedded_segments": 150,
        "no_embedding_segments": 141,
        "nan_count": 0,
        "inf_count": 0,
        "zero_vector_count": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "similarity_ranking_rows": 0,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise AnchorPackageError(
                f"BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH {key}={checks[key]}"
            )
    dim = emb.get("shape")
    if not (isinstance(dim, list) and len(dim) == 2 and dim[1] == 512):
        # also accept embedding_dimension
        if int(emb.get("embedding_dimension") or 0) != 512 and not (
            isinstance(emb.get("shape"), list) and emb["shape"][-1] == 512
        ):
            raise AnchorPackageError(
                "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH embedding_dim"
            )
    if tpl.get("target_id") != "" or tpl.get("target_definition_frozen") is not False:
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH template_not_blank"
        )
    if design.get("automatic_gallery_growth") is not False:
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH automatic_growth"
        )
    if design.get("pseudo_label_enrollment") is not False:
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH pseudo_label"
        )
    if design.get("unknown_identity_preserved") is not True:
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_PREFLIGHT_CONTRACT_MISMATCH unknown"
        )
    snap = Path(config["stage5d_a_preflight"]["snapshot_path"])
    snap_sha = resolve_snapshot_sha(snap, config)
    n, listing = listing_sha(root)
    return {
        "path": config["stage5d_a_preflight"]["path"],
        "final_status": status,
        "listing_file_count": n,
        "listing_sha256": listing,
        "snapshot_sha256": snap_sha,
        "checks": checks,
        "target_template_blank": True,
        "automatic_gallery_growth": False,
        "pseudo_label_enrollment": False,
        "unknown_identity_preserved": True,
    }


def validate_stage5c_closure(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5c_closure"]["path"]
    policy = load_json(root / config["stage5c_closure"]["policy_json"])
    required = {
        "stage5c_status": "closed",
        "automated_parseq_jersey_evidence_enabled": False,
        "automated_parseq_identity_assignment_allowed": False,
        "automated_parseq_identity_veto_allowed": False,
        "automated_parseq_gallery_enrollment_allowed": False,
        "stage5e_automated_jersey_channel_mode": "diagnostic_only",
        "discovery_reserve_opened": False,
        "holdout_reserve_opened": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise AnchorPackageError(
                f"BLOCKED_STAGE5D_B_STAGE5C_CLOSURE_MISMATCH {key}={policy.get(key)!r}"
            )
    return {"path": config["stage5c_closure"]["path"], "policy": required}


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    for path, label in ((video, "video"), (yolo, "yolo"), (osnet, "osnet")):
        if not path.is_file() or path.is_symlink():
            raise AnchorPackageError(f"BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY {label}")
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY video_bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY video_sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY yolo_bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY yolo_sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY osnet_sha")
    return {
        "source_video_sha256": config["source_video"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet_sha256": config["osnet_checkpoint"]["expected_sha256"],
        "inference_run": False,
    }


def load_crop_catalog(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Map crop_id -> metadata + absolute path for baseline and recomputed crops."""
    catalog: dict[str, dict[str, Any]] = {}
    baseline_root = project_root / config["upstream"]["baseline_crops_root"]
    baseline_man = baseline_root / "crop_manifest.jsonl"
    if not baseline_man.is_file():
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY missing_baseline_crop_manifest"
        )
    for line in baseline_man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = str(row["crop_relative_path"])
        path = baseline_root / rel
        catalog[str(row["crop_id"])] = {
            **row,
            "source_type": "reused_baseline",
            "absolute_path": path,
            "crop_relative_path": rel,
        }

    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    seg_man = seg_root / "segment_crop_manifest.jsonl"
    if seg_man.is_file():
        for line in seg_man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rel = str(row["crop_relative_path"])
            path = seg_root / rel
            catalog[str(row["crop_id"])] = {
                **row,
                "source_type": "recomputed_manual_segment",
                "absolute_path": path,
                "crop_relative_path": rel,
            }
    return catalog


def load_stage5c_exclusion_sets(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, set[str]]:
    split = project_root / config["upstream"]["canonical_split"]
    keys: dict[str, set[str]] = {
        "segment_id": set(),
        "raw_track_id": set(),
        "crop_id": set(),
        "source_crop_path": set(),
        "source_crop_sha256": set(),
        "exact_duplicate_group": set(),
        "near_duplicate_component": set(),
        "documented_link_component": set(),
        "temporal_source_window": set(),
        "frame_identity": set(),
        "split_item_id": set(),
    }
    for batch in (
        "discovery_primary",
        "discovery_reserve",
        "holdout_primary",
        "holdout_reserve",
    ):
        man = split / batch / f"{batch}_manifest.jsonl"
        if not man.is_file():
            raise AnchorPackageError(
                f"BLOCKED_STAGE5D_B_GALLERY_EVALUATION_EXCLUSION missing {batch}"
            )
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            keys["split_item_id"].add(str(row["split_item_id"]))
            keys["segment_id"].add(str(row.get("segment_id") or ""))
            keys["raw_track_id"].add(str(row.get("raw_track_id") or ""))
            keys["crop_id"].add(str(row.get("crop_id") or ""))
            keys["source_crop_path"].add(str(row.get("source_crop_path") or ""))
            keys["source_crop_sha256"].add(str(row.get("source_crop_sha256") or ""))
            if row.get("leakage_group_id") not in (None, ""):
                keys["exact_duplicate_group"].add(str(row["leakage_group_id"]))
            if row.get("near_duplicate_cluster_id") not in (None, ""):
                keys["near_duplicate_component"].add(
                    str(row["near_duplicate_cluster_id"])
                )
            if row.get("documented_global_candidate_id") not in (None, ""):
                keys["documented_link_component"].add(
                    str(row["documented_global_candidate_id"])
                )
            if row.get("timeline_bin") not in (None, ""):
                keys["temporal_source_window"].add(str(row["timeline_bin"]))
            keys["frame_identity"].add(
                f"{row.get('segment_id')}:{row.get('frame_index')}"
            )
    return {k: {x for x in v if x} for k, v in keys.items()}


def load_universe_by_segment(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    path = (
        project_root
        / config["upstream"]["visibility_universe"]
        / "clean_review_universe"
        / "clean_review_items.jsonl"
    )
    by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY missing_universe"
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_seg[str(row["segment_id"])].append(row)
    return by_seg


def audit_embeddings(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    npz_path = seg_root / "segment_embeddings.npz"
    index_path = seg_root / "segment_embedding_index.jsonl"
    data = np.load(npz_path)
    vectors = data["vectors"]
    segment_ids = [str(x) for x in data["segment_ids"].tolist()]
    if vectors.shape != (150, 512):
        raise AnchorPackageError(
            f"BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY emb_shape={vectors.shape}"
        )
    if int(np.isnan(vectors).sum()) or int(np.isinf(vectors).sum()):
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY naninf")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if int((norms == 0).sum()):
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY zero")
    if len(segment_ids) != len(set(segment_ids)):
        raise AnchorPackageError("BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY dup_id")

    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedded = [r for r in rows if r.get("embedding_available")]
    no_emb = [r for r in rows if not r.get("embedding_available")]
    if len(embedded) != 150 or len(no_emb) != 141:
        raise AnchorPackageError(
            "BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY coverage"
        )
    meta = {
        "embedding_artifact_path": str(
            Path(config["upstream"]["stage5_replay"])
            / config["upstream"]["segmented_reid_subdir"]
            / "segment_embeddings.npz"
        ),
        "embedding_artifact_sha256": sha256_file(npz_path),
        "metadata_artifact_path": str(
            Path(config["upstream"]["stage5_replay"])
            / config["upstream"]["segmented_reid_subdir"]
            / "segment_embedding_index.jsonl"
        ),
        "metadata_artifact_sha256": sha256_file(index_path),
        "embedded_segment_count": 150,
        "no_embedding_segment_count": 141,
        "embedding_dimension": 512,
        "nan_count": 0,
        "inf_count": 0,
        "zero_vector_count": 0,
    }
    return embedded, no_emb, meta


def exclusion_reasons_for_segment(
    row: Mapping[str, Any],
    *,
    excl: Mapping[str, set[str]],
    catalog: Mapping[str, Mapping[str, Any]],
    universe_by_seg: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    reasons: list[str] = []
    sid = str(row["segment_id"])
    raw = str(row["raw_track_id"])
    if sid in excl["segment_id"]:
        reasons.append("stage5c_segment_id")
    if raw in excl["raw_track_id"]:
        reasons.append("stage5c_raw_track_id")

    for cid in row.get("crop_ids") or []:
        cid_s = str(cid)
        if cid_s in excl["crop_id"]:
            reasons.append("stage5c_crop_id")
            break
        crop = catalog.get(cid_s)
        if crop is None:
            continue
        path = Path(crop["absolute_path"])
        if path.is_file():
            sha = sha256_file(path)
            if sha in excl["source_crop_sha256"]:
                reasons.append("stage5c_source_crop_sha256")
                break
            if str(path) in excl["source_crop_path"]:
                reasons.append("stage5c_source_crop_path")
                break
            frame_id = f"{sid}:{crop.get('frame_index')}"
            if frame_id in excl["frame_identity"]:
                reasons.append("stage5c_frame_identity")
                break

    # Leakage keys from visibility/clean-universe items for this segment.
    for item in universe_by_seg.get(sid, []):
        nd = item.get("near_duplicate_cluster_id")
        if nd not in (None, "") and str(nd) in excl["near_duplicate_component"]:
            reasons.append("stage5c_near_duplicate_component")
            break
        lg = item.get("leakage_group_id")
        if lg not in (None, "") and str(lg) in excl["exact_duplicate_group"]:
            reasons.append("stage5c_exact_duplicate_group")
            break
        dg = item.get("documented_global_candidate_id")
        if dg not in (None, "") and str(dg) in excl["documented_link_component"]:
            reasons.append("stage5c_documented_link_component")
            break
        tb = item.get("timeline_bin")
        if tb not in (None, "") and str(tb) in excl["temporal_source_window"]:
            reasons.append("stage5c_temporal_source_window")
            break
        if str(item.get("source_crop_sha256") or "") in excl["source_crop_sha256"]:
            reasons.append("stage5c_source_crop_sha256")
            break
        if str(item.get("crop_id") or "") in excl["crop_id"]:
            reasons.append("stage5c_crop_id")
            break

    # Documented link via segment embedding global_id when present.
    gid = row.get("global_id")
    if gid not in (None, "") and str(gid) in excl["documented_link_component"]:
        reasons.append("stage5c_documented_link_component")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def representative_crop_score(
    crop: Mapping[str, Any],
    image: np.ndarray,
    *,
    mid_frame: float,
    frame_w: int,
    frame_h: int,
) -> tuple:
    """Lower is better. Identity-blind deterministic ranking."""
    h, w = image.shape[:2]
    area = int(w * h)
    short_side = min(w, h)
    bbox = crop.get("bbox_xyxy")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        margin = min(x0, y0, frame_w - x1, frame_h - y1)
        bbox_area = float(crop.get("bbox_area") or ((x1 - x0) * (y1 - y0)))
    else:
        margin = 0.0
        bbox_area = float(area)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = float(crop.get("quality_score") or 0.0)
    frame_index = int(crop["frame_index"])
    return (
        -bbox_area,
        -short_side,
        -margin,
        -area,
        -lap,
        -quality,
        abs(frame_index - mid_frame),
        str(crop["crop_id"]),
    )


def select_representative_crop(
    row: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    frame_w: int,
    frame_h: int,
) -> dict[str, Any]:
    first = row.get("first_frame")
    last = row.get("last_frame")
    if first is None or last is None:
        mid = 0.0
    else:
        mid = (float(first) + float(last)) / 2.0
    candidates: list[tuple[tuple, dict[str, Any], np.ndarray, str]] = []
    for cid in row.get("crop_ids") or []:
        crop = catalog.get(str(cid))
        if crop is None:
            continue
        path = Path(crop["absolute_path"])
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        sha = sha256_file(path)
        score = representative_crop_score(
            crop, image, mid_frame=mid, frame_w=frame_w, frame_h=frame_h
        )
        candidates.append((score, dict(crop), image, sha))
    if not candidates:
        raise AnchorPackageError(
            f"BLOCKED_STAGE5D_B_SOURCE_UNIVERSE_INTEGRITY no_crop {row['segment_id']}"
        )
    candidates.sort(key=lambda x: x[0])
    score, crop, image, sha = candidates[0]
    h, w = image.shape[:2]
    reason_parts = [
        "deterministic_full_body_preference",
        f"bbox_area_rank={-score[0]}",
        f"short_side={-score[1]}",
        f"edge_margin={-score[2]}",
        f"mid_frame_distance={score[6]}",
    ]
    return {
        "crop": crop,
        "source_crop_path": str(Path(crop["absolute_path"]).resolve()),
        "source_crop_sha256": sha,
        "original_dimensions": {"width": int(w), "height": int(h)},
        "representative_selection_reason": ";".join(reason_parts),
        "quality_diagnostics": {
            "bbox_area": crop.get("bbox_area"),
            "short_side": crop.get("short_side"),
            "quality_score": crop.get("quality_score"),
            "selection_rank_in_segment": crop.get("selection_rank"),
            "laplacian_variance_proxy": float(-score[4]),
            "mid_frame_distance": float(score[6]),
            "candidate_crop_count_considered": len(candidates),
        },
        "image": image,
    }


def render_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    images: Mapping[str, np.ndarray],
    *,
    max_items: int,
    columns: int,
) -> np.ndarray:
    if len(items) > max_items:
        raise AnchorPackageError("contact sheet exceeds max items")
    rows_n = int(math.ceil(len(items) / columns)) if items else 1
    tile_w, tile_h = 300, 380
    sheet = np.full((rows_n * tile_h, columns * tile_w, 3), 24, dtype=np.uint8)
    for index, item in enumerate(items):
        r, c = divmod(index, columns)
        image = images[str(item["anchor_candidate_id"])]
        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        # Visual panel: order + candidate id only (no track/segment/identity proof).
        labels = [
            f"#{item['anchor_order']}",
            str(item["anchor_candidate_id"]),
        ]
        y = 22
        for text in labels:
            cv2.putText(
                tile,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 22
        display = _fit_display(image, tile_w - 20, tile_h - 80)
        dh, dw = display.shape[:2]
        ox = (tile_w - dw) // 2
        oy = 60
        tile[oy : oy + dh, ox : ox + dw] = display
        y0 = r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise AnchorPackageError(f"failed to write png: {path}")


def build_target_definition(
    config: Mapping[str, Any], *, approved_at: str
) -> dict[str, Any]:
    td = config["target_definition"]
    if td["target_id"] != "target_001":
        raise AnchorPackageError("BLOCKED_STAGE5D_B_TARGET_DEFINITION target_id")
    if int(td["human_verified_jersey_number"]) != 5:
        raise AnchorPackageError("BLOCKED_STAGE5D_B_TARGET_DEFINITION jersey")
    if td["jersey_number_provenance"] != "human_verified_by_user_not_automated_ocr":
        raise AnchorPackageError("BLOCKED_STAGE5D_B_TARGET_DEFINITION provenance")
    if td["identity_basis"] != "human_visual_verification_from_source_video":
        raise AnchorPackageError("BLOCKED_STAGE5D_B_TARGET_DEFINITION basis")
    return {
        "schema_version": "reid_target_definition_freeze_v1",
        "target_id": td["target_id"],
        "target_alias": td["target_alias"],
        "target_description": td["target_description"],
        "identity_basis": td["identity_basis"],
        "human_verified_jersey_number": int(td["human_verified_jersey_number"]),
        "jersey_number_provenance": td["jersey_number_provenance"],
        "source_video": {
            "path": config["source_video"]["path"],
            "sha256": config["source_video"]["expected_sha256"],
            "bytes": int(config["source_video"]["expected_bytes"]),
            "frames": int(config["source_video"]["expected_frames"]),
        },
        "allowed_anchor_source_policy": (
            "full_body_segmented_reid_crops_with_existing_osnet_embedding"
        ),
        "forbidden_source_policy": [
            "stage5c_discovery_or_holdout_items",
            "jersey_number_roi_only",
            "automatic_similarity_enrollment",
            "ocr_based_automatic_enrollment",
            "tracker_id_based_automatic_enrollment",
            "parseq_prediction_as_identity_evidence",
        ],
        "reviewer": td["reviewer"],
        "final_approver": td["final_approver"],
        "approved_at": approved_at,
        "target_definition_frozen": True,
        "automated_jersey_used": False,
        "model_identity_prediction_used": False,
        "similarity_score_used": False,
        "notes": (
            "Human-verified jersey metadata is explanatory only; "
            "anchor membership requires separate visual human decisions."
        ),
    }


def build_anchor_review_contract(
    *,
    target_definition_sha: str,
    eligible_count: int,
    exclusion_counts: Mapping[str, int],
    no_embedding_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b_anchor_review_contract_v1",
        "target_id": "target_001",
        "target_definition_sha256": target_definition_sha,
        "eligible_source_universe_count": eligible_count,
        "excluded_no_existing_embedding": no_embedding_count,
        "stage5c_and_leakage_exclusion_counts": dict(exclusion_counts),
        "representative_crop_policy": {
            "one_candidate_per_eligible_segment": True,
            "identity_blind": True,
            "deterministic": True,
            "uses_existing_crops_only": True,
            "no_new_inference": True,
            "preference_order": [
                "full_visibility_proxy",
                "crop_area",
                "edge_margin",
                "image_size",
                "laplacian_blur_proxy",
                "near_segment_midtime",
            ],
            "forbidden": [
                "jersey_number_based_selection",
                "kit_color_target_filtering",
                "similarity_ranking",
                "embedding_target_prediction",
                "tracker_or_global_id_as_identity_proof",
            ],
        },
        "deterministic_ordering": "anchor_candidate_id ascending by segment_id sort",
        "contact_sheet_layout": {
            "max_items_per_sheet": 12,
            "shown_fields": ["anchor_order", "anchor_candidate_id", "full_body_crop"],
            "hidden_fields": [
                "similarity_score",
                "embedding_distance",
                "model_identity_prediction",
                "parseq_ocr_prediction",
                "ocr_confidence",
                "expected_jersey_overlay",
                "global_identity_result",
                "target_positive_prediction",
                "track_or_segment_as_identity_proof",
            ],
        },
        "allowed_manual_decisions": list(ALLOWED_MANUAL_DECISIONS),
        "only_target_anchor_yes_may_become_frozen_anchor_later": True,
        "target_anchor_yes_is_not_gallery_membership_in_stage5d_b": True,
        "no_automatic_enrollment": True,
        "no_similarity_ranking": True,
        "no_ocr_usage": True,
        "no_gallery_membership": True,
        "no_identity_assignment": True,
        "human_review_required": True,
        "final_approval_required": True,
        "anchor_freeze_requires_separate_gate": (
            "STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE"
        ),
        "unknown_identity_preserved": True,
        "manual_decisions": 0,
        "approved_anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise AnchorPackageError("FAILED_STAGE5D_B_ATOMIC_OUTPUT final_exists")

    preflight = validate_stage5d_a_preflight(project_root, config)
    closure = validate_stage5c_closure(project_root, config)
    assets = validate_assets(project_root, config)
    embedded, no_emb, emb_meta = audit_embeddings(project_root, config)
    catalog = load_crop_catalog(project_root, config)
    excl = load_stage5c_exclusion_sets(project_root, config)
    universe_by_seg = load_universe_by_segment(project_root, config)

    tmp = project_root / (
        "outputs/reid/"
        f"_tmp_full_stage4b_rebuild_r2_stage5d_target_001_anchor_review_package_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise AnchorPackageError("FAILED_STAGE5D_B_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    try:
        approved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        target_def = build_target_definition(config, approved_at=approved_at)
        target_path = tmp / "target_definition" / "target_001_definition_frozen.json"
        write_json(target_path, target_def)
        target_sha = sha256_file(target_path)

        frame_w = int(config["source_video"]["expected_width"])
        frame_h = int(config["source_video"]["expected_height"])

        inventory_rows: list[dict[str, Any]] = []
        eligible_rows: list[dict[str, Any]] = []
        exclusion_counter: dict[str, int] = defaultdict(int)

        for row in sorted(embedded, key=lambda r: str(r["segment_id"])):
            reasons = exclusion_reasons_for_segment(
                row, excl=excl, catalog=catalog, universe_by_seg=universe_by_seg
            )
            base = {
                "segment_id": str(row["segment_id"]),
                "raw_track_id": int(row["raw_track_id"]),
                "frame_range": [row.get("first_frame"), row.get("last_frame")],
                "observation_proxy_crop_count": int(row.get("crop_count") or 0),
                "crop_ids": list(row.get("crop_ids") or []),
                "embedding_available": True,
                "embedding_shape": 512,
                "embedding_finite": True,
                "embedding_row": row.get("embedding_row"),
                "embedding_sha256": row.get("embedding_sha256"),
                "representation_source": row.get("representation_source"),
                "source_type": (
                    "recomputed"
                    if "recomput" in str(row.get("representation_source") or "")
                    else "reused"
                ),
                "stage5c_exclusion_reasons": reasons,
            }
            if reasons:
                for reason in reasons:
                    exclusion_counter[reason] += 1
                base["eligibility_status"] = "excluded_stage5c_or_leakage"
                inventory_rows.append(base)
                continue
            selected = select_representative_crop(
                row, catalog, frame_w=frame_w, frame_h=frame_h
            )
            base.update(
                {
                    "eligibility_status": "eligible_anchor_candidate",
                    "selected_crop_id": selected["crop"]["crop_id"],
                    "frame_index": int(selected["crop"]["frame_index"]),
                    "source_crop_path": selected["source_crop_path"],
                    "source_crop_sha256": selected["source_crop_sha256"],
                    "original_dimensions": selected["original_dimensions"],
                    "representative_selection_reason": selected[
                        "representative_selection_reason"
                    ],
                    "quality_diagnostics": selected["quality_diagnostics"],
                    "_image": selected["image"],
                }
            )
            eligible_rows.append(base)
            inventory_rows.append({k: v for k, v in base.items() if k != "_image"})

        for row in no_emb:
            inventory_rows.append(
                {
                    "segment_id": str(row["segment_id"]),
                    "raw_track_id": int(row["raw_track_id"]),
                    "embedding_available": False,
                    "eligibility_status": "excluded_no_existing_embedding",
                    "stage5c_exclusion_reasons": ["no_existing_osnet_embedding"],
                }
            )
            exclusion_counter["excluded_no_existing_embedding"] += 1

        if not eligible_rows:
            raise AnchorPackageError("BLOCKED_STAGE5D_B_NO_ELIGIBLE_ANCHOR_SOURCE")

        # Deterministic candidate IDs / ordering by segment_id.
        eligible_rows.sort(key=lambda r: r["segment_id"])
        candidates: list[dict[str, Any]] = []
        images: dict[str, np.ndarray] = {}
        for i, row in enumerate(eligible_rows, start=1):
            cid = f"target_001_anchor_{i:03d}"
            image = row.pop("_image")
            images[cid] = image
            cand = {
                "anchor_candidate_id": cid,
                "anchor_order": i,
                "target_id": "target_001",
                "segment_id": row["segment_id"],
                "raw_track_id": row["raw_track_id"],
                "frame_index": row["frame_index"],
                "source_crop_path": row["source_crop_path"],
                "source_crop_sha256": row["source_crop_sha256"],
                "original_dimensions": row["original_dimensions"],
                "representative_selection_reason": row[
                    "representative_selection_reason"
                ],
                "quality_diagnostics": row["quality_diagnostics"],
                "stage5c_exclusion_audit": {
                    "excluded": False,
                    "reasons": [],
                },
                "source_type": row["source_type"],
                "representation_source": row["representation_source"],
                "embedding_row": row["embedding_row"],
                "embedding_sha256": row["embedding_sha256"],
                "manual_anchor_decision": "",
                "gallery_member": False,
                "prototype": False,
                "identity_assignment": None,
            }
            candidates.append(cand)

        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        with (inv_dir / "target_001_anchor_candidate_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in inventory_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for cand in candidates:
                # Also emit candidate-resolved rows for audit convenience.
                pass
        # Rewrite inventory to include candidate ids for eligible rows.
        inv_by_seg = {r["segment_id"]: r for r in inventory_rows}
        for cand in candidates:
            inv_by_seg[cand["segment_id"]].update(
                {
                    "anchor_candidate_id": cand["anchor_candidate_id"],
                    "anchor_order": cand["anchor_order"],
                    "selected_as_review_candidate": True,
                }
            )
        with (inv_dir / "target_001_anchor_candidate_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in inventory_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_inv = {
            "schema_version": "reid_stage5d_b_anchor_candidate_summary_v1",
            "target_id": "target_001",
            "embedded_input_segments": 150,
            "no_embedding_segments_excluded": 141,
            "eligible_anchor_candidates": len(candidates),
            "excluded_stage5c_or_leakage_embedded": 150 - len(candidates),
            "exclusion_reason_counts": dict(sorted(exclusion_counter.items())),
            "one_candidate_per_eligible_segment": True,
            "manual_decisions": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "similarity_ranking_rows": 0,
        }
        write_json(inv_dir / "target_001_anchor_candidate_summary.json", summary_inv)

        contract = build_anchor_review_contract(
            target_definition_sha=target_sha,
            eligible_count=len(candidates),
            exclusion_counts=dict(exclusion_counter),
            no_embedding_count=141,
        )
        ar_dir = tmp / "anchor_review"
        ar_dir.mkdir(parents=True)
        write_json(ar_dir / "target_001_anchor_review_contract.json", contract)

        # Contact sheets
        sheet_cfg = config["contact_sheets"]
        max_items = int(sheet_cfg["max_items_per_sheet"])
        columns = int(sheet_cfg["columns"])
        pkg_dir = tmp / "review_packages" / "target_001_anchor_review"
        pkg_dir.mkdir(parents=True)
        sheet_paths: list[str] = []
        sheet_dist: list[dict[str, Any]] = []
        for page_i, start in enumerate(range(0, len(candidates), max_items), start=1):
            chunk = candidates[start : start + max_items]
            sheet = render_contact_sheet(
                chunk, images, max_items=max_items, columns=columns
            )
            name = f"contact_sheet_{page_i:02d}.png"
            out = pkg_dir / name
            write_png(out, sheet)
            rel = f"review_packages/target_001_anchor_review/{name}"
            sheet_paths.append(rel)
            for item in chunk:
                item["contact_sheet_path"] = rel
                item["contact_sheet_page"] = page_i
            sheet_dist.append(
                {
                    "contact_sheet": rel,
                    "item_count": len(chunk),
                    "anchor_candidate_ids": [c["anchor_candidate_id"] for c in chunk],
                }
            )

        # Blank annotation template
        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        tpl_path = tpl_dir / "target_001_anchor_review_annotation_template.csv"
        with tpl_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_FIELDS))
            writer.writeheader()
            for cand in candidates:
                writer.writerow(
                    {
                        "anchor_candidate_id": cand["anchor_candidate_id"],
                        "target_id": cand["target_id"],
                        "segment_id": cand["segment_id"],
                        "raw_track_id": cand["raw_track_id"],
                        "frame_index": cand["frame_index"],
                        "source_crop_path": cand["source_crop_path"],
                        "source_crop_sha256": cand["source_crop_sha256"],
                        "manual_anchor_decision": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )

        # Verify blank decisions
        with tpl_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["manual_anchor_decision"]:
                    raise AnchorPackageError("prefilled decision forbidden")
                if row["manual_anchor_decision"] not in ("",):
                    raise AnchorPackageError("prefilled decision forbidden")

        write_json(
            ar_dir / "target_001_anchor_review_manifest.json",
            {
                "schema_version": "reid_stage5d_b_anchor_review_manifest_v1",
                "target_id": "target_001",
                "target_definition_sha256": target_sha,
                "candidates": [
                    {k: v for k, v in c.items() if k != "_image"} for c in candidates
                ],
                "contact_sheets": sheet_dist,
                "annotation_template": (
                    "templates/target_001_anchor_review_annotation_template.csv"
                ),
                "allowed_manual_decisions": list(ALLOWED_MANUAL_DECISIONS),
                "model_prediction_fields_present": False,
                "similarity_scores_present": False,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
                "manual_decisions": 0,
                "approved_anchors": 0,
            },
        )

        # runtime / effective configs
        runtime = {
            "schema_version": "reid_stage5d_b_runtime_v1",
            "started_at": approved_at,
            "device": "cpu",
            "offline_required": True,
            "network_download": 0,
            "new_model_inference": 0,
            "similarity_inference": 0,
            "parseq_inference": 0,
            "reserve_reads": 0,
            "source_crop_copies": 0,
        }
        write_json(tmp / "runtime" / "runtime.json", runtime)
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        png_count = len(list(tmp.rglob("*.png")))
        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        mp4_count = len(list(tmp.rglob("*.mp4")))
        if jpeg_count != 0 or mp4_count != 0:
            raise AnchorPackageError("FAILED_STAGE5D_B_ATOMIC_OUTPUT media_budget")
        if png_count != len(sheet_paths):
            raise AnchorPackageError("FAILED_STAGE5D_B_ATOMIC_OUTPUT png_budget")

        stage_summary = {
            "schema_version": "reid_stage5d_b_summary_v1",
            "final_status": "COMPLETED_STAGE5D_B_TARGET_001_ANCHOR_REVIEW_PACKAGE_READY",
            "project_head": head,
            "target_id": "target_001",
            "target_alias": target_def["target_alias"],
            "target_definition_frozen": True,
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": target_def["jersey_number_provenance"],
            "automated_jersey_used": False,
            "model_identity_prediction_used": False,
            "similarity_score_used": False,
            "embedded_segments_audited": 150,
            "no_embedding_excluded": 141,
            "eligible_anchor_candidates": len(candidates),
            "contact_sheet_count": len(sheet_paths),
            "contact_sheet_item_distribution": sheet_dist,
            "annotation_template_blank": True,
            "manual_decisions": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "similarity_ranking_rows": 0,
            "source_crop_copies": 0,
            "png_count": png_count,
            "jpeg_count": 0,
            "mp4_count": 0,
            "exact_next_gate": "STAGE5D-B2_TARGET_001_ANCHOR_MANUAL_REVIEW_AND_FREEZE",
            "stage5c_closure": closure["policy"],
            "stage5d_a_preflight_status": preflight["final_status"],
            "embedding_audit": emb_meta,
            "assets": assets,
        }
        write_json(tmp / "stage5d_b_summary.json", stage_summary)

        files_n, files_sha = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_b_manifest_v1",
            "final_status": stage_summary["final_status"],
            "project_head": head,
            "listing_file_count": files_n,
            "listing_sha256": files_sha,
            "target_definition_sha256": target_sha,
            "eligible_anchor_candidates": len(candidates),
            "contact_sheets": sheet_paths,
            "artifacts": {
                "target_definition": "target_definition/target_001_definition_frozen.json",
                "inventory_jsonl": "inventory/target_001_anchor_candidate_inventory.jsonl",
                "inventory_summary": "inventory/target_001_anchor_candidate_summary.json",
                "anchor_review_contract": (
                    "anchor_review/target_001_anchor_review_contract.json"
                ),
                "anchor_review_manifest": (
                    "anchor_review/target_001_anchor_review_manifest.json"
                ),
                "annotation_template": (
                    "templates/target_001_anchor_review_annotation_template.csv"
                ),
                "summary": "stage5d_b_summary.json",
            },
            "preflight_snapshot_sha256": preflight["snapshot_sha256"],
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
        }
        write_json(tmp / "stage5d_b_manifest.json", manifest)

        # Atomic rename
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_anchor_review_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except AnchorPackageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"final_status": summary["final_status"],
                      "eligible": summary["eligible_anchor_candidates"],
                      "sheets": summary["contact_sheet_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
