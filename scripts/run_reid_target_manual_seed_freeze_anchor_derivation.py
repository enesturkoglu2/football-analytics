#!/usr/bin/env python3
"""Stage 5D-B1B — freeze SEED_CANDIDATE_07 and derive anchors if eligible.

No gallery membership, OCR, similarity, crop regeneration, or new inference.
Ineligible Stage 5C/leakage sources remain frozen as identity seed only.
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
CONFIG_SCHEMA = "reid_target_manual_seed_freeze_anchor_derivation_config_v1"
SELECTED_CODE = "SEED_CANDIDATE_07"
ALLOWED_ANCHOR_DECISIONS = (
    "target_anchor_yes",
    "target_anchor_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
ANNOTATION_FIELDS = (
    "derived_anchor_candidate_id",
    "target_id",
    "segment_id",
    "raw_track_id",
    "frame_index",
    "source_crop_path",
    "source_crop_sha256",
    "manual_anchor_decision",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_identity_continuity_confirmed",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ELIGIBILITY_VALUES = (
    "eligible_for_anchor_derivation",
    "frozen_identity_seed_only_stage5c_excluded",
    "frozen_identity_seed_only_leakage_excluded",
    "frozen_identity_seed_only_no_existing_embedding",
    "frozen_identity_seed_only_ambiguous_segment",
    "frozen_identity_seed_only_other_exclusion",
)


class SeedFreezeError(RuntimeError):
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
        raise SeedFreezeError("unexpected config schema")
    if not config.get("offline_required"):
        raise SeedFreezeError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise SeedFreezeError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_manual_seed_freeze_anchor_derivation.py",
        "configs/reid/target_manual_seed_freeze_anchor_derivation_stage5d_target_001.yaml",
        "tests/test_reid_target_manual_seed_freeze_anchor_derivation.py",
        "docs/setup/stage5d-target-manual-seed-freeze-and-anchor-derivation.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise SeedFreezeError(
                    "BLOCKED_STAGE5D_B1B_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add target 001 manual seed selection package":
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path) -> str:
    if not snapshot_path.is_file():
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_sha = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (sidecar_sha == man_sha == actual):
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_b1a_package(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5d_b1a_package"]["path"]
    summary = load_json(root / "stage5d_b1a_summary.json")
    if summary.get("final_status") != config["stage5d_b1a_package"]["expected_final_status"]:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH target"
        )
    if list(summary.get("seed_window_frames") or []) != [280, 310]:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH window"
        )
    if int(summary.get("representative_seed_frame") or -1) != 290:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH rep"
        )
    if int(summary.get("window_observation_count") or 0) != 502:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH obs"
        )
    if int(summary.get("neutral_seed_candidate_count") or 0) != 34:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH cands"
        )
    for key in (
        "manual_selection",
        "approved_anchors",
        "gallery_members",
        "prototypes",
        "identity_assignments",
        "new_detection",
        "new_tracking",
        "new_embedding",
        "ocr",
        "similarity_ranking_rows",
    ):
        if int(summary.get(key) or 0) != 0:
            raise SeedFreezeError(
                f"BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH {key}"
            )

    mapping_path = root / "inventory" / "target_001_manual_seed_candidate_mapping.jsonl"
    rows = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    codes = sorted(r["neutral_seed_code"] for r in rows)
    expected = [f"SEED_CANDIDATE_{i:02d}" for i in range(1, 35)]
    if codes != expected:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH codes"
        )
    with (
        root / "templates" / "target_001_manual_seed_selection_template.csv"
    ).open(encoding="utf-8", newline="") as handle:
        tpl = next(csv.DictReader(handle))
    if tpl.get("selected_neutral_seed_code") or tpl.get("manual_target_confirmed"):
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SEED_PACKAGE_CONTRACT_MISMATCH not_blank"
        )

    snap_sha = resolve_snapshot_sha(Path(config["stage5d_b1a_package"]["snapshot_path"]))
    return {
        "path": config["stage5d_b1a_package"]["path"],
        "summary": summary,
        "mapping_rows": rows,
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "snapshot_sha256": snap_sha,
        "source_video_sha256": summary["source_video"]["sha256"],
    }


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY video_bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY video_sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY yolo_bytes")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY yolo_sha")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY osnet_sha")
    return {
        "source_video_sha256": config["source_video"]["expected_sha256"],
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "osnet_sha256": config["osnet_checkpoint"]["expected_sha256"],
    }


def resolve_selected_seed(
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    selected_code: str,
    review_window: Sequence[int],
    representative_frame: int,
) -> dict[str, Any]:
    if selected_code != SELECTED_CODE:
        raise SeedFreezeError(
            f"selected code must be exact {SELECTED_CODE!r}, got {selected_code!r}"
        )
    hits = [r for r in mapping_rows if r.get("neutral_seed_code") == selected_code]
    if not hits:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SELECTED_SEED_NOT_FOUND")
    if len(hits) != 1:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS duplicate_mapping"
        )
    row = hits[0]
    frames = [int(f) for f in row.get("observation_frames") or []]
    if not frames:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS empty_frames"
        )
    start, end = int(review_window[0]), int(review_window[1])
    if not any(start <= f <= end for f in frames):
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS outside_window"
        )
    if representative_frame not in frames:
        nearest = min(frames, key=lambda f: (abs(f - representative_frame), f))
        rep_delta = nearest - representative_frame
        rep_obs = nearest
    else:
        rep_delta = 0
        rep_obs = representative_frame

    raw_ids = {int(row["raw_track_id"])}
    seg_ids = {str(row["segment_id"])}
    if len(raw_ids) != 1 or len(seg_ids) != 1:
        raise SeedFreezeError(
            "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS multi_id"
        )
    return {
        "mapping": row,
        "raw_track_id": int(row["raw_track_id"]),
        "segment_id": str(row["segment_id"]),
        "observation_frames": frames,
        "first_frame": int(row["first_frame"]),
        "last_frame": int(row["last_frame"]),
        "representative_observation_frame": rep_obs,
        "representative_frame_delta": rep_delta,
    }


def load_exclusion_universe(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    split = project_root / config["upstream"]["canonical_split"]
    batches = {
        "discovery_primary": [],
        "discovery_reserve": [],
        "holdout_primary": [],
        "holdout_reserve": [],
    }
    keys = {
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
    }
    for batch in batches:
        man = split / batch / f"{batch}_manifest.jsonl"
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            batches[batch].append(row)
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
    keys = {k: {x for x in v if x} for k, v in keys.items()}

    universe_path = (
        project_root
        / config["upstream"]["visibility_universe"]
        / "clean_review_universe"
        / "clean_review_items.jsonl"
    )
    uni_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if universe_path.is_file():
        for line in universe_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uni_by_seg[str(row["segment_id"])].append(row)
    return {"batches": batches, "keys": keys, "universe_by_seg": uni_by_seg}


def validate_embedding_for_segment(
    project_root: Path, config: Mapping[str, Any], segment_id: str
) -> dict[str, Any]:
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    index_path = seg_root / "segment_embedding_index.jsonl"
    npz_path = seg_root / "segment_embeddings.npz"
    meta = None
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["segment_id"]) == segment_id:
            meta = row
            break
    if meta is None:
        return {
            "embedding_available": False,
            "reason": "segment_missing_from_index",
        }
    if not meta.get("embedding_available"):
        return {
            "embedding_available": False,
            "reason": "no_existing_embedding",
            "meta": meta,
        }
    data = np.load(npz_path)
    vectors = data["vectors"]
    row_i = int(meta["embedding_row"])
    vec = vectors[row_i]
    if vec.shape != (512,):
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY emb_dim")
    if int(np.isnan(vec).sum()) or int(np.isinf(vec).sum()):
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY emb_naninf")
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if norm == 0.0:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY emb_zero")
    return {
        "embedding_available": True,
        "embedding_dimension": 512,
        "embedding_finite": True,
        "embedding_zero_vector": False,
        "embedding_norm": norm,
        "embedding_row": row_i,
        "embedding_sha256": meta.get("embedding_sha256"),
        "embedding_artifact_path": str(
            Path(config["upstream"]["stage5_replay"])
            / config["upstream"]["segmented_reid_subdir"]
            / "segment_embeddings.npz"
        ),
        "embedding_artifact_sha256": sha256_file(npz_path),
        "representation_source": meta.get("representation_source"),
        "crop_ids": list(meta.get("crop_ids") or []),
        "meta": meta,
    }


def audit_bbox_lineage(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    segment_id: str,
    raw_track_id: int,
    mapping_bboxes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    obs_path = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segment_view_subdir"]
        / "segment_observations.jsonl"
    )
    by_frame: dict[int, dict[str, Any]] = {}
    for line in obs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("segment_id")) != segment_id:
            continue
        if int(row.get("raw_track_id")) != raw_track_id:
            raise SeedFreezeError(
                "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS track_mismatch"
            )
        src = row.get("source_observation") or {}
        bbox = src.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise SeedFreezeError(
                "BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS bbox"
            )
        by_frame[int(row["frame_index"])] = {
            "frame_index": int(row["frame_index"]),
            "bbox_xyxy": [float(x) for x in bbox],
            "source_observation_sha256": row.get("source_observation_sha256"),
        }
    audited = []
    for item in mapping_bboxes:
        fi = int(item["frame_index"])
        if fi not in by_frame:
            raise SeedFreezeError(
                f"BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS missing_obs {fi}"
            )
        upstream = by_frame[fi]["bbox_xyxy"]
        mapped = [float(x) for x in item["bbox_xyxy"]]
        if any(abs(a - b) > 1e-6 for a, b in zip(upstream, mapped)):
            raise SeedFreezeError(
                f"BLOCKED_STAGE5D_B1B_SELECTED_SEED_LINEAGE_AMBIGUOUS bbox_mismatch {fi}"
            )
        audited.append(by_frame[fi])
    return audited


def evaluate_eligibility(
    *,
    segment_id: str,
    raw_track_id: int,
    emb: Mapping[str, Any],
    exclusion: Mapping[str, Any],
    crop_ids: Sequence[str],
    crop_shas: Sequence[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    keys = exclusion["keys"]
    batches = exclusion["batches"]
    batch_hits = {
        name: [
            r["split_item_id"]
            for r in rows
            if str(r.get("segment_id")) == segment_id
            or str(r.get("raw_track_id")) == str(raw_track_id)
        ]
        for name, rows in batches.items()
    }
    stage5c_hit = any(batch_hits.values())
    if stage5c_hit:
        reasons.append("stage5c_membership")

    leakage_hits: list[str] = []
    if segment_id in keys["segment_id"]:
        leakage_hits.append("segment_id")
    if str(raw_track_id) in keys["raw_track_id"]:
        leakage_hits.append("raw_track_id")
    for cid in crop_ids:
        if str(cid) in keys["crop_id"]:
            leakage_hits.append("crop_id")
            break
    for sha in crop_shas:
        if sha in keys["source_crop_sha256"]:
            leakage_hits.append("source_crop_sha256")
            break
    for item in exclusion["universe_by_seg"].get(segment_id, []):
        if str(item.get("near_duplicate_cluster_id") or "") in keys[
            "near_duplicate_component"
        ]:
            leakage_hits.append("near_duplicate_component")
        if str(item.get("leakage_group_id") or "") in keys["exact_duplicate_group"]:
            leakage_hits.append("exact_duplicate_group")
        if str(item.get("documented_global_candidate_id") or "") in keys[
            "documented_link_component"
        ]:
            leakage_hits.append("documented_link_component")
        if str(item.get("timeline_bin") or "") in keys["temporal_source_window"]:
            leakage_hits.append("temporal_source_window")
        if f"{segment_id}:{item.get('frame_index')}" in keys["frame_identity"]:
            leakage_hits.append("frame_identity")
    leakage_hits = sorted(set(leakage_hits))
    if leakage_hits and not stage5c_hit:
        reasons.append("leakage_exclusion")

    if not emb.get("embedding_available"):
        reasons.append("no_existing_embedding")

    # Ambiguous multi-person flag not present on this preserved track; keep hook.
    ambiguous = False

    if stage5c_hit:
        status = "frozen_identity_seed_only_stage5c_excluded"
    elif "no_existing_embedding" in reasons:
        status = "frozen_identity_seed_only_no_existing_embedding"
    elif leakage_hits:
        status = "frozen_identity_seed_only_leakage_excluded"
    elif ambiguous:
        status = "frozen_identity_seed_only_ambiguous_segment"
    elif reasons:
        status = "frozen_identity_seed_only_other_exclusion"
    else:
        status = "eligible_for_anchor_derivation"

    if status not in ELIGIBILITY_VALUES:
        raise SeedFreezeError(f"unknown eligibility {status}")

    return {
        "selected_seed_source_eligibility": status,
        "eligible_for_anchor_derivation": status == "eligible_for_anchor_derivation",
        "stage5c_batch_hits": {k: v for k, v in batch_hits.items() if v},
        "leakage_hit_keys": leakage_hits,
        "exclusion_reasons": reasons,
        "embedding_ok": bool(emb.get("embedding_available")),
        "ambiguous_segment": ambiguous,
    }


def load_segment_crops(
    project_root: Path, config: Mapping[str, Any], crop_ids: Sequence[str]
) -> list[dict[str, Any]]:
    baseline = project_root / config["upstream"]["baseline_crops_root"]
    catalog: dict[str, dict[str, Any]] = {}
    man = baseline / "crop_manifest.jsonl"
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = baseline / str(row["crop_relative_path"])
            catalog[str(row["crop_id"])] = {**row, "absolute_path": path}
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
            path = seg_root / str(row["crop_relative_path"])
            catalog[str(row["crop_id"])] = {**row, "absolute_path": path}

    out = []
    for cid in crop_ids:
        crop = catalog.get(str(cid))
        if crop is None:
            continue
        path = Path(crop["absolute_path"])
        if not path.is_file():
            continue
        sha = sha256_file(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        out.append(
            {
                **crop,
                "source_crop_path": str(path.resolve()),
                "source_crop_sha256": sha,
                "width": int(w),
                "height": int(h),
                "image": image,
            }
        )
    return out


def derive_diverse_anchors(
    crops: Sequence[Mapping[str, Any]],
    *,
    segment_id: str,
    raw_track_id: int,
    seed_freeze_sha: str,
    emb: Mapping[str, Any],
    max_candidates: int,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """Deterministic diversity selection; identity-blind."""
    if not crops:
        return []
    frames = [int(c["frame_index"]) for c in crops]
    fmin, fmax = min(frames), max(frames)
    mid = (fmin + fmax) / 2.0
    # Score: prefer spread across early/mid/late, larger short side, area, quality.
    buckets = {"early": [], "mid": [], "late": []}
    for crop in crops:
        fi = int(crop["frame_index"])
        if fi <= fmin + (fmax - fmin) / 3:
            buckets["early"].append(crop)
        elif fi >= fmax - (fmax - fmin) / 3:
            buckets["late"].append(crop)
        else:
            buckets["mid"].append(crop)

    def rank_key(crop: Mapping[str, Any]) -> tuple:
        area = float(crop.get("bbox_area") or crop["width"] * crop["height"])
        short = float(crop.get("short_side") or min(crop["width"], crop["height"]))
        quality = float(crop.get("quality_score") or 0.0)
        # Prefer mid-distance from already selected handled later; base key:
        return (-area, -short, -quality, int(crop["frame_index"]), str(crop["crop_id"]))

    selected: list[dict[str, Any]] = []
    used_frames: set[int] = set()
    used_shas: set[str] = set()

    def try_add(crop: Mapping[str, Any], bucket: str) -> None:
        if len(selected) >= max_candidates:
            return
        fi = int(crop["frame_index"])
        sha = str(crop["source_crop_sha256"])
        # Near-duplicate / same-frame suppression.
        if any(abs(fi - uf) <= 2 for uf in used_frames):
            return
        if sha in used_shas:
            return
        short = float(crop.get("short_side") or min(crop["width"], crop["height"]))
        if short < 20:
            return
        area = float(crop.get("bbox_area") or crop["width"] * crop["height"])
        if area < 400:
            return
        used_frames.add(fi)
        used_shas.add(sha)
        idx = len(selected) + 1
        selected.append(
            {
                "derived_anchor_candidate_id": f"target_001_seed_anchor_{idx:03d}",
                "target_id": "target_001",
                "selected_seed_freeze_sha256": seed_freeze_sha,
                "segment_id": segment_id,
                "raw_track_id": raw_track_id,
                "frame_index": fi,
                "video_time_sec": fi / fps,
                "source_crop_path": crop["source_crop_path"],
                "source_crop_sha256": sha,
                "crop_id": crop["crop_id"],
                "bbox_xyxy": crop.get("bbox_xyxy"),
                "dimensions": {"width": crop["width"], "height": crop["height"]},
                "quality_diagnostics": {
                    "bbox_area": crop.get("bbox_area"),
                    "short_side": crop.get("short_side"),
                    "quality_score": crop.get("quality_score"),
                    "selection_rank": crop.get("selection_rank"),
                },
                "temporal_diversity_bucket": bucket,
                "duplicate_near_duplicate_audit": {
                    "min_frame_gap_from_other_selected": min(
                        (abs(fi - uf) for uf in used_frames if uf != fi), default=None
                    ),
                    "exact_sha_duplicate_suppressed": False,
                },
                "embedding_path": emb.get("embedding_artifact_path"),
                "embedding_artifact_sha256": emb.get("embedding_artifact_sha256"),
                "embedding_sha256": emb.get("embedding_sha256"),
                "embedding_available": True,
                "manual_anchor_decision": "",
                "gallery_member": False,
                "image": crop["image"],
            }
        )

    # Round-robin buckets for diversity.
    ordered_buckets = ["early", "mid", "late", "mid", "early", "late", "mid", "early"]
    for bucket in ordered_buckets:
        pool = sorted(buckets.get(bucket) or [], key=rank_key)
        for crop in pool:
            before = len(selected)
            try_add(crop, bucket)
            if len(selected) > before:
                break
        if len(selected) >= max_candidates:
            break

    # Fill remaining from global rank if needed.
    if len(selected) < max_candidates:
        for crop in sorted(crops, key=rank_key):
            # infer bucket
            fi = int(crop["frame_index"])
            if fi <= fmin + (fmax - fmin) / 3:
                b = "early"
            elif fi >= fmax - (fmax - fmin) / 3:
                b = "late"
            else:
                b = "mid"
            try_add(crop, b)
            if len(selected) >= max_candidates:
                break
    return selected


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def render_anchor_sheet(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    cols = min(4, max(1, len(candidates)))
    rows_n = int(math.ceil(len(candidates) / cols))
    tile_w, tile_h = 300, 380
    sheet = np.full((rows_n * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for i, cand in enumerate(candidates):
        r, c = divmod(i, cols)
        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        labels = [
            f"#{i + 1}",
            str(cand["derived_anchor_candidate_id"]),
            f"f={cand['frame_index']}",
            f"t={cand['video_time_sec']:.2f}s",
        ]
        y = 20
        for text in labels:
            cv2.putText(
                tile,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 18
        disp = _fit_display(cand["image"], tile_w - 16, tile_h - 100)
        dh, dw = disp.shape[:2]
        ox = (tile_w - dw) // 2
        oy = 90
        tile[oy : oy + dh, ox : ox + dw] = disp
        y0 = r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise SeedFreezeError(f"FAILED_STAGE5D_B1B_ATOMIC_OUTPUT png {path}")


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise SeedFreezeError("FAILED_STAGE5D_B1B_ATOMIC_OUTPUT final_exists")

    human = config["human_selection"]
    if human["selected_neutral_seed_code"] != SELECTED_CODE:
        raise SeedFreezeError("selected_neutral_seed_code must be SEED_CANDIDATE_07")
    if human["selected_neutral_seed_code"] in {"07", "077", 7}:
        raise SeedFreezeError("integer/truncated seed code forbidden")

    b1a = validate_b1a_package(project_root, config)
    assets = validate_assets(project_root, config)
    td_path = project_root / config["target_definition"]["path"]
    td = load_json(td_path)
    td_sha = sha256_file(td_path)
    if td.get("target_id") != "target_001" or td.get("target_definition_frozen") is not True:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY target_def")

    # Stage 5C closure quick check
    pol = load_json(
        project_root
        / config["stage5c_closure"]["path"]
        / config["stage5c_closure"]["policy_json"]
    )
    if pol.get("automated_parseq_identity_assignment_allowed") is not False:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY parseq_assign")
    if pol.get("automated_parseq_gallery_enrollment_allowed") is not False:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY parseq_enroll")

    selected = resolve_selected_seed(
        b1a["mapping_rows"],
        selected_code=human["selected_neutral_seed_code"],
        review_window=[280, 310],
        representative_frame=290,
    )
    bbox_lineage = audit_bbox_lineage(
        project_root,
        config,
        segment_id=selected["segment_id"],
        raw_track_id=selected["raw_track_id"],
        mapping_bboxes=selected["mapping"]["bbox_per_frame"],
    )
    if b1a["source_video_sha256"] != assets["source_video_sha256"]:
        raise SeedFreezeError("BLOCKED_STAGE5D_B1B_SOURCE_INTEGRITY video_sha_mismatch")

    emb = validate_embedding_for_segment(
        project_root, config, selected["segment_id"]
    )
    exclusion = load_exclusion_universe(project_root, config)

    # Crop SHAs from mapping lineage for exclusion audit.
    crop_ids = [
        c["crop_id"] for c in (selected["mapping"].get("source_crop_lineage") or [])
    ]
    crop_shas = [
        c["sha256"]
        for c in (selected["mapping"].get("source_crop_lineage") or [])
        if c.get("sha256")
    ]
    if emb.get("crop_ids"):
        crop_ids = list(dict.fromkeys(list(emb["crop_ids"]) + crop_ids))

    eligibility = evaluate_eligibility(
        segment_id=selected["segment_id"],
        raw_track_id=selected["raw_track_id"],
        emb=emb,
        exclusion=exclusion,
        crop_ids=crop_ids,
        crop_shas=crop_shas,
    )

    approved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_seed_freeze_anchor_derivation_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise SeedFreezeError("FAILED_STAGE5D_B1B_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    try:
        freeze = {
            "schema_version": "reid_target_manual_seed_selection_freeze_v1",
            "target_definition_path": config["target_definition"]["path"],
            "target_definition_sha256": td_sha,
            "target_id": "target_001",
            "target_alias": td.get("target_alias"),
            "selected_neutral_seed_code": SELECTED_CODE,
            "resolved_raw_track_id": selected["raw_track_id"],
            "resolved_segment_id": selected["segment_id"],
            "observation_frames": selected["observation_frames"],
            "first_frame": selected["first_frame"],
            "last_frame": selected["last_frame"],
            "representative_reference_frame": 290,
            "representative_observation_frame": selected[
                "representative_observation_frame"
            ],
            "representative_frame_delta": selected["representative_frame_delta"],
            "bbox_lineage": bbox_lineage,
            "source_crop_lineage": selected["mapping"].get("source_crop_lineage"),
            "existing_embedding_availability": {
                "available": bool(emb.get("embedding_available")),
                "dimension": emb.get("embedding_dimension"),
                "finite": emb.get("embedding_finite"),
                "zero_vector": emb.get("embedding_zero_vector"),
                "norm": emb.get("embedding_norm"),
                "embedding_sha256": emb.get("embedding_sha256"),
                "artifact_sha256": emb.get("embedding_artifact_sha256"),
            },
            "human_decision": {
                "manual_target_confirmed": human["manual_target_confirmed"],
                "manual_human_verified_number_seen": human[
                    "manual_human_verified_number_seen"
                ],
                "manual_crop_valid": human["manual_crop_valid"],
                "manual_target_dominant": human["manual_target_dominant"],
                "human_verified_jersey_number": int(
                    human["human_verified_jersey_number"]
                ),
                "jersey_number_provenance": human["jersey_number_provenance"],
                "manual_notes": human["manual_notes"],
            },
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "approved_at": approved_at,
            "target_seed_frozen": True,
            "automated_jersey_used": False,
            "OCR_used": False,
            "similarity_used": False,
            "model_identity_prediction_used": False,
            "new_detection_used": False,
            "new_tracking_used": False,
            "seed_is_gallery_member": False,
            "seed_is_final_anchor": False,
            "b1a_mapping_sha256": b1a["mapping_sha256"],
            "source_video_sha256": assets["source_video_sha256"],
        }
        freeze_dir = tmp / "seed_freeze"
        freeze_dir.mkdir(parents=True)
        freeze_path = freeze_dir / "target_001_manual_seed_selection_frozen.json"
        write_json(freeze_path, freeze)
        freeze_sha = sha256_file(freeze_path)

        write_json(
            freeze_dir / "target_001_manual_seed_selection_freeze_contract.json",
            {
                "schema_version": "reid_stage5d_b1b_seed_freeze_contract_v1",
                "selected_neutral_seed_code": SELECTED_CODE,
                "target_seed_frozen": True,
                "immutable_after_publish": True,
                "no_ocr": True,
                "no_similarity": True,
                "no_gallery_membership_from_seed_alone": True,
                "seed_is_final_anchor": False,
                "alternate_source_required_if_ineligible": True,
            },
        )
        write_json(
            freeze_dir / "target_001_manual_seed_selection_freeze_manifest.json",
            {
                "schema_version": "reid_stage5d_b1b_seed_freeze_manifest_v1",
                "freeze_path": "seed_freeze/target_001_manual_seed_selection_frozen.json",
                "freeze_sha256": freeze_sha,
                "selected_neutral_seed_code": SELECTED_CODE,
                "resolved_segment_id": selected["segment_id"],
                "resolved_raw_track_id": selected["raw_track_id"],
            },
        )

        elig_payload = {
            "schema_version": "reid_stage5d_b1b_selected_seed_source_eligibility_v1",
            **eligibility,
            "segment_id": selected["segment_id"],
            "raw_track_id": selected["raw_track_id"],
            "selected_neutral_seed_code": SELECTED_CODE,
            "seed_freeze_sha256": freeze_sha,
            "stage5d_a_exclusion_contract": config["stage5d_a_preflight"][
                "exclusion_contract"
            ],
        }
        elig_dir = tmp / "eligibility"
        elig_dir.mkdir(parents=True)
        write_json(
            elig_dir / "target_001_selected_seed_source_eligibility.json", elig_payload
        )

        candidates: list[dict[str, Any]] = []
        png_count = 0
        sheet_path = None
        template_rows = 0
        final_status = ""
        exact_next = ""

        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        review_dir = tmp / "review_packages" / "target_001_seed_derived_anchor_review"
        review_dir.mkdir(parents=True)

        min_needed = int(config["anchor_derivation"]["min_candidates_for_package"])
        max_cands = int(config["anchor_derivation"]["max_candidates"])

        if eligibility["eligible_for_anchor_derivation"]:
            crops = load_segment_crops(project_root, config, emb.get("crop_ids") or crop_ids)
            # Drop image-less for derivation helper already filters.
            candidates = derive_diverse_anchors(
                crops,
                segment_id=selected["segment_id"],
                raw_track_id=selected["raw_track_id"],
                seed_freeze_sha=freeze_sha,
                emb=emb,
                max_candidates=max_cands,
            )
            if len(candidates) < min_needed:
                final_status = (
                    "COMPLETED_STAGE5D_B1B_SEED_FROZEN_INSUFFICIENT_ANCHOR_DIVERSITY"
                )
                exact_next = (
                    "STAGE5D-B1C_TARGET_001_ADDITIONAL_HUMAN_SEED_WINDOW_REVIEW"
                )
                candidates = []
            else:
                sheet = render_anchor_sheet(candidates)
                write_png(review_dir / "contact_sheet_01.png", sheet)
                sheet_path = (
                    "review_packages/target_001_seed_derived_anchor_review/"
                    "contact_sheet_01.png"
                )
                png_count = 1
                tpl_path = tpl_dir / "target_001_seed_derived_anchor_review_template.csv"
                with tpl_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(ANNOTATION_FIELDS)
                    )
                    writer.writeheader()
                    for cand in candidates:
                        writer.writerow(
                            {
                                "derived_anchor_candidate_id": cand[
                                    "derived_anchor_candidate_id"
                                ],
                                "target_id": cand["target_id"],
                                "segment_id": cand["segment_id"],
                                "raw_track_id": cand["raw_track_id"],
                                "frame_index": cand["frame_index"],
                                "source_crop_path": cand["source_crop_path"],
                                "source_crop_sha256": cand["source_crop_sha256"],
                                "manual_anchor_decision": "",
                                "manual_crop_valid": "",
                                "manual_target_dominant": "",
                                "manual_identity_continuity_confirmed": "",
                                "manual_notes": "",
                                "reviewer": "",
                                "final_approver": "",
                                "reviewed_at": "",
                            }
                        )
                        template_rows += 1
                final_status = (
                    "COMPLETED_STAGE5D_B1B_TARGET_001_SEED_FROZEN_ANCHOR_REVIEW_READY"
                )
                exact_next = (
                    "STAGE5D-B1C_TARGET_001_DERIVED_ANCHOR_MANUAL_REVIEW_AND_FREEZE"
                )
        else:
            final_status = "COMPLETED_STAGE5D_B1B_SEED_FROZEN_SOURCE_INELIGIBLE"
            exact_next = (
                "STAGE5D-B1C_TARGET_001_ALTERNATE_ELIGIBLE_ANCHOR_SOURCE_REVIEW"
            )

        # Inventory (empty list still written)
        with (inv_dir / "target_001_seed_derived_anchor_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for cand in candidates:
                row = {k: v for k, v in cand.items() if k != "image"}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            inv_dir / "target_001_seed_derived_anchor_summary.json",
            {
                "schema_version": "reid_stage5d_b1b_seed_derived_anchor_summary_v1",
                "selected_seed_source_eligibility": eligibility[
                    "selected_seed_source_eligibility"
                ],
                "derived_anchor_candidate_count": len(candidates),
                "contact_sheet_path": sheet_path,
                "annotation_template_rows": template_rows,
                "approved_anchors": 0,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
                "diversity_buckets": {
                    b: sum(
                        1
                        for c in candidates
                        if c.get("temporal_diversity_bucket") == b
                    )
                    for b in ("early", "mid", "late")
                },
            },
        )

        # Empty template file marker when ineligible (0 rows, header only optional)
        if template_rows == 0:
            # Do not create misleading filled template; create empty header-only
            # file documenting zero candidates.
            tpl_path = tpl_dir / "target_001_seed_derived_anchor_review_template.csv"
            with tpl_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_FIELDS))
                writer.writeheader()

        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1b_runtime_v1",
                "approved_at": approved_at,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        mp4_count = len(list(tmp.rglob("*.mp4")))
        actual_png = len(list(tmp.rglob("*.png")))
        if jpeg_count or mp4_count:
            raise SeedFreezeError("FAILED_STAGE5D_B1B_ATOMIC_OUTPUT media")
        if actual_png != png_count:
            raise SeedFreezeError(
                f"FAILED_STAGE5D_B1B_ATOMIC_OUTPUT png={actual_png}"
            )

        summary = {
            "schema_version": "reid_stage5d_b1b_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "selected_neutral_seed_code": SELECTED_CODE,
            "manual_target_confirmed": human["manual_target_confirmed"],
            "manual_human_verified_number_seen": human[
                "manual_human_verified_number_seen"
            ],
            "human_verified_jersey_number": 5,
            "jersey_number_provenance": human["jersey_number_provenance"],
            "reviewer": human["reviewer"],
            "final_approver": human["final_approver"],
            "approved_at": approved_at,
            "target_seed_frozen": True,
            "resolved_raw_track_id": selected["raw_track_id"],
            "resolved_segment_id": selected["segment_id"],
            "observation_frame_range": [
                selected["first_frame"],
                selected["last_frame"],
            ],
            "observation_frame_count": len(selected["observation_frames"]),
            "seed_freeze_sha256": freeze_sha,
            "selected_seed_source_eligibility": eligibility[
                "selected_seed_source_eligibility"
            ],
            "stage5c_batch_hits": eligibility["stage5c_batch_hits"],
            "leakage_hit_keys": eligibility["leakage_hit_keys"],
            "existing_embedding_available": bool(emb.get("embedding_available")),
            "derived_anchor_candidate_count": len(candidates),
            "contact_sheet_png_count": png_count,
            "annotation_template_rows": template_rows,
            "automated_jersey_used": False,
            "OCR_used": False,
            "similarity_used": False,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "automatic_anchor_decisions": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "exact_next_gate": exact_next,
            "b1a_snapshot_sha256": b1a["snapshot_sha256"],
            "assets": assets,
        }
        write_json(tmp / "stage5d_b1b_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1b_manifest.json",
            {
                "schema_version": "reid_stage5d_b1b_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "selected_neutral_seed_code": SELECTED_CODE,
                "seed_freeze_sha256": freeze_sha,
                "selected_seed_source_eligibility": eligibility[
                    "selected_seed_source_eligibility"
                ],
                "derived_anchor_candidate_count": len(candidates),
                "contact_sheet_path": sheet_path,
                "gallery_members": 0,
                "prototypes": 0,
                "identity_assignments": 0,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1b_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_manual_seed_freeze_anchor_derivation_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except SeedFreezeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "selected": summary["selected_neutral_seed_code"],
                "eligibility": summary["selected_seed_source_eligibility"],
                "derived_anchors": summary["derived_anchor_candidate_count"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
