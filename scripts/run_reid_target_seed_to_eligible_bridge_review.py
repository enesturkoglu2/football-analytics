#!/usr/bin/env python3
"""Stage 5D-B1C2 — frozen seed to eligible source temporal bridge review.

Frozen SEED_CANDIDATE_07 / raw_222_full is visual continuity only.
No enrollment, scoring, OCR, similarity, new inference, or gallery membership.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "reid_target_seed_to_eligible_bridge_review_config_v1"
FROZEN_SEED_CODE = "SEED_CANDIDATE_07"
FROZEN_VISUAL_LABEL = "FROZEN_HUMAN_SEED_REF"
FROZEN_VISUAL_WARNING = "NOT ENROLLABLE — HUMAN CONTINUITY REFERENCE ONLY"
TEMPLATE_FIELDS = (
    "target_id",
    "bridge_candidate_code",
    "segment_id",
    "raw_track_id",
    "first_frame",
    "last_frame",
    "selected_as_target_continuation",
    "manual_target_confirmed",
    "manual_identity_continuity_observed",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_human_verified_number_seen",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")


class BridgeReviewError(RuntimeError):
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
        raise BridgeReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise BridgeReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise BridgeReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_seed_to_eligible_bridge_review.py",
        "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml",
        "tests/test_reid_target_seed_to_eligible_bridge_review.py",
        "docs/setup/stage5d-target-seed-to-eligible-bridge-review.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise BridgeReviewError(
                    "BLOCKED_STAGE5D_B1C2_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    return head


def validate_frozen_and_b1c(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    b1b_root = project_root / config["stage5d_b1b_package"]["path"]
    summary = load_json(b1b_root / "stage5d_b1b_summary.json")
    freeze = load_json(
        b1b_root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
    )
    td = load_json(project_root / config["target_definition"]["path"])
    exp = config["stage5d_b1b_package"]

    if summary.get("target_id") != "target_001":
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH target"
        )
    if td.get("target_definition_frozen") is not True:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH td")
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH alias"
        )
    if summary.get("selected_neutral_seed_code") != exp["expected_selected_seed"]:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH seed"
        )
    if summary.get("resolved_segment_id") != exp["expected_segment_id"]:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH segment"
        )
    if int(summary.get("resolved_raw_track_id")) != int(exp["expected_raw_track_id"]):
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH track"
        )
    for key in ("manual_target_confirmed", "manual_human_verified_number_seen"):
        if summary.get(key) != "yes":
            raise BridgeReviewError(
                f"BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH {key}"
            )
    hd = freeze.get("human_decision") or {}
    if hd.get("manual_crop_valid") != "yes" or hd.get("manual_target_dominant") != "yes":
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH human_flags"
        )
    if freeze.get("target_seed_frozen") is not True:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH not_frozen"
        )
    if summary.get("selected_seed_source_eligibility") != exp["expected_eligibility"]:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH eligibility"
        )
    if freeze.get("seed_is_gallery_member") is not False:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH gallery_member"
        )
    if freeze.get("seed_is_final_anchor") is not False:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH final_anchor"
        )

    b1c_root = project_root / config["stage5d_b1c_package"]["path"]
    b1c = load_json(b1c_root / "stage5d_b1c_summary.json")
    bexp = config["stage5d_b1c_package"]
    if b1c.get("final_status") != bexp["expected_final_status"]:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_status"
        )
    if list(b1c.get("review_window_frames") or []) != list(bexp["expected_review_window"]):
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_window"
        )
    if int(b1c.get("eligible_candidate_count")) != int(bexp["expected_eligible"]):
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_eligible"
        )
    if int(b1c.get("ineligible_candidate_count")) != int(bexp["expected_ineligible"]):
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_ineligible"
        )
    if b1c.get("original_frozen_seed_code") != FROZEN_SEED_CODE:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_seed"
        )
    if int(b1c.get("gallery_members") or 0) != 0:
        raise BridgeReviewError(
            "BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH b1c_gallery"
        )

    return {
        "summary": summary,
        "freeze": freeze,
        "target_definition": td,
        "b1c": b1c,
        "freeze_sha256": sha256_file(
            b1b_root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
        ),
    }


def validate_stage5c(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    pol = load_json(
        project_root
        / config["stage5c_closure"]["path"]
        / config["stage5c_closure"]["policy_json"]
    )
    required = {
        "stage5c_status": "closed",
        "stage5e_automated_jersey_channel_mode": "diagnostic_only",
        "automated_parseq_gallery_enrollment_allowed": False,
        "discovery_reserve_opened": False,
        "holdout_reserve_opened": False,
    }
    for key, expected in required.items():
        if pol.get(key) != expected:
            raise BridgeReviewError(
                f"BLOCKED_STAGE5D_B1C2_FROZEN_SEED_CONTRACT_MISMATCH {key}"
            )
    return required


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY yolo_b")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY yolo")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY osnet")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY open")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != int(config["source_video"]["expected_frames"]):
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY frames")
    if width != int(config["source_video"]["expected_width"]):
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY width")
    if height != int(config["source_video"]["expected_height"]):
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY height")
    if abs(fps - float(config["source_video"]["expected_fps"])) > 1e-6:
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY fps")
    return {
        "path": config["source_video"]["path"],
        "sha256": config["source_video"]["expected_sha256"],
        "bytes": int(config["source_video"]["expected_bytes"]),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def load_exclusion_universe(
    project_root: Path, config: Mapping[str, Any], frozen: Mapping[str, Any]
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
        if not man.is_file():
            raise BridgeReviewError(f"missing Stage 5C batch {batch}")
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

    frozen_component = {
        "segment_id": {str(config["frozen_seed_forbidden"]["segment_id"])},
        "raw_track_id": {str(config["frozen_seed_forbidden"]["raw_track_id"])},
        "crop_id": set(),
        "source_crop_sha256": set(),
        "exact_duplicate_group": set(),
        "near_duplicate_component": set(),
        "documented_link_component": set(),
        "temporal_source_window": set(),
    }
    for batch_rows in batches.values():
        for row in batch_rows:
            if str(row.get("segment_id")) == str(
                config["frozen_seed_forbidden"]["segment_id"]
            ) or str(row.get("raw_track_id")) == str(
                config["frozen_seed_forbidden"]["raw_track_id"]
            ):
                if row.get("crop_id") not in (None, ""):
                    frozen_component["crop_id"].add(str(row["crop_id"]))
                if row.get("source_crop_sha256") not in (None, ""):
                    frozen_component["source_crop_sha256"].add(
                        str(row["source_crop_sha256"])
                    )
                if row.get("leakage_group_id") not in (None, ""):
                    frozen_component["exact_duplicate_group"].add(
                        str(row["leakage_group_id"])
                    )
                if row.get("near_duplicate_cluster_id") not in (None, ""):
                    frozen_component["near_duplicate_component"].add(
                        str(row["near_duplicate_cluster_id"])
                    )
                if row.get("documented_global_candidate_id") not in (None, ""):
                    frozen_component["documented_link_component"].add(
                        str(row["documented_global_candidate_id"])
                    )
                if row.get("timeline_bin") not in (None, ""):
                    frozen_component["temporal_source_window"].add(
                        str(row["timeline_bin"])
                    )
    for crop in frozen.get("freeze", {}).get("source_crop_lineage") or []:
        if crop.get("sha256"):
            frozen_component["source_crop_sha256"].add(str(crop["sha256"]))
        if crop.get("crop_id"):
            frozen_component["crop_id"].add(str(crop["crop_id"]))

    uni_path = (
        project_root
        / config["upstream"]["visibility_universe"]
        / "clean_review_universe"
        / "clean_review_items.jsonl"
    )
    uni_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if uni_path.is_file():
        for line in uni_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uni_by_seg[str(row["segment_id"])].append(row)

    return {
        "batches": batches,
        "keys": keys,
        "frozen_component": frozen_component,
        "universe_by_seg": uni_by_seg,
    }


def load_crop_catalog(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    baseline = project_root / config["upstream"]["baseline_crops_root"]
    man = baseline / "crop_manifest.jsonl"
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            catalog[str(row["crop_id"])] = baseline / str(row["crop_relative_path"])
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
            catalog[str(row["crop_id"])] = seg_root / str(row["crop_relative_path"])
    return catalog


def load_stage5d_b_eligible(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    inv = (
        project_root
        / config["stage5d_b_anchor_package"]["path"]
        / config["stage5d_b_anchor_package"]["inventory"]
    )
    by_seg: dict[str, dict[str, Any]] = {}
    for line in inv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eligibility_status") != "eligible_anchor_candidate":
            continue
        by_seg[str(row["segment_id"])] = row
    if len(by_seg) != 9:
        raise BridgeReviewError(
            f"expected 9 Stage 5D-B eligible anchors, got {len(by_seg)}"
        )
    return by_seg


def validate_embedding_non_frozen(
    project_root: Path,
    config: Mapping[str, Any],
    segment_id: str,
    *,
    forbidden_segment: str,
) -> dict[str, Any]:
    """Load existing embedding metadata+vector. Never for frozen seed segment."""
    if segment_id == forbidden_segment:
        raise BridgeReviewError(
            "frozen seed embedding read forbidden"
        )
    seg_root = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segmented_reid_subdir"]
    )
    meta = None
    for line in (seg_root / "segment_embedding_index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["segment_id"]) == segment_id:
            meta = row
            break
    if meta is None or not meta.get("embedding_available"):
        return {"embedding_available": False}
    data = np.load(seg_root / "segment_embeddings.npz")
    vec = data["vectors"][int(meta["embedding_row"])]
    if tuple(vec.shape) != (512,):
        return {"embedding_available": False, "reason": "bad_dim"}
    if int(np.isnan(vec).sum()) or int(np.isinf(vec).sum()):
        return {"embedding_available": False, "reason": "naninf"}
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if norm == 0.0:
        return {"embedding_available": False, "reason": "zero"}
    return {
        "embedding_available": True,
        "embedding_dimension": 512,
        "embedding_finite": True,
        "embedding_zero_vector": False,
        "embedding_norm": norm,
        "embedding_row": int(meta["embedding_row"]),
        "embedding_sha256": meta.get("embedding_sha256"),
        "representation_source": meta.get("representation_source"),
        "crop_ids": list(meta.get("crop_ids") or []),
        "embedding_path": str(
            (seg_root / "segment_embeddings.npz").relative_to(project_root)
        ),
        "meta": meta,
    }


def chain_exclusion_reasons(
    *,
    raw_track_id: int,
    segment_id: str,
    emb: Mapping[str, Any],
    crop_catalog: Mapping[str, Path],
    exclusion: Mapping[str, Any],
    forbidden_segment: str,
    forbidden_track: int,
) -> list[str]:
    reasons: list[str] = []
    keys = exclusion["keys"]
    frozen = exclusion["frozen_component"]
    uni = exclusion["universe_by_seg"].get(segment_id, [])

    if segment_id == forbidden_segment or raw_track_id == forbidden_track:
        reasons.append("original_frozen_seed_exclusion")

    if segment_id in keys["segment_id"] or str(raw_track_id) in keys["raw_track_id"]:
        reasons.append("stage5c_membership")

    for item in uni:
        if str(item.get("near_duplicate_cluster_id") or "") in frozen[
            "near_duplicate_component"
        ]:
            reasons.append("frozen_seed_near_duplicate_component")
            break
        if str(item.get("leakage_group_id") or "") in frozen["exact_duplicate_group"]:
            reasons.append("frozen_seed_exact_duplicate_group")
            break
        if str(item.get("documented_global_candidate_id") or "") in frozen[
            "documented_link_component"
        ]:
            reasons.append("frozen_seed_documented_link_component")
            break
        if str(item.get("timeline_bin") or "") in frozen["temporal_source_window"]:
            reasons.append("frozen_seed_temporal_source_window")
            break
        if str(item.get("crop_id") or "") in frozen["crop_id"]:
            reasons.append("frozen_seed_crop_id")
            break
        if str(item.get("source_crop_sha256") or "") in frozen["source_crop_sha256"]:
            reasons.append("frozen_seed_source_crop_sha256")
            break

    for item in uni:
        if str(item.get("near_duplicate_cluster_id") or "") in keys[
            "near_duplicate_component"
        ]:
            reasons.append("stage5d_near_duplicate_component")
            break
        if str(item.get("leakage_group_id") or "") in keys["exact_duplicate_group"]:
            reasons.append("stage5d_exact_duplicate_group")
            break
        if str(item.get("documented_global_candidate_id") or "") in keys[
            "documented_link_component"
        ]:
            reasons.append("stage5d_documented_link_component")
            break
        if str(item.get("timeline_bin") or "") in keys["temporal_source_window"]:
            reasons.append("stage5d_temporal_source_window")
            break
        if f"{segment_id}:{item.get('frame_index')}" in keys["frame_identity"]:
            reasons.append("stage5d_frame_identity")
            break

    if not emb.get("embedding_available"):
        reasons.append("no_existing_embedding")
    else:
        crop_ids = list(emb.get("crop_ids") or [])
        if not crop_ids:
            reasons.append("missing_crop_lineage")
        else:
            verified = 0
            for cid in crop_ids:
                path = crop_catalog.get(str(cid))
                if path is not None and path.is_file():
                    verified += 1
                    if sha256_file(path) in keys["source_crop_sha256"]:
                        reasons.append("stage5c_source_crop_sha256")
                    if str(cid) in keys["crop_id"]:
                        reasons.append("stage5c_crop_id")
                    if str(cid) in frozen["crop_id"]:
                        reasons.append("frozen_seed_crop_id")
            if verified == 0:
                reasons.append("missing_crop_lineage")

    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def assign_bridge_codes(
    eligible_keys: Sequence[tuple[int, str]],
    chains: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[tuple[int, str], str]:
    ordered = sorted(
        eligible_keys,
        key=lambda k: (
            min(int(r["frame_index"]) for r in chains[k]),
            max(int(r["frame_index"]) for r in chains[k]),
            k[0],
            k[1],
        ),
    )
    return {
        key: f"BRIDGE_CANDIDATE_{i:02d}" for i, key in enumerate(ordered, start=1)
    }


def select_sheet_frames(
    targets: Sequence[int], observation_frames: Sequence[int]
) -> list[int]:
    if not observation_frames:
        return []
    obs = sorted(set(int(f) for f in observation_frames))
    selected: list[int] = []
    seen: set[int] = set()
    for target in targets:
        nearest = min(obs, key=lambda f: (abs(f - int(target)), f))
        if nearest not in seen:
            seen.add(nearest)
            selected.append(nearest)
    return selected


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise BridgeReviewError(f"failed read frame {frame_index}")
    return frame


def _fit_display(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate_frame(
    frame: np.ndarray,
    *,
    frozen_items: Sequence[Mapping[str, Any]],
    eligible_items: Sequence[Mapping[str, Any]],
    frame_index: int,
    fps: float,
    frozen_color: Sequence[int],
    eligible_color: Sequence[int],
    bbox_thickness: int,
) -> np.ndarray:
    out = frame.copy()
    for item in frozen_items:
        x0, y0, x1, y1 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            tuple(int(c) for c in frozen_color),
            int(bbox_thickness),
        )
        for i, text in enumerate((FROZEN_VISUAL_LABEL, FROZEN_VISUAL_WARNING)):
            ty = max(18, y0 - 8 - i * 18)
            cv2.putText(
                out,
                text,
                (x0, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if i == 0 else 0.35,
                (10, 10, 10),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (x0, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if i == 0 else 0.35,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
    for item in eligible_items:
        x0, y0, x1, y1 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            tuple(int(c) for c in eligible_color),
            int(bbox_thickness),
        )
        label = str(item["bridge_candidate_code"])
        tx, ty = x0, max(18, y0 - 6)
        cv2.putText(
            out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
        )
    header = [
        f"frame={frame_index}",
        f"t={frame_index / fps:.3f}s",
        "SEED TO ELIGIBLE BRIDGE REVIEW",
    ]
    y = 28
    for text in header:
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 26
    return out


def render_sheet(panels: Sequence[Mapping[str, Any]]) -> np.ndarray:
    cols = min(5, max(1, len(panels)))
    rows_n = int(math.ceil(len(panels) / cols)) if panels else 1
    panel_w, panel_h = 400, 260
    sheet = np.full((rows_n * panel_h, cols * panel_w, 3), 16, dtype=np.uint8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        tile = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
        label = f"f={panel['frame_index']}  t={panel['time_sec']:.2f}s"
        cv2.putText(
            tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
        )
        disp = _fit_display(panel["image"], panel_w - 12, panel_h - 40)
        dh, dw = disp.shape[:2]
        ox = (panel_w - dw) // 2
        oy = 32
        tile[oy : oy + dh, ox : ox + dw] = disp
        y0 = r * panel_h
        x0 = c * panel_w
        sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = tile
    return sheet


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise BridgeReviewError(f"FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT png {path}")


def write_clip(
    path: Path,
    *,
    video_path: Path,
    start: int,
    end: int,
    frozen_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    eligible_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    fps: float,
    frozen_color: Sequence[int],
    eligible_color: Sequence[int],
    bbox_thickness: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY open_clip")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise BridgeReviewError("FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT writer")
    count = 0
    frozen_shown = 0
    eligible_shown = 0
    for frame_index in range(start, end + 1):
        frame = read_frame(cap, frame_index)
        fr = frozen_by_frame.get(frame_index, [])
        el = eligible_by_frame.get(frame_index, [])
        if fr:
            frozen_shown += 1
        if el:
            eligible_shown += 1
        annotated = annotate_frame(
            frame,
            frozen_items=fr,
            eligible_items=el,
            frame_index=frame_index,
            fps=fps,
            frozen_color=frozen_color,
            eligible_color=eligible_color,
            bbox_thickness=bbox_thickness,
        )
        writer.write(annotated)
        count += 1
    writer.release()
    cap.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise BridgeReviewError("FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT empty_clip")
    return {
        "source_frame_range": [start, end],
        "frame_count": count,
        "duration_sec": count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "fps": fps,
        "width": w,
        "height": h,
        "frozen_seed_shown_frame_count": frozen_shown,
        "eligible_candidate_shown_frame_count": eligible_shown,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1c2_seed_to_eligible_bridge_contract_v1",
        "target_id": "target_001",
        "frozen_seed_used_for_human_visual_continuity_only": True,
        "frozen_seed_embedding_used": False,
        "frozen_seed_enrollment_allowed": False,
        "existing_eligible_sources_only": True,
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_new_embedding": True,
        "no_ocr": True,
        "no_similarity": True,
        "no_automatic_continuation_assignment": True,
        "human_decision_required": True,
        "no_anchor_derivation": True,
        "no_gallery_membership": True,
        "unknown_identity_preserved": True,
        "eligible_bridge_manual_selection": 0,
        "derived_anchors": 0,
        "approved_anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "frozen_target_seed_count": 1,
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise BridgeReviewError("FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT final_exists")

    frozen = validate_frozen_and_b1c(project_root, config)
    closure = validate_stage5c(project_root, config)
    assets = validate_assets(project_root, config)
    exclusion = load_exclusion_universe(project_root, config, frozen)
    crop_catalog = load_crop_catalog(project_root, config)
    stage5d_b = load_stage5d_b_eligible(project_root, config)

    start = int(config["bridge_window"]["start_frame"])
    end = int(config["bridge_window"]["end_frame"])
    fps = float(assets["fps"])
    forbidden_seg = str(config["frozen_seed_forbidden"]["segment_id"])
    forbidden_track = int(config["frozen_seed_forbidden"]["raw_track_id"])
    frozen_color = list(config["visualization"]["frozen_bbox_color_bgr"])
    eligible_color = list(config["visualization"]["eligible_bbox_color_bgr"])
    bbox_thickness = int(config["visualization"]["bbox_thickness"])

    obs_path = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segment_view_subdir"]
        / "segment_observations.jsonl"
    )
    chains: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    frozen_chain: list[dict[str, Any]] = []
    window_obs = 0
    for line in obs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fi = int(row["frame_index"])
        if not (start <= fi <= end):
            continue
        src = row.get("source_observation") or {}
        bbox = src.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise BridgeReviewError("missing bbox in bridge window")
        entry = {
            "frame_index": fi,
            "bbox_xyxy": [float(x) for x in bbox],
            "timestamp_sec": float(src.get("timestamp_sec") or fi / fps),
        }
        tid = int(row["raw_track_id"])
        sid = str(row["segment_id"])
        if tid == forbidden_track or sid == forbidden_seg:
            frozen_chain.append(entry)
        else:
            chains[(tid, sid)].append(entry)
        window_obs += 1

    reason_counts: Counter[str] = Counter()
    audits: list[dict[str, Any]] = []
    eligible_keys: list[tuple[int, str]] = []
    emb_cache: dict[str, dict[str, Any]] = {}
    frozen_seed_embedding_used = False

    for key, rows in sorted(chains.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        tid, sid = key
        if sid not in emb_cache:
            emb_cache[sid] = validate_embedding_non_frozen(
                project_root,
                config,
                sid,
                forbidden_segment=forbidden_seg,
            )
        emb = emb_cache[sid]
        reasons = chain_exclusion_reasons(
            raw_track_id=tid,
            segment_id=sid,
            emb=emb,
            crop_catalog=crop_catalog,
            exclusion=exclusion,
            forbidden_segment=forbidden_seg,
            forbidden_track=forbidden_track,
        )
        audit = {
            "raw_track_id": tid,
            "segment_id": sid,
            "observation_count_in_window": len(rows),
            "exclusion_reasons": reasons,
            "eligible_for_anchor_derivation": len(reasons) == 0,
            "embedding_available": bool(emb.get("embedding_available")),
            "stage5d_b_lineage_reference": (
                stage5d_b.get(sid, {}).get("anchor_candidate_id")
            ),
        }
        audits.append(audit)
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
        else:
            eligible_keys.append(key)

    # Frozen seed is never an eligible bridge candidate.
    audits.append(
        {
            "raw_track_id": forbidden_track,
            "segment_id": forbidden_seg,
            "observation_count_in_window": len(frozen_chain),
            "exclusion_reasons": ["original_frozen_seed_exclusion"],
            "eligible_for_anchor_derivation": False,
            "embedding_available": False,
            "frozen_seed_visual_reference_only": True,
            "frozen_seed_embedding_used": False,
        }
    )
    reason_counts["original_frozen_seed_exclusion"] += 1

    code_map = assign_bridge_codes(eligible_keys, chains) if eligible_keys else {}

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_seed_to_eligible_bridge_review_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise BridgeReviewError("FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    video_path = project_root / config["source_video"]["path"]
    try:
        mapping_rows: list[dict[str, Any]] = []
        frozen_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        eligible_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        drawable_frames: set[int] = set()

        for row in frozen_chain:
            fi = int(row["frame_index"])
            frozen_by_frame[fi].append({"bbox_xyxy": row["bbox_xyxy"]})
            drawable_frames.add(fi)

        for key in sorted(code_map.keys(), key=lambda k: code_map[k]):
            tid, sid = key
            rows = sorted(chains[key], key=lambda r: r["frame_index"])
            emb = emb_cache[sid]
            code = code_map[key]
            stage5d_b_ref = stage5d_b.get(sid, {}).get("anchor_candidate_id")
            crop_lineage = []
            for cid in emb.get("crop_ids") or []:
                path = crop_catalog.get(str(cid))
                entry = {
                    "crop_id": cid,
                    "path": str(path) if path else None,
                    "exists": False,
                    "sha256": None,
                }
                if path is not None and path.is_file():
                    entry["exists"] = True
                    entry["sha256"] = sha256_file(path)
                crop_lineage.append(entry)
            for row in rows:
                fi = int(row["frame_index"])
                eligible_by_frame[fi].append(
                    {
                        "bridge_candidate_code": code,
                        "bbox_xyxy": row["bbox_xyxy"],
                    }
                )
                drawable_frames.add(fi)
            for fi in eligible_by_frame:
                eligible_by_frame[fi].sort(key=lambda x: x["bridge_candidate_code"])

            mapping_rows.append(
                {
                    "bridge_candidate_code": code,
                    "target_id": "target_001",
                    "raw_track_id": tid,
                    "segment_id": sid,
                    "stage5d_b_lineage_reference": stage5d_b_ref,
                    "observation_frames": [r["frame_index"] for r in rows],
                    "first_frame": rows[0]["frame_index"],
                    "last_frame": rows[-1]["frame_index"],
                    "bbox_per_observation": [
                        {"frame_index": r["frame_index"], "bbox_xyxy": r["bbox_xyxy"]}
                        for r in rows
                    ],
                    "crop_lineage": crop_lineage,
                    "embedding_lineage": {
                        "available": True,
                        "path": emb.get("embedding_path"),
                        "dimension": emb.get("embedding_dimension"),
                        "row": emb.get("embedding_row"),
                        "sha256": emb.get("embedding_sha256"),
                        "norm": emb.get("embedding_norm"),
                        "is_original_frozen_seed_embedding": False,
                    },
                    "stage5c_exclusion_audit": {"excluded": False, "reasons": []},
                    "stage5d_exclusion_audit": {"excluded": False, "reasons": []},
                    "eligible_for_anchor_derivation": True,
                    "selected_as_target_continuation": "",
                    "manual_target_confirmed": "",
                    "manual_identity_continuity_observed": "",
                    "manual_crop_valid": "",
                    "manual_target_dominant": "",
                    "manual_human_verified_number_seen": "",
                }
            )

        pkg = tmp / "review_packages" / "target_001_seed_to_eligible_bridge"
        pkg.mkdir(parents=True)

        sheet_specs = [
            ("bridge_contact_sheet_01.png", list(config["bridge_window"]["sheet_01_targets"])),
            ("bridge_contact_sheet_02.png", list(config["bridge_window"]["sheet_02_targets"])),
            ("bridge_contact_sheet_03.png", list(config["bridge_window"]["sheet_03_targets"])),
        ]
        sheet_frame_sets: list[list[int]] = []
        drawable_sorted = sorted(drawable_frames)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise BridgeReviewError("BLOCKED_STAGE5D_B1C2_SOURCE_VIDEO_INTEGRITY open")
        sheet_rels = []
        for name, targets in sheet_specs:
            frames = select_sheet_frames(targets, drawable_sorted)
            sheet_frame_sets.append(frames)
            panels = []
            for fi in frames:
                frame = read_frame(cap, fi)
                annotated = annotate_frame(
                    frame,
                    frozen_items=frozen_by_frame.get(fi, []),
                    eligible_items=eligible_by_frame.get(fi, []),
                    frame_index=fi,
                    fps=fps,
                    frozen_color=frozen_color,
                    eligible_color=eligible_color,
                    bbox_thickness=bbox_thickness,
                )
                panels.append(
                    {"frame_index": fi, "time_sec": fi / fps, "image": annotated}
                )
            write_png(pkg / name, render_sheet(panels))
            sheet_rels.append(
                f"review_packages/target_001_seed_to_eligible_bridge/{name}"
            )
        cap.release()

        clip_rel = (
            "review_packages/target_001_seed_to_eligible_bridge/"
            "target_001_seed_to_eligible_bridge.mp4"
        )
        clip_meta = write_clip(
            pkg / "target_001_seed_to_eligible_bridge.mp4",
            video_path=video_path,
            start=start,
            end=end,
            frozen_by_frame=frozen_by_frame,
            eligible_by_frame=eligible_by_frame,
            fps=fps,
            frozen_color=frozen_color,
            eligible_color=eligible_color,
            bbox_thickness=bbox_thickness,
        )

        if eligible_keys:
            final_status = (
                "COMPLETED_STAGE5D_B1C2_SEED_TO_ELIGIBLE_BRIDGE_REVIEW_READY"
            )
            exact_next = (
                "STAGE5D-B1D_TARGET_001_BRIDGE_SOURCE_SELECTION_FREEZE_AND_ANCHOR_DERIVATION"
            )
        else:
            final_status = "COMPLETED_STAGE5D_B1C2_NO_ELIGIBLE_BRIDGE_SOURCE"
            exact_next = "STAGE5D-B1E_TARGET_001_EXTERNAL_ENROLLMENT_CLIP_DESIGN"

        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        with (inv_dir / "target_001_seed_to_eligible_bridge_mapping.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in mapping_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        elig_dir = tmp / "eligibility"
        elig_dir.mkdir(parents=True)
        write_json(
            elig_dir / "target_001_bridge_source_eligibility_audit.json",
            {
                "schema_version": "reid_stage5d_b1c2_bridge_eligibility_audit_v1",
                "bridge_window": [start, end],
                "window_observation_count": window_obs,
                "chain_count_excluding_frozen": len(chains),
                "frozen_seed_observation_count": len(frozen_chain),
                "eligible_count": len(eligible_keys),
                "ineligible_count": len(chains) - len(eligible_keys),
                "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                "audits": audits,
                "stage5d_b_eligible_inventory_count": len(stage5d_b),
                "stage5d_b_lineage_hits_in_window": sum(
                    1
                    for k in eligible_keys
                    if k[1] in stage5d_b
                ),
                "forbidden_original_seed": {
                    "code": FROZEN_SEED_CODE,
                    "segment_id": forbidden_seg,
                    "raw_track_id": forbidden_track,
                    "visual_reference_only": True,
                    "embedding_used": False,
                    "enrollment_allowed": False,
                },
            },
        )

        br = tmp / "bridge_review"
        br.mkdir(parents=True)
        write_json(br / "target_001_seed_to_eligible_bridge_contract.json", build_contract())
        write_json(
            br / "target_001_seed_to_eligible_bridge_manifest.json",
            {
                "schema_version": "reid_stage5d_b1c2_bridge_manifest_v1",
                "target_id": "target_001",
                "bridge_window": [start, end],
                "sheet_paths": sheet_rels,
                "sheet_frames": sheet_frame_sets,
                "clip_path": clip_rel,
                "clip": clip_meta,
                "bridge_candidate_count": len(mapping_rows),
                "eligible_bridge_manual_selection": 0,
                "derived_anchors": 0,
                "gallery_members": 0,
                "original_frozen_seed_code": FROZEN_SEED_CODE,
                "frozen_seed_embedding_used": frozen_seed_embedding_used,
            },
        )

        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        tpl_path = tpl_dir / "target_001_seed_to_eligible_bridge_review_template.csv"
        with tpl_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for row in mapping_rows:
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "bridge_candidate_code": row["bridge_candidate_code"],
                        "segment_id": row["segment_id"],
                        "raw_track_id": row["raw_track_id"],
                        "first_frame": row["first_frame"],
                        "last_frame": row["last_frame"],
                        "selected_as_target_continuation": "",
                        "manual_target_confirmed": "",
                        "manual_identity_continuity_observed": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_human_verified_number_seen": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )

        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(
            tmp / "runtime" / "runtime.json",
            {
                "schema_version": "reid_stage5d_b1c2_runtime_v1",
                "started_at": started,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "frozen_seed_embedding_used": False,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        actual_png = len(list(tmp.rglob("*.png")))
        actual_mp4 = len(list(tmp.rglob("*.mp4")))
        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        if actual_png != 3 or actual_mp4 != 1 or jpeg_count:
            raise BridgeReviewError(
                f"FAILED_STAGE5D_B1C2_ATOMIC_OUTPUT budget png={actual_png} mp4={actual_mp4}"
            )

        summary = {
            "schema_version": "reid_stage5d_b1c2_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": frozen["target_definition"]["target_alias"],
            "frozen_target_seed_count": 1,
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "original_frozen_seed_segment_id": forbidden_seg,
            "original_frozen_seed_raw_track_id": forbidden_track,
            "frozen_seed_visual_only": True,
            "frozen_seed_embedding_used": False,
            "frozen_seed_enrollment_allowed": False,
            "bridge_window_frames": [start, end],
            "bridge_window_time_sec": [start / fps, end / fps],
            "window_observation_count": window_obs,
            "frozen_seed_observation_count": len(frozen_chain),
            "chain_count_excluding_frozen": len(chains),
            "eligible_bridge_candidate_count": len(eligible_keys),
            "ineligible_candidate_count": len(chains) - len(eligible_keys),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "bridge_candidate_codes": [code_map[k] for k in sorted(code_map, key=lambda x: code_map[x])],
            "stage5d_b_lineage_links": [
                {
                    "bridge_candidate_code": code_map[k],
                    "segment_id": k[1],
                    "stage5d_b_lineage_reference": stage5d_b.get(k[1], {}).get(
                        "anchor_candidate_id"
                    ),
                }
                for k in sorted(code_map, key=lambda x: code_map[x])
            ],
            "sheet_frames": sheet_frame_sets,
            "png_count": 3,
            "mp4_count": 1,
            "jpeg_count": 0,
            "eligible_bridge_manual_selection": 0,
            "derived_anchors": 0,
            "approved_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "similarity_ranking_rows": 0,
            "new_detection": 0,
            "new_tracking": 0,
            "new_embedding": 0,
            "ocr": 0,
            "stage5c_closure": closure,
            "source_video": assets,
            "clip": clip_meta,
            "exact_next_gate": exact_next,
        }
        write_json(tmp / "stage5d_b1c2_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1c2_manifest.json",
            {
                "schema_version": "reid_stage5d_b1c2_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "sheet_paths": sheet_rels,
                "clip_path": clip_rel,
                "clip_sha256": clip_meta["sha256"],
                "bridge_candidate_count": len(mapping_rows),
                "original_frozen_seed_code": FROZEN_SEED_CODE,
                "frozen_seed_embedding_used": False,
                "gallery_members": 0,
                "derived_anchors": 0,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1c2_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except BridgeReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "eligible": summary["eligible_bridge_candidate_count"],
                "codes": summary["bridge_candidate_codes"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
