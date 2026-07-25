#!/usr/bin/env python3
"""Stage 5D-B1C — alternate eligible anchor source review for target_001.

Preserves frozen SEED_CANDIDATE_07 / raw_222_full. Does not select or freeze an
alternate source. No OCR, similarity, new inference, or gallery membership.
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
CONFIG_SCHEMA = "reid_target_alternate_eligible_source_review_config_v1"
FROZEN_SEED_CODE = "SEED_CANDIDATE_07"
TEMPLATE_FIELDS = (
    "target_id",
    "review_window_start_frame",
    "review_window_end_frame",
    "selected_alternate_neutral_seed_code",
    "manual_target_confirmed",
    "manual_human_verified_number_seen",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_identity_continuity_observed",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")


class AlternateSourceError(RuntimeError):
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
        raise AlternateSourceError("unexpected config schema")
    if not config.get("offline_required"):
        raise AlternateSourceError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise AlternateSourceError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_target_alternate_eligible_source_review.py",
        "configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml",
        "tests/test_reid_target_alternate_eligible_source_review.py",
        "docs/setup/stage5d-target-alternate-eligible-anchor-source-review.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise AlternateSourceError(
                    "BLOCKED_STAGE5D_B1C_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Freeze target 001 seed and derive anchor candidates":
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_GIT_CONTRACT_MISMATCH message")
    return head


def resolve_snapshot_sha(snapshot_path: Path) -> str:
    if not snapshot_path.is_file():
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH snapshot_sidecar"
        )
    sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_sha = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (sidecar_sha == man_sha == actual):
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_frozen_seed(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = project_root / config["stage5d_b1b_package"]["path"]
    summary = load_json(root / "stage5d_b1b_summary.json")
    freeze = load_json(
        root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
    )
    elig = load_json(
        root / "eligibility" / "target_001_selected_seed_source_eligibility.json"
    )
    td = load_json(project_root / config["target_definition"]["path"])
    exp = config["stage5d_b1b_package"]

    if summary.get("final_status") != exp["expected_final_status"]:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH target"
        )
    if td.get("target_definition_frozen") is not True:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH td"
        )
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH alias"
        )
    if summary.get("selected_neutral_seed_code") != exp["expected_selected_seed"]:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH seed"
        )
    if summary.get("resolved_segment_id") != exp["expected_segment_id"]:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH segment"
        )
    if int(summary.get("resolved_raw_track_id")) != int(exp["expected_raw_track_id"]):
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH track"
        )
    if summary.get("selected_seed_source_eligibility") != exp["expected_eligibility"]:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH eligibility"
        )
    hits = elig.get("stage5c_batch_hits") or {}
    if "holdout_primary" not in hits:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH holdout"
        )
    for key in (
        "manual_target_confirmed",
        "manual_human_verified_number_seen",
    ):
        if summary.get(key) != "yes":
            raise AlternateSourceError(
                f"BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH {key}"
            )
    hd = freeze.get("human_decision") or {}
    if hd.get("manual_crop_valid") != "yes" or hd.get("manual_target_dominant") != "yes":
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH human_flags"
        )
    if freeze.get("target_seed_frozen") is not True:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH not_frozen"
        )
    if int(summary.get("derived_anchor_candidate_count") or 0) != 0:
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH derived"
        )
    for key in ("gallery_members", "prototypes", "identity_assignments"):
        if int(summary.get(key) or 0) != 0:
            raise AlternateSourceError(
                f"BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH {key}"
            )
    if freeze.get("OCR_used") or freeze.get("similarity_used"):
        raise AlternateSourceError(
            "BLOCKED_STAGE5D_B1C_FROZEN_SEED_CONTRACT_MISMATCH ocr_sim"
        )

    snap_sha = resolve_snapshot_sha(Path(exp["snapshot_path"]))
    return {
        "summary": summary,
        "freeze": freeze,
        "eligibility": elig,
        "target_definition": td,
        "snapshot_sha256": snap_sha,
        "freeze_sha256": sha256_file(
            root / "seed_freeze" / "target_001_manual_seed_selection_frozen.json"
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
            raise AlternateSourceError(
                f"BLOCKED_STAGE5D_B1C_EXCLUSION_CONTRACT {key}={pol.get(key)!r}"
            )
    return required


def validate_assets(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    video = project_root / config["source_video"]["path"]
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if video.stat().st_size != int(config["source_video"]["expected_bytes"]):
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY bytes")
    if sha256_file(video) != config["source_video"]["expected_sha256"]:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY sha")
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY yolo_b")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY yolo")
    if sha256_file(osnet) != config["osnet_checkpoint"]["expected_sha256"]:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY osnet")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY open")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames != int(config["source_video"]["expected_frames"]):
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY frames")
    if width != int(config["source_video"]["expected_width"]):
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY width")
    if height != int(config["source_video"]["expected_height"]):
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY height")
    if abs(fps - float(config["source_video"]["expected_fps"])) > 1e-6:
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY fps")
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
            raise AlternateSourceError(
                f"BLOCKED_STAGE5D_B1C_EXCLUSION_CONTRACT missing {batch}"
            )
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

    # Frozen seed exclusion component from holdout/Stage5C rows sharing seed lineage.
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

    # Also include freeze artifact crop SHAs if present.
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


def validate_embedding(
    project_root: Path, config: Mapping[str, Any], segment_id: str
) -> dict[str, Any]:
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

    # Frozen seed component overlap.
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

    # General Stage 5D leakage keys (beyond pure membership already counted).
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

    # Deduplicate preserve order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def assign_alt_codes(
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
    return {key: f"ALT_SEED_CANDIDATE_{i:02d}" for i, key in enumerate(ordered, start=1)}


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
        raise AlternateSourceError(f"failed read frame {frame_index}")
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
    items: Sequence[Mapping[str, Any]],
    frame_index: int,
    fps: float,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> np.ndarray:
    out = frame.copy()
    for item in items:
        x0, y0, x1, y1 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            tuple(int(c) for c in bbox_color),
            int(bbox_thickness),
        )
        label = str(item["alternate_neutral_seed_code"])
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
        "EARLY HUMAN NUMBER REVIEW",
    ]
    y = 28
    for text in header:
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 28
    return out


def render_sheet(panels: Sequence[Mapping[str, Any]]) -> np.ndarray:
    cols = min(4, max(1, len(panels)))
    rows_n = int(math.ceil(len(panels) / cols)) if panels else 1
    panel_w, panel_h = 420, 280
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
        raise AlternateSourceError(f"FAILED_STAGE5D_B1C_ATOMIC_OUTPUT png {path}")


def write_clip(
    path: Path,
    *,
    video_path: Path,
    start: int,
    end: int,
    frame_items: Mapping[int, Sequence[Mapping[str, Any]]],
    fps: float,
    bbox_color: Sequence[int],
    bbox_thickness: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AlternateSourceError("BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY open_clip")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise AlternateSourceError("FAILED_STAGE5D_B1C_ATOMIC_OUTPUT writer")
    count = 0
    for frame_index in range(start, end + 1):
        frame = read_frame(cap, frame_index)
        annotated = annotate_frame(
            frame,
            items=frame_items.get(frame_index, []),
            frame_index=frame_index,
            fps=fps,
            bbox_color=bbox_color,
            bbox_thickness=bbox_thickness,
        )
        writer.write(annotated)
        count += 1
    writer.release()
    cap.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise AlternateSourceError("FAILED_STAGE5D_B1C_ATOMIC_OUTPUT empty_clip")
    return {
        "frame_count": count,
        "duration_sec": count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "fps": fps,
        "width": w,
        "height": h,
    }


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1c_alternate_source_review_contract_v1",
        "target_id": "target_001",
        "frozen_original_seed_preserved": True,
        "frozen_original_seed_code": FROZEN_SEED_CODE,
        "original_excluded_seed_not_used_for_scoring_or_enrollment": True,
        "existing_eligible_observations_only": True,
        "no_new_detection": True,
        "no_new_tracking": True,
        "no_new_embedding": True,
        "no_ocr": True,
        "no_similarity": True,
        "no_automatic_identity": True,
        "human_selection_required": True,
        "alternative_selection_requires_separate_freeze_gate": True,
        "no_gallery_membership": True,
        "unknown_identity_preserved": True,
        "alternate_manual_selection": 0,
        "derived_anchors": 0,
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
        raise AlternateSourceError("FAILED_STAGE5D_B1C_ATOMIC_OUTPUT final_exists")

    frozen = validate_frozen_seed(project_root, config)
    closure = validate_stage5c(project_root, config)
    assets = validate_assets(project_root, config)
    exclusion = load_exclusion_universe(project_root, config, frozen)
    crop_catalog = load_crop_catalog(project_root, config)

    start = int(config["review_window"]["start_frame"])
    end = int(config["review_window"]["end_frame"])
    targets = [int(x) for x in config["review_window"]["sheet_target_frames"]]
    fps = float(assets["fps"])
    forbidden_seg = str(config["frozen_seed_forbidden"]["segment_id"])
    forbidden_track = int(config["frozen_seed_forbidden"]["raw_track_id"])
    bbox_color = list(config["visualization"]["bbox_color_bgr"])
    bbox_thickness = int(config["visualization"]["bbox_thickness"])

    # Ensure original seed embedding is not loaded for scoring: only read other segs.
    obs_path = (
        project_root
        / config["upstream"]["stage5_replay"]
        / config["upstream"]["segment_view_subdir"]
        / "segment_observations.jsonl"
    )
    chains: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
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
            raise AlternateSourceError("missing bbox in window")
        key = (int(row["raw_track_id"]), str(row["segment_id"]))
        chains[key].append(
            {
                "frame_index": fi,
                "bbox_xyxy": [float(x) for x in bbox],
                "timestamp_sec": float(src.get("timestamp_sec") or fi / fps),
                "source_observation_sha256": row.get("source_observation_sha256"),
            }
        )
        window_obs += 1

    reason_counts: Counter[str] = Counter()
    audits: list[dict[str, Any]] = []
    eligible_keys: list[tuple[int, str]] = []
    emb_cache: dict[str, dict[str, Any]] = {}

    for key, rows in sorted(chains.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        tid, sid = key
        if sid not in emb_cache:
            # Never use track 222 / raw_222_full embedding for scoring; still may
            # inspect availability only for exclusion reporting.
            emb_cache[sid] = validate_embedding(project_root, config, sid)
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
        # Explicitly forbid using original seed embedding/path for enrollment.
        if sid == forbidden_seg or tid == forbidden_track:
            if "original_frozen_seed_exclusion" not in reasons:
                reasons.append("original_frozen_seed_exclusion")
        audit = {
            "raw_track_id": tid,
            "segment_id": sid,
            "observation_count_in_window": len(rows),
            "exclusion_reasons": reasons,
            "eligible_for_anchor_derivation": len(reasons) == 0,
            "embedding_available": bool(emb.get("embedding_available")),
        }
        audits.append(audit)
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
        else:
            eligible_keys.append(key)

    # Controlled no-eligible: atomic output root without sheet/clip review media.
    code_map = assign_alt_codes(eligible_keys, chains) if eligible_keys else {}

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_alternate_eligible_source_review_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise AlternateSourceError("FAILED_STAGE5D_B1C_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    video_path = project_root / config["source_video"]["path"]
    try:
        mapping_rows: list[dict[str, Any]] = []
        frame_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        eligible_obs_frames: set[int] = set()

        for key in sorted(code_map.keys(), key=lambda k: code_map[k]):
            tid, sid = key
            rows = sorted(chains[key], key=lambda r: r["frame_index"])
            emb = emb_cache[sid]
            code = code_map[key]
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
                frame_items[int(row["frame_index"])].append(
                    {
                        "alternate_neutral_seed_code": code,
                        "bbox_xyxy": row["bbox_xyxy"],
                    }
                )
                eligible_obs_frames.add(int(row["frame_index"]))
            for fi in frame_items:
                frame_items[fi].sort(key=lambda x: x["alternate_neutral_seed_code"])

            mapping_rows.append(
                {
                    "alternate_neutral_seed_code": code,
                    "target_id": "target_001",
                    "raw_track_id": tid,
                    "segment_id": sid,
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
                        "dimension": emb.get("embedding_dimension"),
                        "row": emb.get("embedding_row"),
                        "sha256": emb.get("embedding_sha256"),
                        "norm": emb.get("embedding_norm"),
                        "representation_source": emb.get("representation_source"),
                        # Explicit: never the frozen seed embedding.
                        "is_original_frozen_seed_embedding": False,
                    },
                    "stage5c_exclusion_audit": {
                        "excluded": False,
                        "membership_batches": [],
                    },
                    "stage5d_exclusion_audit": {
                        "excluded": False,
                        "reasons": [],
                    },
                    "eligible_for_anchor_derivation": True,
                    "manual_target_confirmed": "",
                    "manual_human_verified_number_seen": "",
                    "selected_as_alternate_seed": False,
                }
            )

        png_count = 0
        mp4_count = 0
        sheet_rel = None
        clip_rel = None
        clip_meta = None
        sheet_frames: list[int] = []

        pkg = tmp / "review_packages" / "target_001_alternate_eligible_source_review"
        pkg.mkdir(parents=True)

        if eligible_keys:
            sheet_frames = select_sheet_frames(targets, sorted(eligible_obs_frames))
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise AlternateSourceError(
                    "BLOCKED_STAGE5D_B1C_SOURCE_VIDEO_INTEGRITY open"
                )
            panels = []
            for fi in sheet_frames:
                frame = read_frame(cap, fi)
                annotated = annotate_frame(
                    frame,
                    items=frame_items.get(fi, []),
                    frame_index=fi,
                    fps=fps,
                    bbox_color=bbox_color,
                    bbox_thickness=bbox_thickness,
                )
                panels.append(
                    {
                        "frame_index": fi,
                        "time_sec": fi / fps,
                        "image": annotated,
                    }
                )
            cap.release()
            write_png(pkg / "alternate_seed_selection_sheet_01.png", render_sheet(panels))
            sheet_rel = (
                "review_packages/target_001_alternate_eligible_source_review/"
                "alternate_seed_selection_sheet_01.png"
            )
            png_count = 1

            clip_rel = (
                "review_packages/target_001_alternate_eligible_source_review/"
                "target_001_alternate_eligible_source_window.mp4"
            )
            clip_meta = write_clip(
                pkg / "target_001_alternate_eligible_source_window.mp4",
                video_path=video_path,
                start=start,
                end=end,
                frame_items=frame_items,
                fps=fps,
                bbox_color=bbox_color,
                bbox_thickness=bbox_thickness,
            )
            mp4_count = 1
            final_status = (
                "COMPLETED_STAGE5D_B1C_ALTERNATE_ELIGIBLE_SOURCE_REVIEW_READY"
            )
            exact_next = (
                "STAGE5D-B1D_TARGET_001_ALTERNATE_SOURCE_SELECTION_FREEZE_AND_ANCHOR_DERIVATION"
            )
        else:
            final_status = "COMPLETED_STAGE5D_B1C_NO_ELIGIBLE_SOURCE_IN_EARLY_WINDOW"
            exact_next = "STAGE5D-B1C2_TARGET_001_ADDITIONAL_HUMAN_WINDOW_REVIEW"

        inv_dir = tmp / "inventory"
        inv_dir.mkdir(parents=True)
        with (inv_dir / "target_001_alternate_eligible_source_mapping.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in mapping_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        elig_dir = tmp / "eligibility"
        elig_dir.mkdir(parents=True)
        write_json(
            elig_dir / "target_001_early_window_source_eligibility_audit.json",
            {
                "schema_version": "reid_stage5d_b1c_early_window_eligibility_audit_v1",
                "review_window": [start, end],
                "window_observation_count": window_obs,
                "chain_count": len(chains),
                "eligible_count": len(eligible_keys),
                "ineligible_count": len(chains) - len(eligible_keys),
                "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                "audits": audits,
                "forbidden_original_seed": {
                    "code": FROZEN_SEED_CODE,
                    "segment_id": forbidden_seg,
                    "raw_track_id": forbidden_track,
                    "used_for_scoring": False,
                    "used_for_enrollment": False,
                    "embedding_used": False,
                },
            },
        )

        asr = tmp / "alternate_source_review"
        asr.mkdir(parents=True)
        write_json(
            asr / "target_001_alternate_source_review_contract.json", build_contract()
        )
        write_json(
            asr / "target_001_alternate_source_review_manifest.json",
            {
                "schema_version": "reid_stage5d_b1c_alternate_source_review_manifest_v1",
                "target_id": "target_001",
                "review_window": [start, end],
                "sheet_frames": sheet_frames,
                "sheet_path": sheet_rel,
                "clip_path": clip_rel,
                "clip": clip_meta,
                "alternate_neutral_candidate_count": len(mapping_rows),
                "alternate_manual_selection": 0,
                "derived_anchors": 0,
                "gallery_members": 0,
                "original_frozen_seed_code": FROZEN_SEED_CODE,
                "original_frozen_seed_freeze_sha256": frozen["freeze_sha256"],
            },
        )

        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        tpl_path = tpl_dir / "target_001_alternate_eligible_source_review_template.csv"
        with tpl_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "target_id": "target_001",
                    "review_window_start_frame": start,
                    "review_window_end_frame": end,
                    "selected_alternate_neutral_seed_code": "",
                    "manual_target_confirmed": "",
                    "manual_human_verified_number_seen": "",
                    "manual_crop_valid": "",
                    "manual_target_dominant": "",
                    "manual_identity_continuity_observed": "",
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
                "schema_version": "reid_stage5d_b1c_runtime_v1",
                "started_at": started,
                "offline_required": True,
                "network_download": 0,
                "new_detection": 0,
                "new_tracking": 0,
                "new_embedding": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "original_frozen_seed_embedding_used": False,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)

        actual_png = len(list(tmp.rglob("*.png")))
        actual_mp4 = len(list(tmp.rglob("*.mp4")))
        jpeg_count = len(list(tmp.rglob("*.jpg"))) + len(list(tmp.rglob("*.jpeg")))
        if actual_png != png_count or actual_mp4 != mp4_count or jpeg_count:
            raise AlternateSourceError(
                f"FAILED_STAGE5D_B1C_ATOMIC_OUTPUT budget png={actual_png} mp4={actual_mp4}"
            )

        summary = {
            "schema_version": "reid_stage5d_b1c_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": frozen["target_definition"]["target_alias"],
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "original_frozen_seed_segment_id": forbidden_seg,
            "original_frozen_seed_raw_track_id": forbidden_track,
            "original_frozen_seed_count": 1,
            "original_seed_exclusion": frozen["summary"][
                "selected_seed_source_eligibility"
            ],
            "original_seed_embedding_used": False,
            "review_window_frames": [start, end],
            "review_window_time_sec": [start / fps, end / fps],
            "preferred_human_reference_region": list(
                config["review_window"]["preferred_human_reference_region"]
            ),
            "window_observation_count": window_obs,
            "chain_count": len(chains),
            "eligible_candidate_count": len(eligible_keys),
            "ineligible_candidate_count": len(chains) - len(eligible_keys),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "neutral_alternate_candidate_count": len(mapping_rows),
            "sheet_frames": sheet_frames,
            "png_count": png_count,
            "mp4_count": mp4_count,
            "jpeg_count": 0,
            "alternate_manual_selection": 0,
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
            "b1b_snapshot_sha256": frozen["snapshot_sha256"],
            "source_video": assets,
            "exact_next_gate": exact_next,
        }
        write_json(tmp / "stage5d_b1c_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1c_manifest.json",
            {
                "schema_version": "reid_stage5d_b1c_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "sheet_path": sheet_rel,
                "clip_path": clip_rel,
                "clip_sha256": (clip_meta or {}).get("sha256"),
                "neutral_alternate_candidate_count": len(mapping_rows),
                "original_frozen_seed_code": FROZEN_SEED_CODE,
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

    return load_json(final_dir / "stage5d_b1c_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except AlternateSourceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "eligible": summary["eligible_candidate_count"],
                "alt_codes": summary["neutral_alternate_candidate_count"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
