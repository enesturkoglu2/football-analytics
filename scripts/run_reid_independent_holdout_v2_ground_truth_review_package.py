#!/usr/bin/env python3
"""Stage 5D-F3K — holdout v2 similarity-blind ground-truth review package.

Builds a label-blind human review package from frozen F3J universe metadata.
Does not load gallery/OSNet embeddings, compute similarity/scoring/ranking,
fill ground-truth decisions, assign identity, or rerun detection/tracking/
segmentation.
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
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.ingest.checksum import sha256_file  # noqa: E402
from football_analytics.reid.crop_select import (  # noqa: E402
    clamp_bbox_xyxy,
    float_bbox_to_int_crop,
)
from football_analytics.reid.quality import compute_image_metrics  # noqa: E402

CONFIG_SCHEMA = "reid_independent_holdout_v2_ground_truth_review_config_v1"
STATUS = (
    "COMPLETED_STAGE5D_F3K_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_REVIEW_PACKAGE_READY"
)
READINESS = "TARGET_001_INDEPENDENT_HOLDOUT_V2_GROUND_TRUTH_READY_FOR_HUMAN_REVIEW"
NEXT_GATE = (
    "STAGE5D-F3L_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_MANUAL_REVIEW_AND_FREEZE"
)
TARGET_ID = "target_001"
TARGET_ALIAS = "sarı takım 5 numaralı oyuncu"
ELIGIBLE_N = 141
INELIGIBLE_N = 102
UNIVERSE_N = 243
SHEET_COUNT = 12
ITEMS_PER_FULL_SHEET = 12
LAST_SHEET_ITEMS = 9
VIDEO_PARTS = 3
ITEMS_PER_PART = 47
FFMPEG_BIN = "/usr/bin/ffmpeg"
ALLOWED_DIRTY = {
    "scripts/run_reid_independent_holdout_v2_ground_truth_review_package.py",
    "configs/reid/independent_holdout_v2_ground_truth_review_stage5d_target_001.yaml",
    "tests/test_reid_independent_holdout_v2_ground_truth_review_package.py",
    "docs/setup/stage5d-target-independent-holdout-ground-truth-review-package.md",
}
TEMPLATE_FIELDS = (
    "review_item_id",
    "segment_id",
    "raw_track_code",
    "tracker_native_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "observation_count",
    "primary_representative_frame",
    "representative_crop_path",
    "representative_crop_sha256",
    "manual_ground_truth_decision",
    "manual_target_present",
    "manual_same_target_as_target_001",
    "manual_same_team_as_target",
    "manual_visible_jersey_number",
    "jersey_number_provenance",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_single_person",
    "manual_identity_continuity_observed",
    "manual_track_impurity_observed",
    "manual_quality_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
GT_DECISION_VOCAB = (
    "target_occurrence_yes",
    "target_occurrence_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
SAME_TEAM_VOCAB = ("yes", "no", "uncertain")
FORBIDDEN_FIELDS = frozenset(
    {
        "target_001_predicted",
        "target_probability",
        "target_similarity",
        "distractor_similarity",
        "score_margin",
        "rank",
        "positive_label",
        "negative_label",
        "same_team_label",
        "visible_jersey_number",
        "ocr_prediction",
        "gallery_member_match",
        "human_identity_decision",
        "predicted_target",
        "identity_assignment",
        "jersey_number",
        "team_id",
        "team_label",
    }
)


class GroundTruthReviewError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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
        raise GroundTruthReviewError("unexpected config schema")
    if not config.get("offline_required"):
        raise GroundTruthReviewError("offline_required")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/") or Path(rel).is_absolute():
        raise GroundTruthReviewError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in ALLOWED_DIRTY:
                raise GroundTruthReviewError(
                    "BLOCKED_STAGE5D_F3K_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH snapshot_sha"
        )
    if not listing.read_text(encoding="utf-8").strip():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH listing"
        )
    return actual


def segment_numeric(segment_id: str) -> int:
    return int(str(segment_id).rsplit("_", 1)[-1])


def raw_track_numeric(raw_track_code: str) -> int:
    return int(str(raw_track_code).rsplit("_", 1)[-1])


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def edge_clipping_fraction(
    bbox: Sequence[float], *, width: int, height: int
) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = bbox_area(bbox)
    if area <= 0:
        return 1.0
    ix1, iy1 = max(x1, 0.0), max(y1, 0.0)
    ix2, iy2 = min(x2, float(width)), min(y2, float(height))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(max(0.0, min(1.0, 1.0 - (inter / area))))


def pad_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    fraction: float,
) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    px, py = fraction * bw, fraction * bh
    return clamp_bbox_xyxy(
        [x1 - px, y1 - py, x2 + px, y2 + py],
        video_width=width,
        video_height=height,
    )


def hard_quality_pass(
    *,
    crop_w: int,
    crop_h: int,
    bbox_area_px: float,
    edge_clip: float,
    max_iou: float,
    hq: Mapping[str, Any],
    decode_ok: bool,
) -> bool:
    if not decode_ok:
        return False
    if crop_w < int(hq["min_crop_width_px"]):
        return False
    if crop_h < int(hq["min_crop_height_px"]):
        return False
    if bbox_area_px < float(hq["min_bbox_area_px2"]):
        return False
    if edge_clip > float(hq["max_edge_clipping_fraction"]):
        return False
    if max_iou > float(hq["max_person_iou"]):
        return False
    return True


def forbidden_field_audit(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_FIELDS:
                hits.append(f"{path}.{key}" if path else key)
            hits.extend(forbidden_field_audit(value, f"{path}.{key}" if path else key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:50]):
            hits.extend(forbidden_field_audit(value, f"{path}[{i}]"))
    return hits


def build_access_audit(*, holdout_decode_frames: int) -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3k_access_audit_v1",
        "sample_video_read": False,
        "external_video_read": False,
        "holdout_video_decode_frames": holdout_decode_frames,
        "target_gallery_read_count": 0,
        "distractor_gallery_read_count": 0,
        "gallery_embedding_read_count": 0,
        "embedding_read_count": 0,
        "target_crop_read_count": 0,
        "distractor_crop_read_count": 0,
        "similarity_row_read_count": 0,
        "rank_row_read_count": 0,
        "previous_human_decision_read_count": 0,
        "gt_label_read_count": 0,
        "automated_jersey_metadata_read_count": 0,
        "jersey_metadata_read_count": 0,
        "identity_metadata_used": False,
        "osnet_model_loads": 0,
        "parseq_model_loads": 0,
        "ocr_calls": 0,
        "team_classifier_calls": 0,
        "identity_inference_calls": 0,
        "detection_inference_passes": 0,
        "detection_inference_frames": 0,
        "tracker_passes": 0,
        "tracking_rerun_count": 0,
        "segmentation_passes": 0,
        "segmentation_rerun_count": 0,
        "embeddings": 0,
        "cosine_similarity_computations": 0,
        "query_score_rows": 0,
        "rankings": 0,
        "gt_decisions": 0,
        "metrics": 0,
        "threshold_candidates": 0,
        "identity_assignments": 0,
        "gallery_mutations": 0,
    }


def validate_holdout_source(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    hold = config["holdout_source"]
    path = project_root / hold["path"]
    if not path.is_file() or path.is_symlink():
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH missing")
    if path.stat().st_size != int(hold["expected_bytes"]):
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH bytes")
    sha = sha256_file(path)
    if sha != hold["expected_sha256"]:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH sha")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH open")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (
        width != int(hold["expected_width"])
        or height != int(hold["expected_height"])
        or abs(fps - float(hold["expected_fps"])) > 1e-6
        or frames != int(hold["expected_frames"])
    ):
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH metadata")
    return {
        "path": hold["path"],
        "sha256": sha,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def validate_forbidden_sources(project_root: Path, config: Mapping[str, Any]) -> None:
    sample = project_root / config["evaluation_source"]["path"]
    external = project_root / config["external_enrollment_source"]["path"]
    if config["evaluation_source"].get("read_forbidden") and sample.is_file():
        # Metadata-only stat is allowed; do not decode sample in this gate.
        pass
    if config["external_enrollment_source"].get("read_forbidden") and external.is_file():
        pass
    yolo = project_root / config["yolo_checkpoint"]["path"]
    osnet = Path(config["osnet_checkpoint"]["path"])
    if config["yolo_checkpoint"].get("load_forbidden") and not yolo.is_file():
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH yolo")
    if config["osnet_checkpoint"].get("load_forbidden") and not osnet.is_file():
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH osnet")


def validate_f3j(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_f3j_package"]
    root = project_root / block["path"]
    if not root.is_dir():
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH missing")
    summary = load_json(root / "stage5d_f3j_summary.json")
    contract = load_json(root / "stage5d_f3j_contract.json")
    if summary.get("final_status") != block["expected_final_status"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH status"
        )
    if contract.get("readiness") != block["expected_readiness"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH readiness"
        )
    checks = {
        "segment_count": block["expected_complete_segments"],
        "complete_universe_count": block["expected_complete_segments"],
        "review_eligible_count": block["expected_review_eligible"],
        "review_ineligible_count": block["expected_review_ineligible"],
        "person_detection_count": block["expected_detections"],
        "assigned_detection_count": block["expected_assigned_detections"],
        "unassigned_detection_count": block["expected_unassigned_detections"],
        "raw_track_count": block["expected_raw_tracks"],
        "tracking_observation_count": block["expected_tracking_observations"],
        "pass_through_segment_count": block["expected_pass_through_segments"],
        "split_segment_count": block["expected_split_segments"],
        "representative_frame_candidate_count": block[
            "expected_representative_frame_candidates"
        ],
    }
    for key, expected in checks.items():
        if int(summary.get(key) or 0) != int(expected):
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH {key}"
            )
    if summary.get("holdout_sha256") != config["holdout_source"]["expected_sha256"]:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH holdout_sha"
        )
    if summary.get("replay_deterministic") is not True:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH replay"
        )
    if summary.get("label_blind_forbidden_fields") != 0:
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH forbidden"
        )
    snapshot_sha = resolve_snapshot_sha(
        Path(block["snapshot_path"]), block["expected_snapshot_sha256"]
    )
    f3i = config["stage5d_f3i_package"]
    f3i_root = project_root / f3i["path"]
    if f3i_root.is_dir():
        f3i_summary = load_json(f3i_root / "stage5d_f3i_summary.json")
        if f3i_summary.get("final_status") != f3i["expected_final_status"]:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH f3i_status"
            )
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "snapshot_sha256": snapshot_sha,
        "manifest_sha256": sha256_file(root / "stage5d_f3j_manifest.json"),
    }


def validate_f3h_metric_metadata(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_f3h_package"]
    metric_path = project_root / block["path"] / block["metric_contract_relpath"]
    gt_path = project_root / block["path"] / block["ground_truth_policy_relpath"]
    if not metric_path.is_file() or not gt_path.is_file():
        raise GroundTruthReviewError(
            "BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH f3h_missing"
        )
    metric = load_json(metric_path)
    gt_policy = load_json(gt_path)
    minimum_support = metric.get("minimum_support") or {}
    required = {
        "segment_clean_positive_ge": 5,
        "segment_clean_negative_ge": 20,
        "segment_clean_same_team_negative_ge": 10,
        "component_clean_positive_ge": 3,
        "component_clean_negative_ge": 10,
    }
    for key, expected in required.items():
        if int(minimum_support.get(key) or 0) != expected:
            raise GroundTruthReviewError(
                f"BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH f3h_{key}"
            )
    return {
        "metric_contract_sha256": sha256_file(metric_path),
        "ground_truth_policy_sha256": sha256_file(gt_path),
        "minimum_support": {k: int(minimum_support[k]) for k in required},
        "allowed_human_vocabulary": list(gt_policy.get("allowed_human_vocabulary") or []),
    }


def load_f3j_universe(f3j_root: Path) -> dict[str, Any]:
    return {
        "eligibility": load_jsonl(
            f3j_root / "quality" / "target_001_holdout_v2_segment_review_eligibility.jsonl"
        ),
        "segment_inventory": load_jsonl(
            f3j_root
            / "segmentation"
            / "target_001_holdout_v2_label_blind_segment_inventory.jsonl"
        ),
        "segment_observations": load_jsonl(
            f3j_root
            / "segmentation"
            / "target_001_holdout_v2_label_blind_segment_observations.jsonl"
        ),
        "rep_candidates": load_jsonl(
            f3j_root
            / "quality"
            / "target_001_holdout_v2_objective_representative_frame_candidates.jsonl"
        ),
        "raw_track_observations": load_jsonl(
            f3j_root / "tracking" / "target_001_holdout_v2_raw_track_observations.jsonl"
        ),
    }


def index_observations_by_frame(
    observations: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(dict(row))
    return by_frame


def index_segment_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_seg[str(row["segment_id"])].append(dict(row))
    for sid in by_seg:
        by_seg[sid].sort(key=lambda r: int(r["frame_index"]))
    return by_seg


def select_context_frames(
    obs_frames: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
) -> list[tuple[str, int]]:
    frames = sorted(set(int(f) for f in obs_frames))
    if not frames:
        raise GroundTruthReviewError("segment has no observations")
    midpoint = (int(start_frame) + int(end_frame)) / 2.0

    def nearest_mid() -> int:
        return min(frames, key=lambda f: (abs(f - midpoint), f))

    return [
        ("START", frames[0]),
        ("MIDDLE", nearest_mid()),
        ("END", frames[-1]),
    ]


def compute_candidate_metrics(
    *,
    frame_index: int,
    bbox: Sequence[float],
    candidate_rank: int,
    obs_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    raw_track_code: str,
    frame_cache: Mapping[int, np.ndarray],
    width: int,
    height: int,
    pad_frac: float,
    hq: Mapping[str, Any],
) -> dict[str, Any]:
    edge_clip = edge_clipping_fraction(bbox, width=width, height=height)
    area0 = bbox_area(bbox)
    max_iou = 0.0
    for other in obs_by_frame.get(frame_index, []):
        if str(other.get("raw_track_code")) == raw_track_code:
            continue
        max_iou = max(max_iou, iou_xyxy(bbox, other["bbox_xyxy"]))
    padded = pad_bbox(bbox, width=width, height=height, fraction=pad_frac)
    crop_int = float_bbox_to_int_crop(padded, video_width=width, video_height=height)
    x1, y1, x2, y2 = crop_int
    crop_w, crop_h = max(0, x2 - x1), max(0, y2 - y1)
    sharpness = 0.0
    decode_ok = False
    frame = frame_cache.get(frame_index)
    if frame is not None and crop_w > 0 and crop_h > 0:
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            decode_ok = True
            sharpness = float(compute_image_metrics(crop)["laplacian_variance"])
    hard_pass = hard_quality_pass(
        crop_w=crop_w,
        crop_h=crop_h,
        bbox_area_px=area0,
        edge_clip=edge_clip,
        max_iou=max_iou,
        hq=hq,
        decode_ok=decode_ok,
    )
    return {
        "candidate_rank": candidate_rank,
        "frame_index": frame_index,
        "source_bbox_xyxy": [float(v) for v in bbox],
        "canonical_crop_bbox_xyxy": [float(v) for v in padded],
        "crop_int_bbox_xyxy": [int(v) for v in crop_int],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "crop_area": float(crop_w * crop_h),
        "edge_clipping_fraction": edge_clip,
        "max_person_overlap": float(max_iou),
        "sharpness": sharpness,
        "hard_quality_pass": hard_pass,
        "decode_ok": decode_ok,
    }


def select_representative(
    *,
    candidate_frames: Sequence[int],
    segment_obs: Sequence[Mapping[str, Any]],
    obs_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    raw_track_code: str,
    frame_cache: Mapping[int, np.ndarray],
    width: int,
    height: int,
    pad_frac: float,
    hq: Mapping[str, Any],
) -> dict[str, Any]:
    bbox_by_frame = {int(o["frame_index"]): o for o in segment_obs}
    metrics: list[dict[str, Any]] = []
    for rank, fi in enumerate(candidate_frames):
        obs = bbox_by_frame.get(int(fi))
        if obs is None:
            continue
        metrics.append(
            compute_candidate_metrics(
                frame_index=int(fi),
                bbox=obs["bbox_xyxy"],
                candidate_rank=rank,
                obs_by_frame=obs_by_frame,
                raw_track_code=raw_track_code,
                frame_cache=frame_cache,
                width=width,
                height=height,
                pad_frac=pad_frac,
                hq=hq,
            )
        )
    if not metrics:
        raise GroundTruthReviewError("no representative candidates matched observations")
    metrics.sort(
        key=lambda m: (
            int(m["candidate_rank"]),
            0 if m["hard_quality_pass"] else 1,
            float(m["max_person_overlap"]),
            float(m["edge_clipping_fraction"]),
            -float(m["sharpness"]),
            -float(m["crop_area"]),
            int(m["frame_index"]),
        )
    )
    chosen = dict(metrics[0])
    obs = bbox_by_frame[int(chosen["frame_index"])]
    chosen["observation_id"] = str(obs["observation_id"])
    chosen["selection_reason"] = (
        "objective_f3j_candidate_rank_then_hard_quality_then_overlap_"
        "edge_clip_sharpness_area_frame"
    )
    return chosen


def compute_clip_window(
    *,
    start_frame: int,
    end_frame: int,
    representative_frame: int,
    total_frames: int,
    fps: float,
    pre_pad_sec: float,
    post_pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
) -> tuple[int, int]:
    pre = int(round(pre_pad_sec * fps))
    post = int(round(post_pad_sec * fps))
    min_frames = max(1, int(round(min_clip_sec * fps)))
    max_frames = max(min_frames, int(round(max_clip_sec * fps)))
    clip_start = max(0, int(start_frame) - pre)
    clip_end = min(total_frames - 1, int(end_frame) + post)
    span = clip_end - clip_start + 1
    if span > max_frames:
        half = max_frames // 2
        clip_start = max(0, int(representative_frame) - half)
        clip_end = min(total_frames - 1, clip_start + max_frames - 1)
        if clip_end - clip_start + 1 < max_frames:
            clip_start = max(0, clip_end - max_frames + 1)
    span = clip_end - clip_start + 1
    if span < min_frames:
        need = min_frames - span
        before = need // 2
        after = need - before
        clip_start = max(0, clip_start - before)
        clip_end = min(total_frames - 1, clip_end + after)
        span = clip_end - clip_start + 1
        if span < min_frames:
            clip_start = max(0, min(int(start_frame), total_frames - min_frames))
            clip_end = min(total_frames - 1, clip_start + min_frames - 1)
    return clip_start, clip_end


def build_review_plan(
    universe: Mapping[str, Any],
    *,
    frame_cache: Mapping[int, np.ndarray],
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    pad_frac: float,
    hq: Mapping[str, Any],
    video_sha: str,
    pre_pad_sec: float,
    post_pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inv_by_id = {str(r["segment_id"]): dict(r) for r in universe["segment_inventory"]}
    elig_by_id = {str(r["segment_id"]): dict(r) for r in universe["eligibility"]}
    rep_by_id = {
        str(r["segment_id"]): list(r["representative_frame_candidates"])
        for r in universe["rep_candidates"]
    }
    seg_obs = index_segment_observations(universe["segment_observations"])
    obs_by_frame = index_observations_by_frame(universe["raw_track_observations"])

    eligible_rows = [
        inv_by_id[sid]
        for sid, row in elig_by_id.items()
        if row.get("review_eligible") is True and sid in inv_by_id
    ]
    if len(eligible_rows) != ELIGIBLE_N:
        raise GroundTruthReviewError(
            f"BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH eligible={len(eligible_rows)}"
        )

    eligible_rows.sort(
        key=lambda r: (
            segment_numeric(str(r["segment_id"])),
            str(r["raw_track_code"]),
            int(r["start_frame"]),
        )
    )

    ineligible_rows = [
        {
            "segment_id": str(row["segment_id"]),
            "raw_track_code": str(inv_by_id[str(row["segment_id"])]["raw_track_code"]),
            "start_frame": int(inv_by_id[str(row["segment_id"])]["start_frame"]),
            "end_frame": int(inv_by_id[str(row["segment_id"])]["end_frame"]),
            "observation_count": int(inv_by_id[str(row["segment_id"])]["observation_count"]),
            "review_ineligibility_reasons": list(
                row.get("review_ineligibility_reasons") or []
            ),
            "complete_universe_member": True,
            "review_eligible": False,
            "automatic_negative": False,
            "metric_inclusion": False,
            "ground_truth_decision": "unreviewed_ineligible",
            "label_blind": True,
            "target_label_absent": True,
            "team_absent": True,
            "jersey_absent": True,
        }
        for row in universe["eligibility"]
        if row.get("review_eligible") is not True
    ]
    if len(ineligible_rows) != INELIGIBLE_N:
        raise GroundTruthReviewError(
            f"BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH ineligible={len(ineligible_rows)}"
        )
    ineligible_rows.sort(key=lambda r: segment_numeric(str(r["segment_id"])))

    items: list[dict[str, Any]] = []
    for idx, seg in enumerate(eligible_rows, start=1):
        sid = str(seg["segment_id"])
        raw_code = str(seg["raw_track_code"])
        obs = seg_obs.get(sid) or []
        if not obs:
            raise GroundTruthReviewError(f"missing segment observations {sid}")
        candidates = rep_by_id.get(sid) or []
        if not candidates:
            raise GroundTruthReviewError(f"missing representative candidates {sid}")
        rep = select_representative(
            candidate_frames=candidates,
            segment_obs=obs,
            obs_by_frame=obs_by_frame,
            raw_track_code=raw_code,
            frame_cache=frame_cache,
            width=width,
            height=height,
            pad_frac=pad_frac,
            hq=hq,
        )
        context = select_context_frames(
            [int(o["frame_index"]) for o in obs],
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
        )
        clip_start, clip_end = compute_clip_window(
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
            representative_frame=int(rep["frame_index"]),
            total_frames=total_frames,
            fps=fps,
            pre_pad_sec=pre_pad_sec,
            post_pad_sec=post_pad_sec,
            min_clip_sec=min_clip_sec,
            max_clip_sec=max_clip_sec,
        )
        review_id = f"H2_GT_REVIEW_{idx:06d}"
        bbox_by_frame = {int(o["frame_index"]): o for o in obs}
        tracker_native_id = int(str(raw_code).split("_")[-1])
        elig_row = elig_by_id[sid]
        items.append(
            {
                "review_item_id": review_id,
                "review_rank": idx,
                "target_id": TARGET_ID,
                "segment_id": sid,
                "raw_track_code": raw_code,
                "tracker_native_id": tracker_native_id,
                "start_frame": int(seg["start_frame"]),
                "end_frame": int(seg["end_frame"]),
                "start_time_sec": float(seg["start_time_sec"]),
                "end_time_sec": float(seg["end_time_sec"]),
                "observation_count": int(seg["observation_count"]),
                "span_frames": int(seg["span_frames"]),
                "maximum_gap": int(seg.get("maximum_gap") or 0),
                "objective_eligibility_reasons": list(
                    elig_row.get("review_ineligibility_reasons") or []
                ),
                "representative_frame_candidates": list(candidates),
                "representative": rep,
                "context_frames": context,
                "clip_start_frame": clip_start,
                "clip_end_frame": clip_end,
                "bbox_by_frame": bbox_by_frame,
                "label_blind": True,
                "gt_decision_pending": True,
                "target_label_absent": True,
                "team_absent": True,
                "jersey_absent": True,
                "holdout_sha256": video_sha,
            }
        )

    if len(items) != ELIGIBLE_N:
        raise GroundTruthReviewError("eligible item count mismatch")

    sheet_assignments = []
    for sheet_i in range(1, SHEET_COUNT + 1):
        start = (sheet_i - 1) * ITEMS_PER_FULL_SHEET
        end = start + (
            LAST_SHEET_ITEMS if sheet_i == SHEET_COUNT else ITEMS_PER_FULL_SHEET
        )
        chunk = items[start:end]
        sheet_assignments.append(
            {
                "sheet_index": sheet_i,
                "item_range": [chunk[0]["review_item_id"], chunk[-1]["review_item_id"]],
                "review_item_ids": [r["review_item_id"] for r in chunk],
            }
        )

    video_assignments = []
    for part_i in range(1, VIDEO_PARTS + 1):
        start = (part_i - 1) * ITEMS_PER_PART
        chunk = items[start : start + ITEMS_PER_PART]
        video_assignments.append(
            {
                "part_index": part_i,
                "review_item_ids": [r["review_item_id"] for r in chunk],
            }
        )

    fingerprint_rows = []
    for item in items:
        rep = item["representative"]
        fingerprint_rows.append(
            {
                "review_item_id": item["review_item_id"],
                "segment_id": item["segment_id"],
                "raw_track_code": item["raw_track_code"],
                "representative_frame": int(rep["frame_index"]),
                "source_bbox_xyxy": rep["source_bbox_xyxy"],
                "canonical_crop_bbox_xyxy": rep["canonical_crop_bbox_xyxy"],
                "crop_int_bbox_xyxy": rep["crop_int_bbox_xyxy"],
                "context_frames": item["context_frames"],
                "clip_start_frame": item["clip_start_frame"],
                "clip_end_frame": item["clip_end_frame"],
                "sheet_index": next(
                    s["sheet_index"]
                    for s in sheet_assignments
                    if item["review_item_id"] in s["review_item_ids"]
                ),
                "video_part_index": next(
                    v["part_index"]
                    for v in video_assignments
                    if item["review_item_id"] in v["review_item_ids"]
                ),
            }
        )

    meta = {
        "eligible_count": ELIGIBLE_N,
        "ineligible_count": INELIGIBLE_N,
        "sheet_assignments": sheet_assignments,
        "video_assignments": video_assignments,
        "selection_fingerprint_sha256": canonical_json_sha(fingerprint_rows),
        "fingerprint_rows": fingerprint_rows,
    }
    return items, ineligible_rows, meta


def _fit_bgr(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate_context_frame(
    frame: np.ndarray,
    bbox: Sequence[float],
    *,
    role: str,
    frame_index: int,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
    for i, text in enumerate((role, f"frame={frame_index}", f"t={frame_index / fps:.3f}s")):
        y = 28 + i * 26
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA
        )
    return out


def draw_watermark(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    cv2.putText(
        frame, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA
    )


def render_item_tile(
    *,
    item: Mapping[str, Any],
    crop_bgr: np.ndarray,
    contexts: Sequence[tuple[str, np.ndarray]],
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
    rep = item["representative"]
    header_lines = [
        str(item["review_item_id"]),
        str(item["segment_id"]),
        str(item["raw_track_code"]),
        f"frames {item['start_frame']}-{item['end_frame']}",
        f"t {item['start_time_sec']:.2f}-{item['end_time_sec']:.2f}s",
        f"obs={item['observation_count']}",
        f"rep={rep['frame_index']} ctx="
        + ",".join(f"{role}:{fi}" for role, fi in item["context_frames"]),
    ]
    y = 28
    for text in header_lines:
        cv2.putText(
            tile, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA
        )
        y += 18

    crop_disp = _fit_bgr(crop_bgr, tile_w - 40, int(tile_h * 0.42))
    ch, cw = crop_disp.shape[:2]
    ox = (tile_w - cw) // 2
    oy = 150
    tile[oy : oy + ch, ox : ox + cw] = crop_disp

    ctx_y = oy + ch + 16
    ctx_slot_w = (tile_w - 48) // 3
    ctx_slot_h = tile_h - ctx_y - 16
    for i, (role, img) in enumerate(contexts[:3]):
        disp = _fit_bgr(img, ctx_slot_w - 8, ctx_slot_h - 8)
        dh, dw = disp.shape[:2]
        cx = 16 + i * ctx_slot_w + (ctx_slot_w - dw) // 2
        cy = ctx_y + (ctx_slot_h - dh) // 2
        tile[cy : cy + dh, cx : cx + dw] = disp
        cv2.putText(
            tile,
            role,
            (16 + i * ctx_slot_w + 4, ctx_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 120),
            1,
            cv2.LINE_AA,
        )
    return tile


def render_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    sheet_index: int,
    min_width: int,
    cols: int,
    fps: float,
) -> np.ndarray:
    n = len(items)
    rows_n = int(math.ceil(n / cols)) if n else 1
    tile_w = max(900, int(math.ceil(min_width / cols)))
    tile_h = 780
    header_h = 96
    width = max(min_width, cols * tile_w)
    height = header_h + rows_n * tile_h
    sheet = np.full((height, width, 3), 14, dtype=np.uint8)
    start_id = items[0]["review_item_id"]
    end_id = items[-1]["review_item_id"]
    header_lines = [
        "target_001 independent holdout v2 — LABEL-BLIND HUMAN GROUND-TRUTH REVIEW",
        f"sheet {sheet_index:02d} items {start_id}..{end_id}",
        "NO TARGET/DISTRACTOR SCORE DISPLAYED",
        "HUMAN GROUND-TRUTH REVIEW / NO SIMILARITY — NO AUTOMATIC IDENTITY",
    ]
    y = 24
    for text in header_lines:
        cv2.putText(
            sheet, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 20
    for i, item in enumerate(items):
        r, c = divmod(i, cols)
        tile = render_item_tile(
            item=item,
            crop_bgr=item["crop_bgr"],
            contexts=item["context_panels"],
            tile_w=tile_w,
            tile_h=tile_h,
        )
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


class FfmpegRawWriter:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        if not Path(FFMPEG_BIN).is_file():
            raise GroundTruthReviewError("ffmpeg missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frame_count = 0
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-x264-params",
            "bframes=0:keyint=30:scenecut=0",
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.proc.stdin is None:
            raise GroundTruthReviewError("ffmpeg stdin unavailable")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise GroundTruthReviewError("ffmpeg frame size mismatch")
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
            raise GroundTruthReviewError(f"ffmpeg failed rc={rc}: {err[:500]}")


def draw_title_card(
    *,
    width: int,
    height: int,
    review_item_id: str,
    segment_id: str,
    raw_track_code: str,
) -> np.ndarray:
    card = np.full((height, width, 3), 24, dtype=np.uint8)
    lines = [
        review_item_id,
        segment_id,
        raw_track_code,
        "HUMAN GT DECISION PENDING",
    ]
    y = 80
    for text in lines:
        cv2.putText(
            card, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA
        )
        y += 42
    return card


def annotate_review_frame(
    frame: np.ndarray,
    *,
    item: Mapping[str, Any],
    frame_index: int,
    bbox: Optional[Sequence[float]],
    watermark: str,
) -> np.ndarray:
    out = frame.copy()
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
    lines = [
        str(item["review_item_id"]),
        str(item["segment_id"]),
        str(item["raw_track_code"]),
        f"frame {frame_index} / {item['start_frame']}-{item['end_frame']}",
        "HUMAN GT DECISION PENDING",
    ]
    y = 28
    for text in lines:
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 24
    draw_watermark(out, watermark)
    return out


def write_review_videos(
    *,
    items: Sequence[Mapping[str, Any]],
    video_path: Path,
    out_dir: Path,
    fps: float,
    title_card_sec: float,
    watermark: str,
    crf: int,
    preset: str,
    frame_cache: Mapping[int, np.ndarray],
    width: int,
    height: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    title_frames = max(1, int(round(title_card_sec * fps)))
    rel_paths: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    for part_i in range(1, VIDEO_PARTS + 1):
        chunk = items[(part_i - 1) * ITEMS_PER_PART : part_i * ITEMS_PER_PART]
        if len(chunk) != ITEMS_PER_PART:
            raise GroundTruthReviewError(f"video part {part_i} item count")
        name = f"target_001_holdout_v2_gt_review_part_{part_i:02d}.mp4"
        out_path = out_dir / name
        writer = FfmpegRawWriter(
            out_path, width=width, height=height, fps=fps, crf=crf, preset=preset
        )
        try:
            for item in chunk:
                card = draw_title_card(
                    width=width,
                    height=height,
                    review_item_id=str(item["review_item_id"]),
                    segment_id=str(item["segment_id"]),
                    raw_track_code=str(item["raw_track_code"]),
                )
                for _ in range(title_frames):
                    writer.write(card)
                bbox_by_frame = {
                    int(k): v["bbox_xyxy"] for k, v in item["bbox_by_frame"].items()
                }
                rendered = 0
                bbox_frames = 0
                for fi in range(int(item["clip_start_frame"]), int(item["clip_end_frame"]) + 1):
                    frame = frame_cache.get(fi)
                    if frame is None:
                        raise GroundTruthReviewError(f"missing frame {fi} for video")
                    bbox = bbox_by_frame.get(fi)
                    if bbox is not None:
                        bbox_frames += 1
                    annotated = annotate_review_frame(
                        frame,
                        item=item,
                        frame_index=fi,
                        bbox=bbox,
                        watermark=watermark,
                    )
                    writer.write(annotated)
                    rendered += 1
                manifest_rows.append(
                    {
                        "review_item_id": item["review_item_id"],
                        "segment_id": item["segment_id"],
                        "raw_track_code": item["raw_track_code"],
                        "video_part_index": part_i,
                        "video_relpath": f"videos/{name}",
                        "clip_start_frame": int(item["clip_start_frame"]),
                        "clip_end_frame": int(item["clip_end_frame"]),
                        "rendered_frame_count": rendered + title_frames,
                        "clip_rendered_frame_count": rendered,
                        "title_card_frame_count": title_frames,
                        "bbox_drawn_frame_count": bbox_frames,
                        "bbox_interpolation": False,
                    }
                )
        finally:
            writer.close()
        rel_paths.append(f"videos/{name}")
    if len(manifest_rows) != ELIGIBLE_N:
        raise GroundTruthReviewError("video manifest coverage mismatch")
    return rel_paths, manifest_rows


def decode_frames(video_path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    needed = sorted(set(int(f) for f in frame_indices))
    cache: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise GroundTruthReviewError("holdout open failed")
    try:
        for fi in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise GroundTruthReviewError(f"failed decode frame {fi}")
            cache[fi] = frame
    finally:
        cap.release()
    return cache


def create_temp_root(final_dir: Path) -> Path:
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp = parent / f".tmp_f3k_{token}"
    if tmp.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F3K_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=False, exist_ok=False)
    return tmp


def atomic_publish(tmp: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F3K_ATOMIC_OUTPUT final_exists")
    os.rename(tmp, final_dir)


def run(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    project_root = (project_root or _PROJECT_ROOT).resolve()
    config = load_config(config_path)
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise GroundTruthReviewError("FAILED_STAGE5D_F3K_ATOMIC_OUTPUT final_exists")

    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    holdout = validate_holdout_source(project_root, config)
    validate_forbidden_sources(project_root, config)
    f3j = validate_f3j(project_root, config)
    f3h_meta = validate_f3h_metric_metadata(project_root, config)
    universe = load_f3j_universe(f3j["root"])

    if len(universe["segment_inventory"]) != UNIVERSE_N:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_UNIVERSE_CONTRACT_MISMATCH universe")

    pad_frac = float(config["crop_extraction"]["padding_fraction"])
    hq = config["hard_quality"]
    cs = config["contact_sheets"]
    rv = config["review_videos"]
    fps = float(holdout["fps"])
    width = int(holdout["width"])
    height = int(holdout["height"])
    total_frames = int(holdout["frames"])
    holdout_path = project_root / holdout["path"]
    sha_before = holdout["sha256"]

    # Collect all frame indices needed before decode.
    seg_obs = index_segment_observations(universe["segment_observations"])
    rep_by_id = {
        str(r["segment_id"]): list(r["representative_frame_candidates"])
        for r in universe["rep_candidates"]
    }
    elig_ids = {
        str(r["segment_id"])
        for r in universe["eligibility"]
        if r.get("review_eligible") is True
    }
    inv_by_id = {str(r["segment_id"]): r for r in universe["segment_inventory"]}
    needed_frames: set[int] = set()
    for sid in sorted(elig_ids, key=segment_numeric):
        seg = inv_by_id[sid]
        obs = seg_obs[sid]
        obs_frames = [int(o["frame_index"]) for o in obs]
        needed_frames.update(obs_frames)
        needed_frames.update(rep_by_id.get(sid) or [])
        for role_frames in select_context_frames(
            obs_frames,
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
        ):
            needed_frames.add(role_frames[1])
        rep_guess = (int(seg["start_frame"]) + int(seg["end_frame"])) // 2
        clip_start, clip_end = compute_clip_window(
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
            representative_frame=rep_guess,
            total_frames=total_frames,
            fps=fps,
            pre_pad_sec=float(rv["pre_pad_sec"]),
            post_pad_sec=float(rv["post_pad_sec"]),
            min_clip_sec=float(rv["min_clip_sec"]),
            max_clip_sec=float(rv["max_clip_sec"]),
        )
        needed_frames.update(range(clip_start, clip_end + 1))

    frame_cache = decode_frames(holdout_path, sorted(needed_frames))

    items_pass1, ineligible_rows, plan_meta = build_review_plan(
        universe,
        frame_cache=frame_cache,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        pad_frac=pad_frac,
        hq=hq,
        video_sha=sha_before,
        pre_pad_sec=float(rv["pre_pad_sec"]),
        post_pad_sec=float(rv["post_pad_sec"]),
        min_clip_sec=float(rv["min_clip_sec"]),
        max_clip_sec=float(rv["max_clip_sec"]),
    )
    fp_pass1 = plan_meta["selection_fingerprint_sha256"]

    items_pass2, _, plan_meta2 = build_review_plan(
        universe,
        frame_cache=frame_cache,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        pad_frac=pad_frac,
        hq=hq,
        video_sha=sha_before,
        pre_pad_sec=float(rv["pre_pad_sec"]),
        post_pad_sec=float(rv["post_pad_sec"]),
        min_clip_sec=float(rv["min_clip_sec"]),
        max_clip_sec=float(rv["max_clip_sec"]),
    )
    if plan_meta2["selection_fingerprint_sha256"] != fp_pass1:
        raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_REVIEW_PACKAGE_NONDETERMINISM")

    items = items_pass1
    tmp = create_temp_root(final_dir)
    try:
        review_dir = tmp / "review"
        pkg = tmp / "review_packages" / "target_001_holdout_v2_ground_truth_review"
        crops_dir = tmp / "crops"
        videos_dir = tmp / "videos"
        templates_dir = tmp / "templates"
        exclusions_dir = tmp / "exclusions"
        inventory_dir = tmp / "inventory"
        runtime_dir = tmp / "runtime"
        eff_dir = tmp / "effective_configs"
        for d in (
            review_dir,
            pkg,
            crops_dir,
            videos_dir,
            templates_dir,
            exclusions_dir,
            inventory_dir,
            runtime_dir,
            eff_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, eff_dir / Path(config_path).name)

        crop_contract = {
            "schema_version": "reid_stage5d_f3k_crop_contract_v1",
            "padding_fraction": pad_frac,
            "image_format": config["crop_extraction"]["image_format"],
            "hard_quality": dict(hq),
            "label_blind": True,
            "similarity_blind": True,
        }
        video_contract = {
            "schema_version": "reid_stage5d_f3k_video_contract_v1",
            "ffmpeg_bin": FFMPEG_BIN,
            "codec": rv["codec"],
            "pix_fmt": rv["pix_fmt"],
            "fps": fps,
            "crf": int(rv["crf"]),
            "preset": rv["preset"],
            "title_card_sec": float(rv["title_card_sec"]),
            "pre_pad_sec": float(rv["pre_pad_sec"]),
            "post_pad_sec": float(rv["post_pad_sec"]),
            "min_clip_sec": float(rv["min_clip_sec"]),
            "max_clip_sec": float(rv["max_clip_sec"]),
            "x264_params": "bframes=0:keyint=30:scenecut=0",
            "watermark": rv["watermark"],
            "bbox_interpolation": False,
        }
        write_json(runtime_dir / "crop_contract.json", crop_contract)
        write_json(runtime_dir / "video_contract.json", video_contract)
        write_json(
            runtime_dir / "source_sha_checkpoints.json",
            {"before_render": sha_before},
        )

        crop_manifest_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []

        for item in items:
            rep = item["representative"]
            fi = int(rep["frame_index"])
            x1, y1, x2, y2 = rep["crop_int_bbox_xyxy"]
            crop_img = frame_cache[fi][y1:y2, x1:x2].copy()
            if crop_img.size == 0:
                raise GroundTruthReviewError(f"empty crop {item['review_item_id']}")
            ok, buf = cv2.imencode(".png", crop_img)
            if not ok:
                raise GroundTruthReviewError("png encode failed")
            crop_bytes = buf.tobytes()
            crop_sha = sha256_bytes(crop_bytes)
            crop_name = (
                f"{item['review_item_id']}__{item['segment_id']}__rep.png"
            )
            crop_rel = f"crops/{crop_name}"
            crop_path = crops_dir / crop_name
            crop_path.write_bytes(crop_bytes)
            item["crop_bgr"] = crop_img
            item["representative_crop_path"] = crop_rel
            item["representative_crop_sha256"] = crop_sha

            bbox_by_frame = {
                int(k): v["bbox_xyxy"] for k, v in item["bbox_by_frame"].items()
            }
            panels: list[tuple[str, np.ndarray]] = []
            for role, cfi in item["context_frames"]:
                bbox = bbox_by_frame.get(int(cfi))
                if bbox is None:
                    raise GroundTruthReviewError(
                        f"context frame {cfi} missing bbox for {item['segment_id']}"
                    )
                panels.append(
                    (
                        role,
                        annotate_context_frame(
                            frame_cache[int(cfi)],
                            bbox,
                            role=role,
                            frame_index=int(cfi),
                            fps=fps,
                        ),
                    )
                )
            item["context_panels"] = panels

            crop_manifest_rows.append(
                {
                    "review_item_id": item["review_item_id"],
                    "segment_id": item["segment_id"],
                    "raw_track_code": item["raw_track_code"],
                    "source_frame_index": fi,
                    "observation_id": rep["observation_id"],
                    "source_bbox_xyxy": rep["source_bbox_xyxy"],
                    "canonical_crop_bbox_xyxy": rep["canonical_crop_bbox_xyxy"],
                    "crop_int_bbox_xyxy": rep["crop_int_bbox_xyxy"],
                    "crop_width": rep["crop_width"],
                    "crop_height": rep["crop_height"],
                    "crop_path": crop_rel,
                    "crop_sha256": crop_sha,
                    "holdout_sha256": sha_before,
                    "selection_reason": rep["selection_reason"],
                    "hard_quality_pass": rep["hard_quality_pass"],
                    "target_label_absent": True,
                    "team_absent": True,
                    "jersey_absent": True,
                    "label_blind": True,
                }
            )
            mapping_rows.append(
                {
                    "review_item_id": item["review_item_id"],
                    "target_id": TARGET_ID,
                    "segment_id": item["segment_id"],
                    "raw_track_code": item["raw_track_code"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                    "observation_count": item["observation_count"],
                    "representative_frame": fi,
                    "representative_crop_path": crop_rel,
                    "representative_crop_sha256": crop_sha,
                    "context_frames": [
                        {"role": role, "frame_index": int(cfi)}
                        for role, cfi in item["context_frames"]
                    ],
                    "clip_start_frame": item["clip_start_frame"],
                    "clip_end_frame": item["clip_end_frame"],
                    "review_eligible": True,
                    "all_manual_fields_blank": True,
                    "similarity_computed": False,
                    "rank_computed": False,
                    "label_blind": True,
                }
            )

        write_jsonl(
            crops_dir / "target_001_holdout_v2_gt_review_crop_manifest.jsonl",
            crop_manifest_rows,
        )
        inventory_rows = []
        for item in items:
            inventory_rows.append(
                {
                    "review_item_id": item["review_item_id"],
                    "segment_id": item["segment_id"],
                    "raw_track_code": item["raw_track_code"],
                    "tracker_native_id": item["tracker_native_id"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                    "start_time": item["start_time_sec"],
                    "end_time": item["end_time_sec"],
                    "observation_count": item["observation_count"],
                    "span_frames": item["span_frames"],
                    "maximum_gap": item["maximum_gap"],
                    "objective_eligibility_reasons": item["objective_eligibility_reasons"],
                    "representative_frame_candidate_ids": item[
                        "representative_frame_candidates"
                    ],
                    "selected_representative_frame": item["representative"]["frame_index"],
                    "context_start_frame": next(
                        f for role, f in item["context_frames"] if role == "START"
                    ),
                    "context_middle_frame": next(
                        (f for role, f in item["context_frames"] if role == "MIDDLE"),
                        None,
                    ),
                    "context_end_frame": next(
                        f for role, f in item["context_frames"] if role == "END"
                    ),
                    "label_blind": True,
                    "gt_decision_pending": True,
                }
            )
        write_jsonl(
            inventory_dir / "target_001_holdout_v2_gt_review_item_inventory.jsonl",
            inventory_rows,
        )
        write_json(
            inventory_dir / "target_001_holdout_v2_gt_review_item_mapping.json",
            {
                "schema_version": "reid_stage5d_f3k_gt_review_item_mapping_v1",
                "review_item_count": ELIGIBLE_N,
                "items": mapping_rows,
            },
        )
        write_jsonl(
            exclusions_dir / "target_001_holdout_v2_review_ineligible_segment_inventory.jsonl",
            ineligible_rows,
        )
        write_json(
            exclusions_dir / "target_001_holdout_v2_review_ineligible_summary.json",
            {
                "schema_version": "reid_stage5d_f3k_ineligible_summary_v1",
                "ineligible_count": INELIGIBLE_N,
                "automatic_negative": False,
                "non_player_unless_human_reviewed": False,
                "target_absent_false": True,
                "metric_inclusion": False,
                "ground_truth_decision": "unreviewed_ineligible",
                "not_counted_as_system_failure_negative": True,
                "reason_distribution": dict(
                    Counter(
                        reason
                        for row in ineligible_rows
                        for reason in row["review_ineligibility_reasons"]
                    )
                ),
            },
        )

        sheet_paths: list[str] = []
        sheet_item_counts: list[int] = []
        sheet_manifest_rows: list[dict[str, Any]] = []
        min_width = int(cs["min_width_px"])
        cols = int(cs["grid_cols"])
        for sheet_i in range(1, SHEET_COUNT + 1):
            start = (sheet_i - 1) * ITEMS_PER_FULL_SHEET
            end = start + (
                LAST_SHEET_ITEMS if sheet_i == SHEET_COUNT else ITEMS_PER_FULL_SHEET
            )
            chunk = items[start:end]
            expected = (
                LAST_SHEET_ITEMS if sheet_i == SHEET_COUNT else ITEMS_PER_FULL_SHEET
            )
            if len(chunk) != expected:
                raise GroundTruthReviewError(f"sheet {sheet_i} size")
            sheet = render_contact_sheet(
                chunk, sheet_index=sheet_i, min_width=min_width, cols=cols, fps=fps
            )
            if sheet.shape[1] < min_width:
                raise GroundTruthReviewError("sheet width below minimum")
            name = f"target_001_holdout_v2_gt_review_sheet_{sheet_i:02d}.png"
            out = pkg / name
            if not cv2.imwrite(str(out), sheet):
                raise GroundTruthReviewError(f"sheet write failed {name}")
            rel = (
                "review_packages/target_001_holdout_v2_ground_truth_review/" + name
            )
            sheet_paths.append(rel)
            sheet_item_counts.append(len(chunk))
            sheet_manifest_rows.append(
                {
                    "sheet_index": sheet_i,
                    "path": rel,
                    "item_count": len(chunk),
                    "review_item_ids": [r["review_item_id"] for r in chunk],
                    "item_range": [chunk[0]["review_item_id"], chunk[-1]["review_item_id"]],
                }
            )

        video_paths, video_manifest_rows = write_review_videos(
            items=items,
            video_path=holdout_path,
            out_dir=videos_dir,
            fps=fps,
            title_card_sec=float(rv["title_card_sec"]),
            watermark=str(rv["watermark"]),
            crf=int(rv["crf"]),
            preset=str(rv["preset"]),
            frame_cache=frame_cache,
            width=width,
            height=height,
        )
        write_json(
            videos_dir / "target_001_holdout_v2_gt_review_video_manifest.json",
            {
                "schema_version": "reid_stage5d_f3k_video_manifest_v1",
                "item_count": ELIGIBLE_N,
                "part_count": VIDEO_PARTS,
                "items": video_manifest_rows,
            },
        )
        write_json(
            pkg / "target_001_holdout_v2_gt_contact_sheet_manifest.json",
            {
                "schema_version": "reid_stage5d_f3k_contact_sheet_manifest_v1",
                "sheet_count": SHEET_COUNT,
                "sheets": sheet_manifest_rows,
            },
        )

        with (templates_dir / "target_001_holdout_v2_ground_truth_review_template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "review_item_id": item["review_item_id"],
                        "segment_id": item["segment_id"],
                        "raw_track_code": item["raw_track_code"],
                        "tracker_native_id": item["tracker_native_id"],
                        "start_frame": item["start_frame"],
                        "end_frame": item["end_frame"],
                        "start_time": item["start_time_sec"],
                        "end_time": item["end_time_sec"],
                        "observation_count": item["observation_count"],
                        "primary_representative_frame": item["representative"][
                            "frame_index"
                        ],
                        "representative_crop_path": item["representative_crop_path"],
                        "representative_crop_sha256": item["representative_crop_sha256"],
                        "manual_ground_truth_decision": "",
                        "manual_target_present": "",
                        "manual_same_target_as_target_001": "",
                        "manual_same_team_as_target": "",
                        "manual_visible_jersey_number": "",
                        "jersey_number_provenance": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_single_person": "",
                        "manual_identity_continuity_observed": "",
                        "manual_track_impurity_observed": "",
                        "manual_quality_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )

        review_contract = {
            "schema_version": "reid_target_001_holdout_v2_ground_truth_review_contract_v1",
            "target_id": TARGET_ID,
            "target_alias": TARGET_ALIAS,
            "complete_universe": UNIVERSE_N,
            "review_eligible": ELIGIBLE_N,
            "review_ineligible": INELIGIBLE_N,
            "review_items": ELIGIBLE_N,
            "similarity_blind": True,
            "label_blind": True,
            "manual_ground_truth_decisions": 0,
            "similarity_rows": 0,
            "ranking_rows": 0,
            "gallery_hidden": True,
            "similarity_rank_hidden": True,
            "automated_ocr_forbidden": True,
            "manual_decision_required_before_scoring": True,
            "unreviewed_not_negative": True,
            "ineligible_not_negative": True,
            "target_present_contaminated_not_positive_or_negative": True,
            "gallery_reads": False,
            "embedding_reads": False,
            "detection_rerun": False,
            "tracking_rerun": False,
            "segmentation_rerun": False,
            "holdout_sha256": sha_before,
            "f3j_snapshot_sha256": f3j["snapshot_sha256"],
            "f3h_metric_contract_sha256": f3h_meta["metric_contract_sha256"],
            "minimum_support_metadata_only": f3h_meta["minimum_support"],
            "allowed_manual_ground_truth_decisions": list(GT_DECISION_VOCAB),
            "allowed_manual_same_team_as_target": list(SAME_TEAM_VOCAB),
            "human_jersey_provenance_policy": (
                "fill_manual_visible_jersey_number_only_when_human_clearly_sees;"
                "no_guess;automated_ocr_forbidden"
            ),
            "clean_positive_policy": "target_occurrence_yes_only_when_verifiable_dominant_continuity",
            "clean_negative_policy": "target_occurrence_no_for_player_not_target_001",
            "same_team_negative_policy": "manual_same_team_as_target_yes_with_target_occurrence_no",
            "uncertain_exclusion": True,
            "invalid_exclusion": True,
            "ambiguous_exclusion": True,
            "reviewer_expected_next_gate": "Furkan",
            "final_approver_expected_next_gate": "Furkan",
            "exact_next_gate": NEXT_GATE,
        }
        write_json(
            review_dir / "target_001_holdout_v2_ground_truth_review_contract.json",
            review_contract,
        )
        write_json(
            review_dir / "target_001_holdout_v2_ground_truth_review_manifest.json",
            {
                "schema_version": "reid_target_001_holdout_v2_ground_truth_review_manifest_v1",
                "eligible_count": ELIGIBLE_N,
                "ineligible_count": INELIGIBLE_N,
                "contact_sheets": sheet_paths,
                "sheet_item_counts": sheet_item_counts,
                "videos": video_paths,
                "template": "templates/target_001_holdout_v2_ground_truth_review_template.csv",
                "inventory": "inventory/target_001_holdout_v2_gt_review_item_inventory.jsonl",
                "mapping": "inventory/target_001_holdout_v2_gt_review_item_mapping.json",
                "manual_decisions": 0,
                "similarity_rows": 0,
                "ranking_rows": 0,
            },
        )

        sha_after = sha256_file(holdout_path)
        if sha_after != sha_before:
            raise GroundTruthReviewError("BLOCKED_STAGE5D_F3K_HOLDOUT_SOURCE_MISMATCH post")

        access = build_access_audit(holdout_decode_frames=len(needed_frames))
        write_json(runtime_dir / "access_audit.json", access)
        write_json(
            runtime_dir / "runtime.json",
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "project_head": head,
                "offline_required": True,
                "network_used": False,
                "holdout_decode_frames": len(needed_frames),
                "selection_fingerprint_sha256": fp_pass1,
                "determinism_pass_2_match": True,
                **access,
            },
        )
        runtime_dir.joinpath("source_sha_checkpoints.json").write_text(
            json.dumps(
                {"before_render": sha_before, "after_render": sha_after, "unchanged": True},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        forbidden_hits: list[str] = []
        for label, obj in (
            ("mapping", mapping_rows[:5]),
            ("crop_manifest", crop_manifest_rows[:5]),
            ("review_contract", review_contract),
            ("ineligible", ineligible_rows[:5]),
        ):
            forbidden_hits.extend(f"{label}:{h}" for h in forbidden_field_audit(obj))
        if forbidden_hits:
            raise GroundTruthReviewError(
                "BLOCKED_STAGE5D_F3K_GROUND_TRUTH_REVIEW_LEAKAGE "
                + json.dumps(forbidden_hits)
            )

        crop_pngs = list(crops_dir.glob("*.png"))
        sheet_pngs = list(pkg.glob("target_001_holdout_v2_gt_review_sheet_*.png"))
        mp4s = list(videos_dir.glob("*.mp4"))
        other_png = [
            p
            for p in tmp.rglob("*.png")
            if p not in crop_pngs and p not in sheet_pngs
        ]
        if len(crop_pngs) != ELIGIBLE_N:
            raise GroundTruthReviewError(f"crop png budget {len(crop_pngs)}")
        if len(sheet_pngs) != SHEET_COUNT:
            raise GroundTruthReviewError(f"sheet png budget {len(sheet_pngs)}")
        if len(mp4s) != VIDEO_PARTS:
            raise GroundTruthReviewError(f"mp4 budget {len(mp4s)}")
        if other_png or list(tmp.rglob("*.npy")) or list(tmp.rglob("*.jpg")):
            raise GroundTruthReviewError("artifact budget extra png/npy/jpeg")

        contract = {
            "schema_version": "reid_stage5d_f3k_contract_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "project_head": head,
            "eligible_review_items": ELIGIBLE_N,
            "ineligible_segments": INELIGIBLE_N,
            "universe_segments": UNIVERSE_N,
            "contact_sheets": SHEET_COUNT,
            "review_videos": VIDEO_PARTS,
            "manual_ground_truth_decisions": 0,
            "similarity_rows": 0,
            "ranking_rows": 0,
            "holdout_sha256": sha_before,
            "f3j_snapshot_sha256": f3j["snapshot_sha256"],
            "selection_fingerprint_sha256": fp_pass1,
        }
        write_json(tmp / "stage5d_f3k_contract.json", contract)

        summary = {
            "schema_version": "reid_stage5d_f3k_summary_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "project_head": head,
            "target_id": TARGET_ID,
            "holdout_sha256": sha_before,
            "complete_universe": UNIVERSE_N,
            "review_eligible": ELIGIBLE_N,
            "review_ineligible": INELIGIBLE_N,
            "review_item_count": ELIGIBLE_N,
            "representative_crop_count": ELIGIBLE_N,
            "contact_sheets": SHEET_COUNT,
            "sheet_distribution": sheet_item_counts,
            "sheet_item_counts": sheet_item_counts,
            "review_videos": VIDEO_PARTS,
            "video_distribution": [ITEMS_PER_PART] * VIDEO_PARTS,
            "blank_template_rows": ELIGIBLE_N,
            "manual_decision_count": 0,
            "manual_ground_truth_decisions": 0,
            "automatic_target_team_jersey_prediction_count": 0,
            "gallery_reads": 0,
            "osnet_loads": 0,
            "similarity_rows": 0,
            "rankings": 0,
            "metrics": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "source_sha_unchanged": True,
            "detection_inference_passes": 0,
            "tracker_passes": 0,
            "segmentation_passes": 0,
            "eligible_review_items": ELIGIBLE_N,
            "ineligible_segments": INELIGIBLE_N,
            "universe_segments": UNIVERSE_N,
            "review_item_id_first": "H2_GT_REVIEW_000001",
            "review_item_id_last": "H2_GT_REVIEW_000141",
            "ranking_rows": 0,
            "f3j_snapshot_sha256": f3j["snapshot_sha256"],
            "f3j_manifest_sha256": f3j["manifest_sha256"],
            "selection_fingerprint_sha256": fp_pass1,
            "determinism_pass_2_match": True,
            "network_used": False,
            "access_audit": access,
        }

        n_before, listing_before = listing_sha(tmp)
        manifest = {
            "schema_version": "reid_stage5d_f3k_manifest_v1",
            "final_status": STATUS,
            "project_head": head,
            "output_root": final_rel,
            "listing_file_count_before_manifest": n_before,
            "listing_sha256_before_manifest": listing_before,
            "contact_sheets": sheet_paths,
            "videos": video_paths,
            "f3j_snapshot_sha256": f3j["snapshot_sha256"],
            "selection_fingerprint_sha256": fp_pass1,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(tmp / "stage5d_f3k_manifest.json", manifest)
        n_final, listing_final = listing_sha(tmp)
        manifest["listing_file_count"] = n_final
        manifest["listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3k_manifest.json", manifest)
        summary["output_file_count"] = n_final
        summary["output_listing_sha256"] = listing_final
        write_json(tmp / "stage5d_f3k_summary.json", summary)

        atomic_publish(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f3k_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build similarity-blind holdout v2 ground-truth review package for target_001."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to independent_holdout_v2_ground_truth_review_stage5d_target_001.yaml",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Project root (defaults to repository root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(Path(args.config).resolve(), args.project_root.resolve())
    except GroundTruthReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={summary['final_status']} "
        f"eligible={summary['eligible_review_items']} "
        f"sheets={summary['contact_sheets']} "
        f"videos={summary['review_videos']} "
        f"decisions={summary['manual_ground_truth_decisions']} "
        f"next_gate={summary['next_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
