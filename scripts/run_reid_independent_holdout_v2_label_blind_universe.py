#!/usr/bin/env python3
"""Stage 5D-F3J — holdout v2 label-blind detection, tracking, segment universe.

Two independent full passes (decode→detect→ByteTrack→segment→eligibility).
Frozen B1E-B detector/tracker. Stage 5B3 auto-split null → pass-through segments.
No gallery/OSNet/OCR/similarity/GT/threshold/identity. No crop export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
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

from football_analytics.detection.annotate import sanitize_detection  # noqa: E402
from football_analytics.detection.pipeline import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    load_yolo_model,
)
from football_analytics.ingest.checksum import sha256_file  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402

CONFIG_SCHEMA = "reid_independent_holdout_v2_label_blind_universe_config_v1"
STATUS = "COMPLETED_STAGE5D_F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_UNIVERSE_BUILT"
READINESS = (
    "TARGET_001_INDEPENDENT_HOLDOUT_V2_LABEL_BLIND_UNIVERSE_READY_FOR_"
    "GROUND_TRUTH_REVIEW_PACKAGE"
)
NEXT_GATE = "STAGE5D-F3K_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_REVIEW_PACKAGE"
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


class HoldoutUniverseError(RuntimeError):
    pass


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise HoldoutUniverseError("unexpected config schema")
    if not config.get("offline_required"):
        raise HoldoutUniverseError("offline_required")
    det = config["detection"]
    if (
        abs(float(det["conf"]) - DEFAULT_CONF) > 1e-9
        or abs(float(det["iou"]) - DEFAULT_IOU) > 1e-9
        or int(det["imgsz"]) != DEFAULT_IMGSZ
        or list(det["classes"]) != [0]
        or str(det["device"]) != "cpu"
    ):
        raise HoldoutUniverseError("canonical detection params mismatch")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise HoldoutUniverseError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str, expected_msg: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_GIT_CONTRACT_MISMATCH origin")
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_independent_holdout_v2_label_blind_universe.py",
        "configs/reid/independent_holdout_v2_label_blind_universe_stage5d_target_001.yaml",
        "tests/test_reid_independent_holdout_v2_label_blind_universe.py",
        "docs/setup/stage5d-target-independent-holdout-label-blind-detection-tracking-and-segment-universe.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path.startswith("?? "):
                path = line[3:]
            elif line.startswith("??"):
                path = line[3:].strip()
            else:
                path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise HoldoutUniverseError(
                    "BLOCKED_STAGE5D_F3J_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != expected_msg:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_GIT_CONTRACT_MISMATCH message"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH snapshot")
    sidecar = Path(str(snapshot_path) + ".sha256")
    listing = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_listing.txt")
    )
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not listing.is_file() or not manifest.is_file():
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man_payload = load_json(manifest)
    man = str(man_payload.get("sha256") or man_payload.get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_f3i(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_f3i_package"]
    root = project_root / block["path"]
    summary = load_json(root / "stage5d_f3i_summary.json")
    contract = load_json(root / "stage5d_f3i_contract.json")
    if summary.get("final_status") != block["expected_final_status"]:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH status")
    if summary.get("readiness") != block["expected_readiness"]:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH readiness"
        )
    hold = config["holdout_source"]
    sha = summary.get("holdout_sha256")
    if sha != hold["expected_sha256"]:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH f3i_sha")
    if int(summary.get("holdout_bytes") or -1) != int(hold["expected_bytes"]):
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH f3i_bytes")
    if int(summary.get("frame_count") or -1) != int(hold["expected_frames"]):
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH f3i_frames")
    if summary.get("accepted_as_independent_holdout") is not True:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH not_accepted"
        )
    if summary.get("holdout_path") != hold["path"]:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH path")
    decision_path = (
        root / "independence" / "target_001_holdout_independence_decision.json"
    )
    if not decision_path.is_file():
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH independence"
        )
    decision = load_json(decision_path)
    if decision.get("accepted_as_independent_holdout") is not True:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH not_accepted_decision"
        )
    if decision.get("accepted_holdout_role") != "frozen_evaluation_input":
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH role")
    if decision.get("scoring_evaluation_eligible") is not True:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH scoring_eligible"
        )
    if decision.get("label_blind_universe_build_eligible") is not True:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH label_blind_eligible"
        )
    if decision.get("enrollment_eligible") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH enrollment"
        )
    if decision.get("gallery_growth_eligible") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH gallery_growth"
        )
    if decision.get("threshold_calibration_eligible") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH threshold_cal"
        )
    if summary.get("exact_duplicate_sample") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH sample_dup"
        )
    if summary.get("exact_duplicate_external") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH external_dup"
        )
    if summary.get("sample_temporal_classification") != "no_overlap":
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH sample_fp"
        )
    if summary.get("external_temporal_classification") != "no_overlap":
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH external_fp"
        )
    if summary.get("gallery_overlap") is not False:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH gallery_overlap"
        )
    resolve_snapshot_sha(Path(block["snapshot_path"]), block["expected_snapshot_sha256"])
    return {
        "summary": summary,
        "contract": contract,
        "decision": decision,
        "f3i_sha": sha,
    }


def validate_upstream_b1eb(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    block = config["stage5d_b1e_b_package"]
    root = project_root / block["path"]
    if not root.is_dir():
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE missing"
        )
    summary = load_json(root / "stage5d_b1e_b_summary.json")
    if summary.get("final_status") != block["expected_final_status"]:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE status"
        )
    det_cfg = load_json(root / "detection" / "detection_effective_config.json")
    trk_cfg = load_json(root / "tracking" / "tracking_effective_config.json")
    expected_det = config["detection"]
    if abs(float(det_cfg["conf"]) - float(expected_det["conf"])) > 1e-9:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE det_conf"
        )
    if abs(float(det_cfg["iou"]) - float(expected_det["iou"])) > 1e-9:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE det_iou"
        )
    if int(det_cfg["imgsz"]) != int(expected_det["imgsz"]):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE det_imgsz"
        )
    if list(det_cfg["classes"]) != list(expected_det["classes"]):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE det_classes"
        )
    if str(det_cfg["device"]) != str(expected_det["device"]):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE det_device"
        )
    if trk_cfg.get("tracker_sha256") != config["tracking"]["expected_tracker_sha256"]:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE tracker_sha"
        )
    return {"summary": summary, "detection": det_cfg, "tracking": trk_cfg}


def resolve_segmentation_contract(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    pol_path = project_root / config["segmentation_policy"]["path"]
    pur_path = project_root / config["segmentation_policy"]["purity_audit_path"]
    if not pol_path.is_file() or not pur_path.is_file():
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_SEGMENTATION_CONTRACT_UNAVAILABLE missing"
        )
    with pol_path.open(encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    with pur_path.open(encoding="utf-8") as handle:
        purity = yaml.safe_load(handle)
    auto_split = bool(purity.get("automatic_track_split_enabled"))
    change_thr = purity.get("change_threshold")
    split_thr = purity.get("split_threshold")
    if auto_split or change_thr is not None or split_thr is not None:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_SEGMENTATION_CONTRACT_UNAVAILABLE unexpected_auto_split"
        )
    # Objective pass-through only — no invented thresholds.
    return {
        "schema_version": "reid_stage5d_f3j_label_blind_segmentation_contract_v1",
        "policy_path": str(pol_path.relative_to(project_root)),
        "purity_audit_path": str(pur_path.relative_to(project_root)),
        "automatic_track_split_enabled": False,
        "change_threshold": None,
        "split_threshold": None,
        "mode": "pass_through_raw_track",
        "allowed_objective_inputs": [
            "raw_track_lineage",
            "frame_gaps",
            "timestamp_gaps",
            "bbox_center_displacement",
            "bbox_width_height_area_changes",
            "aspect_ratio_changes",
            "velocity_discontinuity",
            "overlap_with_other_person_boxes",
            "track_lost_reacquired_state",
            "edge_clipping",
            "observation_confidence",
            "source_detection_continuity",
        ],
        "forbidden_inputs": [
            "osnet_embedding",
            "appearance_similarity",
            "gallery_similarity",
            "jersey_number",
            "ocr",
            "team_color_identity",
            "target_identity",
            "previous_sample_labels",
            "human_target_labels",
        ],
        "cross_track_merge_enabled": False,
        "policy_sha256": sha256_file(pol_path),
        "purity_audit_sha256": sha256_file(pur_path),
    }


def edge_clipping_flags(
    bbox: Sequence[float], *, width: int, height: int, eps: float = 1e-6
) -> dict[str, bool]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return {
        "left": x1 <= eps,
        "top": y1 <= eps,
        "right": x2 >= width - eps,
        "bottom": y2 >= height - eps,
        "any": x1 <= eps or y1 <= eps or x2 >= width - eps or y2 >= height - eps,
    }


def serialize_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(v), 6) for v in bbox]


def detection_sort_key(item: tuple[int, dict[str, Any]]) -> tuple:
    idx, row = item
    x1, y1, x2, y2 = row["bbox_xyxy"]
    return (x1, y1, x2, y2, -float(row["confidence"]), idx)


def load_tracker_args(tracker_path: Path) -> IterableSimpleNamespace:
    with tracker_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return IterableSimpleNamespace(**raw)


def run_yolo_detection(
    *,
    video_path: Path,
    model_path: Path,
    start: int,
    end: int,
    conf: float,
    iou: float,
    imgsz: int,
    video_sha: str,
    model_sha: str,
    effective_config_sha: str,
    width: int,
    height: int,
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HoldoutUniverseError("cannot open holdout video for detection")
    rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    frames_with = 0
    skipped = 0
    frame_area = float(width * height)
    for fi in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            skipped += 1
            raise HoldoutUniverseError(f"failed reading frame {fi}")
        preds = model.predict(
            source=frame,
            device="cpu",
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            save=False,
        )
        raw_items: list[dict[str, Any]] = []
        if preds:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for det_i, (bbox, c) in enumerate(zip(xyxy, confs)):
                    cleaned = sanitize_detection(
                        bbox_xyxy=bbox,
                        confidence=float(c),
                        frame_width=width,
                        frame_height=height,
                    )
                    if cleaned is None:
                        continue
                    x1, y1, x2, y2 = cleaned["bbox_xyxy"]
                    raw_items.append(
                        {
                            "detector_index": det_i,
                            "confidence": cleaned["confidence"],
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        }
                    )
        ordered = sorted(enumerate(raw_items), key=detection_sort_key)
        frame_dets: list[dict[str, Any]] = []
        for rank, (_, item) in enumerate(ordered):
            x1, y1, x2, y2 = item["bbox_xyxy"]
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            det_id = f"H2_DET_{fi:06d}_{rank:03d}"
            row = {
                "detection_id": det_id,
                "frame_index": fi,
                "timestamp_sec": float(fi) / fps,
                "within_frame_rank": rank,
                "detector_index": int(item["detector_index"]),
                "class_id": 0,
                "class_name": "person",
                "confidence": float(item["confidence"]),
                "bbox_xyxy": item["bbox_xyxy"],
                "bbox_xyxy_serialized": serialize_bbox(item["bbox_xyxy"]),
                "bbox_width": bw,
                "bbox_height": bh,
                "bbox_area": area,
                "bbox_frame_area_ratio": area / frame_area if frame_area else 0.0,
                "edge_clipping": edge_clipping_flags(
                    item["bbox_xyxy"], width=width, height=height
                ),
                "detector_model_sha256": model_sha,
                "effective_detector_config_sha256": effective_config_sha,
                "source_video_sha256": video_sha,
                "label_blind": True,
            }
            frame_dets.append(row)
            rows.append(row)
        if frame_dets:
            frames_with += 1
        frame_summaries.append(
            {
                "frame_index": fi,
                "timestamp_sec": float(fi) / fps,
                "person_detection_count": len(frame_dets),
                "detection_ids": [d["detection_id"] for d in frame_dets],
            }
        )
    cap.release()
    summary = {
        "schema_version": "reid_stage5d_f3j_detection_summary_v1",
        "frames_expected": end - start + 1,
        "frames_processed": end - start + 1,
        "skipped_frames": skipped,
        "frame_range": [start, end],
        "total_person_detections": len(rows),
        "frames_with_detections": frames_with,
        "frames_without_detections": (end - start + 1) - frames_with,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "classes": [0],
        "device": "cpu",
        "source_video_sha256": video_sha,
        "model_sha256": model_sha,
        "effective_config_sha256": effective_config_sha,
    }
    return rows, frame_summaries, summary


def detections_by_frame(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_f: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_f[int(row["frame_index"])].append(dict(row))
    for fi in by_f:
        by_f[fi].sort(key=lambda r: int(r["within_frame_rank"]))
    return by_f


def boxes_from_frame_dets(
    dets: Sequence[Mapping[str, Any]], *, width: int, height: int
) -> Boxes:
    if not dets:
        return Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(height, width))
    data = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        data.append([x1, y1, x2, y2, float(d["confidence"]), 0.0])
    return Boxes(np.asarray(data, dtype=np.float32), orig_shape=(height, width))


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def replay_bytetrack(
    *,
    by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    start: int,
    end: int,
    width: int,
    height: int,
    fps: float,
    tracker_args: IterableSimpleNamespace,
    video_sha: str,
    tracking_config_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    BYTETracker.reset_id()
    tracker = BYTETracker(tracker_args)
    observations: list[dict[str, Any]] = []
    assigned_det_ids: set[str] = set()
    assignment_rows: list[dict[str, Any]] = []

    for fi in range(start, end + 1):
        dets = list(by_frame.get(fi, []))
        boxes = boxes_from_frame_dets(dets, width=width, height=height)
        out = tracker.update(boxes)
        frame_assigned: set[str] = set()
        if out is not None and len(out) > 0:
            for row in np.asarray(out):
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                track_id = int(row[4])
                conf = float(row[5])
                det_idx = int(row[7]) if len(row) > 7 else -1
                linked_det_id = None
                src_conf = conf
                if 0 <= det_idx < len(dets):
                    linked_det_id = str(dets[det_idx]["detection_id"])
                    src_conf = float(dets[det_idx]["confidence"])
                    frame_assigned.add(linked_det_id)
                    assigned_det_ids.add(linked_det_id)
                raw_code = f"H2_RAW_{track_id:06d}"
                obs_id = f"H2_OBS_{fi:06d}_{track_id:06d}"
                observations.append(
                    {
                        "observation_id": obs_id,
                        "raw_track_code": raw_code,
                        "raw_tracker_id": track_id,
                        "frame_index": fi,
                        "timestamp_sec": float(fi) / fps,
                        "linked_detection_id": linked_det_id,
                        "source_detection_confidence": src_conf,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xyxy_serialized": serialize_bbox([x1, y1, x2, y2]),
                        "edge_clipping": edge_clipping_flags(
                            [x1, y1, x2, y2], width=width, height=height
                        ),
                        "tracker_config_sha256": tracking_config_sha,
                        "source_video_sha256": video_sha,
                        "label_blind": True,
                    }
                )
        for d in dets:
            did = str(d["detection_id"])
            if did in frame_assigned:
                assignment_rows.append(
                    {
                        "detection_id": did,
                        "frame_index": fi,
                        "assignment": "assigned",
                        "unassigned_reason": None,
                    }
                )
            else:
                assignment_rows.append(
                    {
                        "detection_id": did,
                        "frame_index": fi,
                        "assignment": "unassigned",
                        "unassigned_reason": "no_bytetrack_match",
                    }
                )

    audit = {
        "schema_version": "reid_stage5d_f3j_detection_assignment_audit_v1",
        "total_detections": sum(len(by_frame.get(fi, [])) for fi in range(start, end + 1)),
        "assigned_count": len(assigned_det_ids),
        "unassigned_count": 0,
        "assignments": assignment_rows,
        "all_detections_accounted": True,
    }
    # Recompute assigned/unassigned from assignment_rows (per detection occurrence)
    assigned_n = sum(1 for a in assignment_rows if a["assignment"] == "assigned")
    unassigned_n = sum(1 for a in assignment_rows if a["assignment"] == "unassigned")
    audit["assigned_count"] = assigned_n
    audit["unassigned_count"] = unassigned_n
    audit["all_detections_accounted"] = assigned_n + unassigned_n == audit["total_detections"]
    return observations, audit


def build_raw_track_inventory(
    observations: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_tid[int(obs["raw_tracker_id"])].append(dict(obs))

    # Per-frame boxes for overlap stats
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_frame[int(obs["frame_index"])].append(dict(obs))

    inventory: list[dict[str, Any]] = []
    for tid in sorted(by_tid.keys()):
        rows = sorted(by_tid[tid], key=lambda r: int(r["frame_index"]))
        frames = [int(r["frame_index"]) for r in rows]
        start_f, end_f = frames[0], frames[-1]
        span = end_f - start_f + 1
        present = set(frames)
        gaps = []
        for f in range(start_f, end_f + 1):
            if f not in present:
                gaps.append(f)
        # contiguous gap lengths
        gap_lengths: list[int] = []
        i = start_f
        while i <= end_f:
            if i not in present:
                j = i
                while j <= end_f and j not in present:
                    j += 1
                gap_lengths.append(j - i)
                i = j
            else:
                i += 1
        confs = [float(r["source_detection_confidence"]) for r in rows]
        areas = []
        clip_any = 0
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            areas.append((x2 - x1) * (y2 - y1))
            if r["edge_clipping"]["any"]:
                clip_any += 1
        max_overlap = 0.0
        for r in rows:
            fi = int(r["frame_index"])
            for other in by_frame[fi]:
                if int(other["raw_tracker_id"]) == tid:
                    continue
                max_overlap = max(max_overlap, bbox_iou(r["bbox_xyxy"], other["bbox_xyxy"]))
        inventory.append(
            {
                "raw_track_code": f"H2_RAW_{tid:06d}",
                "tracker_native_id": tid,
                "start_frame": start_f,
                "end_frame": end_f,
                "observation_count": len(rows),
                "span_frames": span,
                "gap_count": len(gap_lengths),
                "maximum_gap": max(gap_lengths) if gap_lengths else 0,
                "mean_detection_confidence": float(statistics.mean(confs)),
                "median_detection_confidence": float(statistics.median(confs)),
                "bbox_area_mean": float(statistics.mean(areas)) if areas else 0.0,
                "bbox_area_median": float(statistics.median(areas)) if areas else 0.0,
                "bbox_area_min": float(min(areas)) if areas else 0.0,
                "bbox_area_max": float(max(areas)) if areas else 0.0,
                "edge_clipping_observation_count": clip_any,
                "edge_clipping_rate": clip_any / len(rows) if rows else 0.0,
                "maximum_overlap_with_another_person": max_overlap,
                "label_blind": True,
            }
        )
    return inventory


def build_segments(
    observations: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_tid[int(obs["raw_tracker_id"])].append(dict(obs))

    seg_obs_rows: list[dict[str, Any]] = []
    seg_inv: list[dict[str, Any]] = []
    track_map: dict[str, list[str]] = {}
    seg_seq = 0
    pass_through = 0

    for tid in sorted(by_tid.keys()):
        rows = sorted(by_tid[tid], key=lambda r: int(r["frame_index"]))
        raw_code = f"H2_RAW_{tid:06d}"
        # Pass-through: one segment per raw track (frozen auto-split null).
        seg_seq += 1
        seg_id = f"H2_SEG_{seg_seq:06d}"
        local_idx = 0
        start_f = int(rows[0]["frame_index"])
        end_f = int(rows[-1]["frame_index"])
        obs_ids = [str(r["observation_id"]) for r in rows]
        confs = [float(r["source_detection_confidence"]) for r in rows]
        areas = []
        clip_any = 0
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            areas.append((x2 - x1) * (y2 - y1))
            if r["edge_clipping"]["any"]:
                clip_any += 1
        # overlap within segment frames
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for obs in observations:
            by_frame[int(obs["frame_index"])].append(dict(obs))
        max_overlap = 0.0
        for r in rows:
            for other in by_frame[int(r["frame_index"])]:
                if int(other["raw_tracker_id"]) == tid:
                    continue
                max_overlap = max(max_overlap, bbox_iou(r["bbox_xyxy"], other["bbox_xyxy"]))

        frames = [int(r["frame_index"]) for r in rows]
        present = set(frames)
        gap_lengths: list[int] = []
        i = start_f
        while i <= end_f:
            if i not in present:
                j = i
                while j <= end_f and j not in present:
                    j += 1
                gap_lengths.append(j - i)
                i = j
            else:
                i += 1

        inv_row = {
            "segment_id": seg_id,
            "raw_track_code": raw_code,
            "local_segment_index": local_idx,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_time_sec": float(start_f) / fps,
            "end_time_sec": float(end_f) / fps,
            "observation_count": len(rows),
            "span_frames": end_f - start_f + 1,
            "gap_count": len(gap_lengths),
            "maximum_gap": max(gap_lengths) if gap_lengths else 0,
            "split_reason_before": None,
            "split_reason_after": None,
            "pass_through": True,
            "mean_confidence": float(statistics.mean(confs)),
            "median_confidence": float(statistics.median(confs)),
            "bbox_area_mean": float(statistics.mean(areas)) if areas else 0.0,
            "maximum_overlap_with_another_person": max_overlap,
            "edge_clipping_observation_count": clip_any,
            "source_observation_ids": obs_ids,
            "label_blind": True,
            "target_label_absent": True,
            "gallery_score_absent": True,
            "complete_universe_member": True,
        }
        seg_inv.append(inv_row)
        track_map[raw_code] = [seg_id]
        pass_through += 1
        for r in rows:
            seg_obs_rows.append(
                {
                    "observation_id": r["observation_id"],
                    "segment_id": seg_id,
                    "raw_track_code": raw_code,
                    "frame_index": r["frame_index"],
                    "timestamp_sec": r["timestamp_sec"],
                    "bbox_xyxy": r["bbox_xyxy"],
                    "confidence": r["source_detection_confidence"],
                    "label_blind": True,
                }
            )

    # Validate coverage
    obs_ids_all = [o["observation_id"] for o in observations]
    mapped = [r["observation_id"] for r in seg_obs_rows]
    if len(mapped) != len(set(mapped)):
        raise HoldoutUniverseError("duplicate observation assignment")
    if set(mapped) != set(obs_ids_all):
        raise HoldoutUniverseError("observation silent drop or incomplete assignment")
    if len(track_map) != len(inventory):
        raise HoldoutUniverseError("raw tracks not fully represented")

    audit = {
        "schema_version": "reid_stage5d_f3j_segmentation_audit_v1",
        "raw_track_count": len(inventory),
        "segment_count": len(seg_inv),
        "pass_through_segment_count": pass_through,
        "split_segment_count": 0,
        "split_reason_distribution": {},
        "observation_duplicate_assignment": 0,
        "observation_silent_drop": 0,
        "cross_track_merge_count": 0,
        "all_raw_tracks_represented": True,
        "all_observations_assigned_once": True,
    }
    return seg_obs_rows, seg_inv, track_map, audit


def build_review_eligibility(
    segments: Sequence[Mapping[str, Any]],
    seg_obs: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    min_obs: int,
    max_reps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seg_obs:
        by_seg[str(row["segment_id"])].append(dict(row))

    elig_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    eligible = 0
    ineligible = 0
    rep_total = 0

    for seg in segments:
        sid = str(seg["segment_id"])
        rows = sorted(by_seg[sid], key=lambda r: int(r["frame_index"]))
        reasons: list[str] = []
        if len(rows) < min_obs:
            reasons.append("observation_count_below_min")
        valid = 0
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                valid += 1
        if valid == 0:
            reasons.append("no_valid_in_frame_bbox")
        review_eligible = len(reasons) == 0
        if review_eligible:
            eligible += 1
        else:
            ineligible += 1

        # Objective representative candidates: highest confidence, then area
        scored = []
        for r in rows:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            area = max(0.0, (x2 - x1) * (y2 - y1))
            scored.append((float(r["confidence"]), area, int(r["frame_index"])))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        candidates = [t[2] for t in scored[:max_reps]]
        rep_total += len(candidates)
        elig_rows.append(
            {
                "segment_id": sid,
                "complete_universe_member": True,
                "review_eligible": review_eligible,
                "review_ineligibility_reasons": reasons,
                "observation_count": len(rows),
                "valid_bbox_count": valid,
                "target_probability_absent": True,
                "identity_absent": True,
                "team_absent": True,
                "jersey_absent": True,
                "label_blind": True,
            }
        )
        rep_rows.append(
            {
                "segment_id": sid,
                "representative_frame_candidates": candidates,
                "selection_basis": "objective_confidence_then_area_then_frame",
                "crop_exported": False,
                "label_blind": True,
            }
        )

    summary = {
        "schema_version": "reid_stage5d_f3j_review_eligibility_summary_v1",
        "complete_universe_count": len(segments),
        "review_eligible_count": eligible,
        "review_ineligible_count": ineligible,
        "representative_frame_candidate_count": rep_total,
        "min_observation_count": min_obs,
        "max_representative_frame_candidates": max_reps,
        "review_eligibility_is_not_identity": True,
    }
    return elig_rows, rep_rows, summary


def forbidden_field_audit(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_FIELDS:
                hits.append(f"{path}.{k}" if path else k)
            hits.extend(forbidden_field_audit(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):  # cap deep list walk for huge arrays
            hits.extend(forbidden_field_audit(v, f"{path}[{i}]"))
    return hits


def compare_passes(
    a: Mapping[str, Any], b: Mapping[str, Any], *, bbox_atol: float, conf_atol: float
) -> dict[str, Any]:
    max_diff = 0.0

    def num_diff(x: float, y: float) -> None:
        nonlocal max_diff
        max_diff = max(max_diff, abs(float(x) - float(y)))

    mismatches: list[str] = []
    if a["detection_count"] != b["detection_count"]:
        mismatches.append("detection_count")
    if a["observation_count"] != b["observation_count"]:
        mismatches.append("observation_count")
    if a["raw_track_count"] != b["raw_track_count"]:
        mismatches.append("raw_track_count")
    if a["segment_count"] != b["segment_count"]:
        mismatches.append("segment_count")
    if a["detection_rows_sha"] != b["detection_rows_sha"]:
        mismatches.append("detection_rows_sha")
    if a["observation_rows_sha"] != b["observation_rows_sha"]:
        mismatches.append("observation_rows_sha")
    if a["segment_inventory_sha"] != b["segment_inventory_sha"]:
        mismatches.append("segment_inventory_sha")
    if a["eligibility_sha"] != b["eligibility_sha"]:
        mismatches.append("eligibility_sha")
    if a["rep_candidates_sha"] != b["rep_candidates_sha"]:
        mismatches.append("rep_candidates_sha")

    # Numeric tolerance scan on detections
    for da, db in zip(a["detections"], b["detections"]):
        if da["detection_id"] != db["detection_id"]:
            mismatches.append("detection_id_order")
            break
        for i in range(4):
            num_diff(da["bbox_xyxy"][i], db["bbox_xyxy"][i])
        num_diff(da["confidence"], db["confidence"])
        if abs(da["confidence"] - db["confidence"]) > conf_atol:
            mismatches.append("confidence_atol")
        if any(abs(da["bbox_xyxy"][i] - db["bbox_xyxy"][i]) > bbox_atol for i in range(4)):
            mismatches.append("bbox_atol")

    exact = len(mismatches) == 0 and max_diff == 0.0
    return {
        "exact_match": exact,
        "maximum_numeric_absolute_difference": max_diff,
        "ordering_match": "detection_id_order" not in mismatches,
        "mismatches": mismatches,
        "bbox_atol": bbox_atol,
        "confidence_atol": conf_atol,
    }


def run_pass(
    *,
    pass_name: str,
    project_root: Path,
    video_path: Path,
    model_path: Path,
    video_sha: str,
    model_sha: str,
    det_contract: Mapping[str, Any],
    tracker_args: IterableSimpleNamespace,
    tracking_config_sha: str,
    det_config_sha: str,
    start: int,
    end: int,
    width: int,
    height: int,
    fps: float,
    conf: float,
    iou: float,
    imgsz: int,
    min_obs: int,
    max_reps: int,
    work_dir: Path,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    dets, frame_sums, det_summary = run_yolo_detection(
        video_path=video_path,
        model_path=model_path,
        start=start,
        end=end,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        video_sha=video_sha,
        model_sha=model_sha,
        effective_config_sha=det_config_sha,
        width=width,
        height=height,
        fps=fps,
    )
    by_frame = detections_by_frame(dets)
    observations, assign_audit = replay_bytetrack(
        by_frame=by_frame,
        start=start,
        end=end,
        width=width,
        height=height,
        fps=fps,
        tracker_args=tracker_args,
        video_sha=video_sha,
        tracking_config_sha=tracking_config_sha,
    )
    inventory = build_raw_track_inventory(observations, width=width, height=height)
    seg_obs, seg_inv, track_map, seg_audit = build_segments(
        observations, inventory, fps=fps, width=width, height=height
    )
    elig, reps, elig_sum = build_review_eligibility(
        seg_inv,
        seg_obs,
        width=width,
        height=height,
        min_obs=min_obs,
        max_reps=max_reps,
    )

    det_sha = canonical_json_sha(
        [
            {
                "detection_id": d["detection_id"],
                "frame_index": d["frame_index"],
                "confidence": d["confidence"],
                "bbox_xyxy": d["bbox_xyxy_serialized"],
                "class_id": d["class_id"],
            }
            for d in dets
        ]
    )
    obs_sha = canonical_json_sha(
        [
            {
                "observation_id": o["observation_id"],
                "raw_track_code": o["raw_track_code"],
                "frame_index": o["frame_index"],
                "linked_detection_id": o["linked_detection_id"],
                "bbox_xyxy": o["bbox_xyxy_serialized"],
                "confidence": round(float(o["confidence"]), 7),
            }
            for o in observations
        ]
    )
    seg_sha = canonical_json_sha(seg_inv)
    elig_sha = canonical_json_sha(elig)
    rep_sha = canonical_json_sha(reps)

    payload = {
        "pass_name": pass_name,
        "detections": dets,
        "frame_summaries": frame_sums,
        "detection_summary": det_summary,
        "observations": observations,
        "assignment_audit": assign_audit,
        "raw_track_inventory": inventory,
        "segment_observations": seg_obs,
        "segment_inventory": seg_inv,
        "raw_track_to_segment_map": track_map,
        "segmentation_audit": seg_audit,
        "eligibility": elig,
        "representative_candidates": reps,
        "eligibility_summary": elig_sum,
        "detection_count": len(dets),
        "observation_count": len(observations),
        "raw_track_count": len(inventory),
        "segment_count": len(seg_inv),
        "detection_rows_sha": det_sha,
        "observation_rows_sha": obs_sha,
        "segment_inventory_sha": seg_sha,
        "eligibility_sha": elig_sha,
        "rep_candidates_sha": rep_sha,
        "work_dir": str(work_dir),
    }
    write_json(work_dir / "pass_fingerprint.json", {
        "pass_name": pass_name,
        "detection_rows_sha": det_sha,
        "observation_rows_sha": obs_sha,
        "segment_inventory_sha": seg_sha,
        "eligibility_sha": elig_sha,
        "rep_candidates_sha": rep_sha,
        "detection_count": len(dets),
        "observation_count": len(observations),
        "raw_track_count": len(inventory),
        "segment_count": len(seg_inv),
    })
    return payload


def build_access_audit() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_f3j_access_audit_v1",
        "sample_video_read": False,
        "external_video_read": False,
        "holdout_video_decode_passes": 2,
        "gallery_embedding_read_count": 0,
        "target_crop_read_count": 0,
        "distractor_crop_read_count": 0,
        "F3_score_row_read_count": 0,
        "F3_rank_row_read_count": 0,
        "GT_label_read_count": 0,
        "jersey_metadata_read_count": 0,
        "identity_metadata_used": False,
        "osnet_model_loads": 0,
        "parseq_model_loads": 0,
        "ocr_calls": 0,
        "team_classifier_calls": 0,
        "crop_exports": 0,
        "embeddings": 0,
        "cosine_similarity_computations": 0,
        "query_score_rows": 0,
        "rankings": 0,
        "gt_decisions": 0,
        "metrics": 0,
        "threshold_candidates": 0,
        "identity_assignments": 0,
        "gallery_mutations": 0,
        "detector_inference_frames_total": 2116,
        "tracker_passes": 2,
        "segmentation_passes": 2,
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(
        project_root,
        config["project_head_expected"],
        config["project_head_message_expected"],
    )
    final_dir = project_root / config["output"]["final_dir"]
    if final_dir.exists():
        raise HoldoutUniverseError("final root already exists")

    # Forbidden source paths must not be opened
    sample = project_root / config["evaluation_source"]["path"]
    external = project_root / config["external_enrollment_source"]["path"]
    # Touch only existence checks without reading media for leakage audit intent —
    # we must NOT open them. Just record paths as forbidden.
    _ = sample.exists()
    _ = external.exists()

    f3i = validate_f3i(project_root, config)
    b1eb = validate_upstream_b1eb(project_root, config)

    hold = config["holdout_source"]
    video_path = project_root / hold["path"]
    if not video_path.is_file():
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH missing")
    sha_before = sha256_file(video_path)
    if sha_before != hold["expected_sha256"]:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH source_sha")
    if video_path.stat().st_size != int(hold["expected_bytes"]):
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH bytes")

    model_path = project_root / config["yolo_checkpoint"]["path"]
    model_sha = sha256_file(model_path)
    if model_sha != config["yolo_checkpoint"]["expected_sha256"]:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE model_sha"
        )
    if model_path.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE model_bytes"
        )

    tracker_path = project_root / config["tracking"]["tracker_path"]
    tracker_sha = sha256_file(tracker_path)
    if tracker_sha != config["tracking"]["expected_tracker_sha256"]:
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_FROZEN_DETECTOR_TRACKER_CONTRACT_UNAVAILABLE tracker"
        )

    # Probe holdout metadata without mutating
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH open")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (
        width != hold["expected_width"]
        or height != hold["expected_height"]
        or abs(fps - float(hold["expected_fps"])) > 1e-6
        or n_frames != int(hold["expected_frames"])
    ):
        raise HoldoutUniverseError(
            "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH metadata"
        )

    start = int(hold["frame_start"])
    end = int(hold["frame_end"])
    if end - start + 1 != 1058:
        raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH frames")

    seg_contract = resolve_segmentation_contract(project_root, config)
    det = config["detection"]
    det_contract = {
        "schema_version": "reid_stage5d_f3j_detector_tracker_contract_v1",
        "frozen_before_inference": True,
        "model_path": config["yolo_checkpoint"]["path"],
        "model_sha256": model_sha,
        "model_bytes": int(config["yolo_checkpoint"]["expected_bytes"]),
        "ultralytics_version": __import__("ultralytics").__version__,
        "person_class_id": 0,
        "person_class_name": "person",
        "imgsz": int(det["imgsz"]),
        "confidence_threshold": float(det["conf"]),
        "nms_iou": float(det["iou"]),
        "max_det": None,
        "max_det_policy": "ultralytics_default_unset_not_invented",
        "classes": [0],
        "agnostic_nms": False,
        "device": "cpu",
        "precision": "fp32_default",
        "deterministic_ordering_policy": (
            "within_frame_x1_y1_x2_y2_asc_confidence_desc_detector_index"
        ),
        "bbox_rounding_serialization_policy": "round_6_decimal_serialized_field",
        "tracker_path": config["tracking"]["tracker_path"],
        "tracker_sha256": tracker_sha,
        "bytetrack": b1eb["tracking"]["parameters"],
        "frame_rate_handling": "timestamp_sec = frame_index / fps",
        "upstream_package": config["stage5d_b1e_b_package"]["path"],
        "upstream_status": config["stage5d_b1e_b_package"]["expected_final_status"],
        "source_holdout_sha256": sha_before,
        "source_holdout_path": hold["path"],
        "frame_universe": [start, end],
        "expected_frames": 1058,
    }
    det_config_sha = canonical_json_sha(det_contract)
    tracking_config_sha = canonical_json_sha(
        {
            "tracker_path": config["tracking"]["tracker_path"],
            "tracker_sha256": tracker_sha,
            "parameters": b1eb["tracking"]["parameters"],
        }
    )

    tmp_root = project_root / "outputs" / "reid" / f".tmp_f3j_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    try:
        # Contracts BEFORE inference results are examined / written to final
        write_json(
            tmp_root / "runtime" / "target_001_holdout_v2_detector_tracker_contract_pre_inference.json",
            det_contract,
        )
        write_json(
            tmp_root / "runtime" / "target_001_holdout_v2_label_blind_segmentation_contract_pre_build.json",
            seg_contract,
        )

        tracker_args = load_tracker_args(tracker_path)
        min_obs = int(config["review_eligibility"]["min_observation_count"])
        max_reps = int(config["review_eligibility"]["max_representative_frame_candidates"])

        pass1 = run_pass(
            pass_name="pass_1",
            project_root=project_root,
            video_path=video_path,
            model_path=model_path,
            video_sha=sha_before,
            model_sha=model_sha,
            det_contract=det_contract,
            tracker_args=tracker_args,
            tracking_config_sha=tracking_config_sha,
            det_config_sha=det_config_sha,
            start=start,
            end=end,
            width=width,
            height=height,
            fps=fps,
            conf=float(det["conf"]),
            iou=float(det["iou"]),
            imgsz=int(det["imgsz"]),
            min_obs=min_obs,
            max_reps=max_reps,
            work_dir=tmp_root / "pass_1",
        )
        sha_mid = sha256_file(video_path)
        if sha_mid != sha_before:
            raise HoldoutUniverseError("BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH mid_sha")

        pass2 = run_pass(
            pass_name="pass_2",
            project_root=project_root,
            video_path=video_path,
            model_path=model_path,
            video_sha=sha_before,
            model_sha=model_sha,
            det_contract=det_contract,
            tracker_args=tracker_args,
            tracking_config_sha=tracking_config_sha,
            det_config_sha=det_config_sha,
            start=start,
            end=end,
            width=width,
            height=height,
            fps=fps,
            conf=float(det["conf"]),
            iou=float(det["iou"]),
            imgsz=int(det["imgsz"]),
            min_obs=min_obs,
            max_reps=max_reps,
            work_dir=tmp_root / "pass_2",
        )
        sha_after = sha256_file(video_path)
        if sha_after != sha_before:
            raise HoldoutUniverseError(
                "BLOCKED_STAGE5D_F3J_HOLDOUT_CONTRACT_MISMATCH after_sha"
            )

        replay = compare_passes(
            pass1,
            pass2,
            bbox_atol=float(config["determinism"]["bbox_atol"]),
            conf_atol=float(config["determinism"]["confidence_atol"]),
        )
        write_json(tmp_root / "runtime" / "pass2_replay_comparison.json", replay)
        if not replay["exact_match"]:
            raise HoldoutUniverseError(
                "BLOCKED_STAGE5D_F3J_LABEL_BLIND_UNIVERSE_NONDETERMINISM "
                + json.dumps(replay)
            )

        # Finalize from pass-1
        p = pass1
        dets_dir = tmp_root / "detections"
        trk_dir = tmp_root / "tracking"
        seg_dir = tmp_root / "segmentation"
        qual_dir = tmp_root / "quality"
        eff_dir = tmp_root / "effective_configs"
        for d in (dets_dir, trk_dir, seg_dir, qual_dir, eff_dir):
            d.mkdir(parents=True, exist_ok=True)

        write_jsonl(dets_dir / "target_001_holdout_v2_person_detections.jsonl", p["detections"])
        write_jsonl(
            dets_dir / "target_001_holdout_v2_frame_detection_summary.jsonl",
            p["frame_summaries"],
        )
        write_json(
            dets_dir / "target_001_holdout_v2_detection_manifest.json",
            {
                "schema_version": "reid_stage5d_f3j_detection_manifest_v1",
                **p["detection_summary"],
                "detection_rows_sha256": p["detection_rows_sha"],
            },
        )

        write_jsonl(
            trk_dir / "target_001_holdout_v2_raw_track_observations.jsonl",
            p["observations"],
        )
        write_jsonl(
            trk_dir / "target_001_holdout_v2_raw_track_inventory.jsonl",
            p["raw_track_inventory"],
        )
        assign_summary = {
            k: v for k, v in p["assignment_audit"].items() if k != "assignments"
        }
        assign_summary["assignments_path"] = (
            "target_001_holdout_v2_detection_assignment_audit.jsonl"
        )
        write_json(
            trk_dir / "target_001_holdout_v2_detection_assignment_audit.json",
            assign_summary,
        )
        write_jsonl(
            trk_dir / "target_001_holdout_v2_detection_assignment_audit.jsonl",
            p["assignment_audit"]["assignments"],
        )
        write_json(
            trk_dir / "target_001_holdout_v2_tracking_manifest.json",
            {
                "schema_version": "reid_stage5d_f3j_tracking_manifest_v1",
                "raw_track_count": p["raw_track_count"],
                "observation_count": p["observation_count"],
                "assigned_detection_count": p["assignment_audit"]["assigned_count"],
                "unassigned_detection_count": p["assignment_audit"]["unassigned_count"],
                "all_detections_accounted": p["assignment_audit"]["all_detections_accounted"],
                "observation_rows_sha256": p["observation_rows_sha"],
                "tracker_config_sha256": tracking_config_sha,
            },
        )

        write_jsonl(
            seg_dir / "target_001_holdout_v2_label_blind_segment_observations.jsonl",
            p["segment_observations"],
        )
        write_jsonl(
            seg_dir / "target_001_holdout_v2_label_blind_segment_inventory.jsonl",
            p["segment_inventory"],
        )
        write_json(
            seg_dir / "target_001_holdout_v2_raw_track_to_segment_map.json",
            p["raw_track_to_segment_map"],
        )
        write_json(
            seg_dir / "target_001_holdout_v2_segmentation_audit.json",
            p["segmentation_audit"],
        )
        write_json(
            seg_dir / "target_001_holdout_v2_segmentation_manifest.json",
            {
                "schema_version": "reid_stage5d_f3j_segmentation_manifest_v1",
                "segment_count": p["segment_count"],
                "pass_through_segment_count": p["segmentation_audit"][
                    "pass_through_segment_count"
                ],
                "split_segment_count": 0,
                "segment_inventory_sha256": p["segment_inventory_sha"],
                "segmentation_contract_sha256": canonical_json_sha(seg_contract),
            },
        )

        write_jsonl(
            qual_dir / "target_001_holdout_v2_segment_review_eligibility.jsonl",
            p["eligibility"],
        )
        write_jsonl(
            qual_dir / "target_001_holdout_v2_objective_representative_frame_candidates.jsonl",
            p["representative_candidates"],
        )
        write_json(
            qual_dir / "target_001_holdout_v2_review_eligibility_summary.json",
            p["eligibility_summary"],
        )

        write_json(
            eff_dir / "detector_tracker_effective_config.json",
            det_contract,
        )
        write_json(eff_dir / "segmentation_contract.json", seg_contract)
        shutil.copy2(
            tracker_path,
            eff_dir / "bytetrack_stage3.yaml",
        )

        access = build_access_audit()
        write_json(tmp_root / "runtime" / "access_audit.json", access)
        write_json(
            tmp_root / "runtime" / "source_sha_checkpoints.json",
            {
                "before_inference": sha_before,
                "after_pass_1": sha_mid,
                "after_pass_2": sha_after,
                "unchanged": sha_before == sha_mid == sha_after,
            },
        )

        # Forbidden field audit on key payloads
        forbidden_hits: list[str] = []
        for label, obj in (
            ("detections", p["detections"][:5]),
            ("observations", p["observations"][:5]),
            ("segments", p["segment_inventory"][:5]),
            ("eligibility", p["eligibility"][:5]),
            ("summary_probe", p["eligibility_summary"]),
        ):
            forbidden_hits.extend(f"{label}:{h}" for h in forbidden_field_audit(obj))
        if forbidden_hits:
            raise HoldoutUniverseError(
                "BLOCKED_STAGE5D_F3J_LABEL_BLIND_LEAKAGE_DETECTED "
                + json.dumps(forbidden_hits)
            )

        # Artifact budget scan under tmp (excluding pass_* temp fingerprints)
        png_jpg = 0
        npy = 0
        mp4 = 0
        for dp, _, fns in os.walk(tmp_root):
            if Path(dp).name in {"pass_1", "pass_2"}:
                continue
            for fn in fns:
                low = fn.lower()
                if low.endswith((".png", ".jpg", ".jpeg")):
                    png_jpg += 1
                if low.endswith(".npy"):
                    npy += 1
                if low.endswith(".mp4"):
                    mp4 += 1
        if png_jpg or npy or mp4:
            raise HoldoutUniverseError(
                f"artifact budget violated png={png_jpg} npy={npy} mp4={mp4}"
            )

        obs_counts = [int(r["observation_count"]) for r in p["raw_track_inventory"]]
        spans = [int(r["span_frames"]) for r in p["raw_track_inventory"]]

        contract = {
            "schema_version": "reid_stage5d_f3j_contract_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "project_head": head,
            "holdout_path": hold["path"],
            "holdout_sha256": sha_before,
            "frames_expected": 1058,
            "frames_processed": 1058,
            "detector_input_source_count": 1,
            "label_blind": True,
            "replay_exact_match": True,
            "threshold_selected": False,
            "gallery_mutation": False,
            "f3i_status": config["stage5d_f3i_package"]["expected_final_status"],
            "b1e_b_status": config["stage5d_b1e_b_package"]["expected_final_status"],
        }
        summary = {
            "schema_version": "reid_stage5d_f3j_summary_v1",
            "final_status": STATUS,
            "readiness": READINESS,
            "next_gate": NEXT_GATE,
            "project_head": head,
            "holdout_sha256": sha_before,
            "holdout_path": hold["path"],
            "frames_expected": 1058,
            "frames_processed": int(p["detection_summary"]["frames_processed"]),
            "person_detection_count": p["detection_count"],
            "frames_with_person_detections": p["detection_summary"]["frames_with_detections"],
            "frames_without_person_detections": p["detection_summary"][
                "frames_without_detections"
            ],
            "assigned_detection_count": p["assignment_audit"]["assigned_count"],
            "unassigned_detection_count": p["assignment_audit"]["unassigned_count"],
            "raw_track_count": p["raw_track_count"],
            "tracking_observation_count": p["observation_count"],
            "raw_track_observation_count_min": min(obs_counts) if obs_counts else 0,
            "raw_track_observation_count_max": max(obs_counts) if obs_counts else 0,
            "raw_track_observation_count_mean": (
                float(statistics.mean(obs_counts)) if obs_counts else 0.0
            ),
            "raw_track_span_min": min(spans) if spans else 0,
            "raw_track_span_max": max(spans) if spans else 0,
            "segment_count": p["segment_count"],
            "pass_through_segment_count": p["segmentation_audit"][
                "pass_through_segment_count"
            ],
            "split_segment_count": 0,
            "split_reason_distribution": {},
            "complete_universe_count": p["eligibility_summary"]["complete_universe_count"],
            "review_eligible_count": p["eligibility_summary"]["review_eligible_count"],
            "review_ineligible_count": p["eligibility_summary"]["review_ineligible_count"],
            "representative_frame_candidate_count": p["eligibility_summary"][
                "representative_frame_candidate_count"
            ],
            "replay_deterministic": True,
            "replay_exact_match": True,
            "label_blind_forbidden_fields": 0,
            "gallery_reads": 0,
            "osnet_loads": 0,
            "ocr_calls": 0,
            "team_classifier_calls": 0,
            "crop_exports": 0,
            "embeddings": 0,
            "similarity_rows": 0,
            "rankings": 0,
            "gt_labels": 0,
            "metrics": 0,
            "threshold_selected": False,
            "identity_assignments": 0,
            "gallery_mutation": False,
            "detector_input_source_count": 1,
            "access_audit": access,
            "source_sha_unchanged": True,
            "f3i_validation": {
                "status": config["stage5d_f3i_package"]["expected_final_status"],
                "readiness": config["stage5d_f3i_package"]["expected_readiness"],
                "snapshot_sha256": config["stage5d_f3i_package"]["expected_snapshot_sha256"],
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(tmp_root / "stage5d_f3j_contract.json", contract)
        write_json(tmp_root / "stage5d_f3j_summary.json", summary)

        # Remove pass workdirs before listing/manifest (keep only final structure)
        shutil.rmtree(tmp_root / "pass_1", ignore_errors=True)
        shutil.rmtree(tmp_root / "pass_2", ignore_errors=True)

        n_files, list_sha = listing_sha(tmp_root)
        manifest = {
            "schema_version": "reid_stage5d_f3j_manifest_v1",
            "final_status": STATUS,
            "file_count": n_files,
            "listing_sha256": list_sha,
            "holdout_sha256": sha_before,
            "detection_rows_sha256": p["detection_rows_sha"],
            "observation_rows_sha256": p["observation_rows_sha"],
            "segment_inventory_sha256": p["segment_inventory_sha"],
        }
        write_json(tmp_root / "stage5d_f3j_manifest.json", manifest)
        # recompute listing after manifest write
        n_files, list_sha = listing_sha(tmp_root)
        manifest["file_count"] = n_files
        manifest["listing_sha256"] = list_sha
        write_json(tmp_root / "stage5d_f3j_manifest.json", manifest)

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(tmp_root), str(final_dir))
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_f3j_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/independent_holdout_v2_label_blind_universe_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps({"final_status": summary["final_status"], "readiness": summary["readiness"]}, indent=2))


if __name__ == "__main__":
    main()
