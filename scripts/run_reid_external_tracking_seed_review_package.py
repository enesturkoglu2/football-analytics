#!/usr/bin/env python3
"""Stage 5D-B1E-B — external clip detection, ByteTrack, and seed review package.

YOLO once + ByteTrack replay-twice determinism. No OSNet/OCR/similarity/gallery.
No automatic target selection. Manual fields remain blank.
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

CONFIG_SCHEMA = "reid_external_tracking_seed_review_config_v1"
FROZEN_SEED_CODE = "SEED_CANDIDATE_07"
TEMPLATE_FIELDS = (
    "target_id",
    "external_candidate_code",
    "first_frame",
    "last_frame",
    "representative_frame",
    "manual_occurrence_decision",
    "manual_same_target_as_target_001",
    "manual_human_verified_number_seen",
    "manual_crop_valid",
    "manual_target_dominant",
    "manual_identity_continuity_observed",
    "manual_notes",
    "reviewer",
    "final_approver",
    "reviewed_at",
)
ALLOWED_OCCURRENCE = (
    "target_occurrence_yes",
    "target_occurrence_no",
    "uncertain",
    "invalid",
    "multi_person_ambiguous",
    "non_player",
)
ALLOWED_TRISTATE = ("yes", "no", "uncertain")


class ExternalTrackingError(RuntimeError):
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ExternalTrackingError("unexpected config schema")
    if not config.get("offline_required"):
        raise ExternalTrackingError("offline_required")
    # Canonical detection defaults must match Stage 2/4B.
    det = config["detection"]
    if (
        abs(float(det["conf"]) - DEFAULT_CONF) > 1e-9
        or abs(float(det["iou"]) - DEFAULT_IOU) > 1e-9
        or int(det["imgsz"]) != DEFAULT_IMGSZ
        or list(det["classes"]) != [0]
        or str(det["device"]) != "cpu"
    ):
        raise ExternalTrackingError("canonical detection params mismatch")
    return config


def assert_no_path_traversal(rel: str) -> None:
    if ".." in Path(rel).parts or rel.startswith("/"):
        raise ExternalTrackingError(f"path traversal rejected: {rel}")


def assert_git_contract(project_root: Path, expected_head: str) -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    if head != expected_head:
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_GIT_CONTRACT_MISMATCH HEAD")
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=project_root, text=True
    ).strip()
    if head != origin:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_GIT_CONTRACT_MISMATCH origin"
        )
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    allowed = {
        "scripts/run_reid_external_tracking_seed_review_package.py",
        "configs/reid/external_tracking_seed_review_stage5d_target_001.yaml",
        "tests/test_reid_external_tracking_seed_review_package.py",
        "docs/setup/stage5d-target-external-tracking-and-seed-review-package.md",
    }
    if porcelain:
        for line in porcelain.splitlines():
            path = line[3:] if len(line) > 3 else line
            if path not in allowed:
                raise ExternalTrackingError(
                    "BLOCKED_STAGE5D_B1E_B_GIT_CONTRACT_MISMATCH dirty "
                    + json.dumps(porcelain.splitlines())
                )
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=project_root, text=True
    ).strip()
    if msg != "Add external enrollment clip overlap preflight":
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_GIT_CONTRACT_MISMATCH message"
        )
    # External MP4 must not be tracked.
    tracked = subprocess.check_output(
        ["git", "ls-files", "data/enrollment_clips/target_001_external_enrollment_v1.mp4"],
        cwd=project_root,
        text=True,
    ).strip()
    if tracked:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_GIT_CONTRACT_MISMATCH external_tracked"
        )
    return head


def resolve_snapshot_sha(snapshot_path: Path, expected: str) -> str:
    if not snapshot_path.is_file():
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH snapshot"
        )
    sidecar = Path(str(snapshot_path) + ".sha256")
    manifest = snapshot_path.with_name(
        snapshot_path.name.replace(".tar.gz", "_manifest.json")
    )
    if not sidecar.is_file() or not manifest.is_file():
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH snapshot_sidecar"
        )
    side = sidecar.read_text(encoding="utf-8").split()[0].strip()
    man = str(load_json(manifest).get("archive_sha256") or "")
    actual = sha256_file(snapshot_path)
    if not (side == man == actual == expected):
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH snapshot_sha"
        )
    return actual


def validate_preflight(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root / config["stage5d_b1e_a_package"]["path"]
    summary = load_json(root / "stage5d_b1e_a_summary.json")
    elig = load_json(
        root / "eligibility" / "target_001_external_enrollment_interval_eligibility.json"
    )
    audit = load_json(
        root / "overlap_audit" / "target_001_external_vs_sample_overlap_audit.json"
    )
    exp = config["stage5d_b1e_a_package"]
    if summary.get("final_status") != exp["expected_final_status"]:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH status"
        )
    if summary.get("target_id") != "target_001":
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH target"
        )
    if summary["external_source"]["sha256"] != config["external_enrollment_source"][
        "expected_sha256"
    ]:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH ext_sha"
        )
    if summary.get("exact_file_duplicate") is not False:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH duplicate"
        )
    if int(audit.get("verified_overlapping_pair_count") or 0) != 0:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH pairs"
        )
    if len(audit.get("contiguous_overlapping_intervals") or []) != 0:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH intervals"
        )
    eligible = elig.get("eligible_intervals") or []
    if len(eligible) != 1:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH eligible_count"
        )
    iv = eligible[0]
    if int(iv["start_frame"]) != 0 or int(iv["end_frame"]) != 783:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH eligible_range"
        )
    if abs(float(iv["duration"]) - 26.133333333333333) > 1e-6:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH duration"
        )
    if elig.get("external_source_enrollment_only") is not True:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH enroll_only"
        )
    if elig.get("sample_evaluation_only") is not True:
        raise ExternalTrackingError(
            "BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH eval_only"
        )
    for key in (
        "detection_tracking_rows",
        "crop_embedding_rows",
        "target_selection",
        "derived_anchors",
        "gallery_members",
        "identity_assignments",
    ):
        if int(summary.get(key) or 0) != 0:
            raise ExternalTrackingError(
                f"BLOCKED_STAGE5D_B1E_B_PREFLIGHT_CONTRACT_MISMATCH {key}"
            )
    snap_sha = resolve_snapshot_sha(
        Path(exp["snapshot_path"]), exp["expected_snapshot_sha256"]
    )
    return {"summary": summary, "eligibility": elig, "audit": audit, "snapshot_sha256": snap_sha}


def validate_target_policy(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    td = load_json(project_root / config["target_definition"]["path"])
    b1d = load_json(
        project_root
        / config["stage5d_b1d_package"]["path"]
        / "stage5d_b1d_summary.json"
    )
    handoff = load_json(
        project_root
        / config["stage5d_b1d_package"]["path"]
        / "external_enrollment_handoff"
        / "target_001_external_enrollment_requirements.json"
    )
    if td.get("target_alias") != "sarı takım 5 numaralı oyuncu":
        raise ExternalTrackingError("target alias mismatch")
    if int(handoff.get("human_verified_jersey_number")) != 5:
        raise ExternalTrackingError("jersey mismatch")
    if handoff.get("automated_ocr_used") is not False:
        raise ExternalTrackingError("ocr policy mismatch")
    if handoff.get("automatic_gallery_growth") is not False:
        raise ExternalTrackingError("gallery growth mismatch")
    if handoff.get("unknown_identity_preserved") is not True:
        raise ExternalTrackingError("unknown identity mismatch")
    if b1d.get("original_frozen_seed_code") != FROZEN_SEED_CODE:
        raise ExternalTrackingError("frozen seed mismatch")
    if b1d.get("frozen_seed_enrollment_allowed") is not False:
        raise ExternalTrackingError("frozen seed enroll mismatch")
    return {"target_definition": td, "b1d": b1d, "handoff": handoff}


def validate_external_source(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rel = config["external_enrollment_source"]["path"]
    assert_no_path_traversal(rel)
    path = project_root / rel
    exp = config["external_enrollment_source"]
    if not path.is_file() or path.is_symlink():
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY")
    if path.stat().st_size != int(exp["expected_bytes"]):
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY bytes")
    digest = sha256_file(path)
    if digest != exp["expected_sha256"]:
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY sha")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY open")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width != int(exp["expected_width"]) or height != int(exp["expected_height"]):
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY wh")
    if abs(fps - float(exp["expected_fps"])) > 1e-6:
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY fps")
    if frames != int(exp["expected_frames"]):
        raise ExternalTrackingError("BLOCKED_STAGE5D_B1E_B_EXTERNAL_SOURCE_INTEGRITY frames")
    # Evaluation source immutability (sha only).
    eva = project_root / config["evaluation_source"]["path"]
    if sha256_file(eva) != config["evaluation_source"]["expected_sha256"]:
        raise ExternalTrackingError("evaluation source sha changed")
    yolo = project_root / config["yolo_checkpoint"]["path"]
    if yolo.stat().st_size != int(config["yolo_checkpoint"]["expected_bytes"]):
        raise ExternalTrackingError("yolo bytes mismatch")
    if sha256_file(yolo) != config["yolo_checkpoint"]["expected_sha256"]:
        raise ExternalTrackingError("yolo sha mismatch")
    tracker = project_root / config["tracking"]["tracker_path"]
    if sha256_file(tracker) != config["tracking"]["expected_tracker_sha256"]:
        raise ExternalTrackingError("tracker yaml sha mismatch")
    return {
        "path": rel,
        "absolute_path": str(path),
        "sha256": digest,
        "bytes": int(exp["expected_bytes"]),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "yolo_sha256": config["yolo_checkpoint"]["expected_sha256"],
        "tracker_sha256": config["tracking"]["expected_tracker_sha256"],
    }


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ExternalTrackingError("cannot open external video for detection")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    rows: list[dict[str, Any]] = []
    frames_with = 0
    for fi in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ExternalTrackingError(f"failed reading frame {fi}")
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
        det_index = 0
        frame_hits = 0
        if preds:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for bbox, c in zip(xyxy, confs):
                    cleaned = sanitize_detection(
                        bbox_xyxy=bbox,
                        confidence=float(c),
                        frame_width=width,
                        frame_height=height,
                    )
                    if cleaned is None:
                        continue
                    x1, y1, x2, y2 = cleaned["bbox_xyxy"]
                    rows.append(
                        {
                            "frame_index": fi,
                            "timestamp_sec": fi / fps,
                            "detection_index": det_index,
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": cleaned["confidence"],
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_width": x2 - x1,
                            "bbox_height": y2 - y1,
                            "source_video_sha256": video_sha,
                            "model_sha256": model_sha,
                            "effective_config_sha256": effective_config_sha,
                        }
                    )
                    det_index += 1
                    frame_hits += 1
        if frame_hits:
            frames_with += 1
    cap.release()
    summary = {
        "schema_version": "reid_stage5d_b1e_b_detection_summary_v1",
        "frames_processed": end - start + 1,
        "frame_range": [start, end],
        "total_detections": len(rows),
        "frames_with_detections": frames_with,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "classes": [0],
        "device": "cpu",
        "source_video_sha256": video_sha,
        "model_sha256": model_sha,
        "effective_config_sha256": effective_config_sha,
        "yolo_runs": 1,
    }
    return rows, summary


def detections_by_frame(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_f: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_f[int(row["frame_index"])].append(dict(row))
    for fi in by_f:
        by_f[fi].sort(key=lambda r: int(r["detection_index"]))
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
) -> list[dict[str, Any]]:
    BYTETracker.reset_id()
    tracker = BYTETracker(tracker_args)
    observations: list[dict[str, Any]] = []
    for fi in range(start, end + 1):
        dets = list(by_frame.get(fi, []))
        boxes = boxes_from_frame_dets(dets, width=width, height=height)
        out = tracker.update(boxes)
        if out is None or len(out) == 0:
            continue
        for row in np.asarray(out):
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            track_id = int(row[4])
            conf = float(row[5])
            det_idx = int(row[7]) if len(row) > 7 else -1
            lineage = None
            if 0 <= det_idx < len(dets):
                lineage = {
                    "detection_index": int(dets[det_idx]["detection_index"]),
                    "detection_confidence": float(dets[det_idx]["confidence"]),
                }
            observations.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": fi / fps,
                    "raw_track_id": track_id,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "detection_lineage": lineage,
                    "source_video_sha256": video_sha,
                    "tracking_effective_config_sha256": tracking_config_sha,
                }
            )
    return observations


def observations_fingerprint(obs: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "frame_index": o["frame_index"],
            "raw_track_id": o["raw_track_id"],
            "bbox_xyxy": [round(float(v), 4) for v in o["bbox_xyxy"]],
            "confidence": round(float(o["confidence"]), 6),
        }
        for o in obs
    ]
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def assign_ext_codes(
    tracks: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[int, str]:
    def sort_key(tid: int) -> tuple:
        rows = tracks[tid]
        first = min(rows, key=lambda r: r["frame_index"])
        x1, _, x2, _ = first["bbox_xyxy"]
        x_center = (float(x1) + float(x2)) / 2.0
        return (int(first["frame_index"]), x_center, int(tid))

    ordered = sorted(tracks.keys(), key=sort_key)
    return {tid: f"EXT_{i:03d}" for i, tid in enumerate(ordered, start=1)}


def track_quality(
    rows: Sequence[Mapping[str, Any]], *, width: int, height: int
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(rows) < 3:
        reasons.append("observation_count_below_min")
    valid = 0
    for r in rows:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
            valid += 1
    if valid == 0:
        reasons.append("no_valid_in_frame_bbox")
    return {
        "observation_count": len(rows),
        "valid_bbox_count": valid,
        "review_eligible": len(reasons) == 0,
        "exclusion_reasons": reasons,
    }


def choose_representative(
    rows: Sequence[Mapping[str, Any]], *, width: int, height: int
) -> Mapping[str, Any]:
    frames = [int(r["frame_index"]) for r in rows]
    mid = (min(frames) + max(frames)) / 2.0
    best = None
    best_score = -1e18
    for r in rows:
        x1, y1, x2, y2 = [float(v) for v in r["bbox_xyxy"]]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        edge = min(x1, y1, width - x2, height - y2)
        trunc = max(0.0, -edge)
        mid_dist = abs(int(r["frame_index"]) - mid)
        score = area - 50.0 * trunc - 5.0 * mid_dist + 10.0 * min(w, h)
        if score > best_score:
            best_score = score
            best = r
    assert best is not None
    return best


def draw_readable_label(
    frame: np.ndarray,
    *,
    text: str,
    x: int,
    y: int,
    min_px: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min_px / 30.0)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x0 = max(0, x)
    y0 = max(th + pad + 2, y)
    # Prefer above box; if overflow, place inside.
    top = y0 - th - pad - 2
    if top < 0:
        top = min(frame.shape[0] - th - pad - 2, y + pad)
        y0 = top + th + pad
    cv2.rectangle(
        frame,
        (x0, top),
        (min(frame.shape[1] - 1, x0 + tw + 2 * pad), min(frame.shape[0] - 1, y0 + baseline)),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x0 + pad, y0 - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame_ext(
    frame: np.ndarray,
    *,
    items: Sequence[Mapping[str, Any]],
    color: Sequence[int],
    thickness: int,
    min_px: int,
) -> np.ndarray:
    out = frame.copy()
    # Deterministic label offsets for overlaps: sort by code then nudge.
    ordered = sorted(items, key=lambda x: str(x["external_candidate_code"]))
    used_tops: list[tuple[int, int, int]] = []
    for item in ordered:
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(c) for c in color), int(thickness))
        label_y = y1
        # Nudge if colliding with previous labels near same x.
        for ux, uy, uh in used_tops:
            if abs(ux - x1) < 80 and abs(uy - (y1 - 20)) < uh:
                label_y = y1 + uh
        draw_readable_label(
            out, text=str(item["external_candidate_code"]), x=x1, y=label_y, min_px=min_px
        )
        used_tops.append((x1, label_y - 20, max(28, min_px)))
    return out


def write_review_mp4(
    path: Path,
    *,
    video_path: Path,
    start: int,
    end: int,
    frame_items: Mapping[int, Sequence[Mapping[str, Any]]],
    fps: float,
    color: Sequence[int],
    thickness: int,
    min_px: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ExternalTrackingError("cannot open video for review mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise ExternalTrackingError("FAILED review mp4 writer")
    codes_seen: set[str] = set()
    frames_with = 0
    count = 0
    for fi in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ExternalTrackingError(f"failed read {fi} for mp4")
        items = list(frame_items.get(fi, []))
        if items:
            frames_with += 1
            for it in items:
                codes_seen.add(str(it["external_candidate_code"]))
        annotated = annotate_frame_ext(
            frame, items=items, color=color, thickness=thickness, min_px=min_px
        )
        writer.write(annotated)
        count += 1
    writer.release()
    cap.release()
    return {
        "frame_count": count,
        "fps": fps,
        "duration_sec": count / fps,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frame_coverage": {
            "start": start,
            "end": end,
            "frames_with_boxes": frames_with,
        },
        "candidate_code_coverage": sorted(codes_seen),
        "width": w,
        "height": h,
    }


def _fit(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def render_panel_sheet(
    panels: Sequence[Mapping[str, Any]], *, cols: int, panel_w: int, panel_h: int
) -> np.ndarray:
    rows_n = int(math.ceil(len(panels) / cols)) if panels else 1
    sheet = np.full((rows_n * panel_h, cols * panel_w, 3), 16, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        tile = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
        cv2.putText(
            tile,
            str(panel.get("header", "")),
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        disp = _fit(panel["image"], panel_w - 10, panel_h - 36)
        dh, dw = disp.shape[:2]
        ox = (panel_w - dw) // 2
        oy = 28
        tile[oy : oy + dh, ox : ox + dw] = disp
        y0 = r * panel_h
        x0 = c * panel_w
        sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = tile
    return sheet


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "reid_stage5d_b1e_b_contract_v1",
        "target_id": "target_001",
        "external_enrollment_only": True,
        "sample_evaluation_only": True,
        "yolo_detection_once": True,
        "bytetrack_from_saved_detections": True,
        "two_replay_determinism_required": True,
        "no_osnet": True,
        "no_ocr": True,
        "no_similarity": True,
        "no_reid_assisted_tracking": True,
        "no_automatic_target_selection": True,
        "no_team_or_jersey_filtering": True,
        "no_bbox_interpolation": True,
        "manual_selections": 0,
        "approved_target_tracklets": 0,
        "embeddings": 0,
        "anchors": 0,
        "gallery_members": 0,
        "prototypes": 0,
        "identity_assignments": 0,
        "multiple_positive_occurrences_allowed_later": True,
        "exact_next_gate": "STAGE5D-B1E-C_TARGET_001_EXTERNAL_SEED_MANUAL_REVIEW_AND_FREEZE",
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    head = assert_git_contract(project_root, config["project_head_expected"])
    final_rel = config["output"]["final_dir"]
    assert_no_path_traversal(final_rel)
    final_dir = project_root / final_rel
    if final_dir.exists():
        raise ExternalTrackingError("FAILED_STAGE5D_B1E_B_ATOMIC_OUTPUT final_exists")

    preflight = validate_preflight(project_root, config)
    policy = validate_target_policy(project_root, config)
    source = validate_external_source(project_root, config)

    start = int(config["eligible_interval"]["start_frame"])
    end = int(config["eligible_interval"]["end_frame"])
    if start != 0 or end != 783:
        raise ExternalTrackingError("eligible interval must be 0-783")

    det_cfg = {
        "conf": float(config["detection"]["conf"]),
        "iou": float(config["detection"]["iou"]),
        "imgsz": int(config["detection"]["imgsz"]),
        "classes": [0],
        "device": "cpu",
        "frame_range": [start, end],
    }
    det_cfg_sha = hashlib.sha256(
        json.dumps(det_cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tracker_path = project_root / config["tracking"]["tracker_path"]
    tracker_args = load_tracker_args(tracker_path)
    track_cfg_sha = sha256_file(tracker_path)

    tmp = project_root / (
        "outputs/reid/"
        "_tmp_full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package_"
        f"{uuid.uuid4().hex}"
    )
    if tmp.exists():
        raise ExternalTrackingError("FAILED_STAGE5D_B1E_B_ATOMIC_OUTPUT tmp_exists")
    tmp.mkdir(parents=True)

    video_path = Path(source["absolute_path"])
    try:
        det_rows, det_summary = run_yolo_detection(
            video_path=video_path,
            model_path=project_root / config["yolo_checkpoint"]["path"],
            start=start,
            end=end,
            conf=det_cfg["conf"],
            iou=det_cfg["iou"],
            imgsz=det_cfg["imgsz"],
            video_sha=source["sha256"],
            model_sha=source["yolo_sha256"],
            effective_config_sha=det_cfg_sha,
        )
        det_dir = tmp / "detection"
        det_dir.mkdir(parents=True)
        with (det_dir / "detections.jsonl").open("w", encoding="utf-8") as handle:
            for row in det_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(det_dir / "detection_summary.json", det_summary)
        write_json(det_dir / "detection_effective_config.json", det_cfg)

        by_frame = detections_by_frame(det_rows)
        replay_kwargs = dict(
            by_frame=by_frame,
            start=start,
            end=end,
            width=int(source["width"]),
            height=int(source["height"]),
            fps=float(source["fps"]),
            tracker_args=tracker_args,
            video_sha=source["sha256"],
            tracking_config_sha=track_cfg_sha,
        )
        obs_a = replay_bytetrack(**replay_kwargs)
        obs_b = replay_bytetrack(**replay_kwargs)
        fp_a = observations_fingerprint(obs_a)
        fp_b = observations_fingerprint(obs_b)
        if fp_a != fp_b or len(obs_a) != len(obs_b):
            raise ExternalTrackingError(
                "BLOCKED_STAGE5D_B1E_B_TRACKING_NONDETERMINISTIC"
            )
        # Exact pairwise equality.
        for a, b in zip(obs_a, obs_b):
            if (
                a["frame_index"] != b["frame_index"]
                or a["raw_track_id"] != b["raw_track_id"]
                or any(
                    abs(float(x) - float(y)) > 1e-6
                    for x, y in zip(a["bbox_xyxy"], b["bbox_xyxy"])
                )
            ):
                raise ExternalTrackingError(
                    "BLOCKED_STAGE5D_B1E_B_TRACKING_NONDETERMINISTIC mapping"
                )

        track_dir = tmp / "tracking"
        track_dir.mkdir(parents=True)
        with (track_dir / "tracks.jsonl").open("w", encoding="utf-8") as handle:
            for row in obs_a:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            track_dir / "tracking_summary.json",
            {
                "schema_version": "reid_stage5d_b1e_b_tracking_summary_v1",
                "total_observations": len(obs_a),
                "unique_raw_tracks": len({o["raw_track_id"] for o in obs_a}),
                "frame_range": [start, end],
                "two_replay_determinism": True,
                "replay_fingerprint_sha256": fp_a,
                "tracker_path": config["tracking"]["tracker_path"],
                "tracker_sha256": track_cfg_sha,
                "source_video_sha256": source["sha256"],
                "yolo_inference_in_tracking": False,
                "bytetrack_from_saved_detections": True,
            },
        )
        write_json(
            track_dir / "tracking_effective_config.json",
            {
                "tracker_yaml": config["tracking"]["tracker_path"],
                "tracker_sha256": track_cfg_sha,
                "parameters": dict(yaml.safe_load(tracker_path.read_text(encoding="utf-8"))),
            },
        )

        tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in obs_a:
            tracks[int(row["raw_track_id"])].append(row)
        code_map = assign_ext_codes(tracks)

        mapping_rows: list[dict[str, Any]] = []
        review_codes: dict[int, str] = {}
        for tid, code in sorted(code_map.items(), key=lambda kv: kv[1]):
            rows = sorted(tracks[tid], key=lambda r: r["frame_index"])
            quality = track_quality(
                rows, width=int(source["width"]), height=int(source["height"])
            )
            rep = choose_representative(
                rows, width=int(source["width"]), height=int(source["height"])
            )
            mapping_rows.append(
                {
                    "external_candidate_code": code,
                    "raw_external_track_id": tid,
                    "first_frame": rows[0]["frame_index"],
                    "last_frame": rows[-1]["frame_index"],
                    "observation_count": len(rows),
                    "observation_frames": [r["frame_index"] for r in rows],
                    "bbox_per_observation": [
                        {"frame_index": r["frame_index"], "bbox_xyxy": r["bbox_xyxy"]}
                        for r in rows
                    ],
                    "detection_lineage": [r.get("detection_lineage") for r in rows],
                    "representative_frame": rep["frame_index"],
                    "representative_bbox": rep["bbox_xyxy"],
                    "quality_diagnostics": quality,
                    "review_eligible": quality["review_eligible"],
                    "source_video_sha256": source["sha256"],
                    "manual_fields_blank": True,
                    "manual_occurrence_decision": "",
                    "manual_same_target_as_target_001": "",
                }
            )
            if quality["review_eligible"]:
                review_codes[tid] = code

        inv = tmp / "inventory"
        inv.mkdir(parents=True)
        with (inv / "target_001_external_track_candidate_mapping.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in mapping_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Frame items for review visuals: only review-eligible tracks.
        frame_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for tid, code in review_codes.items():
            for r in tracks[tid]:
                frame_items[int(r["frame_index"])].append(
                    {
                        "external_candidate_code": code,
                        "bbox_xyxy": r["bbox_xyxy"],
                    }
                )
        for fi in frame_items:
            frame_items[fi].sort(key=lambda x: x["external_candidate_code"])

        color = list(config["visualization"]["bbox_color_bgr"])
        thickness = int(config["visualization"]["bbox_thickness"])
        min_px = int(config["visualization"]["label_min_px"])
        fps = float(source["fps"])

        pkg = tmp / "review_packages" / "target_001_external_seed_review"
        pkg.mkdir(parents=True)
        clip_meta = write_review_mp4(
            pkg / "target_001_external_tracking_seed_review.mp4",
            video_path=video_path,
            start=start,
            end=end,
            frame_items=frame_items,
            fps=fps,
            color=color,
            thickness=thickness,
            min_px=min_px,
        )
        write_json(pkg / "target_001_external_tracking_seed_review_mp4_manifest.json", clip_meta)

        # Temporal overview sheets.
        stride = int(config["visualization"]["temporal_stride"])
        max_panels = int(config["visualization"]["max_panels_per_temporal_sheet"])
        temporal_frames = list(range(start, end + 1, stride))
        if end not in temporal_frames:
            # Gate: sample every 1s; last exact 783 optional — include if not already.
            pass
        # Exact: 0,30,...,780 => 27 panels (783 not required if stride lands on 780).
        cap = cv2.VideoCapture(str(video_path))
        temporal_panels = []
        for fi in temporal_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ExternalTrackingError(f"temporal sheet read fail {fi}")
            annotated = annotate_frame_ext(
                frame,
                items=frame_items.get(fi, []),
                color=color,
                thickness=thickness,
                min_px=min_px,
            )
            temporal_panels.append(
                {
                    "header": f"f={fi} t={fi/fps:.2f}s",
                    "image": annotated,
                    "frame_index": fi,
                }
            )
        cap.release()
        sheet_paths = []
        for sheet_i, offset in enumerate(range(0, len(temporal_panels), max_panels), start=1):
            chunk = temporal_panels[offset : offset + max_panels]
            name = f"temporal_overview_sheet_{sheet_i:02d}.png"
            out = pkg / name
            if not cv2.imwrite(
                str(out),
                render_panel_sheet(chunk, cols=min(4, len(chunk) or 1), panel_w=360, panel_h=240),
            ):
                raise ExternalTrackingError(f"failed write {name}")
            sheet_paths.append(
                f"review_packages/target_001_external_seed_review/{name}"
            )
        if len(sheet_paths) != 4:
            raise ExternalTrackingError(
                f"expected 4 temporal sheets, got {len(sheet_paths)}"
            )

        # Candidate index sheets.
        idx_dir = pkg / "candidate_index"
        idx_dir.mkdir(parents=True)
        eligible_map = [m for m in mapping_rows if m["review_eligible"]]
        max_cand = int(config["visualization"]["max_candidates_per_index_sheet"])
        cap = cv2.VideoCapture(str(video_path))
        index_panels = []
        for m in eligible_map:
            fi = int(m["representative_frame"])
            bbox = m["representative_bbox"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ExternalTrackingError(f"index crop read fail {fi}")
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(frame.shape[1] - 1, x1))
            x2 = max(0, min(frame.shape[1], x2))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            y2 = max(0, min(frame.shape[0], y2))
            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame.copy()
            index_panels.append(
                {
                    "header": (
                        f"{m['external_candidate_code']} f={fi} "
                        f"t={fi/fps:.2f}s n={m['observation_count']}"
                    ),
                    "image": crop,
                }
            )
        cap.release()
        index_paths = []
        for sheet_i, offset in enumerate(range(0, len(index_panels), max_cand), start=1):
            chunk = index_panels[offset : offset + max_cand]
            name = f"candidate_index_sheet_{sheet_i:02d}.png"
            out = idx_dir / name
            if not cv2.imwrite(
                str(out),
                render_panel_sheet(chunk, cols=4, panel_w=220, panel_h=260),
            ):
                raise ExternalTrackingError(f"failed write {name}")
            index_paths.append(
                "review_packages/target_001_external_seed_review/"
                f"candidate_index/{name}"
            )

        # Blank template.
        tpl_dir = tmp / "templates"
        tpl_dir.mkdir(parents=True)
        with (
            tpl_dir / "target_001_external_seed_manual_review_template.csv"
        ).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_FIELDS))
            writer.writeheader()
            # One blank row per EXT code (eligible and excluded); no prefilled decisions.
            for m in mapping_rows:
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "external_candidate_code": m["external_candidate_code"],
                        "first_frame": m["first_frame"],
                        "last_frame": m["last_frame"],
                        "representative_frame": m["representative_frame"],
                        "manual_occurrence_decision": "",
                        "manual_same_target_as_target_001": "",
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
                "schema_version": "reid_stage5d_b1e_b_runtime_v1",
                "started_at": started,
                "offline_required": True,
                "network_download": 0,
                "yolo_runs": 1,
                "bytetrack_replays": 2,
                "osnet_inference": 0,
                "ocr": 0,
                "similarity_inference": 0,
                "manual_selections": 0,
            },
        )
        eff = tmp / "effective_configs"
        eff.mkdir(parents=True)
        shutil.copy2(config_path, eff / config_path.name)
        shutil.copy2(tracker_path, eff / tracker_path.name)

        write_json(tmp / "stage5d_b1e_b_contract.json", build_contract())

        png_count = len(list(tmp.rglob("*.png")))
        mp4_count = len(list(tmp.rglob("*.mp4")))
        if mp4_count != 1:
            raise ExternalTrackingError(
                f"FAILED_STAGE5D_B1E_B_ATOMIC_OUTPUT mp4_count={mp4_count}"
            )
        if png_count != 4 + len(index_paths):
            raise ExternalTrackingError(
                f"FAILED_STAGE5D_B1E_B_ATOMIC_OUTPUT png_count={png_count}"
            )

        final_status = "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY"
        summary = {
            "schema_version": "reid_stage5d_b1e_b_summary_v1",
            "final_status": final_status,
            "project_head": head,
            "target_id": "target_001",
            "target_alias": policy["target_definition"]["target_alias"],
            "human_verified_jersey_number": 5,
            "original_frozen_seed_code": FROZEN_SEED_CODE,
            "frozen_seed_used_for_tracking_or_ranking": False,
            "external_source": source,
            "eligible_interval": [start, end],
            "detection_total": len(det_rows),
            "detection_frames_with_boxes": det_summary["frames_with_detections"],
            "detection_effective_config_sha256": det_cfg_sha,
            "tracking_total_observations": len(obs_a),
            "raw_track_count": len(tracks),
            "review_eligible_candidate_count": len(eligible_map),
            "ext_candidate_count": len(code_map),
            "two_replay_determinism": True,
            "tracking_replay_fingerprint_sha256": fp_a,
            "temporal_overview_png_count": 4,
            "candidate_index_png_count": len(index_paths),
            "annotated_mp4_count": 1,
            "manual_selections": 0,
            "approved_target_tracklets": 0,
            "embeddings": 0,
            "osnet_inference": 0,
            "ocr": 0,
            "similarity_ranking_rows": 0,
            "derived_anchors": 0,
            "gallery_members": 0,
            "prototypes": 0,
            "identity_assignments": 0,
            "b1e_a_snapshot_sha256": preflight["snapshot_sha256"],
            "exact_next_gate": (
                "STAGE5D-B1E-C_TARGET_001_EXTERNAL_SEED_MANUAL_REVIEW_AND_FREEZE"
            ),
        }
        write_json(tmp / "stage5d_b1e_b_summary.json", summary)

        n_files, listing = listing_sha(tmp)
        write_json(
            tmp / "stage5d_b1e_b_manifest.json",
            {
                "schema_version": "reid_stage5d_b1e_b_manifest_v1",
                "final_status": final_status,
                "project_head": head,
                "listing_file_count": n_files,
                "listing_sha256": listing,
                "mp4_sha256": clip_meta["sha256"],
                "temporal_sheets": sheet_paths,
                "candidate_index_sheets": index_paths,
                "ext_candidate_count": len(code_map),
                "review_eligible_candidate_count": len(eligible_map),
                "gallery_members": 0,
                "manual_selections": 0,
            },
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, final_dir)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    return load_json(final_dir / "stage5d_b1e_b_summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "configs/reid/external_tracking_seed_review_stage5d_target_001.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = run(args.config.resolve(), args.project_root.resolve())
    except ExternalTrackingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_status": summary["final_status"],
                "detections": summary["detection_total"],
                "tracks": summary["raw_track_count"],
                "ext_codes": summary["ext_candidate_count"],
                "review_eligible": summary["review_eligible_candidate_count"],
                "next": summary["exact_next_gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
