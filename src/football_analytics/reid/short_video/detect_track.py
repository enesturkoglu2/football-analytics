"""Validated YOLO person detection + ByteTrack replay (B1E-B contract, no new models)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from football_analytics.detection.annotate import sanitize_detection
from football_analytics.detection.pipeline import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    load_yolo_model,
)
from football_analytics.ingest.checksum import sha256_file

YOLO_EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
YOLO_EXPECTED_BYTES = 5613764
TRACKER_EXPECTED_SHA256 = "b951014a9ef48b14eb4c13003d2b83c579b260abc9079156c986818898c8549b"
DEFAULT_TRACKER_REL = "configs/tracking/bytetrack_stage3.yaml"
DEFAULT_YOLO_REL = "models/yolo11n.pt"


class ShortVideoDetectTrackError(RuntimeError):
    pass


def assert_frozen_checkpoints(project_root: Path) -> dict[str, str]:
    yolo = project_root / DEFAULT_YOLO_REL
    tracker = project_root / DEFAULT_TRACKER_REL
    if not yolo.is_file():
        raise ShortVideoDetectTrackError(f"missing yolo checkpoint: {yolo}")
    if yolo.stat().st_size != YOLO_EXPECTED_BYTES:
        raise ShortVideoDetectTrackError("yolo bytes mismatch")
    yolo_sha = sha256_file(yolo)
    if yolo_sha != YOLO_EXPECTED_SHA256:
        raise ShortVideoDetectTrackError("yolo sha mismatch")
    tracker_sha = sha256_file(tracker)
    if tracker_sha != TRACKER_EXPECTED_SHA256:
        raise ShortVideoDetectTrackError("tracker yaml sha mismatch")
    return {"yolo_sha256": yolo_sha, "tracker_sha256": tracker_sha}


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
    video_sha: str,
    model_sha: str,
    effective_config_sha: str,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ShortVideoDetectTrackError("cannot open video for detection")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    rows: list[dict[str, Any]] = []
    frames_with = 0
    dropped = 0
    for fi in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            dropped += 1
            continue
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
                    detection_id = f"det_{fi:06d}_{det_index:03d}"
                    rows.append(
                        {
                            "detection_id": detection_id,
                            "frame_index": fi,
                            "timestamp_sec": fi / fps if fps else 0.0,
                            "detection_index": det_index,
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": cleaned["confidence"],
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_width": x2 - x1,
                            "bbox_height": y2 - y1,
                            "source_video_sha256": video_sha,
                            "model_sha256": model_sha,
                            "model_path": str(model_path),
                            "effective_config_sha256": effective_config_sha,
                        }
                    )
                    det_index += 1
                    frame_hits += 1
        if frame_hits:
            frames_with += 1
    cap.release()
    summary = {
        "schema_version": "short_video_detection_summary_v1",
        "frames_processed": end - start + 1 - dropped,
        "frames_requested": end - start + 1,
        "dropped_frame_count": dropped,
        "decode_failures": dropped,
        "frame_range": [start, end],
        "total_detections": len(rows),
        "player_detection_count": len(rows),
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
        "referee_spectator_auto_hide": False,
        "note": "COCO person class only; no separate referee/spectator filter evidence applied",
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
            detection_id = None
            if 0 <= det_idx < len(dets):
                lineage = {
                    "detection_index": int(dets[det_idx]["detection_index"]),
                    "detection_confidence": float(dets[det_idx]["confidence"]),
                    "detection_id": dets[det_idx].get("detection_id"),
                }
                detection_id = dets[det_idx].get("detection_id")
            observations.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": fi / fps if fps else 0.0,
                    "raw_track_id": track_id,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "detection_id": detection_id,
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


def assign_ext_codes(tracks: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[int, str]:
    def sort_key(tid: int) -> tuple:
        rows = tracks[tid]
        first = min(rows, key=lambda r: r["frame_index"])
        x1, _, x2, _ = first["bbox_xyxy"]
        x_center = (float(x1) + float(x2)) / 2.0
        return (int(first["frame_index"]), x_center, int(tid))

    ordered = sorted(tracks.keys(), key=sort_key)
    return {tid: f"EXT_{i:03d}" for i, tid in enumerate(ordered, start=1)}


def track_quality(
    rows: Sequence[Mapping[str, Any]], *, width: int, height: int, min_obs: int = 3
) -> dict[str, Any]:
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


def build_mapping_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    video_sha: str,
    min_obs: int = 3,
) -> list[dict[str, Any]]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        tracks[int(row["raw_track_id"])].append(dict(row))
    code_map = assign_ext_codes(tracks)
    mapping_rows: list[dict[str, Any]] = []
    for tid, code in sorted(code_map.items(), key=lambda kv: kv[1]):
        rows = sorted(tracks[tid], key=lambda r: r["frame_index"])
        quality = track_quality(rows, width=width, height=height, min_obs=min_obs)
        frames = [int(r["frame_index"]) for r in rows]
        coverage = len(set(frames)) / max(1, (max(frames) - min(frames) + 1))
        rep = choose_representative(rows, width=width, height=height)
        mapping_rows.append(
            {
                "external_candidate_code": code,
                "raw_external_track_id": tid,
                "raw_track_id": tid,
                "segment_id": f"EXT_SEG_{code.replace('EXT_', '')}",
                "first_frame": rows[0]["frame_index"],
                "last_frame": rows[-1]["frame_index"],
                "start_frame": rows[0]["frame_index"],
                "end_frame": rows[-1]["frame_index"],
                "duration_frames": int(rows[-1]["frame_index"]) - int(rows[0]["frame_index"]) + 1,
                "observation_count": len(rows),
                "observation_coverage": coverage,
                "observation_frames": frames,
                "bbox_per_observation": [
                    {
                        "frame_index": r["frame_index"],
                        "bbox_xyxy": r["bbox_xyxy"],
                        "detection_id": r.get("detection_id"),
                        "confidence": r.get("confidence"),
                    }
                    for r in rows
                ],
                "detection_lineage": [r.get("detection_lineage") for r in rows],
                "representative_frame": rep["frame_index"],
                "representative_bbox": rep["bbox_xyxy"],
                "quality_diagnostics": quality,
                "review_eligible": quality["review_eligible"],
                "track_state": "terminated",
                "termination_reason": "tracker_lost_or_ended",
                "source_video_sha256": video_sha,
                "team_metadata": {"team_label": "unknown", "is_identity_proof": False},
            }
        )
    return mapping_rows


def effective_detection_config_sha(
    *, conf: float, iou: float, imgsz: int, model_sha: str
) -> str:
    blob = json.dumps(
        {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "classes": [0],
            "device": "cpu",
            "model_sha256": model_sha,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()
