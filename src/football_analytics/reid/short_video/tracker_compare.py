"""Comparative tracker replay on frozen detections (no YOLO rerun, no ReID confirm)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from football_analytics.reid.short_video.detect_track import (
    build_mapping_rows,
    detections_by_frame,
    observations_fingerprint,
)
from football_analytics.reid.short_video.tracking_audit import continuity_probe


def _ns(params: Mapping[str, Any]) -> IterableSimpleNamespace:
    return IterableSimpleNamespace(**dict(params))


def boxes_from_dets(
    dets: Sequence[Mapping[str, Any]], *, width: int, height: int
) -> Boxes:
    if not dets:
        return Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(height, width))
    data = [[*d["bbox_xyxy"], float(d["confidence"]), 0.0] for d in dets]
    return Boxes(np.asarray(data, dtype=np.float32), orig_shape=(height, width))


def replay_tracker(
    *,
    tracker_kind: str,
    params: Mapping[str, Any],
    by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    start: int,
    end: int,
    width: int,
    height: int,
    fps: float,
    video_path: Path | None = None,
    video_sha: str = "",
    tracking_config_sha: str = "",
) -> list[dict[str, Any]]:
    """Replay ByteTrack or BoT-SORT (with_reid=False) from saved detections."""
    args = _ns(params)
    if tracker_kind == "bytetrack":
        BYTETracker.reset_id()
        tracker = BYTETracker(args)
        need_img = False
    elif tracker_kind == "botsort":
        from ultralytics.trackers.bot_sort import BOTSORT

        BYTETracker.reset_id()
        # BOTSORT inherits reset_id usage via BYTETracker
        tracker = BOTSORT(args)
        need_img = str(params.get("gmc_method") or "none") != "none"
    else:
        raise ValueError(f"unknown tracker_kind: {tracker_kind}")

    cap = None
    if need_img:
        if video_path is None:
            raise RuntimeError("video_path required for GMC tracker")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video for GMC: {video_path}")

    observations: list[dict[str, Any]] = []
    try:
        for fi in range(start, end + 1):
            dets = list(by_frame.get(fi, []))
            boxes = boxes_from_dets(dets, width=width, height=height)
            img = None
            if need_img and cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                ok, frame = cap.read()
                if ok and frame is not None:
                    img = frame
            out = tracker.update(boxes, img=img)
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
    finally:
        if cap is not None:
            cap.release()
    return observations


def summarize_variant(
    observations: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    width: int,
    height: int,
    video_sha: str,
    seed_bbox: Sequence[float] | None,
    seed_frame: int | None,
    runtime_sec: float,
    fingerprint: str,
    fingerprint_b: str | None,
) -> dict[str, Any]:
    by_track: dict[int, list] = defaultdict(list)
    for row in observations:
        by_track[int(row["raw_track_id"])].append(row)
    durs = []
    for rows in by_track.values():
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        durs.append((int(rows[-1]["frame_index"]) - int(rows[0]["frame_index"]) + 1) / fps)
    durs_a = np.asarray(durs, dtype=float) if durs else np.asarray([0.0])
    mapping = build_mapping_rows(
        observations, width=width, height=height, video_sha=video_sha
    )
    eligible = sum(1 for r in mapping if r.get("review_eligible"))
    probe = None
    if seed_bbox is not None and seed_frame is not None:
        probe = continuity_probe(
            observations, seed_bbox=seed_bbox, seed_frame=seed_frame, fps=fps
        )
    return {
        "raw_track_count": len(by_track),
        "eligible_raw_tracks": eligible,
        "median_track_duration_sec": float(np.median(durs_a)),
        "mean_track_duration_sec": float(np.mean(durs_a)),
        "short_tracks_lt_1s": int((durs_a < 1.0).sum()),
        "short_tracks_lt_0_5s": int((durs_a < 0.5).sum()),
        "fragmentation_index": float(len(by_track) / max(1e-9, float(np.sum(durs_a)))),
        "runtime_sec": runtime_sec,
        "determinism_ok": fingerprint_b is None or fingerprint == fingerprint_b,
        "fingerprint_sha256": fingerprint,
        "continuity_probe": probe,
        "mapping_rows": mapping,
        "observations": list(observations),
    }


BASELINE_PARAMS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def bounded_variants() -> list[dict[str, Any]]:
    """Evidence-driven small sweep (not random)."""
    return [
        {
            "variant_id": "A_current_bytetrack",
            "tracker_kind": "bytetrack",
            "params": dict(BASELINE_PARAMS),
            "rationale": "Frozen stage3 ByteTrack contract",
        },
        {
            "variant_id": "B1_buffer60_match07_new035",
            "tracker_kind": "bytetrack",
            "params": {
                **BASELINE_PARAMS,
                "track_buffer": 60,
                "match_thresh": 0.7,
                "new_track_thresh": 0.35,
                "track_high_thresh": 0.28,
            },
            "rationale": "Longer occlusion buffer + slightly easier match + fewer spurious births",
        },
        {
            "variant_id": "B2_buffer90_match065_new04",
            "tracker_kind": "bytetrack",
            "params": {
                **BASELINE_PARAMS,
                "track_buffer": 90,
                "match_thresh": 0.65,
                "new_track_thresh": 0.4,
                "track_high_thresh": 0.3,
                "track_low_thresh": 0.1,
            },
            "rationale": "Stronger occlusion hold; stricter new-track gate for crowded football",
        },
        {
            "variant_id": "B3_buffer45_match075_new03",
            "tracker_kind": "bytetrack",
            "params": {
                **BASELINE_PARAMS,
                "track_buffer": 45,
                "match_thresh": 0.75,
                "new_track_thresh": 0.3,
            },
            "rationale": "Conservative mid-point between baseline and aggressive buffer",
        },
        {
            "variant_id": "C_botsort_gmc_sparseOptFlow",
            "tracker_kind": "botsort",
            "params": {
                "tracker_type": "botsort",
                "track_high_thresh": 0.28,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.35,
                "track_buffer": 60,
                "match_thresh": 0.7,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,
                "model": "auto",
            },
            "rationale": "Camera-motion compensation via GMC; appearance ReID disabled (no auto identity)",
        },
    ]


def config_sha(params: Mapping[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def write_yaml(path: Path, params: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(params), handle, sort_keys=False)


def select_best(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Safety-first ranking: no identity risk → continuity → fragmentation → runtime."""
    scored = []
    for row in results:
        probe = row.get("continuity_probe") or {}
        det_ok = bool(row.get("determinism_ok", False))
        # Without GT, treat spatial-jump absence + determinism as safety proxy;
        # never reward merging via raw-count alone.
        continuity = float(probe.get("uninterrupted_duration_sec") or 0.0)
        frag = float(row.get("fragmentation_index") or 1e9)
        short = int(row.get("short_tracks_lt_1s") or 0)
        runtime = float(row.get("runtime_sec") or 1e9)
        # Hard reject non-deterministic
        if not det_ok:
            score = (-1e18, 0, 0, 0, 0)
        else:
            score = (
                1.0,  # placeholder identity-safety OK (no forced merges)
                continuity,
                -short,
                -frag,
                -runtime,
            )
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    winner = scored[0][1]
    baseline = next(r for r in results if r["variant_id"] == "A_current_bytetrack")
    improved = (
        float((winner.get("continuity_probe") or {}).get("uninterrupted_duration_sec") or 0)
        > float((baseline.get("continuity_probe") or {}).get("uninterrupted_duration_sec") or 0)
        + 0.5
        and int(winner.get("short_tracks_lt_1s") or 0)
        < int(baseline.get("short_tracks_lt_1s") or 0)
    )
    return {
        "selected_variant_id": winner["variant_id"],
        "keep_baseline_if_not_clearly_better": not improved,
        "product_candidate_variant_id": winner["variant_id"]
        if improved
        else "A_current_bytetrack",
        "improved_vs_baseline": improved,
        "ranking_order": [r["variant_id"] for _, r in scored],
        "selection_rule": [
            "false_target_identity_switch_proxy_safe",
            "target_continuity_probe",
            "occlusion_buffer_via_short_track_reduction",
            "lower_fragmentation",
            "runtime",
        ],
    }
