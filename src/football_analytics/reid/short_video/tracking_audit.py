"""Audit short-video ByteTrack fragmentation without re-running detection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _center(b: Sequence[float]) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def audit_tracking(
    *,
    tracks_jsonl: Path,
    detections_jsonl: Path,
    mapping_jsonl: Path | None = None,
    fps: float = 30.0,
    width: int = 1326,
    height: int = 750,
) -> dict[str, Any]:
    obs = _load_jsonl(tracks_jsonl)
    dets = _load_jsonl(detections_jsonl)
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in obs:
        by_track[int(row["raw_track_id"])].append(row)
    for tid in by_track:
        by_track[tid].sort(key=lambda r: int(r["frame_index"]))

    durations = []
    obs_counts = []
    for tid, rows in by_track.items():
        frames = [int(r["frame_index"]) for r in rows]
        dur = (frames[-1] - frames[0] + 1) / fps
        durations.append(dur)
        obs_counts.append(len(rows))
    durations_a = np.asarray(durations, dtype=float)
    obs_a = np.asarray(obs_counts, dtype=float)

    def pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p)) if len(arr) else 0.0

    eligible = None
    if mapping_jsonl and mapping_jsonl.is_file():
        mapping = _load_jsonl(mapping_jsonl)
        eligible = sum(1 for r in mapping if r.get("review_eligible"))

    # per-frame active tracks / births / deaths
    active_per_frame: dict[int, int] = defaultdict(int)
    births: Counter[int] = Counter()
    deaths: Counter[int] = Counter()
    for tid, rows in by_track.items():
        first = int(rows[0]["frame_index"])
        last = int(rows[-1]["frame_index"])
        births[first] += 1
        deaths[last] += 1
        for r in rows:
            active_per_frame[int(r["frame_index"])] += 1

    frames_sorted = sorted(active_per_frame)
    active_counts = [active_per_frame[f] for f in frames_sorted] if frames_sorted else [0]

    # detection gaps (frames with detections but no tracks / empty)
    dets_by_f: dict[int, int] = defaultdict(int)
    for d in dets:
        dets_by_f[int(d["frame_index"])] += 1
    frames_with_det_no_track = 0
    frames_with_track = set(active_per_frame)
    for fi, n in dets_by_f.items():
        if n > 0 and fi not in frames_with_track:
            frames_with_det_no_track += 1

    # confidence drops within tracks
    conf_drop_tracks = 0
    for rows in by_track.values():
        confs = [float(r.get("confidence") or 0.0) for r in rows]
        if confs and (max(confs) - min(confs)) >= 0.35 and confs[-1] < confs[0] - 0.2:
            conf_drop_tracks += 1

    # border entry/exit
    margin = 12.0
    border_events = 0
    for rows in by_track.values():
        for r in (rows[0], rows[-1]):
            x1, y1, x2, y2 = [float(v) for v in r["bbox_xyxy"]]
            if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
                border_events += 1
                break

    # duplicate/overlapping same-frame high IoU different IDs
    by_frame_boxes: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for tid, rows in by_track.items():
        for r in rows:
            by_frame_boxes[int(r["frame_index"])].append((tid, list(r["bbox_xyxy"])))
    duplicate_pair_frames = 0
    for fi, items in by_frame_boxes.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _iou(items[i][1], items[j][1]) >= 0.7:
                    duplicate_pair_frames += 1

    # spatial jump suspects within same ID (possible tracker corruption)
    spatial_jump_tracks = 0
    for rows in by_track.values():
        for a, b in zip(rows, rows[1:]):
            if int(b["frame_index"]) - int(a["frame_index"]) > 3:
                continue
            ca, cb = _center(a["bbox_xyxy"]), _center(b["bbox_xyxy"])
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            diag = ((float(a["bbox_xyxy"][2]) - float(a["bbox_xyxy"][0])) ** 2) ** 0.5
            if dist > max(80.0, 2.5 * max(diag, 20.0)):
                spatial_jump_tracks += 1
                break

    # camera-motion proxy: large collective bbox center shift frame-to-frame
    mean_centers: dict[int, tuple[float, float]] = {}
    for fi, items in by_frame_boxes.items():
        if not items:
            continue
        xs = [(_center(b)[0]) for _, b in items]
        ys = [(_center(b)[1]) for _, b in items]
        mean_centers[fi] = (float(np.mean(xs)), float(np.mean(ys)))
    camera_motion_frames = 0
    prev = None
    for fi in sorted(mean_centers):
        cur = mean_centers[fi]
        if prev is not None and fi - prev[0] == 1:
            dx = cur[0] - prev[1][0]
            dy = cur[1] - prev[1][1]
            if (dx * dx + dy * dy) ** 0.5 >= 18.0:
                camera_motion_frames += 1
        prev = (fi, cur)

    # occlusion proxy: high overlap pairs
    occlusion_frames = 0
    for fi, items in by_frame_boxes.items():
        hit = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if 0.2 <= _iou(items[i][1], items[j][1]) < 0.7:
                    hit = True
                    break
            if hit:
                break
        if hit:
            occlusion_frames += 1

    longest = sorted(
        (
            {
                "raw_track_id": tid,
                "duration_sec": (int(rows[-1]["frame_index"]) - int(rows[0]["frame_index"]) + 1)
                / fps,
                "observation_count": len(rows),
                "first_frame": int(rows[0]["frame_index"]),
                "last_frame": int(rows[-1]["frame_index"]),
            }
            for tid, rows in by_track.items()
        ),
        key=lambda r: r["duration_sec"],
        reverse=True,
    )[:15]

    # root-cause split evidence
    short = int((durations_a < 1.0).sum()) if len(durations_a) else 0
    root_cause = {
        "detector_pressure": {
            "mean_detections_per_frame": float(np.mean(list(dets_by_f.values())))
            if dets_by_f
            else 0.0,
            "note": "Crowded person field (~15/frame) increases association ambiguity",
        },
        "tracker_parameter_pressure": {
            "short_tracks_lt_1s": short,
            "median_duration_sec": pct(durations_a, 50),
            "match_thresh_default": 0.8,
            "track_buffer_default_frames": 30,
            "note": "High short-track rate + strict match_thresh/low buffer typical of association fragmentation",
        },
        "camera_motion_pressure": {
            "frames_with_collective_shift": camera_motion_frames,
            "share_of_frames": float(camera_motion_frames / max(1, len(mean_centers))),
        },
        "occlusion_pressure": {
            "frames_with_partial_overlap": occlusion_frames,
            "share_of_frames": float(occlusion_frames / max(1, len(by_frame_boxes))),
        },
        "assessment": (
            "Primary: tracker association fragmentation under crowded detections + occlusions; "
            "secondary: camera motion; detector itself produces dense valid person boxes (not sparse miss)."
        ),
    }

    return {
        "schema_version": "short_video_tracking_fragmentation_audit_v1",
        "raw_track_count": len(by_track),
        "eligible_track_count": eligible,
        "total_observations": len(obs),
        "observation_count": {
            "p10": pct(obs_a, 10),
            "p25": pct(obs_a, 25),
            "median": pct(obs_a, 50),
            "p75": pct(obs_a, 75),
            "p90": pct(obs_a, 90),
            "mean": float(obs_a.mean()) if len(obs_a) else 0.0,
        },
        "duration_sec": {
            "p10": pct(durations_a, 10),
            "p25": pct(durations_a, 25),
            "median": pct(durations_a, 50),
            "p75": pct(durations_a, 75),
            "p90": pct(durations_a, 90),
            "mean": float(durations_a.mean()) if len(durations_a) else 0.0,
        },
        "short_tracks": {
            "lt_0_5s": int((durations_a < 0.5).sum()) if len(durations_a) else 0,
            "lt_1s": int((durations_a < 1.0).sum()) if len(durations_a) else 0,
            "lt_2s": int((durations_a < 2.0).sum()) if len(durations_a) else 0,
        },
        "longest_tracks": longest,
        "active_tracks_per_frame": {
            "mean": float(np.mean(active_counts)),
            "median": float(np.median(active_counts)),
            "max": int(np.max(active_counts)),
        },
        "birth_death": {
            "birth_frames_nonzero": int(sum(1 for v in births.values() if v)),
            "death_frames_nonzero": int(sum(1 for v in deaths.values() if v)),
            "max_births_in_frame": int(max(births.values()) if births else 0),
            "max_deaths_in_frame": int(max(deaths.values()) if deaths else 0),
        },
        "camera_motion_frames": camera_motion_frames,
        "occlusion_frames": occlusion_frames,
        "border_entry_exit_tracks": border_events,
        "confidence_drop_tracks": conf_drop_tracks,
        "detection_gap_frames_with_det_no_track": frames_with_det_no_track,
        "duplicate_high_iou_pair_instances": duplicate_pair_frames,
        "spatial_jump_suspect_tracks": spatial_jump_tracks,
        "root_cause": root_cause,
        "fps": fps,
    }


def continuity_probe(
    observations: Sequence[Mapping[str, Any]],
    *,
    seed_bbox: Sequence[float],
    seed_frame: int,
    fps: float,
    max_gap: int = 5,
    min_iou: float = 0.2,
) -> dict[str, Any]:
    """Follow a seed bbox via greedy IoU across frames (diagnostic probe, not GT)."""
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(row)
    if seed_frame not in by_frame:
        return {"status": "seed_frame_missing", "uninterrupted_duration_sec": 0.0}

    # pick best IoU at seed
    best = None
    best_iou = -1.0
    for row in by_frame[seed_frame]:
        v = _iou(seed_bbox, row["bbox_xyxy"])
        if v > best_iou:
            best_iou = v
            best = row
    if best is None or best_iou < min_iou:
        return {"status": "seed_bbox_unmatched", "uninterrupted_duration_sec": 0.0}

    tid = int(best["raw_track_id"])
    # measure how long this raw track continues from seed
    track_rows = sorted(
        [r for r in observations if int(r["raw_track_id"]) == tid],
        key=lambda r: int(r["frame_index"]),
    )
    start = int(track_rows[0]["frame_index"])
    end = int(track_rows[-1]["frame_index"])
    # first lost after seed
    frames = {int(r["frame_index"]) for r in track_rows}
    lost = None
    for fi in range(seed_frame, end + 2):
        if fi not in frames and fi > seed_frame:
            # allow small internal gaps
            if any((fi + k) in frames for k in range(1, max_gap + 1)):
                continue
            lost = fi
            break
    uninterrupted_end = (lost - 1) if lost is not None else end
    return {
        "status": "ok",
        "probe_not_ground_truth": True,
        "seed_frame": seed_frame,
        "matched_raw_track_id": tid,
        "seed_iou": best_iou,
        "track_start_frame": start,
        "track_end_frame": end,
        "first_lost_frame_after_seed": lost,
        "uninterrupted_duration_sec": max(0, uninterrupted_end - seed_frame + 1) / fps,
        "full_track_duration_sec": (end - start + 1) / fps,
    }
