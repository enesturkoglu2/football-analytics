"""Segment-level continuation candidates + conservative stitching with kit guards."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from football_analytics.reid.candidates import temporal_gap_frames
from football_analytics.reid.target_tracking_r1.candidates import (
    build_track_index,
    exact_frame_conflict,
)
from football_analytics.reid.target_tracking_r2.policy import R2_POLICY


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _center(b: Sequence[float]) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _bbox_at(obs, tid: str, fi: int):
    for r in obs.get(str(fi)) or []:
        if str(r.get("raw_track_id")) == str(tid):
            return [float(v) for v in r["bbox_xyxy"]]
    return None


def summarize_segment_kit(
    evidence_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_track_id: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    rows = [
        r
        for r in evidence_by_track.get(str(raw_track_id), [])
        if int(start_frame) <= int(r["frame_index"]) <= int(end_frame) and r.get("crop_quality_ok")
    ]
    if not rows:
        return {"kit_state": "UNKNOWN", "reliable": False, "yellow": 0.0, "white": 0.0}
    # majority of reliable kit_state
    counts: dict[str, int] = {}
    y_vals, w_vals = [], []
    for r in rows:
        k = str(r.get("kit_state") or "UNKNOWN")
        counts[k] = counts.get(k, 0) + 1
        if r.get("yellow_evidence") is not None:
            y_vals.append(float(r["yellow_evidence"]))
        if r.get("white_evidence") is not None:
            w_vals.append(float(r["white_evidence"]))
    kit = max(counts, key=counts.get)
    return {
        "kit_state": kit,
        "reliable": kit in {"YELLOW", "WHITE"} and counts[kit] >= max(2, len(rows) // 3),
        "yellow": sum(y_vals) / len(y_vals) if y_vals else 0.0,
        "white": sum(w_vals) / len(w_vals) if w_vals else 0.0,
        "n": len(rows),
    }


def generate_segment_candidates(
    *,
    seed_segment: Mapping[str, Any],
    track_index: Mapping[str, Mapping[str, Any]],
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_width: int,
    frame_height: int,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pol = dict(policy or R2_POLICY)
    end_f = int(seed_segment["end_frame"])
    prev_tid = str(seed_segment["parent_raw_track_id"])
    last_bbox = _bbox_at(observations_by_frame, prev_tid, end_f)
    if last_bbox is None:
        for fi in range(end_f, max(-1, end_f - 5), -1):
            last_bbox = _bbox_at(observations_by_frame, prev_tid, fi)
            if last_bbox:
                end_f = fi
                break
    seed_kit = "YELLOW"
    cands: list[dict[str, Any]] = []
    max_gap = int(pol["max_local_gap_frames"])

    for tid, tr in track_index.items():
        if tid == prev_tid:
            continue
        if int(tr["first_frame"]) <= end_f:
            continue
        gap = temporal_gap_frames(
            first_a=int(seed_segment["start_frame"]),
            last_a=end_f,
            first_b=int(tr["first_frame"]),
            last_b=int(tr["last_frame"]),
        )
        if gap > max_gap:
            continue
        if pol.get("require_review_eligible", True) and not tr.get("review_eligible"):
            continue
        duration = int(tr["last_frame"]) - int(tr["first_frame"]) + 1
        if duration < int(pol["min_candidate_duration_frames"]):
            continue
        start_bbox = _bbox_at(observations_by_frame, tid, int(tr["first_frame"]))
        if last_bbox is None or start_bbox is None:
            continue
        if _area(last_bbox) <= 1 or _area(start_bbox) <= 1:
            continue
        # exact-frame vs seed parent track frames after seed end should be empty for tid
        # Use track frames if present
        prev = {
            "frames": set(range(int(seed_segment["start_frame"]), end_f + 1)),
            "first_frame": int(seed_segment["start_frame"]),
            "last_frame": end_f,
        }
        if exact_frame_conflict(prev, tr):
            continue
        disp = math.hypot(
            _center(last_bbox)[0] - _center(start_bbox)[0],
            _center(last_bbox)[1] - _center(start_bbox)[1],
        )
        if disp > float(pol["max_candidate_center_displacement_px"]):
            continue
        scale = _area(start_bbox) / _area(last_bbox)
        if scale < float(pol["min_scale_ratio"]) or scale > float(pol["max_scale_ratio"]):
            continue
        dt = max(1, gap + 1)
        vel = disp / float(dt)
        if vel > float(pol["max_velocity_px_per_frame"]):
            continue

        kit = summarize_segment_kit(
            evidence_by_track,
            tid,
            int(tr["first_frame"]),
            min(int(tr["first_frame"]) + 20, int(tr["last_frame"])),
        )
        hard_rejects: list[str] = []
        if (
            pol.get("cross_team_hard_reject")
            and seed_kit == "YELLOW"
            and kit.get("reliable")
            and kit.get("kit_state") == "WHITE"
        ):
            hard_rejects.append("CROSS_TEAM_KIT_MISMATCH")

        gap_n = gap / float(max(1, pol["max_auto_stitch_gap_frames"]))
        disp_n = disp / float(max(1.0, pol["max_auto_center_displacement_px"]))
        cost = 0.5 * min(1.5, gap_n) + 0.5 * min(1.5, disp_n)

        cands.append(
            {
                "candidate_unit": "clean_track_segment",
                "candidate_raw_track_id": tid,
                "candidate_segment_id": f"raw{tid}_full",
                "parent_raw_track_id": tid,
                "start_frame": int(tr["first_frame"]),
                "end_frame": int(tr["last_frame"]),
                "temporal_gap_frames": gap,
                "center_displacement_px": disp,
                "bbox_scale_ratio": scale,
                "velocity_px_per_frame": vel,
                "kit": kit,
                "hard_rejects": hard_rejects,
                "hard_gates_passed": len(hard_rejects) == 0,
                "cost": cost,
                "appearance_reid": "UNAVAILABLE",
                "purity_state": "UNKNOWN_UNTIL_AUDITED",
            }
        )
    cands.sort(key=lambda c: (not c["hard_gates_passed"], float(c["cost"]), c["candidate_raw_track_id"]))
    return cands


def decide_segment_stitch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pol = dict(policy or R2_POLICY)
    passed = [c for c in candidates if c.get("hard_gates_passed")]
    rejected_cross = [
        c for c in candidates if "CROSS_TEAM_KIT_MISMATCH" in (c.get("hard_rejects") or [])
    ]
    if not passed:
        return {
            "decision": "TARGET_UNRESOLVED",
            "reason": "no_hard_gate_passing_candidate",
            "selected": None,
            "rejected_cross_team_count": len(rejected_cross),
            "created_at": _utc(),
        }
    best, second = passed[0], (passed[1] if len(passed) > 1 else None)
    margin = None if second is None else float(second["cost"]) - float(best["cost"])
    reasons = []
    ok = True
    if int(best["temporal_gap_frames"]) > int(pol["max_auto_stitch_gap_frames"]):
        ok = False
        reasons.append("gap")
    if float(best["center_displacement_px"]) > float(pol["max_auto_center_displacement_px"]):
        ok = False
        reasons.append("disp")
    if float(best["cost"]) > float(pol["max_auto_cost"]):
        ok = False
        reasons.append("cost")
    if second is not None and (margin is None or margin < float(pol["min_score_margin"])):
        ok = False
        reasons.append("margin")
    # Prefer yellow-consistent when reliable; unknown does not hard-reject
    kit = best.get("kit") or {}
    if kit.get("reliable") and kit.get("kit_state") == "WHITE":
        ok = False
        reasons.append("white_kit")
    if not ok:
        return {
            "decision": "TARGET_UNRESOLVED",
            "reason": ",".join(reasons) or "ambiguous",
            "selected": None,
            "best_candidate": dict(best),
            "runner_up": dict(second) if second else None,
            "margin": margin,
            "rejected_cross_team_count": len(rejected_cross),
            "created_at": _utc(),
        }
    return {
        "decision": "AUTO_STITCH",
        "reason": "segment_hard_gates_passed_clear_margin",
        "selected": dict(best),
        "runner_up": dict(second) if second else None,
        "margin": margin if second is not None else float("inf"),
        "rejected_cross_team_count": len(rejected_cross),
        "created_at": _utc(),
    }


def build_r2_timeline(
    *,
    persistent_target_id: str,
    target_id: str,
    seed_segment: Mapping[str, Any],
    conflict_segment: Mapping[str, Any] | None,
    stitch_decision: Mapping[str, Any],
    fps: float,
) -> dict[str, Any]:
    intervals = [
        {
            "interval_id": "tiv_seed_segment",
            "kind": "HUMAN_SEED_SEGMENT",
            "start_frame": int(seed_segment["start_frame"]),
            "end_frame": int(seed_segment["end_frame"]),
            "start_time": float(seed_segment["start_time"]),
            "end_time": float(seed_segment["end_time"]),
            "raw_track_id": seed_segment["parent_raw_track_id"],
            "segment_id": seed_segment["segment_id"],
            "analysis_eligible": True,
            "label": "TARGET SEED — PURE YELLOW SEGMENT",
        }
    ]
    if conflict_segment is not None:
        intervals.append(
            {
                "interval_id": "tiv_conflict_excluded",
                "kind": "IDENTITY_CONFLICT_EXCLUDED",
                "start_frame": int(conflict_segment["start_frame"]),
                "end_frame": int(conflict_segment["end_frame"]),
                "start_time": float(conflict_segment["start_time"]),
                "end_time": float(conflict_segment["end_time"]),
                "raw_track_id": conflict_segment["parent_raw_track_id"],
                "segment_id": conflict_segment["segment_id"],
                "analysis_eligible": False,
                "reason": "CROSS_TEAM_IDENTITY_CONFLICT",
                "label": "IDENTITY CONFLICT — CROSS TEAM",
            }
        )
    # After seed end → temporarily lost / unresolved unless stitch
    lost_start = int(seed_segment["end_frame"]) + 1
    if stitch_decision.get("decision") == "AUTO_STITCH" and stitch_decision.get("selected"):
        sel = stitch_decision["selected"]
        if lost_start < int(sel["start_frame"]):
            intervals.append(
                {
                    "interval_id": "tiv_gap",
                    "kind": "TARGET_TEMPORARILY_LOST",
                    "start_frame": lost_start,
                    "end_frame": int(sel["start_frame"]) - 1,
                    "start_time": lost_start / fps,
                    "end_time": int(sel["start_frame"]) / fps,
                    "analysis_eligible": False,
                    "label": "TARGET TEMPORARILY LOST",
                }
            )
        intervals.append(
            {
                "interval_id": "tiv_auto_seg",
                "kind": "AUTO_STITCHED_SEGMENT",
                "start_frame": int(sel["start_frame"]),
                "end_frame": int(sel["end_frame"]),
                "start_time": int(sel["start_frame"]) / fps,
                "end_time": (int(sel["end_frame"]) + 1) / fps,
                "raw_track_id": sel["candidate_raw_track_id"],
                "segment_id": sel.get("candidate_segment_id"),
                "analysis_eligible": True,
                "label": "TARGET — AUTO STITCHED SEGMENT",
                "score": {
                    "cost": sel.get("cost"),
                    "gap": sel.get("temporal_gap_frames"),
                    "disp": sel.get("center_displacement_px"),
                    "kit": sel.get("kit"),
                },
            }
        )
    else:
        intervals.append(
            {
                "interval_id": "tiv_unresolved",
                "kind": "TARGET_UNRESOLVED",
                "start_frame": lost_start,
                "end_frame": lost_start + 45,
                "start_time": lost_start / fps,
                "end_time": (lost_start + 46) / fps,
                "analysis_eligible": False,
                "label": "TARGET UNRESOLVED",
                "reason": stitch_decision.get("reason"),
            }
        )
    return {
        "schema_version": "target_tracking_r2_timeline_v1",
        "persistent_target_id": persistent_target_id,
        "target_id": target_id,
        "intervals": intervals,
        "full_metrics": {
            "target_idf1": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "target_recall": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "false_identity_switch_rate": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
        },
        "human_acceptance": "HUMAN_VISUAL_ACCEPTANCE_PENDING",
    }
