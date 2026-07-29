"""Local continuation candidate generation for Target Tracking R1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from football_analytics.reid.candidates import span_interval_overlap, temporal_gap_frames
from football_analytics.reid.target_tracking_r1.policy import STITCH_POLICY


def _center(b: Sequence[float]) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _bbox_at(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_track_id: str,
    frame_index: int,
) -> list[float] | None:
    for r in observations_by_frame.get(str(int(frame_index))) or []:
        if str(r.get("raw_track_id")) == str(raw_track_id):
            return [float(v) for v in r["bbox_xyxy"]]
    return None


def build_track_index(
    inventory_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in inventory_rows:
        tid = str(r["raw_track_id"])
        out[tid] = {
            "raw_track_id": tid,
            "segment_id": r.get("segment_id"),
            "first_frame": int(r["first_frame"]),
            "last_frame": int(r["last_frame"]),
            "observation_count": int(r.get("observation_count") or 0),
            "review_eligible": bool(r.get("review_eligible")),
            "frames": set(int(x) for x in (r.get("observation_frames") or [])),
        }
    return out


def exact_frame_conflict(
    track_a: Mapping[str, Any],
    track_b: Mapping[str, Any],
) -> bool:
    """True if tracks share any observation frame (hard reject)."""
    fa = track_a.get("frames") or set()
    fb = track_b.get("frames") or set()
    if fa and fb:
        return bool(fa & fb)
    # fallback: inclusive span overlap is not exact-frame; only treat as soft
    return False


def generate_continuation_candidates(
    *,
    previous_track_id: str,
    track_index: Mapping[str, Mapping[str, Any]],
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_width: int,
    frame_height: int,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """All hard-gate-passing local births after previous track end (no top-k hiding)."""
    pol = dict(policy or STITCH_POLICY)
    prev = track_index[str(previous_track_id)]
    end_f = int(prev["last_frame"])
    last_bbox = _bbox_at(observations_by_frame, str(previous_track_id), end_f)
    if last_bbox is None:
        # walk back a few frames
        for fi in range(end_f, max(-1, end_f - 5), -1):
            last_bbox = _bbox_at(observations_by_frame, str(previous_track_id), fi)
            if last_bbox:
                end_f = fi
                break
    candidates: list[dict[str, Any]] = []
    max_gap = int(pol["max_local_gap_frames"])
    max_vel = float(pol["max_velocity_px_per_frame"])
    border = float(pol["border_margin_px"])

    for tid, tr in track_index.items():
        if tid == str(previous_track_id):
            continue
        # must start after previous end (local continuation)
        if int(tr["first_frame"]) <= end_f:
            # exact-frame conflict check for any overlap
            if exact_frame_conflict(prev, tr) or span_interval_overlap(
                first_a=int(prev["first_frame"]),
                last_a=int(prev["last_frame"]),
                first_b=int(tr["first_frame"]),
                last_b=int(tr["last_frame"]),
            ):
                # overlapping active spans → never same player simultaneously
                continue
            continue
        gap = temporal_gap_frames(
            first_a=int(prev["first_frame"]),
            last_a=end_f,
            first_b=int(tr["first_frame"]),
            last_b=int(tr["last_frame"]),
        )
        if gap > max_gap:
            continue
        start_bbox = _bbox_at(observations_by_frame, tid, int(tr["first_frame"]))
        hard_rejects: list[str] = []
        if last_bbox is None or start_bbox is None:
            hard_rejects.append("invalid_or_missing_bbox")
            continue
        # invalid bbox
        if _area(last_bbox) <= 1 or _area(start_bbox) <= 1:
            hard_rejects.append("invalid_bbox_area")
            continue
        if exact_frame_conflict(prev, tr):
            hard_rejects.append("exact_frame_conflict")
            continue

        pc = _center(last_bbox)
        nc = _center(start_bbox)
        disp = math.hypot(pc[0] - nc[0], pc[1] - nc[1])
        dt = max(1, gap + 1)
        vel = disp / float(dt)
        if vel > max_vel:
            hard_rejects.append("impossible_image_motion")
            continue

        scale = _area(start_bbox) / _area(last_bbox)
        if scale < float(pol["min_scale_ratio"]) or scale > float(pol["max_scale_ratio"]):
            hard_rejects.append("implausible_scale_change")
            # keep as rejected record? Spec: hard-gate failing not in auto set
            # Still include in manifest with hard_reject for audit — but "hard-gate geçen"
            # means only passers. Skip.
            continue

        if pol.get("require_review_eligible") and not tr.get("review_eligible"):
            hard_rejects.append("not_review_eligible")
            continue

        duration = int(tr["last_frame"]) - int(tr["first_frame"]) + 1
        if duration < int(pol["min_candidate_duration_frames"]):
            hard_rejects.append("candidate_too_short")
            continue

        if disp > float(pol["max_candidate_center_displacement_px"]):
            hard_rejects.append("displacement_too_large")
            continue

        # border exit/entry hints
        border_exit = (
            pc[0] < border
            or pc[0] > frame_width - border
            or pc[1] < border
            or pc[1] > frame_height - border
        )
        border_entry = (
            nc[0] < border
            or nc[0] > frame_width - border
            or nc[1] < border
            or nc[1] > frame_height - border
        )

        # detector continuity: frames in gap with any detection
        det_cont = sum(
            1
            for g in range(end_f + 1, int(tr["first_frame"]))
            if observations_by_frame.get(str(g))
        )
        # crowd: other boxes on start frame
        crowd = max(0, len(observations_by_frame.get(str(tr["first_frame"])) or []) - 1)

        # predicted motion: assume constant velocity from last 5 frames of prev
        motion_consistency = None
        prev_b2 = _bbox_at(observations_by_frame, str(previous_track_id), max(0, end_f - 5))
        if prev_b2 is not None and last_bbox is not None:
            v_est = (
                (pc[0] - _center(prev_b2)[0]) / 5.0,
                (pc[1] - _center(prev_b2)[1]) / 5.0,
            )
            pred = (pc[0] + v_est[0] * dt, pc[1] + v_est[1] * dt)
            motion_consistency = math.hypot(pred[0] - nc[0], pred[1] - nc[1])

        # normalized costs in [0,1]
        gap_n = gap / float(max(1, pol["max_auto_stitch_gap_frames"]))
        disp_n = disp / float(max(1.0, pol["max_auto_center_displacement_px"]))
        scale_n = abs(math.log(max(1e-6, scale))) / abs(math.log(2.0))
        motion_n = (
            (motion_consistency or 0.0) / float(max(1.0, pol["max_auto_center_displacement_px"]))
        )
        cost = 0.45 * min(1.5, gap_n) + 0.45 * min(1.5, disp_n) + 0.05 * min(1.5, scale_n) + 0.05 * min(
            1.5, motion_n
        )

        candidates.append(
            {
                "candidate_raw_track_id": tid,
                "segment_id": tr.get("segment_id"),
                "previous_raw_track_id": str(previous_track_id),
                "previous_end_frame": end_f,
                "candidate_start_frame": int(tr["first_frame"]),
                "candidate_end_frame": int(tr["last_frame"]),
                "temporal_gap_frames": gap,
                "center_displacement_px": disp,
                "normalized_center_displacement": disp_n,
                "bbox_scale_ratio": scale,
                "velocity_px_per_frame": vel,
                "predicted_motion_error_px": motion_consistency,
                "border_exit": border_exit,
                "border_entry": border_entry,
                "crowd_overlap_count": crowd,
                "detector_continuity_frames_in_gap": det_cont,
                "candidate_duration_frames": duration,
                "review_eligible": bool(tr.get("review_eligible")),
                "kit_team_evidence": "UNAVAILABLE",
                "crop_quality": "UNAVAILABLE",
                "appearance_reid": "UNAVAILABLE",
                "track_purity_risk": "UNAVAILABLE",
                "cost": cost,
                "hard_rejects": hard_rejects,
                "hard_gates_passed": True,
            }
        )

    candidates.sort(key=lambda c: (float(c["cost"]), int(c["temporal_gap_frames"]), c["candidate_raw_track_id"]))
    return candidates
