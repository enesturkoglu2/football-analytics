"""Detector snapping for bridge-projected bboxes with cross-team safety."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from football_analytics.reid.target_tracking_r1.candidates import exact_frame_conflict
from football_analytics.reid.target_tracking_r3.policy import R3_POLICY


def _center(b: Sequence[float]) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def score_detector_candidates(
    *,
    projected_bbox: Sequence[float],
    detections: Sequence[Mapping[str, Any]],
    target_kit_state: str,
    kit_by_detection: Mapping[str, Mapping[str, Any]] | None,
    excluded_raw_track_ids: Sequence[str],
    seed_track_meta: Mapping[str, Any] | None,
    track_index: Mapping[str, Mapping[str, Any]] | None,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank detector boxes for snapping; hard-reject reliable white vs yellow target."""
    pol = dict(policy or R3_POLICY)
    kit_map = kit_by_detection or {}
    excluded = {str(x) for x in excluded_raw_track_ids}
    pc = _center(projected_bbox)
    pa = _area(projected_bbox)
    out: list[dict[str, Any]] = []

    for det in detections:
        tid = str(det.get("raw_track_id") or "")
        bbox = [float(v) for v in det["bbox_xyxy"]]
        hard: list[str] = []
        kit_info = kit_map.get(f"{det.get('frame_index')}:{tid}") or kit_map.get(tid) or {}
        cand_kit = str(kit_info.get("kit_state") or det.get("kit_state") or "UNKNOWN")
        reliable_kit = bool(kit_info.get("reliable", True))

        if tid in excluded:
            hard.append("EXCLUDED_IMPURE_PARENT_CONTINUATION")

        # Cross-team hard reject
        if (
            pol["cross_team_hard_reject"]
            and str(target_kit_state).upper() == "YELLOW"
            and cand_kit == "WHITE"
            and reliable_kit
        ):
            hard.append("CROSS_TEAM_KIT_MISMATCH")
        # Unknown must not hard-reject
        if cand_kit == "UNKNOWN" and "CROSS_TEAM_KIT_MISMATCH" in hard:
            hard.remove("CROSS_TEAM_KIT_MISMATCH")

        # Exact-frame conflict with seed track span when candidate overlaps seed active frames
        if seed_track_meta and track_index and tid in track_index:
            if exact_frame_conflict(seed_track_meta, track_index[tid]):
                # Sharing frames with impure parent is expected near overlap — only reject
                # if candidate is a different identity that co-exists for long overlap.
                # Soft: do not hard reject solely for overlapping the parent near bridge.
                pass

        cc = _center(bbox)
        dist = math.hypot(cc[0] - pc[0], cc[1] - pc[1])
        if dist > float(pol["max_snap_center_dist_px"]):
            hard.append("IMPOSSIBLE_OR_EXCESS_DISPLACEMENT")
        iou = _iou(projected_bbox, bbox)
        if iou < float(pol["min_snap_iou"]) and dist > float(pol["max_snap_center_dist_px"]) * 0.5:
            hard.append("LOW_OVERLAP_WITH_PROJECTION")
        ca = _area(bbox)
        if pa > 0 and ca > 0:
            ratio = ca / pa
            if ratio < float(pol["min_scale_ratio"]) or ratio > float(pol["max_scale_ratio"]):
                hard.append("SCALE_INCONSISTENT")
        if ca <= 1:
            hard.append("INVALID_BBOX")

        # Score: higher better (only used when no hard rejects)
        score = 0.55 * iou + 0.35 * max(0.0, 1.0 - dist / float(pol["max_snap_center_dist_px"]))
        if cand_kit == "YELLOW":
            score += 0.08
        elif cand_kit == "UNKNOWN":
            score += 0.02
        # no fake appearance embedding score

        out.append(
            {
                "raw_track_id": tid,
                "detection_id": det.get("detection_id"),
                "bbox_xyxy": bbox,
                "center_xy": list(cc),
                "iou_with_projection": float(iou),
                "center_distance_px": float(dist),
                "kit_state": cand_kit,
                "score": float(score),
                "hard_rejects": hard,
                "eligible": len(hard) == 0,
            }
        )

    out.sort(key=lambda c: (bool(c["eligible"]), float(c["score"])), reverse=True)
    return out


def decide_snap(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pol = dict(policy or R3_POLICY)
    eligible = [c for c in candidates if c.get("eligible")]
    cross_rejects = [
        c for c in candidates if "CROSS_TEAM_KIT_MISMATCH" in (c.get("hard_rejects") or [])
    ]
    if not eligible:
        return {
            "decision": "TARGET_UNRESOLVED",
            "reason": "NO_ELIGIBLE_DETECTOR_CANDIDATE",
            "selected": None,
            "cross_team_rejection_count": len(cross_rejects),
            "candidate_count": len(candidates),
        }
    best = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    margin = float(best["score"]) - (float(second["score"]) if second else 0.0)
    if second is not None and margin < float(pol["min_score_margin"]):
        return {
            "decision": "TARGET_UNRESOLVED",
            "reason": "AMBIGUOUS_CANDIDATE_MARGIN",
            "selected": None,
            "best": dict(best),
            "second": dict(second),
            "margin": margin,
            "cross_team_rejection_count": len(cross_rejects),
            "candidate_count": len(candidates),
        }
    return {
        "decision": "DETECTOR_SNAP",
        "reason": "CLEAR_MARGIN",
        "selected": dict(best),
        "margin": margin if second else float(best["score"]),
        "cross_team_rejection_count": len(cross_rejects),
        "candidate_count": len(candidates),
    }
