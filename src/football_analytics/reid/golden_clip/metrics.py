"""Target-specific tracking metrics against accepted ground truth."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip import (
    CORRECT_ASSIGNMENT_STATES,
    NOT_MEASURABLE_WITHOUT_GT,
    VISIBLE_STATES,
)
from football_analytics.reid.golden_clip.schema import active_intervals

# IoU threshold for frame-level association match (operational definition).
MATCH_IOU = 0.5


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


def _gt_frame_map(gt: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for iv in active_intervals(gt):
        for fi in range(int(iv["start_frame"]), int(iv["end_frame"]) + 1):
            out[fi] = iv
    return out


def _obs_by_frame_track(
    observations: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_f: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_f[int(row["frame_index"])].append(dict(row))
    return by_f


def _best_match(
    gt_bbox: Sequence[float] | None,
    dets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if gt_bbox is None or not dets:
        return None, 0.0
    best = None
    best_iou = -1.0
    for d in dets:
        v = _iou(gt_bbox, d["bbox_xyxy"])
        if v > best_iou:
            best_iou = v
            best = d
    if best is None or best_iou < MATCH_IOU:
        return None, max(0.0, best_iou)
    return best, best_iou


def _gt_bbox_at(iv: Mapping[str, Any], fi: int) -> list[float] | None:
    # Prefer true target bbox for WRONG_TARGET_ASSIGNED; else associated obs for ASSOCIATED
    if iv.get("target_state") == "WRONG_TARGET_ASSIGNED":
        for row in iv.get("true_target_bbox_observations") or []:
            if int(row.get("frame_index", -1)) == fi:
                return [float(v) for v in row["bbox_xyxy"]]
    for row in iv.get("bbox_observations") or []:
        if int(row.get("frame_index", -1)) == fi:
            return [float(v) for v in row["bbox_xyxy"]]
    return None


def metrics_without_gt() -> dict[str, Any]:
    return {
        "status": NOT_MEASURABLE_WITHOUT_GT,
        "false_target_identity_switch": NOT_MEASURABLE_WITHOUT_GT,
        "false_target_identity_switch_count": NOT_MEASURABLE_WITHOUT_GT,
        "note": (
            "Accepted target_identity_ground_truth_v1 required. "
            "Placeholder false_target_identity_switch=0 from prior stabilization "
            "must not be treated as a measured result."
        ),
        "seed_iou_continuity_proxy_seconds": None,
        "seed_iou_continuity_proxy_is_not_target_accuracy": True,
    }


def evaluate_variant_against_gt(
    *,
    ground_truth: Mapping[str, Any],
    variant_observations: Sequence[Mapping[str, Any]],
    fps: float,
    variant_id: str,
    runtime_sec: float | None = None,
    seed_iou_continuity_proxy_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute target-specific metrics.

    Operational definitions (final report mirrors these):
    - target_visible_duration: frames where GT state in VISIBLE_STATES, / fps
    - correctly_assigned_duration: visible frames where best tracker box IoU>=0.5
      matches a GT associated raw_track_id for ASSOCIATED, OR for MISSED is never
      correctly assigned; for WRONG_TARGET_ASSIGNED correct only if tracker matches
      true-target association when provided via true_target bbox IoU and track id
      continuity is not required for precision numerator.
    - target_precision: correctly_assigned / frames where tracker claims target
      (heuristic: frames with any track overlapping GT visible bbox at IoU>=0.5)
    - target_recall: correctly_assigned / (visible_frames - uncertain_visible_excl)
    - association_f1: harmonic mean of precision and recall (IDF1-equivalent proxy
      when single-target; not multi-ID MOT IDF1)
    - false_target_identity_switch_count: times consecutive correctly-assigned
      frames change raw_track_id incorrectly relative to GT association changes
      that are not explained by GT WRONG→CORRECT transitions
    """
    if not ground_truth.get("accepted"):
        out = metrics_without_gt()
        out["variant_id"] = variant_id
        out["seed_iou_continuity_proxy_seconds"] = seed_iou_continuity_proxy_seconds
        return out

    gt_map = _gt_frame_map(ground_truth)
    by_f = _obs_by_frame_track(variant_observations)
    fps = float(fps) if fps else 30.0

    visible_frames = 0
    uncertain_frames = 0
    occluded_frames = 0
    oof_frames = 0
    missed_frames = 0
    correct_frames = 0
    false_assign_frames = 0
    false_assign_events = 0
    claim_frames = 0
    loss_events = 0
    recovery_latencies: list[float] = []
    longest_correct = 0
    cur_correct = 0
    prev_correct_tid: str | None = None
    identity_switches = 0
    was_lost = False
    lost_at: int | None = None
    track_ids_used: set[str] = set()
    in_false_run = False

    # correctness denominator excludes TARGET_UNCERTAIN
    for fi in sorted(gt_map):
        iv = gt_map[fi]
        state = str(iv["target_state"])
        if state == "TARGET_UNCERTAIN":
            uncertain_frames += 1
            continue
        if state == "TARGET_OCCLUDED":
            occluded_frames += 1
            continue
        if state == "TARGET_OUT_OF_FRAME":
            oof_frames += 1
            continue

        if state in VISIBLE_STATES:
            visible_frames += 1

        gt_bbox = _gt_bbox_at(iv, fi)
        dets = by_f.get(fi, [])
        match, iou = _best_match(gt_bbox, dets)
        assoc_ids = {str(x) for x in (iv.get("associated_raw_track_ids") or [])}

        claimed = match is not None
        if claimed:
            claim_frames += 1
            track_ids_used.add(str(match["raw_track_id"]))

        correct = False
        if state == "TARGET_VISIBLE_ASSOCIATED":
            if match is not None and str(match["raw_track_id"]) in assoc_ids:
                correct = True
            elif match is not None and assoc_ids and str(match["raw_track_id"]) not in assoc_ids:
                false_assign_frames += 1
                if not in_false_run:
                    false_assign_events += 1
                    in_false_run = True
            else:
                in_false_run = False
                missed_frames += 1
        elif state == "TARGET_VISIBLE_BUT_MISSED":
            # GT says tracker missed; any claim overlapping is false assignment
            if match is not None:
                false_assign_frames += 1
                if not in_false_run:
                    false_assign_events += 1
                    in_false_run = True
            else:
                in_false_run = False
                missed_frames += 1
        elif state == "WRONG_TARGET_ASSIGNED":
            # associated_* are wrong; correct if tracker is NOT those wrong ids
            # and overlaps true target bbox (if present); else false if matches wrong
            if match is not None and str(match["raw_track_id"]) in assoc_ids:
                false_assign_frames += 1
                if not in_false_run:
                    false_assign_events += 1
                    in_false_run = True
            elif match is not None and gt_bbox is not None and iou >= MATCH_IOU:
                correct = True
                in_false_run = False
            else:
                in_false_run = False
                missed_frames += 1

        if correct:
            correct_frames += 1
            cur_correct += 1
            longest_correct = max(longest_correct, cur_correct)
            tid = str(match["raw_track_id"]) if match else None
            if (
                prev_correct_tid is not None
                and tid is not None
                and tid != prev_correct_tid
                and state == "TARGET_VISIBLE_ASSOCIATED"
                and tid not in assoc_ids
            ):
                identity_switches += 1
            # identity switch when GT association is stable but tracker id flips
            if (
                prev_correct_tid is not None
                and tid is not None
                and tid != prev_correct_tid
                and state == "TARGET_VISIBLE_ASSOCIATED"
                and len(assoc_ids) == 1
                and tid in assoc_ids
                and prev_correct_tid in assoc_ids
            ):
                # both in assoc set (multi-id) — not a false switch
                pass
            elif (
                prev_correct_tid is not None
                and tid is not None
                and tid != prev_correct_tid
                and state == "TARGET_VISIBLE_ASSOCIATED"
                and tid in assoc_ids
                and prev_correct_tid not in assoc_ids
            ):
                identity_switches += 1
            prev_correct_tid = tid
            if was_lost and lost_at is not None:
                recovery_latencies.append((fi - lost_at) / fps)
                was_lost = False
                lost_at = None
        else:
            cur_correct = 0
            if state in VISIBLE_STATES and not was_lost:
                loss_events += 1
                was_lost = True
                lost_at = fi
            prev_correct_tid = None

    denom_vis = max(1, visible_frames)
    precision = correct_frames / claim_frames if claim_frames else 0.0
    recall = correct_frames / denom_vis if visible_frames else 0.0
    if precision + recall > 0:
        assoc_f1 = 2 * precision * recall / (precision + recall)
    else:
        assoc_f1 = 0.0

    duration_sec = (max(gt_map) + 1) / fps if gt_map else 0.0
    interventions = false_assign_events + loss_events  # operational proxy
    interventions_per_min = (
        interventions / (duration_sec / 60.0) if duration_sec > 0 else 0.0
    )

    return {
        "status": "ok",
        "variant_id": variant_id,
        "target_visible_duration": visible_frames / fps,
        "correctly_assigned_duration": correct_frames / fps,
        "target_visible_coverage": correct_frames / denom_vis if visible_frames else 0.0,
        "target_precision": precision,
        "target_recall": recall,
        "target_association_f1": assoc_f1,
        "target_association_f1_note": (
            "Single-target association F1 (precision/recall harmonic mean); "
            "IDF1-equivalent proxy for one identity, not multi-ID MOTChallenge IDF1."
        ),
        "false_target_assignment_count": false_assign_events,
        "false_target_assignment_duration": false_assign_frames / fps,
        "false_target_identity_switch_count": identity_switches,
        "target_loss_count": loss_events,
        "visible_but_missed_duration": missed_frames / fps,
        "longest_uninterrupted_correct_duration": longest_correct / fps,
        "recovery_latency_distribution": {
            "count": len(recovery_latencies),
            "mean_sec": (
                float(sum(recovery_latencies) / len(recovery_latencies))
                if recovery_latencies
                else None
            ),
            "max_sec": max(recovery_latencies) if recovery_latencies else None,
            "values_sec": recovery_latencies[:50],
        },
        "unresolved_duration": uncertain_frames / fps,
        "manual_interventions_per_minute": interventions_per_min,
        "target_track_fragmentation": len(track_ids_used),
        "out_of_frame_duration": oof_frames / fps,
        "occluded_duration": occluded_frames / fps,
        "uncertain_duration": uncertain_frames / fps,
        "runtime_sec": runtime_sec,
        "seed_iou_continuity_proxy_seconds": seed_iou_continuity_proxy_seconds,
        "seed_iou_continuity_proxy_is_not_target_accuracy": True,
        "match_iou_threshold": MATCH_IOU,
    }


def select_best_variant(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Priority: false assign → identity switch → coverage → uninterrupted → interventions → runtime."""
    scored = []
    for row in results:
        if row.get("status") != "ok":
            scored.append(((-1e18,), row))
            continue
        score = (
            -int(row.get("false_target_assignment_count") or 0),
            -int(row.get("false_target_identity_switch_count") or 0),
            float(row.get("target_visible_coverage") or 0.0),
            float(row.get("longest_uninterrupted_correct_duration") or 0.0),
            -float(row.get("manual_interventions_per_minute") or 0.0),
            -float(row.get("runtime_sec") or 1e9),
        )
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    winner = scored[0][1] if scored else {}
    return {
        "selected_variant_id": winner.get("variant_id"),
        "ranking_order": [r.get("variant_id") for _, r in scored],
        "selection_rule": [
            "lowest_false_target_assignment",
            "lowest_false_identity_switch",
            "highest_correctly_assigned_visible_coverage",
            "longest_uninterrupted_correct_duration",
            "lowest_intervention_per_minute",
            "runtime",
        ],
        "raw_track_count_not_success_criterion": True,
    }
