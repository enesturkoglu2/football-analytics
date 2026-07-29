"""Classify target-tracking failures from accepted GT + variant observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip.metrics import MATCH_IOU, _gt_bbox_at, _iou
from football_analytics.reid.golden_clip.schema import active_intervals

FAILURE_CLASSES = (
    "DETECTION_MISS",
    "SHORT_OCCLUSION_FRAGMENTATION",
    "RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION",
    "WRONG_PLAYER_ID_SWITCH",
    "CAMERA_MOTION_ASSOCIATION_FAILURE",
    "BORDER_EXIT_REENTRY",
    "LONG_OUT_OF_FRAME_REENTRY",
    "LOW_RESOLUTION_AMBIGUITY",
    "CROWD_OVERLAP",
    "UNKNOWN",
)


def classify_failures(
    *,
    ground_truth: Mapping[str, Any],
    variant_observations: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]] | None,
    fps: float,
    variant_id: str,
    width: int = 1326,
    height: int = 750,
) -> dict[str, Any]:
    """Heuristic taxonomy for next-gate selection (not automatic labeling of GT)."""
    if not ground_truth.get("accepted"):
        return {
            "status": "NOT_MEASURABLE_WITHOUT_GT",
            "variant_id": variant_id,
            "classes": {c: {"event_count": 0, "total_duration_sec": 0.0} for c in FAILURE_CLASSES},
        }

    by_f_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in variant_observations:
        by_f_track[int(row["frame_index"])].append(dict(row))
    by_f_det: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in detections or []:
        by_f_det[int(row["frame_index"])].append(dict(row))

    events: dict[str, list[dict[str, Any]]] = {c: [] for c in FAILURE_CLASSES}
    margin = 12.0

    for iv in active_intervals(ground_truth):
        state = str(iv["target_state"])
        start, end = int(iv["start_frame"]), int(iv["end_frame"])
        dur = (end - start + 1) / fps
        rep = {"start_frame": start, "end_frame": end, "duration_sec": dur}

        if state == "TARGET_VISIBLE_BUT_MISSED":
            # check if detections exist near GT bbox
            has_det = False
            for fi in range(start, end + 1):
                gb = _gt_bbox_at(iv, fi)
                if gb is None:
                    continue
                for d in by_f_det.get(fi, []):
                    if _iou(gb, d["bbox_xyxy"]) >= MATCH_IOU:
                        has_det = True
                        break
                if has_det:
                    break
            cls = "DETECTION_MISS" if not has_det else "RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"
            events[cls].append(rep)
        elif state == "WRONG_TARGET_ASSIGNED":
            events["WRONG_PLAYER_ID_SWITCH"].append(rep)
        elif state == "TARGET_OCCLUDED" and dur <= 1.5:
            events["SHORT_OCCLUSION_FRAGMENTATION"].append(rep)
        elif state == "TARGET_OUT_OF_FRAME":
            cls = "LONG_OUT_OF_FRAME_REENTRY" if dur >= 2.0 else "BORDER_EXIT_REENTRY"
            # border check on edges of adjacent associated intervals
            events[cls].append(rep)
        elif state == "TARGET_VISIBLE_ASSOCIATED":
            # fragmentation: multiple associated raw track ids in one short span? skip
            tids = iv.get("associated_raw_track_ids") or []
            if len(tids) > 1:
                events["RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"].append(rep)
            # crowd overlap proxy
            crowd = 0
            for fi in range(start, min(end, start + 30) + 1):
                boxes = by_f_track.get(fi, [])
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        if 0.2 <= _iou(boxes[i]["bbox_xyxy"], boxes[j]["bbox_xyxy"]) < 0.7:
                            crowd += 1
            if crowd >= 5:
                events["CROWD_OVERLAP"].append(rep)
            # low-res ambiguity: tiny bbox
            tiny = 0
            for row in iv.get("bbox_observations") or []:
                b = row["bbox_xyxy"]
                if (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) < 800:
                    tiny += 1
            if tiny >= 5:
                events["LOW_RESOLUTION_AMBIGUITY"].append(rep)
            # border
            for row in (iv.get("bbox_observations") or [])[:1] + (iv.get("bbox_observations") or [])[-1:]:
                b = row.get("bbox_xyxy")
                if not b:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b]
                if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
                    events["BORDER_EXIT_REENTRY"].append(rep)
                    break
        elif state == "TARGET_UNCERTAIN":
            events["UNKNOWN"].append(rep)

    summary = {}
    for cls in FAILURE_CLASSES:
        items = events[cls]
        summary[cls] = {
            "event_count": len(items),
            "total_duration_sec": float(sum(x["duration_sec"] for x in items)),
            "representative_timestamps": items[:5],
            "associated_tracking_variants": [variant_id] if items else [],
            "available_old_identity_signals": [
                "quality",
                "kit",
                "purity",
                "manual_segments",
                "linking_policy_manual_only",
            ]
            if items
            else [],
        }
    return {"status": "ok", "variant_id": variant_id, "classes": summary}


def choose_next_gate(error_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick exactly one next implementation gate from aggregated failure classes."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for summary in error_summaries:
        if summary.get("status") != "ok":
            continue
        for cls, payload in (summary.get("classes") or {}).items():
            totals[cls] += float(payload.get("total_duration_sec") or 0.0)
            counts[cls] += int(payload.get("event_count") or 0)

    # Decision rules (single gate)
    if counts["DETECTION_MISS"] >= max(1, counts["RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"]):
        if totals["DETECTION_MISS"] >= totals["RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"]:
            gate = "TARGET_DETECTION_RECALL_REMEDIATION"
            rationale = "Hedef görünürken detection miss süresi/olayı baskın."
        else:
            gate = "EXISTING_TRACKLET_STITCHING_INTEGRATION"
            rationale = "Detection devam ederken raw-track parçalanması baskın."
    elif counts["SHORT_OCCLUSION_FRAGMENTATION"] > 0 and totals[
        "SHORT_OCCLUSION_FRAGMENTATION"
    ] >= 0.5 * max(1.0, totals["RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"]):
        gate = "TARGET_CONDITIONED_SHORT_OCCLUSION_BRIDGE"
        rationale = "Kısa örtüşme sonrası track kopması baskın; yerel devam kanıtı güçlü."
    elif counts["RAW_TRACK_ID_FRAGMENTATION_WITH_CONTINUOUS_DETECTION"] > 0:
        gate = "EXISTING_TRACKLET_STITCHING_INTEGRATION"
        rationale = "Detection devamlılığı yüksek; raw-track ID parçalanması sık."
    elif counts["WRONG_PLAYER_ID_SWITCH"] > 0:
        # Wrong switches still need persistent target_id observation layer in product
        gate = "TARGET_ID_STATE_AND_OBSERVATION_LAYER"
        rationale = (
            "Yanlış atama olayları var; product runtime kalıcı target_id "
            "observation katmanı olmadan güvenli ilerlemiyor."
        )
    else:
        gate = "TARGET_ID_STATE_AND_OBSERVATION_LAYER"
        rationale = (
            "Annotation/metric altyapısı hazır; product tarafında kalıcı "
            "target_id observation katmanı bir sonraki güvenli kapı."
        )

    return {
        "exact_next_gate": gate,
        "rationale_tr": rationale,
        "class_duration_sec": dict(totals),
        "class_event_count": dict(counts),
        "single_gate_only": True,
    }
