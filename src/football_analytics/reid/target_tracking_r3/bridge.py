"""Short-occlusion bridge loop: flow predict → detector snap → safety gates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.target_tracking_r3.flow import (
    extract_quality_gated_features,
    project_bbox,
    track_flow_forward_backward,
)
from football_analytics.reid.target_tracking_r3.policy import R3_POLICY
from football_analytics.reid.target_tracking_r3.snap import decide_snap, score_detector_candidates


def resolve_bridge_window(
    *,
    last_reliable_frame: int,
    refined_change_point: int,
    contamination_frames: Sequence[int],
    nearby_birth_frames: Sequence[int],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dynamically bound the short local bridge window (no random fixed duration)."""
    pol = dict(policy or R3_POLICY)
    start = int(last_reliable_frame) + 1
    contam = [int(f) for f in contamination_frames if int(f) >= start]
    births = [int(f) for f in nearby_birth_frames if int(f) >= start]
    end_candidates = [
        int(refined_change_point) + int(pol["overlap_extend_after_contam_frames"]),
        start + int(pol["max_bridge_frames"]) - 1,
    ]
    if contam:
        end_candidates.append(max(contam) + int(pol["overlap_extend_after_contam_frames"]))
    if births:
        end_candidates.append(min(births) + int(pol["snap_confirm_frames"]) + 5)
    end = min(end_candidates)
    end = max(end, start)  # at least one frame attempt
    span = end - start + 1
    long_gap = span > int(pol["long_gap_frames"])
    return {
        "bridge_start_frame": start,
        "bridge_end_frame": end if not long_gap else start - 1,
        "bridge_span_frames": 0 if long_gap else span,
        "long_gap_review_required": long_gap,
        "resolution_inputs": {
            "refined_change_point": int(refined_change_point),
            "contamination_frames": contam[:20],
            "nearby_birth_frames": births[:20],
            "max_bridge_frames": int(pol["max_bridge_frames"]),
        },
    }


def run_bridge(
    *,
    video_path: str,
    bridge_state: Mapping[str, Any],
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    kit_lookup: Mapping[str, Mapping[str, Any]],
    excluded_raw_track_ids: Sequence[str],
    seed_track_meta: Mapping[str, Any] | None,
    track_index: Mapping[str, Mapping[str, Any]] | None,
    window: Mapping[str, Any],
    fps: float,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute bounded bridge; never assign permanent identity to a raw_track_id."""
    pol = dict(policy or R3_POLICY)
    if window.get("long_gap_review_required"):
        return {
            "schema_version": "target_tracking_r3_bridge_result_v1",
            "bridge_decision": "LONG_GAP_REVIEW_REQUIRED",
            "accepted": False,
            "frames": [],
            "metrics": {"bridge_event_count": 0},
        }
    if bridge_state.get("bridge_status") != "READY":
        return {
            "schema_version": "target_tracking_r3_bridge_result_v1",
            "bridge_decision": "TARGET_UNRESOLVED",
            "reason": "BRIDGE_STATE_NOT_READY",
            "accepted": False,
            "frames": [],
            "metrics": {"bridge_event_count": 0},
        }

    start = int(window["bridge_start_frame"])
    end = int(window["bridge_end_frame"])
    cap = cv2.VideoCapture(str(video_path))
    # seek to last reliable frame
    last_f = int(bridge_state["last_reliable_frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_f)
    ok, prev_bgr = cap.read()
    if not ok or prev_bgr is None:
        cap.release()
        return {
            "schema_version": "target_tracking_r3_bridge_result_v1",
            "bridge_decision": "BLOCKED_TARGET_TRACKING_R3_FLOW",
            "accepted": False,
            "frames": [],
            "metrics": {},
        }
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    bbox = [float(v) for v in bridge_state["last_reliable_bbox"]]
    pts = extract_quality_gated_features(prev_gray, bbox, policy=pol)
    if pts is None:
        cap.release()
        return {
            "schema_version": "target_tracking_r3_bridge_result_v1",
            "bridge_decision": "TARGET_UNRESOLVED",
            "reason": "BRIDGE_FLOW_UNRELIABLE",
            "accepted": False,
            "frames": [],
            "metrics": {"flow_reliability": "UNRELIABLE_AT_INIT"},
        }

    vel = bridge_state.get("velocity_estimate_px_per_frame") or {"vx": 0.0, "vy": 0.0}
    target_kit = str(
        (bridge_state.get("torso_kit_descriptor") or {}).get("dominant_kit_state") or "YELLOW"
    )
    frame_records: list[dict[str, Any]] = []
    snap_confirm_run = 0
    accepted_snap: dict[str, Any] | None = None
    cross_team_rejects = 0
    flow_unreliable_count = 0
    detector_snapped_frames = 0

    for fi in range(start, end + 1):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        flow = track_flow_forward_backward(prev_gray, gray, pts, policy=pol)
        status = "FLOW_PREDICTION"
        projected = None
        snap_decision = None
        selected = None
        cands: list[dict[str, Any]] = []

        if not flow["reliable"]:
            flow_unreliable_count += 1
            # motion fallback using velocity estimate
            projected = project_bbox(
                bbox, (float(vel["vx"]), float(vel["vy"])), scale_factor=1.0
            )
            status = "BRIDGE_FLOW_UNRELIABLE"
            # still attempt detector snap with velocity projection
        else:
            projected = project_bbox(bbox, flow["median_delta"], scale_factor=1.0)
            pts = flow["inlier_points"]

        dets = list(observations_by_frame.get(str(fi)) or [])
        # attach frame_index for kit lookup
        det_enriched = []
        for d in dets:
            dd = dict(d)
            dd["frame_index"] = fi
            det_enriched.append(dd)

        cands = score_detector_candidates(
            projected_bbox=projected,
            detections=det_enriched,
            target_kit_state=target_kit,
            kit_by_detection=kit_lookup,
            excluded_raw_track_ids=excluded_raw_track_ids,
            seed_track_meta=seed_track_meta,
            track_index=track_index,
            policy=pol,
        )
        cross_team_rejects += sum(
            1 for c in cands if "CROSS_TEAM_KIT_MISMATCH" in (c.get("hard_rejects") or [])
        )
        snap_decision = decide_snap(cands, policy=pol)

        if snap_decision["decision"] == "DETECTOR_SNAP" and flow["reliable"]:
            selected = snap_decision["selected"]
            bbox = list(selected["bbox_xyxy"])
            status = "DETECTOR_SNAP"
            detector_snapped_frames += 1
            snap_confirm_run += 1
            # re-seed features on snapped box
            new_pts = extract_quality_gated_features(gray, bbox, policy=pol)
            if new_pts is not None:
                pts = new_pts
            if snap_confirm_run >= int(pol["snap_confirm_frames"]):
                accepted_snap = {
                    "frame_index": fi,
                    "raw_track_id": selected["raw_track_id"],
                    "bbox_xyxy": bbox,
                    "confirm_frames": snap_confirm_run,
                }
                frame_records.append(
                    {
                        "frame_index": fi,
                        "timestamp_sec": fi / fps,
                        "bbox_xyxy": bbox,
                        "projected_bbox": projected,
                        "status": "BRIDGE_ACCEPTED",
                        "flow_confidence": flow.get("inlier_ratio"),
                        "flow_reliable": flow["reliable"],
                        "median_delta": flow.get("median_delta"),
                        "snap": snap_decision,
                        "candidates": cands[:8],
                        "supporting_detection": selected,
                        "provenance": "DETECTOR_SNAPPED_CONTINUATION",
                    }
                )
                # continue a few more frames optional — stop after accept for short event
                prev_gray = gray
                break
        elif snap_decision["decision"] == "DETECTOR_SNAP" and not flow["reliable"]:
            # detector support without reliable flow → unresolved (policy: both required)
            snap_confirm_run = 0
            status = "TARGET_UNRESOLVED"
            reason = "FLOW_UNRELIABLE_DESPITE_DETECTOR"
        else:
            snap_confirm_run = 0
            status = "TARGET_UNRESOLVED" if status != "BRIDGE_FLOW_UNRELIABLE" else status
            # keep projected bbox for visualization only — not committed as identity
            bbox = list(projected)

        frame_records.append(
            {
                "frame_index": fi,
                "timestamp_sec": fi / fps,
                "bbox_xyxy": list(bbox) if status in {"DETECTOR_SNAP", "BRIDGE_ACCEPTED"} else None,
                "projected_bbox": projected,
                "status": status,
                "flow_confidence": flow.get("inlier_ratio"),
                "flow_reliable": bool(flow.get("reliable")),
                "median_delta": flow.get("median_delta"),
                "n_flow_inliers": flow.get("n_inliers"),
                "snap": snap_decision,
                "candidates": cands[:8],
                "supporting_detection": selected,
                "provenance": (
                    "TARGET_CONDITIONED_BRIDGE"
                    if status in {"FLOW_PREDICTION", "DETECTOR_SNAP"}
                    else "UNRESOLVED_GAP"
                ),
                "reason": snap_decision.get("reason") if snap_decision else flow.get("reason"),
            }
        )
        prev_gray = gray
        # if too many consecutive unreliable flow frames, abort
        if flow_unreliable_count >= 5 and accepted_snap is None:
            break

    cap.release()

    if accepted_snap is not None:
        decision = "BRIDGE_ACCEPTED"
        accepted = True
    elif any(r["status"] == "BRIDGE_FLOW_UNRELIABLE" for r in frame_records) and not any(
        r["status"] == "DETECTOR_SNAP" for r in frame_records
    ):
        decision = "TARGET_UNRESOLVED"
        accepted = False
        reason = "BRIDGE_FLOW_UNRELIABLE"
    else:
        decision = "TARGET_UNRESOLVED"
        accepted = False
        reason = "NO_CLEAR_SNAP"

    return {
        "schema_version": "target_tracking_r3_bridge_result_v1",
        "bridge_decision": decision,
        "accepted": accepted,
        "reason": None if accepted else locals().get("reason", "UNRESOLVED"),
        "accepted_snap": accepted_snap,
        "frames": frame_records,
        "window": dict(window),
        "metrics": {
            "bridge_event_count": 1,
            "bridge_accepted": accepted,
            "bridge_rejected": not accepted,
            "bridge_frame_duration": len(frame_records),
            "detector_snapped_frame_count": detector_snapped_frames,
            "flow_reliability": (
                "MIXED"
                if flow_unreliable_count and detector_snapped_frames
                else ("UNRELIABLE" if flow_unreliable_count else "RELIABLE")
            ),
            "flow_unreliable_frames": flow_unreliable_count,
            "cross_team_rejection_count": cross_team_rejects,
            "candidate_count_last": len(frame_records[-1]["candidates"]) if frame_records else 0,
        },
    }


def build_r3_timeline(
    *,
    persistent_target_id: str,
    target_id: str,
    refined_seed_start: int,
    refined_seed_end: int,
    seed_segment_id: str,
    parent_raw_track_id: str,
    bridge_result: Mapping[str, Any],
    fps: float,
    frame_count: int,
) -> dict[str, Any]:
    """Persistent target intervals with provenance (no raw_id identity binding)."""
    intervals: list[dict[str, Any]] = [
        {
            "kind": "HUMAN_SEED_SEGMENT",
            "segment_id": seed_segment_id,
            "parent_raw_track_id": str(parent_raw_track_id),
            "start_frame": int(refined_seed_start),
            "end_frame": int(refined_seed_end),
            "start_time": int(refined_seed_start) / fps,
            "end_time": (int(refined_seed_end) + 1) / fps,
            "analysis_eligible": True,
            "provenance": "HUMAN_SEED_SEGMENT",
        }
    ]
    observations: list[dict[str, Any]] = []
    for fr in bridge_result.get("frames") or []:
        st = fr.get("status")
        if st in {"DETECTOR_SNAP", "BRIDGE_ACCEPTED"} and fr.get("bbox_xyxy"):
            kind = (
                "DETECTOR_SNAPPED_CONTINUATION"
                if st == "BRIDGE_ACCEPTED" or st == "DETECTOR_SNAP"
                else "TARGET_CONDITIONED_BRIDGE"
            )
            observations.append(
                {
                    "frame_index": fr["frame_index"],
                    "timestamp_sec": fr["timestamp_sec"],
                    "bbox_xyxy": fr["bbox_xyxy"],
                    "supporting_detection": fr.get("supporting_detection"),
                    "raw_track_id": (fr.get("supporting_detection") or {}).get("raw_track_id"),
                    "flow_confidence": fr.get("flow_confidence"),
                    "snap_evidence": fr.get("snap"),
                    "kit_evidence": (fr.get("supporting_detection") or {}).get("kit_state"),
                    "status": st,
                    "provenance": fr.get("provenance") or kind,
                }
            )
        elif st in {"FLOW_PREDICTION"} and fr.get("projected_bbox"):
            observations.append(
                {
                    "frame_index": fr["frame_index"],
                    "timestamp_sec": fr["timestamp_sec"],
                    "bbox_xyxy": None,
                    "projected_bbox": fr.get("projected_bbox"),
                    "status": "TARGET_CONDITIONED_BRIDGE",
                    "provenance": "TARGET_CONDITIONED_BRIDGE",
                    "flow_confidence": fr.get("flow_confidence"),
                    "note": "projection_only_not_identity",
                }
            )

    if bridge_result.get("accepted") and observations:
        snap_frames = [
            o["frame_index"]
            for o in observations
            if o.get("status") in {"DETECTOR_SNAP", "BRIDGE_ACCEPTED"}
        ]
        if snap_frames:
            intervals.append(
                {
                    "kind": "DETECTOR_SNAPPED_CONTINUATION",
                    "start_frame": min(snap_frames),
                    "end_frame": max(snap_frames),
                    "start_time": min(snap_frames) / fps,
                    "end_time": (max(snap_frames) + 1) / fps,
                    "analysis_eligible": True,
                    "provenance": "DETECTOR_SNAPPED_CONTINUATION",
                    "note": "temporary_support_not_permanent_raw_id_binding",
                }
            )
    else:
        gap_start = int(refined_seed_end) + 1
        intervals.append(
            {
                "kind": "UNRESOLVED_GAP",
                "start_frame": gap_start,
                "end_frame": int(frame_count) - 1,
                "start_time": gap_start / fps,
                "end_time": frame_count / fps,
                "analysis_eligible": False,
                "provenance": "UNRESOLVED_GAP",
                "status": "TARGET_UNRESOLVED",
            }
        )

    eligible_frames = sum(
        int(iv["end_frame"]) - int(iv["start_frame"]) + 1
        for iv in intervals
        if iv.get("analysis_eligible")
    )
    return {
        "schema_version": "target_tracking_r3_timeline_v1",
        "persistent_target_id": persistent_target_id,
        "target_id": target_id,
        "intervals": intervals,
        "bridge_observations": observations,
        "persistent_target_duration_frames": eligible_frames,
        "persistent_target_duration_sec": eligible_frames / fps,
        "raw_track_permanent_binding": False,
    }
