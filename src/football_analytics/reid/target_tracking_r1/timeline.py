"""Target timeline from persistent state + track spans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


INTERVAL_KINDS = frozenset(
    {
        "HUMAN_CONFIRMED",
        "RAW_TRACK_CONTINUATION",
        "AUTO_STITCHED_CONTINUATION",
        "UNRESOLVED",
        "OUT_OF_FRAME",
        "REVIEW_REQUIRED",
    }
)


def build_target_timeline(
    *,
    persistent_target_id: str,
    target_id: str,
    seed_raw_track_id: str,
    chain_raw_track_ids: Sequence[str],
    track_index: Mapping[str, Mapping[str, Any]],
    stitch_events: Sequence[Mapping[str, Any]],
    fps: float,
) -> dict[str, Any]:
    intervals: list[dict[str, Any]] = []
    # Seed track full span: HUMAN_CONFIRMED (user seed establishes identity)
    seed = track_index[str(seed_raw_track_id)]
    intervals.append(
        {
            "interval_id": f"tiv_seed_{seed_raw_track_id}",
            "kind": "HUMAN_CONFIRMED",
            "start_frame": int(seed["first_frame"]),
            "end_frame": int(seed["last_frame"]),
            "start_time": int(seed["first_frame"]) / fps,
            "end_time": (int(seed["last_frame"]) + 1) / fps,
            "raw_track_id": str(seed_raw_track_id),
            "stitching_source": "HUMAN_SEED",
            "candidate_score_breakdown": None,
            "provenance": {"seed_raw_track_id": str(seed_raw_track_id)},
            "confidence_class": "human_seed",
            "analysis_eligible": True,
        }
    )

    stitch_by_prev = {
        str(e["previous_raw_track_id"]): e
        for e in stitch_events
        if (e.get("decision") or {}).get("decision") == "AUTO_STITCH"
    }

    for prev, nxt in zip(chain_raw_track_ids, chain_raw_track_ids[1:]):
        ev = stitch_by_prev.get(str(prev))
        tr = track_index[str(nxt)]
        sel = (ev or {}).get("decision", {}).get("selected") or {}
        # gap interval
        gap_start = int(track_index[str(prev)]["last_frame"]) + 1
        gap_end = int(tr["first_frame"]) - 1
        if gap_end >= gap_start:
            intervals.append(
                {
                    "interval_id": f"tiv_gap_{prev}_{nxt}",
                    "kind": "UNRESOLVED",
                    "start_frame": gap_start,
                    "end_frame": gap_end,
                    "start_time": gap_start / fps,
                    "end_time": (gap_end + 1) / fps,
                    "raw_track_id": None,
                    "stitching_source": "AUTO_STITCH_GAP",
                    "candidate_score_breakdown": {
                        "cost": sel.get("cost"),
                        "temporal_gap_frames": sel.get("temporal_gap_frames"),
                        "center_displacement_px": sel.get("center_displacement_px"),
                    },
                    "provenance": {"bridged_by_auto_stitch": True},
                    "confidence_class": "bridged_gap",
                    "analysis_eligible": False,
                }
            )
        intervals.append(
            {
                "interval_id": f"tiv_stitch_{nxt}",
                "kind": "AUTO_STITCHED_CONTINUATION",
                "start_frame": int(tr["first_frame"]),
                "end_frame": int(tr["last_frame"]),
                "start_time": int(tr["first_frame"]) / fps,
                "end_time": (int(tr["last_frame"]) + 1) / fps,
                "raw_track_id": str(nxt),
                "stitching_source": "AUTO_STITCHED_TRACKLET",
                "candidate_score_breakdown": {
                    "cost": sel.get("cost"),
                    "temporal_gap_frames": sel.get("temporal_gap_frames"),
                    "center_displacement_px": sel.get("center_displacement_px"),
                    "bbox_scale_ratio": sel.get("bbox_scale_ratio"),
                    "margin": (ev or {}).get("decision", {}).get("margin"),
                },
                "provenance": {
                    "previous_raw_track_id": str(prev),
                    "event_id": (ev or {}).get("event_id"),
                },
                "confidence_class": "auto_stitch_conservative",
                "analysis_eligible": True,
            }
        )

    # Trailing unresolved after last chain track if last event was unresolved
    if stitch_events:
        last = stitch_events[-1]
        dec = (last.get("decision") or {}).get("decision")
        if dec in {"TARGET_UNRESOLVED", "LONG_GAP_REVIEW_REQUIRED"}:
            prev = str(last["previous_raw_track_id"])
            start = int(track_index[prev]["last_frame"]) + 1
            # open-ended short window marker (does not claim full video coverage)
            end = start + 30
            intervals.append(
                {
                    "interval_id": f"tiv_tail_{prev}",
                    "kind": "REVIEW_REQUIRED" if dec == "LONG_GAP_REVIEW_REQUIRED" else "UNRESOLVED",
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / fps,
                    "end_time": (end + 1) / fps,
                    "raw_track_id": None,
                    "stitching_source": dec,
                    "candidate_score_breakdown": {
                        "reason": (last.get("decision") or {}).get("reason"),
                        "best": (last.get("decision") or {}).get("best_candidate"),
                        "runner_up": (last.get("decision") or {}).get("runner_up"),
                        "margin": (last.get("decision") or {}).get("margin"),
                    },
                    "provenance": {"event_id": last.get("event_id")},
                    "confidence_class": "unresolved",
                    "analysis_eligible": False,
                }
            )

    eligible = [i for i in intervals if i.get("analysis_eligible")]
    stitched_frames = sum(
        int(i["end_frame"]) - int(i["start_frame"]) + 1
        for i in eligible
        if i["kind"] in {"HUMAN_CONFIRMED", "AUTO_STITCHED_CONTINUATION", "RAW_TRACK_CONTINUATION"}
    )
    baseline_frames = int(seed["last_frame"]) - int(seed["first_frame"]) + 1

    return {
        "schema_version": "target_tracking_r1_timeline_v1",
        "persistent_target_id": persistent_target_id,
        "target_id": target_id,
        "intervals": intervals,
        "structural_metrics": {
            "target_raw_tracks_linked": len(chain_raw_track_ids),
            "automatic_stitch_count": max(0, len(chain_raw_track_ids) - 1),
            "baseline_seed_duration_frames": baseline_frames,
            "baseline_seed_duration_sec": baseline_frames / fps,
            "stitched_timeline_duration_frames": stitched_frames,
            "stitched_timeline_duration_sec": stitched_frames / fps,
            "longest_linked_chain": len(chain_raw_track_ids),
        },
        "full_metrics": {
            "target_idf1": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "target_recall": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "false_identity_switch_rate": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "correctness_percentage": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
        },
        "human_acceptance": "HUMAN_VISUAL_ACCEPTANCE_PENDING",
    }
