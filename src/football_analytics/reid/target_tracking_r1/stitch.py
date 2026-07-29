"""Conservative local tracklet stitching decisions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from football_analytics.reid.target_tracking_r1.candidates import (
    generate_continuation_candidates,
)
from football_analytics.reid.target_tracking_r1.policy import STITCH_POLICY
from football_analytics.reid.target_tracking_r1.state import append_observation


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decide_stitch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return AUTO_STITCH, UNRESOLVED, or LONG_GAP_REVIEW_REQUIRED decision."""
    pol = dict(policy or STITCH_POLICY)
    passed = [c for c in candidates if c.get("hard_gates_passed")]
    if not passed:
        return {
            "decision": "TARGET_UNRESOLVED",
            "reason": "no_hard_gate_passing_candidate",
            "selected": None,
            "runner_up": None,
            "margin": None,
            "created_at": _utc(),
        }

    # Long-gap only candidates → review required (should already be filtered by max_local_gap)
    best = passed[0]
    second = passed[1] if len(passed) > 1 else None
    margin = None
    if second is not None:
        margin = float(second["cost"]) - float(best["cost"])

    auto_ok = True
    reasons: list[str] = []
    if int(best["temporal_gap_frames"]) > int(pol["max_auto_stitch_gap_frames"]):
        auto_ok = False
        reasons.append("gap_exceeds_auto_stitch_window")
    if float(best["center_displacement_px"]) > float(pol["max_auto_center_displacement_px"]):
        auto_ok = False
        reasons.append("displacement_exceeds_auto_threshold")
    if float(best["cost"]) > float(pol["max_auto_cost"]):
        auto_ok = False
        reasons.append("cost_above_max_auto_cost")
    if second is not None and (margin is None or margin < float(pol["min_score_margin"])):
        auto_ok = False
        reasons.append("insufficient_margin_vs_runner_up")
    # Conflicting near-ties: another candidate with similar cost and different place
    if second is not None and margin is not None and margin < float(pol["min_score_margin"]):
        auto_ok = False
    # Appearance unavailable → identity proof is geometry-only; still allow if strict gates pass
    if pol.get("appearance_evidence") == "UNAVAILABLE":
        # already enforced via stricter geometric thresholds
        pass

    if not auto_ok:
        decision = "TARGET_UNRESOLVED"
        if int(best["temporal_gap_frames"]) > int(pol["max_local_gap_frames"]):
            decision = "LONG_GAP_REVIEW_REQUIRED"
        return {
            "decision": decision,
            "reason": ",".join(reasons) or "ambiguous",
            "selected": None,
            "best_candidate": dict(best),
            "runner_up": dict(second) if second else None,
            "margin": margin,
            "created_at": _utc(),
            "identity_proof": "spatial_temporal_geometry_only",
            "appearance_evidence": "UNAVAILABLE",
        }

    return {
        "decision": "AUTO_STITCH",
        "reason": "hard_gates_passed_clear_margin_low_cost",
        "selected": dict(best),
        "runner_up": dict(second) if second else None,
        "margin": margin if second is not None else float("inf"),
        "created_at": _utc(),
        "identity_proof": "spatial_temporal_geometry_only",
        "appearance_evidence": "UNAVAILABLE",
    }


def run_local_stitch_chain(
    *,
    seed_raw_track_id: str,
    track_index: Mapping[str, Mapping[str, Any]],
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_width: int,
    frame_height: int,
    state: dict[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Greedy local chain from seed; stop at first unresolved / long gap."""
    pol = dict(policy or STITCH_POLICY)
    chain_track_ids = [str(seed_raw_track_id)]
    events: list[dict[str, Any]] = []
    current = str(seed_raw_track_id)
    stitches = 0

    while stitches < int(pol["max_chain_stitches"]):
        if current not in track_index:
            break
        cands = generate_continuation_candidates(
            previous_track_id=current,
            track_index=track_index,
            observations_by_frame=observations_by_frame,
            frame_width=frame_width,
            frame_height=frame_height,
            policy=pol,
        )
        decision = decide_stitch(cands, policy=pol)
        event = {
            "event_id": f"stitch_{uuid.uuid4().hex[:10]}",
            "previous_raw_track_id": current,
            "previous_end_frame": int(track_index[current]["last_frame"]),
            "candidates": cands,
            "decision": decision,
        }
        events.append(event)

        if decision["decision"] != "AUTO_STITCH" or not decision.get("selected"):
            # unresolved / long gap observation
            src = "UNRESOLVED_GAP"
            status = (
                "TARGET_REVIEW_REQUIRED"
                if decision["decision"] == "LONG_GAP_REVIEW_REQUIRED"
                else "TARGET_UNRESOLVED"
            )
            state = append_observation(
                state,
                {
                    "observation_id": f"tobs_{uuid.uuid4().hex[:12]}",
                    "source": src,
                    "status_hint": status,
                    "frame_index": int(track_index[current]["last_frame"]) + 1,
                    "raw_track_id": None,
                    "provenance": {
                        "kind": decision["decision"],
                        "reason": decision.get("reason"),
                        "best_candidate": (decision.get("best_candidate") or {}).get(
                            "candidate_raw_track_id"
                        ),
                        "margin": decision.get("margin"),
                    },
                },
            )
            break

        sel = decision["selected"]
        nxt = str(sel["candidate_raw_track_id"])
        chain_track_ids.append(nxt)
        state = append_observation(
            state,
            {
                "observation_id": f"tobs_{uuid.uuid4().hex[:12]}",
                "source": "AUTO_STITCHED_TRACKLET",
                "status_hint": "TARGET_TRACKING",
                "frame_index": int(sel["candidate_start_frame"]),
                "raw_track_id": nxt,
                "segment_id": sel.get("segment_id"),
                "provenance": {
                    "kind": "AUTO_STITCH",
                    "previous_raw_track_id": current,
                    "temporal_gap_frames": sel.get("temporal_gap_frames"),
                    "center_displacement_px": sel.get("center_displacement_px"),
                    "cost": sel.get("cost"),
                    "margin": decision.get("margin"),
                    "primary_evidence": [
                        "temporal_gap",
                        "center_displacement",
                        "scale_ratio",
                        "exact_frame_conflict_free",
                    ],
                },
            },
        )
        current = nxt
        stitches += 1
        # If no further candidates after this track, loop continues and may unresolved

    return {
        "chain_raw_track_ids": chain_track_ids,
        "automatic_stitch_count": stitches,
        "events": events,
        "state": state,
        "policy": pol,
    }
