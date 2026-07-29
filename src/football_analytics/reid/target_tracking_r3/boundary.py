"""R2 purity-boundary audit + algorithmic refinement (no human hard-code)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.target_tracking_r3.policy import R3_POLICY


def _contam_coverage(row: Mapping[str, Any]) -> float:
    c = row.get("contamination") or {}
    return float(c.get("union_other_person_crop_coverage") or 0.0)


def _contam_iou(row: Mapping[str, Any]) -> float:
    c = row.get("contamination") or {}
    return float(c.get("max_other_person_iou") or 0.0)


def audit_seed_segment_white_leak(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    seed_segment_start: int,
    seed_segment_end: int,
    r2_change_point_frame: int,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect reliable white / severe contamination inside R2 clean seed segment."""
    pol = dict(policy or R3_POLICY)
    white_in_seed: list[dict[str, Any]] = []
    severe_contam: list[dict[str, Any]] = []
    for r in evidence_rows:
        fi = int(r["frame_index"])
        if fi < int(seed_segment_start) or fi > int(seed_segment_end):
            continue
        kit = str(r.get("kit_state") or "UNKNOWN")
        reliable = bool(r.get("reliable") or r.get("crop_quality_ok"))
        if reliable and kit == "WHITE":
            white_in_seed.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": float(r["timestamp_sec"]),
                    "white_evidence": r.get("white_evidence"),
                    "yellow_evidence": r.get("yellow_evidence"),
                }
            )
        if _contam_coverage(r) >= float(pol["refine_severe_contam_coverage"]):
            severe_contam.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": float(r["timestamp_sec"]),
                    "coverage": _contam_coverage(r),
                    "iou": _contam_iou(r),
                    "kit_state": kit,
                }
            )
    return {
        "schema_version": "target_tracking_r3_boundary_audit_v1",
        "r2_seed_segment": [int(seed_segment_start), int(seed_segment_end)],
        "r2_change_point_frame": int(r2_change_point_frame),
        "reliable_white_frames_in_seed": white_in_seed,
        "severe_contamination_frames_in_seed": severe_contam,
        "refinement_required": bool(white_in_seed)
        or bool(severe_contam and white_in_seed),
        "white_player_continuation_painted_as_target_risk": bool(white_in_seed),
    }


def refine_purity_boundary(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    seed_frame: int,
    r2_change_point_frame: int,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Move purity boundary earlier using kit + overlap evidence (temporal confirm).

    Does not hard-code the human ~1.6s observation as the algorithm truth.
    """
    pol = dict(policy or R3_POLICY)
    confirm = int(pol["refine_confirm_frames"])
    w_high = float(pol["refine_white_frac_high"])
    collapse = float(pol["refine_yellow_frac_collapse"])
    contam_iou_min = float(pol["refine_contam_iou_min"])
    human_t = float(pol["human_reported_transition_sec_approx"])

    # Seed baseline yellow (same spirit as R2)
    radius = 8
    seed_yellow = [
        float(r["yellow_evidence"])
        for r in evidence_rows
        if abs(int(r["frame_index"]) - int(seed_frame)) <= radius
        and r.get("yellow_evidence") is not None
        and (r.get("crop_quality_ok") or r.get("reliable"))
    ]
    baseline_yellow = (
        sorted(seed_yellow)[len(seed_yellow) // 2] if seed_yellow else None
    )

    flags: list[dict[str, Any]] = []
    for r in evidence_rows:
        fi = int(r["frame_index"])
        if fi > int(r2_change_point_frame) + 5:
            break
        y = r.get("yellow_evidence")
        w = r.get("white_evidence")
        kit = str(r.get("kit_state") or "UNKNOWN")
        reliable = bool(r.get("crop_quality_ok") or r.get("reliable"))
        yellow_collapse = (
            reliable
            and y is not None
            and baseline_yellow is not None
            and float(y) < collapse * baseline_yellow
        )
        white_gain = reliable and w is not None and float(w) >= w_high
        white_kit = reliable and kit == "WHITE"
        contam = _contam_iou(r) >= contam_iou_min or _contam_coverage(r) >= float(
            pol["refine_severe_contam_coverage"]
        )
        signal = white_kit or (white_gain and (yellow_collapse or contam))
        flags.append(
            {
                "frame_index": fi,
                "timestamp_sec": float(r["timestamp_sec"]),
                "signal": bool(signal),
                "reliable": reliable,
                "kit_state": kit,
                "reasons": {
                    "white_kit": white_kit,
                    "white_gain": white_gain,
                    "yellow_collapse": yellow_collapse,
                    "contamination": contam,
                },
            }
        )

    refined = None
    i = 0
    while i < len(flags):
        if not flags[i]["signal"] or not flags[i]["reliable"]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]["signal"] and flags[j]["reliable"]:
            j += 1
        run = flags[i:j]
        if len(run) >= confirm and int(run[0]["frame_index"]) > int(seed_frame):
            refined = {
                "change_point_frame": int(run[0]["frame_index"]),
                "change_point_time_sec": float(run[0]["timestamp_sec"]),
                "confirmed_run_frames": len(run),
                "kind": "REFINED_YELLOW_TO_WHITE_OR_OVERLAP",
                "evidence_reasons": dict(run[0]["reasons"]),
            }
            break
        i = max(j, i + 1)

    r2_f = int(r2_change_point_frame)
    if refined is None:
        refined_f = r2_f
        source = "R2_UNCHANGED"
    else:
        refined_f = min(int(refined["change_point_frame"]), r2_f)
        source = "R3_REFINED" if refined_f < r2_f else "R2_ALIGNED"

    # Seed segment must still contain seed frame
    seed_end = max(int(seed_frame), refined_f - 1)

    return {
        "schema_version": "target_tracking_r3_boundary_refine_v1",
        "r2_change_point_frame": r2_f,
        "refined_change_point_frame": int(refined_f),
        "refined_change_point_time_sec": float(refined_f) / 30.0,  # overwritten by runner fps
        "refined_seed_segment_end_frame": int(seed_end),
        "refinement_source": source,
        "refined_detail": refined,
        "human_reported_transition_sec_approx": human_t,
        "human_reported_transition_is_algorithm_truth": False,
        "delta_refined_vs_human_sec": None,  # filled by runner with real fps
        "baseline_yellow_evidence": baseline_yellow,
        "policy_refine_confirm_frames": confirm,
    }
