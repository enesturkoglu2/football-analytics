"""Change-point detection inside an immutable raw track (no one-frame splits)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.target_tracking_r2.policy import R2_POLICY


def detect_kit_change_points(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    seed_frame: int,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Find sustained yellow→white (or seed-kit loss) change points.

    Human approx transition is reported separately and must not be hard-coded.
    """
    pol = dict(policy or R2_POLICY)
    confirm = int(pol["change_confirm_frames"])
    y_low = float(pol["yellow_frac_low"])
    w_high = float(pol["white_frac_high"])
    human_t = float(pol["human_reported_transition_sec_approx"])

    # Seed baseline yellow evidence
    radius = int(pol["seed_baseline_radius_frames"])
    seed_rows = [
        r
        for r in evidence_rows
        if abs(int(r["frame_index"]) - int(seed_frame)) <= radius and r.get("crop_quality_ok")
    ]
    seed_yellow = [
        float(r.get("yellow_evidence") or 0.0)
        for r in seed_rows
        if r.get("yellow_evidence") is not None
    ]
    baseline_yellow = (
        sorted(seed_yellow)[len(seed_yellow) // 2] if seed_yellow else None
    )

    # Candidate flags per frame
    flags: list[dict[str, Any]] = []
    for r in evidence_rows:
        y = r.get("yellow_evidence")
        w = r.get("white_evidence")
        kit = str(r.get("kit_state") or "UNKNOWN")
        cont = float((r.get("contamination") or {}).get("union_other_person_crop_coverage") or 0.0)
        reliable = bool(r.get("crop_quality_ok"))
        yellow_loss = (
            reliable
            and y is not None
            and (
                float(y) <= y_low
                or (baseline_yellow is not None and float(y) < 0.4 * baseline_yellow)
            )
        )
        white_gain = reliable and w is not None and float(w) >= w_high
        cross_team = kit == "WHITE" or (yellow_loss and white_gain)
        support = cont >= float(pol["contamination_support_min"]) or (
            r.get("nearby_player_overlap") or 0
        ) > 0
        flags.append(
            {
                "frame_index": int(r["frame_index"]),
                "timestamp_sec": float(r["timestamp_sec"]),
                "cross_team_signal": bool(cross_team),
                "yellow_loss": bool(yellow_loss),
                "white_gain": bool(white_gain),
                "kit_state": kit,
                "contamination_support": bool(support),
                "reliable": reliable,
            }
        )

    # Require confirm consecutive frames with cross_team_signal AND (contamination support
    # somewhere in window OR strong white dominant)
    change_points: list[dict[str, Any]] = []
    i = 0
    while i < len(flags):
        if not flags[i]["cross_team_signal"] or not flags[i]["reliable"]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]["cross_team_signal"] and flags[j]["reliable"]:
            j += 1
        run = flags[i:j]
        if len(run) >= confirm:
            # first frame of sustained run
            cont_ok = any(f["contamination_support"] for f in run) or all(
                f["kit_state"] == "WHITE" for f in run[:confirm]
            )
            if cont_ok:
                cp = run[0]
                change_points.append(
                    {
                        "change_point_frame": cp["frame_index"],
                        "change_point_time_sec": cp["timestamp_sec"],
                        "confirmed_run_frames": len(run),
                        "end_frame": run[-1]["frame_index"],
                        "kind": "YELLOW_TO_WHITE_CROSS_TEAM",
                        "error_classes": [
                            "RAW_TRACK_PURITY_FAILURE",
                            "CROSS_TEAM_IDENTITY_SWITCH",
                            "CROWD_OVERLAP",
                        ],
                    }
                )
                break  # first sustained change is the purity boundary for R2
        i = max(j, i + 1)

    algorithmic = change_points[0] if change_points else None
    return {
        "schema_version": "target_tracking_r2_changepoint_v1",
        "seed_frame": int(seed_frame),
        "baseline_yellow_evidence": baseline_yellow,
        "algorithmic_change_point": algorithmic,
        "all_change_points": change_points,
        "human_reported_transition_sec_approx": human_t,
        "human_reported_transition_is_algorithm_truth": False,
        "delta_algorithmic_vs_human_sec": (
            None
            if algorithmic is None
            else float(algorithmic["change_point_time_sec"]) - human_t
        ),
        "policy_confirm_frames": confirm,
    }
