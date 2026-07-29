"""Build target_short_occlusion_state_v1 from clean seed observations only."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.target_tracking_r3.policy import R3_POLICY


def _center(b: Sequence[float]) -> tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _scale(b: Sequence[float]) -> float:
    return max(1.0, float(b[2]) - float(b[0])) * max(1.0, float(b[3]) - float(b[1]))


def _contam_coverage(row: Mapping[str, Any]) -> float:
    c = row.get("contamination") or {}
    return float(c.get("union_other_person_crop_coverage") or 0.0)


def select_bridge_template_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    seed_start: int,
    refined_seed_end: int,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Last quality-gated yellow/unknown rows; exclude white/impure."""
    pol = dict(policy or R3_POLICY)
    max_hist = int(pol["bridge_history_frames"])
    max_contam = float(pol["max_contam_coverage_for_template"])
    eligible: list[dict[str, Any]] = []
    for r in evidence_rows:
        fi = int(r["frame_index"])
        if fi < int(seed_start) or fi > int(refined_seed_end):
            continue
        kit = str(r.get("kit_state") or "UNKNOWN")
        if kit == "WHITE":
            continue  # never include impure/white in bridge template
        if not (r.get("crop_quality_ok") or r.get("reliable")):
            continue
        if _contam_coverage(r) > max_contam:
            continue
        if float(r.get("bbox_area") or 0) < float(pol["min_crop_area_px"]):
            continue
        lap = float(r.get("laplacian_variance") or 0)
        if lap and lap < float(pol["min_laplacian_variance"]):
            continue
        eligible.append(dict(r))
    return eligible[-max_hist:]


def build_short_occlusion_state(
    *,
    persistent_target_id: str,
    source_segment_id: str,
    parent_raw_track_id: str,
    template_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble target_short_occlusion_state_v1."""
    pol = dict(policy or R3_POLICY)
    if not template_rows:
        return {
            "schema_version": "target_short_occlusion_state_v1",
            "persistent_target_id": persistent_target_id,
            "source_segment": source_segment_id,
            "parent_raw_track_id": str(parent_raw_track_id),
            "bridge_status": "BRIDGE_STATE_EMPTY",
            "contamination_state": "UNKNOWN",
            "provenance": "CLEAN_SEED_SEGMENT_ONLY",
            "error": "no_quality_gated_template_rows",
        }

    last = template_rows[-1]
    centers = [_center(r["bbox_xyxy"]) for r in template_rows]
    scales = [_scale(r["bbox_xyxy"]) for r in template_rows]
    velocities: list[tuple[float, float]] = []
    for i in range(1, len(centers)):
        velocities.append(
            (centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1])
        )
    if velocities:
        vx = sum(v[0] for v in velocities) / len(velocities)
        vy = sum(v[1] for v in velocities) / len(velocities)
    else:
        vx = vy = 0.0
    scale_trend = 0.0
    if len(scales) >= 2:
        scale_trend = (scales[-1] - scales[0]) / max(1, len(scales) - 1)

    kits = [str(r.get("kit_state") or "UNKNOWN") for r in template_rows]
    yellow_vals = [
        float(r["yellow_evidence"])
        for r in template_rows
        if r.get("yellow_evidence") is not None
    ]
    torso = {
        "dominant_kit_state": "YELLOW"
        if kits.count("YELLOW") >= max(1, len(kits) // 2)
        else (kits[-1] if kits else "UNKNOWN"),
        "median_yellow_evidence": (
            sorted(yellow_vals)[len(yellow_vals) // 2] if yellow_vals else None
        ),
        "kit_labels": kits,
    }

    return {
        "schema_version": "target_short_occlusion_state_v1",
        "persistent_target_id": persistent_target_id,
        "source_segment": source_segment_id,
        "parent_raw_track_id": str(parent_raw_track_id),
        "last_reliable_frame": int(last["frame_index"]),
        "last_reliable_time_sec": float(last["timestamp_sec"]),
        "last_reliable_bbox": [float(v) for v in last["bbox_xyxy"]],
        "recent_center_trajectory": [
            {"frame_index": int(r["frame_index"]), "center_xy": list(_center(r["bbox_xyxy"]))}
            for r in template_rows
        ],
        "velocity_estimate_px_per_frame": {"vx": float(vx), "vy": float(vy)},
        "bbox_scale_trend": float(scale_trend),
        "quality_gated_crop_refs": [
            {
                "frame_index": int(r["frame_index"]),
                "crop_sha256": r.get("crop_sha256"),
                "bbox_xyxy": [float(v) for v in r["bbox_xyxy"]],
                "kit_state": r.get("kit_state"),
            }
            for r in template_rows
        ],
        "torso_kit_descriptor": torso,
        "contamination_state": "CLEAN_TEMPLATE",
        "bridge_status": "READY",
        "provenance": "CLEAN_SEED_SEGMENT_ONLY",
        "impure_observations_excluded": True,
        "policy_history_frames": int(pol["bridge_history_frames"]),
    }
