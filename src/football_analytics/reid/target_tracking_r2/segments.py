"""Derived clean segments over an immutable parent raw track."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence


def build_derived_segments(
    *,
    parent_raw_track_id: str,
    parent_first_frame: int,
    parent_last_frame: int,
    seed_frame: int,
    change_point: Mapping[str, Any] | None,
    evidence_rows: Sequence[Mapping[str, Any]],
    fps: float,
) -> dict[str, Any]:
    """Split parent track into seed-pure vs conflict segments without mutating raw track."""
    parent = str(parent_raw_track_id)
    if change_point is None:
        # Entire track treated cautiously: still mark as seed segment only if kit stays yellow
        seg_a = {
            "segment_id": f"raw{parent}_seg_A_{uuid.uuid4().hex[:6]}",
            "parent_raw_track_id": parent,
            "start_frame": int(parent_first_frame),
            "end_frame": int(parent_last_frame),
            "start_time": int(parent_first_frame) / fps,
            "end_time": (int(parent_last_frame) + 1) / fps,
            "dominant_kit_state": "YELLOW",
            "purity_state": "ASSUMED_PURE_NO_CHANGEPOINT",
            "contamination_state": "UNKNOWN",
            "target_eligibility": "TARGET_SEED_SEGMENT",
            "contains_seed_frame": True,
            "evidence": {"change_point": None},
        }
        return {
            "schema_version": "target_tracking_r2_derived_segments_v1",
            "parent_raw_track_id": parent,
            "parent_immutable": True,
            "segments": [seg_a],
            "seed_segment_id": seg_a["segment_id"],
            "conflict_segment_ids": [],
        }

    cp_f = int(change_point["change_point_frame"])
    # Seed segment ends at frame before change point
    seed_end = max(int(parent_first_frame), cp_f - 1)
    if seed_frame > seed_end:
        # safety: ensure seed frame included
        seed_end = int(seed_frame)

    # Dominant kit on seed side from evidence
    seed_kits = [
        r.get("kit_state")
        for r in evidence_rows
        if int(parent_first_frame) <= int(r["frame_index"]) <= seed_end and r.get("reliable")
    ]
    conflict_kits = [
        r.get("kit_state")
        for r in evidence_rows
        if int(r["frame_index"]) >= cp_f and r.get("reliable")
    ]

    seg_a = {
        "segment_id": f"raw{parent}_seg_A",
        "parent_raw_track_id": parent,
        "start_frame": int(parent_first_frame),
        "end_frame": int(seed_end),
        "start_time": int(parent_first_frame) / fps,
        "end_time": (int(seed_end) + 1) / fps,
        "dominant_kit_state": "YELLOW",
        "purity_state": "PURE_RELATIVE_TO_SEED",
        "contamination_state": "PRE_CONFLICT",
        "target_eligibility": "TARGET_SEED_SEGMENT",
        "contains_seed_frame": int(parent_first_frame) <= int(seed_frame) <= int(seed_end),
        "evidence": {
            "change_point": dict(change_point),
            "reliable_kit_labels": seed_kits[:20],
        },
    }
    seg_b = {
        "segment_id": f"raw{parent}_seg_B",
        "parent_raw_track_id": parent,
        "start_frame": int(cp_f),
        "end_frame": int(parent_last_frame),
        "start_time": int(cp_f) / fps,
        "end_time": (int(parent_last_frame) + 1) / fps,
        "dominant_kit_state": "WHITE",
        "purity_state": "IMPURE_CROSS_TEAM",
        "contamination_state": "IDENTITY_CONFLICT",
        "target_eligibility": "TARGET_INELIGIBLE_IDENTITY_CONFLICT",
        "contains_seed_frame": False,
        "analysis_eligible": False,
        "reason": "CROSS_TEAM_IDENTITY_CONFLICT",
        "evidence": {
            "change_point": dict(change_point),
            "reliable_kit_labels": conflict_kits[:20],
        },
    }
    assert seg_a["contains_seed_frame"], "seed frame must lie in TARGET_SEED_SEGMENT"
    return {
        "schema_version": "target_tracking_r2_derived_segments_v1",
        "parent_raw_track_id": parent,
        "parent_immutable": True,
        "parent_first_frame": int(parent_first_frame),
        "parent_last_frame": int(parent_last_frame),
        "segments": [seg_a, seg_b],
        "seed_segment_id": seg_a["segment_id"],
        "conflict_segment_ids": [seg_b["segment_id"]],
        "clean_seed_duration_frames": int(seed_end) - int(parent_first_frame) + 1,
        "excluded_impure_duration_frames": int(parent_last_frame) - int(cp_f) + 1,
        "original_raw_track_duration_frames": int(parent_last_frame)
        - int(parent_first_frame)
        + 1,
    }
