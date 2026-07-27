"""Overlap/conflict detection for target timelines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def find_interval_overlaps(
    intervals: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
) -> list[dict[str, Any]]:
    """Fail-closed overlap detection for same-target occupancy."""
    rows = [
        i
        for i in intervals
        if i.get("target_id") == target_id
        and i.get("status") in {"human_confirmed", "tracker_continuation"}
    ]
    rows = sorted(rows, key=lambda r: (int(r["start_frame"]), int(r["end_frame"]), r["interval_id"]))
    conflicts: list[dict[str, Any]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if int(b["start_frame"]) > int(a["end_frame"]):
                break
            # overlap if ranges intersect
            if int(a["start_frame"]) <= int(b["end_frame"]) and int(b["start_frame"]) <= int(
                a["end_frame"]
            ):
                if a.get("segment_id") == b.get("segment_id") and a.get("raw_track_id") == b.get(
                    "raw_track_id"
                ):
                    continue
                conflicts.append(
                    {
                        "conflict_type": "overlapping_target_segments",
                        "target_id": target_id,
                        "interval_a": a["interval_id"],
                        "interval_b": b["interval_id"],
                        "segment_a": a.get("segment_id"),
                        "segment_b": b.get("segment_id"),
                        "raw_track_a": a.get("raw_track_id"),
                        "raw_track_b": b.get("raw_track_id"),
                        "overlap_start_frame": max(int(a["start_frame"]), int(b["start_frame"])),
                        "overlap_end_frame": min(int(a["end_frame"]), int(b["end_frame"])),
                    }
                )
    return conflicts


def find_dual_target_segment_bindings(
    intervals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_seg: dict[str, set[str]] = {}
    for row in intervals:
        seg = row.get("segment_id")
        tid = row.get("target_id")
        if not seg or not tid:
            continue
        if row.get("status") not in {"human_confirmed", "tracker_continuation"}:
            continue
        by_seg.setdefault(str(seg), set()).add(str(tid))
    conflicts = []
    for seg, targets in sorted(by_seg.items()):
        if len(targets) > 1:
            conflicts.append(
                {
                    "conflict_type": "segment_bound_to_multiple_targets",
                    "segment_id": seg,
                    "target_ids": sorted(targets),
                }
            )
    return conflicts
