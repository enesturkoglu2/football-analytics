"""Segment provenance validation against frozen tracking artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.hil.timeline.schema import TimelineError


def load_segment_index(inventory_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(inventory_path)
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index[str(row["segment_id"])] = row
    return index


def load_first_last_bbox(
    observations_path: str | Path,
    *,
    segment_id: str,
) -> tuple[list[float] | None, list[float] | None, int]:
    path = Path(observations_path)
    first = None
    last = None
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("segment_id") != segment_id:
                continue
            bbox = row.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            count += 1
            if first is None:
                first = [float(v) for v in bbox]
            last = [float(v) for v in bbox]
    return first, last, count


def validate_decision_against_segment(
    decision: Mapping[str, Any],
    *,
    segment_index: Mapping[str, Mapping[str, Any]],
    expected_video_sha256: str | None = None,
) -> dict[str, Any]:
    segment_id = decision.get("selected_segment_id")
    if not segment_id:
        if decision.get("direct_bbox_selection"):
            return {
                "ok": False,
                "reason": "direct_bbox_without_segment_id_not_linkable_in_hil_c",
                "segment": None,
            }
        return {"ok": False, "reason": "missing_selected_segment_id", "segment": None}

    seg = segment_index.get(str(segment_id))
    if seg is None:
        raise TimelineError(
            f"BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH: unknown segment_id={segment_id}"
        )

    raw = decision.get("selected_raw_track_id")
    seg_raw = seg.get("raw_track_code") or seg.get("raw_track_id")
    if raw and seg_raw and str(raw) != str(seg_raw):
        raise TimelineError(
            "BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH: "
            f"raw_track mismatch decision={raw} segment={seg_raw}"
        )

    frame = decision.get("selected_frame_index")
    if frame is not None:
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        if not (start <= int(frame) <= end):
            raise TimelineError(
                "BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH: "
                f"selected_frame_index {frame} outside segment [{start},{end}]"
            )

    if expected_video_sha256 and decision.get("video_sha256"):
        if str(decision["video_sha256"]).lower() != expected_video_sha256.lower():
            raise TimelineError(
                "BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH: video_sha256 mismatch"
            )

    return {
        "ok": True,
        "reason": None,
        "segment": {
            "segment_id": seg["segment_id"],
            "raw_track_id": seg_raw,
            "start_frame": int(seg["start_frame"]),
            "end_frame": int(seg["end_frame"]),
            "observation_count": int(seg.get("observation_count") or 0),
        },
    }
