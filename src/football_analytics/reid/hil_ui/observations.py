"""Read-only sparse observation lookup for review packages (no tracking)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_observation_lookup(
    path: str | Path | None,
    *,
    segment_ids: set[str],
    frame_min: int,
    frame_max: int,
) -> dict[tuple[str, int], list[float]]:
    """Load bbox observations for requested segments/frames only (fail-soft)."""
    if path is None:
        return {}
    obs_path = Path(path)
    if not obs_path.is_file():
        return {}
    out: dict[tuple[str, int], list[float]] = {}
    with obs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seg = str(row.get("segment_id") or "")
            if seg not in segment_ids:
                continue
            frame = int(row["frame_index"])
            if frame < frame_min or frame > frame_max:
                continue
            bbox = row.get("bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                out[(seg, frame)] = [float(v) for v in bbox]
    return out


def candidate_observation_audit(
    candidates: list[dict[str, Any]],
    observation_lookup: dict[tuple[str, int], list[float]],
) -> dict[str, Any]:
    rows = []
    for cand in candidates:
        frames_manifest = sorted(
            {int(r["frame_index"]) for r in cand.get("bbox_references") or []}
        )
        seg = cand["segment_id"]
        frames_obs = sorted(f for (s, f) in observation_lookup if s == seg)
        rows.append(
            {
                "candidate_id": cand["candidate_id"],
                "segment_id": seg,
                "raw_track_id": cand["raw_track_id"],
                "start_middle_end": [
                    cand.get("start_frame"),
                    cand.get("middle_frame"),
                    cand.get("end_frame"),
                ],
                "manifest_observation_frames": frames_manifest,
                "lookup_observation_frames": frames_obs,
                "sparse": len(frames_manifest) <= 2,
            }
        )
    return {
        "schema_version": "hil_b_r1_sparse_tracklet_observation_audit_v1",
        "candidates": rows,
        "note": "Sparse manifest refs are review metadata, not continuous tracking.",
    }
