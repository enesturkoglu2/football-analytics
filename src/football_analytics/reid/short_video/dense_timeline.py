"""Dense interactive_video_bbox_timeline_v1 builder (all player tracks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.short_video import SCHEMA_DENSE_TIMELINE


def build_dense_bbox_timeline(
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    video_sha256: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    include_ineligible: bool = True,
) -> dict[str, Any]:
    """Build time-indexed dense observations for every track observation."""
    observations_by_frame: dict[str, list[dict[str, Any]]] = {}
    track_coverage: list[dict[str, Any]] = []

    for row in mapping_rows:
        code = str(row["external_candidate_code"])
        raw = str(row.get("raw_external_track_id", row.get("raw_track_id")))
        seg = str(row.get("segment_id") or f"EXT_SEG_{code.replace('EXT_', '')}")
        eligible = bool(row.get("review_eligible", True))
        selectable = eligible or include_ineligible
        reason = "review_eligible" if eligible else "listed_but_below_quality_min"
        team = dict(row.get("team_metadata") or {"team_label": "unknown", "is_identity_proof": False})
        obs_list = list(row.get("bbox_per_observation") or [])
        track_coverage.append(
            {
                "raw_track_id": raw,
                "segment_id": seg,
                "external_candidate_code": code,
                "first_frame": int(row["first_frame"]),
                "last_frame": int(row["last_frame"]),
                "observation_count": len(obs_list),
                "observation_coverage": float(row.get("observation_coverage") or 0.0),
                "selectable": selectable,
            }
        )
        for item in obs_list:
            fi = int(item["frame_index"])
            key = str(fi)
            observations_by_frame.setdefault(key, []).append(
                {
                    "video_id": video_id,
                    "frame_index": fi,
                    "timestamp": fi / fps if fps else 0.0,
                    "detection_id": item.get("detection_id"),
                    "raw_track_id": raw,
                    "segment_id": seg,
                    "external_candidate_code": code,
                    "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                    "confidence": item.get("confidence"),
                    "selectable": selectable,
                    "eligibility_reason": reason,
                    "team_metadata": team,
                    "provenance": {
                        "source": "bytetrack_replay_from_yolo_detections",
                        "source_video_sha256": video_sha256,
                    },
                    "candidate_id": None,
                }
            )

    for key in observations_by_frame:
        observations_by_frame[key].sort(
            key=lambda r: (str(r["segment_id"]), str(r["raw_track_id"]))
        )

    return {
        "schema_version": SCHEMA_DENSE_TIMELINE,
        "video_id": video_id,
        "source_video_sha256": video_sha256,
        "fps": fps,
        "frame_count": frame_count,
        "resolution": {"width": width, "height": height},
        "track_count": len(mapping_rows),
        "frame_keys_with_observations": len(observations_by_frame),
        "observations_by_frame": observations_by_frame,
        "track_coverage": track_coverage,
        "legacy_ext_mapping_fallback_used": False,
        "game_state_executed": False,
        "clip_executed": False,
    }


def attach_candidate_ids_to_timeline(
    timeline: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    out = dict(timeline)
    by_seg = {str(c["segment_id"]): str(c["candidate_id"]) for c in candidates}
    obs = {}
    for fi, rows in (timeline.get("observations_by_frame") or {}).items():
        updated = []
        for row in rows:
            item = dict(row)
            item["candidate_id"] = by_seg.get(str(row.get("segment_id")))
            updated.append(item)
        obs[str(fi)] = updated
    out["observations_by_frame"] = obs
    return out


def observations_for_component(timeline: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Subset fields expected by the interactive video component."""
    out: dict[str, list[dict[str, Any]]] = {}
    for fi, rows in (timeline.get("observations_by_frame") or {}).items():
        out[str(fi)] = [
            {
                "bbox_xyxy": r["bbox_xyxy"],
                "segment_id": r["segment_id"],
                "raw_track_id": str(r["raw_track_id"]),
                "external_candidate_code": r.get("external_candidate_code"),
                "candidate_id": r.get("candidate_id"),
                "selectable": r.get("selectable", True),
            }
            for r in rows
            if r.get("selectable", True)
        ]
    return out


def write_dense_timeline(path: Path, timeline: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")


def load_dense_timeline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
