"""target_identity_ground_truth_v1 schema builders and validators."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from football_analytics.reid.golden_clip import SCHEMA_ANNOTATION_EVENT, SCHEMA_GT, TARGET_STATES
from football_analytics.reid.hil.common import HilValidationError, require_bool, require_int, require_str


class GoldenClipError(HilValidationError):
    """Raised when golden-clip GT contracts fail."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_annotation_id() -> str:
    return f"ann_{uuid.uuid4().hex[:12]}"


def new_session_id() -> str:
    return f"gtsess_{uuid.uuid4().hex[:12]}"


def frames_to_time(frame: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame) / float(fps)


def validate_target_state(value: Any) -> str:
    state = require_str(value, field="target_state")
    if state not in TARGET_STATES:
        raise GoldenClipError(f"invalid target_state: {state!r}")
    return state


def build_annotation_interval(
    *,
    annotation_id: str | None = None,
    start_frame: int,
    end_frame: int,
    fps: float,
    target_state: str,
    associated_detection_ids: Sequence[str] | None = None,
    associated_raw_track_ids: Sequence[str] | None = None,
    associated_segment_ids: Sequence[str] | None = None,
    bbox_observations: Sequence[Mapping[str, Any]] | None = None,
    true_target_bbox_observations: Sequence[Mapping[str, Any]] | None = None,
    occlusion_state: str | None = None,
    visibility_confidence: str = "medium",
    reviewer_comment: str = "",
    supersedes_annotation_id: str | None = None,
    active: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one annotation interval.

    Semantics for WRONG_TARGET_ASSIGNED:
    - associated_* fields refer to the *wrongly assigned* tracker/detection.
    - true_target_bbox_observations (optional) hold the true target location
      for overlay/review when separately known.
    """
    start = require_int(start_frame, field="start_frame", min_value=0)
    end = require_int(end_frame, field="end_frame", min_value=0)
    if end < start:
        raise GoldenClipError(f"end_frame ({end}) < start_frame ({start})")
    state = validate_target_state(target_state)
    aid = annotation_id or new_annotation_id()
    return {
        "annotation_id": aid,
        "start_frame": start,
        "end_frame": end,
        "start_time": frames_to_time(start, fps),
        "end_time": frames_to_time(end + 1, fps),
        "target_state": state,
        "associated_detection_ids": [str(x) for x in (associated_detection_ids or [])],
        "associated_raw_track_ids": [str(x) for x in (associated_raw_track_ids or [])],
        "associated_segment_ids": [str(x) for x in (associated_segment_ids or [])],
        "bbox_observations": [dict(x) for x in (bbox_observations or [])],
        "true_target_bbox_observations": [
            dict(x) for x in (true_target_bbox_observations or [])
        ],
        "occlusion_state": occlusion_state,
        "visibility_confidence": visibility_confidence,
        "reviewer_comment": reviewer_comment or "",
        "supersedes_annotation_id": supersedes_annotation_id,
        "active": require_bool(active, field="active"),
        "provenance": dict(provenance or {}),
    }


def validate_annotation_interval(row: Mapping[str, Any], *, fps: float) -> dict[str, Any]:
    require_str(row.get("annotation_id"), field="annotation_id")
    start = require_int(row.get("start_frame"), field="start_frame", min_value=0)
    end = require_int(row.get("end_frame"), field="end_frame", min_value=0)
    if end < start:
        raise GoldenClipError("end_frame < start_frame")
    validate_target_state(row.get("target_state"))
    require_bool(row.get("active"), field="active")
    for key in (
        "associated_detection_ids",
        "associated_raw_track_ids",
        "associated_segment_ids",
        "bbox_observations",
        "true_target_bbox_observations",
    ):
        if key in row and row[key] is not None and not isinstance(row[key], list):
            raise GoldenClipError(f"{key} must be a list")
    out = dict(row)
    out["start_time"] = frames_to_time(start, fps)
    out["end_time"] = frames_to_time(end + 1, fps)
    return out


def build_annotation_event(
    *,
    action: str,
    interval: Mapping[str, Any],
    reviewer: str,
    annotation_session_id: str,
    source_video_sha256: str,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    event_id: str | None = None,
    comment: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_ANNOTATION_EVENT,
        "event_id": event_id or f"gtevt_{uuid.uuid4().hex[:12]}",
        "action": action,
        "created_at": _utc(),
        "reviewer": reviewer,
        "annotation_session_id": annotation_session_id,
        "source_video_sha256": source_video_sha256,
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "comment": comment,
        "interval": dict(interval),
    }


def empty_ground_truth(
    *,
    source_video_sha256: str,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    annotation_session_id: str,
    reviewer: str,
    frame_count: int,
    fps: float,
    revision: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_GT,
        "source_video_sha256": source_video_sha256,
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "annotation_session_id": annotation_session_id,
        "reviewer": reviewer,
        "created_at": _utc(),
        "revision": int(revision),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "accepted": False,
        "intervals": [],
    }


def active_intervals(gt: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(iv) for iv in (gt.get("intervals") or []) if iv.get("active") is True]
