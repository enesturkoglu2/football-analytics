"""Target timeline schemas and enums (HIL-C)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from football_analytics.reid.hil.common import (
    HilValidationError,
    require_bool,
    require_int,
    require_mapping,
    require_sha256,
    require_str,
    validate_no_path_traversal,
)

TIMELINE_SCHEMA_VERSION = "target_timeline_v1"
DECISION_SOURCE_MANIFEST_SCHEMA = "target_timeline_decision_source_manifest_v1"
GENERATOR_VERSION = "hil_c_timeline_reconstructor_v1"


class TimelineError(HilValidationError):
    """Raised when timeline construction/validation fails."""


class IntervalStatus(str, Enum):
    HUMAN_CONFIRMED = "human_confirmed"
    TRACKER_CONTINUATION = "tracker_continuation"
    APPEARANCE_SUPPORTED = "appearance_supported"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"
    INVALID = "invalid"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"


class TimelineStatus(str, Enum):
    OK = "ok"
    CONFLICTED = "conflicted"
    NO_APPROVED_PRODUCT_DECISIONS = "no_approved_product_decisions"
    EMPTY = "empty"


class DecisionSourceClass(str, Enum):
    PRODUCT_APPROVED = "PRODUCT_APPROVED"
    PRODUCT_UNQUALIFIED_TEST_DECISION = "PRODUCT_UNQUALIFIED_TEST_DECISION"
    ACCEPTANCE_ISOLATED = "ACCEPTANCE_ISOLATED"
    FIXTURE_DEMO = "FIXTURE_DEMO"
    REVOKED_OR_SUPERSEDED = "REVOKED_OR_SUPERSEDED"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"


DEFAULT_ANALYSIS_ELIGIBLE = {
    IntervalStatus.HUMAN_CONFIRMED,
    IntervalStatus.TRACKER_CONTINUATION,
}


def analysis_eligible_for_status(status: IntervalStatus | str) -> bool:
    value = IntervalStatus(status) if not isinstance(status, IntervalStatus) else status
    # appearance_supported alone is never analysis-eligible in HIL-C
    return value in DEFAULT_ANALYSIS_ELIGIBLE


def validate_interval(row: Mapping[str, Any]) -> dict[str, Any]:
    data = require_mapping(row, field="interval")
    status = IntervalStatus(require_str(data.get("status"), field="status"))
    start = require_int(data.get("start_frame"), field="start_frame", min_value=0)
    end = require_int(data.get("end_frame"), field="end_frame", min_value=0)
    if end < start:
        raise TimelineError("end_frame < start_frame")
    analysis_eligible = require_bool(
        data.get("analysis_eligible"), field="analysis_eligible"
    )
    expected = analysis_eligible_for_status(status)
    if status == IntervalStatus.APPEARANCE_SUPPORTED and analysis_eligible:
        raise TimelineError("appearance_supported must not be analysis_eligible")
    if analysis_eligible and not expected and status not in {
        IntervalStatus.HUMAN_CONFIRMED,
        IntervalStatus.TRACKER_CONTINUATION,
    }:
        raise TimelineError(f"{status.value} must not be analysis_eligible")
    return {
        "interval_id": require_str(data.get("interval_id"), field="interval_id"),
        "target_id": require_str(data.get("target_id"), field="target_id"),
        "segment_id": data.get("segment_id"),
        "raw_track_id": data.get("raw_track_id"),
        "start_frame": start,
        "end_frame": end,
        "start_time_seconds": float(data.get("start_time_seconds", 0.0)),
        "end_time_seconds": float(data.get("end_time_seconds", 0.0)),
        "duration_seconds": float(data.get("duration_seconds", 0.0)),
        "status": status.value,
        "evidence_source": require_str(
            data.get("evidence_source"), field="evidence_source", allow_empty=True
        ),
        "source_event_ids": list(data.get("source_event_ids") or []),
        "source_decision_ids": list(data.get("source_decision_ids") or []),
        "tracker_provenance": dict(data.get("tracker_provenance") or {}),
        "bbox_observation_count": int(data.get("bbox_observation_count") or 0),
        "first_bbox": data.get("first_bbox"),
        "last_bbox": data.get("last_bbox"),
        "confidence_class": data.get("confidence_class"),
        "analysis_eligible": analysis_eligible,
        "exclusion_reason": data.get("exclusion_reason"),
        "superseded_by": data.get("superseded_by"),
        "metadata": dict(data.get("metadata") or {}),
        "requires_calibration": bool(data.get("requires_calibration", False)),
    }


def validate_timeline(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = require_mapping(payload, field="timeline")
    if require_str(row.get("schema_version"), field="schema_version") != TIMELINE_SCHEMA_VERSION:
        raise TimelineError("invalid timeline schema_version")
    status = TimelineStatus(require_str(row.get("timeline_status"), field="timeline_status"))
    intervals = [validate_interval(i) for i in (row.get("intervals") or [])]
    unresolved = [validate_interval(i) for i in (row.get("unresolved_intervals") or [])]
    excluded = [validate_interval(i) for i in (row.get("excluded_intervals") or [])]
    return {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline_id": require_str(row.get("timeline_id"), field="timeline_id"),
        "project_id": require_str(row.get("project_id"), field="project_id"),
        "run_id": require_str(row.get("run_id"), field="run_id"),
        "target_id": require_str(row.get("target_id"), field="target_id"),
        "video_id": require_str(row.get("video_id"), field="video_id"),
        "video_path": validate_no_path_traversal(row.get("video_path"), field="video_path"),
        "video_sha256": require_sha256(row.get("video_sha256"), field="video_sha256"),
        "source_segment_manifest_path": row.get("source_segment_manifest_path"),
        "source_segment_manifest_sha256": row.get("source_segment_manifest_sha256"),
        "decision_source_manifest_path": row.get("decision_source_manifest_path"),
        "decision_source_manifest_sha256": row.get("decision_source_manifest_sha256"),
        "decision_log_paths": list(row.get("decision_log_paths") or []),
        "decision_log_sha256": dict(row.get("decision_log_sha256") or {}),
        "generated_at": require_str(row.get("generated_at"), field="generated_at"),
        "generator_version": require_str(row.get("generator_version"), field="generator_version"),
        "timeline_status": status.value,
        "frame_rate": float(row.get("frame_rate") or 0.0),
        "total_video_frames": int(row.get("total_video_frames") or 0),
        "total_video_duration_seconds": float(row.get("total_video_duration_seconds") or 0.0),
        "intervals": intervals,
        "unresolved_intervals": unresolved,
        "excluded_intervals": excluded,
        "conflicts": list(row.get("conflicts") or []),
        "coverage_summary": dict(row.get("coverage_summary") or {}),
        "provenance": dict(row.get("provenance") or {}),
    }
