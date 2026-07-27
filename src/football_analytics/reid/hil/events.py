"""Target recovery event model and validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from football_analytics.reid.hil.common import (
    HilValidationError,
    reject_mutable_runtime_leak,
    require_bool,
    require_int,
    require_mapping,
    require_sha256,
    require_sha256_or_none,
    require_str,
    validate_no_path_traversal,
)

EVENT_SCHEMA_VERSION = "target_recovery_event_v1"


class EventError(HilValidationError):
    """Raised when a recovery event fails validation."""


class EventType(str, Enum):
    INITIAL_TARGET_ENROLLMENT = "INITIAL_TARGET_ENROLLMENT"
    TARGET_TRACK_CONTINUATION_UNCERTAIN = "TARGET_TRACK_CONTINUATION_UNCERTAIN"
    TARGET_LOST = "TARGET_LOST"
    TARGET_REENTRY_CANDIDATES_AVAILABLE = "TARGET_REENTRY_CANDIDATES_AVAILABLE"
    MULTIPLE_PLAUSIBLE_CANDIDATES = "MULTIPLE_PLAUSIBLE_CANDIDATES"
    NO_PLAUSIBLE_CANDIDATE = "NO_PLAUSIBLE_CANDIDATE"
    TRACK_IDENTITY_SWITCH_SUSPECTED = "TRACK_IDENTITY_SWITCH_SUSPECTED"
    MANUAL_CORRECTION_REQUIRED = "MANUAL_CORRECTION_REQUIRED"
    TARGET_CONFIRMED = "TARGET_CONFIRMED"
    TARGET_REJECTED = "TARGET_REJECTED"
    TARGET_DEFERRED = "TARGET_DEFERRED"


class EventStatus(str, Enum):
    OPEN = "open"
    DEFERRED = "deferred"
    CLOSED = "closed"


_TYPES_REQUIRING_WINDOW = {
    EventType.TARGET_LOST,
    EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE,
    EventType.MULTIPLE_PLAUSIBLE_CANDIDATES,
    EventType.NO_PLAUSIBLE_CANDIDATE,
    EventType.TARGET_TRACK_CONTINUATION_UNCERTAIN,
    EventType.TRACK_IDENTITY_SWITCH_SUSPECTED,
}

_TYPES_REQUIRING_MANIFEST = {
    EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE,
    EventType.MULTIPLE_PLAUSIBLE_CANDIDATES,
}


def parse_event_type(value: Any) -> EventType:
    if not isinstance(value, str):
        raise EventError(f"unknown or invalid event_type: {value!r}")
    try:
        return EventType(value)
    except ValueError as exc:
        raise EventError(f"unknown event_type rejected: {value!r}") from exc


def validate_recovery_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized recovery event mapping."""
    row = require_mapping(payload, field="event")
    reject_mutable_runtime_leak(row, field="event")

    if require_str(row.get("schema_version"), field="schema_version") != EVENT_SCHEMA_VERSION:
        raise EventError(
            f"schema_version must be {EVENT_SCHEMA_VERSION}, got {row.get('schema_version')!r}"
        )

    event_type = parse_event_type(row.get("event_type"))
    event_id = require_str(row.get("event_id"), field="event_id")
    project_id = require_str(row.get("project_id"), field="project_id")
    run_id = require_str(row.get("run_id"), field="run_id")
    target_id = require_str(row.get("target_id"), field="target_id")
    video_id = require_str(row.get("video_id"), field="video_id")
    video_path = validate_no_path_traversal(row.get("video_path"), field="video_path")
    video_sha256 = require_sha256(row.get("video_sha256"), field="video_sha256")
    created_at = require_str(row.get("created_at"), field="created_at")

    status_raw = require_str(row.get("status"), field="status")
    try:
        status = EventStatus(status_raw)
    except ValueError as exc:
        raise EventError(f"invalid event status: {status_raw!r}") from exc

    trigger_source = require_str(row.get("trigger_source"), field="trigger_source")
    trigger_reason = require_str(row.get("trigger_reason"), field="trigger_reason", allow_empty=True)

    last_confirmed_segment_id = row.get("last_confirmed_segment_id")
    if last_confirmed_segment_id is not None:
        last_confirmed_segment_id = require_str(
            last_confirmed_segment_id, field="last_confirmed_segment_id"
        )

    last_confirmed_frame_index = row.get("last_confirmed_frame_index")
    if last_confirmed_frame_index is not None:
        last_confirmed_frame_index = require_int(
            last_confirmed_frame_index, field="last_confirmed_frame_index", min_value=0
        )

    start = require_int(row.get("review_window_start_frame"), field="review_window_start_frame", min_value=0)
    end = require_int(row.get("review_window_end_frame"), field="review_window_end_frame", min_value=0)
    if end < start:
        raise EventError("review_window_end_frame must be >= review_window_start_frame")

    if event_type in _TYPES_REQUIRING_WINDOW and end == start and event_type != EventType.NO_PLAUSIBLE_CANDIDATE:
        # Zero-width window is allowed only when explicitly empty-search; otherwise warn via fail-closed for lost/reentry.
        if event_type in {
            EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE,
            EventType.MULTIPLE_PLAUSIBLE_CANDIDATES,
        }:
            raise EventError(f"{event_type.value} requires a non-empty review window")

    candidate_manifest_path = row.get("candidate_manifest_path")
    candidate_manifest_sha256 = row.get("candidate_manifest_sha256")
    candidate_count = require_int(row.get("candidate_count"), field="candidate_count", min_value=0)

    if event_type in _TYPES_REQUIRING_MANIFEST:
        candidate_manifest_path = validate_no_path_traversal(
            candidate_manifest_path, field="candidate_manifest_path"
        )
        candidate_manifest_sha256 = require_sha256(
            candidate_manifest_sha256, field="candidate_manifest_sha256"
        )
    else:
        if candidate_manifest_path is not None:
            candidate_manifest_path = validate_no_path_traversal(
                candidate_manifest_path, field="candidate_manifest_path"
            )
        candidate_manifest_sha256 = require_sha256_or_none(
            candidate_manifest_sha256, field="candidate_manifest_sha256"
        )

    if event_type == EventType.NO_PLAUSIBLE_CANDIDATE and candidate_count != 0:
        raise EventError("NO_PLAUSIBLE_CANDIDATE requires candidate_count == 0")

    if event_type == EventType.TARGET_LOST:
        if last_confirmed_segment_id is None or last_confirmed_frame_index is None:
            raise EventError("TARGET_LOST requires last_confirmed_segment_id and last_confirmed_frame_index")

    requires_calibration = require_bool(row.get("requires_calibration"), field="requires_calibration")

    evidence_paths = row.get("evidence_paths", [])
    if not isinstance(evidence_paths, list):
        raise EventError("evidence_paths must be a list")
    evidence_paths = [validate_no_path_traversal(p, field=f"evidence_paths[{i}]") for i, p in enumerate(evidence_paths)]

    evidence_sha256 = row.get("evidence_sha256", [])
    if not isinstance(evidence_sha256, list):
        raise EventError("evidence_sha256 must be a list")
    evidence_sha256 = [require_sha256(s, field=f"evidence_sha256[{i}]") for i, s in enumerate(evidence_sha256)]
    if len(evidence_paths) != len(evidence_sha256):
        raise EventError("evidence_paths and evidence_sha256 length mismatch")

    provenance = require_mapping(row.get("provenance", {}), field="provenance")
    metadata = require_mapping(row.get("metadata", {}), field="metadata")
    reject_mutable_runtime_leak(metadata, field="metadata")

    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "project_id": project_id,
        "run_id": run_id,
        "target_id": target_id,
        "event_type": event_type.value,
        "video_id": video_id,
        "video_path": video_path,
        "video_sha256": video_sha256,
        "created_at": created_at,
        "status": status.value,
        "trigger_source": trigger_source,
        "trigger_reason": trigger_reason,
        "last_confirmed_segment_id": last_confirmed_segment_id,
        "last_confirmed_frame_index": last_confirmed_frame_index,
        "review_window_start_frame": start,
        "review_window_end_frame": end,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_count": candidate_count,
        "requires_calibration": requires_calibration,
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_sha256,
        "provenance": dict(provenance),
        "metadata": dict(metadata),
    }


def validate_event_list_unique_ids(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in events:
        event = validate_recovery_event(raw)
        if event["event_id"] in seen:
            raise EventError(f"duplicate event_id: {event['event_id']}")
        seen.add(event["event_id"])
        validated.append(event)
    return validated
