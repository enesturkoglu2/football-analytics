"""Human decision schema and invariants for HIL recovery."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from football_analytics.reid.hil.candidates import candidate_ids, validate_candidate_manifest
from football_analytics.reid.hil.common import (
    HilValidationError,
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
    reject_mutable_runtime_leak,
    require_bool,
    require_int,
    require_mapping,
    require_sha256,
    require_sha256_or_none,
    require_str,
    validate_no_path_traversal,
)
from football_analytics.reid.schema import validate_bbox_xyxy

DECISION_SCHEMA_VERSION = "target_recovery_decision_v1"


class DecisionError(HilValidationError):
    """Raised when a human decision fails validation."""


class DecisionAction(str, Enum):
    CONFIRM_TARGET = "CONFIRM_TARGET"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    NONE_OF_THESE = "NONE_OF_THESE"
    UNKNOWN = "UNKNOWN"
    INVALID_SEGMENT = "INVALID_SEGMENT"
    DEFER = "DEFER"
    REVOKE = "REVOKE"
    CORRECT_PREVIOUS_DECISION = "CORRECT_PREVIOUS_DECISION"


class DecisionConfidence(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


_NO_SELECTION_ACTIONS = {
    DecisionAction.NONE_OF_THESE,
    DecisionAction.UNKNOWN,
    DecisionAction.DEFER,
    DecisionAction.INVALID_SEGMENT,
}


def parse_action(value: Any) -> DecisionAction:
    if not isinstance(value, str):
        raise DecisionError(f"invalid action: {value!r}")
    try:
        return DecisionAction(value)
    except ValueError as exc:
        raise DecisionError(f"unknown decision action rejected: {value!r}") from exc


def build_decision(
    *,
    decision_id: str,
    project_id: str,
    run_id: str,
    target_id: str,
    event_id: str,
    video_id: str,
    video_path: str,
    video_sha256: str,
    reviewer: str,
    created_at: str,
    revision: int,
    action: str | DecisionAction,
    confidence: str | DecisionConfidence = DecisionConfidence.UNKNOWN,
    status: str | DecisionStatus = DecisionStatus.ACTIVE,
    supersedes_decision_id: str | None = None,
    selected_candidate_id: str | None = None,
    selected_segment_id: str | None = None,
    selected_raw_track_id: str | None = None,
    selected_frame_index: int | None = None,
    selected_bbox_xyxy: list[float] | None = None,
    direct_bbox_selection: bool = False,
    candidate_manifest_path: str | None = None,
    candidate_manifest_sha256: str | None = None,
    displayed_model_id: str | None = None,
    displayed_checkpoint_sha256: str | None = None,
    displayed_rank: int | None = None,
    displayed_score: float | None = None,
    displayed_T_max: float | None = None,
    displayed_D_max: float | None = None,
    evidence_paths: list[str] | None = None,
    evidence_sha256: list[str] | None = None,
    comment: str = "",
    training_use_approved: bool = False,
    gallery_use_approved: bool = False,
) -> dict[str, Any]:
    """Build a decision mapping with defaults; always validate before append."""
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_id": decision_id,
        "project_id": project_id,
        "run_id": run_id,
        "target_id": target_id,
        "event_id": event_id,
        "video_id": video_id,
        "video_path": video_path,
        "video_sha256": video_sha256,
        "reviewer": reviewer,
        "created_at": created_at,
        "revision": revision,
        "supersedes_decision_id": supersedes_decision_id,
        "action": action.value if isinstance(action, DecisionAction) else action,
        "selected_candidate_id": selected_candidate_id,
        "selected_segment_id": selected_segment_id,
        "selected_raw_track_id": selected_raw_track_id,
        "selected_frame_index": selected_frame_index,
        "selected_bbox": selected_bbox_xyxy,
        "direct_bbox_selection": direct_bbox_selection,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "displayed_model_id": displayed_model_id,
        "displayed_checkpoint_sha256": displayed_checkpoint_sha256,
        "displayed_rank": displayed_rank,
        "displayed_score": displayed_score,
        "displayed_T_max": displayed_T_max,
        "displayed_D_max": displayed_D_max,
        "evidence_paths": list(evidence_paths or []),
        "evidence_sha256": list(evidence_sha256 or []),
        "comment": comment,
        "confidence": confidence.value
        if isinstance(confidence, DecisionConfidence)
        else confidence,
        "status": status.value if isinstance(status, DecisionStatus) else status,
        "training_use_approved": training_use_approved,
        "gallery_use_approved": gallery_use_approved,
        "model_auto_filled": False,
    }


def validate_decision(
    payload: Mapping[str, Any],
    *,
    event: Mapping[str, Any] | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
    known_decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row = require_mapping(payload, field="decision")
    reject_mutable_runtime_leak(row, field="decision")

    if require_str(row.get("schema_version"), field="schema_version") != DECISION_SCHEMA_VERSION:
        raise DecisionError(f"schema_version must be {DECISION_SCHEMA_VERSION}")

    # Model must never auto-fill human decision fields.
    if row.get("model_auto_filled") is True:
        raise DecisionError("model must not auto-fill human decisions")
    if "auto_confirm" in row or row.get("automatic_confirmation") is True:
        raise DecisionError("automatic confirmation is forbidden")

    decision_id = require_str(row.get("decision_id"), field="decision_id")
    project_id = require_str(row.get("project_id"), field="project_id")
    run_id = require_str(row.get("run_id"), field="run_id")
    target_id = require_str(row.get("target_id"), field="target_id")
    event_id = require_str(row.get("event_id"), field="event_id")
    video_id = require_str(row.get("video_id"), field="video_id")
    video_path = validate_no_path_traversal(row.get("video_path"), field="video_path")
    video_sha256 = require_sha256(row.get("video_sha256"), field="video_sha256")
    reviewer = require_str(row.get("reviewer"), field="reviewer")
    created_at = require_str(row.get("created_at"), field="created_at")
    revision = require_int(row.get("revision"), field="revision", min_value=1)

    supersedes = row.get("supersedes_decision_id")
    if supersedes is not None:
        supersedes = require_str(supersedes, field="supersedes_decision_id")

    action = parse_action(row.get("action"))

    confidence_raw = require_str(row.get("confidence"), field="confidence")
    try:
        confidence = DecisionConfidence(confidence_raw)
    except ValueError as exc:
        raise DecisionError(f"invalid confidence: {confidence_raw!r}") from exc

    status_raw = require_str(row.get("status"), field="status")
    try:
        status = DecisionStatus(status_raw)
    except ValueError as exc:
        raise DecisionError(f"invalid status: {status_raw!r}") from exc

    training_use_approved = require_bool(
        row.get("training_use_approved", False), field="training_use_approved"
    )
    gallery_use_approved = require_bool(
        row.get("gallery_use_approved", False), field="gallery_use_approved"
    )
    # Defaults must remain false unless explicitly approved in the record.
    # (Explicit true is allowed only as human flag; never inferred.)

    direct_bbox = require_bool(
        row.get("direct_bbox_selection", False), field="direct_bbox_selection"
    )
    selected_candidate_id = row.get("selected_candidate_id")
    selected_segment_id = row.get("selected_segment_id")
    selected_raw_track_id = row.get("selected_raw_track_id")
    selected_frame_index = row.get("selected_frame_index")
    selected_bbox = row.get("selected_bbox")
    if selected_bbox is None:
        selected_bbox = row.get("selected_bbox_xyxy")

    if selected_candidate_id is not None:
        selected_candidate_id = require_str(
            selected_candidate_id, field="selected_candidate_id"
        )
    if selected_segment_id is not None:
        selected_segment_id = require_str(selected_segment_id, field="selected_segment_id")
    if selected_raw_track_id is not None:
        selected_raw_track_id = require_str(
            selected_raw_track_id, field="selected_raw_track_id"
        )
    if selected_frame_index is not None:
        selected_frame_index = require_int(
            selected_frame_index, field="selected_frame_index", min_value=0
        )
    if selected_bbox is not None:
        selected_bbox = validate_bbox_xyxy(selected_bbox, field="selected_bbox")

    candidate_manifest_path = row.get("candidate_manifest_path")
    if candidate_manifest_path is not None:
        candidate_manifest_path = validate_no_path_traversal(
            candidate_manifest_path, field="candidate_manifest_path"
        )
    candidate_manifest_sha256 = require_sha256_or_none(
        row.get("candidate_manifest_sha256"), field="candidate_manifest_sha256"
    )

    displayed_model_id = row.get("displayed_model_id")
    displayed_checkpoint_sha256 = row.get("displayed_checkpoint_sha256")
    displayed_rank = row.get("displayed_rank")
    displayed_score = row.get("displayed_score")
    if displayed_score is None:
        displayed_score = row.get("displayed_score_S")
    displayed_T_max = row.get("displayed_T_max")
    displayed_D_max = row.get("displayed_D_max")

    if any(
        v is not None
        for v in (displayed_rank, displayed_score, displayed_T_max, displayed_D_max)
    ):
        displayed_model_id = require_str(displayed_model_id, field="displayed_model_id")
        if displayed_model_id != SPORTSREID_MODEL_ID:
            raise DecisionError(
                "displayed_model_id must be SportsReID helper model "
                f"({SPORTSREID_MODEL_ID}); no Market1501/R2B fallback"
            )
        displayed_checkpoint_sha256 = require_sha256(
            displayed_checkpoint_sha256, field="displayed_checkpoint_sha256"
        )
        if displayed_checkpoint_sha256 != SPORTSREID_CHECKPOINT_SHA256:
            raise DecisionError("displayed_checkpoint_sha256 mismatch for SportsReID helper")
        if displayed_rank is not None:
            displayed_rank = require_int(displayed_rank, field="displayed_rank", min_value=1)

    # Score fields are metadata only — never drive automatic confirmation.
    if row.get("action_from_score") is True or row.get("confirmed_by_model") is True:
        raise DecisionError("model rank/score must not determine human action")

    evidence_paths = row.get("evidence_paths", [])
    evidence_sha256 = row.get("evidence_sha256", row.get("evidence_shas", []))
    if not isinstance(evidence_paths, list) or not isinstance(evidence_sha256, list):
        raise DecisionError("evidence_paths/evidence_sha256 must be lists")
    evidence_paths = [
        validate_no_path_traversal(p, field=f"evidence_paths[{i}]")
        for i, p in enumerate(evidence_paths)
    ]
    evidence_sha256 = [
        require_sha256(s, field=f"evidence_sha256[{i}]") for i, s in enumerate(evidence_sha256)
    ]
    if len(evidence_paths) != len(evidence_sha256):
        raise DecisionError("evidence_paths and evidence_sha256 length mismatch")

    comment = require_str(row.get("comment", ""), field="comment", allow_empty=True)

    if event is not None:
        if event.get("event_id") != event_id:
            raise DecisionError("decision.event_id must match event.event_id")
        if event.get("target_id") != target_id:
            raise DecisionError("decision.target_id must match event.target_id")
        if event.get("project_id") != project_id:
            raise DecisionError("decision.project_id must match event.project_id")
        if event.get("video_sha256") != video_sha256:
            raise DecisionError("decision.video_sha256 must match event.video_sha256")

    # Action-specific requirements
    if action == DecisionAction.CONFIRM_TARGET:
        if not selected_segment_id and not (
            direct_bbox and selected_frame_index is not None and selected_bbox is not None
        ):
            raise DecisionError(
                "CONFIRM_TARGET requires selected_segment_id or direct bbox selection"
            )
        if direct_bbox:
            if selected_frame_index is None or selected_bbox is None:
                raise DecisionError(
                    "direct_bbox_selection requires selected_frame_index and selected_bbox"
                )
        elif selected_candidate_id is None:
            raise DecisionError(
                "CONFIRM_TARGET without direct_bbox_selection requires selected_candidate_id"
            )

    if action == DecisionAction.REJECT_CANDIDATE:
        if selected_candidate_id is None:
            raise DecisionError("REJECT_CANDIDATE requires selected_candidate_id")

    if action in _NO_SELECTION_ACTIONS:
        if selected_candidate_id is not None or selected_segment_id is not None:
            # Allow INVALID_SEGMENT to optionally point at a segment being invalidated.
            if action != DecisionAction.INVALID_SEGMENT:
                raise DecisionError(
                    f"{action.value} must not select a candidate/segment"
                )

    if action in {DecisionAction.REVOKE, DecisionAction.CORRECT_PREVIOUS_DECISION}:
        if supersedes is None:
            raise DecisionError(f"{action.value} requires supersedes_decision_id")

    if status in {DecisionStatus.SUPERSEDED, DecisionStatus.REVOKED}:
        # Raw historical rows may be marked superseded/revoked only via later writers.
        # New appends should normally be active; allow for fixture history rows.
        pass

    if candidate_manifest is not None:
        manifest = validate_candidate_manifest(candidate_manifest)
        ids = candidate_ids(manifest)
        if selected_candidate_id is not None and not direct_bbox:
            if selected_candidate_id not in ids:
                raise DecisionError(
                    "selected_candidate_id must exist in candidate manifest "
                    "when direct_bbox_selection is false"
                )
        if candidate_manifest_sha256 is None:
            raise DecisionError(
                "candidate_manifest_sha256 required when validating against a manifest"
            )

    if known_decisions is not None and supersedes is not None:
        prior = known_decisions.get(supersedes)
        if prior is None:
            raise DecisionError(f"supersedes_decision_id not found: {supersedes}")
        if prior.get("target_id") != target_id:
            raise DecisionError("cannot supersede a decision for another target")
        if prior.get("event_id") != event_id:
            raise DecisionError("cannot supersede a decision for another event")
        # Cycle detection: walk chain
        seen = {decision_id}
        cursor = supersedes
        while cursor is not None:
            if cursor in seen:
                raise DecisionError("supersedes chain cycle detected")
            seen.add(cursor)
            node = known_decisions.get(cursor)
            if node is None:
                break
            cursor = node.get("supersedes_decision_id")

    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_id": decision_id,
        "project_id": project_id,
        "run_id": run_id,
        "target_id": target_id,
        "event_id": event_id,
        "video_id": video_id,
        "video_path": video_path,
        "video_sha256": video_sha256,
        "reviewer": reviewer,
        "created_at": created_at,
        "revision": revision,
        "supersedes_decision_id": supersedes,
        "action": action.value,
        "selected_candidate_id": selected_candidate_id,
        "selected_segment_id": selected_segment_id,
        "selected_raw_track_id": selected_raw_track_id,
        "selected_frame_index": selected_frame_index,
        "selected_bbox": selected_bbox,
        "direct_bbox_selection": direct_bbox,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "displayed_model_id": displayed_model_id,
        "displayed_checkpoint_sha256": displayed_checkpoint_sha256,
        "displayed_rank": displayed_rank,
        "displayed_score": displayed_score,
        "displayed_T_max": displayed_T_max,
        "displayed_D_max": displayed_D_max,
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_sha256,
        "comment": comment,
        "confidence": confidence.value,
        "status": status.value,
        "training_use_approved": training_use_approved,
        "gallery_use_approved": gallery_use_approved,
        "model_auto_filled": False,
        "score_semantics": "similarity_margin_not_probability",
    }
