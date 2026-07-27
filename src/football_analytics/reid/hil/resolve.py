"""Derived effective decision and event review state resolution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.decisions import DecisionAction, DecisionStatus
from football_analytics.reid.hil.log import DecisionLog


class EventReviewState(str, Enum):
    UNREVIEWED = "unreviewed"
    DEFERRED = "deferred"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NONE_SELECTED = "none_selected"
    UNKNOWN = "unknown"
    INVALID = "invalid"
    REVOKED_NEEDS_REVIEW = "revoked_needs_review"


def _latest_chain_heads(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Return the newest decision per event that is not superseded by a later record."""
    by_id = {d["decision_id"]: d for d in decisions}
    superseded: set[str] = set()
    for d in decisions:
        prior = d.get("supersedes_decision_id")
        if prior:
            superseded.add(str(prior))

    heads: dict[str, Mapping[str, Any]] = {}
    for d in decisions:
        if d["decision_id"] in superseded:
            continue
        event_id = d["event_id"]
        current = heads.get(event_id)
        if current is None or int(d["revision"]) > int(current["revision"]):
            heads[event_id] = d
    # Also ensure we didn't keep a head that is itself marked superseded/revoked status
    # without a newer active replacement — still report it as the chain tip.
    _ = by_id
    return heads


def resolve_effective_decisions(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Deterministically resolve one effective decision view per event_id."""
    heads = _latest_chain_heads(decisions)
    effective: dict[str, dict[str, Any]] = {}
    for event_id, head in heads.items():
        status = head.get("status")
        action = head.get("action")
        is_active = status == DecisionStatus.ACTIVE.value
        # If tip is REVOKE, there is no effective positive assignment.
        effective[event_id] = {
            "event_id": event_id,
            "effective_decision_id": head["decision_id"] if is_active else head["decision_id"],
            "action": action,
            "status": status,
            "revision": head["revision"],
            "is_active": is_active,
            "supersedes_decision_id": head.get("supersedes_decision_id"),
            "selected_candidate_id": head.get("selected_candidate_id"),
            "selected_segment_id": head.get("selected_segment_id"),
            "direct_bbox_selection": bool(head.get("direct_bbox_selection")),
            "training_use_approved": bool(head.get("training_use_approved")),
            "gallery_use_approved": bool(head.get("gallery_use_approved")),
        }
    return effective


def resolve_event_review_state(
    *,
    event_id: str,
    decisions: Sequence[Mapping[str, Any]],
) -> EventReviewState:
    """Derive review state from the append-only log without mutating event records."""
    event_decisions = [d for d in decisions if d.get("event_id") == event_id]
    if not event_decisions:
        return EventReviewState.UNREVIEWED

    effective = resolve_effective_decisions(event_decisions).get(event_id)
    if effective is None:
        return EventReviewState.UNREVIEWED

    action = effective["action"]
    if action == DecisionAction.REVOKE.value:
        return EventReviewState.REVOKED_NEEDS_REVIEW
    if action == DecisionAction.DEFER.value:
        return EventReviewState.DEFERRED
    if action == DecisionAction.CONFIRM_TARGET.value:
        return EventReviewState.CONFIRMED
    if action == DecisionAction.REJECT_CANDIDATE.value:
        return EventReviewState.REJECTED
    if action == DecisionAction.NONE_OF_THESE.value:
        return EventReviewState.NONE_SELECTED
    if action == DecisionAction.UNKNOWN.value:
        return EventReviewState.UNKNOWN
    if action == DecisionAction.INVALID_SEGMENT.value:
        return EventReviewState.INVALID
    if action == DecisionAction.CORRECT_PREVIOUS_DECISION.value:
        # Correction tip should itself carry the corrected action fields in overrides;
        # if bare correction, treat as needs review.
        return EventReviewState.REVOKED_NEEDS_REVIEW
    return EventReviewState.UNREVIEWED


def derive_effective_state_summary(
    log: DecisionLog,
    *,
    event_ids: Sequence[str],
) -> dict[str, Any]:
    decisions = log.validate_full_log()
    effective = resolve_effective_decisions(decisions)
    states = {
        event_id: resolve_event_review_state(event_id=event_id, decisions=decisions).value
        for event_id in event_ids
    }
    return {
        "schema_version": "hil_effective_state_summary_v1",
        "decision_log_sha256": log.integrity_report()["sha256"],
        "effective_decisions": effective,
        "event_review_states": states,
    }
