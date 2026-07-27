"""Recovery queue filtering from events + effective decision state."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.resolve import (
    EventReviewState,
    resolve_event_review_state,
)

QUEUE_FILTERS = {
    "unresolved",
    "deferred",
    "confirmed",
    "invalid",
    "revoked-needs-review",
    "all",
}

_UNRESOLVED_STATES = {
    EventReviewState.UNREVIEWED,
    EventReviewState.REVOKED_NEEDS_REVIEW,
    EventReviewState.UNKNOWN,
    EventReviewState.NONE_SELECTED,
}


def build_queue_rows(
    events: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event["event_id"])
        state = resolve_event_review_state(event_id=event_id, decisions=decisions)
        event_decisions = [d for d in decisions if d.get("event_id") == event_id]
        effective = None
        if event_decisions:
            from football_analytics.reid.hil.resolve import resolve_effective_decisions

            effective = resolve_effective_decisions(event_decisions).get(event_id)
        rows.append(
            {
                "event_id": event_id,
                "event_type": event.get("event_type"),
                "review_window_start_frame": event.get("review_window_start_frame"),
                "review_window_end_frame": event.get("review_window_end_frame"),
                "created_at": event.get("created_at"),
                "candidate_count": event.get("candidate_count"),
                "last_confirmed_segment_id": event.get("last_confirmed_segment_id"),
                "review_status": state.value,
                "deferred": state == EventReviewState.DEFERRED,
                "revoked_needs_review": state == EventReviewState.REVOKED_NEEDS_REVIEW,
                "current_effective_decision": effective,
                "priority": (event.get("metadata") or {}).get("priority"),
            }
        )
    return rows


def filter_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    filter_name: str = "all",
) -> list[dict[str, Any]]:
    if filter_name not in QUEUE_FILTERS:
        raise ValueError(f"unsupported queue filter: {filter_name!r}")
    if filter_name == "all":
        return [dict(r) for r in rows]
    out: list[dict[str, Any]] = []
    for row in rows:
        state = EventReviewState(row["review_status"])
        if filter_name == "unresolved" and state in _UNRESOLVED_STATES:
            out.append(dict(row))
        elif filter_name == "deferred" and state == EventReviewState.DEFERRED:
            out.append(dict(row))
        elif filter_name == "confirmed" and state == EventReviewState.CONFIRMED:
            out.append(dict(row))
        elif filter_name == "invalid" and state == EventReviewState.INVALID:
            out.append(dict(row))
        elif (
            filter_name == "revoked-needs-review"
            and state == EventReviewState.REVOKED_NEEDS_REVIEW
        ):
            out.append(dict(row))
    return out
