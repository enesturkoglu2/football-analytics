"""Human-in-the-loop target recovery (HIL) domain package."""

from __future__ import annotations

from football_analytics.reid.hil.candidates import (
    CandidateManifestError,
    validate_candidate_manifest,
)
from football_analytics.reid.hil.decisions import (
    DecisionError,
    DecisionAction,
    DecisionConfidence,
    DecisionStatus,
    build_decision,
    validate_decision,
)
from football_analytics.reid.hil.events import (
    EventError,
    EventStatus,
    EventType,
    validate_recovery_event,
)
from football_analytics.reid.hil.log import (
    AppendOnlyLogError,
    DecisionLog,
    compute_log_sha256,
)
from football_analytics.reid.hil.resolve import (
    EventReviewState,
    resolve_effective_decisions,
    resolve_event_review_state,
)

__all__ = [
    "AppendOnlyLogError",
    "CandidateManifestError",
    "DecisionAction",
    "DecisionConfidence",
    "DecisionError",
    "DecisionLog",
    "DecisionStatus",
    "EventError",
    "EventReviewState",
    "EventStatus",
    "EventType",
    "build_decision",
    "compute_log_sha256",
    "resolve_effective_decisions",
    "resolve_event_review_state",
    "validate_candidate_manifest",
    "validate_decision",
    "validate_recovery_event",
]
