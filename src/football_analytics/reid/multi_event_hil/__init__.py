"""Multi-event HIL package exports."""

from __future__ import annotations

from football_analytics.reid.multi_event_hil.identity import (
    MATCH_ID,
    PACKAGE_ID,
    TARGET_ID,
    VIDEO_ID,
    VIDEO_SHA,
    AnalysisIdentity,
    build_identity,
)
from football_analytics.reid.multi_event_hil.source_audit import audit_multi_event_sources

__all__ = [
    "AnalysisIdentity",
    "MATCH_ID",
    "PACKAGE_ID",
    "TARGET_ID",
    "VIDEO_ID",
    "VIDEO_SHA",
    "audit_multi_event_sources",
    "build_identity",
]
