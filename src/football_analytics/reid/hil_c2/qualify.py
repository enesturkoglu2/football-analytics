"""Qualify product decisions against timeline approvals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil.log import DecisionLog
from football_analytics.reid.hil.resolve import resolve_effective_decisions
from football_analytics.reid.hil.timeline.approvals import (
    ApprovalLog,
    resolve_active_approvals,
)
from football_analytics.reid.hil.timeline.sources import audit_decision_sources


def qualify_product_session(
    *,
    decision_log_path: str | Path,
    approval_log_path: str | Path,
    review_package_mode: str = "product",
    excluded_source_audits: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify product decisions + approvals for timeline eligibility."""
    dpath = Path(decision_log_path).expanduser().resolve()
    apath = Path(approval_log_path).expanduser().resolve()
    decision_sha_before = sha256_file(dpath) if dpath.is_file() else ""
    approval_log = ApprovalLog(apath)
    approval_rows = approval_log.read_raw()
    active = resolve_active_approvals(approval_rows)
    approved_ids = set(active.keys())

    product_audit = audit_decision_sources(
        [{"path": str(dpath), "review_package_mode": review_package_mode}],
        approved_decision_ids=approved_ids,
        require_timeline_approval=True,
    )

    decisions = DecisionLog(dpath).read_raw()
    effective = resolve_effective_decisions(decisions)

    revoked_or_superseded = [
        d
        for d in product_audit["decisions"]
        if d["source_classification"] == "REVOKED_OR_SUPERSEDED"
        or (d.get("action") == "REVOKE" and d.get("effective"))
    ]
    unknown_defer = [
        d
        for d in product_audit["decisions"]
        if d.get("effective") and d.get("action") in {"UNKNOWN", "DEFER", "NONE_OF_THESE"}
    ]
    unapproved_confirms = [
        d
        for d in product_audit["decisions"]
        if d.get("action") == "CONFIRM_TARGET"
        and d.get("effective")
        and d["decision_id"] not in approved_ids
        and d["source_classification"] == "PRODUCT_APPROVED"
    ]

    decision_sha_after = sha256_file(dpath) if dpath.is_file() else ""
    return {
        "schema_version": "hil_c2_product_decision_qualification_v1",
        "decision_log_path": str(dpath),
        "decision_log_sha256": decision_sha_after,
        "decision_log_immutable": decision_sha_before == decision_sha_after,
        "approval_log_path": str(apath),
        "approval_log_sha256": approval_log.sha256(),
        "active_approvals": active,
        "approved_decision_ids": sorted(approved_ids),
        "product_audit": product_audit,
        "effective_decisions": effective,
        "timeline_eligible_decisions": product_audit["timeline_eligible_decisions"],
        "revoked_or_superseded": revoked_or_superseded,
        "unknown_defer_none": unknown_defer,
        "unapproved_confirms": unapproved_confirms,
        "counts": {
            "product_decisions": len(product_audit["decisions"]),
            "active_approvals": len(active),
            "timeline_eligible": len(product_audit["timeline_eligible_decisions"]),
            "revoked_or_superseded": len(revoked_or_superseded),
            "unknown_defer_none": len(unknown_defer),
            "unapproved_confirms": len(unapproved_confirms),
        },
        "excluded_source_audits": list(excluded_source_audits or []),
    }
