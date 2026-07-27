"""Append-only product timeline decision approvals (HIL-C2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import (
    HilValidationError,
    require_mapping,
    require_sha256,
    require_str,
    sha256_file,
    validate_no_path_traversal,
)
from football_analytics.reid.hil.timeline.sources import (
    ACCEPTANCE_LOG_SHA,
    FIXTURE_RUN_IDS,
    UNQUALIFIED_PRODUCT_DECISION_IDS,
)

APPROVAL_SCHEMA_VERSION = "target_timeline_decision_approval_v1"


class ApprovalError(HilValidationError):
    """Raised when an approval record fails validation."""


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_approval(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = require_mapping(payload, field="approval")
    schema = require_str(row.get("schema_version"), field="schema_version")
    if schema != APPROVAL_SCHEMA_VERSION:
        raise ApprovalError(
            f"schema_version must be {APPROVAL_SCHEMA_VERSION}, got {schema!r}"
        )
    status_raw = require_str(row.get("approval_status"), field="approval_status")
    try:
        status = ApprovalStatus(status_raw)
    except ValueError as exc:
        raise ApprovalError(f"invalid approval_status: {status_raw!r}") from exc

    purpose = require_str(row.get("purpose"), field="purpose")
    if purpose != "product_target_timeline":
        raise ApprovalError("purpose must be product_target_timeline")

    decision_id = require_str(row.get("decision_id"), field="decision_id")
    if decision_id in UNQUALIFIED_PRODUCT_DECISION_IDS:
        raise ApprovalError(
            f"refusing approval for unqualified product test decision: {decision_id}"
        )

    provenance = require_mapping(row.get("provenance", {}), field="provenance")
    if provenance.get("acceptance_isolated") or provenance.get("fixture_demo"):
        raise ApprovalError("refusing approval for acceptance/fixture provenance")

    supersedes = row.get("supersedes_approval_id")
    if supersedes is not None:
        supersedes = require_str(supersedes, field="supersedes_approval_id")

    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_id": require_str(row.get("approval_id"), field="approval_id"),
        "decision_id": decision_id,
        "event_id": require_str(row.get("event_id"), field="event_id"),
        "target_id": require_str(row.get("target_id"), field="target_id"),
        "product_package_id": require_str(
            row.get("product_package_id"), field="product_package_id"
        ),
        "video_id": require_str(row.get("video_id"), field="video_id"),
        "video_sha256": require_sha256(row.get("video_sha256"), field="video_sha256"),
        "reviewer": require_str(row.get("reviewer"), field="reviewer"),
        "approved_at": require_str(row.get("approved_at"), field="approved_at"),
        "purpose": purpose,
        "approval_status": status.value,
        "decision_log_path": validate_no_path_traversal(
            row.get("decision_log_path"), field="decision_log_path"
        ),
        "decision_log_sha256_at_approval": require_sha256(
            row.get("decision_log_sha256_at_approval"),
            field="decision_log_sha256_at_approval",
        ),
        "candidate_manifest_sha256": require_sha256(
            row.get("candidate_manifest_sha256"), field="candidate_manifest_sha256"
        )
        if row.get("candidate_manifest_sha256") is not None
        else None,
        "segment_manifest_sha256": require_sha256(
            row.get("segment_manifest_sha256"), field="segment_manifest_sha256"
        )
        if row.get("segment_manifest_sha256") is not None
        else None,
        "comment": require_str(row.get("comment", ""), field="comment", allow_empty=True),
        "supersedes_approval_id": supersedes,
        "provenance": dict(provenance),
    }


def assert_decision_approvable(
    decision: Mapping[str, Any],
    *,
    log_path: str | Path,
    log_sha256: str,
    review_package_mode: str | None,
    product_package_id: str,
    expected_target_id: str,
    expected_video_sha256: str,
) -> None:
    """Fail-closed checks before writing an approval."""
    decision_id = require_str(decision.get("decision_id"), field="decision_id")
    if decision_id in UNQUALIFIED_PRODUCT_DECISION_IDS:
        raise ApprovalError(f"decision_id not approvable: {decision_id}")
    if log_sha256 == ACCEPTANCE_LOG_SHA or "acceptance" in str(log_path).lower():
        raise ApprovalError("acceptance log decisions cannot be timeline-approved")
    run_id = str(decision.get("run_id") or "")
    if run_id in FIXTURE_RUN_IDS or review_package_mode in {"fixture", "hil_a_fixture"}:
        raise ApprovalError("fixture/demo decisions cannot be timeline-approved")
    if review_package_mode not in {"product", "product_review"}:
        raise ApprovalError(
            f"review_package_mode must be product for approval, got {review_package_mode!r}"
        )
    if str(decision.get("target_id")) != expected_target_id:
        raise ApprovalError("cross-target approval rejection")
    if str(decision.get("video_sha256") or "").lower() != expected_video_sha256.lower():
        raise ApprovalError("cross-video approval rejection")
    if str(decision.get("action")) != "CONFIRM_TARGET":
        raise ApprovalError("only CONFIRM_TARGET decisions may be timeline-approved")
    _ = product_package_id


class ApprovalLog:
    """Append-only approval log; never rewrite prior rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def read_raw(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return rows
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_approval(payload)
        existing_ids = {r.get("approval_id") for r in self.read_raw()}
        if validated["approval_id"] in existing_ids:
            raise ApprovalError(f"duplicate approval_id: {validated['approval_id']}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
        return validated

    def sha256(self) -> str:
        return sha256_file(self.path) if self.path.is_file() else ""


def resolve_active_approvals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return decision_id → active approved tip (latest non-revoked/rejected wins)."""
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = validate_approval(raw)
        by_decision.setdefault(row["decision_id"], []).append(row)

    active: dict[str, dict[str, Any]] = {}
    for decision_id, chain in by_decision.items():
        tip = chain[-1]
        if tip["approval_status"] == ApprovalStatus.APPROVED.value:
            active[decision_id] = tip
    return active


def build_approval_record(
    *,
    approval_id: str,
    decision: Mapping[str, Any],
    product_package_id: str,
    decision_log_path: str,
    decision_log_sha256_at_approval: str,
    approval_status: str = ApprovalStatus.APPROVED.value,
    comment: str = "",
    supersedes_approval_id: str | None = None,
    candidate_manifest_sha256: str | None = None,
    segment_manifest_sha256: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    approved_at: str | None = None,
) -> dict[str, Any]:
    return validate_approval(
        {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "approval_id": approval_id,
            "decision_id": decision["decision_id"],
            "event_id": decision["event_id"],
            "target_id": decision["target_id"],
            "product_package_id": product_package_id,
            "video_id": decision["video_id"],
            "video_sha256": decision["video_sha256"],
            "reviewer": decision.get("reviewer") or "unknown",
            "approved_at": approved_at or _utc_now(),
            "purpose": "product_target_timeline",
            "approval_status": approval_status,
            "decision_log_path": decision_log_path,
            "decision_log_sha256_at_approval": decision_log_sha256_at_approval,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "segment_manifest_sha256": segment_manifest_sha256,
            "comment": comment,
            "supersedes_approval_id": supersedes_approval_id,
            "provenance": dict(provenance or {"explicit_user_approval": True}),
        }
    )
