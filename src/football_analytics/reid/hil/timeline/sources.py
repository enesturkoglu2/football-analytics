"""Decision source qualification for timeline reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil.decisions import DecisionAction
from football_analytics.reid.hil.log import DecisionLog
from football_analytics.reid.hil.resolve import (
    resolve_effective_decisions,
    resolve_event_review_state,
)
from football_analytics.reid.hil.timeline.schema import (
    DECISION_SOURCE_MANIFEST_SCHEMA,
    DecisionSourceClass,
)

# Known isolated / unqualified sources from HIL-B acceptance lineage.
ACCEPTANCE_LOG_SHA = "11d7e0b6972f93da6b94da5ab02d3087af6794a38c21643db120987faf57ee49"
UNQUALIFIED_PRODUCT_DECISION_IDS = frozenset(
    {
        "dec_6fbbcc997aff",
        "dec_50d455b82dec",  # same product test log chain as rev1
    }
)
FIXTURE_RUN_IDS = frozenset({"hil_a_demo", "hil_b_fixture", "hil_b_r2_acceptance"})
ACCEPTANCE_RUN_IDS = frozenset(
    {"hil_b_r2_existing_acceptance", "hil_b_existing_artifact_dev"}
)


def classify_decision(
    decision: Mapping[str, Any],
    *,
    log_path: str,
    log_sha256: str,
    review_package_mode: str | None = None,
    effective_decision_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Classify one decision for timeline eligibility (never mutates the log)."""
    decision_id = str(decision.get("decision_id"))
    action = str(decision.get("action"))
    run_id = str(decision.get("run_id") or "")
    path_l = str(log_path).lower()

    classification = DecisionSourceClass.PRODUCT_APPROVED
    exclusion = None
    eligible = True

    # Acceptance isolation by SHA or path/run
    if log_sha256 == ACCEPTANCE_LOG_SHA or "acceptance" in path_l or "/tmp/hil_b_r2_acceptance" in str(log_path):
        classification = DecisionSourceClass.ACCEPTANCE_ISOLATED
        eligible = False
        exclusion = "acceptance_isolated_log"
    elif review_package_mode in {"fixture", "hil_a_fixture"} or run_id in FIXTURE_RUN_IDS or "fixture" in path_l:
        classification = DecisionSourceClass.FIXTURE_DEMO
        eligible = False
        exclusion = "fixture_or_demo_decision"
    elif decision_id in UNQUALIFIED_PRODUCT_DECISION_IDS or run_id in {
        "hil_b_existing_artifact_dev"
    }:
        # Product path test/acceptance decisions without explicit product timeline approval.
        classification = DecisionSourceClass.PRODUCT_UNQUALIFIED_TEST_DECISION
        eligible = False
        exclusion = "unqualified_product_test_decision_not_user_approved_for_timeline"
    elif effective_decision_ids is not None and decision_id not in effective_decision_ids:
        # Non-head records that are superseded by later appends
        if decision.get("supersedes_decision_id") or action == DecisionAction.REVOKE.value:
            # Keep explicit revoked/superseded class for non-effective rows when not already excluded
            pass

    # Effective tip REVOKE → not timeline-eligible as confirm
    if eligible and action == DecisionAction.REVOKE.value:
        classification = DecisionSourceClass.REVOKED_OR_SUPERSEDED
        eligible = False
        exclusion = "effective_or_recorded_revoke_not_confirm"

    if eligible and action != DecisionAction.CONFIRM_TARGET.value:
        # Only CONFIRM can create confirmed intervals; others may create unresolved later.
        # Mark non-confirm as not interval-confirm-eligible but may still be referenced.
        if action in {
            DecisionAction.NONE_OF_THESE.value,
            DecisionAction.UNKNOWN.value,
            DecisionAction.DEFER.value,
            DecisionAction.INVALID_SEGMENT.value,
            DecisionAction.REJECT_CANDIDATE.value,
            DecisionAction.CORRECT_PREVIOUS_DECISION.value,
        }:
            # Still PRODUCT_APPROVED class if otherwise approved, but not confirm-eligible
            pass

    is_effective = bool(
        effective_decision_ids is None or decision_id in (effective_decision_ids or set())
    )

    confirm_eligible = bool(
        eligible
        and is_effective
        and action == DecisionAction.CONFIRM_TARGET.value
    )

    # Provenance invalid if missing required ids for confirm
    if confirm_eligible:
        if not decision.get("selected_segment_id") and not decision.get("direct_bbox_selection"):
            classification = DecisionSourceClass.INVALID_PROVENANCE
            confirm_eligible = False
            eligible = False
            exclusion = "confirm_missing_segment_or_direct_bbox"

    if (
        not confirm_eligible
        and classification == DecisionSourceClass.PRODUCT_APPROVED
        and not is_effective
    ):
        classification = DecisionSourceClass.REVOKED_OR_SUPERSEDED
        exclusion = exclusion or "not_effective_chain_head"

    return {
        "decision_id": decision_id,
        "event_id": decision.get("event_id"),
        "revision": decision.get("revision"),
        "action": action,
        "status": decision.get("status"),
        "effective": is_effective,
        "log_path": str(log_path),
        "log_sha256": log_sha256,
        "review_package_mode": review_package_mode,
        "run_id": run_id,
        "reviewer": decision.get("reviewer"),
        "source_classification": classification.value,
        "timeline_eligible": confirm_eligible,
        "exclusion_reason": exclusion,
        "approval_provenance": {
            "explicit_product_timeline_approval": classification
            == DecisionSourceClass.PRODUCT_APPROVED
            and confirm_eligible,
            "assumed_approval": False,
        },
        "selected_segment_id": decision.get("selected_segment_id"),
        "selected_raw_track_id": decision.get("selected_raw_track_id"),
        "selected_frame_index": decision.get("selected_frame_index"),
        "direct_bbox_selection": bool(decision.get("direct_bbox_selection")),
        "video_sha256": decision.get("video_sha256"),
        "target_id": decision.get("target_id"),
    }


def audit_decision_log(
    path: str | Path,
    *,
    review_package_mode: str | None = None,
) -> dict[str, Any]:
    log_path = Path(path).expanduser().resolve()
    log = DecisionLog(log_path)
    rows = log.read_raw()
    sha = sha256_file(log_path) if log_path.is_file() else ""
    effective = resolve_effective_decisions(rows)
    effective_ids = {v["effective_decision_id"] for v in effective.values()}
    classified = [
        classify_decision(
            row,
            log_path=str(log_path),
            log_sha256=sha,
            review_package_mode=review_package_mode,
            effective_decision_ids=effective_ids,
        )
        for row in rows
    ]
    # Mark non-effective rows that were product-approved class as superseded/revoked class
    for item, row in zip(classified, rows):
        if item["timeline_eligible"]:
            continue
        if item["source_classification"] != DecisionSourceClass.PRODUCT_APPROVED.value:
            continue
        if row.get("decision_id") not in effective_ids:
            item["source_classification"] = DecisionSourceClass.REVOKED_OR_SUPERSEDED.value
            item["exclusion_reason"] = item.get("exclusion_reason") or "not_effective_chain_head"

    event_states = {
        eid: resolve_event_review_state(event_id=eid, decisions=rows).value
        for eid in sorted({str(r.get("event_id")) for r in rows})
    }
    return {
        "schema_version": DECISION_SOURCE_MANIFEST_SCHEMA,
        "log_path": str(log_path),
        "log_sha256": sha,
        "review_package_mode": review_package_mode,
        "record_count": len(rows),
        "decisions": classified,
        "effective_decisions": effective,
        "event_review_states": event_states,
        "counts": _count_classes(classified),
    }


def audit_decision_sources(
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit multiple logs. Each source: {path, review_package_mode?}."""
    manifests = []
    all_decisions: list[dict[str, Any]] = []
    for src in sources:
        man = audit_decision_log(
            src["path"], review_package_mode=src.get("review_package_mode")
        )
        manifests.append(man)
        all_decisions.extend(man["decisions"])
    counts = _count_classes(all_decisions)
    eligible = [d for d in all_decisions if d["timeline_eligible"]]
    return {
        "schema_version": DECISION_SOURCE_MANIFEST_SCHEMA,
        "logs": manifests,
        "decisions": all_decisions,
        "timeline_eligible_decisions": eligible,
        "counts": counts,
        "product_approved_confirm_count": sum(
            1
            for d in all_decisions
            if d["source_classification"] == DecisionSourceClass.PRODUCT_APPROVED.value
            and d["timeline_eligible"]
        ),
    }


def _count_classes(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {c.value: 0 for c in DecisionSourceClass}
    counts["timeline_eligible"] = 0
    counts["excluded"] = 0
    for d in decisions:
        counts[str(d["source_classification"])] = counts.get(str(d["source_classification"]), 0) + 1
        if d.get("timeline_eligible"):
            counts["timeline_eligible"] += 1
        else:
            counts["excluded"] += 1
    return counts


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
