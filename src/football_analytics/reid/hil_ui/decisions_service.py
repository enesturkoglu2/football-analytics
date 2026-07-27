"""Decision submission service wrapping HIL-A append-only log."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from football_analytics.reid.hil.decisions import DecisionAction, build_decision
from football_analytics.reid.hil.log import DecisionLog
from football_analytics.reid.hil.resolve import (
    resolve_effective_decisions,
    resolve_event_review_state,
)
from football_analytics.reid.hil_ui.actions import preview_confirmation


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def next_revision(log: DecisionLog, event_id: str) -> int:
    rows = [r for r in log.read_raw() if r.get("event_id") == event_id]
    if not rows:
        return 1
    return max(int(r["revision"]) for r in rows) + 1


def effective_decision_for_event(log: DecisionLog, event_id: str) -> dict[str, Any] | None:
    rows = log.validate_full_log()
    return resolve_effective_decisions(rows).get(event_id)


def history_for_event(log: DecisionLog, event_id: str) -> dict[str, Any]:
    rows = log.get_history(event_id=event_id)
    effective = resolve_effective_decisions(rows).get(event_id)
    chain = []
    for row in rows:
        chain.append(
            {
                "decision_id": row["decision_id"],
                "revision": row["revision"],
                "action": row["action"],
                "status": row["status"],
                "supersedes_decision_id": row.get("supersedes_decision_id"),
                "reviewer": row.get("reviewer"),
                "created_at": row.get("created_at"),
                "comment": row.get("comment", ""),
            }
        )
    return {
        "event_id": event_id,
        "raw_records": rows,
        "supersedes_chain": chain,
        "effective_active_decision": effective,
        "review_state": resolve_event_review_state(event_id=event_id, decisions=rows).value,
        "decision_log_sha256": log.integrity_report()["sha256"],
    }


def submit_decision(
    log: DecisionLog,
    *,
    event: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any] | None,
    action: DecisionAction | str,
    reviewer: str,
    selection: Mapping[str, Any] | None = None,
    comment: str = "",
    confidence: str = "unknown",
    training_use_approved: bool = False,
    gallery_use_approved: bool = False,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action_enum = action if isinstance(action, DecisionAction) else DecisionAction(action)
    # Defaults must remain false unless explicitly True from caller.
    if training_use_approved is not True:
        training_use_approved = False
    if gallery_use_approved is not True:
        gallery_use_approved = False

    preview = preview_confirmation(
        action=action_enum,
        selection=selection,
        reviewer=reviewer,
        comment=comment,
        confidence=confidence,
        training_use_approved=training_use_approved,
        gallery_use_approved=gallery_use_approved,
    )
    sel = dict(selection or {})
    meta = dict(model_metadata or {})
    decision = build_decision(
        decision_id=_new_id("dec"),
        project_id=str(event["project_id"]),
        run_id=str(event["run_id"]),
        target_id=str(event["target_id"]),
        event_id=str(event["event_id"]),
        video_id=str(event["video_id"]),
        video_path=str(event["video_path"]),
        video_sha256=str(event["video_sha256"]),
        reviewer=reviewer,
        created_at=_utc_now(),
        revision=next_revision(log, str(event["event_id"])),
        action=action_enum,
        confidence=confidence,
        selected_candidate_id=sel.get("selected_candidate_id"),
        selected_segment_id=sel.get("selected_segment_id"),
        selected_raw_track_id=sel.get("selected_raw_track_id"),
        selected_frame_index=sel.get("selected_frame_index"),
        selected_bbox_xyxy=sel.get("selected_bbox_xyxy"),
        direct_bbox_selection=bool(sel.get("direct_bbox_selection", False)),
        candidate_manifest_path=event.get("candidate_manifest_path"),
        candidate_manifest_sha256=event.get("candidate_manifest_sha256"),
        displayed_model_id=meta.get("model_id") or sel.get("displayed_model_id"),
        displayed_checkpoint_sha256=meta.get("checkpoint_sha256")
        or sel.get("displayed_checkpoint_sha256"),
        displayed_rank=sel.get("displayed_rank"),
        displayed_score=sel.get("displayed_score"),
        displayed_T_max=sel.get("displayed_T_max"),
        displayed_D_max=sel.get("displayed_D_max"),
        comment=comment,
        training_use_approved=training_use_approved,
        gallery_use_approved=gallery_use_approved,
    )
    appended = log.append(
        decision, event=event, candidate_manifest=candidate_manifest
    )
    return {
        "decision": appended,
        "confirmation_preview": preview.to_dict(),
        "log_integrity": log.integrity_report(),
    }


def undo_last_decision(
    log: DecisionLog,
    *,
    event: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any] | None,
    reviewer: str,
    comment: str = "undo last decision",
) -> dict[str, Any]:
    """Undo by appending a superseding REVOKE; never edits prior rows."""
    event_id = str(event["event_id"])
    effective = effective_decision_for_event(log, event_id)
    if effective is None:
        raise ValueError("no effective decision to undo")
    prior_id = str(effective["effective_decision_id"])
    appended = log.revoke_active_decision(
        prior_decision_id=prior_id,
        new_decision_id=_new_id("dec"),
        reviewer=reviewer,
        created_at=_utc_now(),
        revision=next_revision(log, event_id),
        comment=comment,
        event=event,
        candidate_manifest=candidate_manifest,
    )
    return {
        "decision": appended,
        "supersedes": prior_id,
        "log_integrity": log.integrity_report(),
    }


def correct_or_revoke(
    log: DecisionLog,
    *,
    event: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any] | None,
    reviewer: str,
    action: DecisionAction,
    comment: str = "",
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    effective = effective_decision_for_event(log, event_id)
    if effective is None:
        raise ValueError("no effective decision to correct/revoke")
    prior_id = str(effective["effective_decision_id"])
    sel = dict(selection or {})
    if action == DecisionAction.REVOKE:
        appended = log.revoke_active_decision(
            prior_decision_id=prior_id,
            new_decision_id=_new_id("dec"),
            reviewer=reviewer,
            created_at=_utc_now(),
            revision=next_revision(log, event_id),
            comment=comment,
            event=event,
            candidate_manifest=candidate_manifest,
        )
    else:
        appended = log.create_superseding_decision(
            prior_decision_id=prior_id,
            new_decision_id=_new_id("dec"),
            action=action,
            reviewer=reviewer,
            created_at=_utc_now(),
            revision=next_revision(log, event_id),
            comment=comment,
            event=event,
            candidate_manifest=candidate_manifest,
            selected_candidate_id=sel.get("selected_candidate_id"),
            selected_segment_id=sel.get("selected_segment_id"),
            selected_raw_track_id=sel.get("selected_raw_track_id"),
            selected_frame_index=sel.get("selected_frame_index"),
            selected_bbox=sel.get("selected_bbox_xyxy"),
            direct_bbox_selection=bool(sel.get("direct_bbox_selection", False)),
        )
    return {
        "decision": appended,
        "supersedes": prior_id,
        "log_integrity": log.integrity_report(),
    }
