"""One-click CONFIRM_TARGET + Timeline Approval for short-video UI."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil.decisions import DecisionAction
from football_analytics.reid.hil.timeline.approvals import (
    ApprovalLog,
    assert_decision_approvable,
    build_approval_record,
)
from football_analytics.reid.hil_ui.decisions_service import submit_decision
from football_analytics.reid.hil.log import DecisionLog
from football_analytics.reid.short_video.provisional_timeline import (
    append_provisional_from_confirm,
    empty_provisional,
    load_provisional,
    write_provisional,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def one_click_confirm_and_approve(
    *,
    decision_log: DecisionLog,
    approval_log_path: Path,
    event: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
    package: Mapping[str, Any],
    reviewer: str = "short_video_reviewer",
    comment: str = "one_click_confirm_and_continue",
    track_start_frame: int,
    track_end_frame: int,
    fps: float,
    provisional_path: Path,
    approved_timeline_path: Path,
    link_same_target: bool = False,
) -> dict[str, Any]:
    """Append CONFIRM_TARGET then explicit Timeline Approval; update timelines.

    Decision and approval remain separate append-only records.
    Timeline interval uses full raw-track span; provenance keeps selected frame.
    """
    sel = dict(selection)
    sel["selected_track_interval"] = {
        "start_frame": int(track_start_frame),
        "end_frame": int(track_end_frame),
        "selected_frame_index": sel.get("selected_frame_index"),
        "interval_basis": "full_raw_track_observations",
    }
    if link_same_target:
        comment = f"{comment} | same_target_relink"

    result = submit_decision(
        decision_log,
        event=event,
        candidate_manifest=candidate_manifest,
        action=DecisionAction.CONFIRM_TARGET,
        reviewer=reviewer,
        selection=sel,
        comment=comment,
        confidence="confirmed",
        training_use_approved=False,
        gallery_use_approved=False,
        model_metadata={
            "one_click_workflow": True,
            "full_track_interval": True,
            "link_same_target": link_same_target,
        },
    )
    decision = result["decision"]
    log_path = Path(package["decision_log_resolved"])
    log_sha = sha256_file(log_path)

    # Treat short_video_product as product for approval gate
    assert_decision_approvable(
        decision,
        log_path=log_path,
        log_sha256=log_sha,
        review_package_mode="product",
        product_package_id=str(package["package_id"]),
        expected_target_id=str(package["target_id"]),
        expected_video_sha256=str(package["source_video_sha256"]),
    )
    approval_id = f"appr_{decision['decision_id']}_0001"
    cand_shas = list((package.get("candidate_manifest_sha256") or {}).values())
    record = build_approval_record(
        approval_id=approval_id,
        decision=decision,
        product_package_id=str(package["package_id"]),
        decision_log_path=str(log_path),
        decision_log_sha256_at_approval=log_sha,
        comment="one_click_explicit_timeline_approval",
        candidate_manifest_sha256=cand_shas[0] if cand_shas else None,
        segment_manifest_sha256=(package.get("provenance") or {}).get(
            "segment_inventory_sha256"
        ),
        provenance={
            "explicit_user_approval": True,
            "package_mode": "product",
            "ui_profile": "short_video",
            "one_click_workflow": True,
            "full_raw_track_interval": True,
        },
    )
    ApprovalLog(approval_log_path).append(record)

    provisional = load_provisional(provisional_path)
    if not provisional:
        provisional = empty_provisional(
            target_id=str(package["target_id"]),
            video_id=str((package.get("provenance") or {}).get("video_id") or ""),
        )
    provisional = append_provisional_from_confirm(
        provisional,
        decision_id=decision["decision_id"],
        event_id=str(event["event_id"]),
        segment_id=str(sel.get("selected_segment_id")),
        raw_track_id=str(sel.get("selected_raw_track_id")),
        start_frame=int(track_start_frame),
        end_frame=int(track_end_frame),
        fps=fps,
        status="TRACKER CONTINUATION",
    )
    # Mark human-verified covering full interval for simple UI strip
    for iv in provisional.get("intervals") or []:
        if iv.get("decision_id") == decision["decision_id"] and iv.get("status") == "HUMAN VERIFIED":
            iv["end_frame"] = int(track_end_frame)
            iv["end_timestamp"] = int(track_end_frame) / fps if fps else 0.0
            iv["status"] = "VERIFIED"
            iv["approval_id"] = approval_id
            iv["analysis_eligible"] = True
        if iv.get("decision_id") == decision["decision_id"] and "CONTINUATION" in str(
            iv.get("status")
        ):
            iv["approval_id"] = approval_id
            iv["analysis_eligible"] = True
            iv["status"] = "TRACKER CONTINUATION"
    write_provisional(provisional_path, provisional)

    approved = {
        "schema_version": "short_video_approved_timeline_v1",
        "target_id": package["target_id"],
        "video_id": (package.get("provenance") or {}).get("video_id"),
        "updated_at": _utc(),
        "analysis_eligible": True,
        "intervals": [
            {
                "interval_id": f"approved_{decision['decision_id']}",
                "start_frame": int(track_start_frame),
                "end_frame": int(track_end_frame),
                "start_timestamp": int(track_start_frame) / fps if fps else 0.0,
                "end_timestamp": int(track_end_frame) / fps if fps else 0.0,
                "segment_id": sel.get("selected_segment_id"),
                "raw_track_id": str(sel.get("selected_raw_track_id")),
                "status": "VERIFIED",
                "decision_id": decision["decision_id"],
                "approval_id": approval_id,
                "selected_frame_index": sel.get("selected_frame_index"),
                "analysis_eligible": True,
            }
        ],
        "unresolved_intervals": list(provisional.get("unresolved_intervals") or []),
    }
    # Merge prior approved intervals if file exists
    if approved_timeline_path.is_file():
        import json

        prev = json.loads(approved_timeline_path.read_text(encoding="utf-8"))
        prev_iv = list(prev.get("intervals") or [])
        approved["intervals"] = prev_iv + approved["intervals"]
    approved_timeline_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    approved_timeline_path.write_text(
        json.dumps(approved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "decision": decision,
        "approval": record,
        "provisional_timeline": provisional,
        "approved_timeline": approved,
        "log_integrity": result["log_integrity"],
    }


def rank_recovery_candidates(
    candidates: list[Mapping[str, Any]],
    *,
    lost_frame: int,
    previous_bbox: list[float] | None,
    fps: float,
) -> list[dict[str, Any]]:
    """Helper ranking only — never auto-confirm; never hide candidates."""
    ranked = []
    px = py = None
    if previous_bbox and len(previous_bbox) == 4:
        px = (float(previous_bbox[0]) + float(previous_bbox[2])) / 2.0
        py = (float(previous_bbox[1]) + float(previous_bbox[3])) / 2.0
        prev_h = max(1.0, float(previous_bbox[3]) - float(previous_bbox[1]))
    else:
        prev_h = None

    for c in candidates:
        signals: dict[str, Any] = {}
        score = 0.0
        start = int(c.get("start_frame") or 0)
        gap = start - int(lost_frame)
        signals["time_gap_frames"] = gap
        if 0 <= gap <= int(1.5 * fps):
            score += 3.0
            signals["time_gap"] = "near"
        elif gap < 0:
            score -= 1.0
            signals["time_gap"] = "overlaps_lost"
        else:
            signals["time_gap"] = "far"

        refs = list(c.get("bbox_references") or [])
        entry = refs[0]["bbox_xyxy"] if refs else None
        if entry and px is not None:
            cx = (float(entry[0]) + float(entry[2])) / 2.0
            cy = (float(entry[1]) + float(entry[3])) / 2.0
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            signals["spatial_distance_px"] = dist
            if dist < 120:
                score += 3.0
            elif dist < 250:
                score += 1.0
            h = max(1.0, float(entry[3]) - float(entry[1]))
            if prev_h:
                scale = h / prev_h
                signals["bbox_scale_ratio"] = scale
                if 0.7 <= scale <= 1.4:
                    score += 1.0
        team = (c.get("team_evidence") or {}).get("team_label") or "unknown"
        signals["team_label"] = team
        signals["team_is_identity_proof"] = False
        # unknown preserved; never hide other team — only mild rank penalty if labeled other
        if team not in {"unknown", None, ""}:
            score += 0.1  # weak helper only
        ranked.append(
            {
                **dict(c),
                "helper_rank_score": score,
                "helper_signals": signals,
                "helper_only": True,
                "auto_confirm_forbidden": True,
            }
        )
    ranked.sort(key=lambda r: (-float(r["helper_rank_score"]), int(r.get("display_order") or 0)))
    # reassign display_order but keep all
    for i, row in enumerate(ranked, start=1):
        row["display_order"] = i
    return ranked
