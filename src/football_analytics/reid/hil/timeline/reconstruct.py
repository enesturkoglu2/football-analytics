"""Deterministic target timeline reconstruction from qualified decisions."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import sha256_file, sha256_json_canonical
from football_analytics.reid.hil.decisions import DecisionAction
from football_analytics.reid.hil.resolve import resolve_effective_decisions, resolve_event_review_state
from football_analytics.reid.hil.timeline.conflicts import (
    find_dual_target_segment_bindings,
    find_interval_overlaps,
)
from football_analytics.reid.hil.timeline.coverage import build_coverage_summary
from football_analytics.reid.hil.timeline.schema import (
    GENERATOR_VERSION,
    TIMELINE_SCHEMA_VERSION,
    IntervalStatus,
    TimelineStatus,
    analysis_eligible_for_status,
    validate_timeline,
)
from football_analytics.reid.hil.timeline.segments import validate_decision_against_segment
from football_analytics.reid.hil.timeline.sources import audit_decision_sources


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frames_to_seconds(frame: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame) / float(fps)


def _interval_from_segment(
    *,
    interval_id: str,
    target_id: str,
    segment: Mapping[str, Any],
    decision: Mapping[str, Any],
    status: IntervalStatus,
    fps: float,
    evidence_source: str,
    first_bbox: list[float] | None = None,
    last_bbox: list[float] | None = None,
    bbox_count: int = 0,
) -> dict[str, Any]:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    st = _frames_to_seconds(start, fps)
    et = _frames_to_seconds(end + 1, fps)  # inclusive end frame duration
    return {
        "interval_id": interval_id,
        "target_id": target_id,
        "segment_id": segment.get("segment_id"),
        "raw_track_id": segment.get("raw_track_id"),
        "start_frame": start,
        "end_frame": end,
        "start_time_seconds": st,
        "end_time_seconds": et,
        "duration_seconds": max(0.0, et - st),
        "status": status.value,
        "evidence_source": evidence_source,
        "source_event_ids": [decision.get("event_id")],
        "source_decision_ids": [decision.get("decision_id")],
        "tracker_provenance": {
            "raw_track_id": segment.get("raw_track_id"),
            "segment_id": segment.get("segment_id"),
        },
        "bbox_observation_count": bbox_count or int(segment.get("observation_count") or 0),
        "first_bbox": first_bbox,
        "last_bbox": last_bbox,
        "confidence_class": "human",
        "analysis_eligible": analysis_eligible_for_status(status),
        "exclusion_reason": None,
        "superseded_by": None,
        "metadata": {
            "direct_bbox_selection": bool(decision.get("direct_bbox_selection")),
        },
        "requires_calibration": False,
    }


def _unresolved_gap(
    *,
    interval_id: str,
    target_id: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    reason: str,
    event_ids: list[Any] | None = None,
) -> dict[str, Any]:
    st = _frames_to_seconds(start_frame, fps)
    et = _frames_to_seconds(end_frame + 1, fps)
    return {
        "interval_id": interval_id,
        "target_id": target_id,
        "segment_id": None,
        "raw_track_id": None,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_seconds": st,
        "end_time_seconds": et,
        "duration_seconds": max(0.0, et - st),
        "status": IntervalStatus.UNRESOLVED.value,
        "evidence_source": "gap_between_confirmed_intervals",
        "source_event_ids": list(event_ids or []),
        "source_decision_ids": [],
        "tracker_provenance": {},
        "bbox_observation_count": 0,
        "first_bbox": None,
        "last_bbox": None,
        "confidence_class": "unresolved",
        "analysis_eligible": False,
        "exclusion_reason": reason,
        "superseded_by": None,
        "metadata": {"auto_filled": False},
        "requires_calibration": True,
    }


def generate_gaps(
    confirmed: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
    fps: float,
    video_end_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Produce unresolved gaps between confirmed intervals; never interpolates identity."""
    rows = sorted(confirmed, key=lambda r: (int(r["start_frame"]), int(r["end_frame"])))
    gaps: list[dict[str, Any]] = []
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        gap_start = int(a["end_frame"]) + 1
        gap_end = int(b["start_frame"]) - 1
        if gap_end >= gap_start:
            gaps.append(
                _unresolved_gap(
                    interval_id=f"gap_{gap_start}_{gap_end}",
                    target_id=target_id,
                    start_frame=gap_start,
                    end_frame=gap_end,
                    fps=fps,
                    reason="unresolved_recovery_gap",
                    event_ids=list(a.get("source_event_ids") or [])
                    + list(b.get("source_event_ids") or []),
                )
            )
    return gaps


def reconstruct_timeline(
    *,
    project_id: str,
    run_id: str,
    target_id: str,
    video_id: str,
    video_path: str,
    video_sha256: str,
    frame_rate: float,
    total_video_frames: int,
    total_video_duration_seconds: float,
    decision_sources: Sequence[Mapping[str, Any]],
    segment_index: Mapping[str, Mapping[str, Any]] | None = None,
    source_segment_manifest_path: str | None = None,
    source_segment_manifest_sha256: str | None = None,
    initial_enrollment: Mapping[str, Any] | None = None,
    allow_tracker_continuation: bool = False,
    timeline_id: str | None = None,
    generated_at: str | None = None,
    approved_decision_ids: set[str] | None = None,
    require_timeline_approval: bool = False,
) -> dict[str, Any]:
    """Reconstruct timeline. Does not mutate decision logs."""
    source_audit = audit_decision_sources(
        decision_sources,
        approved_decision_ids=approved_decision_ids,
        require_timeline_approval=require_timeline_approval,
    )
    eligible = list(source_audit["timeline_eligible_decisions"])
    intervals: list[dict[str, Any]] = []
    excluded_intervals: list[dict[str, Any]] = []
    provenance_validation: list[dict[str, Any]] = []

    # Optional initial enrollment (must be explicit human-confirmed)
    if initial_enrollment and initial_enrollment.get("human_confirmed") is True:
        seg = {
            "segment_id": initial_enrollment["segment_id"],
            "raw_track_id": initial_enrollment["raw_track_id"],
            "start_frame": int(initial_enrollment["start_frame"]),
            "end_frame": int(initial_enrollment["end_frame"]),
            "observation_count": int(initial_enrollment.get("observation_count") or 0),
        }
        intervals.append(
            _interval_from_segment(
                interval_id=f"enroll_{seg['segment_id']}",
                target_id=target_id,
                segment=seg,
                decision={
                    "event_id": initial_enrollment.get("event_id"),
                    "decision_id": initial_enrollment.get("decision_id"),
                    "direct_bbox_selection": False,
                },
                status=IntervalStatus.HUMAN_CONFIRMED,
                fps=frame_rate,
                evidence_source="initial_enrollment_human_confirmed",
            )
        )

    for d in eligible:
        # Only CONFIRM_TARGET eligible rows reach here
        if segment_index is None:
            # Fixture-only path: use decision fields as segment span metadata if provided
            if d.get("selected_segment_id") and d.get("metadata_segment_span"):
                span = d["metadata_segment_span"]
                seg = {
                    "segment_id": d["selected_segment_id"],
                    "raw_track_id": d.get("selected_raw_track_id"),
                    "start_frame": int(span["start_frame"]),
                    "end_frame": int(span["end_frame"]),
                    "observation_count": int(span.get("observation_count") or 0),
                }
                provenance_validation.append({"decision_id": d["decision_id"], "ok": True})
            else:
                provenance_validation.append(
                    {
                        "decision_id": d["decision_id"],
                        "ok": False,
                        "reason": "no_segment_index_and_no_span",
                    }
                )
                continue
        else:
            # Need full decision row for bbox/frame checks — reload from classified fields
            fake_decision = {
                "selected_segment_id": d.get("selected_segment_id"),
                "selected_raw_track_id": d.get("selected_raw_track_id"),
                "selected_frame_index": d.get("selected_frame_index"),
                "direct_bbox_selection": d.get("direct_bbox_selection"),
                "video_sha256": d.get("video_sha256"),
                "event_id": d.get("event_id"),
                "decision_id": d.get("decision_id"),
            }
            try:
                result = validate_decision_against_segment(
                    fake_decision,
                    segment_index=segment_index,
                    expected_video_sha256=video_sha256,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "blocked_status": "BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH",
                    "error": str(exc),
                    "source_audit": source_audit,
                }
            provenance_validation.append({"decision_id": d["decision_id"], **result})
            if not result["ok"]:
                continue
            seg = result["segment"]

        intervals.append(
            _interval_from_segment(
                interval_id=f"confirm_{d['decision_id']}",
                target_id=target_id,
                segment=seg,
                decision=d,
                status=IntervalStatus.HUMAN_CONFIRMED,
                fps=frame_rate,
                evidence_source="human_confirm_target",
            )
        )

    # Tracker continuation is opt-in and calibration-gated; disabled by default in HIL-C.
    _ = allow_tracker_continuation

    conflicts = find_interval_overlaps(intervals, target_id=target_id)
    conflicts.extend(find_dual_target_segment_bindings(intervals))

    unresolved = generate_gaps(
        [
            i
            for i in intervals
            if i["status"]
            in {
                IntervalStatus.HUMAN_CONFIRMED.value,
                IntervalStatus.TRACKER_CONTINUATION.value,
            }
        ],
        target_id=target_id,
        fps=frame_rate,
        video_end_frame=total_video_frames - 1 if total_video_frames else None,
    )

    # Also emit unresolved for effective NONE/UNKNOWN/DEFER on approved logs only
    for log in source_audit["logs"]:
        for d in log["decisions"]:
            if d["source_classification"] != "PRODUCT_APPROVED":
                continue
            if not d.get("effective"):
                continue
            action = d.get("action")
            if action in {
                DecisionAction.NONE_OF_THESE.value,
                DecisionAction.UNKNOWN.value,
                DecisionAction.DEFER.value,
            }:
                # event-scoped placeholder unresolved without inventing frames
                unresolved.append(
                    {
                        "interval_id": f"event_unresolved_{d['event_id']}",
                        "target_id": target_id,
                        "segment_id": None,
                        "raw_track_id": None,
                        "start_frame": 0,
                        "end_frame": 0,
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 0.0,
                        "duration_seconds": 0.0,
                        "status": IntervalStatus.UNRESOLVED.value,
                        "evidence_source": f"effective_{action.lower()}",
                        "source_event_ids": [d.get("event_id")],
                        "source_decision_ids": [d.get("decision_id")],
                        "tracker_provenance": {},
                        "bbox_observation_count": 0,
                        "first_bbox": None,
                        "last_bbox": None,
                        "confidence_class": "unresolved",
                        "analysis_eligible": False,
                        "exclusion_reason": action,
                        "superseded_by": None,
                        "metadata": {"zero_width_event_marker": True},
                        "requires_calibration": True,
                    }
                )
            if action == DecisionAction.INVALID_SEGMENT.value:
                excluded_intervals.append(
                    {
                        "interval_id": f"invalid_{d['decision_id']}",
                        "target_id": target_id,
                        "segment_id": d.get("selected_segment_id"),
                        "raw_track_id": d.get("selected_raw_track_id"),
                        "start_frame": 0,
                        "end_frame": 0,
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 0.0,
                        "duration_seconds": 0.0,
                        "status": IntervalStatus.INVALID.value,
                        "evidence_source": "invalid_segment_decision",
                        "source_event_ids": [d.get("event_id")],
                        "source_decision_ids": [d.get("decision_id")],
                        "tracker_provenance": {},
                        "bbox_observation_count": 0,
                        "first_bbox": None,
                        "last_bbox": None,
                        "confidence_class": "invalid",
                        "analysis_eligible": False,
                        "exclusion_reason": "INVALID_SEGMENT",
                        "superseded_by": None,
                        "metadata": {},
                        "requires_calibration": False,
                    }
                )

    approved_count = int(source_audit.get("product_approved_confirm_count") or 0)
    if conflicts:
        timeline_status = TimelineStatus.CONFLICTED
    elif approved_count == 0 and not intervals:
        timeline_status = TimelineStatus.NO_APPROVED_PRODUCT_DECISIONS
    elif not intervals:
        timeline_status = TimelineStatus.EMPTY
    else:
        timeline_status = TimelineStatus.OK

    coverage = build_coverage_summary(
        total_video_duration_seconds=total_video_duration_seconds,
        intervals=intervals,
        unresolved_intervals=unresolved,
        excluded_intervals=excluded_intervals,
        conflicts=conflicts,
        excluded_decision_counts=source_audit["counts"],
    )

    decision_log_paths = [str(s["path"]) for s in decision_sources]
    decision_log_sha256 = {
        str(log["log_path"]): log["log_sha256"] for log in source_audit["logs"]
    }

    timeline = {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline_id": timeline_id or f"timeline_{target_id}_{run_id}",
        "project_id": project_id,
        "run_id": run_id,
        "target_id": target_id,
        "video_id": video_id,
        "video_path": video_path,
        "video_sha256": video_sha256,
        "source_segment_manifest_path": source_segment_manifest_path,
        "source_segment_manifest_sha256": source_segment_manifest_sha256,
        "decision_source_manifest_path": None,
        "decision_source_manifest_sha256": None,
        "decision_log_paths": decision_log_paths,
        "decision_log_sha256": decision_log_sha256,
        "generated_at": generated_at or _utc_now(),
        "generator_version": GENERATOR_VERSION,
        "timeline_status": timeline_status.value,
        "frame_rate": float(frame_rate),
        "total_video_frames": int(total_video_frames),
        "total_video_duration_seconds": float(total_video_duration_seconds),
        "intervals": intervals,
        "unresolved_intervals": unresolved,
        "excluded_intervals": excluded_intervals,
        "conflicts": conflicts,
        "coverage_summary": coverage,
        "provenance": {
            "source_audit_counts": source_audit["counts"],
            "product_approved_confirm_count": approved_count,
            "segment_provenance_validation": provenance_validation,
            "no_gap_interpolation": True,
            "no_reid_inference": True,
            "no_detection_tracking_rerun": True,
        },
    }
    validated = validate_timeline(timeline)
    return {
        "timeline": validated,
        "source_audit": source_audit,
        "blocked_status": (
            "BLOCKED_HIL_C_TIMELINE_CONFLICT"
            if timeline_status == TimelineStatus.CONFLICTED
            else None
        ),
    }


def reconstruct_twice_for_determinism(**kwargs: Any) -> dict[str, Any]:
    a = reconstruct_timeline(**kwargs)
    b = reconstruct_timeline(**kwargs)
    # Drop generated_at for semantic compare if auto-now; force same timestamp
    ta = copy.deepcopy(a["timeline"])
    tb = copy.deepcopy(b["timeline"])
    ta["generated_at"] = "COMPARE"
    tb["generated_at"] = "COMPARE"
    same = sha256_json_canonical(ta) == sha256_json_canonical(tb)
    return {
        "deterministic": same,
        "sha_a": sha256_json_canonical(ta),
        "sha_b": sha256_json_canonical(tb),
        "max_numeric_diff": 0,
        "timeline": a["timeline"],
        "source_audit": a["source_audit"],
        "blocked_status": a.get("blocked_status"),
    }


def dump_intervals_jsonl(path: Any, intervals: Sequence[Mapping[str, Any]]) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in intervals:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
