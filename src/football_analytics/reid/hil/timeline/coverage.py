"""Coverage summary for target timelines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.timeline.schema import IntervalStatus


def _dur(intervals: Sequence[Mapping[str, Any]], status: str | None = None) -> float:
    total = 0.0
    for row in intervals:
        if status is not None and row.get("status") != status:
            continue
        total += float(row.get("duration_seconds") or 0.0)
    return total


def build_coverage_summary(
    *,
    total_video_duration_seconds: float,
    intervals: Sequence[Mapping[str, Any]],
    unresolved_intervals: Sequence[Mapping[str, Any]],
    excluded_intervals: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    excluded_decision_counts: Mapping[str, int],
) -> dict[str, Any]:
    all_iv = list(intervals) + list(unresolved_intervals) + list(excluded_intervals)
    confirmed = _dur(intervals, IntervalStatus.HUMAN_CONFIRMED.value)
    continuation = _dur(intervals, IntervalStatus.TRACKER_CONTINUATION.value)
    unresolved = _dur(unresolved_intervals, IntervalStatus.UNRESOLVED.value)
    rejected = _dur(all_iv, IntervalStatus.REJECTED.value)
    invalid = _dur(all_iv, IntervalStatus.INVALID.value)
    revoked = _dur(all_iv, IntervalStatus.REVOKED.value)
    eligible = sum(
        float(i.get("duration_seconds") or 0.0)
        for i in intervals
        if i.get("analysis_eligible") is True
    )
    total = float(total_video_duration_seconds or 0.0)
    verified = confirmed + continuation
    raw_tracks = sorted(
        {
            str(i.get("raw_track_id"))
            for i in intervals
            if i.get("raw_track_id") and i.get("status")
            in {
                IntervalStatus.HUMAN_CONFIRMED.value,
                IntervalStatus.TRACKER_CONTINUATION.value,
            }
        }
    )
    return {
        "total_video_duration_seconds": total,
        "human_confirmed_duration_seconds": confirmed,
        "tracker_continuation_duration_seconds": continuation,
        "unresolved_duration_seconds": unresolved,
        "rejected_duration_seconds": rejected,
        "invalid_duration_seconds": invalid,
        "revoked_duration_seconds": revoked,
        "analysis_eligible_duration_seconds": eligible,
        "verified_coverage_percentage": (100.0 * verified / total) if total > 0 else 0.0,
        "unresolved_percentage": (100.0 * unresolved / total) if total > 0 else 0.0,
        "number_of_confirmed_segments": sum(
            1
            for i in intervals
            if i.get("status") == IntervalStatus.HUMAN_CONFIRMED.value
        ),
        "number_of_raw_track_ids_linked": len(raw_tracks),
        "raw_track_ids_linked": raw_tracks,
        "number_of_recovery_gaps": len(unresolved_intervals),
        "number_of_conflicts": len(conflicts),
        "number_of_excluded_test_acceptance_decisions": int(
            excluded_decision_counts.get("excluded", 0)
        ),
        "excluded_decision_counts": dict(excluded_decision_counts),
    }
