"""Provisional vs approved live timeline helpers for short-video UI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_provisional(*, target_id: str, video_id: str) -> dict[str, Any]:
    return {
        "schema_version": "short_video_provisional_timeline_v1",
        "target_id": target_id,
        "video_id": video_id,
        "updated_at": _utc(),
        "analysis_eligible": False,
        "intervals": [],
        "unresolved_intervals": [],
        "note": "Provisional only; not analysis-eligible until Timeline Approval",
    }


def append_provisional_from_confirm(
    provisional: Mapping[str, Any],
    *,
    decision_id: str,
    event_id: str,
    segment_id: str,
    raw_track_id: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    status: str = "TRACKER CONTINUATION",
) -> dict[str, Any]:
    """Update provisional timeline immediately after human CONFIRM (not approved)."""
    out = dict(provisional)
    intervals = list(out.get("intervals") or [])
    # Human verified at start frame; continuation through track end
    human = {
        "interval_id": f"prov_human_{decision_id}",
        "start_frame": int(start_frame),
        "end_frame": int(start_frame),
        "start_timestamp": int(start_frame) / fps if fps else 0.0,
        "end_timestamp": int(start_frame) / fps if fps else 0.0,
        "segment_id": segment_id,
        "raw_track_id": str(raw_track_id),
        "status": "HUMAN VERIFIED",
        "decision_id": decision_id,
        "approval_id": None,
        "event_id": event_id,
        "analysis_eligible": False,
    }
    cont = {
        "interval_id": f"prov_cont_{decision_id}",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_timestamp": int(start_frame) / fps if fps else 0.0,
        "end_timestamp": int(end_frame) / fps if fps else 0.0,
        "segment_id": segment_id,
        "raw_track_id": str(raw_track_id),
        "status": status,
        "decision_id": decision_id,
        "approval_id": None,
        "event_id": event_id,
        "analysis_eligible": False,
    }
    intervals.append(human)
    if end_frame > start_frame:
        intervals.append(cont)
    out["intervals"] = intervals
    out["updated_at"] = _utc()
    return out


def mark_unresolved(
    provisional: Mapping[str, Any],
    *,
    start_frame: int,
    end_frame: int | None,
    fps: float,
    previous_segment_id: str | None,
    previous_raw_track_id: str | None,
) -> dict[str, Any]:
    out = dict(provisional)
    gaps = list(out.get("unresolved_intervals") or [])
    end = int(end_frame) if end_frame is not None else int(start_frame)
    gaps.append(
        {
            "interval_id": f"prov_unresolved_{start_frame}_{end}",
            "start_frame": int(start_frame),
            "end_frame": end,
            "start_timestamp": int(start_frame) / fps if fps else 0.0,
            "end_timestamp": end / fps if fps else 0.0,
            "status": "UNRESOLVED",
            "previous_segment_id": previous_segment_id,
            "previous_raw_track_id": str(previous_raw_track_id)
            if previous_raw_track_id is not None
            else None,
            "analysis_eligible": False,
        }
    )
    out["unresolved_intervals"] = gaps
    out["updated_at"] = _utc()
    return out


def write_provisional(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_provisional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def target_status_from_selection(
    *,
    selected_raw_track_id: str | None,
    track_end_frame: int | None,
    current_frame: int | None,
    has_unresolved: bool,
) -> str:
    if has_unresolved and (
        selected_raw_track_id is None
        or (
            track_end_frame is not None
            and current_frame is not None
            and current_frame > track_end_frame
        )
    ):
        return "UNRESOLVED"
    if selected_raw_track_id is None:
        return "REVIEW REQUIRED"
    if (
        track_end_frame is not None
        and current_frame is not None
        and current_frame > track_end_frame
    ):
        return "LOST"
    return "TRACKING"
