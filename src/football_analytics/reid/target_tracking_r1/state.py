"""Persistent target state (independent of raw_track_id)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

STATE_SCHEMA = "persistent_target_state_v1"

TARGET_STATUSES = frozenset(
    {
        "TARGET_CONFIRMED",
        "TARGET_TRACKING",
        "TARGET_TEMPORARILY_LOST",
        "TARGET_UNRESOLVED",
        "TARGET_OUT_OF_FRAME",
        "TARGET_REVIEW_REQUIRED",
    }
)

OBSERVATION_SOURCES = frozenset(
    {
        "HUMAN_SEED",
        "RAW_TRACK_OBSERVATION",
        "AUTO_STITCHED_TRACKLET",
        "HUMAN_CONFIRMED_STITCH",
        "UNRESOLVED_GAP",
    }
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_persistent_target_id(target_id: str) -> str:
    """Logical id distinct from any raw_track_id."""
    return f"ptarget_{target_id}_{uuid.uuid4().hex[:10]}"


def empty_persistent_state(
    *,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    source_video_sha256: str,
    persistent_target_id: str | None = None,
) -> dict[str, Any]:
    ptid = persistent_target_id or new_persistent_target_id(target_id)
    if ptid == target_id:
        raise ValueError("persistent_target_id must not equal product target_id alone as raw equality")
    return {
        "schema_version": STATE_SCHEMA,
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "persistent_target_id": ptid,
        "status": "TARGET_CONFIRMED",
        "created_at": _utc(),
        "updated_at": _utc(),
        "current_raw_track_id": None,
        "seed": None,
        "observations": [],
        "note": "persistent_target_id is independent of raw_track_id",
    }


def apply_human_seed(
    state: dict[str, Any],
    *,
    raw_track_id: str,
    seed_frame: int,
    seed_time: float,
    segment_id: str | None,
    bbox_xyxy: Sequence[float] | None,
) -> dict[str, Any]:
    if str(raw_track_id) == str(state["persistent_target_id"]):
        raise ValueError("raw_track_id must not equal persistent_target_id")
    obs = {
        "observation_id": f"tobs_{uuid.uuid4().hex[:12]}",
        "source": "HUMAN_SEED",
        "status_hint": "TARGET_CONFIRMED",
        "frame_index": int(seed_frame),
        "timestamp_sec": float(seed_time),
        "raw_track_id": str(raw_track_id),
        "segment_id": segment_id,
        "bbox_xyxy": list(bbox_xyxy) if bbox_xyxy else None,
        "provenance": {
            "kind": "HUMAN_SEED",
            "reviewer": "user_provided_seed",
            "seed_frame": int(seed_frame),
        },
    }
    state = dict(state)
    state["seed"] = {
        "raw_track_id": str(raw_track_id),
        "seed_frame": int(seed_frame),
        "seed_time": float(seed_time),
        "segment_id": segment_id,
    }
    state["current_raw_track_id"] = str(raw_track_id)
    state["status"] = "TARGET_TRACKING"
    state["observations"] = list(state.get("observations") or []) + [obs]
    state["updated_at"] = _utc()
    return state


def append_observation(state: dict[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    src = str(observation.get("source") or "")
    if src not in OBSERVATION_SOURCES:
        raise ValueError(f"invalid observation source: {src}")
    state = dict(state)
    state["observations"] = list(state.get("observations") or []) + [dict(observation)]
    if observation.get("raw_track_id"):
        state["current_raw_track_id"] = str(observation["raw_track_id"])
    if observation.get("status_hint") in TARGET_STATUSES:
        state["status"] = observation["status_hint"]
    state["updated_at"] = _utc()
    return state
