"""Append-only R1 human visual rejection record (does not mutate R1 stitch outputs)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REVIEW_SCHEMA = "target_tracking_visual_review_v1"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_r1_rejection_record(
    *,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    persistent_target_id_r1: str | None,
    r1_chain: list[str],
    reviewer: str = "human_visual_reviewer",
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA,
        "created_at": _utc(),
        "review_of": "TARGET_TRACKING_R1",
        "result": "REJECTED",
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "persistent_target_id_r1": persistent_target_id_r1,
        "seed_raw_track_id": "10",
        "seed_frame": 29,
        "seed_time_sec": 0.97,
        "approximate_first_wrong_identity_transition_sec": 1.6,
        "original_target_kit": "yellow",
        "wrong_assigned_kit": "white",
        "observed_errors": [
            "RAW_TRACK_PURITY_FAILURE",
            "CROSS_TEAM_IDENTITY_SWITCH",
            "CROWD_OVERLAP",
        ],
        "r1_chain": list(r1_chain),
        "r1_correctness": "NOT_ACCEPTED",
        "reviewer": reviewer,
        "reviewer_comment": (
            "raw track 10 changes from the yellow seeded player to a white player "
            "during overlap; later stitch therefore inherits contaminated identity."
        ),
        "r1_artifacts_not_overwritten": True,
        "human_acceptance": "REJECTED",
    }
