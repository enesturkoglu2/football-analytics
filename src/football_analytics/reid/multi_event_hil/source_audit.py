"""Multi-event HIL source selection (read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil_c2.source_audit import ProductSourceClass


def audit_multi_event_sources(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    ext_video = root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
    track_root = (
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
    )
    mapping = track_root / "inventory/target_001_external_track_candidate_mapping.jsonl"
    tracks = track_root / "tracking/tracks.jsonl"
    detections = track_root / "detection/detections.jsonl"

    duration_s = 26.154646
    ready = all(p.is_file() for p in (ext_video, mapping, tracks, detections))
    # Multi-event evidence: EXT_004 then later EXT_183 / EXT_198
    occurrence_codes = []
    if mapping.is_file():
        with mapping.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("external_candidate_code") in {"EXT_004", "EXT_183", "EXT_198"}:
                    occurrence_codes.append(
                        {
                            "code": row["external_candidate_code"],
                            "raw_track_id": row["raw_external_track_id"],
                            "first_frame": row["first_frame"],
                            "last_frame": row["last_frame"],
                            "observation_count": row["observation_count"],
                        }
                    )

    external = {
        "video_id": "target_001_external_enrollment_v1",
        "video_path": str(ext_video),
        "video_sha256": sha256_file(ext_video) if ext_video.is_file() else None,
        "duration_seconds": duration_s,
        "preferred_duration_band_1_to_5_min": False,
        "duration_note": (
            "Clip is ~26s (below preferred 1–5 min). Selected because it is the only "
            "preprocessed real-user enrollment source with detection/tracking and "
            "multiple temporally separated occurrence tracklets."
        ),
        "target_visible_at_start": True,
        "track_id_changes_or_exits_estimated": len(occurrence_codes) >= 2,
        "detection_run_present": detections.is_file(),
        "tracking_run_present": tracks.is_file(),
        "segment_mapping_present": mapping.is_file(),
        "bbox_observations_present": tracks.is_file(),
        "development_holdout_gt_prefill": False,
        "old_development_gallery_forbidden": True,
        "occurrence_tracklets": occurrence_codes,
        "classification": (
            ProductSourceClass.READY_WITH_ADAPTER.value
            if ready
            else ProductSourceClass.INVALID_OR_INCOMPLETE.value
        ),
        "review_package_ready": ready,
        "tracking_root": str(track_root),
    }

    holdout = {
        "video_id": "target_001_independent_holdout_v2",
        "classification": ProductSourceClass.DEVELOPMENT_HOLDOUT_ONLY.value,
        "review_package_ready": False,
        "note": "Must not be used as product match or gallery source.",
    }
    sample = {
        "video_id": "sample",
        "classification": ProductSourceClass.DEVELOPMENT_HOLDOUT_ONLY.value,
        "review_package_ready": False,
        "note": "Enrollment forbidden; incomplete HIL segment schema.",
    }

    selected = external if ready else None
    return {
        "schema_version": "multi_event_hil_source_audit_v1",
        "sources": [external, holdout, sample],
        "selected_source": selected,
        "selected_video_id": None if selected is None else selected["video_id"],
        "blocked_status": None
        if selected is not None
        else "BLOCKED_MULTI_EVENT_HIL_NO_PREPROCESSED_SOURCE",
        "exact_next_gate_if_blocked": "REID_PRODUCT_VIDEO_DETECTION_TRACKING_AND_REVIEW_PACKAGE_BUILD",
        "no_new_detection_tracking_default": True,
        "no_clip": True,
        "no_game_state": True,
    }
