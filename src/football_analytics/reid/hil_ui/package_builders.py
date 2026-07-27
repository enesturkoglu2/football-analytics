"""Builders for fixture and existing-artifact review packages (dev/diagnostic only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.candidates import validate_candidate_manifest
from football_analytics.reid.hil.common import (
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
    sha256_file,
)
from football_analytics.reid.hil.events import EventType, validate_recovery_event
from football_analytics.reid.hil_ui.package import (
    REVIEW_PACKAGE_SCHEMA_VERSION,
    validate_review_package,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_synthetic_fixture_video(path: Path, *, frames: int = 160, w: int = 320, h: int = 240) -> str:
    """Create a tiny deterministic mp4 for offline UI smoke (no source overwrite)."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    try:
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (40 + (i % 50), 40, 40)
            # Draw candidate-like boxes on middle frames
            boxes = [
                (10, 20, 40, 80),
                (50, 20, 90, 90),
                (100, 30, 150, 100),
                (160, 40, 220, 120),
                (200, 10, 280, 70),  # extra non-candidate for direct selection
            ]
            for j, (x1, y1, x2, y2) in enumerate(boxes):
                if 110 <= i <= 145:
                    color = (0, 255, 255) if j < 4 else (255, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            writer.write(frame)
    finally:
        writer.release()
    return sha256_file(path)


def build_fixture_review_package(root: Path) -> dict[str, Any]:
    """HIL-A-style fixture package with synthetic media and empty decision log."""
    root = root.resolve()
    media = root / "media"
    writable = root / "session"
    readonly = root / "readonly_sources"
    readonly.mkdir(parents=True, exist_ok=True)
    writable.mkdir(parents=True, exist_ok=True)
    video_path = media / "fixture_review.mp4"
    video_sha = create_synthetic_fixture_video(video_path)

    bboxes = [
        [10.0, 20.0, 40.0, 80.0],
        [50.0, 20.0, 90.0, 90.0],
        [100.0, 30.0, 150.0, 100.0],
        [160.0, 40.0, 220.0, 120.0],
    ]
    candidates = []
    for i, bbox in enumerate(bboxes, start=1):
        mid = 120 + i
        candidates.append(
            {
                "candidate_id": f"cand_{i:03d}",
                "segment_id": f"H2_SEG_DEMO_{i:03d}",
                "raw_track_id": f"raw_track_demo_{i:03d}",
                "start_frame": mid - 10,
                "middle_frame": mid,
                "end_frame": mid + 20,
                "bbox_references": [{"frame_index": mid, "bbox_xyxy": bbox}],
                "crop_path": f"fixtures/crops/cand_{i:03d}.jpg",
                "crop_sha256": f"{i:064x}",
                "context_paths": {},
                "context_sha256": {},
                "short_clip_path": None,
                "short_clip_sha256": None,
                "team_evidence": {"same_team_predicted": True, "is_identity_proof": False},
                "visibility": {"score": 0.9},
                "quality": {"score": 0.8},
                "contamination": {"multi_person": False},
                "sportsreid_model_id": SPORTSREID_MODEL_ID,
                "sportsreid_checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
                "appearance_rank": i,
                "T_max": 0.55 - 0.05 * (i - 1),
                "D_max": 0.5,
                "S": 0.05 * max(0, 3 - i),
                "temporal_distance": float(i),
                "spatial_distance": None,
                "eligibility": True,
                "rejection_reason": None,
                "display_order": i,
            }
        )
    manifest = validate_candidate_manifest(
        {
            "schema_version": "target_recovery_candidate_manifest_v1",
            "event_id": "evt_demo_reentry_001",
            "target_id": "target_001",
            "candidate_count": 4,
            "eligible_count": 4,
            "supports_direct_bbox_selection": True,
            "appearance_rank_is_helper_only": True,
            "rank_does_not_hide_candidates": True,
            "candidates": candidates,
        }
    )
    man_path = root / "demo_candidate_manifest.json"
    _write_json(man_path, manifest)
    man_sha = sha256_file(man_path)

    lost = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": "evt_demo_target_lost_001",
            "project_id": "football-analytics",
            "run_id": "hil_b_fixture",
            "target_id": "target_001",
            "event_type": EventType.TARGET_LOST.value,
            "video_id": "fixture_video",
            "video_path": str(video_path.relative_to(root)),
            "video_sha256": video_sha,
            "created_at": "2026-07-27T00:00:00Z",
            "status": "open",
            "trigger_source": "fixture",
            "trigger_reason": "demo TARGET_LOST",
            "last_confirmed_segment_id": "H2_SEG_DEMO_000",
            "last_confirmed_frame_index": 100,
            "review_window_start_frame": 101,
            "review_window_end_frame": 150,
            "candidate_manifest_path": None,
            "candidate_manifest_sha256": None,
            "candidate_count": 0,
            "requires_calibration": True,
            "evidence_paths": [],
            "evidence_sha256": [],
            "provenance": {"fixture": True},
            "metadata": {},
        }
    )
    reentry = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": "evt_demo_reentry_001",
            "project_id": "football-analytics",
            "run_id": "hil_b_fixture",
            "target_id": "target_001",
            "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
            "video_id": "fixture_video",
            "video_path": str(video_path.relative_to(root)),
            "video_sha256": video_sha,
            "created_at": "2026-07-27T00:00:01Z",
            "status": "open",
            "trigger_source": "fixture",
            "trigger_reason": "demo reentry candidates",
            "last_confirmed_segment_id": "H2_SEG_DEMO_000",
            "last_confirmed_frame_index": 100,
            "review_window_start_frame": 110,
            "review_window_end_frame": 145,
            "candidate_manifest_path": "demo_candidate_manifest.json",
            "candidate_manifest_sha256": man_sha,
            "candidate_count": 4,
            "requires_calibration": True,
            "evidence_paths": [],
            "evidence_sha256": [],
            "provenance": {"fixture": True},
            "metadata": {},
        }
    )
    events_path = root / "demo_recovery_events.jsonl"
    _write_jsonl(events_path, [lost, reentry])
    events_sha = sha256_file(events_path)

    decision_rel = "decision_log.jsonl"
    (writable / decision_rel).write_text("", encoding="utf-8")

    package = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": "pkg_hil_b_fixture_v1",
        "project_id": "football-analytics",
        "run_id": "hil_b_fixture",
        "target_id": "target_001",
        "source_video_path": str(video_path.relative_to(root)),
        "source_video_sha256": video_sha,
        "event_manifest_path": "demo_recovery_events.jsonl",
        "event_manifest_sha256": events_sha,
        "candidate_manifest_paths": ["demo_candidate_manifest.json"],
        "candidate_manifest_sha256": {"demo_candidate_manifest.json": man_sha},
        "decision_log_path": decision_rel,
        "target_gallery_reference": None,
        "model_metadata": {
            "helper_model_id": SPORTSREID_MODEL_ID,
            "helper_checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
            "role": "appearance_helper_ranker_only",
        },
        "media_root": "media",
        "read_only_source_roots": ["media", "readonly_sources"],
        "writable_session_root": "session",
        "created_at": _utc_now(),
        "provenance": {
            "mode": "hil_a_fixture",
            "development_only": True,
            "extra_frame_bboxes": {
                "121": [
                    {
                        "bbox_id": "extra_direct_001",
                        "bbox_xyxy": [200.0, 10.0, 280.0, 70.0],
                        "provenance": {
                            "listed_candidate": False,
                            "segment_id": "DIRECT_SEG_EXTRA",
                            "raw_track_id": "raw_direct_extra",
                        },
                    }
                ]
            },
        },
        "media_status": "synthetic_fixture",
    }
    validated = validate_review_package(package, package_dir=root, verify_sources=True)
    out = root / "review_package.json"
    _write_json(out, package)
    validated["package_file"] = str(out)
    return validated


def build_existing_artifact_review_package(
    root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Read-only existing holdout segments → diagnostic review package (no GT prefill)."""
    root = root.resolve()
    project_root = project_root.resolve()
    writable = root / "session"
    writable.mkdir(parents=True, exist_ok=True)

    video_rel = "data/test_clips/target_001_independent_holdout_v2.mp4"
    video_abs = project_root / video_rel
    video_sha = sha256_file(video_abs)

    obs_path = (
        project_root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
        / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl"
    )
    elig_path = (
        project_root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
        / "quality/target_001_holdout_v2_segment_review_eligibility.jsonl"
    )
    eligible = set()
    with elig_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("review_eligible"):
                eligible.add(row["segment_id"])


    # Pick first frame that has >=4 eligible segment observations
    by_frame: dict[int, list[dict[str, Any]]] = {}
    with obs_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["segment_id"] not in eligible:
                continue
            by_frame.setdefault(int(row["frame_index"]), []).append(row)
            if len(by_frame) > 400:
                # keep scanning a bounded prefix for determinism/speed
                break

    chosen_frame = None
    chosen_rows: list[dict[str, Any]] = []
    for frame_index in sorted(by_frame):
        # unique segments
        uniq: dict[str, dict[str, Any]] = {}
        for row in by_frame[frame_index]:
            uniq.setdefault(row["segment_id"], row)
        if len(uniq) >= 4:
            chosen_frame = frame_index
            chosen_rows = list(uniq.values())[:4]
            break
    if chosen_frame is None:
        raise RuntimeError("could not find a frame with >=4 eligible segment bboxes")

    # Collect sparse start/end refs for the same segments (not continuous tracking).
    end_frame = chosen_frame + 5
    by_seg_frame: dict[tuple[str, int], list[float]] = {
        (row["segment_id"], chosen_frame): list(row["bbox_xyxy"]) for row in chosen_rows
    }
    with obs_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["segment_id"], int(row["frame_index"]))
            if key[0] in {r["segment_id"] for r in chosen_rows} and key[1] == end_frame:
                by_seg_frame[key] = list(row["bbox_xyxy"])

    candidates = []
    for i, row in enumerate(chosen_rows, start=1):
        seg = row["segment_id"]
        refs = [
            {
                "frame_index": chosen_frame,
                "bbox_xyxy": list(by_seg_frame[(seg, chosen_frame)]),
            }
        ]
        end_key = (seg, end_frame)
        if end_key in by_seg_frame:
            refs.append(
                {
                    "frame_index": end_frame,
                    "bbox_xyxy": list(by_seg_frame[end_key]),
                }
            )
        candidates.append(
            {
                "candidate_id": f"real_cand_{i:03d}",
                "segment_id": seg,
                "raw_track_id": row["raw_track_code"],
                "start_frame": chosen_frame,
                "middle_frame": chosen_frame,
                "end_frame": end_frame,
                "bbox_references": refs,
                "crop_path": None,
                "crop_sha256": None,
                "context_paths": {},
                "context_sha256": {},
                "short_clip_path": None,
                "short_clip_sha256": None,
                "team_evidence": {"is_identity_proof": False},
                "visibility": {},
                "quality": {"source": "label_blind_observation"},
                "contamination": {},
                "sportsreid_model_id": None,
                "sportsreid_checkpoint_sha256": None,
                "appearance_rank": None,
                "T_max": None,
                "D_max": None,
                "S": None,
                "temporal_distance": None,
                "spatial_distance": None,
                "eligibility": True,
                "rejection_reason": None,
                "display_order": i,
            }
        )
    manifest = validate_candidate_manifest(
        {
            "schema_version": "target_recovery_candidate_manifest_v1",
            "event_id": "evt_real_holdout_reentry_001",
            "target_id": "target_001",
            "candidate_count": 4,
            "eligible_count": 4,
            "supports_direct_bbox_selection": True,
            "appearance_rank_is_helper_only": True,
            "rank_does_not_hide_candidates": True,
            "candidates": candidates,
        }
    )
    man_path = root / "existing_candidate_manifest.json"
    _write_json(man_path, manifest)
    man_sha = sha256_file(man_path)

    event = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": "evt_real_holdout_reentry_001",
            "project_id": "football-analytics",
            "run_id": "hil_b_existing_artifact_dev",
            "target_id": "target_001",
            "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
            "video_id": "target_001_independent_holdout_v2",
            "video_path": video_rel,
            "video_sha256": video_sha,
            "created_at": _utc_now(),
            "status": "open",
            "trigger_source": "existing_artifact_diagnostic",
            "trigger_reason": "development-only HIL-B smoke; GT not applied",
            "last_confirmed_segment_id": "H2_SEG_UNKNOWN_PRIOR",
            "last_confirmed_frame_index": max(0, chosen_frame - 30),
            "review_window_start_frame": max(0, chosen_frame - 10),
            "review_window_end_frame": chosen_frame + 10,
            "candidate_manifest_path": "existing_candidate_manifest.json",
            "candidate_manifest_sha256": man_sha,
            "candidate_count": 4,
            "requires_calibration": True,
            "evidence_paths": [],
            "evidence_sha256": [],
            "provenance": {
                "development_only": True,
                "gt_prefill": False,
                "source_observations": obs_path.name,
                "chosen_frame": chosen_frame,
            },
            "metadata": {"priority": "diagnostic"},
        }
    )
    events_path = root / "existing_recovery_events.jsonl"
    _write_jsonl(events_path, [event])
    events_sha = sha256_file(events_path)
    decision_rel = "decision_log.jsonl"
    (writable / decision_rel).write_text("", encoding="utf-8")

    # Absolute read-only roots: project data + label-blind universe
    ro_roots = [
        str(project_root / "data"),
        str(
            project_root
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
        ),
    ]
    package = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": "pkg_hil_b_existing_holdout_dev_v1",
        "project_id": "football-analytics",
        "run_id": "hil_b_existing_artifact_dev",
        "target_id": "target_001",
        "source_video_path": str(video_abs),
        "source_video_sha256": video_sha,
        "event_manifest_path": "existing_recovery_events.jsonl",
        "event_manifest_sha256": events_sha,
        "candidate_manifest_paths": ["existing_candidate_manifest.json"],
        "candidate_manifest_sha256": {"existing_candidate_manifest.json": man_sha},
        "decision_log_path": decision_rel,
        "target_gallery_reference": {
            "note": "reference only; not auto-applied",
            "path": None,
        },
        "model_metadata": {
            "inference": False,
            "sportsreid_scores_present": False,
        },
        "media_root": str(project_root / "data/test_clips"),
        "read_only_source_roots": ro_roots,
        "writable_session_root": "session",
        "created_at": _utc_now(),
        "provenance": {
            "mode": "existing_artifact_readonly",
            "development_only": True,
            "product_decision_log_isolated": True,
            "gt_prefill": False,
            "chosen_frame": chosen_frame,
            "candidate_segment_ids": [c["segment_id"] for c in candidates],
            "observation_index_path": str(obs_path),
            "sparse_observations": True,
            "not_continuous_tracking_preview": True,
        },
        "media_status": "verified",
    }
    validated = validate_review_package(package, package_dir=root, verify_sources=True)
    out = root / "review_package.json"
    _write_json(out, package)
    validated["package_file"] = str(out)
    validated["chosen_frame"] = chosen_frame
    return validated
