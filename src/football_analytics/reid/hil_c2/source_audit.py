"""Product video source audit for HIL-C2 (read-only, no inference)."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.common import sha256_file


class ProductSourceClass(str, Enum):
    PRODUCT_REVIEW_READY = "PRODUCT_REVIEW_READY"
    READY_WITH_ADAPTER = "READY_WITH_ADAPTER"
    REQUIRES_DETECTION_TRACKING_PREPROCESS = "REQUIRES_DETECTION_TRACKING_PREPROCESS"
    DEVELOPMENT_HOLDOUT_ONLY = "DEVELOPMENT_HOLDOUT_ONLY"
    INVALID_OR_INCOMPLETE = "INVALID_OR_INCOMPLETE"


def _file_info(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "bytes": path.stat().st_size if exists else 0,
    }


def audit_product_video_sources(project_root: Path) -> dict[str, Any]:
    """Classify real media stacks; never present development holdout as product match."""
    root = project_root.resolve()
    sources: list[dict[str, Any]] = []

    # --- External enrollment (real user enrollment clip) ---
    ext_video = root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
    ext_track_root = (
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
    )
    ext_det = ext_track_root / "detection/detections.jsonl"
    ext_tracks = ext_track_root / "tracking/tracks.jsonl"
    ext_map = (
        ext_track_root / "inventory/target_001_external_track_candidate_mapping.jsonl"
    )
    ext_track_sum = ext_track_root / "tracking/tracking_summary.json"
    gallery = (
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1/gallery/target_001_gallery_members.jsonl"
    )
    ext_tech = {
        "fps": 30.0,
        "frame_count": 784,
        "duration_seconds": 784 / 30.0,
        "half_or_match": None,
    }
    if ext_track_sum.is_file():
        summary = json.loads(ext_track_sum.read_text(encoding="utf-8"))
        fr = summary.get("frame_range") or [0, 783]
        ext_tech["frame_count"] = int(fr[1]) - int(fr[0]) + 1
        ext_tech["duration_seconds"] = ext_tech["frame_count"] / ext_tech["fps"]
    ext_ready = all(p.is_file() for p in (ext_video, ext_det, ext_tracks, ext_map))
    sources.append(
        {
            "video_id": "target_001_external_enrollment_v1",
            "video_path": str(ext_video),
            "video_sha256": sha256_file(ext_video) if ext_video.is_file() else None,
            "duration_seconds": ext_tech["duration_seconds"],
            "fps": ext_tech["fps"],
            "frame_count": ext_tech["frame_count"],
            "half_or_match": None,
            "detection_run_present": ext_det.is_file(),
            "tracking_run_present": ext_tracks.is_file(),
            "segment_manifest_present": ext_map.is_file(),  # adapter maps tracks→segments
            "bbox_observations_present": ext_tracks.is_file(),
            "source_provenance": {
                "tracking_root": str(ext_track_root),
                "gallery_members": str(gallery) if gallery.is_file() else None,
                "role": "external_enrollment_real_user_clip",
            },
            "target_001_enrollment_info_present": gallery.is_file(),
            "review_package_ready": ext_ready,
            "classification": (
                ProductSourceClass.READY_WITH_ADAPTER.value
                if ext_ready
                else ProductSourceClass.INVALID_OR_INCOMPLETE.value
            ),
            "artifacts": {
                "detection": _file_info(ext_det),
                "tracks": _file_info(ext_tracks),
                "candidate_mapping": _file_info(ext_map),
                "gallery": _file_info(gallery) if gallery.exists() else None,
            },
        }
    )

    # --- Independent holdout v2 (evaluation / development holdout — not product match) ---
    holdout_video = root / "data/test_clips/target_001_independent_holdout_v2.mp4"
    holdout_uni = (
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
    )
    holdout_seg = (
        holdout_uni
        / "segmentation/target_001_holdout_v2_label_blind_segment_inventory.jsonl"
    )
    holdout_obs = (
        holdout_uni
        / "segmentation/target_001_holdout_v2_label_blind_segment_observations.jsonl"
    )
    holdout_det = holdout_uni / "detections/target_001_holdout_v2_detection_manifest.json"
    independence = (
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ingestion_and_preflight"
        / "independence/target_001_holdout_independence_decision.json"
    )
    enroll_eligible = None
    if independence.is_file():
        enroll_eligible = json.loads(independence.read_text(encoding="utf-8")).get(
            "enrollment_eligible"
        )
    sources.append(
        {
            "video_id": "target_001_independent_holdout_v2",
            "video_path": str(holdout_video),
            "video_sha256": sha256_file(holdout_video) if holdout_video.is_file() else None,
            "duration_seconds": 1058 / 30.0,
            "fps": 30.0,
            "frame_count": 1058,
            "half_or_match": None,
            "detection_run_present": holdout_det.is_file(),
            "tracking_run_present": (
                holdout_uni / "tracking/target_001_holdout_v2_tracking_manifest.json"
            ).is_file(),
            "segment_manifest_present": holdout_seg.is_file(),
            "bbox_observations_present": holdout_obs.is_file(),
            "source_provenance": {
                "universe_root": str(holdout_uni),
                "accepted_holdout_role": "frozen_evaluation_input",
                "enrollment_eligible": enroll_eligible,
                "development_only": True,
            },
            "target_001_enrollment_info_present": False,
            "review_package_ready": False,
            "classification": ProductSourceClass.DEVELOPMENT_HOLDOUT_ONLY.value,
            "note": "Must not be presented as a real product match.",
        }
    )

    # --- sample.mp4 evaluation ---
    sample = root / "data/test_clips/sample.mp4"
    sample_root = root / "outputs/reid/full_stage4b_rebuild_r2"
    sources.append(
        {
            "video_id": "sample",
            "video_path": str(sample),
            "video_sha256": sha256_file(sample) if sample.is_file() else None,
            "duration_seconds": 1023 / 30.0,
            "fps": 30.0,
            "frame_count": 1023,
            "half_or_match": None,
            "detection_run_present": (sample_root / "detection/detections.jsonl").is_file(),
            "tracking_run_present": (sample_root / "tracking/tracks.jsonl").is_file(),
            "segment_manifest_present": False,
            "bbox_observations_present": (sample_root / "tracking/tracks.jsonl").is_file(),
            "source_provenance": {
                "rebuild_root": str(sample_root),
                "enrollment_use_forbidden": True,
            },
            "target_001_enrollment_info_present": False,
            "review_package_ready": False,
            "classification": ProductSourceClass.DEVELOPMENT_HOLDOUT_ONLY.value,
            "alternate_classification": ProductSourceClass.REQUIRES_DETECTION_TRACKING_PREPROCESS.value,
            "note": "Evaluation source; no HIL segment schema.",
        }
    )

    # --- fixture (invalid for product) ---
    fixture_pkg = (
        root
        / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/fixture/review_package.json"
    )
    sources.append(
        {
            "video_id": "fixture_video",
            "video_path": None,
            "video_sha256": None,
            "detection_run_present": False,
            "tracking_run_present": False,
            "segment_manifest_present": False,
            "bbox_observations_present": False,
            "source_provenance": {"package": str(fixture_pkg) if fixture_pkg.is_file() else None},
            "target_001_enrollment_info_present": False,
            "review_package_ready": False,
            "classification": ProductSourceClass.INVALID_OR_INCOMPLETE.value,
            "note": "Synthetic fixture; never product timeline source.",
        }
    )

    selected = None
    for s in sources:
        if s["classification"] == ProductSourceClass.READY_WITH_ADAPTER.value and s.get(
            "review_package_ready"
        ):
            selected = s
            break
    if selected is None:
        for s in sources:
            if s["classification"] == ProductSourceClass.PRODUCT_REVIEW_READY.value:
                selected = s
                break

    blocked = selected is None
    return {
        "schema_version": "hil_c2_product_video_source_audit_v1",
        "sources": sources,
        "selected_source_video_id": None if selected is None else selected["video_id"],
        "selected_source": selected,
        "blocked_status": "BLOCKED_HIL_C2_NO_PRODUCT_REVIEW_SOURCE" if blocked else None,
        "exact_next_gate_if_blocked": "REID_PRODUCT_VIDEO_DETECTION_TRACKING_AND_REVIEW_PACKAGE_BUILD",
        "no_new_detection_tracking": True,
        "no_reid_inference": True,
    }
