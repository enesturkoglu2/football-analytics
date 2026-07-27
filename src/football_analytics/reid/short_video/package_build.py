"""Build short-video product review package (all-player dense bbox, no EXT fallback)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.candidates import validate_candidate_manifest
from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil.events import EventType, validate_recovery_event
from football_analytics.reid.hil_c2.product_package import _bbox_refs_for_row, _segment_id
from football_analytics.reid.hil_ui.package import (
    REVIEW_PACKAGE_SCHEMA_VERSION,
    validate_review_package,
)
from football_analytics.reid.short_video import PACKAGE_MODE, UI_PROFILE
from football_analytics.reid.short_video.dense_timeline import (
    attach_candidate_ids_to_timeline,
    build_dense_bbox_timeline,
    observations_for_component,
    write_dense_timeline,
)
from football_analytics.reid.short_video.identity import ShortVideoIdentity
from football_analytics.reid.short_video.provisional_timeline import (
    empty_provisional,
    write_provisional,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidates_from_rows(
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    target_id: str,
    codes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    by_code = {str(r["external_candidate_code"]): r for r in mapping_rows}
    selected = list(codes) if codes is not None else [
        str(r["external_candidate_code"])
        for r in mapping_rows
        if r.get("review_eligible", True)
    ]
    candidates: list[dict[str, Any]] = []
    for i, code in enumerate(selected, start=1):
        row = by_code[code]
        refs = _bbox_refs_for_row(row)
        mid = refs[len(refs) // 2]["frame_index"] if refs else int(row["first_frame"])
        candidates.append(
            {
                "candidate_id": f"sv_cand_{i:04d}",
                "segment_id": _segment_id(code),
                "raw_track_id": str(row["raw_external_track_id"]),
                "start_frame": int(row["first_frame"]),
                "middle_frame": mid,
                "end_frame": int(row["last_frame"]),
                "bbox_references": refs,
                "crop_path": None,
                "crop_sha256": None,
                "context_paths": {},
                "context_sha256": {},
                "short_clip_path": None,
                "short_clip_sha256": None,
                "team_evidence": {
                    "team_label": "unknown",
                    "is_identity_proof": False,
                },
                "visibility": {},
                "quality": {"source": "short_video_track_mapping"},
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
                "metadata": {
                    "external_candidate_code": code,
                    "event_id": event_id,
                    "observation_coverage": row.get("observation_coverage"),
                },
            }
        )
    return candidates


def _tracks_active_in_window(
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    start: int,
    end: int,
) -> list[str]:
    codes: list[str] = []
    for row in mapping_rows:
        if not row.get("review_eligible", True):
            continue
        first = int(row["first_frame"])
        last = int(row["last_frame"])
        if last < start or first > end:
            continue
        codes.append(str(row["external_candidate_code"]))
    return codes


def build_short_video_review_package(
    root: Path,
    *,
    project_root: Path,
    identity: ShortVideoIdentity,
    video_abs: Path,
    mapping_rows: list[dict[str, Any]],
    dense_timeline: dict[str, Any],
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    preprocess_root: Path,
    max_recovery_events: int = 15,
) -> dict[str, Any]:
    root = root.resolve()
    project_root = project_root.resolve()
    writable = root / "session"
    writable.mkdir(parents=True, exist_ok=True)

    video_sha = identity.video_sha256
    if sha256_file(video_abs) != video_sha:
        raise RuntimeError("video SHA mismatch for short-video package")

    inventory = []
    for row in mapping_rows:
        code = str(row["external_candidate_code"])
        inventory.append(
            {
                "segment_id": _segment_id(code),
                "raw_track_code": str(row["raw_external_track_id"]),
                "raw_track_id": str(row["raw_external_track_id"]),
                "external_candidate_code": code,
                "start_frame": int(row["first_frame"]),
                "end_frame": int(row["last_frame"]),
                "observation_count": int(row["observation_count"]),
                "observation_coverage": row.get("observation_coverage"),
                "video_sha256": video_sha,
                "source": "short_video_track_mapping",
                "team_metadata": row.get("team_metadata"),
            }
        )
    inv_path = root / "segment_inventory.jsonl"
    _write_jsonl(inv_path, inventory)

    # Mapping copy for UI/debug (not legacy EXT_004/183/198 fallback)
    map_path = root / "track_candidate_mapping.jsonl"
    _write_jsonl(map_path, mapping_rows)

    enroll_event_id = "evt_sv_initial_enrollment_001"
    enroll_codes = [
        str(r["external_candidate_code"])
        for r in mapping_rows
        if r.get("review_eligible", True)
    ]
    enroll_cands = _candidates_from_rows(
        mapping_rows, event_id=enroll_event_id, target_id=identity.target_id, codes=enroll_codes
    )
    enroll_manifest = validate_candidate_manifest(
        {
            "schema_version": "target_recovery_candidate_manifest_v1",
            "event_id": enroll_event_id,
            "target_id": identity.target_id,
            "candidate_count": len(enroll_cands),
            "eligible_count": len(enroll_cands),
            "supports_direct_bbox_selection": True,
            "appearance_rank_is_helper_only": True,
            "rank_does_not_hide_candidates": True,
            "candidates": enroll_cands,
        }
    )
    enroll_name = "enrollment_candidate_manifest.json"
    _write_json(root / enroll_name, enroll_manifest)
    enroll_sha = sha256_file(root / enroll_name)

    # Dense timeline with candidate ids
    dens = attach_candidate_ids_to_timeline(dense_timeline, enroll_cands)
    dens_path = root / "dense_bbox_timeline.json"
    write_dense_timeline(dens_path, dens)
    dens_component = observations_for_component(dens)
    ir = writable / "interactive_review"
    ir.mkdir(parents=True, exist_ok=True)
    (ir / "dense_observations.json").write_text(
        json.dumps(dens_component, ensure_ascii=False), encoding="utf-8"
    )

    enroll_end = min(90, max(0, frame_count - 1))
    events: list[dict[str, Any]] = [
        validate_recovery_event(
            {
                "schema_version": "target_recovery_event_v1",
                "event_id": enroll_event_id,
                "project_id": "football-analytics",
                "run_id": identity.analysis_run_id,
                "target_id": identity.target_id,
                "event_type": EventType.INITIAL_TARGET_ENROLLMENT.value,
                "video_id": identity.video_id,
                "video_path": str(video_abs),
                "video_sha256": video_sha,
                "created_at": _utc(),
                "status": "open",
                "trigger_source": "short_video_preprocess",
                "trigger_reason": "initial target selection on full-player dense bbox timeline",
                "last_confirmed_segment_id": None,
                "last_confirmed_frame_index": None,
                "review_window_start_frame": 0,
                "review_window_end_frame": enroll_end,
                "candidate_manifest_path": enroll_name,
                "candidate_manifest_sha256": enroll_sha,
                "candidate_count": len(enroll_cands),
                "requires_calibration": False,
                "evidence_paths": [],
                "evidence_sha256": [],
                "provenance": {
                    "package_mode": PACKAGE_MODE,
                    "ui_profile": UI_PROFILE,
                    "match_id": identity.match_id,
                    "analysis_run_id": identity.analysis_run_id,
                    "gt_prefill": False,
                    "automatic_identity": False,
                    "old_development_gallery": False,
                    "legacy_ext_mapping_fallback": False,
                },
                "metadata": {"priority": "required_enrollment"},
            }
        )
    ]

    cand_paths = [enroll_name]
    cand_shas = {enroll_name: enroll_sha}

    # Recovery events: longest mid-video terminations; ALL eligible tracks in window
    mid_ends = [
        r
        for r in mapping_rows
        if r.get("review_eligible", True)
        and int(r["last_frame"]) < frame_count - 30
        and int(r["observation_count"]) >= 30
    ]
    mid_ends.sort(key=lambda r: int(r["observation_count"]), reverse=True)
    for row in mid_ends[:max_recovery_events]:
        code = str(row["external_candidate_code"])
        lost = int(row["last_frame"])
        win_start = lost + 1
        win_end = min(frame_count - 1, lost + 60)
        if win_start > win_end:
            continue
        codes = _tracks_active_in_window(mapping_rows, start=win_start, end=win_end)
        if not codes:
            continue
        event_id = f"evt_sv_reentry_{code}"
        cands = _candidates_from_rows(
            mapping_rows,
            event_id=event_id,
            target_id=identity.target_id,
            codes=codes,
        )
        man = validate_candidate_manifest(
            {
                "schema_version": "target_recovery_candidate_manifest_v1",
                "event_id": event_id,
                "target_id": identity.target_id,
                "candidate_count": len(cands),
                "eligible_count": len(cands),
                "supports_direct_bbox_selection": True,
                "appearance_rank_is_helper_only": True,
                "rank_does_not_hide_candidates": True,
                "candidates": cands,
            }
        )
        man_name = f"reentry_{code}_candidate_manifest.json"
        _write_json(root / man_name, man)
        man_sha = sha256_file(root / man_name)
        cand_paths.append(man_name)
        cand_shas[man_name] = man_sha
        events.append(
            validate_recovery_event(
                {
                    "schema_version": "target_recovery_event_v1",
                    "event_id": event_id,
                    "project_id": "football-analytics",
                    "run_id": identity.analysis_run_id,
                    "target_id": identity.target_id,
                    "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
                    "video_id": identity.video_id,
                    "video_path": str(video_abs),
                    "video_sha256": video_sha,
                    "created_at": _utc(),
                    "status": "open",
                    "trigger_source": "short_video_track_termination",
                    "trigger_reason": (
                        f"raw track {row['raw_external_track_id']} / {code} ended @ {lost}; "
                        "all eligible re-entry tracks listed; no automatic switch"
                    ),
                    "last_confirmed_segment_id": _segment_id(code),
                    "last_confirmed_frame_index": lost,
                    "review_window_start_frame": win_start,
                    "review_window_end_frame": win_end,
                    "candidate_manifest_path": man_name,
                    "candidate_manifest_sha256": man_sha,
                    "candidate_count": len(cands),
                    "requires_calibration": False,
                    "evidence_paths": [],
                    "evidence_sha256": [],
                    "provenance": {
                        "package_mode": PACKAGE_MODE,
                        "match_id": identity.match_id,
                        "lost_raw_track_id": str(row["raw_external_track_id"]),
                        "gt_prefill": False,
                        "automatic_identity": False,
                        "top_k_only": False,
                    },
                    "metadata": {
                        "priority": "recovery",
                        "previous_segment_id": _segment_id(code),
                        "previous_raw_track_id": str(row["raw_external_track_id"]),
                    },
                }
            )
        )

    events_path = root / "recovery_events.jsonl"
    _write_jsonl(events_path, events)
    events_sha = sha256_file(events_path)
    inv_sha = sha256_file(inv_path)

    for name in ("decision_log.jsonl", "timeline_approval_log.jsonl", "gallery_approval_log.jsonl"):
        p = writable / name
        if not p.exists():
            p.write_text("", encoding="utf-8")

    write_provisional(
        writable / "provisional_timeline.json",
        empty_provisional(target_id=identity.target_id, video_id=identity.video_id),
    )
    _write_json(
        writable / "approved_timeline.json",
        {
            "schema_version": "short_video_approved_timeline_v1",
            "target_id": identity.target_id,
            "video_id": identity.video_id,
            "intervals": [],
            "unresolved_intervals": [],
            "analysis_eligible": True,
            "note": "Populated only after active decision + Timeline Approval",
        },
    )

    target_profile = {
        "schema_version": "short_video_target_profile_v1",
        **identity.to_dict(),
        "fps": fps,
        "frame_count": frame_count,
        "resolution": {"width": width, "height": height},
        "created_at": _utc(),
    }
    _write_json(root / "target_profile.json", target_profile)

    data_ro = str((project_root / "data").resolve())
    preprocess_ro = str(preprocess_root.resolve())

    package = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": identity.product_package_id,
        "project_id": "football-analytics",
        "run_id": identity.analysis_run_id,
        "target_id": identity.target_id,
        "source_video_path": str(video_abs.resolve()),
        "source_video_sha256": video_sha,
        "event_manifest_path": "recovery_events.jsonl",
        "event_manifest_sha256": events_sha,
        "candidate_manifest_paths": cand_paths,
        "candidate_manifest_sha256": cand_shas,
        "decision_log_path": "decision_log.jsonl",
        "target_gallery_reference": {
            "note": "gallery optional; quality gate required; not success criterion",
            "path": None,
            "development_gallery_forbidden": True,
        },
        "model_metadata": {
            "inference_for_confirm": False,
            "role": "no_appearance_helper_required",
            "sportsreid_scores_present": False,
        },
        "media_root": str((project_root / "data").resolve()),
        "read_only_source_roots": [data_ro, preprocess_ro],
        "writable_session_root": "session",
        "created_at": _utc(),
        "provenance": {
            "mode": "product",
            "package_mode": PACKAGE_MODE,
            "ui_profile": UI_PROFILE,
            "match_id": identity.match_id,
            "analysis_run_id": identity.analysis_run_id,
            "video_id": identity.video_id,
            "composite_key": identity.composite_key,
            "dense_bbox_timeline_path": "dense_bbox_timeline.json",
            "track_mapping_path": "track_candidate_mapping.jsonl",
            "segment_inventory_path": str(inv_path),
            "segment_inventory_sha256": inv_sha,
            "legacy_ext_004_183_198_fallback": False,
            "game_state": False,
            "clip": False,
            "automatic_identity": False,
            "old_development_gallery": False,
            "fps": fps,
            "frame_count": frame_count,
            "video_width": width,
            "video_height": height,
            "approval_log_path": str(writable / "timeline_approval_log.jsonl"),
        },
        "media_status": "verified",
        "metadata": {
            "product_next_gate": "USER_ACTION_INTERACTIVE_TARGET_SELECTION",
            "gallery_required": False,
        },
    }
    validated = validate_review_package(package, package_dir=root, verify_sources=True)
    _write_json(root / "review_package.json", package)
    return {
        "package": validated,
        "event_count": len(events),
        "candidate_count_enrollment": len(enroll_cands),
        "dense_path": str(dens_path),
        "recovery_event_count": max(0, len(events) - 1),
    }
