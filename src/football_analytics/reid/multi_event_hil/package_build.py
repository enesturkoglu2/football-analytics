"""Build isolated multi-event product review package (no development gallery)."""

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
from football_analytics.reid.hil_c2.product_package import (
    VIDEO_REL,
    VIDEO_SHA,
    _bbox_refs_for_row,
    _segment_id,
    build_segment_inventory_from_mapping,
    load_external_mapping,
)
from football_analytics.reid.hil_ui.package import (
    REVIEW_PACKAGE_SCHEMA_VERSION,
    validate_review_package,
)
from football_analytics.reid.multi_event_hil.identity import AnalysisIdentity


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


def _candidates_from_codes(
    mapping_by_code: dict[str, dict[str, Any]],
    codes: list[str],
    *,
    event_id: str,
    ranks: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i, code in enumerate(codes, start=1):
        row = mapping_by_code[code]
        refs = _bbox_refs_for_row(row)
        mid = refs[len(refs) // 2]["frame_index"] if refs else int(row["first_frame"])
        rank_meta = (ranks or {}).get(code) or {}
        candidates.append(
            {
                "candidate_id": f"mehil_cand_{i:03d}",
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
                "team_evidence": {"is_identity_proof": False},
                "visibility": {},
                "quality": {"source": "external_track_mapping"},
                "contamination": {},
                "sportsreid_model_id": rank_meta.get("sportsreid_model_id"),
                "sportsreid_checkpoint_sha256": rank_meta.get("sportsreid_checkpoint_sha256"),
                "appearance_rank": rank_meta.get("appearance_rank"),
                "T_max": rank_meta.get("T_max"),
                "D_max": rank_meta.get("D_max"),
                "S": rank_meta.get("S"),
                "temporal_distance": None,
                "spatial_distance": None,
                "eligibility": True,
                "rejection_reason": None,
                "display_order": i,
                "metadata": {"external_candidate_code": code, "event_id": event_id},
            }
        )
    return candidates


def build_multi_event_review_package(
    root: Path,
    *,
    project_root: Path,
    identity: AnalysisIdentity,
    helper_ranks: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Create a fresh product package isolated from HIL-C2 session logs and old galleries."""
    root = root.resolve()
    project_root = project_root.resolve()
    writable = root / "session"
    writable.mkdir(parents=True, exist_ok=True)

    video_abs = project_root / VIDEO_REL
    video_sha = sha256_file(video_abs)
    if video_sha != VIDEO_SHA or video_sha != identity.video_sha256:
        raise RuntimeError("video SHA mismatch for multi-event package")

    track_root = (
        project_root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
    )
    mapping_path = track_root / "inventory/target_001_external_track_candidate_mapping.jsonl"
    mapping_rows = load_external_mapping(mapping_path)
    mapping_by_code = {r["external_candidate_code"]: r for r in mapping_rows}
    if "EXT_004" not in mapping_by_code:
        raise RuntimeError("EXT_004 missing")

    inventory = build_segment_inventory_from_mapping(mapping_rows, video_sha256=video_sha)
    inv_path = root / "segment_inventory.jsonl"
    _write_jsonl(inv_path, inventory)
    inv_sha = sha256_file(inv_path)

    # Sparse observations for UI
    obs_rows: list[dict[str, Any]] = []
    for code in ("EXT_004", "EXT_001", "EXT_002", "EXT_003", "EXT_183", "EXT_198"):
        if code not in mapping_by_code:
            continue
        row = mapping_by_code[code]
        for item in row.get("bbox_per_observation") or []:
            fi = int(item["frame_index"])
            if fi % 5 != 0 and fi not in {int(row["first_frame"]), int(row["last_frame"])}:
                continue
            obs_rows.append(
                {
                    "segment_id": _segment_id(code),
                    "raw_track_code": str(row["raw_external_track_id"]),
                    "frame_index": fi,
                    "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                }
            )
    obs_path = root / "segment_observations_sparse.jsonl"
    _write_jsonl(obs_path, obs_rows)

    enroll_event_id = "evt_mehil_initial_enrollment_001"
    enroll_codes = [c for c in ("EXT_004", "EXT_001", "EXT_002", "EXT_003") if c in mapping_by_code]
    enroll_cands = _candidates_from_codes(
        mapping_by_code, enroll_codes, event_id=enroll_event_id
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
    enroll_man_name = "enrollment_candidate_manifest.json"
    _write_json(root / enroll_man_name, enroll_manifest)
    enroll_man_sha = sha256_file(root / enroll_man_name)

    enroll_frame = int(mapping_by_code["EXT_004"]["first_frame"])
    enroll_end = min(enroll_frame + 30, int(mapping_by_code["EXT_004"]["last_frame"]))
    events = [
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
                "created_at": _utc_now(),
                "status": "open",
                "trigger_source": "multi_event_hil",
                "trigger_reason": "isolated match-specific initial enrollment",
                "last_confirmed_segment_id": None,
                "last_confirmed_frame_index": None,
                "review_window_start_frame": enroll_frame,
                "review_window_end_frame": enroll_end,
                "candidate_manifest_path": enroll_man_name,
                "candidate_manifest_sha256": enroll_man_sha,
                "candidate_count": len(enroll_cands),
                "requires_calibration": False,
                "evidence_paths": [],
                "evidence_sha256": [],
                "provenance": {
                    "package_mode": "product",
                    "match_id": identity.match_id,
                    "analysis_run_id": identity.analysis_run_id,
                    "gt_prefill": False,
                    "automatic_identity": False,
                    "old_development_gallery": False,
                },
                "metadata": {"priority": "required_enrollment"},
            }
        )
    ]
    cand_paths = [enroll_man_name]
    cand_shas = {enroll_man_name: enroll_man_sha}

    recovery_codes = [c for c in ("EXT_183", "EXT_198") if c in mapping_by_code]
    for code in recovery_codes:
        evt_suffix = code.replace("EXT_", "")
        event_id = f"evt_mehil_reentry_{evt_suffix}"
        row = mapping_by_code[code]
        start_f = int(row["first_frame"])
        codes = [code]
        for other in mapping_rows:
            oc = other["external_candidate_code"]
            if oc == code:
                continue
            if abs(int(other["first_frame"]) - start_f) <= 2 and len(codes) < 6:
                codes.append(oc)
        ranks = (helper_ranks or {}).get(event_id)
        cands = _candidates_from_codes(
            mapping_by_code, codes, event_id=event_id, ranks=ranks
        )
        # Ensure all eligible listed — do not top-k hide
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
        man_name = f"reentry_{evt_suffix}_candidate_manifest.json"
        _write_json(root / man_name, man)
        man_sha = sha256_file(root / man_name)
        cand_paths.append(man_name)
        cand_shas[man_name] = man_sha
        end_f = min(start_f + 25, int(row["last_frame"]))
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
                    "created_at": _utc_now(),
                    "status": "open",
                    "trigger_source": "multi_event_hil_track_termination",
                    "trigger_reason": (
                        f"raw track for EXT_004 ended; occurrence {code} available; "
                        "SportsReID helper ranks optional; no auto CONFIRM"
                    ),
                    "last_confirmed_segment_id": _segment_id("EXT_004"),
                    "last_confirmed_frame_index": int(mapping_by_code["EXT_004"]["last_frame"]),
                    "review_window_start_frame": start_f,
                    "review_window_end_frame": end_f,
                    "candidate_manifest_path": man_name,
                    "candidate_manifest_sha256": man_sha,
                    "candidate_count": len(cands),
                    "requires_calibration": True,
                    "evidence_paths": [],
                    "evidence_sha256": [],
                    "provenance": {
                        "package_mode": "product",
                        "match_id": identity.match_id,
                        "occurrence_code": code,
                        "gt_prefill": False,
                        "automatic_identity": False,
                    },
                    "metadata": {"priority": "recovery"},
                }
            )
        )

    events_path = root / "recovery_events.jsonl"
    _write_jsonl(events_path, events)
    events_sha = sha256_file(events_path)

    decision_path = writable / "decision_log.jsonl"
    if not decision_path.exists():
        decision_path.write_text("", encoding="utf-8")
    approval_path = writable / "timeline_approval_log.jsonl"
    if not approval_path.exists():
        approval_path.write_text("", encoding="utf-8")
    gallery_approval_path = writable / "gallery_approval_log.jsonl"
    if not gallery_approval_path.exists():
        gallery_approval_path.write_text("", encoding="utf-8")

    profile = {
        "schema_version": "match_specific_target_profile_v1",
        "match_id": identity.match_id,
        "analysis_run_id": identity.analysis_run_id,
        "target_id": identity.target_id,
        "product_package_id": identity.product_package_id,
        "video_id": identity.video_id,
        "video_sha256": identity.video_sha256,
        "composite_key": identity.composite_key,
        "isolated_from_development_gallery": True,
        "isolated_from_hil_c2_session_logs": True,
        "automatic_identity": False,
        "created_at": _utc_now(),
    }
    _write_json(root / "target_profile.json", profile)

    package = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": identity.product_package_id,
        "project_id": "football-analytics",
        "run_id": identity.analysis_run_id,
        "target_id": identity.target_id,
        "source_video_path": str(video_abs),
        "source_video_sha256": video_sha,
        "event_manifest_path": "recovery_events.jsonl",
        "event_manifest_sha256": events_sha,
        "candidate_manifest_paths": cand_paths,
        "candidate_manifest_sha256": cand_shas,
        "decision_log_path": "decision_log.jsonl",
        "target_gallery_reference": {
            "note": "match-specific gallery only after explicit crop approvals; not development gallery",
            "path": None,
            "development_gallery_forbidden": True,
        },
        "model_metadata": {
            "inference_for_confirm": False,
            "helper_model_id": SPORTSREID_MODEL_ID,
            "helper_checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
            "role": "appearance_helper_ranker_only",
            "sportsreid_scores_present": bool(helper_ranks),
        },
        "media_root": str(project_root / "data/enrollment_clips"),
        "read_only_source_roots": [
            str(project_root / "data"),
            str(track_root),
        ],
        "writable_session_root": "session",
        "created_at": _utc_now(),
        "provenance": {
            "mode": "product",
            "package_mode": "product",
            "match_id": identity.match_id,
            "analysis_run_id": identity.analysis_run_id,
            "development_only": False,
            "fixture": False,
            "acceptance": False,
            "gt_prefill": False,
            "automatic_identity": False,
            "old_development_gallery": False,
            "segment_inventory_path": str(inv_path),
            "segment_inventory_sha256": inv_sha,
            "observation_index_path": str(obs_path),
            "approval_log_path": str(approval_path),
            "gallery_approval_log_path": str(gallery_approval_path),
            "recovery_event_count": len(recovery_codes),
            "sparse_observations": True,
        },
        "media_status": "verified",
    }
    validated = validate_review_package(package, package_dir=root, verify_sources=True)
    out = root / "review_package.json"
    _write_json(out, package)
    validated["package_file"] = str(out)
    validated["segment_inventory_path"] = str(inv_path)
    validated["segment_inventory_sha256"] = inv_sha
    validated["decision_log_path"] = str(decision_path)
    validated["approval_log_path"] = str(approval_path)
    validated["gallery_approval_log_path"] = str(gallery_approval_path)
    validated["event_count"] = len(events)
    validated["recovery_event_count"] = len(recovery_codes)
    validated["identity"] = identity.as_dict()
    validated["mapping_by_code"] = {
        k: {
            "raw_track_id": v["raw_external_track_id"],
            "first_frame": v["first_frame"],
            "last_frame": v["last_frame"],
            "observation_count": v["observation_count"],
        }
        for k, v in mapping_by_code.items()
        if k in {"EXT_004", "EXT_183", "EXT_198", "EXT_001", "EXT_002", "EXT_003"}
    }
    return validated
