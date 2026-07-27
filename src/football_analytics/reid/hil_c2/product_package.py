"""Build a product-mode target recovery review package from external enrollment artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.candidates import validate_candidate_manifest
from football_analytics.reid.hil.common import sha256_file
from football_analytics.reid.hil.events import EventType, validate_recovery_event
from football_analytics.reid.hil_ui.package import (
    REVIEW_PACKAGE_SCHEMA_VERSION,
    validate_review_package,
)

PRODUCT_PACKAGE_ID = "pkg_hil_c2_product_external_enrollment_v1"
PRODUCT_RUN_ID = "hil_c2_product_external_enrollment"
VIDEO_ID = "target_001_external_enrollment_v1"
VIDEO_REL = "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
VIDEO_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


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


def _segment_id(code: str) -> str:
    return f"EXT_SEG_{code.replace('EXT_', '')}"


def load_external_mapping(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_segment_inventory_from_mapping(
    mapping_rows: list[dict[str, Any]],
    *,
    video_sha256: str,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
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
                "video_sha256": video_sha256,
                "source": "external_track_candidate_mapping_adapter",
            }
        )
    return inventory


def _bbox_refs_for_row(row: dict[str, Any], *, max_refs: int = 3) -> list[dict[str, Any]]:
    obs = list(row.get("bbox_per_observation") or [])
    if not obs:
        return []
    idxs = [0]
    if len(obs) > 1:
        idxs.append(len(obs) // 2)
    if len(obs) > 2:
        idxs.append(len(obs) - 1)
    # unique preserve order
    seen: set[int] = set()
    refs: list[dict[str, Any]] = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        item = obs[i]
        refs.append(
            {
                "frame_index": int(item["frame_index"]),
                "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
            }
        )
        if len(refs) >= max_refs:
            break
    return refs


def _candidates_from_codes(
    mapping_by_code: dict[str, dict[str, Any]],
    codes: list[str],
    *,
    event_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i, code in enumerate(codes, start=1):
        row = mapping_by_code[code]
        refs = _bbox_refs_for_row(row)
        mid = refs[len(refs) // 2]["frame_index"] if refs else int(row["first_frame"])
        candidates.append(
            {
                "candidate_id": f"prod_cand_{i:03d}",
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
                "metadata": {"external_candidate_code": code, "event_id": event_id},
            }
        )
    return candidates


def build_product_external_review_package(
    root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Create a new product package (never reuse HIL-B existing_artifact package)."""
    root = root.resolve()
    project_root = project_root.resolve()
    writable = root / "session"
    writable.mkdir(parents=True, exist_ok=True)

    video_abs = project_root / VIDEO_REL
    video_sha = sha256_file(video_abs)
    if video_sha != VIDEO_SHA:
        raise RuntimeError(
            f"external enrollment video SHA mismatch: expected {VIDEO_SHA}, got {video_sha}"
        )

    track_root = (
        project_root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
    )
    mapping_path = track_root / "inventory/target_001_external_track_candidate_mapping.jsonl"
    mapping_rows = load_external_mapping(mapping_path)
    mapping_by_code = {r["external_candidate_code"]: r for r in mapping_rows}

    # Known human-reviewed occurrence codes from external occurrence freeze.
    enrollment_codes = ["EXT_004", "EXT_001", "EXT_002", "EXT_003"]
    enrollment_codes = [c for c in enrollment_codes if c in mapping_by_code]
    if "EXT_004" not in enrollment_codes:
        raise RuntimeError("EXT_004 missing from external track mapping")

    inventory = build_segment_inventory_from_mapping(mapping_rows, video_sha256=video_sha)
    inv_path = root / "product_segment_inventory.jsonl"
    _write_jsonl(inv_path, inventory)
    inv_sha = sha256_file(inv_path)

    # Sparse observation index for UI (candidate segments only; no full track copy).
    obs_rows: list[dict[str, Any]] = []
    for code in enrollment_codes + ["EXT_183", "EXT_198"]:
        if code not in mapping_by_code:
            continue
        row = mapping_by_code[code]
        for item in row.get("bbox_per_observation") or []:
            # Keep every Nth + ends to bound size without inventing bboxes
            fi = int(item["frame_index"])
            if fi % 5 != 0 and fi != int(row["first_frame"]) and fi != int(row["last_frame"]):
                continue
            obs_rows.append(
                {
                    "segment_id": _segment_id(code),
                    "raw_track_code": str(row["raw_external_track_id"]),
                    "frame_index": fi,
                    "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                }
            )
    obs_path = root / "product_segment_observations_sparse.jsonl"
    _write_jsonl(obs_path, obs_rows)

    enroll_event_id = "evt_product_initial_enrollment_001"
    enroll_cands = _candidates_from_codes(
        mapping_by_code, enrollment_codes, event_id=enroll_event_id
    )
    enroll_manifest = validate_candidate_manifest(
        {
            "schema_version": "target_recovery_candidate_manifest_v1",
            "event_id": enroll_event_id,
            "target_id": "target_001",
            "candidate_count": len(enroll_cands),
            "eligible_count": len(enroll_cands),
            "supports_direct_bbox_selection": True,
            "appearance_rank_is_helper_only": True,
            "rank_does_not_hide_candidates": True,
            "candidates": enroll_cands,
        }
    )
    enroll_man_path = root / "product_enrollment_candidate_manifest.json"
    _write_json(enroll_man_path, enroll_manifest)
    enroll_man_sha = sha256_file(enroll_man_path)

    enroll_frame = int(mapping_by_code["EXT_004"]["first_frame"])
    enroll_end = min(enroll_frame + 30, int(mapping_by_code["EXT_004"]["last_frame"]))
    enrollment_event = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": enroll_event_id,
            "project_id": "football-analytics",
            "run_id": PRODUCT_RUN_ID,
            "target_id": "target_001",
            "event_type": EventType.INITIAL_TARGET_ENROLLMENT.value,
            "video_id": VIDEO_ID,
            "video_path": str(video_abs),
            "video_sha256": video_sha,
            "created_at": _utc_now(),
            "status": "open",
            "trigger_source": "product_hil_c2",
            "trigger_reason": "human initial target enrollment required for product timeline",
            "last_confirmed_segment_id": None,
            "last_confirmed_frame_index": None,
            "review_window_start_frame": enroll_frame,
            "review_window_end_frame": enroll_end,
            "candidate_manifest_path": "product_enrollment_candidate_manifest.json",
            "candidate_manifest_sha256": enroll_man_sha,
            "candidate_count": len(enroll_cands),
            "requires_calibration": False,
            "evidence_paths": [],
            "evidence_sha256": [],
            "provenance": {
                "package_mode": "product",
                "adapter": "external_track_candidate_mapping",
                "suggested_occurrence_code": "EXT_004",
                "gallery_reference_only": True,
                "gt_prefill": False,
                "automatic_identity": False,
            },
            "metadata": {"priority": "required_enrollment"},
        }
    )

    events = [enrollment_event]
    cand_paths = ["product_enrollment_candidate_manifest.json"]
    cand_shas = {"product_enrollment_candidate_manifest.json": enroll_man_sha}

    # Recovery readiness: later occurrence tracks exist; thresholds not invented.
    recovery_readiness = {
        "schema_version": "hil_c2_recovery_event_readiness_v1",
        "initial_confirmed_target_segment": "pending_human_enrollment",
        "known_occurrence_codes": ["EXT_004", "EXT_183", "EXT_198"],
        "recovery_events_auto_generated": False,
        "requires_calibration": True,
        "reason": (
            "Numeric reentry confidence thresholds not calibrated in HIL-C2; "
            "partial timeline may proceed from approved initial enrollment only."
        ),
        "candidate_tracklets_present": {
            "EXT_183": "EXT_183" in mapping_by_code,
            "EXT_198": "EXT_198" in mapping_by_code,
        },
    }

    # Optional review-only reentry windows (human must decide; requires_calibration).
    for code, evt_suffix in (("EXT_183", "183"), ("EXT_198", "198")):
        if code not in mapping_by_code:
            continue
        row = mapping_by_code[code]
        event_id = f"evt_product_reentry_{evt_suffix}"
        # Include target occurrence + nearby-in-time codes on same start frame neighborhood
        codes = [code]
        start_f = int(row["first_frame"])
        for other in mapping_rows:
            oc = other["external_candidate_code"]
            if oc == code:
                continue
            if abs(int(other["first_frame"]) - start_f) <= 2 and len(codes) < 4:
                codes.append(oc)
        cands = _candidates_from_codes(mapping_by_code, codes, event_id=event_id)
        man = validate_candidate_manifest(
            {
                "schema_version": "target_recovery_candidate_manifest_v1",
                "event_id": event_id,
                "target_id": "target_001",
                "candidate_count": len(cands),
                "eligible_count": len(cands),
                "supports_direct_bbox_selection": True,
                "appearance_rank_is_helper_only": True,
                "rank_does_not_hide_candidates": True,
                "candidates": cands,
            }
        )
        man_name = f"product_reentry_{evt_suffix}_candidate_manifest.json"
        man_path = root / man_name
        _write_json(man_path, man)
        man_sha = sha256_file(man_path)
        cand_paths.append(man_name)
        cand_shas[man_name] = man_sha
        end_f = min(start_f + 20, int(row["last_frame"]))
        events.append(
            validate_recovery_event(
                {
                    "schema_version": "target_recovery_event_v1",
                    "event_id": event_id,
                    "project_id": "football-analytics",
                    "run_id": PRODUCT_RUN_ID,
                    "target_id": "target_001",
                    "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
                    "video_id": VIDEO_ID,
                    "video_path": str(video_abs),
                    "video_sha256": video_sha,
                    "created_at": _utc_now(),
                    "status": "open",
                    "trigger_source": "product_hil_c2_adapter",
                    "trigger_reason": (
                        f"occurrence tracklet {code} available; "
                        "identity not auto-assigned; calibration required for auto trigger"
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
                        "gt_prefill": False,
                        "automatic_identity": False,
                        "occurrence_code": code,
                    },
                    "metadata": {"priority": "optional_recovery"},
                }
            )
        )

    events_path = root / "product_recovery_events.jsonl"
    _write_jsonl(events_path, events)
    events_sha = sha256_file(events_path)

    decision_rel = "decision_log.jsonl"
    decision_path = writable / decision_rel
    if not decision_path.exists():
        decision_path.write_text("", encoding="utf-8")
    approval_path = writable / "timeline_approval_log.jsonl"
    if not approval_path.exists():
        approval_path.write_text("", encoding="utf-8")

    gallery_path = (
        project_root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1/gallery/target_001_gallery_members.jsonl"
    )
    ro_roots = [
        str(project_root / "data"),
        str(track_root),
        str(
            project_root
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1"
        ),
    ]

    package = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": PRODUCT_PACKAGE_ID,
        "project_id": "football-analytics",
        "run_id": PRODUCT_RUN_ID,
        "target_id": "target_001",
        "source_video_path": str(video_abs),
        "source_video_sha256": video_sha,
        "event_manifest_path": "product_recovery_events.jsonl",
        "event_manifest_sha256": events_sha,
        "candidate_manifest_paths": cand_paths,
        "candidate_manifest_sha256": cand_shas,
        "decision_log_path": decision_rel,
        "target_gallery_reference": {
            "note": "reference only; not auto-applied; not timeline approval",
            "path": str(gallery_path) if gallery_path.is_file() else None,
        },
        "model_metadata": {
            "inference": False,
            "sportsreid_scores_present": False,
            "role": "none",
        },
        "media_root": str(project_root / "data/enrollment_clips"),
        "read_only_source_roots": ro_roots,
        "writable_session_root": "session",
        "created_at": _utc_now(),
        "provenance": {
            "mode": "product",
            "package_mode": "product",
            "development_only": False,
            "fixture": False,
            "acceptance": False,
            "gt_prefill": False,
            "automatic_identity": False,
            "adapter": "external_enrollment_track_mapping_v1",
            "segment_inventory_path": str(inv_path),
            "segment_inventory_sha256": inv_sha,
            "observation_index_path": str(obs_path),
            "approval_log_path": str(approval_path),
            "recovery_readiness": recovery_readiness,
            "not_reused_existing_artifact_package": True,
            "sparse_observations": True,
        },
        "media_status": "verified",
    }
    validated = validate_review_package(package, package_dir=root, verify_sources=True)
    out = root / "review_package.json"
    _write_json(out, package)
    _write_json(root / "recovery_event_readiness.json", recovery_readiness)
    validated["package_file"] = str(out)
    validated["segment_inventory_path"] = str(inv_path)
    validated["segment_inventory_sha256"] = inv_sha
    validated["approval_log_path"] = str(approval_path)
    validated["decision_log_path"] = str(decision_path)
    validated["recovery_readiness"] = recovery_readiness
    validated["event_count"] = len(events)
    validated["candidate_manifest_count"] = len(cand_paths)
    return validated
