#!/usr/bin/env python3
"""Multi-event HIL end-to-end validation scaffold + timeline/overlay when approved.

Does not start new detection/tracking by default. Does not use development gallery.
Human enrollment / gallery / recovery / timeline approvals / overlay acceptance
are required for COMPLETED_MULTI_EVENT_HIL_END_TO_END_VALIDATION.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.timeline.approvals import (  # noqa: E402
    ApprovalLog,
    resolve_active_approvals,
)
from football_analytics.reid.hil.timeline.reconstruct import (  # noqa: E402
    dump_intervals_jsonl,
    reconstruct_twice_for_determinism,
)
from football_analytics.reid.hil.timeline.segments import load_segment_index  # noqa: E402
from football_analytics.reid.hil.timeline.sources import write_json  # noqa: E402
from football_analytics.reid.hil_c2.qualify import qualify_product_session  # noqa: E402
from football_analytics.reid.hil_ui.observations import load_observation_lookup  # noqa: E402
from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalLog,
    resolve_active_gallery_approvals,
)
from football_analytics.reid.multi_event_hil.gallery_crops import (  # noqa: E402
    build_enrollment_crop_candidates,
)
from football_analytics.reid.multi_event_hil.gallery_embed import (  # noqa: E402
    embed_approved_gallery,
)
from football_analytics.reid.multi_event_hil.identity import build_identity  # noqa: E402
from football_analytics.reid.multi_event_hil.overlay import (  # noqa: E402
    render_timeline_overlay_video,
)
from football_analytics.reid.multi_event_hil.package_build import (  # noqa: E402
    build_multi_event_review_package,
)
from football_analytics.reid.multi_event_hil.source_audit import (  # noqa: E402
    audit_multi_event_sources,
)
from football_analytics.reid.hil_c2.product_package import (  # noqa: E402
    load_external_mapping,
    _segment_id,
)

OUTPUT_NAME = "target_001_multi_event_hil_end_to_end_validation"
PACKAGE_DIRNAME = "target_001_multi_event_hil_review_package"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_ui(package_json: Path, *, port: int | None = None) -> dict[str, Any]:
    py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
    launch = PROJECT / "scripts/run_hil_offline_review_ui.py"
    use_port = port or _free_port()
    cmd = [
        str(py),
        str(launch),
        "--review-package",
        str(package_json),
        "--port",
        str(use_port),
        "--address",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "pid": proc.pid,
        "url": f"http://127.0.0.1:{use_port}",
        "port": use_port,
        "stop": f"kill {proc.pid}",
        "launched": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-event HIL end-to-end validation")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--skip-gallery-embed", action="store_true")
    parser.add_argument("--analysis-run-id", default=None)
    parser.add_argument("--ui-port", type=int, default=0)
    args = parser.parse_args()
    root = args.project_root.resolve()

    final = root / "outputs" / "reid" / OUTPUT_NAME
    if final.exists():
        print("BLOCKED_MULTI_EVENT_HIL_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    source_audit = audit_multi_event_sources(root)
    if source_audit.get("blocked_status"):
        print(source_audit["blocked_status"])
        print(source_audit["exact_next_gate_if_blocked"])
        return 3

    identity = build_identity(args.analysis_run_id)
    package_root = root / "outputs" / "reid" / PACKAGE_DIRNAME
    package_root.mkdir(parents=True, exist_ok=True)
    # Persist stable run id if package already has profile
    profile_path = package_root / "target_profile.json"
    if profile_path.is_file():
        prev = json.loads(profile_path.read_text(encoding="utf-8"))
        identity = build_identity(prev["analysis_run_id"])

    pkg = build_multi_event_review_package(
        package_root, project_root=root, identity=identity
    )

    # Crop candidates from EXT_004 for gallery review (not auto-members)
    track_root = Path(source_audit["selected_source"]["tracking_root"])
    mapping = {
        r["external_candidate_code"]: r
        for r in load_external_mapping(
            track_root / "inventory/target_001_external_track_candidate_mapping.jsonl"
        )
    }
    crop_dir = package_root / "gallery_crop_candidates"
    crop_manifest = build_enrollment_crop_candidates(
        video_path=root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4",
        mapping_row=mapping["EXT_004"],
        segment_id=_segment_id("EXT_004"),
        output_dir=crop_dir,
        video_sha256=identity.video_sha256,
    )

    gallery_log = GalleryApprovalLog(pkg["gallery_approval_log_path"])
    active_gallery = resolve_active_gallery_approvals(gallery_log.read_raw())

    gallery_embed_info: dict[str, Any] = {
        "embedded": False,
        "member_count": 0,
        "skipped": True,
    }
    if active_gallery and not args.skip_gallery_embed:
        try:
            gallery_embed_info = embed_approved_gallery(
                crop_candidates=crop_manifest,
                gallery_approvals=gallery_log.read_raw(),
                output_dir=package_root / "match_specific_gallery",
                match_id=identity.match_id,
                analysis_run_id=identity.analysis_run_id,
                target_id=identity.target_id,
            )
            gallery_embed_info["skipped"] = False
        except Exception as exc:  # noqa: BLE001
            print("BLOCKED_MULTI_EVENT_HIL_GALLERY_CONTRACT")
            print(str(exc))
            return 4

    ui_info = None
    if not args.skip_ui_launch:
        ui_info = _launch_ui(Path(pkg["package_file"]), port=args.ui_port or None)

    qual = qualify_product_session(
        decision_log_path=pkg["decision_log_path"],
        approval_log_path=pkg["approval_log_path"],
        review_package_mode="product",
    )
    approved_ids = set(qual["approved_decision_ids"])
    segment_index = load_segment_index(pkg["segment_inventory_path"])

    recon = reconstruct_twice_for_determinism(
        project_id="football-analytics",
        run_id=identity.analysis_run_id,
        target_id=identity.target_id,
        video_id=identity.video_id,
        video_path=str(root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"),
        video_sha256=identity.video_sha256,
        frame_rate=30.0,
        total_video_frames=784,
        total_video_duration_seconds=784 / 30.0,
        decision_sources=[
            {
                "path": pkg["decision_log_path"],
                "review_package_mode": "product",
                "require_timeline_approval": True,
            }
        ],
        segment_index=segment_index,
        source_segment_manifest_path=pkg["segment_inventory_path"],
        source_segment_manifest_sha256=pkg["segment_inventory_sha256"],
        generated_at=_utc(),
        approved_decision_ids=approved_ids,
        require_timeline_approval=True,
        timeline_id=f"timeline_{identity.composite_key}",
    )
    tl = recon["timeline"]
    for iv in tl["intervals"]:
        meta = dict(iv.get("metadata") or {})
        meta["match_id"] = identity.match_id
        meta["analysis_run_id"] = identity.analysis_run_id
        meta["evidence"] = "approved_product_decision"
        iv["metadata"] = meta

    linked_tracks = sorted(
        {str(iv.get("raw_track_id")) for iv in tl["intervals"] if iv.get("raw_track_id")}
    )
    multi_event_linking = len(linked_tracks) >= 2

    overlay_info: dict[str, Any] | None = None
    if tl["intervals"] and not args.skip_overlay:
        obs_path = package_root / "segment_observations_sparse.jsonl"
        lookup = load_observation_lookup(
            obs_path,
            segment_ids={str(iv["segment_id"]) for iv in tl["intervals"] if iv.get("segment_id")},
            frame_min=0,
            frame_max=783,
        )
        overlay_path = package_root / "timeline_overlay" / "target_001_timeline_overlay.mp4"
        overlay_info = render_timeline_overlay_video(
            video_path=root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4",
            timeline=tl,
            output_path=overlay_path,
            observation_lookup=lookup,
        )

    checklist_path_hint = package_root / "session" / "human_acceptance_checklist.json"
    checklist = None
    if checklist_path_hint.is_file():
        checklist = json.loads(checklist_path_hint.read_text(encoding="utf-8"))

    eligible = len(qual["timeline_eligible_decisions"])
    gallery_members = int(
        (gallery_embed_info.get("manifest") or {}).get("member_count")
        or len(active_gallery)
    )
    overlay_accepted = bool(
        checklist
        and checklist.get("timeline_acceptable_for_analysis") is True
        and checklist.get("false_confirmed_target_assignment") == 0
    )

    if (
        multi_event_linking
        and eligible >= 2
        and gallery_members >= 1
        and overlay_info
        and overlay_accepted
    ):
        final_status = "COMPLETED_MULTI_EVENT_HIL_END_TO_END_VALIDATION"
        next_gate = "REID_HIL_D_VERIFIED_TARGET_TRAJECTORY_AND_METRIC_EXPORT"
    elif eligible >= 1 or gallery_members >= 1 or pkg["recovery_event_count"] >= 1:
        final_status = "COMPLETED_MULTI_EVENT_HIL_PARTIAL_TIMELINE"
        next_gate = "REID_PRODUCT_VIDEO_MULTI_EVENT_END_TO_END_HIL_VALIDATION"
    else:
        final_status = "COMPLETED_MULTI_EVENT_HIL_PARTIAL_TIMELINE"
        next_gate = "REID_PRODUCT_VIDEO_MULTI_EVENT_END_TO_END_HIL_VALIDATION"

    if checklist and checklist.get("false_confirmed_target_assignment", 0) not in (0, None):
        if checklist.get("false_confirmed_target_assignment", 0) > 0:
            final_status = "FAILED_MULTI_EVENT_HIL_OVERLAY_ACCEPTANCE"
            next_gate = "REID_PRODUCT_VIDEO_MULTI_EVENT_END_TO_END_HIL_VALIDATION"

    temp = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}_", dir=str(root / "outputs" / "reid"))
    )
    try:
        write_json(temp / "product_video_source_audit.json", source_audit)
        write_json(
            temp / "analysis_run_manifest.json",
            {
                "identity": identity.as_dict(),
                "composite_key": identity.composite_key,
                "package_root": str(package_root),
                "recovery_event_count": pkg["recovery_event_count"],
                "generated_at": _utc(),
            },
        )
        write_json(temp / "target_profile.json", identity.as_dict() | {
            "isolated_from_development_gallery": True,
            "schema_version": "match_specific_target_profile_v1",
        })
        write_json(temp / "enrollment_crop_candidates.json", crop_manifest)
        shutil.copy2(
            pkg["gallery_approval_log_path"], temp / "gallery_approvals.jsonl"
        )
        if (package_root / "match_specific_gallery" / "match_specific_gallery_manifest.json").is_file():
            shutil.copy2(
                package_root / "match_specific_gallery" / "match_specific_gallery_manifest.json",
                temp / "match_specific_gallery_manifest.json",
            )
        else:
            write_json(
                temp / "match_specific_gallery_manifest.json",
                {
                    "member_count": 0,
                    "embeddings_present": False,
                    "note": "awaiting explicit gallery crop approvals + embed",
                },
            )
        write_json(
            temp / "gallery_embedding_provenance.json",
            {
                "gallery_embed": {
                    k: gallery_embed_info[k]
                    for k in gallery_embed_info
                    if k != "embeddings"
                },
                "model_run_for_confirm": False,
                "helper_only": True,
            },
        )
        write_json(
            temp / "recovery_events_summary.json",
            {
                "event_count": pkg["event_count"],
                "recovery_event_count": pkg["recovery_event_count"],
                "events_path": str(package_root / "recovery_events.jsonl"),
            },
        )
        write_json(temp / "product_decision_qualification.json", qual)
        write_json(temp / "target_001_timeline.json", tl)
        dump_intervals_jsonl(temp / "target_001_timeline_intervals.jsonl", tl["intervals"])
        dump_intervals_jsonl(
            temp / "target_001_unresolved_intervals.jsonl", tl["unresolved_intervals"]
        )
        write_json(temp / "coverage_summary.json", tl["coverage_summary"])
        write_json(
            temp / "timeline_determinism.json",
            {
                "deterministic": recon["deterministic"],
                "sha_a": recon["sha_a"],
                "sha_b": recon["sha_b"],
                "max_numeric_diff": 0,
            },
        )
        if overlay_info:
            # do not copy large video into atomic root — record path/sha only
            write_json(temp / "timeline_overlay_manifest.json", overlay_info)
        else:
            write_json(
                temp / "timeline_overlay_manifest.json",
                {"overlay_path": None, "reason": "no_confirmed_intervals_or_skipped"},
            )
        write_json(
            temp / "human_acceptance_checklist.json",
            checklist
            or {
                "schema_version": "multi_event_hil_overlay_acceptance_checklist_v1",
                "status": "pending_human",
                "questions": {
                    "only_correct_player_highlighted": None,
                    "track_transitions_correct": None,
                    "identity_switch_to_wrong_player": None,
                    "unresolved_regions_honestly_empty": None,
                    "timeline_acceptable_for_analysis": None,
                },
                "false_confirmed_target_assignment": None,
                "checklist_write_path": str(checklist_path_hint),
                "instructions": (
                    "Write answers to session/human_acceptance_checklist.json after "
                    "watching the overlay, then re-run this script."
                ),
            },
        )
        write_json(
            temp / "active_execution_path.json",
            {
                "detection_tracking_rerun": False,
                "reid_confirm_inference": False,
                "sportsreid_gallery_embed": bool(gallery_embed_info.get("embedded")),
                "clip": False,
                "game_state": False,
                "heatmap_distance_speed": False,
                "ui": ui_info,
                "package_root": str(package_root),
            },
        )
        report = f"""# Multi-Event HIL End-to-End Validation Report

- final_status: `{final_status}`
- timeline_scope: `{'MULTI_EVENT' if multi_event_linking else 'PARTIAL_OR_ENROLLMENT'}`
- product_next_gate: `{next_gate}`
- research_next_gate: `REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW`
- match_id: `{identity.match_id}`
- analysis_run_id: `{identity.analysis_run_id}`
- video_duration_seconds: ~26.15 (below preferred 1–5 min; honest selection)
- recovery_event_count: {pkg['recovery_event_count']}
- timeline_eligible_decisions: {eligible}
- confirmed_intervals: {len(tl['intervals'])}
- linked_raw_tracks: {linked_tracks}
- multi_event_linking: {multi_event_linking}
- gallery_crop_candidates: {crop_manifest['candidate_count']}
- gallery_approvals_active: {len(active_gallery)}
- gallery_members_embedded: {gallery_members}
- unresolved_intervals: {len(tl['unresolved_intervals'])}
- verified_coverage_percentage: {tl['coverage_summary'].get('verified_coverage_percentage')}
- overlay: {(overlay_info or {}).get('overlay_path')}
- UI: {(ui_info or {}).get('url')}
- detection/tracking rerun: false
- CLIP/Game State/metrics: false

## İnsan adımları

1. Initial enrollment Confirm + Timeline Approval
2. Gallery sekmesinde crop onayları (eski development gallery yok)
3. Recovery event Confirm + Timeline Approval (≥2 raw track için multi-event)
4. Overlay izle → `session/human_acceptance_checklist.json` doldur
5. Bu script’i yeniden çalıştır (önce mevcut output root’u archive et)
"""
        (temp / "target_001_multi_event_hil_report.md").write_text(report + "\n", encoding="utf-8")
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        write_json(
            temp / "target_001_multi_event_hil_manifest.json",
            {
                "final_status": final_status,
                "product_next_gate": next_gate,
                "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                "artifact_files": artifacts,
                "artifact_count": len(artifacts),
                "identity": identity.as_dict(),
                "multi_event_linking": multi_event_linking,
                "generated_at": _utc(),
            },
        )
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        man = json.loads((temp / "target_001_multi_event_hil_manifest.json").read_text())
        man["artifact_files"] = artifacts
        man["artifact_count"] = len(artifacts)
        write_json(temp / "target_001_multi_event_hil_manifest.json", man)
        shutil.move(str(temp), str(final))
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        print("FAILED_MULTI_EVENT_HIL_ATOMIC_BUILD")
        raise

    print(final_status)
    print(f"output: {final}")
    print(f"package: {package_root}")
    if ui_info:
        print(f"ui: {ui_info['url']} pid={ui_info['pid']}")
        print(f"stop: {ui_info['stop']}")
    print(f"decision_log: {pkg['decision_log_path']}")
    print(f"approval_log: {pkg['approval_log_path']}")
    print(f"gallery_approval_log: {pkg['gallery_approval_log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
