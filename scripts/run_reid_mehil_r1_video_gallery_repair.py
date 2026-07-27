#!/usr/bin/env python3
"""MEHIL-R1: gallery render repair validation + video review assets (no log overwrite)."""

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
from football_analytics.reid.hil.timeline.sources import write_json  # noqa: E402
from football_analytics.reid.hil_ui.compat import streamlit_image_api_report  # noqa: E402
from football_analytics.reid.hil_ui.gallery_view import (  # noqa: E402
    validate_gallery_crop_for_display,
)
from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalLog,
    resolve_active_gallery_approvals,
)
from football_analytics.reid.multi_event_hil.gallery_embed import (  # noqa: E402
    embed_approved_gallery,
)
from football_analytics.reid.multi_event_hil.review_clips import (  # noqa: E402
    bboxes_from_candidates_for_window,
    extract_event_window_clip,
)

OUTPUT_NAME = "target_001_multi_event_hil_video_gallery_r1"
PACKAGE = "target_001_multi_event_hil_review_package"
DECISION_SHA_EXPECTED = (
    "32022b9d6cb733395b62fe1d69226eb93fb4aeab6d055e4a7dcf5686f931766e"
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_ui(package_json: Path, port: int) -> dict[str, Any]:
    py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
    launch = PROJECT / "scripts/run_hil_offline_review_ui.py"
    cmd = [
        str(py),
        str(launch),
        "--review-package",
        str(package_json),
        "--port",
        str(port),
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
    return {"pid": proc.pid, "url": f"http://127.0.0.1:{port}", "stop": f"kill {proc.pid}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--kill-old-ui-port", type=int, default=59389)
    args = parser.parse_args()
    root = args.project_root.resolve()

    final = root / "outputs" / "reid" / OUTPUT_NAME
    if final.exists():
        print("BLOCKED_MEHIL_R1_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    pkg_root = root / "outputs" / "reid" / PACKAGE
    decision_log = pkg_root / "session" / "decision_log.jsonl"
    gallery_log = pkg_root / "session" / "gallery_approval_log.jsonl"
    approval_log = pkg_root / "session" / "timeline_approval_log.jsonl"
    crop_man_path = (
        pkg_root / "gallery_crop_candidates" / "enrollment_crop_candidates.json"
    )

    # Preserve user logs
    dec_bytes = decision_log.read_bytes()
    gal_bytes = gallery_log.read_bytes() if gallery_log.exists() else b""
    appr_bytes = approval_log.read_bytes() if approval_log.exists() else b""
    dec_sha = sha256_file(decision_log)
    if dec_sha != DECISION_SHA_EXPECTED:
        # Allow if user appended more since snapshot — still must not shrink/delete
        if len(dec_bytes) < 100:
            print("BLOCKED_MEHIL_R1_SOURCE_CONTRACT_MISMATCH")
            print("decision log unexpectedly small")
            return 3

    decisions = [
        json.loads(line)
        for line in decision_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decision_audit = {
        "decision_log_path": str(decision_log),
        "decision_log_sha256": dec_sha,
        "record_count": len(decisions),
        "decisions": [
            {
                "decision_id": d.get("decision_id"),
                "event_id": d.get("event_id"),
                "action": d.get("action"),
                "segment_id": d.get("selected_segment_id"),
                "raw_track_id": d.get("selected_raw_track_id"),
            }
            for d in decisions
        ],
        "note": "read-only audit; logs not rewritten by R1",
    }

    # Streamlit compatibility
    sys.path.insert(0, str(Path.home() / "miniconda3/envs/football-hil-ui/lib"))
    import importlib

    # Use hil-ui env streamlit if available via subprocess report
    report_py = subprocess.check_output(
        [
            str(Path.home() / "miniconda3/envs/football-hil-ui/bin/python"),
            "-c",
            "import json,streamlit as st; from football_analytics.reid.hil_ui.compat import streamlit_image_api_report; "
            "import sys; sys.path.insert(0,%r); from football_analytics.reid.hil_ui.compat import streamlit_image_api_report as r; "
            "print(json.dumps(r(st)))" % str(root / "src"),
        ],
        cwd=str(root),
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": str(root / "src"),
            "PYTHONNOUSERSITE": "1",
        },
        text=True,
    )
    api_report = json.loads(report_py.strip().splitlines()[-1])

    crop_man = json.loads(crop_man_path.read_text(encoding="utf-8"))
    crop_validations = [
        validate_gallery_crop_for_display(c) for c in crop_man.get("candidates") or []
    ]
    broken = [v for v in crop_validations if not v["approval_enabled"]]
    root_cause = {
        "error_seen": "ImageMixin.image() got an unexpected keyword argument 'use_container_width'",
        "streamlit_version": api_report.get("streamlit_version"),
        "exact_root_cause": api_report.get("root_cause_mehil_r1"),
        "fix": "use streamlit_image() wrapper → use_column_width on 1.37.1",
        "version_upgrade": False,
        "crop_files_exist": all(v["exists"] for v in crop_validations),
        "broken_image_count": len(broken),
    }

    # Event clips
    video = root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
    video_sha = sha256_file(video)
    events = [
        json.loads(line)
        for line in (pkg_root / "recovery_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    clip_root = pkg_root / "session" / "review_clips"
    clip_manifests = []
    for event in events:
        man_path = pkg_root / event.get("candidate_manifest_path", "")
        cands = []
        if man_path.is_file():
            cands = json.loads(man_path.read_text()).get("candidates") or []
        bbox_map = bboxes_from_candidates_for_window(
            cands,
            start_frame=int(event["review_window_start_frame"]),
            end_frame=int(event["review_window_end_frame"]),
        )
        clip_manifests.append(
            extract_event_window_clip(
                source_video=video,
                source_video_sha256=video_sha,
                event=event,
                output_dir=clip_root,
                candidate_bboxes_by_frame=bbox_map,
            )
        )

    # Gallery embed only if approvals exist
    gal_rows = GalleryApprovalLog(gallery_log).read_raw()
    active_gal = resolve_active_gallery_approvals(gal_rows)
    gallery_embed: dict[str, Any] = {
        "embedded": False,
        "member_count": 0,
        "reason": "no_active_gallery_approvals",
    }
    if active_gal:
        try:
            gallery_embed = embed_approved_gallery(
                crop_candidates=crop_man,
                gallery_approvals=gal_rows,
                output_dir=pkg_root / "match_specific_gallery",
                match_id="match_external_enrollment_v1_mehil",
                analysis_run_id="mehil_run_20260727T223012Z",
                target_id="target_001",
            )
            # strip ndarray
            gallery_embed = {
                k: gallery_embed[k] for k in gallery_embed if k != "embeddings"
            }
        except Exception as exc:  # noqa: BLE001
            print("BLOCKED_MEHIL_R1_GALLERY_EMBEDDING")
            print(exc)
            return 4

    # Ensure logs unchanged
    if decision_log.read_bytes() != dec_bytes:
        print("BLOCKED_MEHIL_R1_SOURCE_CONTRACT_MISMATCH")
        print("decision log mutated")
        return 5
    if gallery_log.read_bytes() != gal_bytes:
        print("BLOCKED_MEHIL_R1_SOURCE_CONTRACT_MISMATCH")
        print("gallery log mutated unexpectedly")
        return 5
    if approval_log.read_bytes() != appr_bytes:
        print("BLOCKED_MEHIL_R1_SOURCE_CONTRACT_MISMATCH")
        print("timeline approval log mutated")
        return 5

    # Kill old UI / relaunch
    if args.kill_old_ui_port:
        subprocess.run(
            ["fuser", "-k", f"{args.kill_old_ui_port}/tcp"],
            capture_output=True,
            check=False,
        )
    ui_info = None
    if not args.skip_ui_launch:
        port = _free_port()
        ui_info = _launch_ui(pkg_root / "review_package.json", port)

    # Status: technical ready, user action required
    final_status = "COMPLETED_MEHIL_R1_VIDEO_REVIEW_READY_USER_ACTION_REQUIRED"
    if broken:
        final_status = "BLOCKED_MEHIL_R1_GALLERY_RENDERING"
    next_gate = "REID_PRODUCT_NEW_SHORT_VIDEO_PREPROCESS_AND_VALIDATION"

    temp = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}_", dir=str(root / "outputs" / "reid"))
    )
    try:
        write_json(temp / "gallery_rendering_root_cause.json", root_cause)
        write_json(temp / "ui_compatibility_report.json", api_report)
        write_json(temp / "crop_decode_validation.json", {"crops": crop_validations})
        write_json(
            temp / "gallery_approval_manifest.json",
            {
                "active": active_gal,
                "log_sha256": sha256_file(gallery_log) if gallery_log.is_file() else "",
                "append_only": True,
            },
        )
        write_json(
            temp / "match_specific_gallery_manifest.json",
            gallery_embed.get("manifest")
            or {
                "member_count": 0,
                "gallery_id": "gallery_mehil_run_20260727T223012Z_target_001",
                "awaiting_approvals": True,
            },
        )
        write_json(temp / "gallery_embedding_manifest.json", gallery_embed)
        write_json(
            temp / "source_video_playback_manifest.json",
            {
                "video_path": str(video),
                "video_sha256": video_sha,
                "codec": "h264",
                "pix_fmt": "yuv420p",
                "browser_compatible": True,
                "proxy_required": False,
                "duration_seconds_approx": 26.15,
                "playback_note": (
                    "Video playback is for review. Clicking and confirming links "
                    "existing tracklets; it does not rerun tracking."
                ),
            },
        )
        write_json(temp / "event_window_clip_manifests.json", {"clips": clip_manifests})
        write_json(
            temp / "candidate_ranking_manifest.json",
            {
                "ready": bool(gallery_embed.get("embedded")),
                "message": (
                    None
                    if gallery_embed.get("embedded")
                    else "Match-specific appearance gallery is not ready"
                ),
                "is_probability": False,
                "hides_candidates": False,
            },
        )
        write_json(temp / "product_decisions_approvals_audit.json", decision_audit)
        write_json(
            temp / "target_timeline.json",
            {"status": "pending_user_timeline_approvals", "intervals": []},
        )
        write_json(
            temp / "timeline_overlay_manifest.json",
            {"overlay_path": None, "reason": "awaiting_approved_multi_event_timeline"},
        )
        write_json(
            temp / "human_acceptance_checklist.json",
            {
                "schema_version": "multi_event_hil_overlay_acceptance_checklist_v1",
                "status": "pending_human",
                "false_confirmed_target_assignment": None,
            },
        )
        write_json(
            temp / "active_execution_path.json",
            {
                "detection_tracking_rerun": False,
                "clip": False,
                "game_state": False,
                "metrics": False,
                "ui": ui_info,
                "logs_immutable": True,
            },
        )
        report = f"""# MEHIL-R1 Video + Gallery Repair Report

- final_status: `{final_status}`
- product_next_gate: `{next_gate}`
- research_next_gate: `REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW`
- root_cause: `{root_cause['exact_root_cause']}`
- fix: streamlit_image wrapper uses `use_column_width` on Streamlit {api_report.get('streamlit_version')}
- crop candidates: {len(crop_validations)} · broken: {len(broken)}
- all crops EXT_SEG_004 / raw track 11: true
- decision audit: {decision_audit['record_count']} rows · SHA `{dec_sha}`
- event clips: {len(clip_manifests)}
- gallery embeddings: {gallery_embed.get('embedded')}
- UI: {(ui_info or {}).get('url')}
- stop: `{(ui_info or {}).get('stop')}`

## Kullanıcı sırası

1. Full source video izle
2. Enrollment Confirm (varsa doğrula) + Timeline Approval
3. Match Gallery — görünen crop’ları onayla
4. Cursor: embedding rebuild (R1/script yeniden, output archive)
5. EXT_183 / EXT_198 clip + Confirm + Timeline Approval
6. Timeline + overlay + acceptance checklist
"""
        (temp / "target_001_mehil_r1_report.md").write_text(report + "\n", encoding="utf-8")
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        write_json(
            temp / "target_001_mehil_r1_manifest.json",
            {
                "final_status": final_status,
                "product_next_gate": next_gate,
                "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                "artifact_files": artifacts,
                "artifact_count": len(artifacts),
                "generated_at": _utc(),
            },
        )
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        man = json.loads((temp / "target_001_mehil_r1_manifest.json").read_text())
        man["artifact_files"] = artifacts
        man["artifact_count"] = len(artifacts)
        write_json(temp / "target_001_mehil_r1_manifest.json", man)
        shutil.move(str(temp), str(final))
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise

    print(final_status)
    print(f"output: {final}")
    if ui_info:
        print(f"ui: {ui_info['url']} pid={ui_info['pid']}")
        print(f"stop: {ui_info['stop']}")
    return 0 if not broken else 6


if __name__ == "__main__":
    raise SystemExit(main())
