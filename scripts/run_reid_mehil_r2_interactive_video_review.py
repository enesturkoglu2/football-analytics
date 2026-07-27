#!/usr/bin/env python3
"""MEHIL-R2: interactive video component readiness + gallery quality gate (logs immutable)."""

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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.timeline.sources import write_json  # noqa: E402
from football_analytics.reid.hil_c2.product_package import _segment_id  # noqa: E402
from football_analytics.reid.hil_ui.dense_observations import (  # noqa: E402
    attach_candidate_ids,
    build_dense_observations_from_mapping,
    load_mapping_jsonl,
)
from football_analytics.reid.hil_ui.gallery_quality import audit_gallery_candidates  # noqa: E402
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)
from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalLog,
    resolve_active_gallery_approvals,
)
from football_analytics.reid.multi_event_hil.gallery_embed import (  # noqa: E402
    embed_approved_gallery,
)

OUTPUT = "target_001_multi_event_hil_interactive_video_r2"
PACKAGE = "target_001_multi_event_hil_review_package"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--kill-old-ui-ports", default="58131,59389")
    args = parser.parse_args()
    root = args.project_root.resolve()
    final = root / "outputs" / "reid" / OUTPUT
    if final.exists():
        print("BLOCKED_MEHIL_R2_OUTPUT_EXISTS")
        return 2

    pkg = root / "outputs" / "reid" / PACKAGE
    dec = pkg / "session" / "decision_log.jsonl"
    gal = pkg / "session" / "gallery_approval_log.jsonl"
    appr = pkg / "session" / "timeline_approval_log.jsonl"
    dec_b, gal_b, appr_b = dec.read_bytes(), gal.read_bytes(), appr.read_bytes()

    video = root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
    video_sha = sha256_file(video)
    proxy = ensure_interactive_review_proxy(
        source_video=video,
        source_video_sha256=video_sha,
        output_path=pkg / "session" / "interactive_review" / "source_proxy_960.mp4",
    )

    mapping = load_mapping_jsonl(
        root
        / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
        / "inventory/target_001_external_track_candidate_mapping.jsonl"
    )
    dens = build_dense_observations_from_mapping(
        mapping,
        codes=["EXT_004", "EXT_183", "EXT_198", "EXT_001", "EXT_002", "EXT_003"],
        segment_id_fn=_segment_id,
    )
    # attach ids from enrollment manifest if present
    enroll_man = pkg / "enrollment_candidate_manifest.json"
    if enroll_man.is_file():
        dens = attach_candidate_ids(
            dens, json.loads(enroll_man.read_text()).get("candidates") or []
        )
    dens_path = pkg / "session" / "interactive_review" / "dense_observations.json"
    dens_path.write_text(json.dumps(dens), encoding="utf-8")

    crop_man = json.loads(
        (pkg / "gallery_crop_candidates" / "enrollment_crop_candidates.json").read_text()
    )
    quality = audit_gallery_candidates(
        crop_man.get("candidates") or [], source_frame_size=(1332, 746)
    )

    active_gal = resolve_active_gallery_approvals(GalleryApprovalLog(gal).read_raw())
    embed_info = {"embedded": False, "member_count": 0}
    if active_gal and quality["usable_count"] > 0:
        # only embed if approved crops are also usable
        usable_ids = {
            a["crop_id"]
            for a in quality["audits"]
            if a.get("quality_class") == "USABLE_GALLERY_CANDIDATE"
        }
        if any(cid in usable_ids for cid in active_gal):
            try:
                emb = embed_approved_gallery(
                    crop_candidates=crop_man,
                    gallery_approvals=GalleryApprovalLog(gal).read_raw(),
                    output_dir=pkg / "match_specific_gallery",
                    match_id="match_external_enrollment_v1_mehil",
                    analysis_run_id="mehil_run_20260727T223012Z",
                    target_id="target_001",
                )
                embed_info = {k: emb[k] for k in emb if k != "embeddings"}
            except Exception as exc:  # noqa: BLE001
                print("BLOCKED_MEHIL_R2_GALLERY_QUALITY")
                print(exc)
                return 4
    elif quality["usable_count"] == 0:
        embed_info = {
            "embedded": False,
            "member_count": 0,
            "reason": "all_crops_failed_quality_gate",
            "helper_ranking": "unavailable",
        }

    component_dir = (
        root
        / "src/football_analytics/reid/hil_ui/interactive_video_component/frontend/index.html"
    )
    if not component_dir.is_file():
        print("BLOCKED_MEHIL_R2_VIDEO_COMPONENT")
        return 5

    # immutability
    if dec.read_bytes() != dec_b or gal.read_bytes() != gal_b or appr.read_bytes() != appr_b:
        print("BLOCKED_MEHIL_R2_LOGS_MUTATED")
        return 6

    for port in str(args.kill_old_ui_ports).split(","):
        port = port.strip()
        if port:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)

    ui = None
    if not args.skip_ui_launch:
        port = _free_port()
        py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
        cmd = [
            str(py),
            str(PROJECT / "scripts/run_hil_offline_review_ui.py"),
            "--review-package",
            str(pkg / "review_package.json"),
            "--port",
            str(port),
            "--address",
            "127.0.0.1",
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
        ui = {"pid": proc.pid, "url": f"http://127.0.0.1:{port}", "stop": f"kill {proc.pid}"}

    status = "COMPLETED_MEHIL_R2_INTERACTIVE_VIDEO_READY_USER_ACTION_REQUIRED"
    next_gate = "REID_PRODUCT_NEW_SHORT_VIDEO_PREPROCESS_AND_VALIDATION"

    temp = Path(tempfile.mkdtemp(prefix=f".{OUTPUT}_", dir=str(root / "outputs" / "reid")))
    try:
        write_json(
            temp / "interactive_video_component_report.json",
            {
                "component": "hil_interactive_video",
                "frontend": str(component_dir),
                "local_only": True,
                "public_hosting": False,
                "st_video_click_hack": False,
                "proxy": {k: proxy[k] for k in proxy if k != "data_url"},
                "dense_observation_frames": len(dens),
                "dense_observation_path": str(dens_path),
            },
        )
        write_json(temp / "gallery_quality_audit.json", quality)
        write_json(
            temp / "match_specific_gallery_manifest.json",
            embed_info.get("manifest")
            or {
                "member_count": 0,
                "embeddings_present": False,
                "development_gallery_used": False,
            },
        )
        write_json(temp / "gallery_embedding_manifest.json", embed_info)
        write_json(
            temp / "product_decision_approval_audit.json",
            {
                "decision_log_sha256": sha256_file(dec),
                "timeline_approval_log_sha256": sha256_file(appr),
                "gallery_approval_log_sha256": sha256_file(gal),
                "immutable": True,
            },
        )
        write_json(
            temp / "human_acceptance_checklist.json",
            {"status": "pending_human", "false_confirmed_target_assignment": None},
        )
        write_json(
            temp / "active_execution_path.json",
            {
                "detection_tracking_rerun": False,
                "clip": False,
                "game_state": False,
                "ui": ui,
            },
        )
        report = f"""# MEHIL-R2 Interactive Video + Gallery Quality

- final_status: `{status}`
- product_next_gate: `{next_gate}`
- research_next_gate: `REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW`
- interactive component: local `declare_component` + HTML5 video/canvas
- proxy_sha256: `{proxy['proxy_sha256']}`
- dense observation frames: {len(dens)}
- gallery usable_count: {quality['usable_count']} / {quality['candidate_count']}
- gallery embed: {embed_info.get('embedded')}
- UI: {(ui or {}).get('url')}
- stop: `{(ui or {}).get('stop')}`

## Kullanıcı

1. Interactive Video sekmesinde hareketli bbox’lara tıkla
2. Confirm + Timeline Approval
3. Recovery marker’lara git, video üzerinden seç
4. Match Gallery — yalnız USABLE crop onayla (bulanıklar disabled)
5. Overlay kabul
"""
        (temp / "target_001_mehil_r2_report.md").write_text(report + "\n", encoding="utf-8")
        arts = sorted(p.name for p in temp.iterdir() if p.is_file())
        write_json(
            temp / "target_001_mehil_r2_manifest.json",
            {
                "final_status": status,
                "product_next_gate": next_gate,
                "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                "artifact_files": arts,
                "artifact_count": len(arts),
                "generated_at": _utc(),
            },
        )
        arts = sorted(p.name for p in temp.iterdir() if p.is_file())
        man = json.loads((temp / "target_001_mehil_r2_manifest.json").read_text())
        man["artifact_files"] = arts
        man["artifact_count"] = len(arts)
        write_json(temp / "target_001_mehil_r2_manifest.json", man)
        shutil.move(str(temp), str(final))
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(status)
    print(f"output: {final}")
    if ui:
        print(f"ui: {ui['url']} pid={ui['pid']}")
        print(f"stop: {ui['stop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
