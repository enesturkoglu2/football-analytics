#!/usr/bin/env python3
"""Product short-video full detection/tracking preprocess + interactive UI readiness.

No Game State / CLIP / heatmap / metrics. Does not mutate the Windows source file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.detection.pipeline import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
)
from football_analytics.ingest.checksum import sha256_file  # noqa: E402
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)
from football_analytics.reid.multi_event_hil.overlay import (  # noqa: E402
    render_timeline_overlay_video,
)
from football_analytics.reid.short_video.dense_timeline import (  # noqa: E402
    build_dense_bbox_timeline,
)
from football_analytics.reid.short_video.detect_track import (  # noqa: E402
    DEFAULT_TRACKER_REL,
    DEFAULT_YOLO_REL,
    ShortVideoDetectTrackError,
    assert_frozen_checkpoints,
    build_mapping_rows,
    detections_by_frame,
    effective_detection_config_sha,
    load_tracker_args,
    observations_fingerprint,
    replay_bytetrack,
    run_yolo_detection,
)
from football_analytics.reid.short_video.identity import (  # noqa: E402
    build_identity,
    new_analysis_run_id,
)
from football_analytics.reid.short_video.package_build import (  # noqa: E402
    build_short_video_review_package,
)
from football_analytics.reid.short_video.path_resolve import (  # noqa: E402
    ShortVideoInputError,
    copy_to_product_input,
    resolve_short_video_path,
)
from football_analytics.reid.short_video.source_audit import audit_short_video  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        required=True,
        help="Windows or WSL path to the new short football video",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--analysis-run-id", default=None)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--skip-detection", action="store_true", help="Reuse existing preprocess artifacts")
    args = parser.parse_args()
    root = args.project_root.resolve()

    windows_stated = args.video
    try:
        source = resolve_short_video_path(args.video)
    except ShortVideoInputError as exc:
        print(str(exc))
        return 2

    # Forbidden legacy clips
    forbidden_names = {
        "sample.mp4",
        "target_001_independent_holdout_v2.mp4",
        "target_001_external_enrollment_v1.mp4",
    }
    if source.name in forbidden_names:
        print("BLOCKED_SHORT_VIDEO_SOURCE_CONTRACT_MISMATCH")
        print(f"refusing legacy/development clip: {source}")
        return 3

    audit = audit_short_video(
        source,
        windows_path_stated=windows_stated,
        windows_path_resolved=str(source),
    )
    if not audit["decode_integrity_ok"]:
        print("BLOCKED_SHORT_VIDEO_DETECTION")
        print("decode integrity failed")
        return 4

    video_sha = audit["sha256"]
    run_id = args.analysis_run_id or new_analysis_run_id()
    identity = build_identity(video_sha256=video_sha, analysis_run_id=run_id)

    out_root = (
        root
        / "outputs"
        / "reid"
        / "product_new_short_video_preprocess_validation"
        / identity.analysis_run_id
    )
    if out_root.exists() and not args.skip_detection:
        print(f"BLOCKED_SHORT_VIDEO_SOURCE_CONTRACT_MISMATCH: output exists {out_root}")
        return 5
    out_root.mkdir(parents=True, exist_ok=True)

    product_input_dir = (
        root
        / "data"
        / "product_inputs"
        / f"short_video_{video_sha[:12]}"
    )
    video_local = copy_to_product_input(source, product_input_dir)
    if sha256_file(video_local) != video_sha:
        print("BLOCKED_SHORT_VIDEO_SOURCE_CONTRACT_MISMATCH")
        print("copied product input SHA mismatch")
        return 6

    _write_json(out_root / "source_audit.json", audit)
    _write_json(out_root / "identity.json", identity.to_dict())

    try:
        ckpt = assert_frozen_checkpoints(root)
    except ShortVideoDetectTrackError as exc:
        print("BLOCKED_SHORT_VIDEO_DETECTION")
        print(exc)
        return 7

    width = int(audit["resolution"]["width"])
    height = int(audit["resolution"]["height"])
    fps = float(audit["fps"])
    frame_count = int(audit["frame_count_decoded"])
    start, end = 0, frame_count - 1

    det_dir = out_root / "detection"
    track_dir = out_root / "tracking"
    inv_dir = out_root / "inventory"

    t0 = time.time()
    if args.skip_detection and (track_dir / "tracks.jsonl").is_file():
        det_rows = [
            json.loads(line)
            for line in (det_dir / "detections.jsonl").read_text().splitlines()
            if line.strip()
        ]
        det_summary = json.loads((det_dir / "detection_summary.json").read_text())
        obs_a = [
            json.loads(line)
            for line in (track_dir / "tracks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        mapping_rows = [
            json.loads(line)
            for line in (inv_dir / "track_candidate_mapping.jsonl").read_text().splitlines()
            if line.strip()
        ]
        detect_runtime = float(det_summary.get("runtime_sec") or 0.0)
    else:
        model_path = root / DEFAULT_YOLO_REL
        tracker_path = root / DEFAULT_TRACKER_REL
        eff_sha = effective_detection_config_sha(
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            imgsz=DEFAULT_IMGSZ,
            model_sha=ckpt["yolo_sha256"],
        )
        print(f"Running YOLO detection frames {start}-{end} …")
        try:
            det_rows, det_summary = run_yolo_detection(
                video_path=video_local,
                model_path=model_path,
                start=start,
                end=end,
                video_sha=video_sha,
                model_sha=ckpt["yolo_sha256"],
                effective_config_sha=eff_sha,
            )
        except Exception as exc:  # noqa: BLE001
            print("BLOCKED_SHORT_VIDEO_DETECTION")
            print(exc)
            return 8
        detect_runtime = time.time() - t0
        det_summary["runtime_sec"] = detect_runtime
        det_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(det_dir / "detections.jsonl", det_rows)
        _write_json(det_dir / "detection_summary.json", det_summary)
        _write_json(
            det_dir / "detection_effective_config.json",
            {
                "conf": DEFAULT_CONF,
                "iou": DEFAULT_IOU,
                "imgsz": DEFAULT_IMGSZ,
                "classes": [0],
                "device": "cpu",
                "model_path": DEFAULT_YOLO_REL,
                "model_sha256": ckpt["yolo_sha256"],
                "effective_config_sha256": eff_sha,
            },
        )

        by_frame = detections_by_frame(det_rows)
        tracker_args = load_tracker_args(tracker_path)
        track_cfg_sha = ckpt["tracker_sha256"]
        print("Replaying ByteTrack (x2 determinism) …")
        try:
            replay_kwargs = dict(
                by_frame=by_frame,
                start=start,
                end=end,
                width=width,
                height=height,
                fps=fps,
                tracker_args=tracker_args,
                video_sha=video_sha,
                tracking_config_sha=track_cfg_sha,
            )
            obs_a = replay_bytetrack(**replay_kwargs)
            obs_b = replay_bytetrack(**replay_kwargs)
            fp_a = observations_fingerprint(obs_a)
            fp_b = observations_fingerprint(obs_b)
            if fp_a != fp_b or len(obs_a) != len(obs_b):
                print("BLOCKED_SHORT_VIDEO_TRACKING")
                print("ByteTrack replay non-deterministic")
                return 9
        except Exception as exc:  # noqa: BLE001
            print("BLOCKED_SHORT_VIDEO_TRACKING")
            print(exc)
            return 10

        track_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(track_dir / "tracks.jsonl", obs_a)
        _write_json(
            track_dir / "tracking_summary.json",
            {
                "schema_version": "short_video_tracking_summary_v1",
                "total_observations": len(obs_a),
                "unique_raw_tracks": len({o["raw_track_id"] for o in obs_a}),
                "frame_range": [start, end],
                "two_replay_determinism": True,
                "replay_fingerprint_sha256": fp_a,
                "tracker_path": DEFAULT_TRACKER_REL,
                "tracker_sha256": track_cfg_sha,
                "source_video_sha256": video_sha,
                "yolo_inference_in_tracking": False,
                "bytetrack_from_saved_detections": False,
                "runtime_sec": time.time() - t0 - detect_runtime,
            },
        )
        mapping_rows = build_mapping_rows(
            obs_a, width=width, height=height, video_sha=video_sha
        )
        inv_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(inv_dir / "track_candidate_mapping.jsonl", mapping_rows)
        _write_json(
            inv_dir / "observation_coverage_summary.json",
            {
                "track_count": len(mapping_rows),
                "eligible_count": sum(1 for r in mapping_rows if r.get("review_eligible")),
                "tracks": [
                    {
                        "raw_track_id": r["raw_track_id"],
                        "segment_id": r["segment_id"],
                        "observation_count": r["observation_count"],
                        "observation_coverage": r["observation_coverage"],
                        "first_frame": r["first_frame"],
                        "last_frame": r["last_frame"],
                    }
                    for r in mapping_rows
                ],
            },
        )

    try:
        dense = build_dense_bbox_timeline(
            mapping_rows,
            video_id=identity.video_id,
            video_sha256=video_sha,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            include_ineligible=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("BLOCKED_SHORT_VIDEO_DENSE_BBOX_MANIFEST")
        print(exc)
        return 11

    dens_path = out_root / "dense_bbox_timeline.json"
    dens_path.write_text(json.dumps(dense, ensure_ascii=False), encoding="utf-8")

    pkg_root = out_root / "product_review_package"
    try:
        pkg_info = build_short_video_review_package(
            pkg_root,
            project_root=root,
            identity=identity,
            video_abs=video_local,
            mapping_rows=mapping_rows,
            dense_timeline=dense,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            preprocess_root=out_root,
        )
    except Exception as exc:  # noqa: BLE001
        print("BLOCKED_SHORT_VIDEO_DENSE_BBOX_MANIFEST")
        print("package build failed:", exc)
        return 12

    # Interactive proxy (no source overwrite)
    try:
        proxy = ensure_interactive_review_proxy(
            source_video=video_local,
            source_video_sha256=video_sha,
            output_path=pkg_root / "session" / "interactive_review" / "source_proxy_960.mp4",
        )
    except Exception as exc:  # noqa: BLE001
        print("BLOCKED_SHORT_VIDEO_INTERACTIVE_SYNC")
        print(exc)
        return 13

    # Empty overlay scaffold (user acceptance after HIL decisions)
    overlay_dir = out_root / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    empty_timeline = {
        "intervals": [],
        "unresolved_intervals": [
            {
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "status": "unresolved",
                "metadata": {"preprocess_placeholder": True},
            }
        ],
    }
    overlay_path = overlay_dir / "diagnostic_overlay_preprocess_placeholder.mp4"
    try:
        overlay_meta = render_timeline_overlay_video(
            video_path=video_local,
            timeline=empty_timeline,
            output_path=overlay_path,
            observation_lookup=None,
            max_frames=min(frame_count, 90),
        )
    except Exception as exc:  # noqa: BLE001
        overlay_meta = {"error": str(exc), "note": "overlay prep deferred to post-approval"}
        overlay_path = None

    checklist = {
        "schema_version": "short_video_human_acceptance_checklist_v1",
        "items": [
            "all player bboxes visible on interactive video",
            "initial target click selects raw track",
            "highlight persists on same raw track",
            "track end shows TARGET TRACK ENDED / LOST / UNRESOLVED",
            "no automatic raw-track switch",
            "re-entry tracks selectable",
            "provisional timeline updates on CONFIRM",
            "approved timeline only after Timeline Approval",
            "overlay acceptance after multi-track linking",
        ],
        "status": "AWAITING_USER",
    }
    _write_json(out_root / "human_acceptance_checklist.json", checklist)

    coverage = {
        "processed_frame_count": det_summary.get("frames_processed"),
        "player_detection_count": det_summary.get("player_detection_count"),
        "dropped_frame_count": det_summary.get("dropped_frame_count"),
        "decode_failures": det_summary.get("decode_failures"),
        "detection_runtime_sec": detect_runtime,
        "unique_raw_tracks": len(mapping_rows),
        "eligible_tracks": sum(1 for r in mapping_rows if r.get("review_eligible")),
        "dense_frame_keys": dense.get("frame_keys_with_observations"),
        "total_track_observations": sum(int(r["observation_count"]) for r in mapping_rows),
        "proxy_sha256": proxy.get("proxy_sha256"),
        "legacy_ext_fallback": False,
        "game_state_executed": False,
        "clip_executed": False,
    }
    _write_json(out_root / "coverage_summary.json", coverage)

    status = "COMPLETED_SHORT_VIDEO_PREPROCESS_UI_READY_USER_ACTION_REQUIRED"
    report = {
        "final_status": status,
        "created_at": _utc(),
        "identity": identity.to_dict(),
        "source_audit": {
            "absolute_path": audit["absolute_path"],
            "product_input_path": str(video_local),
            "sha256": video_sha,
            "duration_sec": audit["duration_sec"],
            "fps": fps,
            "frame_count": frame_count,
            "resolution": audit["resolution"],
            "codec": audit["codec"],
            "decode_integrity_ok": audit["decode_integrity_ok"],
            "camera_cuts_estimate": audit.get("camera_cuts_estimate"),
        },
        "coverage": coverage,
        "package": {
            "path": str(pkg_root / "review_package.json"),
            "event_count": pkg_info["event_count"],
            "recovery_event_count": pkg_info["recovery_event_count"],
            "enrollment_candidates": pkg_info["candidate_count_enrollment"],
        },
        "overlay": overlay_meta if isinstance(overlay_meta, dict) else {},
        "product_next_gate": "USER_INTERACTIVE_TARGET_SELECTION_AND_TIMELINE_APPROVAL",
        "forbidden_next": ["HIL-D", "CLIP", "Game State", "heatmap", "sprint metrics"],
    }
    _write_json(out_root / "final_manifest.json", report)
    report_md = out_root / "final_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Short Video Preprocess — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- analysis_run_id: `{identity.analysis_run_id}`",
                f"- match_id: `{identity.match_id}`",
                f"- video_sha256: `{video_sha}`",
                f"- duration_sec: {audit['duration_sec']}",
                f"- frames: {frame_count} @ {fps} fps",
                f"- detections: {coverage['player_detection_count']}",
                f"- raw_tracks: {coverage['unique_raw_tracks']} (eligible {coverage['eligible_tracks']})",
                f"- dense frames with obs: {coverage['dense_frame_keys']}",
                f"- package: `{pkg_root / 'review_package.json'}`",
                "",
                "## Notlar",
                "",
                "- Eski enrollment/holdout/sample klipleri kullanılmadı.",
                "- Game State / CLIP / saha metriği çalıştırılmadı.",
                "- Kullanıcı Interactive Video üzerinden hedef seçmeli.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ui_url = None
    if not args.skip_ui_launch:
        port = _free_port()
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(root / "src"),
            "HIL_REVIEW_PACKAGE": str(pkg_root / "review_package.json"),
            "STREAMLIT_CONFIG_DIR": str(root / "configs/reid/hil_ui/.streamlit"),
        }
        py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
        log_path = out_root / "ui_launch.log"
        cmd = [
            str(py),
            str(root / "scripts/run_hil_offline_review_ui.py"),
            "--review-package",
            str(pkg_root / "review_package.json"),
            "--port",
            str(port),
        ]
        with log_path.open("w", encoding="utf-8") as logf:
            subprocess.Popen(cmd, cwd=str(root), env=env, stdout=logf, stderr=subprocess.STDOUT)
        ui_url = f"http://127.0.0.1:{port}"
        _write_json(out_root / "ui_session.json", {"url": ui_url, "port": port, "log": str(log_path)})

    print(status)
    print(f"output_root={out_root}")
    print(f"review_package={pkg_root / 'review_package.json'}")
    if ui_url:
        print(f"local_url={ui_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
