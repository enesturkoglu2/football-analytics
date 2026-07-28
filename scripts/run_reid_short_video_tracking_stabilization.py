#!/usr/bin/env python3
"""Short-video tracking stabilization: audit, comparative trackers, versioned manifests.

Does not mutate existing detection manifests or prior decision/approval logs.
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

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.ingest.checksum import sha256_file  # noqa: E402
from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: E402
    ensure_interactive_review_proxy,
)
from football_analytics.reid.short_video.dense_timeline import (  # noqa: E402
    build_dense_bbox_timeline,
    observations_for_component,
    write_dense_timeline,
)
from football_analytics.reid.short_video.detect_track import (  # noqa: E402
    detections_by_frame,
)
from football_analytics.reid.short_video.tracker_compare import (  # noqa: E402
    bounded_variants,
    config_sha,
    observations_fingerprint,
    replay_tracker,
    select_best,
    summarize_variant,
    write_yaml,
)
from football_analytics.reid.short_video.tracking_audit import (  # noqa: E402
    audit_tracking,
    continuity_probe,
)


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


def _render_comparison_overlay(
    *,
    video_path: Path,
    observations: list[dict],
    output_path: Path,
    label: str,
    max_frames: int | None = None,
) -> dict:
    import cv2
    from collections import defaultdict

    by_f = defaultdict(list)
    for r in observations:
        by_f[int(r["frame_index"])].append(r)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n = min(n, max_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    births = {int(min(v, key=lambda r: r["frame_index"])["frame_index"]) for v in
              [ [x for x in observations if int(x["raw_track_id"])==tid] for tid in {int(o["raw_track_id"]) for o in observations} ]
              if v}
    # simpler birth/death sets
    track_span = {}
    for r in observations:
        tid = int(r["raw_track_id"])
        fi = int(r["frame_index"])
        if tid not in track_span:
            track_span[tid] = [fi, fi]
        else:
            track_span[tid][0] = min(track_span[tid][0], fi)
            track_span[tid][1] = max(track_span[tid][1], fi)
    birth_f = {s[0] for s in track_span.values()}
    death_f = {s[1] for s in track_span.values()}
    written = 0
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        for r in by_f.get(fi, []):
            x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(
                frame,
                f"id={r['raw_track_id']}",
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        tag = label
        if fi in birth_f:
            tag += " | BIRTH"
        if fi in death_f:
            tag += " | DEATH"
        cv2.putText(frame, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return {"path": str(output_path), "sha256": sha256_file(output_path), "frames": written}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--max-overlay-frames", type=int, default=450)
    args = parser.parse_args()
    root = args.project_root.resolve()

    run = (
        root
        / "outputs/reid/product_new_short_video_preprocess_validation/sv_run_20260727T234854Z"
    )
    if not run.is_dir():
        print("BLOCKED_TRACKING_AUDIT")
        print("missing analysis run", run)
        return 2

    out = run / "tracking_stabilization"
    out.mkdir(parents=True, exist_ok=True)

    video = root / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    video_sha = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"
    if sha256_file(video) != video_sha:
        print("BLOCKED_TRACKING_AUDIT")
        print("video sha mismatch")
        return 3

    # Freeze decision log bytes (must not mutate)
    dec = run / "product_review_package/session/decision_log.jsonl"
    appr = run / "product_review_package/session/timeline_approval_log.jsonl"
    dec_b, appr_b = dec.read_bytes(), appr.read_bytes()

    audit = audit_tracking(
        tracks_jsonl=run / "tracking/tracks.jsonl",
        detections_jsonl=run / "detection/detections.jsonl",
        mapping_jsonl=run / "inventory/track_candidate_mapping.jsonl",
        fps=30.0,
        width=1326,
        height=750,
    )
    _write_json(out / "tracking_fragmentation_audit.json", audit)

    # Target-specific: no user decisions yet
    target_diag = {
        "schema_version": "short_video_target_diagnostic_v1",
        "active_user_selection": False,
        "decision_log_empty": dec.stat().st_size == 0,
        "note": "NO_ACTIVE_TARGET_SELECTION — kullanıcıdan Interactive Video'da başlangıç bbox seçimi bekleniyor",
        "probe_not_ground_truth": True,
    }
    # Continuity probe using longest current track seed (diagnostic only)
    longest = (audit.get("longest_tracks") or [None])[0]
    seed_bbox = None
    seed_frame = None
    if longest:
        # load first bbox of that track
        for line in (run / "tracking/tracks.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["raw_track_id"]) == int(longest["raw_track_id"]) and int(
                row["frame_index"]
            ) == int(longest["first_frame"]):
                seed_bbox = row["bbox_xyxy"]
                seed_frame = int(row["frame_index"])
                break
        if seed_bbox is not None:
            obs_cur = [
                json.loads(l)
                for l in (run / "tracking/tracks.jsonl").read_text().splitlines()
                if l.strip()
            ]
            target_diag["continuity_probe_on_longest_track"] = continuity_probe(
                obs_cur, seed_bbox=seed_bbox, seed_frame=seed_frame, fps=30.0
            )
            target_diag["probe_raw_track_id"] = longest["raw_track_id"]
    _write_json(out / "target_specific_diagnostic.json", target_diag)

    # Load detections once
    det_rows = [
        json.loads(l)
        for l in (run / "detection/detections.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by_frame = detections_by_frame(det_rows)
    start, end = 0, 1356
    width, height, fps = 1326, 750, 30.0

    variants = bounded_variants()
    results_summary = []
    overlays = []
    selected_payload = None

    for variant in variants:
        vid = variant["variant_id"]
        print(f"Running variant {vid} …")
        params = variant["params"]
        cfg_sha = config_sha(params)
        write_yaml(out / "configs" / f"{vid}.yaml", params)
        t0 = time.time()
        try:
            if variant["tracker_kind"] == "botsort":
                # availability already known in env
                pass
            obs_a = replay_tracker(
                tracker_kind=variant["tracker_kind"],
                params=params,
                by_frame=by_frame,
                start=start,
                end=end,
                width=width,
                height=height,
                fps=fps,
                video_path=video if variant["tracker_kind"] == "botsort" else None,
                video_sha=video_sha,
                tracking_config_sha=cfg_sha,
            )
            obs_b = replay_tracker(
                tracker_kind=variant["tracker_kind"],
                params=params,
                by_frame=by_frame,
                start=start,
                end=end,
                width=width,
                height=height,
                fps=fps,
                video_path=video if variant["tracker_kind"] == "botsort" else None,
                video_sha=video_sha,
                tracking_config_sha=cfg_sha,
            )
        except Exception as exc:  # noqa: BLE001
            if variant["tracker_kind"] == "botsort":
                results_summary.append(
                    {
                        "variant_id": vid,
                        "status": "TRACKER_CANDIDATE_NOT_AVAILABLE",
                        "error": str(exc),
                        "determinism_ok": False,
                        "raw_track_count": None,
                        "continuity_probe": None,
                        "runtime_sec": time.time() - t0,
                    }
                )
                continue
            print("BLOCKED_TRACKER_CANDIDATE")
            print(exc)
            return 4
        runtime = time.time() - t0
        fp_a = observations_fingerprint(obs_a)
        fp_b = observations_fingerprint(obs_b)
        summary = summarize_variant(
            obs_a,
            fps=fps,
            width=width,
            height=height,
            video_sha=video_sha,
            seed_bbox=seed_bbox,
            seed_frame=seed_frame,
            runtime_sec=runtime,
            fingerprint=fp_a,
            fingerprint_b=fp_b,
        )
        summary["variant_id"] = vid
        summary["tracker_kind"] = variant["tracker_kind"]
        summary["params"] = params
        summary["rationale"] = variant["rationale"]
        summary["false_target_identity_switch"] = 0  # no auto-merge policy; probe only
        summary["status"] = "ok"
        # persist versioned tracking
        vdir = out / "variants" / vid
        vdir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(vdir / "tracks.jsonl", obs_a)
        _write_jsonl(vdir / "track_candidate_mapping.jsonl", summary["mapping_rows"])
        slim = {k: v for k, v in summary.items() if k not in {"observations", "mapping_rows"}}
        _write_json(vdir / "summary.json", slim)
        results_summary.append(slim)
        selected_payload = selected_payload  # placeholder
        # keep heavy fields only on disk
        variant["_obs"] = obs_a
        variant["_mapping"] = summary["mapping_rows"]
        variant["_summary"] = slim

    selection = select_best([r for r in results_summary if r.get("status") == "ok"])
    product_id = selection["product_candidate_variant_id"]
    _write_json(out / "variant_comparison.json", {"results": results_summary, "selection": selection})

    # Overlays for up to 3 variants
    overlay_ids = []
    for vid in ["A_current_bytetrack", "B2_buffer90_match065_new04", "C_botsort_gmc_sparseOptFlow"]:
        if any(r.get("variant_id") == vid and r.get("status") == "ok" for r in results_summary):
            overlay_ids.append(vid)
    overlay_ids = overlay_ids[:3]
    for vid in overlay_ids:
        meta = next(v for v in variants if v["variant_id"] == vid)
        ov = _render_comparison_overlay(
            video_path=video,
            observations=meta["_obs"],
            output_path=out / "overlays" / f"{vid}.mp4",
            label=vid,
            max_frames=args.max_overlay_frames,
        )
        overlays.append({"variant_id": vid, **ov})
    _write_json(out / "comparison_overlays.json", {"overlays": overlays})

    # Install product candidate dense timeline into package session (versioned; do not overwrite old tracking/)
    winner = next(v for v in variants if v["variant_id"] == product_id)
    win_map = winner["_mapping"]
    dens = build_dense_bbox_timeline(
        win_map,
        video_id="video_short_f2f6d8a077ca",
        video_sha256=video_sha,
        fps=fps,
        frame_count=1357,
        width=width,
        height=height,
        include_ineligible=True,
    )
    dens["tracking_variant_id"] = product_id
    dens["stabilization_note"] = "versioned candidate; original tracking/ untouched"
    write_dense_timeline(out / "dense_bbox_timeline_stabilized.json", dens)
    # also version under variants
    write_dense_timeline(out / "variants" / product_id / "dense_bbox_timeline.json", dens)

    pkg = run / "product_review_package"
    from football_analytics.reid.hil.candidates import validate_candidate_manifest
    from football_analytics.reid.hil.common import sha256_file as _sha
    from football_analytics.reid.hil.events import EventType, validate_recovery_event
    from football_analytics.reid.short_video.dense_timeline import (
        attach_candidate_ids_to_timeline,
    )
    from football_analytics.reid.short_video.package_build import (
        _candidates_from_rows,
        _tracks_active_in_window,
    )

    enroll_event_id = "evt_sv_initial_enrollment_001"
    enroll_codes = [
        str(r["external_candidate_code"]) for r in win_map if r.get("review_eligible")
    ]
    enroll_cands = _candidates_from_rows(
        win_map,
        event_id=enroll_event_id,
        target_id="target_001",
        codes=enroll_codes,
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
    dens2 = attach_candidate_ids_to_timeline(dens, enroll_cands)
    write_dense_timeline(pkg / "dense_bbox_timeline_stabilized.json", dens2)
    ir = pkg / "session" / "interactive_review"
    ir.mkdir(parents=True, exist_ok=True)
    comp = observations_for_component(dens2)
    (ir / "dense_observations_stabilized.json").write_text(
        json.dumps(comp, ensure_ascii=False), encoding="utf-8"
    )
    active = ir / "dense_observations.json"
    if active.is_file() and not (ir / "dense_observations_baseline_backup.json").exists():
        shutil.copy2(active, ir / "dense_observations_baseline_backup.json")
    active.write_text(json.dumps(comp, ensure_ascii=False), encoding="utf-8")
    main_dense = pkg / "dense_bbox_timeline.json"
    if main_dense.is_file() and not (pkg / "dense_bbox_timeline_baseline_backup.json").exists():
        shutil.copy2(main_dense, pkg / "dense_bbox_timeline_baseline_backup.json")
    write_dense_timeline(main_dense, dens2)

    enroll = pkg / "enrollment_candidate_manifest.json"
    enroll_name = "enrollment_candidate_manifest.json"
    if enroll.is_file() and not (pkg / "enrollment_candidate_manifest_baseline_backup.json").exists():
        shutil.copy2(enroll, pkg / "enrollment_candidate_manifest_baseline_backup.json")
    _write_json(enroll, enroll_manifest)
    enroll_sha = _sha(enroll)

    events = [
        validate_recovery_event(
            {
                "schema_version": "target_recovery_event_v1",
                "event_id": enroll_event_id,
                "project_id": "football-analytics",
                "run_id": "sv_run_20260727T234854Z",
                "target_id": "target_001",
                "event_type": EventType.INITIAL_TARGET_ENROLLMENT.value,
                "video_id": "video_short_f2f6d8a077ca",
                "video_path": str(video.resolve()),
                "video_sha256": video_sha,
                "created_at": _utc(),
                "status": "open",
                "trigger_source": "short_video_tracking_stabilization",
                "trigger_reason": "stabilized tracking re-enrollment (all eligible tracks)",
                "last_confirmed_segment_id": None,
                "last_confirmed_frame_index": None,
                "review_window_start_frame": 0,
                "review_window_end_frame": min(90, 1356),
                "candidate_manifest_path": enroll_name,
                "candidate_manifest_sha256": enroll_sha,
                "candidate_count": len(enroll_cands),
                "requires_calibration": False,
                "evidence_paths": [],
                "evidence_sha256": [],
                "provenance": {
                    "package_mode": "short_video_product",
                    "ui_profile": "short_video",
                    "tracking_variant_id": product_id,
                    "one_click_workflow": True,
                    "gt_prefill": False,
                    "automatic_identity": False,
                },
                "metadata": {"priority": "required_enrollment"},
            }
        )
    ]
    cand_paths = [enroll_name]
    cand_shas = {enroll_name: enroll_sha}
    # keep a few recovery windows from longest mid-ends on stabilized mapping
    mid_ends = [
        r
        for r in win_map
        if r.get("review_eligible", True)
        and int(r["last_frame"]) < 1356 - 30
        and int(r["observation_count"]) >= 30
    ]
    mid_ends.sort(key=lambda r: int(r["observation_count"]), reverse=True)
    for row in mid_ends[:8]:
        code = str(row["external_candidate_code"])
        lost = int(row["last_frame"])
        win_start, win_end = lost + 1, min(1356, lost + 60)
        codes = _tracks_active_in_window(win_map, start=win_start, end=win_end)
        if not codes:
            continue
        event_id = f"evt_sv_reentry_{code}"
        cands = _candidates_from_rows(
            win_map, event_id=event_id, target_id="target_001", codes=codes
        )
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
        man_name = f"reentry_{code}_candidate_manifest.json"
        _write_json(pkg / man_name, man)
        man_sha = _sha(pkg / man_name)
        cand_paths.append(man_name)
        cand_shas[man_name] = man_sha
        from football_analytics.reid.hil_c2.product_package import _segment_id

        events.append(
            validate_recovery_event(
                {
                    "schema_version": "target_recovery_event_v1",
                    "event_id": event_id,
                    "project_id": "football-analytics",
                    "run_id": "sv_run_20260727T234854Z",
                    "target_id": "target_001",
                    "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
                    "video_id": "video_short_f2f6d8a077ca",
                    "video_path": str(video.resolve()),
                    "video_sha256": video_sha,
                    "created_at": _utc(),
                    "status": "open",
                    "trigger_source": "short_video_track_termination_stabilized",
                    "trigger_reason": f"stabilized track {row['raw_external_track_id']} ended @ {lost}",
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
                        "package_mode": "short_video_product",
                        "top_k_only": False,
                        "automatic_identity": False,
                        "tracking_variant_id": product_id,
                    },
                    "metadata": {"priority": "recovery"},
                }
            )
        )
    events_path = pkg / "recovery_events.jsonl"
    # backup baseline events once
    if events_path.is_file() and not (pkg / "recovery_events_baseline_backup.jsonl").exists():
        shutil.copy2(events_path, pkg / "recovery_events_baseline_backup.jsonl")
    _write_jsonl(events_path, events)
    events_sha = _sha(events_path)

    rp = json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))
    rp["event_manifest_sha256"] = events_sha
    rp["candidate_manifest_paths"] = cand_paths
    rp["candidate_manifest_sha256"] = cand_shas
    rp.setdefault("provenance", {})["tracking_variant_id"] = product_id
    rp["provenance"]["stabilization_root"] = str(out.relative_to(root))
    rp["provenance"]["one_click_workflow"] = True
    rp["provenance"]["ui_profile"] = "short_video"
    rp["provenance"]["stabilized_enrollment_manifest"] = enroll_name
    rp["provenance"]["stabilized_enrollment_sha256"] = enroll_sha
    (pkg / "review_package.json").write_text(
        json.dumps(rp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ensure_interactive_review_proxy(
        source_video=video,
        source_video_sha256=video_sha,
        output_path=ir / "source_proxy_960.mp4",
    )

    # immutability of user logs
    if dec.read_bytes() != dec_b or appr.read_bytes() != appr_b:
        print("FAILED_TRACKING_IDENTITY_SAFETY")
        print("decision/approval logs mutated")
        return 5

    baseline = next(r for r in results_summary if r["variant_id"] == "A_current_bytetrack")
    winner_sum = next(r for r in results_summary if r["variant_id"] == product_id)
    improved = bool(selection.get("improved_vs_baseline"))
    if improved:
        status = "COMPLETED_TRACKING_STABILIZATION_UI_READY_USER_ACTION_REQUIRED"
    else:
        status = "COMPLETED_UI_WORKFLOW_ONLY_TRACKING_STILL_FRAGMENTED"

    report = {
        "final_status": status,
        "created_at": _utc(),
        "analysis_run_id": "sv_run_20260727T234854Z",
        "match_id": "match_short_video_f2f6d8a077ca",
        "audit_highlights": {
            "raw_tracks": audit["raw_track_count"],
            "eligible": audit["eligible_track_count"],
            "median_duration_sec": audit["duration_sec"]["median"],
            "lt_1s": audit["short_tracks"]["lt_1s"],
            "root_cause": audit["root_cause"]["assessment"],
        },
        "selection": selection,
        "baseline_summary": baseline,
        "product_candidate_summary": winner_sum,
        "overlays": overlays,
        "false_target_identity_switch": 0,
        "detection_rerun": False,
        "original_tracking_dir_mutated": False,
        "target_diagnostic": target_diag,
        "forbidden_next": ["HIL-D", "Game State", "CLIP", "2D", "heatmap"],
    }
    _write_json(out / "final_manifest.json", report)
    (out / "final_report.md").write_text(
        "\n".join(
            [
                "# Short Video Tracking Stabilization — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- product_candidate: `{product_id}`",
                f"- improved_vs_baseline: `{improved}`",
                f"- baseline raw tracks: {baseline.get('raw_track_count')}",
                f"- candidate raw tracks: {winner_sum.get('raw_track_id', winner_sum.get('raw_track_count'))}",
                f"- baseline median duration: {baseline.get('median_track_duration_sec')}",
                f"- candidate median duration: {winner_sum.get('median_track_duration_sec')}",
                f"- false_target_identity_switch: 0",
                "",
                "## Root cause",
                "",
                audit["root_cause"]["assessment"],
                "",
                "## Notlar",
                "",
                "- Detection yeniden çalıştırılmadı; `tracking/` orijinali korundu.",
                "- Decision/approval logları değiştirilmedi.",
                "- Kullanıcı one-click UI ile seçim/onay yapmalı.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ui_url = None
    if not args.skip_ui_launch:
        port = _free_port()
        log = out / "ui_launch.log"
        py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
        cmd = [
            str(py),
            str(root / "scripts/run_hil_offline_review_ui.py"),
            "--review-package",
            str(pkg / "review_package.json"),
            "--port",
            str(port),
            "--address",
            "127.0.0.1",
        ]
        with log.open("w", encoding="utf-8") as handle:
            # Prefer host-visible launch via setsid when possible
            subprocess.Popen(
                ["setsid", *cmd],
                cwd=str(root),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        ui_url = f"http://127.0.0.1:{port}"
        _write_json(out / "ui_session.json", {"url": ui_url, "port": port, "log": str(log)})

    print(status)
    print(f"output={out}")
    if ui_url:
        print(f"local_url={ui_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
