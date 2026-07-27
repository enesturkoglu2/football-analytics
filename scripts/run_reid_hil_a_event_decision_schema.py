#!/usr/bin/env python3
"""HIL-A validation/demo entrypoint (no UI, no inference, no network)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.candidates import validate_candidate_manifest  # noqa: E402
from football_analytics.reid.hil.common import sha256_file, sha256_json_canonical  # noqa: E402
from football_analytics.reid.hil.decisions import build_decision, validate_decision  # noqa: E402
from football_analytics.reid.hil.events import (  # noqa: E402
    EventType,
    validate_event_list_unique_ids,
    validate_recovery_event,
)
from football_analytics.reid.hil.log import DecisionLog, compute_log_sha256  # noqa: E402
from football_analytics.reid.hil.resolve import derive_effective_state_summary  # noqa: E402

VIDEO_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
CKPT = "c61e0da2007f7c7f4d889cb68774dfeecf8c4c433e0bfe3858b48b8655f83e91"
MODEL = "osnet_x1_0_sportsreid_soccernet"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_demo_fixtures(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    lost = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": "evt_demo_target_lost_001",
            "project_id": "football-analytics",
            "run_id": "hil_a_demo",
            "target_id": "target_001",
            "event_type": EventType.TARGET_LOST.value,
            "video_id": "demo_video",
            "video_path": "data/test_clips/target_001_independent_holdout_v2.mp4",
            "video_sha256": VIDEO_SHA,
            "created_at": "2026-07-27T00:00:00Z",
            "status": "open",
            "trigger_source": "fixture",
            "trigger_reason": "demo TARGET_LOST",
            "last_confirmed_segment_id": "H2_SEG_DEMO_000",
            "last_confirmed_frame_index": 100,
            "review_window_start_frame": 101,
            "review_window_end_frame": 250,
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

    candidates = []
    for i in range(1, 5):
        candidates.append(
            {
                "candidate_id": f"cand_{i:03d}",
                "segment_id": f"H2_SEG_DEMO_{i:03d}",
                "raw_track_id": f"raw_track_demo_{i:03d}",
                "start_frame": 110 + i,
                "middle_frame": 120 + i,
                "end_frame": 140 + i,
                "bbox_references": [
                    {"frame_index": 120 + i, "bbox_xyxy": [10.0, 20.0, 40.0, 80.0]}
                ],
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
                "sportsreid_model_id": MODEL,
                "sportsreid_checkpoint_sha256": CKPT,
                "appearance_rank": i,
                "T_max": 0.6 - 0.05 * i,
                "D_max": 0.5,
                "S": 0.1 - 0.05 * i,
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
            "supports_direct_bbox_selection": True,
            "candidates": candidates,
            "metadata": {"fixture": True},
        }
    )
    manifest_path = out_dir / "demo_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    reentry = validate_recovery_event(
        {
            "schema_version": "target_recovery_event_v1",
            "event_id": "evt_demo_reentry_001",
            "project_id": "football-analytics",
            "run_id": "hil_a_demo",
            "target_id": "target_001",
            "event_type": EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE.value,
            "video_id": "demo_video",
            "video_path": "data/test_clips/target_001_independent_holdout_v2.mp4",
            "video_sha256": VIDEO_SHA,
            "created_at": "2026-07-27T00:00:01Z",
            "status": "open",
            "trigger_source": "fixture",
            "trigger_reason": "demo reentry candidates",
            "last_confirmed_segment_id": "H2_SEG_DEMO_000",
            "last_confirmed_frame_index": 100,
            "review_window_start_frame": 101,
            "review_window_end_frame": 250,
            "candidate_manifest_path": str(manifest_path.name),
            "candidate_manifest_sha256": manifest_sha,
            "candidate_count": 4,
            "requires_calibration": True,
            "evidence_paths": [],
            "evidence_sha256": [],
            "provenance": {"fixture": True},
            "metadata": {},
        }
    )

    events = validate_event_list_unique_ids([lost, reentry])
    events_path = out_dir / "demo_recovery_events.jsonl"
    _write_jsonl(events_path, events)

    log_path = out_dir / "demo_decision_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    log = DecisionLog(log_path)

    # 1) defer on lost event
    defer = build_decision(
        decision_id="dec_demo_001",
        project_id="football-analytics",
        run_id="hil_a_demo",
        target_id="target_001",
        event_id="evt_demo_target_lost_001",
        video_id="demo_video",
        video_path="data/test_clips/target_001_independent_holdout_v2.mp4",
        video_sha256=VIDEO_SHA,
        reviewer="fixture_reviewer",
        created_at="2026-07-27T00:01:00Z",
        revision=1,
        action="DEFER",
        confidence="unknown",
        comment="defer lost event",
    )
    log.append(defer, event=lost)

    # 2) confirm via listed candidate on reentry
    confirm = build_decision(
        decision_id="dec_demo_002",
        project_id="football-analytics",
        run_id="hil_a_demo",
        target_id="target_001",
        event_id="evt_demo_reentry_001",
        video_id="demo_video",
        video_path="data/test_clips/target_001_independent_holdout_v2.mp4",
        video_sha256=VIDEO_SHA,
        reviewer="fixture_reviewer",
        created_at="2026-07-27T00:02:00Z",
        revision=1,
        action="CONFIRM_TARGET",
        confidence="confirmed",
        selected_candidate_id="cand_004",
        selected_segment_id="H2_SEG_DEMO_004",
        selected_raw_track_id="raw_track_demo_004",
        selected_frame_index=124,
        selected_bbox_xyxy=[10.0, 20.0, 40.0, 80.0],
        direct_bbox_selection=False,
        candidate_manifest_path=str(manifest_path.name),
        candidate_manifest_sha256=manifest_sha,
        displayed_model_id=MODEL,
        displayed_checkpoint_sha256=CKPT,
        displayed_rank=4,
        displayed_score=-0.1,
        displayed_T_max=0.4,
        displayed_D_max=0.5,
        comment="confirm low-rank candidate (rank helper only)",
    )
    log.append(confirm, event=reentry, candidate_manifest=manifest)

    # 3) undo/supersede confirm via REVOKE
    log.revoke_active_decision(
        prior_decision_id="dec_demo_002",
        new_decision_id="dec_demo_003",
        reviewer="fixture_reviewer",
        created_at="2026-07-27T00:03:00Z",
        revision=2,
        comment="undo confirm",
        event=reentry,
        candidate_manifest=manifest,
    )

    # 4) direct bbox selection confirm after undo
    direct = build_decision(
        decision_id="dec_demo_004",
        project_id="football-analytics",
        run_id="hil_a_demo",
        target_id="target_001",
        event_id="evt_demo_reentry_001",
        video_id="demo_video",
        video_path="data/test_clips/target_001_independent_holdout_v2.mp4",
        video_sha256=VIDEO_SHA,
        reviewer="fixture_reviewer",
        created_at="2026-07-27T00:04:00Z",
        revision=3,
        action="CONFIRM_TARGET",
        confidence="confirmed",
        selected_candidate_id=None,
        selected_segment_id="H2_SEG_DIRECT_001",
        selected_raw_track_id="raw_track_direct_001",
        selected_frame_index=180,
        selected_bbox_xyxy=[50.0, 60.0, 90.0, 160.0],
        direct_bbox_selection=True,
        candidate_manifest_path=str(manifest_path.name),
        candidate_manifest_sha256=manifest_sha,
        displayed_model_id=MODEL,
        displayed_checkpoint_sha256=CKPT,
        displayed_rank=None,
        displayed_score=None,
        comment="direct bbox selection off candidate list",
        supersedes_decision_id="dec_demo_003",
    )
    log.append(direct, event=reentry, candidate_manifest=manifest)

    summary = derive_effective_state_summary(
        log, event_ids=["evt_demo_target_lost_001", "evt_demo_reentry_001"]
    )
    _write_json(out_dir / "demo_effective_decisions.json", summary)
    integrity = log.integrity_report()
    _write_json(out_dir / "append_only_integrity_report.json", integrity)

    return {
        "events_path": str(events_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "decision_log_path": str(log_path),
        "decision_log_sha256": integrity["sha256"],
        "effective_summary": summary,
        "event_types": [e.value for e in EventType],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HIL-A schema validation / demo fixtures")
    parser.add_argument(
        "--demo-out",
        type=Path,
        default=None,
        help="Write demo fixtures to this directory",
    )
    parser.add_argument("--validate-event", type=Path, default=None)
    parser.add_argument("--validate-manifest", type=Path, default=None)
    parser.add_argument("--validate-log", type=Path, default=None)
    parser.add_argument("--list-history", type=Path, default=None)
    parser.add_argument("--event-id", type=str, default=None)
    args = parser.parse_args(argv)

    if args.validate_event:
        rows = [
            json.loads(line)
            for line in args.validate_event.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_event_list_unique_ids(rows)
        print(json.dumps({"ok": True, "events": len(rows)}))
        return 0

    if args.validate_manifest:
        payload = json.loads(args.validate_manifest.read_text(encoding="utf-8"))
        manifest = validate_candidate_manifest(payload)
        print(json.dumps({"ok": True, "candidate_count": manifest["candidate_count"]}))
        return 0

    if args.validate_log:
        log = DecisionLog(args.validate_log)
        rows = log.validate_full_log()
        print(
            json.dumps(
                {
                    "ok": True,
                    "records": len(rows),
                    "sha256": compute_log_sha256(Path(args.validate_log)),
                }
            )
        )
        return 0

    if args.list_history:
        log = DecisionLog(args.list_history)
        hist = log.get_history(event_id=args.event_id)
        print(json.dumps(hist, indent=2, ensure_ascii=False))
        return 0

    if args.demo_out:
        result = build_demo_fixtures(args.demo_out)
        print(json.dumps({k: result[k] for k in result if k != "effective_summary"}, indent=2))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
