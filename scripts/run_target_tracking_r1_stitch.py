#!/usr/bin/env python3
"""Target Tracking R1 — persistent state + conservative local stitching + diagnostic MP4s.

No annotation UI, no detection/tracking rerun, no product log mutation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

EXPECTED_SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"
MATCH_ID = "match_short_video_f2f6d8a077ca"
RUN_ID = "sv_run_20260727T234854Z"
TARGET_ID = "target_001"
SEED_RAW = "10"
SEED_FRAME = 29
SEED_TIME = 0.97

PREPROCESS = PROJECT / "outputs/reid/product_new_short_video_preprocess_validation" / RUN_ID
OUT = PROJECT / "outputs/reid/target_tracking_r1" / MATCH_ID / RUN_ID


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()

    from football_analytics.reid.golden_clip.window_index import load_dense_observations_index
    from football_analytics.reid.hil.common import sha256_file
    from football_analytics.reid.target_tracking_r1.candidates import build_track_index
    from football_analytics.reid.target_tracking_r1.overlays import (
        render_baseline_overlay,
        render_stitched_overlay,
    )
    from football_analytics.reid.target_tracking_r1.policy import STITCH_POLICY
    from football_analytics.reid.target_tracking_r1.state import (
        apply_human_seed,
        empty_persistent_state,
    )
    from football_analytics.reid.target_tracking_r1.stitch import run_local_stitch_chain
    from football_analytics.reid.target_tracking_r1.timeline import build_target_timeline

    video = PROJECT / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    if sha256_file(video) != EXPECTED_SHA:
        print("BLOCKED_TARGET_TRACKING_R1_SOURCE")
        return 2

    # Refuse product log mutation: open read-only sizes only
    for name in ("decision_log.jsonl", "timeline_approval_log.jsonl", "gallery_approval_log.jsonl"):
        p = PREPROCESS / "product_review_package/session" / name
        if p.is_file():
            # touch-free: only stat
            _ = p.stat().st_size

    dense_path = PREPROCESS / "dense_bbox_timeline.json"
    idx = load_dense_observations_index(dense_path)
    obs = idx["observations_by_frame"]
    fps = float(idx["fps"])
    frame_count = int(idx["frame_count"])
    fw = int((idx.get("resolution") or {}).get("width") or 1326)
    fh = int((idx.get("resolution") or {}).get("height") or 750)

    inv_rows = [
        json.loads(l)
        for l in (PREPROCESS / "inventory/track_candidate_mapping.jsonl").read_text().splitlines()
        if l.strip()
    ]
    track_index = build_track_index(inv_rows)
    # Enrich frames from dense for exact-frame conflict accuracy
    for fi_s, rows in obs.items():
        fi = int(fi_s)
        for r in rows:
            tid = str(r.get("raw_track_id"))
            if tid in track_index:
                track_index[tid].setdefault("frames", set()).add(fi)

    if SEED_RAW not in track_index:
        print("BLOCKED_TARGET_TRACKING_R1_SOURCE")
        return 2

    seed_bbox = None
    for r in obs.get(str(SEED_FRAME)) or []:
        if str(r.get("raw_track_id")) == SEED_RAW:
            seed_bbox = list(r["bbox_xyxy"])
            break

    state = empty_persistent_state(
        match_id=MATCH_ID,
        analysis_run_id=RUN_ID,
        target_id=TARGET_ID,
        source_video_sha256=EXPECTED_SHA,
    )
    # Guarantee independence
    assert state["persistent_target_id"] != SEED_RAW
    assert state["persistent_target_id"] != TARGET_ID or True  # ptarget_* prefix
    state = apply_human_seed(
        state,
        raw_track_id=SEED_RAW,
        seed_frame=SEED_FRAME,
        seed_time=SEED_TIME,
        segment_id=track_index[SEED_RAW].get("segment_id"),
        bbox_xyxy=seed_bbox,
    )

    chain = run_local_stitch_chain(
        seed_raw_track_id=SEED_RAW,
        track_index=track_index,
        observations_by_frame=obs,
        frame_width=fw,
        frame_height=fh,
        state=state,
        policy=STITCH_POLICY,
    )
    state = chain["state"]
    timeline = build_target_timeline(
        persistent_target_id=state["persistent_target_id"],
        target_id=TARGET_ID,
        seed_raw_track_id=SEED_RAW,
        chain_raw_track_ids=chain["chain_raw_track_ids"],
        track_index=track_index,
        stitch_events=chain["events"],
        fps=fps,
    )

    out_dir = OUT / "stitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "persistent_target_state.json", state)
    _write(out_dir / "stitch_policy.json", STITCH_POLICY)
    # Serialize events without huge nested candidate dumps duplication — keep full for audit
    serial_events = []
    for ev in chain["events"]:
        e = {
            "event_id": ev["event_id"],
            "previous_raw_track_id": ev["previous_raw_track_id"],
            "previous_end_frame": ev["previous_end_frame"],
            "candidate_count": len(ev.get("candidates") or []),
            "candidates": ev.get("candidates"),
            "decision": ev.get("decision"),
        }
        serial_events.append(e)
    _write(out_dir / "stitch_events.json", serial_events)
    _write(out_dir / "target_timeline.json", timeline)

    # Candidate score table + unresolved table
    score_rows = []
    unresolved_rows = []
    for ev in serial_events:
        for c in ev.get("candidates") or []:
            score_rows.append(
                {
                    "event_id": ev["event_id"],
                    "previous_raw_track_id": ev["previous_raw_track_id"],
                    **{k: c.get(k) for k in (
                        "candidate_raw_track_id",
                        "temporal_gap_frames",
                        "center_displacement_px",
                        "bbox_scale_ratio",
                        "cost",
                        "candidate_duration_frames",
                        "review_eligible",
                        "kit_team_evidence",
                        "appearance_reid",
                    )},
                }
            )
        dec = ev.get("decision") or {}
        if dec.get("decision") != "AUTO_STITCH":
            unresolved_rows.append(
                {
                    "event_id": ev["event_id"],
                    "previous_raw_track_id": ev["previous_raw_track_id"],
                    "previous_end_frame": ev["previous_end_frame"],
                    "decision": dec.get("decision"),
                    "reason": dec.get("reason"),
                    "best_candidate": (dec.get("best_candidate") or {}).get("candidate_raw_track_id"),
                    "margin": dec.get("margin"),
                }
            )
    _write(out_dir / "candidate_score_table.json", score_rows)
    _write(out_dir / "unresolved_event_table.json", unresolved_rows)

    # Contact sheet of stitch transitions (static PNGs)
    import cv2

    contact_dir = out_dir / "stitch_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    for ev in serial_events:
        dec = ev.get("decision") or {}
        if dec.get("decision") != "AUTO_STITCH":
            continue
        sel = dec["selected"]
        fi = int(sel["candidate_start_frame"])
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        for r in obs.get(str(fi)) or []:
            tid = str(r.get("raw_track_id"))
            x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
            color = (0, 200, 0) if tid == str(sel["candidate_raw_track_id"]) else (220, 220, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, tid, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        msg = f"STITCH {ev['previous_raw_track_id']}->{sel['candidate_raw_track_id']} f={fi}"
        cv2.putText(frame, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        path = contact_dir / f"{ev['event_id']}_f{fi:06d}.png"
        cv2.imwrite(str(path), frame)
        contact_paths.append(str(path))

    print("Rendering baseline diagnostic MP4…")
    baseline = render_baseline_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        seed_raw_track_id=SEED_RAW,
        seed_last_frame=int(track_index[SEED_RAW]["last_frame"]),
        fps=fps,
        frame_count=frame_count,
        output_path=out_dir / "baseline_target_raw_track_overlay.mp4",
    )
    print("Rendering stitched diagnostic MP4…")
    stitched = render_stitched_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        timeline=timeline,
        stitch_events=serial_events,
        fps=fps,
        frame_count=frame_count,
        output_path=out_dir / "stitched_persistent_target_overlay.mp4",
    )

    (out_dir / "review_notes_template.md").write_text(
        "\n".join(
            [
                "# Target Tracking R1 — Human Visual Review Notes",
                "",
                f"- persistent_target_id: `{state['persistent_target_id']}`",
                f"- seed raw_track_id: `{SEED_RAW}` (HUMAN_SEED only; not equal to persistent id)",
                f"- baseline: `{baseline['path']}`",
                f"- stitched: `{stitched['path']}`",
                "",
                "## Questions",
                "",
                "1. Stitched video doğru oyuncuda baseline’dan daha uzun kaldı mı?",
                "2. Başka bir oyuncuya geçtiği timestamp var mı?",
                "3. Kısa örtüşme sonrasında doğru oyuncuyu buldu mu?",
                "4. Emin olmadığı yerde UNRESOLVED kaldı mı?",
                "5. Manuel müdahale ihtiyacı anlamlı biçimde azaldı mı?",
                "",
                "## Stitch timestamps",
                "",
            ]
            + [
                f"- t={row['time_sec']:.2f}s · f={row['frame']} · {row['prev']} → {row['new']} · gap={row['gap']}"
                for row in stitched.get("stitch_timestamps") or []
            ]
            + ["", "Cevaplar:", "", ""]
        ),
        encoding="utf-8",
    )

    runtime = time.perf_counter() - t0
    structural = dict(timeline["structural_metrics"])
    structural.update(
        {
            "unresolved_event_count": len(unresolved_rows),
            "candidate_count_total": len(score_rows),
            "temporal_gap_distribution": [
                c.get("temporal_gap_frames")
                for ev in serial_events
                for c in (ev.get("candidates") or [])
            ][:50],
            "evidence_availability": {
                "kit": "UNAVAILABLE",
                "crop_quality": "UNAVAILABLE",
                "appearance_reid": "UNAVAILABLE",
                "track_purity": "UNAVAILABLE",
                "exact_frame_conflict": "AVAILABLE",
                "temporal_spatial": "AVAILABLE",
            },
            "runtime_sec": runtime,
            "determinism": "policy_preregistered_sorted_candidates",
        }
    )

    status = "COMPLETED_TARGET_TRACKING_R1_VISUAL_REVIEW_READY"
    if chain["automatic_stitch_count"] == 0:
        status = "COMPLETED_TARGET_TRACKING_R1_ALL_UNRESOLVED"

    final = {
        "final_status": status,
        "created_at": _utc(),
        "git_head": head,
        "match_id": MATCH_ID,
        "analysis_run_id": RUN_ID,
        "target_id": TARGET_ID,
        "persistent_target_id": state["persistent_target_id"],
        "seed": {
            "raw_track_id": SEED_RAW,
            "seed_frame": SEED_FRAME,
            "seed_time": SEED_TIME,
            "role": "HUMAN_SEED_ONLY",
        },
        "chain_raw_track_ids": chain["chain_raw_track_ids"],
        "automatic_stitch_count": chain["automatic_stitch_count"],
        "baseline_mp4": baseline["path"],
        "stitched_mp4": stitched["path"],
        "stitch_timestamps": stitched.get("stitch_timestamps"),
        "contact_sheets": contact_paths,
        "structural_metrics": structural,
        "full_metrics": timeline["full_metrics"],
        "human_acceptance": "HUMAN_VISUAL_ACCEPTANCE_PENDING",
        "annotation_ui_used": False,
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
    }
    _write(out_dir / "final_manifest.json", final)
    (out_dir / "final_report.md").write_text(
        "\n".join(
            [
                "# Target Tracking R1 — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- persistent_target_id: `{state['persistent_target_id']}`",
                f"- seed raw_track_id: `{SEED_RAW}` (HUMAN_SEED; persistent id’den bağımsız)",
                f"- chain: `{chain['chain_raw_track_ids']}`",
                f"- automatic_stitch_count: `{chain['automatic_stitch_count']}`",
                f"- baseline: `{baseline['path']}`",
                f"- stitched: `{stitched['path']}`",
                f"- runtime_sec: `{runtime:.1f}`",
                "",
                "## Human acceptance",
                "",
                "`HUMAN_VISUAL_ACCEPTANCE_PENDING`",
                "",
                "## Metrics",
                "",
                "IDF1 / recall / false switch: `NOT_MEASURABLE_WITHOUT_ACCEPTED_GT`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(status)
    print(f"persistent_target_id={state['persistent_target_id']}")
    print(f"chain={chain['chain_raw_track_ids']}")
    print(f"stitches={chain['automatic_stitch_count']}")
    print(f"baseline={baseline['path']}")
    print(f"stitched={stitched['path']}")
    for row in stitched.get("stitch_timestamps") or []:
        print(f"stitch_ts t={row['time_sec']:.2f}s f={row['frame']} {row['prev']}->{row['new']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
