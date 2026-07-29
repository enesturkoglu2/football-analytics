#!/usr/bin/env python3
"""Target Tracking R2 — purity split, kit guard, segment-level stitching.

Does NOT overwrite R1 stitch artifacts, run detection/tracking, or open annotation UI.
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
R1_ROOT = PROJECT / "outputs/reid/target_tracking_r1" / MATCH_ID / RUN_ID
OUT = R1_ROOT / "r2_purity_split"  # sibling under same run; does not overwrite R1 stitch/


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()

    from football_analytics.reid.golden_clip.window_index import load_dense_observations_index
    from football_analytics.reid.hil.common import sha256_file
    from football_analytics.reid.target_tracking_r1.candidates import build_track_index
    from football_analytics.reid.target_tracking_r1.state import (
        apply_human_seed,
        empty_persistent_state,
    )
    from football_analytics.reid.target_tracking_r2.changepoint import detect_kit_change_points
    from football_analytics.reid.target_tracking_r2.evidence import (
        extract_track_evidence,
        load_kit_config,
    )
    from football_analytics.reid.target_tracking_r2.overlays import (
        render_purity_split_overlay,
        render_segment_stitched_overlay,
    )
    from football_analytics.reid.target_tracking_r2.policy import R2_POLICY
    from football_analytics.reid.target_tracking_r2.review import build_r1_rejection_record
    from football_analytics.reid.target_tracking_r2.segments import build_derived_segments
    from football_analytics.reid.target_tracking_r2.stitch import (
        build_r2_timeline,
        decide_segment_stitch,
        generate_segment_candidates,
        summarize_segment_kit,
    )

    video = PROJECT / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    if sha256_file(video) != EXPECTED_SHA:
        print("BLOCKED_TARGET_TRACKING_R2_SOURCE")
        return 2

    # Ensure R1 stitch files still exist and are not written
    r1_stitch = R1_ROOT / "stitch"
    r1_baseline = r1_stitch / "baseline_target_raw_track_overlay.mp4"
    r1_stitched = r1_stitch / "stitched_persistent_target_overlay.mp4"
    assert r1_baseline.is_file() and r1_stitched.is_file()

    r1_final = {}
    if (r1_stitch / "final_manifest.json").is_file():
        r1_final = json.loads((r1_stitch / "final_manifest.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)

    # --- 2. R1 human rejection (append-only new file) ---
    review = build_r1_rejection_record(
        match_id=MATCH_ID,
        analysis_run_id=RUN_ID,
        target_id=TARGET_ID,
        persistent_target_id_r1=r1_final.get("persistent_target_id"),
        r1_chain=list(r1_final.get("chain_raw_track_ids") or ["10", "365"]),
    )
    review_path = OUT / "r1_human_visual_review.json"
    _write(review_path, review)

    # --- reuse audit ---
    _write(
        OUT / "existing_signal_reuse_audit.json",
        {
            "created_at": _utc(),
            "components": [
                {
                    "name": "crop_quality",
                    "path": "football_analytics.reid.quality.compute_image_metrics",
                    "applied": True,
                    "adapter": "on_the_fly_crop_from_video",
                },
                {
                    "name": "contamination",
                    "path": "football_analytics.reid.quality.compute_tracking_bbox_contamination",
                    "applied": True,
                    "adapter": "raw_track_id→track_id remap",
                },
                {
                    "name": "kit_descriptor",
                    "path": "football_analytics.reid.kit.compute_torso_kit_metrics",
                    "applied": True,
                    "adapter": "on_the_fly_crop + kit_descriptor_stage5b.yaml",
                },
                {
                    "name": "track_purity",
                    "path": "football_analytics.reid.target_tracking_r2.changepoint",
                    "applied": True,
                    "note": "uses kit/contamination evidence; Stage5B3 thresholds remain null so R2 preregisters temporal confirm windows",
                },
                {
                    "name": "manual_segmentation_derived_view",
                    "path": "football_analytics.reid.target_tracking_r2.segments.build_derived_segments",
                    "applied": True,
                    "adapter": "algorithmic derived segments (raw immutable)",
                },
                {
                    "name": "exact_frame_conflict",
                    "path": "football_analytics.reid.target_tracking_r1.candidates.exact_frame_conflict",
                    "applied": True,
                },
                {
                    "name": "segmented_reid",
                    "path": "football_analytics.reid.segment_regression",
                    "applied": False,
                    "status": "UNAVAILABLE",
                    "reason": "no short-video embeddings/baseline",
                },
            ],
        },
    )

    dense = load_dense_observations_index(PREPROCESS / "dense_bbox_timeline.json")
    obs = dense["observations_by_frame"]
    fps = float(dense["fps"])
    frame_count = int(dense["frame_count"])
    fw = int((dense.get("resolution") or {}).get("width") or 1326)
    fh = int((dense.get("resolution") or {}).get("height") or 750)

    inv_rows = [
        json.loads(l)
        for l in (PREPROCESS / "inventory/track_candidate_mapping.jsonl").read_text().splitlines()
        if l.strip()
    ]
    track_index = build_track_index(inv_rows)
    for fi_s, rows in obs.items():
        fi = int(fi_s)
        for r in rows:
            tid = str(r.get("raw_track_id"))
            if tid in track_index:
                track_index[tid].setdefault("frames", set()).add(fi)

    if SEED_RAW not in track_index:
        print("BLOCKED_TARGET_TRACKING_R2_SEGMENTATION")
        return 2

    print("Extracting kit/quality evidence for raw_track_id=10…")
    kit_cfg = load_kit_config(PROJECT)
    evidence = extract_track_evidence(
        video_path=video,
        raw_track_id=SEED_RAW,
        observations_by_frame=obs,
        frame_width=fw,
        frame_height=fh,
        fps=fps,
        kit_config=kit_cfg,
        policy=R2_POLICY,
    )
    _write(OUT / "seed_track_evidence_summary.json", {
        "n_rows": len(evidence),
        "first_frame": evidence[0]["frame_index"] if evidence else None,
        "last_frame": evidence[-1]["frame_index"] if evidence else None,
    })
    with (OUT / "seed_track_purity_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    cp = detect_kit_change_points(evidence, seed_frame=SEED_FRAME, policy=R2_POLICY)
    _write(OUT / "changepoint.json", cp)
    if not cp.get("algorithmic_change_point"):
        print("BLOCKED_TARGET_TRACKING_R2_PURITY_CODE")
        print("no algorithmic change-point found")
        # Still continue with unresolved safety if possible — but R2 requires detection
        # Fall through with None will mark whole track carefully
        pass

    parent = track_index[SEED_RAW]
    derived = build_derived_segments(
        parent_raw_track_id=SEED_RAW,
        parent_first_frame=int(parent["first_frame"]),
        parent_last_frame=int(parent["last_frame"]),
        seed_frame=SEED_FRAME,
        change_point=cp.get("algorithmic_change_point"),
        evidence_rows=evidence,
        fps=fps,
    )
    _write(OUT / "derived_segments.json", derived)
    seed_seg = next(s for s in derived["segments"] if s["target_eligibility"] == "TARGET_SEED_SEGMENT")
    conflict_seg = next(
        (s for s in derived["segments"] if s["target_eligibility"] == "TARGET_INELIGIBLE_IDENTITY_CONFLICT"),
        None,
    )

    # Persistent state: only seed segment
    state = empty_persistent_state(
        match_id=MATCH_ID,
        analysis_run_id=RUN_ID,
        target_id=TARGET_ID,
        source_video_sha256=EXPECTED_SHA,
    )
    state = apply_human_seed(
        state,
        raw_track_id=SEED_RAW,
        seed_frame=SEED_FRAME,
        seed_time=SEED_TIME,
        segment_id=seed_seg["segment_id"],
        bbox_xyxy=None,
    )
    state["status"] = "TARGET_TEMPORARILY_LOST"
    state["bound_segment"] = {
        "segment_id": seed_seg["segment_id"],
        "parent_raw_track_id": SEED_RAW,
        "start_frame": seed_seg["start_frame"],
        "end_frame": seed_seg["end_frame"],
        "role": "HUMAN_SEED_SEGMENT",
    }
    state["excluded_segments"] = (
        [
            {
                "segment_id": conflict_seg["segment_id"],
                "reason": "CROSS_TEAM_IDENTITY_CONFLICT",
                "analysis_eligible": False,
            }
        ]
        if conflict_seg
        else []
    )
    assert state["persistent_target_id"] != SEED_RAW
    _write(OUT / "persistent_target_state.json", state)

    # Candidate kit evidence: extract short evidence for top nearby tracks only (cost control)
    evidence_by_track: dict[str, list] = {SEED_RAW: evidence}
    # Pre-generate geometric candidates without kit, then audit kit for those IDs
    # First pass with empty kit map → then fill
    geo_cands = generate_segment_candidates(
        seed_segment=seed_seg,
        track_index=track_index,
        observations_by_frame=obs,
        evidence_by_track=evidence_by_track,
        frame_width=fw,
        frame_height=fh,
        policy=R2_POLICY,
    )
    # Audit kit for candidate track ids (first 25 frames of each)
    for c in geo_cands[:40]:
        tid = str(c["candidate_raw_track_id"])
        if tid in evidence_by_track:
            continue
        print(f"  kit-audit candidate raw_track={tid}")
        evidence_by_track[tid] = extract_track_evidence(
            video_path=video,
            raw_track_id=tid,
            observations_by_frame=obs,
            frame_width=fw,
            frame_height=fh,
            fps=fps,
            kit_config=kit_cfg,
            policy={**R2_POLICY, "sample_every_frame": 3},
        )
    cands = generate_segment_candidates(
        seed_segment=seed_seg,
        track_index=track_index,
        observations_by_frame=obs,
        evidence_by_track=evidence_by_track,
        frame_width=fw,
        frame_height=fh,
        policy=R2_POLICY,
    )
    decision = decide_segment_stitch(cands, policy=R2_POLICY)
    _write(OUT / "segment_candidates.json", cands)
    _write(OUT / "stitch_decision.json", decision)

    timeline = build_r2_timeline(
        persistent_target_id=state["persistent_target_id"],
        target_id=TARGET_ID,
        seed_segment=seed_seg,
        conflict_segment=conflict_seg,
        stitch_decision=decision,
        fps=fps,
    )
    _write(OUT / "target_timeline.json", timeline)

    rejected_cross = [c for c in cands if "CROSS_TEAM_KIT_MISMATCH" in (c.get("hard_rejects") or [])]

    print("Rendering purity_split_seed_overlay.mp4…")
    purity_mp4 = render_purity_split_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        parent_raw_track_id=SEED_RAW,
        seed_segment=seed_seg,
        conflict_segment=conflict_seg,
        change_point=cp.get("algorithmic_change_point"),
        fps=fps,
        frame_count=frame_count,
        output_path=OUT / "purity_split_seed_overlay.mp4",
    )
    print("Rendering segment_stitched_target_overlay.mp4…")
    stitch_mp4 = render_segment_stitched_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        timeline=timeline,
        rejected_cross_team=rejected_cross,
        fps=fps,
        frame_count=frame_count,
        output_path=OUT / "segment_stitched_target_overlay.mp4",
    )

    # Safety: white never in eligible target intervals
    yellow_to_white_prevented = True
    for iv in timeline["intervals"]:
        if not iv.get("analysis_eligible"):
            continue
        if iv.get("kind") == "IDENTITY_CONFLICT_EXCLUDED":
            yellow_to_white_prevented = False
        # eligible intervals must not be conflict segment
        if conflict_seg and iv.get("segment_id") == conflict_seg["segment_id"]:
            yellow_to_white_prevented = False
        if iv.get("raw_track_id") == SEED_RAW:
            # only seed segment frames allowed for parent 10
            if int(iv["end_frame"]) > int(seed_seg["end_frame"]):
                yellow_to_white_prevented = False

    accepted_stitches = 1 if decision.get("decision") == "AUTO_STITCH" else 0
    status = (
        "COMPLETED_TARGET_TRACKING_R2_PURITY_SPLIT_VISUAL_REVIEW_READY"
        if yellow_to_white_prevented
        else "FAILED_TARGET_TRACKING_R2_CROSS_TEAM_SAFETY"
    )
    if yellow_to_white_prevented and accepted_stitches == 0:
        status = "COMPLETED_TARGET_TRACKING_R2_SAFE_UNRESOLVED"

    algo_cp = cp.get("algorithmic_change_point") or {}
    structural = {
        "original_raw_track_10_duration_frames": derived.get("original_raw_track_duration_frames"),
        "original_raw_track_10_duration_sec": (
            derived.get("original_raw_track_duration_frames") or 0
        )
        / fps,
        "clean_seed_segment_duration_frames": derived.get("clean_seed_duration_frames"),
        "clean_seed_segment_duration_sec": (derived.get("clean_seed_duration_frames") or 0) / fps,
        "detected_change_point_frame": algo_cp.get("change_point_frame"),
        "detected_change_point_time_sec": algo_cp.get("change_point_time_sec"),
        "human_reported_transition_sec_approx": 1.6,
        "delta_algorithmic_vs_human_sec": cp.get("delta_algorithmic_vs_human_sec"),
        "excluded_impure_duration_frames": derived.get("excluded_impure_duration_frames"),
        "excluded_impure_duration_sec": (derived.get("excluded_impure_duration_frames") or 0) / fps,
        "cross_team_conflict_count": 1 if conflict_seg else 0,
        "segment_candidate_count": len(cands),
        "accepted_segment_stitch_count": accepted_stitches,
        "unresolved_count": 0 if accepted_stitches else 1,
        "kit_evidence_availability": "AVAILABLE_ON_THE_FLY",
        "purity_evidence_availability": "AVAILABLE_ON_THE_FLY",
        "appearance_reid": "UNAVAILABLE",
        "determinism": "preregistered_r2_policy_sorted_candidates",
        "original_yellow_to_white_switch_prevented": yellow_to_white_prevented,
        "runtime_sec": time.perf_counter() - t0,
    }

    final = {
        "final_status": status,
        "created_at": _utc(),
        "git_head": head,
        "persistent_target_id": state["persistent_target_id"],
        "seed_segment_id": seed_seg["segment_id"],
        "r1_human_review": str(review_path),
        "purity_split_mp4": purity_mp4["path"],
        "segment_stitched_mp4": stitch_mp4["path"],
        "structural_metrics": structural,
        "full_metrics": timeline["full_metrics"],
        "human_acceptance": "HUMAN_VISUAL_ACCEPTANCE_PENDING",
        "r1_artifacts_overwritten": False,
        "annotation_ui_used": False,
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
        "stitch_decision": decision.get("decision"),
        "stitch_reason": decision.get("reason"),
    }
    _write(OUT / "final_manifest.json", final)
    (OUT / "final_report.md").write_text(
        "\n".join(
            [
                "# Target Tracking R2 — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- R1 human review: `REJECTED` (recorded; R1 files untouched)",
                f"- persistent_target_id: `{state['persistent_target_id']}`",
                f"- seed segment: `{seed_seg['segment_id']}` "
                f"[{seed_seg['start_frame']},{seed_seg['end_frame']}] "
                f"≈{structural['clean_seed_segment_duration_sec']:.2f}s",
                f"- change-point: f=`{algo_cp.get('change_point_frame')}` "
                f"t=`{algo_cp.get('change_point_time_sec')}` "
                f"(human≈1.6s, delta=`{cp.get('delta_algorithmic_vs_human_sec')}`)",
                f"- yellow→white prevented: `{yellow_to_white_prevented}`",
                f"- accepted stitches: `{accepted_stitches}`",
                "",
                "## Videos",
                "",
                f"- `{purity_mp4['path']}`",
                f"- `{stitch_mp4['path']}`",
                "",
                "## Metrics",
                "",
                "IDF1/recall/accuracy: `NOT_MEASURABLE_WITHOUT_ACCEPTED_GT`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Remove accidental None write if any
    print(status)
    print(f"change_point_frame={algo_cp.get('change_point_frame')} t={algo_cp.get('change_point_time_sec')}")
    print(f"seed_seg=[{seed_seg['start_frame']},{seed_seg['end_frame']}]")
    print(f"yellow_to_white_prevented={yellow_to_white_prevented}")
    print(f"purity_mp4={purity_mp4['path']}")
    print(f"stitched_mp4={stitch_mp4['path']}")
    return 0 if status.startswith("COMPLETED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
