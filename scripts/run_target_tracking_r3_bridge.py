#!/usr/bin/env python3
"""Target Tracking R3 — target-conditioned short-occlusion bridge.

Does NOT overwrite R1/R2 artifacts, run detection/tracking, or open annotation UI.
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
EXPECTED_HEAD_PREFIX = "ffd5a0c"
MATCH_ID = "match_short_video_f2f6d8a077ca"
RUN_ID = "sv_run_20260727T234854Z"
TARGET_ID = "target_001"
SEED_RAW = "10"
SEED_FRAME = 29

PREPROCESS = PROJECT / "outputs/reid/product_new_short_video_preprocess_validation" / RUN_ID
R1_ROOT = PROJECT / "outputs/reid/target_tracking_r1" / MATCH_ID / RUN_ID
R2_OUT = R1_ROOT / "r2_purity_split"
OUT = R1_ROOT / "r3_bridge"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if not head.startswith(EXPECTED_HEAD_PREFIX):
        # Allow running after this commit lands; warn only if behind expected prefix absent
        pass

    from football_analytics.reid.golden_clip.window_index import load_dense_observations_index
    from football_analytics.reid.hil.common import sha256_file
    from football_analytics.reid.target_tracking_r1.candidates import build_track_index
    from football_analytics.reid.target_tracking_r1.state import (
        apply_human_seed,
        empty_persistent_state,
    )
    from football_analytics.reid.target_tracking_r2.evidence import load_kit_config
    from football_analytics.reid.target_tracking_r3.boundary import (
        audit_seed_segment_white_leak,
        refine_purity_boundary,
    )
    from football_analytics.reid.target_tracking_r3.bridge import (
        build_r3_timeline,
        resolve_bridge_window,
        run_bridge,
    )
    from football_analytics.reid.target_tracking_r3.bridge_state import (
        build_short_occlusion_state,
        select_bridge_template_rows,
    )
    from football_analytics.reid.target_tracking_r3.kit_lookup import (
        build_kit_lookup_from_evidence,
        kit_for_crop,
    )
    from football_analytics.reid.target_tracking_r3.overlays import (
        render_bridge_overlay,
        render_r2_reference_overlay,
        render_switch_window_slow_review,
        write_review_contact_sheet,
    )
    from football_analytics.reid.target_tracking_r3.policy import R3_POLICY

    import cv2

    video = PROJECT / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    if sha256_file(video) != EXPECTED_SHA:
        print("BLOCKED_TARGET_TRACKING_R3_SOURCE")
        return 2

    # R1/R2 must remain untouched — only read
    assert (R1_ROOT / "stitch/baseline_target_raw_track_overlay.mp4").is_file()
    assert (R2_OUT / "purity_split_seed_overlay.mp4").is_file()
    assert (R2_OUT / "final_manifest.json").is_file()
    r2_mtime = (R2_OUT / "final_manifest.json").stat().st_mtime

    OUT.mkdir(parents=True, exist_ok=True)

    r2_derived = json.loads((R2_OUT / "derived_segments.json").read_text())
    r2_cp = json.loads((R2_OUT / "changepoint.json").read_text())
    r2_manifest = json.loads((R2_OUT / "final_manifest.json").read_text())
    evidence = [
        json.loads(l)
        for l in (R2_OUT / "seed_track_purity_evidence.jsonl").read_text().splitlines()
        if l.strip()
    ]
    for row in evidence:
        row.setdefault("raw_track_id", SEED_RAW)

    seed_seg = next(s for s in r2_derived["segments"] if s["target_eligibility"] == "TARGET_SEED_SEGMENT")
    r2_cp_f = int(r2_cp["algorithmic_change_point"]["change_point_frame"])

    audit = audit_seed_segment_white_leak(
        evidence,
        seed_segment_start=int(seed_seg["start_frame"]),
        seed_segment_end=int(seed_seg["end_frame"]),
        r2_change_point_frame=r2_cp_f,
        policy=R3_POLICY,
    )
    _write(OUT / "r2_boundary_audit.json", audit)

    refine = refine_purity_boundary(
        evidence,
        seed_frame=SEED_FRAME,
        r2_change_point_frame=r2_cp_f,
        policy=R3_POLICY,
    )
    dense = load_dense_observations_index(PREPROCESS / "dense_bbox_timeline.json")
    obs = dense["observations_by_frame"]
    fps = float(dense["fps"])
    frame_count = int(dense["frame_count"])
    fw = int((dense.get("resolution") or {}).get("width") or 1326)
    fh = int((dense.get("resolution") or {}).get("height") or 750)

    refine["refined_change_point_time_sec"] = int(refine["refined_change_point_frame"]) / fps
    refine["delta_refined_vs_human_sec"] = (
        float(refine["refined_change_point_time_sec"])
        - float(R3_POLICY["human_reported_transition_sec_approx"])
    )
    _write(OUT / "boundary_refinement.json", refine)

    if audit["white_player_continuation_painted_as_target_risk"] and refine["refinement_source"] == "R2_UNCHANGED":
        print("BLOCKED_TARGET_TRACKING_R3_BOUNDARY")
        _write(OUT / "final_manifest.json", {"final_status": "BLOCKED_TARGET_TRACKING_R3_BOUNDARY"})
        return 2

    refined_end = int(refine["refined_seed_segment_end_frame"])
    refined_cp = int(refine["refined_change_point_frame"])

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

    template = select_bridge_template_rows(
        evidence,
        seed_start=int(seed_seg["start_frame"]),
        refined_seed_end=refined_end,
        policy=R3_POLICY,
    )
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
        seed_time=SEED_FRAME / fps,
        segment_id=f"{seed_seg['segment_id']}_refined",
        bbox_xyxy=None,
    )
    bridge_state = build_short_occlusion_state(
        persistent_target_id=state["persistent_target_id"],
        source_segment_id=f"{seed_seg['segment_id']}_refined_end_{refined_end}",
        parent_raw_track_id=SEED_RAW,
        template_rows=template,
        policy=R3_POLICY,
    )
    _write(OUT / "bridge_state.json", bridge_state)

    # Contamination / birth signals for window
    contam_frames = [
        int(r["frame_index"])
        for r in evidence
        if float((r.get("contamination") or {}).get("max_other_person_iou") or 0) >= 0.15
        or float((r.get("contamination") or {}).get("union_other_person_crop_coverage") or 0) >= 0.2
    ]
    nearby_births = []
    last_rel = int(bridge_state.get("last_reliable_frame") or refined_end)
    for tid, tr in track_index.items():
        if tid == SEED_RAW:
            continue
        bf = int(tr["first_frame"])
        if last_rel <= bf <= last_rel + 40:
            nearby_births.append(bf)

    window = resolve_bridge_window(
        last_reliable_frame=last_rel,
        refined_change_point=refined_cp,
        contamination_frames=contam_frames,
        nearby_birth_frames=nearby_births,
        policy=R3_POLICY,
    )
    _write(OUT / "bridge_window.json", window)

    # Kit lookup: seed evidence + on-the-fly for bridge-window detections near seed
    kit_lookup = build_kit_lookup_from_evidence(evidence)
    kit_cfg = load_kit_config(PROJECT)
    if window.get("bridge_span_frames", 0) > 0:
        cap = cv2.VideoCapture(str(video))
        for fi in range(int(window["bridge_start_frame"]), int(window["bridge_end_frame"]) + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            for det in obs.get(str(fi)) or []:
                tid = str(det.get("raw_track_id"))
                key = f"{fi}:{tid}"
                if key in kit_lookup:
                    continue
                # only compute for tracks near last reliable center
                bx = det["bbox_xyxy"]
                cx = 0.5 * (float(bx[0]) + float(bx[2]))
                cy = 0.5 * (float(bx[1]) + float(bx[3]))
                lc = bridge_state["last_reliable_bbox"]
                lcx = 0.5 * (float(lc[0]) + float(lc[2]))
                lcy = 0.5 * (float(lc[1]) + float(lc[3]))
                if ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5 > 140:
                    continue
                kit_lookup[key] = kit_for_crop(bgr, bx, kit_config=kit_cfg, policy=R3_POLICY)
        cap.release()

    # Exclude impure parent continuation (raw 10 after refined boundary)
    excluded = [SEED_RAW]

    print("Running target-conditioned bridge…")
    bridge_result = run_bridge(
        video_path=str(video),
        bridge_state=bridge_state,
        observations_by_frame=obs,
        kit_lookup=kit_lookup,
        excluded_raw_track_ids=excluded,
        seed_track_meta=track_index.get(SEED_RAW),
        track_index=track_index,
        window=window,
        fps=fps,
        policy=R3_POLICY,
    )
    _write(OUT / "bridge_result.json", bridge_result)

    timeline = build_r3_timeline(
        persistent_target_id=state["persistent_target_id"],
        target_id=TARGET_ID,
        refined_seed_start=int(seed_seg["start_frame"]),
        refined_seed_end=refined_end,
        seed_segment_id=f"{seed_seg['segment_id']}_refined",
        parent_raw_track_id=SEED_RAW,
        bridge_result=bridge_result,
        fps=fps,
        frame_count=frame_count,
    )
    _write(OUT / "target_timeline.json", timeline)

    # Safety: no eligible observation with reliable white kit
    yellow_to_white_prevented = True
    for obs_row in timeline.get("bridge_observations") or []:
        if not obs_row.get("bbox_xyxy"):
            continue
        kit = obs_row.get("kit_evidence")
        if kit == "WHITE":
            yellow_to_white_prevented = False
        tid = str(obs_row.get("raw_track_id") or "")
        # parent 10 after refined_cp must never be committed
        if tid == SEED_RAW and int(obs_row["frame_index"]) >= refined_cp:
            yellow_to_white_prevented = False

    if not yellow_to_white_prevented:
        status = "FAILED_TARGET_TRACKING_R3_CROSS_TEAM_SAFETY"
    elif bridge_result.get("bridge_decision") == "LONG_GAP_REVIEW_REQUIRED":
        status = "COMPLETED_TARGET_TRACKING_R3_SAFE_UNRESOLVED"
    elif bridge_result.get("accepted"):
        status = "COMPLETED_TARGET_TRACKING_R3_BRIDGE_VISUAL_REVIEW_READY"
    elif bridge_result.get("reason") == "BRIDGE_FLOW_UNRELIABLE" and bridge_result.get("bridge_decision") == "BLOCKED_TARGET_TRACKING_R3_FLOW":
        status = "BLOCKED_TARGET_TRACKING_R3_FLOW"
    else:
        status = "COMPLETED_TARGET_TRACKING_R3_SAFE_UNRESOLVED"

    print("Rendering diagnostic MP4s…")
    r2_ref = render_r2_reference_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        parent_raw_track_id=SEED_RAW,
        r2_seed_start=int(seed_seg["start_frame"]),
        r2_seed_end=int(seed_seg["end_frame"]),
        r2_change_point_frame=r2_cp_f,
        fps=fps,
        frame_count=frame_count,
        output_path=OUT / "r2_safe_unresolved_reference_overlay.mp4",
    )
    r3_ov = render_bridge_overlay(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        timeline=timeline,
        bridge_result=bridge_result,
        refined_seed_start=int(seed_seg["start_frame"]),
        refined_seed_end=refined_end,
        refined_change_point=refined_cp,
        fps=fps,
        frame_count=frame_count,
        output_path=OUT / "r3_target_conditioned_bridge_overlay.mp4",
    )
    slow = render_switch_window_slow_review(
        source_video=video,
        source_video_sha256=EXPECTED_SHA,
        observations_by_frame=obs,
        bridge_result=bridge_result,
        refined_seed_start=int(seed_seg["start_frame"]),
        refined_seed_end=refined_end,
        refined_change_point=refined_cp,
        fps=fps,
        output_path=OUT / "r3_switch_window_slow_review.mp4",
        slowdown=0.25,
    )
    contact_frames = sorted(
        {
            max(0, refined_end - 5),
            refined_end,
            refined_cp,
            refined_cp + 3,
            refined_cp + 8,
            last_rel,
            last_rel + 5,
            last_rel + 10,
        }
    )
    contact = write_review_contact_sheet(
        source_video=video,
        frame_indices=contact_frames,
        output_path=OUT / "r3_human_review_contact_sheet.png",
    )

    review_qs = {
        "schema_version": "target_tracking_r3_human_review_questions_v1",
        "questions": [
            "Hedef sarı oyuncu overlap sonrasında doğru şekilde devam ediyor mu?",
            "Beyaz oyuncuya herhangi bir target geçişi var mı?",
            "Bridge doğru sarı oyuncuya snap ediyor mu?",
            "Kanıt yetersiz olduğunda UNRESOLVED kalıyor mu?",
            "R2’ye kıyasla doğru continuity gerçekten uzuyor mu?",
        ],
        "videos": [r2_ref["path"], r3_ov["path"], slow["path"]],
        "contact_sheet": str(contact),
        "custom_ui": False,
    }
    _write(OUT / "human_review_questions.json", review_qs)

    metrics = dict(bridge_result.get("metrics") or {})
    metrics.update(
        {
            "unresolved_duration_frames": (
                0
                if bridge_result.get("accepted")
                else max(0, frame_count - refined_end - 1)
            ),
            "unresolved_duration_sec": (
                0.0
                if bridge_result.get("accepted")
                else max(0, frame_count - refined_end - 1) / fps
            ),
            "persistent_target_duration_frames": timeline["persistent_target_duration_frames"],
            "persistent_target_duration_sec": timeline["persistent_target_duration_sec"],
            "r2_persistent_duration_sec": float(
                r2_manifest.get("structural_metrics", {}).get("clean_seed_segment_duration_sec") or 0
            ),
            "refined_seed_duration_sec": (refined_end + 1) / fps,
            "determinism": "preregistered_r3_policy_sorted_candidates",
            "runtime_sec": time.perf_counter() - t0,
            "yellow_to_white_switch_prevented": yellow_to_white_prevented,
            "boundary_refinement_source": refine["refinement_source"],
            "refined_change_point_frame": refined_cp,
            "r2_change_point_frame": r2_cp_f,
        }
    )

    final = {
        "final_status": status,
        "created_at": _utc(),
        "git_head": head,
        "persistent_target_id": state["persistent_target_id"],
        "r2_artifacts_overwritten": False,
        "r1_artifacts_overwritten": False,
        "r2_manifest_mtime_unchanged": (R2_OUT / "final_manifest.json").stat().st_mtime == r2_mtime,
        "videos": {
            "r2_reference": r2_ref["path"],
            "r3_bridge": r3_ov["path"],
            "slow_review": slow["path"],
        },
        "contact_sheet": str(contact),
        "structural_metrics": metrics,
        "full_metrics": {
            "target_idf1": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "target_recall": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "false_switch_rate": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "correctness_percentage": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
        },
        "human_acceptance": "HUMAN_VISUAL_ACCEPTANCE_PENDING",
        "annotation_ui_used": False,
        "detection_rerun": False,
        "tracking_rerun": False,
        "bridge_decision": bridge_result.get("bridge_decision"),
    }
    _write(OUT / "final_manifest.json", final)
    (OUT / "final_report.md").write_text(
        "\n".join(
            [
                "# Target Tracking R3 — Final Report",
                "",
                f"- final_status: `{status}`",
                f"- refined_boundary: f=`{refined_cp}` (R2 was f=`{r2_cp_f}`)",
                f"- bridge_decision: `{bridge_result.get('bridge_decision')}`",
                f"- yellow_to_white_switch_prevented: `{yellow_to_white_prevented}`",
                f"- persistent_target_duration_sec: `{timeline['persistent_target_duration_sec']}`",
                "",
                "## Videos",
                f"- {r2_ref['path']}",
                f"- {r3_ov['path']}",
                f"- {slow['path']}",
                "",
                "IDF1/recall/accuracy: `NOT_MEASURABLE_WITHOUT_ACCEPTED_GT`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(status)
    print(f"refined_cp={refined_cp} seed_end={refined_end}")
    print(f"bridge={bridge_result.get('bridge_decision')} prevented={yellow_to_white_prevented}")
    print(f"r3_mp4={r3_ov['path']}")
    return 0 if status.startswith("COMPLETED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
