#!/usr/bin/env python3
"""Target Tracking R1 — seed contact sheet when no human-approved seed exists.

Does NOT open annotation UI, run detection/tracking, or mutate product logs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

EXPECTED_SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"
MATCH_ID = "match_short_video_f2f6d8a077ca"
RUN_ID = "sv_run_20260727T234854Z"
TARGET_ID = "target_001"
PREPROCESS = (
    PROJECT
    / "outputs/reid/product_new_short_video_preprocess_validation"
    / RUN_ID
)
OUT = (
    PROJECT
    / "outputs/reid/target_tracking_r1"
    / MATCH_ID
    / RUN_ID
    / "seed_request"
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT, text=True
    )
    lr = subprocess.check_output(
        ["git", "rev-list", "--left-right", "--count", "origin/main...HEAD"],
        cwd=PROJECT,
        text=True,
    ).strip()

    decision = PREPROCESS / "product_review_package/session/decision_log.jsonl"
    approval = PREPROCESS / "product_review_package/session/timeline_approval_log.jsonl"
    gallery = PREPROCESS / "product_review_package/session/gallery_approval_log.jsonl"
    ann = (
        PROJECT
        / "outputs/reid/target_golden_clip_r1"
        / MATCH_ID
        / RUN_ID
        / "session/annotation_log.jsonl"
    )

    seed_audit = {
        "decision_log_bytes": decision.stat().st_size if decision.is_file() else None,
        "timeline_approval_log_bytes": approval.stat().st_size if approval.is_file() else None,
        "gallery_approval_log_bytes": gallery.stat().st_size if gallery.is_file() else None,
        "annotation_log_bytes": ann.stat().st_size if ann.is_file() else None,
        "human_approved_seed_found": False,
        "reason_tr": (
            "Product decision/approval logları boş (0 byte). "
            "Golden-clip annotation yalnız INCOMPLETE_UI_FREEZE_EVENT içeriyor. "
            "Otomatik seed seçilmedi."
        ),
    }

    refs_video = PROJECT / "data/product_inputs/short_video_f2f6d8a077ca/kisa_mac_klip.mp4"
    from football_analytics.reid.hil.common import sha256_file

    sha = sha256_file(refs_video)
    if sha != EXPECTED_SHA:
        print("BLOCKED_TARGET_TRACKING_R1_SOURCE")
        return 2

    enroll = json.loads(
        (PREPROCESS / "product_review_package/enrollment_candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    eligible_ids = {str(c["raw_track_id"]) for c in enroll.get("candidates") or []}
    dense = json.loads(
        (PREPROCESS / "dense_bbox_timeline.json").read_text(encoding="utf-8")
    )
    obs = dense.get("observations_by_frame") or {}
    fps = float(dense.get("fps") or 30.0)
    frame_count = int(dense.get("frame_count") or 1357)

    # Single clear start frame in enrollment window [0, 90]: max eligible bboxes
    best_fi, best_n = 30, -1
    for fi in range(0, min(91, frame_count)):
        rows = obs.get(str(fi)) or []
        n = sum(1 for r in rows if str(r.get("raw_track_id")) in eligible_ids)
        if n > best_n:
            best_n = n
            best_fi = fi

    rows = [
        r
        for r in (obs.get(str(best_fi)) or [])
        if str(r.get("raw_track_id")) in eligible_ids
    ]
    # Prefer selectable player rows
    rows = [r for r in rows if r.get("selectable", True)] or rows

    cap = cv2.VideoCapture(str(refs_video))
    if not cap.isOpened():
        print("BLOCKED_TARGET_TRACKING_R1_SOURCE")
        return 2
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(best_fi))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        print("BLOCKED_TARGET_TRACKING_R1_SOURCE")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    drawn = frame.copy()
    visible = []
    for r in rows:
        tid = str(r.get("raw_track_id"))
        x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(
            drawn,
            tid,
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 220),
            2,
            cv2.LINE_AA,
        )
        visible.append(
            {
                "raw_track_id": tid,
                "segment_id": r.get("segment_id"),
                "bbox_xyxy": list(r["bbox_xyxy"]),
                "detection_id": r.get("detection_id"),
            }
        )

    # Banner
    banner = (
        f"SEED REQUEST · f={best_fi} · t={best_fi/fps:.2f}s · "
        f"eligible_on_frame={len(visible)} · SHA={EXPECTED_SHA[:12]}"
    )
    cv2.rectangle(drawn, (0, 0), (drawn.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        drawn,
        banner,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    png = OUT / f"seed_contact_sheet_frame_{best_fi:06d}.png"
    cv2.imwrite(str(png), drawn)

    # Also write a compact ID table for easy reply
    table = {
        "schema_version": "target_tracking_r1_seed_contact_sheet_v1",
        "created_at": _utc(),
        "match_id": MATCH_ID,
        "analysis_run_id": RUN_ID,
        "target_id": TARGET_ID,
        "source_video_sha256": EXPECTED_SHA,
        "frame_index": best_fi,
        "timestamp_sec": best_fi / fps,
        "fps": fps,
        "eligible_universe_count": len(eligible_ids),
        "visible_eligible_count": len(visible),
        "visible_raw_track_ids": [v["raw_track_id"] for v in visible],
        "visible_boxes": visible,
        "contact_sheet_png": str(png),
        "instruction_tr": (
            "Lütfen hedef oyuncunun raw_track_id değerini bildirin "
            "(ör. 875). Otomatik seçim yapılmayacak."
        ),
        "seed_audit": seed_audit,
        "git": {
            "head": head,
            "branch": branch,
            "origin_main_left_right": lr,
            "working_tree_clean": status.strip() == "",
        },
        "annotation_ui_used": False,
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
    }
    _write(OUT / "seed_contact_sheet_manifest.json", table)

    status_name = "TARGET_TRACKING_R1_USER_SEED_REQUIRED"
    final = {
        "final_status": status_name,
        "created_at": _utc(),
        "exact_next_gate": "TARGET_TRACKING_R1_AWAIT_USER_RAW_TRACK_ID",
        "seed_contact_sheet": str(png),
        "seed_manifest": str(OUT / "seed_contact_sheet_manifest.json"),
        "frame_index": best_fi,
        "timestamp_sec": best_fi / fps,
        "visible_eligible_count": len(visible),
        "human_approved_seed_found": False,
        "metrics": {
            "target_idf1": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "target_recall": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
            "false_identity_switch_rate": "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT",
        },
        "git_head": head,
        "note_tr": (
            "Onaylı seed yok. Contact sheet üretildi. "
            "Stitching / persistent state / diagnostic MP4 bu seed gelmeden çalıştırılmadı."
        ),
    }
    _write(OUT / "final_manifest.json", final)
    (OUT / "final_report.md").write_text(
        "\n".join(
            [
                "# Target Tracking R1 — Seed Required",
                "",
                f"- final_status: `{status_name}`",
                f"- HEAD: `{head}`",
                f"- frame: `{best_fi}` · t=`{best_fi/fps:.2f}`s",
                f"- visible eligible boxes: `{len(visible)}`",
                f"- contact sheet: `{png}`",
                "",
                "## Seed audit",
                "",
                f"- decision_log: `{seed_audit['decision_log_bytes']}` bytes",
                f"- timeline_approval_log: `{seed_audit['timeline_approval_log_bytes']}` bytes",
                f"- gallery_approval_log: `{seed_audit['gallery_approval_log_bytes']}` bytes",
                "",
                "## Kullanıcı aksiyonu",
                "",
                "Contact sheet PNG’deki etiketlerden hedef oyuncunun "
                "`raw_track_id` değerini bildiriniz.",
                "",
                "Örnek görünür ID’ler (ilk 30): "
                + ", ".join(table["visible_raw_track_ids"][:30]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    # Signal reuse audit stub (no model runs)
    _write(
        OUT / "existing_signal_reuse_audit.json",
        {
            "created_at": _utc(),
            "run_id": RUN_ID,
            "signals": [
                {
                    "name": "crop_quality",
                    "importable": True,
                    "module": "football_analytics.reid.quality",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                    "reason": "no crop JPEGs for short-video run",
                },
                {
                    "name": "contamination",
                    "importable": True,
                    "module": "football_analytics.reid.quality",
                    "artifact_for_run": False,
                    "usable_input": "partial_from_cooccurrence",
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                    "reason": "deferred until seed provided; no precomputed contamination",
                },
                {
                    "name": "kit_team_descriptor",
                    "importable": True,
                    "module": "football_analytics.reid.kit",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                },
                {
                    "name": "track_purity",
                    "importable": True,
                    "module": "football_analytics.reid.purity",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                },
                {
                    "name": "manual_segmentation_derived_view",
                    "importable": True,
                    "module": "football_analytics.reid.segments",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                },
                {
                    "name": "segmented_reid",
                    "importable": True,
                    "module": "football_analytics.reid.segment_regression",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                },
                {
                    "name": "sportsreid_safe_adapter",
                    "importable": True,
                    "module": "football_analytics.reid.safe_checkpoint",
                    "artifact_for_run": False,
                    "usable_input": False,
                    "included_in_stitch_score": False,
                    "status": "UNAVAILABLE",
                    "reason": "adapter registered elsewhere; no short-video embeddings",
                },
                {
                    "name": "exact_frame_conflict",
                    "importable": True,
                    "module": "football_analytics.reid.candidates",
                    "artifact_for_run": True,
                    "usable_input": True,
                    "included_in_stitch_score": "planned_after_seed",
                    "status": "AVAILABLE_AFTER_SEED",
                    "reason": "recomputable from tracks.jsonl / dense timeline",
                },
                {
                    "name": "temporal_spatial_pair_evidence",
                    "importable": True,
                    "module": "football_analytics.reid.candidates / short_video.one_click",
                    "artifact_for_run": True,
                    "usable_input": True,
                    "included_in_stitch_score": "planned_after_seed",
                    "status": "AVAILABLE_AFTER_SEED",
                },
                {
                    "name": "candidate_ranking",
                    "importable": True,
                    "module": "football_analytics.reid.short_video.one_click",
                    "artifact_for_run": True,
                    "usable_input": True,
                    "included_in_stitch_score": "planned_after_seed",
                    "status": "AVAILABLE_AFTER_SEED",
                    "reason": "geometry helper only; appearance rank UNAVAILABLE",
                },
            ],
        },
    )

    print(status_name)
    print(f"contact_sheet={png}")
    print(f"frame={best_fi} visible={len(visible)}")
    print("visible_ids=" + ",".join(table["visible_raw_track_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
