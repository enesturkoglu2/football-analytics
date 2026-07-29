#!/usr/bin/env python3
"""R1.2 static failure-window readiness artifacts (no detection/tracking rerun)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

ROOT = (
    PROJECT
    / "outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
)
EXPECTED_SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    refs = json.loads((ROOT / "read_only_refs" / "artifact_pointers.json").read_text())
    if refs["source_video_sha256"] != EXPECTED_SHA:
        print("BLOCKED_TARGET_FAILURE_WINDOW_SELECTION")
        return 2

    # Pre-build enrollment only (fast); failure packages after user seed selection.
    from football_analytics.reid.golden_clip.static_packages import (
        build_enrollment_package,
        pick_enrollment_frame,
    )
    from football_analytics.reid.golden_clip.window_index import load_dense_observations_index

    dense = Path(refs["dense_bbox_timeline"])
    if not dense.is_file():
        dense = PROJECT / refs["dense_bbox_timeline"]
    video = Path(refs["source_video"])
    if not video.is_file():
        video = PROJECT / refs["source_video"]

    idx = load_dense_observations_index(dense)
    fi = pick_enrollment_frame(idx["observations_by_frame"], frame_count=idx["frame_count"])
    enroll_dir = ROOT / "session" / "static_failure_windows" / "enrollment"
    enroll = build_enrollment_package(
        source_video=video,
        source_video_sha256=refs["source_video_sha256"],
        observations_by_frame=idx["observations_by_frame"],
        frame_index=fi,
        output_dir=enroll_dir,
    )

    status = "COMPLETED_TARGET_FAILURE_WINDOW_STATIC_UI_READY_USER_ACTION_REQUIRED"
    report = {
        "final_status": status,
        "created_at": _utc(),
        "coverage_scope": "FAILURE_WINDOW_PILOT",
        "ui_default": "static",
        "custom_interactive_component_default": False,
        "enrollment_frame_index": enroll["frame_index"],
        "enrollment_candidate_count": enroll["candidate_count"],
        "enrollment_png": enroll["png"]["path"],
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
        "previous_blocker": "BLOCKED_TARGET_GOLDEN_CLIP_UI_INTERACTION_FREEZE",
        "git_head": head,
        "next_user_action": (
            "Open static UI, click target on enrollment PNG, label >=3 failure windows"
        ),
        "full_metrics": "NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT",
    }
    _write(ROOT / "reports" / "r12_final_manifest.json", report)
    (ROOT / "reports" / "r12_final_report.md").write_text(
        "\n".join(
            [
                "# Target Failure Window Pilot R1.2 — Final Report",
                "",
                f"- final_status: `{status}`",
                "- R1.1 interactive component kullanıcı ortamında tekrar dondu; ana yol olmaktan çıkarıldı.",
                "",
                "## Varsayılan UI",
                "",
                "- Statik enrollment PNG + `streamlit-image-coordinates`",
                "- Native `st.video` event clip",
                "- Statik candidate PNG",
                "- `st.form` ile append-only pilot annotation",
                "- Custom interactive component yalnız Advanced/Experimental altında, kapalı",
                "",
                f"- Enrollment frame: `{enroll['frame_index']}` · candidates: `{enroll['candidate_count']}`",
                "",
                "## Kullanıcı aksiyonu",
                "",
                "1. Hedef oyuncuya statik karede tıkla",
                "2. En az 3 failure window etiketle (tercihen 5)",
                "3. Fragment/occlusion/border için continuation candidate seç",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(status)
    print(f"enrollment_frame={enroll['frame_index']} candidates={enroll['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
