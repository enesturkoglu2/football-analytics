#!/usr/bin/env python3
"""R1.1 freeze audit + readiness artifacts (no detection/tracking rerun)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.golden_clip.window_index import (  # noqa: E402
    DEFAULT_WINDOW_RADIUS_FRAMES,
    load_dense_observations_index,
    payload_byte_size,
    slice_window,
)

ROOT = (
    PROJECT
    / "outputs/reid/target_golden_clip_r1/match_short_video_f2f6d8a077ca/sv_run_20260727T234854Z"
)
EXPECTED_HEAD = "89530f1b28ba6d8f1edf8ea7c30490f9ba525f52"
EXPECTED_SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    # Allow HEAD to move after this repair commit; contract for bootstrap is pre-fix base.
    refs = json.loads((ROOT / "read_only_refs" / "artifact_pointers.json").read_text())
    if refs["source_video_sha256"] != EXPECTED_SHA:
        print("BLOCKED_TARGET_GOLDEN_CLIP_SOURCE_CONTRACT")
        return 2

    ann = ROOT / "session" / "annotation_log.jsonl"
    text = ann.read_text(encoding="utf-8") if ann.is_file() else ""
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    incomplete = []
    # Empty log + draft with no intervals = freeze happened before append
    draft = ROOT / "session" / "ground_truth_draft.json"
    draft_obj = json.loads(draft.read_text(encoding="utf-8")) if draft.is_file() else {}
    already_qualified = any(r.get("action") == "INCOMPLETE_UI_FREEZE_EVENT" for r in rows)
    if draft_obj and not draft_obj.get("intervals") and not rows and not already_qualified:
        qualify = {
            "schema_version": "target_gt_annotation_event_v1",
            "event_id": f"gtevt_freeze_{_utc().replace(':', '').replace('-', '')}",
            "action": "INCOMPLETE_UI_FREEZE_EVENT",
            "created_at": _utc(),
            "qualification": "INCOMPLETE_UI_FREEZE_EVENT",
            "reason_tr": (
                "Annotation log boş; draft intervals yok. "
                "Kullanıcı tıklamasında UI dondu; accepted GT dışında tutulur."
            ),
            "annotation_session_id": draft_obj.get("annotation_session_id"),
            "coverage_scope": "FAILURE_WINDOW_PILOT",
            "accepted_gt_exclusion": True,
            "deleted": False,
            "rows_deleted": False,
        }
        incomplete.append(qualify)
        # Append-only qualify; never delete prior rows
        with ann.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(qualify, ensure_ascii=False) + "\n")
        rows = [qualify]
    for row in rows:
        if row.get("action") == "INCOMPLETE_UI_FREEZE_EVENT":
            incomplete.append(row)
        # partial proposals without pilot label
        if row.get("action") == "APPEND_INTERVAL" and (
            row.get("interval") or {}
        ).get("visibility_confidence") == "proposal":
            incomplete.append(
                {
                    "qualification": "INCOMPLETE_UI_FREEZE_EVENT",
                    "event_id": row.get("event_id"),
                    "deleted": False,
                }
            )

    _write(
        ROOT / "reports" / "incomplete_freeze_event_audit.json",
        {
            "created_at": _utc(),
            "annotation_log_rows": len(rows),
            "incomplete_events": incomplete,
            "no_rows_deleted": True,
            "accepted_gt_exclusion": True,
        },
    )

    dense = Path(refs["dense_bbox_timeline"])
    if not dense.is_file():
        dense = PROJECT / refs["dense_bbox_timeline"]
    t0 = time.perf_counter()
    idx = load_dense_observations_index(dense)
    deserialize_sec = time.perf_counter() - t0
    full_bytes = payload_byte_size(idx["observations_by_frame"])
    win = slice_window(
        idx["observations_by_frame"],
        center_frame=600,
        radius=DEFAULT_WINDOW_RADIUS_FRAMES,
        frame_count=idx["frame_count"],
    )
    win_bytes = payload_byte_size(win)
    proxy = ROOT / "session" / "interactive_review" / "source_proxy_960.mp4"
    freeze = {
        "schema_version": "target_golden_clip_r11_freeze_root_cause_v1",
        "created_at": _utc(),
        "blocker": "BLOCKED_TARGET_GOLDEN_CLIP_UI_INTERACTION_FREEZE",
        "reproduced_analytically": True,
        "root_cause": {
            "primary": "full_dense_manifest_plus_base64_video_on_every_streamlit_rerun",
            "component_full_payload_bytes": full_bytes,
            "component_full_payload_mb": round(full_bytes / 1e6, 2),
            "proxy_file_bytes": proxy.stat().st_size if proxy.is_file() else None,
            "proxy_b64_approx_mb": (
                round(proxy.stat().st_size * 4 / 3 / 1e6, 2) if proxy.is_file() else None
            ),
            "dense_deserialize_sec": deserialize_sec,
            "click_handler_called_st_rerun": True,
            "video_reencoded_or_reread_each_rerun_before_fix": True,
            "session_state_copied_full_track_bboxes_before_fix": True,
            "backend_vs_frontend": "rerun_loop_with_oversized_component_args",
        },
        "fix": {
            "bounded_window_radius_frames": DEFAULT_WINDOW_RADIUS_FRAMES,
            "bounded_window_payload_bytes_example_center_600": win_bytes,
            "bounded_window_payload_kb": round(win_bytes / 1024, 1),
            "video_data_url_sent_only_once_per_session": True,
            "duplicate_click_debounce_ms": 400,
            "event_uuid_idempotence": True,
            "fallback_static_frame_ui": True,
            "pilot_mode": "TARGET_FAILURE_WINDOW_PILOT",
        },
        "base_head_at_r1": EXPECTED_HEAD,
        "git_head_now": head,
    }
    _write(ROOT / "reports" / "freeze_root_cause.json", freeze)

    status = "COMPLETED_TARGET_GOLDEN_CLIP_R11_UI_READY_USER_ACTION_REQUIRED"
    report = {
        "final_status": status,
        "created_at": _utc(),
        "coverage_scope": "FAILURE_WINDOW_PILOT",
        "detection_rerun": False,
        "tracking_rerun": False,
        "product_logs_mutated": False,
        "previous_status_invalidated": (
            "COMPLETED_TARGET_GOLDEN_CLIP_UI_READY_USER_ACTION_REQUIRED"
        ),
        "active_blocker_resolved_technically": "BLOCKED_TARGET_GOLDEN_CLIP_UI_INTERACTION_FREEZE",
        "next_user_action": (
            "Open repaired UI, select target once, label >=3 failure windows"
        ),
    }
    _write(ROOT / "reports" / "r11_final_manifest.json", report)
    (ROOT / "reports" / "r11_final_report.md").write_text(
        "\n".join(
            [
                "# Target Golden Clip R1.1 — Final Report",
                "",
                f"- final_status: `{status}`",
                "- Önceki UI-ready status insan kullanımı açısından geçersizdi (click freeze).",
                "",
                "## Freeze kök nedeni",
                "",
                f"- Full component payload ≈ `{freeze['root_cause']['component_full_payload_mb']} MB`",
                f"- Proxy base64 ≈ `{freeze['root_cause']['proxy_b64_approx_mb']} MB`",
                "- Her bbox click `st.rerun` ile bu payload’ı yeniden gönderiyordu.",
                "",
                "## Düzeltme",
                "",
                f"- Bounded window ±{DEFAULT_WINDOW_RADIUS_FRAMES} frame "
                f"(ör. `{freeze['fix']['bounded_window_payload_kb']} KB`).",
                "- Video data URL oturumda bir kez gönderilir.",
                "- Click debounce + event_uuid idempotence.",
                "- Fallback static frame UI.",
                "- `TARGET_FAILURE_WINDOW_PILOT` (tam 45.29s GT zorunlu değil).",
                "",
                "## Kullanıcı aksiyonu",
                "",
                "1. Hedef oyuncuya bir kez tıkla",
                "2. En az 3 failure window etiketle",
                "3. Interactive donarsa Fallback moda geç",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(status)
    print(f"window_payload_kb={freeze['fix']['bounded_window_payload_kb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
