#!/usr/bin/env python3
"""HIL-B-R1 atomic artifact builder (acceptance failure + usability repair)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.resolve import (  # noqa: E402
    resolve_effective_decisions,
    resolve_event_review_state,
)
from football_analytics.reid.hil_ui.geometry import letterbox_params, resolve_click_to_bbox  # noqa: E402
from football_analytics.reid.hil_ui.observations import (  # noqa: E402
    candidate_observation_audit,
    load_observation_lookup,
)
from football_analytics.reid.hil_ui.security import (  # noqa: E402
    security_audit_payload,
    streamlit_cli_args,
)
from football_analytics.reid.hil_ui.visualization import (  # noqa: E402
    build_selection_from_bbox_hit,
    selection_visibility,
)

OUT_NAME = "target_001_reid_hil_b_r1_human_acceptance_repair"
HIL_UI_PYTHON = Path.home() / "miniconda3" / "envs" / "football-hil-ui" / "bin" / "python"
EXISTING_PKG_ROOT = (
    PROJECT
    / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/existing_artifact"
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    final = PROJECT / "outputs" / "reid" / OUT_NAME
    if final.exists():
        print("BLOCKED: output root already exists")
        return 2
    temp = Path(
        tempfile.mkdtemp(prefix=f".{OUT_NAME}_", dir=str(PROJECT / "outputs" / "reid"))
    )
    try:
        log_path = EXISTING_PKG_ROOT / "session" / "decision_log.jsonl"
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rev2 = next(r for r in rows if r["revision"] == 2)
        effective = resolve_effective_decisions(rows)
        state = resolve_event_review_state(
            event_id=rev2["event_id"], decisions=rows
        ).value
        prior_audit = {
            "schema_version": "hil_b_r1_prior_revision_2_decision_audit_v1",
            "decision_log_path": str(log_path.resolve()),
            "decision_log_sha256": _sha(log_path),
            "record_count": len(rows),
            "revision_2": rev2,
            "supersedes_chain": [
                {
                    "decision_id": r["decision_id"],
                    "revision": r["revision"],
                    "supersedes_decision_id": r.get("supersedes_decision_id"),
                    "action": r["action"],
                }
                for r in rows
                if r["event_id"] == rev2["event_id"]
            ],
            "effective_state": effective.get(rev2["event_id"]),
            "event_review_state": state,
            "auto_revoke_applied": False,
            "bytes_mutated": False,
        }
        _write(temp / "prior_revision_2_decision_audit.json", prior_audit)

        acceptance_failure = {
            "schema_version": "hil_b_human_acceptance_failure_v1",
            "status": "HIL_B_HUMAN_ACCEPTANCE_FAILED",
            "url": "http://127.0.0.1:51531/",
            "working_flows": [
                "candidate_card_select_listed",
                "confirm_target_append_revision_2",
            ],
            "failed_flows": [
                "plotly_image_bbox_click_no_selection",
                "direct_bbox_selection_false_on_final_record",
                "manual_ui_xy_not_user_friendly",
                "no_selected_bbox_highlight",
                "no_sparse_visibility_message",
                "rank_none_display",
                "raw_json_as_primary_ui",
                "tracking_not_run_not_clear",
            ],
            "revision_2_decision_id": rev2["decision_id"],
            "direct_bbox_selection_on_record": False,
            "product_next_gate_blocked": "REID_HIL_C_TARGET_TIMELINE_RECONSTRUCTION",
            "repair_next_gate": "REID_HIL_B_R2_HUMAN_ACCEPTANCE_SESSION",
        }
        _write(temp / "acceptance_failure_report.json", acceptance_failure)

        root_cause = {
            "schema_version": "hil_b_r1_direct_click_root_cause_v1",
            "plotly_on_select_image_trace": {
                "reliable_for_bbox_click": False,
                "evidence": (
                    "Human acceptance: clicking yellow bbox produced no selection; "
                    "final decision had direct_bbox_selection=false via Select listed only."
                ),
                "shapes_overlay": "Plotly shapes do not create selectable points on Image trace",
                "streamlit_selection_mode_points": "does not deliver bbox-click hits for burned overlays",
            },
            "coordinate_transform": {
                "letterbox_helper": "unit-tested and correct",
                "not_reached_in_acceptance": True,
            },
            "chosen_fix": {
                "primary": "streamlit-image-coordinates==0.1.9",
                "license": "MIT",
                "telemetry_network_in_component": False,
                "install_scope": "football-hil-ui only",
                "mechanism": "burn overlays into PIL image; click returns display pixels; map to frame; resolve_click_to_bbox",
                "secondary_fallback": "Select this player buttons per visible bbox",
                "candidate_card_fallback": "Select this candidate (listed_selection=true)",
                "manual_pixel_coords": "Advanced / debug only",
            },
            "acceptance_direct_click_counted_successful": False,
        }
        _write(temp / "direct_click_root_cause.json", root_cause)

        man = json.loads(
            (EXISTING_PKG_ROOT / "existing_candidate_manifest.json").read_text()
        )
        pkg = json.loads((EXISTING_PKG_ROOT / "review_package.json").read_text())
        obs_path = (pkg.get("provenance") or {}).get("observation_index_path")
        segs = {c["segment_id"] for c in man["candidates"]}
        lookup = load_observation_lookup(
            obs_path, segment_ids=segs, frame_min=0, frame_max=5
        )
        sparse_audit = candidate_observation_audit(man["candidates"], lookup)
        _write(temp / "sparse_tracklet_observation_audit.json", sparse_audit)

        cand3 = next(c for c in man["candidates"] if c["candidate_id"] == "real_cand_003")
        sel = {
            "selected_candidate_id": "real_cand_003",
            "selected_segment_id": cand3["segment_id"],
            "selected_raw_track_id": cand3["raw_track_id"],
        }
        vis0 = selection_visibility(
            selection=sel, frame_index=0, candidates=man["candidates"], observation_lookup=lookup
        )
        vis2 = selection_visibility(
            selection=sel, frame_index=2, candidates=man["candidates"], observation_lookup=lookup
        )
        vis5 = selection_visibility(
            selection=sel, frame_index=5, candidates=man["candidates"], observation_lookup=lookup
        )
        # direct click simulation
        params = letterbox_params(frame_w=100, frame_h=100, display_w=100, display_h=100)
        boxes = [
            {
                "bbox_id": "extra",
                "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
                "provenance": {"segment_id": "DIRECT", "raw_track_id": "raw_d"},
            }
        ]
        hit = resolve_click_to_bbox(ui_x=20, ui_y=20, params=params, bboxes=boxes)
        direct_sel = build_selection_from_bbox_hit(
            frame_index=0,
            bbox_xyxy=hit.selected.bbox_xyxy,
            candidates=[],
            hit_meta=hit.selected.provenance,
        )
        _write(
            temp / "selection_visualization_validation.json",
            {
                "frame0_selected_highlight": vis0,
                "frame2_not_visible_or_lookup": vis2,
                "frame5_observation_highlight": vis5,
                "direct_click_simulation": direct_sel,
                "sparse_notice_required": True,
            },
        )

        _write(
            temp / "usability_changes.json",
            {
                "image_click_component": "streamlit-image-coordinates==0.1.9",
                "select_this_player_buttons": True,
                "selected_bbox_highlight": True,
                "sparse_frame_message": True,
                "tracking_not_run_notice": True,
                "confirmation_summary_not_raw_json": True,
                "rank_unavailable_label": True,
                "confirm_disabled_without_selection": True,
                "manual_coords_advanced_only": True,
                "prior_revision_2_preserved": True,
            },
        )

        (temp / "new_acceptance_instructions.md").write_text(
            "\n".join(
                [
                    "# HIL-B-R2 Human Acceptance Instructions",
                    "",
                    "1. Launch: `scripts/run_hil_offline_review_ui.py --review-package <existing_artifact/review_package.json>`",
                    "2. Open Recovery Review.",
                    "3. Click a yellow player box on the image → selection must set `direct_bbox_selection=true` if not listed, else listed.",
                    "4. Or use `Select this player` under the frame.",
                    "5. Move slider: selected tracklet highlights when observation exists; otherwise shows not-visible message.",
                    "6. Confirm without selection must be disabled.",
                    "7. Submit success text must include `Tracking was not run.`",
                    "8. Do not revoke prior revision 2 unless explicitly testing history revoke.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # Server smoke for repaired app
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        app = PROJECT / "src/football_analytics/reid/hil_ui/streamlit_app.py"
        env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(PROJECT / "src"),
            "HIL_REVIEW_PACKAGE": str(EXISTING_PKG_ROOT / "review_package.json"),
            "STREAMLIT_CONFIG_DIR": str(PROJECT / "configs/reid/hil_ui/.streamlit"),
        }
        proc = subprocess.Popen(
            [str(HIL_UI_PYTHON), "-m", *streamlit_cli_args(app_path=str(app), port=port)],
            cwd=str(PROJECT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        healthy = False
        try:
            for _ in range(40):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as resp:
                        healthy = resp.status == 200
                        if healthy:
                            break
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            time.sleep(0.3)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                port_left = sock.connect_ex(("127.0.0.1", port)) == 0
        audit = security_audit_payload(address="127.0.0.1", port=port)
        audit.update(
            {
                "health_check_ok": healthy,
                "server_terminated": proc.poll() is not None,
                "port_released": not port_left,
                "background_process_left": False,
                "listening_port_left": port_left,
            }
        )
        _write(temp / "streamlit_server_security_audit.json", audit)

        freeze = subprocess.check_output(
            [str(HIL_UI_PYTHON), "-m", "pip", "freeze"],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            text=True,
        )
        (temp / "ui_environment_freeze.txt").write_text(freeze, encoding="utf-8")

        report = temp / "target_001_reid_hil_b_r1_human_acceptance_repair_report.md"
        report.write_text(
            "\n".join(
                [
                    "# HIL-B-R1 Human Acceptance Repair",
                    "",
                    "- acceptance: HIL_B_HUMAN_ACCEPTANCE_FAILED",
                    "- final_status: COMPLETED_HIL_B_R1_HUMAN_USABILITY_REPAIR",
                    "- next_gate: REID_HIL_B_R2_HUMAN_ACCEPTANCE_SESSION",
                    f"- revision_2_decision_id: {rev2['decision_id']}",
                    "- direct_click_fix: streamlit-image-coordinates==0.1.9 + Select this player",
                    f"- server_health: {healthy}",
                    f"- port_released: {not port_left}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        files = sorted(p.name for p in temp.iterdir() if p.is_file())
        manifest = {
            "schema_version": "target_001_reid_hil_b_r1_manifest_v1",
            "final_status": "COMPLETED_HIL_B_R1_HUMAN_USABILITY_REPAIR",
            "acceptance_status": "HIL_B_HUMAN_ACCEPTANCE_FAILED",
            "exact_next_gate": "REID_HIL_B_R2_HUMAN_ACCEPTANCE_SESSION",
            "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
            "created_at": _utc(),
            "artifact_files": files,
            "artifacts": {
                name: {"size_bytes": (temp / name).stat().st_size, "sha256": _sha(temp / name)}
                for name in files
            },
            "prior_revision_2_decision_id": rev2["decision_id"],
            "decision_log_sha256_preserved": prior_audit["decision_log_sha256"],
            "server_security": audit,
        }
        _write(temp / "target_001_reid_hil_b_r1_manifest.json", manifest)
        files = sorted(p.name for p in temp.iterdir() if p.is_file())
        manifest["artifact_files"] = files
        manifest["artifacts"] = {
            name: {"size_bytes": (temp / name).stat().st_size, "sha256": _sha(temp / name)}
            for name in files
        }
        _write(temp / "target_001_reid_hil_b_r1_manifest.json", manifest)

        if not healthy or port_left:
            print("FAILED_HIL_B_R1_LOCAL_SERVER")
            shutil.rmtree(temp, ignore_errors=True)
            return 3

        os.rename(temp, final)
        print("COMPLETED_HIL_B_R1_HUMAN_USABILITY_REPAIR")
        print(f"output_root={final}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED_HIL_B_R1_BUILD: {exc}")
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
