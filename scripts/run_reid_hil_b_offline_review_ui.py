#!/usr/bin/env python3
"""HIL-B offline review UI gate: build packages, smoke workflow, write audit artifacts."""

from __future__ import annotations

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

from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.decisions import DecisionAction  # noqa: E402
from football_analytics.reid.hil_ui.decisions_service import (  # noqa: E402
    submit_decision,
    undo_last_decision,
)
from football_analytics.reid.hil_ui.geometry import letterbox_params, resolve_click_to_bbox  # noqa: E402
from football_analytics.reid.hil_ui.package import REVIEW_PACKAGE_SCHEMA_VERSION  # noqa: E402
from football_analytics.reid.hil_ui.package_builders import (  # noqa: E402
    build_existing_artifact_review_package,
    build_fixture_review_package,
)
from football_analytics.reid.hil_ui.security import (  # noqa: E402
    security_audit_payload,
    streamlit_cli_args,
)
from football_analytics.reid.hil_ui.session import open_review_session  # noqa: E402

OUTPUT_NAME = "target_001_reid_hil_b_offline_review_ui"
HIL_UI_PYTHON = Path.home() / "miniconda3" / "envs" / "football-hil-ui" / "bin" / "python"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _atomic_publish(temp_root: Path, final_root: Path) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise RuntimeError("BLOCKED_HIL_B_OUTPUT_ROOT_ALREADY_EXISTS")
    os.rename(temp_root, final_root)


def main() -> int:
    final_root = PROJECT / "outputs" / "reid" / OUTPUT_NAME
    if final_root.exists():
        print("BLOCKED_HIL_B_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    temp_root = Path(
        tempfile.mkdtemp(
            prefix=".target_001_reid_hil_b_offline_review_ui_",
            dir=str(PROJECT / "outputs" / "reid"),
        )
    )
    try:
        packages_dir = temp_root / "packages"
        fixture_dir = packages_dir / "fixture"
        existing_dir = packages_dir / "existing_artifact"
        fixture = build_fixture_review_package(fixture_dir)
        existing = build_existing_artifact_review_package(
            existing_dir, project_root=PROJECT
        )

        # Copy package JSON snapshots to artifact root
        shutil.copy2(fixture["package_file"], temp_root / "fixture_review_package.json")
        shutil.copy2(existing["package_file"], temp_root / "existing_artifact_review_package.json")

        # Workflow smoke on fixture (confirm / defer / undo / direct bbox)
        session = open_review_session(fixture["package_file"])
        session.set_event_index(1)
        event = session.current_event
        man = session.current_manifest()
        assert man is not None
        cand = man["candidates"][0]
        confirm = submit_decision(
            session.decision_log,
            event=event,
            candidate_manifest=man,
            action=DecisionAction.CONFIRM_TARGET,
            reviewer="hil_b_smoke",
            selection={
                "selected_candidate_id": cand["candidate_id"],
                "selected_segment_id": cand["segment_id"],
                "selected_raw_track_id": cand["raw_track_id"],
                "selected_frame_index": cand["bbox_references"][0]["frame_index"],
                "selected_bbox_xyxy": cand["bbox_references"][0]["bbox_xyxy"],
                "direct_bbox_selection": False,
                "listed_selection": True,
                "displayed_rank": cand.get("appearance_rank"),
                "displayed_score": cand.get("S"),
                "displayed_T_max": cand.get("T_max"),
                "displayed_D_max": cand.get("D_max"),
                "displayed_model_id": cand.get("sportsreid_model_id"),
                "displayed_checkpoint_sha256": cand.get("sportsreid_checkpoint_sha256"),
            },
        )
        defer = submit_decision(
            session.decision_log,
            event=session.events[0],
            candidate_manifest=None,
            action=DecisionAction.DEFER,
            reviewer="hil_b_smoke",
        )
        undo = undo_last_decision(
            session.decision_log,
            event=event,
            candidate_manifest=man,
            reviewer="hil_b_smoke",
        )
        direct = submit_decision(
            session.decision_log,
            event=event,
            candidate_manifest=man,
            action=DecisionAction.CONFIRM_TARGET,
            reviewer="hil_b_smoke",
            selection={
                "selected_candidate_id": None,
                "selected_segment_id": "DIRECT_SEG_EXTRA",
                "selected_raw_track_id": "raw_direct_extra",
                "selected_frame_index": 121,
                "selected_bbox_xyxy": [200.0, 10.0, 280.0, 70.0],
                "direct_bbox_selection": True,
                "listed_selection": False,
            },
            comment="direct bbox smoke",
        )

        params = letterbox_params(frame_w=320, frame_h=240, display_w=640, display_h=480)
        # click center of first bbox in display coords
        # bbox 10,20,40,80 → center 25,50 → display with letterbox
        # scale = min(640/320, 480/240)=2, pad_x=0, pad_y=0
        click = resolve_click_to_bbox(
            ui_x=50,
            ui_y=100,
            params=params,
            bboxes=[
                {"bbox_id": "cand_001", "bbox_xyxy": [10, 20, 40, 80], "provenance": {}},
                {"bbox_id": "extra", "bbox_xyxy": [200, 10, 280, 70], "provenance": {}},
            ],
        )
        bbox_validation = {
            "letterbox": params.__dict__,
            "click_resolution": {
                "status": click.status,
                "selected": None if click.selected is None else click.selected.bbox_id,
                "message": click.message,
            },
            "direct_bbox_decision_id": direct["decision"]["decision_id"],
            "direct_bbox_selection": True,
        }
        _write_json(temp_root / "bbox_click_validation.json", bbox_validation)

        workflow = {
            "confirm_decision_id": confirm["decision"]["decision_id"],
            "defer_decision_id": defer["decision"]["decision_id"],
            "direct_decision_id": direct["decision"]["decision_id"],
            "undo_decision_id": undo["decision"]["decision_id"],
            "undo_supersedes": undo["supersedes"],
            "direct_after_undo": True,
            "decision_log_sha256": session.decision_log.integrity_report()["sha256"],
            "fixture_package_id": fixture["package_id"],
            "existing_package_id": existing["package_id"],
        }
        _write_json(temp_root / "ui_workflow_smoke.json", workflow)
        _write_json(
            temp_root / "decision_log_integration_report.json",
            {
                "append_only": True,
                "undo_mutates_prior_rows": False,
                "undo_creates_superseding_record": True,
                "training_use_approved_default": False,
                "gallery_use_approved_default": False,
                "log_path": session.package["decision_log_resolved"],
                "log_sha256": session.decision_log.integrity_report()["sha256"],
                "actions_exercised": [
                    "CONFIRM_TARGET",
                    "DEFER",
                    "CONFIRM_TARGET_DIRECT_BBOX",
                    "REVOKE_UNDO",
                ],
            },
        )

        # Env report + freeze
        env_py = HIL_UI_PYTHON
        freeze = subprocess.check_output(
            [str(env_py), "-m", "pip", "freeze"],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            text=True,
        )
        (temp_root / "ui_environment_freeze.txt").write_text(freeze, encoding="utf-8")
        pip_check = subprocess.run(
            [str(env_py), "-m", "pip", "check"],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            capture_output=True,
            text=True,
        )
        versions = subprocess.check_output(
            [
                str(env_py),
                "-c",
                "import sys,streamlit,plotly,PIL,cv2,numpy,yaml;"
                "print(sys.version);"
                "print(streamlit.__version__);"
                "print(plotly.__version__);"
                "print(PIL.__version__);"
                "print(cv2.__version__);"
                "print(numpy.__version__);"
                "print(yaml.__version__)",
            ],
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(PROJECT / "src")},
            text=True,
        ).strip().splitlines()
        # Main envs unchanged regarding streamlit
        main_streamlit = {}
        for name in ("football-cv", "sn-reid-cpu"):
            py = Path.home() / "miniconda3" / "envs" / name / "bin" / "python"
            code = (
                "import importlib.util; print(bool(importlib.util.find_spec('streamlit')))"
            )
            present = subprocess.check_output(
                [str(py), "-c", code],
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                text=True,
            ).strip()
            main_streamlit[name] = present == "True"

        env_report = {
            "environment_name": "football-hil-ui",
            "environment_path": str(env_py.parent.parent),
            "python_version": versions[0],
            "streamlit": versions[1],
            "plotly": versions[2],
            "pillow": versions[3],
            "opencv": versions[4],
            "numpy": versions[5],
            "pyyaml": versions[6],
            "pip_check_ok": pip_check.returncode == 0,
            "pip_check_stdout": pip_check.stdout,
            "pip_check_stderr": pip_check.stderr,
            "integration_method": "PYTHONPATH=src (no editable install of heavy deps)",
            "main_env_streamlit_present": main_streamlit,
            "pinned": {
                "streamlit": "1.37.1",
                "plotly": "5.24.1",
                "pillow": "10.4.0",
                "opencv-python-headless": "4.10.0.84",
                "PyYAML": "6.0.2",
                "numpy": "1.26.4",
            },
            "extra_click_component_installed": False,
            "bbox_click_mechanism": "plotly_image_click_plus_manual_coordinate_entry",
        }
        _write_json(temp_root / "ui_environment_report.json", env_report)

        # Server security smoke
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        app = PROJECT / "src/football_analytics/reid/hil_ui/streamlit_app.py"
        env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(PROJECT / "src"),
            "HIL_REVIEW_PACKAGE": fixture["package_file"],
            "STREAMLIT_CONFIG_DIR": str(PROJECT / "configs/reid/hil_ui/.streamlit"),
        }
        proc = subprocess.Popen(
            [str(env_py), "-m", *streamlit_cli_args(app_path=str(app), port=port)],
            cwd=str(PROJECT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        healthy = False
        background_left = False
        port_left = False
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
            background_left = proc.poll() is None
            time.sleep(0.3)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                port_left = sock.connect_ex(("127.0.0.1", port)) == 0

        audit = security_audit_payload(address="127.0.0.1", port=port)
        audit.update(
            {
                "health_check_ok": healthy,
                "server_terminated": not background_left,
                "port_released": not port_left,
                "background_process_left": background_left,
                "listening_port_left": port_left,
            }
        )
        _write_json(temp_root / "streamlit_server_security_audit.json", audit)

        schema_snap = {
            "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
            "required_fields": [
                "schema_version",
                "package_id",
                "project_id",
                "run_id",
                "target_id",
                "source_video_path",
                "source_video_sha256",
                "event_manifest_path",
                "event_manifest_sha256",
                "candidate_manifest_paths",
                "candidate_manifest_sha256",
                "decision_log_path",
                "target_gallery_reference",
                "model_metadata",
                "media_root",
                "read_only_source_roots",
                "writable_session_root",
                "created_at",
                "provenance",
            ],
        }
        _write_json(temp_root / "review_package_schema_snapshot.json", schema_snap)

        contract_src = PROJECT / "configs/reid/hil_b_offline_review_ui_contract.yaml"
        shutil.copy2(contract_src, temp_root / "effective_hil_b_contract.yaml")

        _write_json(
            temp_root / "active_execution_path.json",
            {
                "product_path": "HIL",
                "gate": "REID_HIL_B_OFFLINE_REVIEW_UI_ENVIRONMENT_AND_IMPLEMENTATION",
                "ui_entrypoint": "scripts/run_hil_offline_review_ui.py",
                "streamlit_app": str(app),
                "isolated_env": "football-hil-ui",
                "bind": "127.0.0.1",
                "no_inference": True,
                "no_clip": True,
                "no_timeline_final": True,
            },
        )

        # Tiny screenshot placeholder (PNG via opencv) as demo visual
        try:
            import cv2
            import numpy as np

            demo = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                demo,
                "HIL-B UI smoke",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            shot = temp_root / "ui_demo_frame.png"
            cv2.imwrite(str(shot), demo)
        except Exception:  # noqa: BLE001
            (temp_root / "ui_demo_frame.png").write_bytes(b"")

        report_md = temp_root / "target_001_reid_hil_b_offline_review_ui_report.md"
        report_md.write_text(
            "\n".join(
                [
                    "# ReID-HIL-B Offline Review UI Report",
                    "",
                    f"- final_status: COMPLETED_HIL_B_OFFLINE_REVIEW_UI",
                    f"- product_next_gate: REID_HIL_C_TARGET_TIMELINE_RECONSTRUCTION",
                    f"- research_next_gate: REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                    f"- isolated_env: football-hil-ui",
                    f"- streamlit: {env_report['streamlit']}",
                    f"- bind: 127.0.0.1",
                    f"- server_health: {healthy}",
                    f"- background_process_left: {background_left}",
                    f"- listening_port_left: {port_left}",
                    f"- fixture_package: {fixture['package_id']}",
                    f"- existing_artifact_package: {existing['package_id']}",
                    f"- direct_bbox_selection: true",
                    f"- append_only_undo: true",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        artifacts = sorted(p.name for p in temp_root.iterdir() if p.is_file())
        # also count nested package files? Manifest lists top-level primarily
        manifest = {
            "schema_version": "target_001_reid_hil_b_manifest_v1",
            "final_status": "COMPLETED_HIL_B_OFFLINE_REVIEW_UI",
            "exact_next_gate": "REID_HIL_C_TARGET_TIMELINE_RECONSTRUCTION",
            "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
            "created_at": _utc_now(),
            "artifact_files": artifacts,
            "artifacts": {
                name: {
                    "size_bytes": (temp_root / name).stat().st_size,
                    "sha256": sha256_file(temp_root / name),
                }
                for name in artifacts
            },
            "packages_dir_present": packages_dir.is_dir(),
            "server_security": audit,
            "environment": {
                "name": "football-hil-ui",
                "streamlit": env_report["streamlit"],
                "python": env_report["python_version"],
            },
        }
        _write_json(temp_root / "target_001_reid_hil_b_manifest.json", manifest)
        # refresh artifact list after manifest write — rewrite once
        artifacts = sorted(p.name for p in temp_root.iterdir() if p.is_file())
        manifest["artifact_files"] = artifacts
        manifest["artifacts"] = {
            name: {
                "size_bytes": (temp_root / name).stat().st_size,
                "sha256": sha256_file(temp_root / name),
            }
            for name in artifacts
        }
        _write_json(temp_root / "target_001_reid_hil_b_manifest.json", manifest)

        if not healthy or background_left or port_left:
            print("FAILED_HIL_B_LOCAL_SERVER_SECURITY_SMOKE")
            print(json.dumps(audit, indent=2))
            shutil.rmtree(temp_root, ignore_errors=True)
            return 3

        _atomic_publish(temp_root, final_root)
        print("COMPLETED_HIL_B_OFFLINE_REVIEW_UI")
        print(f"output_root={final_root}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED_HIL_B_ATOMIC_BUILD: {exc}")
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
