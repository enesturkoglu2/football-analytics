#!/usr/bin/env python3
"""HIL-C2: product review package, timeline approval audit, timeline population.

No detection/tracking/ReID inference. Does not mutate acceptance or legacy product logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.timeline.approvals import ApprovalLog  # noqa: E402
from football_analytics.reid.hil.timeline.reconstruct import (  # noqa: E402
    dump_intervals_jsonl,
    reconstruct_twice_for_determinism,
)
from football_analytics.reid.hil.timeline.segments import load_segment_index  # noqa: E402
from football_analytics.reid.hil.timeline.sources import (  # noqa: E402
    ACCEPTANCE_LOG_SHA,
    audit_decision_sources,
    write_json,
)
from football_analytics.reid.hil_c2.product_package import (  # noqa: E402
    PRODUCT_PACKAGE_ID,
    PRODUCT_RUN_ID,
    VIDEO_ID,
    VIDEO_SHA,
    build_product_external_review_package,
)
from football_analytics.reid.hil_c2.qualify import qualify_product_session  # noqa: E402
from football_analytics.reid.hil_c2.source_audit import audit_product_video_sources  # noqa: E402

OUTPUT_NAME = "target_001_reid_hil_c2_product_timeline_population"
PACKAGE_NAME = "target_001_reid_hil_c2_product_review_package"
LEGACY_PRODUCT_LOG_SHA = (
    "9ace4d2177bbf8993ec2a0093bd694c3bff6f07e2a606c87c9e318110b886623"
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_ui(package_json: Path, *, port: int | None = None) -> dict[str, Any]:
    py = Path.home() / "miniconda3/envs/football-hil-ui/bin/python"
    launch = PROJECT / "scripts/run_hil_offline_review_ui.py"
    use_port = port or _free_port()
    cmd = [
        str(py),
        str(launch),
        "--review-package",
        str(package_json),
        "--port",
        str(use_port),
        "--address",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "pid": proc.pid,
        "url": f"http://127.0.0.1:{use_port}",
        "port": use_port,
        "command": cmd,
        "stop": f"kill {proc.pid}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HIL-C2 product timeline population")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--skip-ui-launch", action="store_true")
    parser.add_argument("--ui-port", type=int, default=0)
    args = parser.parse_args()
    root = args.project_root.resolve()

    final = root / "outputs" / "reid" / OUTPUT_NAME
    if final.exists():
        print("BLOCKED_HIL_C2_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    # Immutable legacy logs
    legacy_product = (
        root
        / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/existing_artifact/session/decision_log.jsonl"
    )
    acceptance = Path(
        "/tmp/hil_b_r2_acceptance_existing/session_acceptance_r2/decision_log.jsonl"
    )
    legacy_sha = sha256_file(legacy_product)
    if legacy_sha != LEGACY_PRODUCT_LOG_SHA:
        print("BLOCKED_HIL_C2_SOURCE_CONTRACT_MISMATCH")
        print(f"legacy product log SHA mismatch: {legacy_sha}")
        return 2
    acc_sha = sha256_file(acceptance) if acceptance.is_file() else None
    if acceptance.is_file() and acc_sha != ACCEPTANCE_LOG_SHA:
        print("BLOCKED_HIL_C2_SOURCE_CONTRACT_MISMATCH")
        print(f"acceptance log SHA mismatch: {acc_sha}")
        return 2
    legacy_bytes = legacy_product.read_bytes()
    acc_bytes = acceptance.read_bytes() if acceptance.is_file() else None

    source_audit = audit_product_video_sources(root)
    if source_audit.get("blocked_status"):
        # Still allow writing a minimal blocked report under temp then move? Spec says stop.
        print(source_audit["blocked_status"])
        print(source_audit.get("exact_next_gate_if_blocked"))
        return 3

    package_root = root / "outputs" / "reid" / PACKAGE_NAME
    package_root.mkdir(parents=True, exist_ok=True)
    pkg = build_product_external_review_package(package_root, project_root=root)

    ui_info = None
    if not args.skip_ui_launch:
        try:
            ui_info = _launch_ui(Path(pkg["package_file"]), port=args.ui_port or None)
        except OSError as exc:
            ui_info = {"error": str(exc), "launched": False}

    excluded_audits = audit_decision_sources(
        [
            {
                "path": str(legacy_product),
                "review_package_mode": "existing_artifact_product_path",
            },
            *(
                [
                    {
                        "path": str(acceptance),
                        "review_package_mode": "acceptance_isolated",
                    }
                ]
                if acceptance.is_file()
                else []
            ),
        ]
    )

    qual = qualify_product_session(
        decision_log_path=pkg["decision_log_path"],
        approval_log_path=pkg["approval_log_path"],
        review_package_mode="product",
        excluded_source_audits=[
            {
                "note": "legacy/acceptance excluded from product timeline",
                "counts": excluded_audits["counts"],
            }
        ],
    )

    approved_ids = set(qual["approved_decision_ids"])
    segment_index = load_segment_index(pkg["segment_inventory_path"])

    recon = reconstruct_twice_for_determinism(
        project_id="football-analytics",
        run_id=PRODUCT_RUN_ID,
        target_id="target_001",
        video_id=VIDEO_ID,
        video_path=str(root / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"),
        video_sha256=VIDEO_SHA,
        frame_rate=30.0,
        total_video_frames=784,
        total_video_duration_seconds=784 / 30.0,
        decision_sources=[
            {
                "path": pkg["decision_log_path"],
                "review_package_mode": "product",
                "require_timeline_approval": True,
            }
        ],
        segment_index=segment_index,
        source_segment_manifest_path=pkg["segment_inventory_path"],
        source_segment_manifest_sha256=pkg["segment_inventory_sha256"],
        generated_at=_utc(),
        approved_decision_ids=approved_ids,
        require_timeline_approval=True,
        timeline_id=f"timeline_target_001_{PRODUCT_RUN_ID}",
    )

    if legacy_product.read_bytes() != legacy_bytes:
        print("BLOCKED_HIL_C2_DECISION_PROVENANCE")
        print("legacy product log mutated")
        return 4
    if acceptance.is_file() and acceptance.read_bytes() != acc_bytes:
        print("BLOCKED_HIL_C2_DECISION_PROVENANCE")
        print("acceptance log mutated")
        return 4

    eligible_count = len(qual["timeline_eligible_decisions"])
    if eligible_count > 0:
        final_status = "COMPLETED_HIL_C2_PRODUCT_TIMELINE_POPULATED"
        next_gate = "REID_HIL_D_VERIFIED_TARGET_TRAJECTORY_AND_METRIC_EXPORT"
    else:
        final_status = "COMPLETED_HIL_C2_NO_USER_APPROVED_DECISIONS"
        next_gate = "REID_HIL_D_VERIFIED_TARGET_TRAJECTORY_AND_METRIC_EXPORT"

    tl = recon["timeline"]
    cov = tl["coverage_summary"]

    temp = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}_", dir=str(root / "outputs" / "reid"))
    )
    try:
        contract = {
            "gate": "REID_HIL_C2_PRODUCT_REVIEW_PACKAGE_AND_TIMELINE_APPROVAL",
            "final_status": final_status,
            "product_next_gate": next_gate,
            "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
            "product_package_id": PRODUCT_PACKAGE_ID,
            "selected_video_id": source_audit["selected_source_video_id"],
            "require_timeline_approval": True,
            "no_detection_tracking_rerun": True,
            "no_reid_inference": True,
            "legacy_product_log_sha256": legacy_sha,
            "acceptance_log_sha256": acc_sha,
        }
        (temp / "effective_hil_c2_contract.yaml").write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in contract.items()) + "\n",
            encoding="utf-8",
        )
        write_json(temp / "product_video_source_audit.json", source_audit)
        write_json(
            temp / "product_review_package.json",
            {
                "package_file": pkg["package_file"],
                "package_id": PRODUCT_PACKAGE_ID,
                "run_id": PRODUCT_RUN_ID,
                "event_count": pkg["event_count"],
                "candidate_manifest_count": pkg["candidate_manifest_count"],
                "decision_log_path": pkg["decision_log_path"],
                "approval_log_path": pkg["approval_log_path"],
                "segment_inventory_path": pkg["segment_inventory_path"],
                "segment_inventory_sha256": pkg["segment_inventory_sha256"],
                "recovery_readiness": pkg["recovery_readiness"],
                "package_mode": "product",
            },
        )
        write_json(
            temp / "product_decision_log_audit.json",
            {
                "product": qual["product_audit"],
                "excluded_legacy_and_acceptance": excluded_audits,
            },
        )
        # Copy approval log snapshot (append-only source remains in package session)
        shutil.copy2(pkg["approval_log_path"], temp / "product_timeline_approval_log.jsonl")
        write_json(temp / "product_decision_qualification.json", qual)
        write_json(
            temp / "segment_provenance_validation.json",
            tl["provenance"].get("segment_provenance_validation") or [],
        )
        write_json(temp / "target_001_product_timeline.json", tl)
        dump_intervals_jsonl(temp / "target_001_product_timeline_intervals.jsonl", tl["intervals"])
        dump_intervals_jsonl(
            temp / "target_001_product_unresolved_intervals.jsonl",
            tl["unresolved_intervals"],
        )
        write_json(temp / "product_timeline_coverage.json", cov)
        write_json(
            temp / "timeline_determinism.json",
            {
                "deterministic": recon["deterministic"],
                "sha_a": recon["sha_a"],
                "sha_b": recon["sha_b"],
                "max_numeric_diff": recon["max_numeric_diff"],
                "product_decision_log_sha256": qual["decision_log_sha256"],
                "legacy_product_log_sha256": legacy_sha,
                "acceptance_log_sha256": acc_sha,
                "logs_immutable": True,
            },
        )
        write_json(
            temp / "active_execution_path.json",
            {
                "detection_tracking_rerun": False,
                "reid_inference": False,
                "clip_download": False,
                "game_state": False,
                "physical_metrics": False,
                "ui_launch": ui_info,
                "package_root": str(package_root),
            },
        )
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        # report + manifest written below; recount after
        report_lines = [
            "# HIL-C2 Product Timeline Population Report",
            "",
            f"- final_status: `{final_status}`",
            f"- product_next_gate: `{next_gate}`",
            "- research_next_gate: `REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW`",
            f"- selected_video: `{source_audit['selected_source_video_id']}`",
            f"- product_package_id: `{PRODUCT_PACKAGE_ID}`",
            f"- approved_decision_count: {eligible_count}",
            f"- interval_count: {len(tl['intervals'])}",
            f"- unresolved_count: {len(tl['unresolved_intervals'])}",
            f"- conflicts: {len(tl['conflicts'])}",
            f"- verified_coverage_percentage: {cov.get('verified_coverage_percentage')}",
            f"- analysis_eligible_duration_seconds: {cov.get('analysis_eligible_duration_seconds')}",
            f"- unresolved_duration_seconds: {cov.get('unresolved_duration_seconds')}",
            f"- deterministic: {recon['deterministic']}",
            f"- legacy_product_log_sha256: `{legacy_sha}`",
            f"- acceptance_log_sha256: `{acc_sha}`",
            f"- detection/tracking/ReID run: false",
            "",
            "## UI session",
            "",
        ]
        if ui_info and ui_info.get("url"):
            report_lines.extend(
                [
                    f"- URL: {ui_info['url']}",
                    f"- PID: {ui_info['pid']}",
                    f"- stop: `{ui_info['stop']}`",
                    f"- decision_log: `{pkg['decision_log_path']}`",
                    f"- approval_log: `{pkg['approval_log_path']}`",
                    "",
                    "İnsan enrollment + Confirm + Timeline Approval tamamlanmadan",
                    "timeline eligible karar oluşmaz.",
                ]
            )
        else:
            report_lines.append("- UI launch skipped or failed")
        (temp / "target_001_reid_hil_c2_product_timeline_report.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        write_json(
            temp / "target_001_reid_hil_c2_manifest.json",
            {
                "final_status": final_status,
                "product_next_gate": next_gate,
                "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                "artifact_files": artifacts + ["target_001_reid_hil_c2_manifest.json"],
                "artifact_count": len(artifacts) + 1,
                "output_root": str(final),
                "package_root": str(package_root),
                "generated_at": _utc(),
            },
        )
        # refresh artifact list including manifest
        artifacts = sorted(p.name for p in temp.iterdir() if p.is_file())
        man_path = temp / "target_001_reid_hil_c2_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["artifact_files"] = artifacts
        man["artifact_count"] = len(artifacts)
        write_json(man_path, man)

        shutil.move(str(temp), str(final))
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        print("FAILED_HIL_C2_ATOMIC_BUILD")
        raise

    print(final_status)
    print(f"output: {final}")
    if ui_info and ui_info.get("url"):
        print(f"ui: {ui_info['url']} pid={ui_info['pid']}")
        print(f"stop: {ui_info['stop']}")
    print(f"decision_log: {pkg['decision_log_path']}")
    print(f"approval_log: {pkg['approval_log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
