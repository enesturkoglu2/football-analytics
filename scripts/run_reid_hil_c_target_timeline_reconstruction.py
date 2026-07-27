#!/usr/bin/env python3
"""HIL-C verified target timeline reconstruction (no inference, no UI)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from football_analytics.reid.hil.common import sha256_file  # noqa: E402
from football_analytics.reid.hil.log import DecisionLog  # noqa: E402
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
from football_analytics.reid.hil.timeline.schema import GENERATOR_VERSION  # noqa: E402

OUTPUT_NAME = "target_001_reid_hil_c_target_timeline_reconstruction"
VIDEO_SHA = "bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description="HIL-C target timeline reconstruction")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    final = root / "outputs" / "reid" / OUTPUT_NAME
    if final.exists():
        print("BLOCKED_HIL_C_OUTPUT_ROOT_ALREADY_EXISTS")
        return 2

    temp = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}_", dir=str(root / "outputs" / "reid"))
    )
    try:
        product_log = (
            root
            / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/existing_artifact/session/decision_log.jsonl"
        )
        acceptance_log = Path(
            "/tmp/hil_b_r2_acceptance_existing/session_acceptance_r2/decision_log.jsonl"
        )
        fixture_log = (
            root
            / "outputs/reid/target_001_reid_hil_a_event_decision_schema/demo_decision_log.jsonl"
        )
        fixture_pkg_log = (
            root
            / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/fixture/session/decision_log.jsonl"
        )

        sources = [
            {"path": str(product_log), "review_package_mode": "existing_artifact_product_path"},
        ]
        if acceptance_log.is_file():
            sources.append(
                {"path": str(acceptance_log), "review_package_mode": "acceptance_isolated"}
            )
        if fixture_log.is_file():
            sources.append({"path": str(fixture_log), "review_package_mode": "fixture"})
        if fixture_pkg_log.is_file() and fixture_pkg_log.stat().st_size >= 0:
            sources.append(
                {"path": str(fixture_pkg_log), "review_package_mode": "fixture"}
            )

        # Prove product log immutable snapshot before/after
        product_sha_before = sha256_file(product_log)
        product_bytes_before = product_log.read_bytes()

        segment_inv = (
            root
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe/segmentation/target_001_holdout_v2_label_blind_segment_inventory.jsonl"
        )
        segment_index = load_segment_index(segment_inv)
        seg_sha = sha256_file(segment_inv)

        tech = json.loads(
            (
                root
                / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ingestion_and_preflight/input/target_001_independent_holdout_v2_technical_metadata.json"
            ).read_text(encoding="utf-8")
        )

        result = reconstruct_twice_for_determinism(
            project_id="football-analytics",
            run_id="hil_c_target_timeline",
            target_id="target_001",
            video_id="target_001_independent_holdout_v2",
            video_path="data/test_clips/target_001_independent_holdout_v2.mp4",
            video_sha256=VIDEO_SHA,
            frame_rate=float(tech["fps_float"]),
            total_video_frames=int(tech["frame_count"]),
            total_video_duration_seconds=float(tech["video_duration"]),
            decision_sources=sources,
            segment_index=segment_index,
            source_segment_manifest_path=str(segment_inv),
            source_segment_manifest_sha256=seg_sha,
            initial_enrollment=None,
            allow_tracker_continuation=False,
            generated_at=_utc(),
        )
        if result.get("blocked_status") == "BLOCKED_HIL_C_TIMELINE_CONFLICT":
            write_json(temp / "timeline_conflicts.json", result["timeline"]["conflicts"])
            print(result["blocked_status"])
            shutil.rmtree(temp, ignore_errors=True)
            return 3

        timeline = result["timeline"]
        source_audit = result["source_audit"]

        # Attach source manifest SHA into timeline
        source_manifest_path = temp / "target_timeline_decision_source_manifest.json"
        write_json(source_manifest_path, source_audit)
        source_sha = sha256_file(source_manifest_path)
        timeline["decision_source_manifest_path"] = str(source_manifest_path.name)
        timeline["decision_source_manifest_sha256"] = source_sha

        # Immutability check
        product_sha_after = sha256_file(product_log)
        assert product_sha_after == product_sha_before
        assert product_log.read_bytes() == product_bytes_before

        # Effective resolution artifact
        product_rows = DecisionLog(product_log).read_raw()
        from football_analytics.reid.hil.resolve import (
            resolve_effective_decisions,
            resolve_event_review_state,
        )

        effective = resolve_effective_decisions(product_rows)
        event_states = {
            eid: resolve_event_review_state(event_id=eid, decisions=product_rows).value
            for eid in sorted({r["event_id"] for r in product_rows})
        }
        write_json(
            temp / "effective_decision_resolution.json",
            {
                "product_log_sha256": product_sha_after,
                "effective_decisions": effective,
                "event_review_states": event_states,
            },
        )

        # Decision source audit summary
        dec_6f = next(
            (
                d
                for d in source_audit["decisions"]
                if d["decision_id"] == "dec_6fbbcc997aff"
            ),
            None,
        )
        acc_entries = [
            d
            for d in source_audit["decisions"]
            if d["log_sha256"] == ACCEPTANCE_LOG_SHA
            or d["source_classification"] == "ACCEPTANCE_ISOLATED"
        ]
        write_json(
            temp / "decision_source_audit.json",
            {
                "counts": source_audit["counts"],
                "product_approved_confirm_count": source_audit[
                    "product_approved_confirm_count"
                ],
                "dec_6fbbcc997aff": dec_6f,
                "acceptance_log_sha_expected": ACCEPTANCE_LOG_SHA,
                "acceptance_excluded_count": len(acc_entries),
                "acceptance_all_timeline_eligible_false": all(
                    not d["timeline_eligible"] for d in acc_entries
                )
                if acc_entries
                else True,
            },
        )
        write_json(temp / "target_timeline_decision_source_manifest.json", source_audit)

        # Segment provenance validation (for unqualified confirms — documented, not applied)
        from football_analytics.reid.hil.timeline.segments import (
            validate_decision_against_segment,
        )

        seg_val = []
        for row in product_rows:
            if row.get("action") != "CONFIRM_TARGET":
                continue
            try:
                seg_val.append(
                    {
                        "decision_id": row["decision_id"],
                        **validate_decision_against_segment(
                            row,
                            segment_index=segment_index,
                            expected_video_sha256=VIDEO_SHA,
                        ),
                        "timeline_applied": False,
                        "reason": "decision_classified_unqualified_test",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                seg_val.append(
                    {
                        "decision_id": row["decision_id"],
                        "ok": False,
                        "error": str(exc),
                        "timeline_applied": False,
                    }
                )
        write_json(temp / "segment_provenance_validation.json", {"validations": seg_val})

        write_json(temp / "target_001_timeline.json", timeline)
        dump_intervals_jsonl(temp / "target_001_timeline_intervals.jsonl", timeline["intervals"])
        dump_intervals_jsonl(
            temp / "target_001_unresolved_intervals.jsonl",
            timeline["unresolved_intervals"],
        )
        write_json(temp / "target_001_timeline_coverage.json", timeline["coverage_summary"])
        write_json(temp / "timeline_conflicts.json", timeline["conflicts"])
        write_json(
            temp / "timeline_determinism.json",
            {
                "deterministic": result["deterministic"],
                "sha_a": result["sha_a"],
                "sha_b": result["sha_b"],
                "max_numeric_diff": 0,
                "product_log_immutable": True,
                "product_log_sha256": product_sha_after,
            },
        )

        shutil.copy2(
            root / "configs/reid/hil_c_target_timeline_reconstruction_contract.yaml",
            temp / "effective_hil_c_contract.yaml",
        )
        write_json(
            temp / "active_execution_path.json",
            {
                "gate": "REID_HIL_C_TARGET_TIMELINE_RECONSTRUCTION",
                "generator_version": GENERATOR_VERSION,
                "entrypoint": "scripts/run_reid_hil_c_target_timeline_reconstruction.py",
                "no_inference": True,
                "no_detection_tracking": True,
                "no_clip": True,
                "no_metrics": True,
            },
        )

        status = (
            "COMPLETED_HIL_C_NO_APPROVED_PRODUCT_DECISIONS"
            if timeline["timeline_status"] == "no_approved_product_decisions"
            else "COMPLETED_HIL_C_TARGET_TIMELINE_RECONSTRUCTION"
        )
        next_gate = "REID_HIL_D_VERIFIED_TARGET_TRAJECTORY_AND_METRIC_EXPORT"

        report = temp / "target_001_reid_hil_c_target_timeline_report.md"
        report.write_text(
            "\n".join(
                [
                    "# ReID-HIL-C Target Timeline Reconstruction",
                    "",
                    f"- final_status: {status}",
                    f"- timeline_status: {timeline['timeline_status']}",
                    f"- product_approved_confirm_count: {source_audit['product_approved_confirm_count']}",
                    f"- acceptance_excluded: {len(acc_entries)}",
                    f"- dec_6fbbcc997aff class: {(dec_6f or {}).get('source_classification')}",
                    f"- verified_coverage_percentage: {timeline['coverage_summary']['verified_coverage_percentage']}",
                    f"- deterministic: {result['deterministic']}",
                    f"- product_next_gate: {next_gate}",
                    f"- research_next_gate: REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        files = sorted(p.name for p in temp.iterdir() if p.is_file())
        manifest = {
            "schema_version": "target_001_reid_hil_c_manifest_v1",
            "final_status": status,
            "exact_next_gate": next_gate,
            "research_next_gate": "REID_CLIP_A_CONTROLLED_ASSET_ACQUISITION_AND_SECURITY_REVIEW",
            "created_at": _utc(),
            "timeline_status": timeline["timeline_status"],
            "product_approved_confirm_count": source_audit["product_approved_confirm_count"],
            "artifact_files": files,
            "artifacts": {
                name: {
                    "size_bytes": (temp / name).stat().st_size,
                    "sha256": sha256_file(temp / name),
                }
                for name in files
            },
            "acceptance_log_sha256": ACCEPTANCE_LOG_SHA,
            "product_log_sha256": product_sha_after,
            "detection_tracking_reid_run": False,
        }
        write_json(temp / "target_001_reid_hil_c_manifest.json", manifest)
        files = sorted(p.name for p in temp.iterdir() if p.is_file())
        manifest["artifact_files"] = files
        manifest["artifacts"] = {
            name: {
                "size_bytes": (temp / name).stat().st_size,
                "sha256": sha256_file(temp / name),
            }
            for name in files
        }
        write_json(temp / "target_001_reid_hil_c_manifest.json", manifest)

        os.rename(temp, final)
        print(status)
        print(f"output_root={final}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED_HIL_C_ATOMIC_BUILD: {exc}")
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
