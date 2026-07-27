"""Tests for HIL-C2 product package, approvals, and timeline qualification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil.decisions import build_decision  # noqa: E402
from football_analytics.reid.hil.log import DecisionLog  # noqa: E402
from football_analytics.reid.hil.timeline.approvals import (  # noqa: E402
    ApprovalError,
    ApprovalLog,
    assert_decision_approvable,
    build_approval_record,
    resolve_active_approvals,
)
from football_analytics.reid.hil.timeline.reconstruct import (  # noqa: E402
    reconstruct_twice_for_determinism,
)
from football_analytics.reid.hil.timeline.sources import (  # noqa: E402
    ACCEPTANCE_LOG_SHA,
    classify_decision,
)
from football_analytics.reid.hil_c2.product_package import (  # noqa: E402
    PRODUCT_PACKAGE_ID,
    VIDEO_SHA,
    build_product_external_review_package,
)
from football_analytics.reid.hil_c2.qualify import qualify_product_session  # noqa: E402
from football_analytics.reid.hil_c2.source_audit import (  # noqa: E402
    ProductSourceClass,
    audit_product_video_sources,
)

VIDEO_SHA_TEST = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


def _confirm_decision(**kwargs):
    defaults = dict(
        decision_id="dec_prod_001",
        project_id="football-analytics",
        run_id="hil_c2_product_external_enrollment",
        target_id="target_001",
        event_id="evt_product_initial_enrollment_001",
        video_id="target_001_external_enrollment_v1",
        video_path="data/enrollment_clips/target_001_external_enrollment_v1.mp4",
        video_sha256=VIDEO_SHA_TEST,
        reviewer="tester",
        created_at="2026-07-28T00:00:00Z",
        revision=1,
        action="CONFIRM_TARGET",
        selected_candidate_id="prod_cand_001",
        selected_segment_id="EXT_SEG_004",
        selected_raw_track_id="11",
        selected_frame_index=0,
        selected_bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        confidence="confirmed",
    )
    defaults.update(kwargs)
    return build_decision(**defaults)


class SourceAuditTests(unittest.TestCase):
    def test_product_source_classification(self):
        audit = audit_product_video_sources(_PROJECT_ROOT)
        by_id = {s["video_id"]: s for s in audit["sources"]}
        self.assertEqual(
            by_id["target_001_independent_holdout_v2"]["classification"],
            ProductSourceClass.DEVELOPMENT_HOLDOUT_ONLY.value,
        )
        self.assertEqual(
            by_id["fixture_video"]["classification"],
            ProductSourceClass.INVALID_OR_INCOMPLETE.value,
        )
        self.assertNotEqual(
            audit.get("selected_source_video_id"),
            "target_001_independent_holdout_v2",
        )
        if by_id["target_001_external_enrollment_v1"].get("review_package_ready"):
            self.assertEqual(
                by_id["target_001_external_enrollment_v1"]["classification"],
                ProductSourceClass.READY_WITH_ADAPTER.value,
            )
            self.assertEqual(
                audit["selected_source_video_id"],
                "target_001_external_enrollment_v1",
            )


class ApprovalTests(unittest.TestCase):
    def test_approval_schema_append_only_and_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            dpath = Path(tmp) / "decision_log.jsonl"
            apath = Path(tmp) / "approval_log.jsonl"
            log = DecisionLog(dpath)
            dec = _confirm_decision()
            log.append(dec)
            # Without approval → not eligible
            row = classify_decision(
                dec,
                log_path=str(dpath),
                log_sha256="a" * 64,
                review_package_mode="product",
                effective_decision_ids={dec["decision_id"]},
                approved_decision_ids=set(),
                require_timeline_approval=True,
            )
            self.assertFalse(row["timeline_eligible"])
            self.assertEqual(row["exclusion_reason"], "missing_product_timeline_approval")

            alog = ApprovalLog(apath)
            before = apath.read_bytes()
            record = build_approval_record(
                approval_id="appr_001",
                decision=dec,
                product_package_id=PRODUCT_PACKAGE_ID,
                decision_log_path=str(dpath),
                decision_log_sha256_at_approval="b" * 64,
            )
            alog.append(record)
            self.assertEqual(len(alog.read_raw()), 1)
            after = apath.read_bytes()
            self.assertTrue(after.startswith(before))

            active = resolve_active_approvals(alog.read_raw())
            self.assertIn(dec["decision_id"], active)
            row2 = classify_decision(
                dec,
                log_path=str(dpath),
                log_sha256="a" * 64,
                review_package_mode="product",
                effective_decision_ids={dec["decision_id"]},
                approved_decision_ids=set(active.keys()),
                require_timeline_approval=True,
            )
            self.assertTrue(row2["timeline_eligible"])

            # revoke approval
            alog.append(
                build_approval_record(
                    approval_id="appr_002",
                    decision=dec,
                    product_package_id=PRODUCT_PACKAGE_ID,
                    decision_log_path=str(dpath),
                    decision_log_sha256_at_approval="b" * 64,
                    approval_status="revoked",
                    supersedes_approval_id="appr_001",
                )
            )
            active2 = resolve_active_approvals(alog.read_raw())
            self.assertNotIn(dec["decision_id"], active2)

    def test_acceptance_and_fixture_approval_rejection(self):
        dec = _confirm_decision(decision_id="dec_6fbbcc997aff")
        with self.assertRaises(ApprovalError):
            build_approval_record(
                approval_id="appr_bad",
                decision=dec,
                product_package_id=PRODUCT_PACKAGE_ID,
                decision_log_path="/tmp/x",
                decision_log_sha256_at_approval="c" * 64,
            )
        ok = _confirm_decision()
        with self.assertRaises(ApprovalError):
            assert_decision_approvable(
                ok,
                log_path="/tmp/hil_b_r2_acceptance/x.jsonl",
                log_sha256=ACCEPTANCE_LOG_SHA,
                review_package_mode="product",
                product_package_id=PRODUCT_PACKAGE_ID,
                expected_target_id="target_001",
                expected_video_sha256=VIDEO_SHA_TEST,
            )
        with self.assertRaises(ApprovalError):
            assert_decision_approvable(
                ok,
                log_path="/tmp/dec.jsonl",
                log_sha256="d" * 64,
                review_package_mode="fixture",
                product_package_id=PRODUCT_PACKAGE_ID,
                expected_target_id="target_001",
                expected_video_sha256=VIDEO_SHA_TEST,
            )

    def test_cross_video_and_target_rejection(self):
        ok = _confirm_decision()
        with self.assertRaises(ApprovalError):
            assert_decision_approvable(
                ok,
                log_path="/tmp/dec.jsonl",
                log_sha256="d" * 64,
                review_package_mode="product",
                product_package_id=PRODUCT_PACKAGE_ID,
                expected_target_id="target_002",
                expected_video_sha256=VIDEO_SHA_TEST,
            )
        with self.assertRaises(ApprovalError):
            assert_decision_approvable(
                ok,
                log_path="/tmp/dec.jsonl",
                log_sha256="d" * 64,
                review_package_mode="product",
                product_package_id=PRODUCT_PACKAGE_ID,
                expected_target_id="target_001",
                expected_video_sha256="e" * 64,
            )


class PackageAndReconstructionTests(unittest.TestCase):
    def test_product_package_and_reconstruction_with_approval(self):
        audit = audit_product_video_sources(_PROJECT_ROOT)
        if audit.get("selected_source_video_id") != "target_001_external_enrollment_v1":
            self.skipTest("external enrollment artifacts not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = build_product_external_review_package(root, project_root=_PROJECT_ROOT)
            self.assertEqual(pkg.get("package_file") or "", pkg["package_file"])
            package = json.loads(Path(pkg["package_file"]).read_text(encoding="utf-8"))
            self.assertEqual(package["package_id"], PRODUCT_PACKAGE_ID)
            self.assertEqual(package["provenance"]["package_mode"], "product")
            self.assertFalse(package["provenance"].get("gt_prefill"))
            self.assertTrue(package["provenance"].get("not_reused_existing_artifact_package"))

            # Decision without approval → empty timeline
            log = DecisionLog(pkg["decision_log_path"])
            # Use real segment bounds from inventory
            inv = [
                json.loads(line)
                for line in Path(pkg["segment_inventory_path"]).read_text().splitlines()
                if line.strip()
            ]
            seg = next(s for s in inv if s["segment_id"] == "EXT_SEG_004")
            dec = _confirm_decision(
                selected_frame_index=int(seg["start_frame"]),
                selected_raw_track_id=str(seg["raw_track_code"]),
            )
            log.append(dec)
            qual0 = qualify_product_session(
                decision_log_path=pkg["decision_log_path"],
                approval_log_path=pkg["approval_log_path"],
            )
            self.assertEqual(qual0["counts"]["timeline_eligible"], 0)

            ApprovalLog(pkg["approval_log_path"]).append(
                build_approval_record(
                    approval_id="appr_test_001",
                    decision=dec,
                    product_package_id=PRODUCT_PACKAGE_ID,
                    decision_log_path=pkg["decision_log_path"],
                    decision_log_sha256_at_approval="f" * 64,
                    segment_manifest_sha256=pkg["segment_inventory_sha256"],
                )
            )
            qual1 = qualify_product_session(
                decision_log_path=pkg["decision_log_path"],
                approval_log_path=pkg["approval_log_path"],
            )
            self.assertEqual(qual1["counts"]["timeline_eligible"], 1)

            segment_index = {s["segment_id"]: s for s in inv}
            out = reconstruct_twice_for_determinism(
                project_id="football-analytics",
                run_id="hil_c2_product_external_enrollment",
                target_id="target_001",
                video_id="target_001_external_enrollment_v1",
                video_path=str(
                    _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
                ),
                video_sha256=VIDEO_SHA,
                frame_rate=30.0,
                total_video_frames=784,
                total_video_duration_seconds=784 / 30.0,
                decision_sources=[
                    {"path": pkg["decision_log_path"], "review_package_mode": "product"}
                ],
                segment_index=segment_index,
                source_segment_manifest_path=pkg["segment_inventory_path"],
                source_segment_manifest_sha256=pkg["segment_inventory_sha256"],
                generated_at="2026-07-28T00:00:00Z",
                approved_decision_ids=set(qual1["approved_decision_ids"]),
                require_timeline_approval=True,
            )
            self.assertTrue(out["deterministic"])
            self.assertGreaterEqual(len(out["timeline"]["intervals"]), 1)
            # unresolved gap after enrollment segment until video end may exist
            self.assertTrue(
                any(i["status"] == "human_confirmed" for i in out["timeline"]["intervals"])
            )
            # decision log not mutated by reconstruction
            sha_before = Path(pkg["decision_log_path"]).read_bytes()
            reconstruct_twice_for_determinism(
                project_id="football-analytics",
                run_id="hil_c2_product_external_enrollment",
                target_id="target_001",
                video_id="target_001_external_enrollment_v1",
                video_path=str(
                    _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
                ),
                video_sha256=VIDEO_SHA,
                frame_rate=30.0,
                total_video_frames=784,
                total_video_duration_seconds=784 / 30.0,
                decision_sources=[
                    {"path": pkg["decision_log_path"], "review_package_mode": "product"}
                ],
                segment_index=segment_index,
                generated_at="2026-07-28T00:00:00Z",
                approved_decision_ids=set(qual1["approved_decision_ids"]),
                require_timeline_approval=True,
            )
            self.assertEqual(sha_before, Path(pkg["decision_log_path"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
