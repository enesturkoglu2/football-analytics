"""Tests for HIL-C target timeline reconstruction."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil.decisions import DecisionAction, build_decision  # noqa: E402
from football_analytics.reid.hil.log import DecisionLog  # noqa: E402
from football_analytics.reid.hil.timeline.conflicts import find_interval_overlaps  # noqa: E402
from football_analytics.reid.hil.timeline.reconstruct import (  # noqa: E402
    generate_gaps,
    reconstruct_timeline,
    reconstruct_twice_for_determinism,
)
from football_analytics.reid.hil.timeline.schema import (  # noqa: E402
    IntervalStatus,
    analysis_eligible_for_status,
)
from football_analytics.reid.hil.timeline.sources import (  # noqa: E402
    ACCEPTANCE_LOG_SHA,
    audit_decision_log,
    classify_decision,
)

VIDEO_SHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _confirm(
    log: DecisionLog,
    *,
    decision_id: str,
    event_id: str,
    revision: int,
    segment_id: str,
    raw_track_id: str,
    frame: int,
    run_id: str = "product_timeline_v1",
    supersedes: str | None = None,
    action: str = "CONFIRM_TARGET",
) -> dict:
    payload = build_decision(
        decision_id=decision_id,
        project_id="football-analytics",
        run_id=run_id,
        target_id="target_001",
        event_id=event_id,
        video_id="v1",
        video_path="data/video.mp4",
        video_sha256=VIDEO_SHA,
        reviewer="tester",
        created_at="2026-07-28T00:00:00Z",
        revision=revision,
        action=action,
        selected_candidate_id="c1" if action == "CONFIRM_TARGET" else None,
        selected_segment_id=segment_id if action != "REVOKE" else None,
        selected_raw_track_id=raw_track_id if action != "REVOKE" else None,
        selected_frame_index=frame if action != "REVOKE" else None,
        selected_bbox_xyxy=[1.0, 2.0, 3.0, 4.0] if action != "REVOKE" else None,
        supersedes_decision_id=supersedes,
        confidence="confirmed" if action == "CONFIRM_TARGET" else "unknown",
    )
    return log.append(payload)


class SourceClassificationTests(unittest.TestCase):
    def test_acceptance_and_unqualified_exclusion(self):
        acc = classify_decision(
            {
                "decision_id": "dec_x",
                "event_id": "e",
                "revision": 1,
                "action": "CONFIRM_TARGET",
                "status": "active",
                "run_id": "hil_b_r2_existing_acceptance",
                "selected_segment_id": "s",
            },
            log_path="/tmp/hil_b_r2_acceptance_existing/session_acceptance_r2/decision_log.jsonl",
            log_sha256=ACCEPTANCE_LOG_SHA,
            review_package_mode="acceptance_isolated",
            effective_decision_ids={"dec_x"},
        )
        self.assertEqual(acc["source_classification"], "ACCEPTANCE_ISOLATED")
        self.assertFalse(acc["timeline_eligible"])

        unq = classify_decision(
            {
                "decision_id": "dec_6fbbcc997aff",
                "event_id": "e",
                "revision": 2,
                "action": "CONFIRM_TARGET",
                "status": "active",
                "run_id": "hil_b_existing_artifact_dev",
                "selected_segment_id": "H2_SEG_000003",
            },
            log_path="outputs/.../decision_log.jsonl",
            log_sha256="9ace4d2177bbf8993ec2a0093bd694c3bff6f07e2a606c87c9e318110b886623",
            review_package_mode="existing_artifact_product_path",
            effective_decision_ids={"dec_6fbbcc997aff"},
        )
        self.assertEqual(unq["source_classification"], "PRODUCT_UNQUALIFIED_TEST_DECISION")
        self.assertFalse(unq["timeline_eligible"])

        fix = classify_decision(
            {
                "decision_id": "dec_f",
                "event_id": "e",
                "revision": 1,
                "action": "CONFIRM_TARGET",
                "status": "active",
                "run_id": "hil_a_demo",
                "selected_segment_id": "s",
            },
            log_path="demo_decision_log.jsonl",
            log_sha256="a" * 64,
            review_package_mode="fixture",
            effective_decision_ids={"dec_f"},
        )
        self.assertEqual(fix["source_classification"], "FIXTURE_DEMO")
        self.assertFalse(fix["timeline_eligible"])


class TimelineLogicTests(unittest.TestCase):
    def test_confirm_revoke_gap_and_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(log_path)
            _confirm(
                log,
                decision_id="dec_a",
                event_id="evt_a",
                revision=1,
                segment_id="SEG_A",
                raw_track_id="RAW_A",
                frame=5,
            )
            # second event confirm later in time
            _confirm(
                log,
                decision_id="dec_b",
                event_id="evt_b",
                revision=1,
                segment_id="SEG_B",
                raw_track_id="RAW_B",
                frame=50,
            )
            segment_index = {
                "SEG_A": {
                    "segment_id": "SEG_A",
                    "raw_track_code": "RAW_A",
                    "start_frame": 0,
                    "end_frame": 10,
                    "observation_count": 11,
                },
                "SEG_B": {
                    "segment_id": "SEG_B",
                    "raw_track_code": "RAW_B",
                    "start_frame": 40,
                    "end_frame": 60,
                    "observation_count": 21,
                },
            }
            out = reconstruct_timeline(
                project_id="football-analytics",
                run_id="product_timeline_v1",
                target_id="target_001",
                video_id="v1",
                video_path="data/video.mp4",
                video_sha256=VIDEO_SHA,
                frame_rate=10.0,
                total_video_frames=100,
                total_video_duration_seconds=10.0,
                decision_sources=[{"path": str(log_path), "review_package_mode": "product"}],
                segment_index=segment_index,
                generated_at="2026-07-28T00:00:00Z",
            )
            tl = out["timeline"]
            self.assertEqual(tl["timeline_status"], "ok")
            self.assertEqual(len(tl["intervals"]), 2)
            gaps = tl["unresolved_intervals"]
            self.assertTrue(any(g["start_frame"] == 11 and g["end_frame"] == 39 for g in gaps))
            self.assertFalse(any(g.get("metadata", {}).get("auto_filled") for g in gaps))
            for iv in tl["intervals"]:
                self.assertTrue(iv["analysis_eligible"])
            self.assertFalse(analysis_eligible_for_status(IntervalStatus.APPEARANCE_SUPPORTED))
            self.assertFalse(analysis_eligible_for_status(IntervalStatus.UNRESOLVED))

            # revoke first confirm
            _confirm(
                log,
                decision_id="dec_a_revoke",
                event_id="evt_a",
                revision=2,
                segment_id="SEG_A",
                raw_track_id="RAW_A",
                frame=5,
                supersedes="dec_a",
                action="REVOKE",
            )
            out2 = reconstruct_timeline(
                project_id="football-analytics",
                run_id="product_timeline_v1",
                target_id="target_001",
                video_id="v1",
                video_path="data/video.mp4",
                video_sha256=VIDEO_SHA,
                frame_rate=10.0,
                total_video_frames=100,
                total_video_duration_seconds=10.0,
                decision_sources=[{"path": str(log_path), "review_package_mode": "product"}],
                segment_index=segment_index,
                generated_at="2026-07-28T00:00:00Z",
            )
            segs = {i["segment_id"] for i in out2["timeline"]["intervals"]}
            self.assertNotIn("SEG_A", segs)
            self.assertIn("SEG_B", segs)
            # log immutable bytes still append-only with both records
            self.assertGreaterEqual(len(log.read_raw()), 3)

    def test_overlap_conflict(self):
        intervals = [
            {
                "interval_id": "i1",
                "target_id": "target_001",
                "segment_id": "S1",
                "raw_track_id": "R1",
                "start_frame": 0,
                "end_frame": 20,
                "status": "human_confirmed",
            },
            {
                "interval_id": "i2",
                "target_id": "target_001",
                "segment_id": "S2",
                "raw_track_id": "R2",
                "start_frame": 10,
                "end_frame": 30,
                "status": "human_confirmed",
            },
        ]
        conflicts = find_interval_overlaps(intervals, target_id="target_001")
        self.assertEqual(len(conflicts), 1)

    def test_gap_generation_helper(self):
        gaps = generate_gaps(
            [
                {
                    "start_frame": 0,
                    "end_frame": 1000,
                    "source_event_ids": ["e1"],
                },
                {
                    "start_frame": 1100,
                    "end_frame": 1200,
                    "source_event_ids": ["e2"],
                },
            ],
            target_id="target_001",
            fps=30.0,
        )
        self.assertEqual(gaps[0]["start_frame"], 1001)
        self.assertEqual(gaps[0]["end_frame"], 1099)
        self.assertEqual(gaps[0]["status"], "unresolved")
        self.assertFalse(gaps[0]["analysis_eligible"])

    def test_none_unknown_defer_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "d.jsonl"
            log = DecisionLog(log_path)
            for i, action in enumerate(
                ["NONE_OF_THESE", "UNKNOWN", "DEFER", "INVALID_SEGMENT"], start=1
            ):
                payload = build_decision(
                    decision_id=f"dec_{i}",
                    project_id="football-analytics",
                    run_id="product_timeline_v1",
                    target_id="target_001",
                    event_id=f"evt_{i}",
                    video_id="v1",
                    video_path="data/video.mp4",
                    video_sha256=VIDEO_SHA,
                    reviewer="t",
                    created_at="2026-07-28T00:00:00Z",
                    revision=1,
                    action=action,
                    selected_segment_id="SEG_X" if action == "INVALID_SEGMENT" else None,
                    selected_raw_track_id="RAW_X" if action == "INVALID_SEGMENT" else None,
                )
                log.append(payload)
            out = reconstruct_timeline(
                project_id="football-analytics",
                run_id="product_timeline_v1",
                target_id="target_001",
                video_id="v1",
                video_path="data/video.mp4",
                video_sha256=VIDEO_SHA,
                frame_rate=10.0,
                total_video_frames=50,
                total_video_duration_seconds=5.0,
                decision_sources=[{"path": str(log_path), "review_package_mode": "product"}],
                segment_index={},
                generated_at="2026-07-28T00:00:00Z",
            )
            self.assertEqual(out["timeline"]["timeline_status"], "no_approved_product_decisions")
            self.assertTrue(
                any(u["status"] == "unresolved" for u in out["timeline"]["unresolved_intervals"])
            )
            self.assertTrue(
                any(e["status"] == "invalid" for e in out["timeline"]["excluded_intervals"])
            )

    def test_empty_when_only_unqualified_product(self):
        prod = (
            _PROJECT_ROOT
            / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/existing_artifact/session/decision_log.jsonl"
        )
        if not prod.is_file():
            self.skipTest("product log missing")
        before = prod.read_bytes()
        inv = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe/segmentation/target_001_holdout_v2_label_blind_segment_inventory.jsonl"
        )
        from football_analytics.reid.hil.timeline.segments import load_segment_index

        out = reconstruct_twice_for_determinism(
            project_id="football-analytics",
            run_id="hil_c",
            target_id="target_001",
            video_id="holdout",
            video_path="data/test_clips/target_001_independent_holdout_v2.mp4",
            video_sha256="bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f",
            frame_rate=30.0,
            total_video_frames=1058,
            total_video_duration_seconds=35.266633,
            decision_sources=[
                {"path": str(prod), "review_package_mode": "existing_artifact_product_path"}
            ],
            segment_index=load_segment_index(inv),
            generated_at="2026-07-28T00:00:00Z",
        )
        self.assertTrue(out["deterministic"])
        self.assertEqual(out["timeline"]["timeline_status"], "no_approved_product_decisions")
        self.assertEqual(out["timeline"]["coverage_summary"]["verified_coverage_percentage"], 0.0)
        self.assertEqual(prod.read_bytes(), before)
        man = audit_decision_log(
            prod, review_package_mode="existing_artifact_product_path"
        )
        row = next(d for d in man["decisions"] if d["decision_id"] == "dec_6fbbcc997aff")
        self.assertEqual(row["source_classification"], "PRODUCT_UNQUALIFIED_TEST_DECISION")

    def test_segment_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "d.jsonl"
            log = DecisionLog(log_path)
            _confirm(
                log,
                decision_id="dec_bad",
                event_id="evt",
                revision=1,
                segment_id="SEG_MISSING",
                raw_track_id="RAW",
                frame=1,
            )
            out = reconstruct_timeline(
                project_id="football-analytics",
                run_id="product_timeline_v1",
                target_id="target_001",
                video_id="v1",
                video_path="data/video.mp4",
                video_sha256=VIDEO_SHA,
                frame_rate=10.0,
                total_video_frames=10,
                total_video_duration_seconds=1.0,
                decision_sources=[{"path": str(log_path), "review_package_mode": "product"}],
                segment_index={"SEG_OTHER": {"segment_id": "SEG_OTHER", "raw_track_code": "X", "start_frame": 0, "end_frame": 1}},
                generated_at="2026-07-28T00:00:00Z",
            )
            self.assertEqual(out.get("blocked_status"), "BLOCKED_HIL_C_SEGMENT_PROVENANCE_MISMATCH")

    def test_acceptance_sha_constant(self):
        acc = Path("/tmp/hil_b_r2_acceptance_existing/session_acceptance_r2/decision_log.jsonl")
        if not acc.is_file():
            self.skipTest("acceptance log missing")
        self.assertEqual(hashlib.sha256(acc.read_bytes()).hexdigest(), ACCEPTANCE_LOG_SHA)
        man = audit_decision_log(acc, review_package_mode="acceptance_isolated")
        self.assertTrue(all(not d["timeline_eligible"] for d in man["decisions"]))
        self.assertTrue(
            all(d["source_classification"] == "ACCEPTANCE_ISOLATED" for d in man["decisions"])
        )


if __name__ == "__main__":
    unittest.main()
