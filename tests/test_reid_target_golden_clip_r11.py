"""Tests for Target Golden Clip R1.1 freeze repair + failure-window pilot."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.golden_clip.annotation_log import (  # noqa: E402
    AnnotationLog,
    assert_not_product_log,
)
from football_analytics.reid.golden_clip.intervals import rebuild_gt_from_events  # noqa: E402
from football_analytics.reid.golden_clip.pilot import (  # noqa: E402
    COVERAGE_SCOPE,
    NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
    build_pilot_label_event,
    choose_pilot_next_gate,
    generate_failure_windows,
    summarize_pilot_labels,
)
from football_analytics.reid.golden_clip.schema import empty_ground_truth  # noqa: E402
from football_analytics.reid.golden_clip.window_index import (  # noqa: E402
    payload_byte_size,
    slice_window,
    track_span_lightweight,
)


class BoundedPayloadTests(unittest.TestCase):
    def test_slice_window_much_smaller_than_full(self):
        obs = {
            str(i): [
                {
                    "bbox_xyxy": [0, 0, 10, 10],
                    "raw_track_id": "1",
                    "segment_id": "S1",
                    "detection_id": f"d{i}",
                }
            ]
            for i in range(1357)
        }
        full = payload_byte_size(obs)
        win = slice_window(obs, center_frame=600, radius=45, frame_count=1357)
        small = payload_byte_size(win)
        self.assertLess(small * 10, full)
        self.assertEqual(len(win), 91)

    def test_track_span_lightweight_sparse(self):
        obs = {
            str(i): [
                {
                    "bbox_xyxy": [0, 0, 10, 10],
                    "raw_track_id": "7",
                    "segment_id": "S7",
                    "detection_id": f"d{i}",
                }
            ]
            for i in range(0, 300)
        }
        span = track_span_lightweight(obs, raw_track_id="7", sample_every=30)
        self.assertEqual(span["first_frame"], 0)
        self.assertEqual(span["last_frame"], 299)
        self.assertLess(len(span["sparse_bbox_observations"]), 300)


class DebounceFrontendContractTests(unittest.TestCase):
    def test_frontend_has_event_uuid_and_debounce_and_keep_video(self):
        html = (
            _SRC
            / "football_analytics/reid/hil_ui/interactive_video_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("event_uuid", html)
        self.assertIn("lastClickFingerprint", html)
        self.assertIn("400", html)
        self.assertIn("keep current src", html)
        self.assertIn("detection_id", html)


class PilotSchemaTests(unittest.TestCase):
    def test_pilot_append_only_and_partial_coverage_honesty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation_log.jsonl"
            log = AnnotationLog(path)
            assert_not_product_log(path)
            w = {
                "window_id": "fw_x",
                "kind": "selected_raw_track_end",
                "center_frame": 100,
                "start_frame": 70,
                "end_frame": 130,
                "start_time": 70 / 30,
                "end_time": 131 / 30,
                "previous_raw_track_id": "7",
            }
            ev = build_pilot_label_event(
                window=w,
                label="RAW_TRACK_FRAGMENT_SAME_TARGET",
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256="a" * 64,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
                selected_next_raw_track_id="8",
            )
            self.assertEqual(ev["coverage_scope"], COVERAGE_SCOPE)
            log.append(ev)
            # rebuild full GT ignores pilot events
            base = empty_ground_truth(
                source_video_sha256="a" * 64,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
                annotation_session_id="s",
                reviewer="t",
                frame_count=1357,
                fps=30.0,
            )
            gt = rebuild_gt_from_events(log.read_raw(), base_gt=base, fps=30.0)
            self.assertEqual(gt["intervals"], [])
            summary = summarize_pilot_labels(log.read_raw())
            self.assertEqual(summary["coverage_scope"], COVERAGE_SCOPE)
            self.assertEqual(
                summary["full_metrics"]["target_idf1"],
                NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
            )
            gate = choose_pilot_next_gate(summary)
            self.assertEqual(
                gate["exact_next_gate"],
                "TARGET_FAILURE_WINDOW_PILOT_MORE_USER_EVIDENCE_REQUIRED",
            )

    def test_generate_failure_windows_and_gate_with_labels(self):
        obs = {}
        for i in range(0, 50):
            obs[str(i)] = [
                {
                    "bbox_xyxy": [10, 10, 30, 50],
                    "raw_track_id": "1",
                    "segment_id": "S1",
                }
            ]
        for i in range(55, 80):
            obs[str(i)] = [
                {
                    "bbox_xyxy": [12, 12, 32, 52],
                    "raw_track_id": "2",
                    "segment_id": "S2",
                }
            ]
        wins = generate_failure_windows(
            observations_by_frame=obs,
            selected_raw_track_id="1",
            selected_segment_id="S1",
            track_first=0,
            track_last=49,
            fps=30.0,
            frame_count=100,
            max_windows=5,
        )
        self.assertGreaterEqual(len(wins), 1)
        events = []
        for i, lab in enumerate(
            [
                "RAW_TRACK_FRAGMENT_SAME_TARGET",
                "RAW_TRACK_FRAGMENT_SAME_TARGET",
                "SHORT_OCCLUSION_FRAGMENTATION",
            ]
        ):
            events.append(
                build_pilot_label_event(
                    window=wins[min(i, len(wins) - 1)],
                    label=lab,
                    reviewer="t",
                    annotation_session_id="s",
                    source_video_sha256="a" * 64,
                    match_id="m",
                    analysis_run_id="r",
                    target_id="target_001",
                    selected_next_raw_track_id="2",
                )
            )
        summary = summarize_pilot_labels(events)
        self.assertEqual(summary["labeled_window_count"], 3)
        gate = choose_pilot_next_gate(summary)
        self.assertIn(
            gate["exact_next_gate"],
            {
                "EXISTING_TRACKLET_STITCHING_INTEGRATION",
                "TARGET_CONDITIONED_SHORT_OCCLUSION_BRIDGE",
                "TARGET_DETECTION_RECALL_REMEDIATION",
                "TARGET_ID_STATE_AND_OBSERVATION_LAYER",
            },
        )


class NoContaminationTests(unittest.TestCase):
    def test_refuse_product_log_and_no_rerun_in_audit_script(self):
        with self.assertRaises(Exception):
            assert_not_product_log(Path("/tmp/product_review_package/decision_log.jsonl"))
        script = (
            _PROJECT / "scripts/run_target_golden_clip_r11_freeze_repair_audit.py"
        ).read_text(encoding="utf-8")
        self.assertIn("no detection/tracking rerun", script.lower())
        app = (
            _SRC / "football_analytics/reid/golden_clip/streamlit_app.py"
        ).read_text(encoding="utf-8")
        # R1.2: static is default; interactive only under Advanced/Experimental
        self.assertIn("gt_r12_enrollment_click_stable", app)
        self.assertIn("submitted_event_uuids", app)
        self.assertIn("Advanced / Experimental Interactive Mode", app)
        self.assertNotIn("observations_for_component(dens_full)", app)


class IncompleteFreezeExclusionTests(unittest.TestCase):
    def test_incomplete_freeze_does_not_enter_accepted_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation_log.jsonl"
            log = AnnotationLog(path)
            log.append(
                {
                    "schema_version": "target_gt_annotation_event_v1",
                    "event_id": "gtevt_x",
                    "action": "INCOMPLETE_UI_FREEZE_EVENT",
                    "created_at": "2026-07-29T00:00:00Z",
                    "qualification": "INCOMPLETE_UI_FREEZE_EVENT",
                    "accepted_gt_exclusion": True,
                }
            )
            base = empty_ground_truth(
                source_video_sha256="a" * 64,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
                annotation_session_id="s",
                reviewer="t",
                frame_count=1357,
                fps=30.0,
            )
            gt = rebuild_gt_from_events(log.read_raw(), base_gt=base, fps=30.0)
            self.assertEqual(gt["intervals"], [])
            self.assertFalse(gt.get("accepted"))


class LatencyAndKeyContractTests(unittest.TestCase):
    def test_app_reports_latency_and_stable_key_and_cache(self):
        app = (
            _SRC / "football_analytics/reid/golden_clip/streamlit_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("selection_latency_ms", app)
        self.assertIn('key="gt_r12_enrollment_click_stable"', app)
        self.assertIn("st.cache_data", app)
        self.assertIn("submitted_event_uuids", app)
        self.assertIn("TARGET_FAILURE_WINDOW_PILOT", app)
        self.assertIn("st.form", app)
        self.assertIn("st.video", app)


if __name__ == "__main__":
    unittest.main()
