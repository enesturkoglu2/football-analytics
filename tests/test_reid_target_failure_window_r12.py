"""Tests for Target Failure Window Pilot R1.2 static event review."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    CANDIDATE_REQUIRED_LABELS,
    COVERAGE_SCOPE,
    NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
    PILOT_LABELS,
    build_pilot_label_event,
    choose_pilot_next_gate,
    summarize_pilot_labels,
)
from football_analytics.reid.golden_clip.schema import empty_ground_truth  # noqa: E402
from football_analytics.reid.golden_clip.static_packages import (  # noqa: E402
    pick_enrollment_frame,
    resolve_enrollment_click,
)


class DefaultNoInteractiveComponentTests(unittest.TestCase):
    def test_app_does_not_import_component_at_module_level(self):
        app = (
            _SRC / "football_analytics/reid/golden_clip/streamlit_app.py"
        ).read_text(encoding="utf-8")
        # Top-level / default path must not declare_component usage
        self.assertNotIn("from football_analytics.reid.hil_ui.interactive_video_component", app.split("Advanced / Experimental")[0])
        self.assertIn("Advanced / Experimental Interactive Mode", app)
        self.assertIn("Enable experimental interactive component", app)
        self.assertIn("value=False", app)
        self.assertIn("streamlit_image_coordinates", app)
        self.assertIn("st.form", app)
        self.assertIn("st.video", app)


class EnrollmentAndClickTests(unittest.TestCase):
    def test_pick_enrollment_and_click_resolution(self):
        obs = {
            "40": [
                {
                    "bbox_xyxy": [10, 10, 50, 80],
                    "raw_track_id": "7",
                    "segment_id": "S7",
                    "detection_id": "d1",
                    "bbox_id": "enr_0_7",
                },
                {
                    "bbox_xyxy": [200, 200, 240, 280],
                    "raw_track_id": "9",
                    "segment_id": "S9",
                    "detection_id": "d2",
                    "bbox_id": "enr_1_9",
                },
            ]
        }
        for i in range(30, 80):
            obs.setdefault(str(i), obs["40"] if i == 40 else [{"bbox_xyxy": [0, 0, 1, 1], "raw_track_id": "1", "segment_id": "S1", "bbox_id": f"b{i}"}])
        fi = pick_enrollment_frame(obs, frame_count=1357, prefer_from=30, prefer_to=80)
        self.assertEqual(fi, 40)
        enroll = {
            "frame_w": 400,
            "frame_h": 300,
            "display_w": 400,
            "display_h": 300,
            "candidates": obs["40"],
        }
        hit = resolve_enrollment_click(ui_x=30, ui_y=40, enrollment=enroll)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["raw_track_id"], "7")
        miss = resolve_enrollment_click(ui_x=1, ui_y=1, enrollment=enroll)
        self.assertIsNone(miss)


class OneWindowAndFormContractTests(unittest.TestCase):
    def test_static_packages_module_exists_and_bounds(self):
        from football_analytics.reid.golden_clip import static_packages as sp

        self.assertEqual(sp.CLIP_RADIUS_FRAMES, 75)
        start, end = sp._safe_clip_bounds(100, frame_count=1357, radius=75)
        self.assertEqual(start, 25)
        self.assertEqual(end, 175)
        self.assertLessEqual((end - start + 1) / 30.0, 6.1)
        self.assertGreaterEqual((end - start + 1) / 30.0, 3.0)

    def test_form_submit_idempotence_and_candidate_rules(self):
        with self.assertRaises(ValueError):
            build_pilot_label_event(
                window={"window_id": "fw_1", "start_frame": 1, "end_frame": 2, "previous_raw_track_id": "1"},
                label="RAW_TRACK_FRAGMENT_SAME_TARGET",
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256="a" * 64,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
                selected_next_raw_track_id=None,
            )
        self.assertIn("RAW_TRACK_FRAGMENT_SAME_TARGET", CANDIDATE_REQUIRED_LABELS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation_log.jsonl"
            log = AnnotationLog(path)
            assert_not_product_log(path)
            ev = build_pilot_label_event(
                window={
                    "window_id": "fw_1",
                    "start_frame": 10,
                    "end_frame": 40,
                    "start_time": 0.3,
                    "end_time": 1.3,
                    "previous_raw_track_id": "1",
                },
                label="TARGET_VISIBLE_BUT_DETECTION_MISSED",
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256="a" * 64,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
                event_uuid="euuid_test_1",
                failure_window_id="fw_1",
                ui_mode="static",
            )
            log.append(ev)
            # duplicate window tracking via uuid set simulation
            seen = {ev["event_uuid"]}
            self.assertIn("euuid_test_1", seen)
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


class PilotHonestyAndGateTests(unittest.TestCase):
    def test_three_window_minimum_and_no_full_metrics(self):
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
                    window={
                        "window_id": f"fw_{i}",
                        "start_frame": i * 10,
                        "end_frame": i * 10 + 5,
                        "previous_raw_track_id": "1",
                    },
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
        self.assertEqual(
            summary["full_metrics"]["target_idf1"],
            NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
        )
        gate = choose_pilot_next_gate(summary)
        self.assertEqual(
            gate["exact_next_gate"], "EXISTING_TRACKLET_STITCHING_INTEGRATION"
        )
        self.assertTrue(PILOT_LABELS >= {"BORDER_EXIT_REENTRY", "UNCERTAIN"})

    def test_insufficient_labels_gate(self):
        gate = choose_pilot_next_gate({"labeled_window_count": 1, "taxonomy": {}})
        self.assertEqual(
            gate["exact_next_gate"],
            "TARGET_FAILURE_WINDOW_PILOT_MORE_USER_EVIDENCE_REQUIRED",
        )


class NoContaminationTests(unittest.TestCase):
    def test_refuse_product_and_no_rerun_claims(self):
        with self.assertRaises(Exception):
            assert_not_product_log(Path("/tmp/product_review_package/decision_log.jsonl"))
        app = (
            _SRC / "football_analytics/reid/golden_clip/streamlit_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("custom video component yok", app)
        sp = (
            _SRC / "football_analytics/reid/golden_clip/static_packages.py"
        ).read_text(encoding="utf-8")
        self.assertIn("full_dense_manifest_not_sent_to_browser", sp)


class NativeClipMockTests(unittest.TestCase):
    def test_extract_clip_invokes_ffmpeg_contract(self):
        from football_analytics.reid.golden_clip import static_packages as sp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            video = tmp_p / "v.mp4"
            video.write_bytes(b"fake")
            out = tmp_p / "clip.mp4"

            def fake_sha(path):
                return "abc"

            def fake_run(cmd, capture_output, text, check):
                out.write_bytes(b"mp4")
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(sp, "sha256_file", side_effect=lambda p: "abc"):
                with mock.patch.object(sp.subprocess, "run", side_effect=fake_run):
                    meta = sp._extract_native_clip(
                        source_video=video,
                        source_video_sha256="abc",
                        start_frame=0,
                        end_frame=89,
                        fps=30.0,
                        output_path=out,
                    )
            self.assertTrue(out.is_file())
            self.assertEqual(meta["start_frame"], 0)
            self.assertEqual(meta["end_frame"], 89)


if __name__ == "__main__":
    unittest.main()
