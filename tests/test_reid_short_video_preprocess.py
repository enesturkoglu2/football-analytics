"""Tests for short-video dense bbox / path resolve / provisional timeline (no Game State)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.short_video.dense_timeline import (  # noqa: E402
    build_dense_bbox_timeline,
    observations_for_component,
)
from football_analytics.reid.short_video.path_resolve import (  # noqa: E402
    ShortVideoInputError,
    resolve_short_video_path,
    windows_to_wsl_candidates,
)
from football_analytics.reid.short_video.provisional_timeline import (  # noqa: E402
    append_provisional_from_confirm,
    empty_provisional,
    mark_unresolved,
    target_status_from_selection,
)


class PathResolveTests(unittest.TestCase):
    def test_windows_candidates_include_mnt(self):
        cands = windows_to_wsl_candidates(r"C:\Users\enest\Downloads\kisa_mac_klip.mp4")
        self.assertTrue(any(str(p).startswith("/mnt/c/") for p in cands))

    def test_refuse_missing(self):
        with self.assertRaises(ShortVideoInputError):
            resolve_short_video_path("/tmp/does_not_exist_short_video_xyz.mp4")


class DenseTimelineTests(unittest.TestCase):
    def test_all_tracks_dense_no_legacy_fallback(self):
        mapping = [
            {
                "external_candidate_code": "EXT_001",
                "raw_external_track_id": 7,
                "first_frame": 0,
                "last_frame": 2,
                "observation_coverage": 1.0,
                "review_eligible": True,
                "bbox_per_observation": [
                    {"frame_index": 0, "bbox_xyxy": [1, 2, 3, 4], "detection_id": "d0"},
                    {"frame_index": 1, "bbox_xyxy": [2, 3, 4, 5], "detection_id": "d1"},
                    {"frame_index": 2, "bbox_xyxy": [3, 4, 5, 6], "detection_id": "d2"},
                ],
            },
            {
                "external_candidate_code": "EXT_002",
                "raw_external_track_id": 9,
                "first_frame": 1,
                "last_frame": 1,
                "observation_coverage": 1.0,
                "review_eligible": True,
                "bbox_per_observation": [
                    {"frame_index": 1, "bbox_xyxy": [10, 10, 20, 20], "detection_id": "d9"},
                ],
            },
        ]
        dens = build_dense_bbox_timeline(
            mapping,
            video_id="v",
            video_sha256="a" * 64,
            fps=30.0,
            frame_count=3,
            width=100,
            height=100,
        )
        self.assertEqual(dens["schema_version"], "interactive_video_bbox_timeline_v1")
        self.assertFalse(dens["legacy_ext_mapping_fallback_used"])
        self.assertFalse(dens["game_state_executed"])
        self.assertEqual(len(dens["observations_by_frame"]["1"]), 2)
        comp = observations_for_component(dens)
        self.assertEqual(comp["1"][0]["raw_track_id"], "7")


class ProvisionalTimelineTests(unittest.TestCase):
    def test_confirm_and_unresolved_no_auto_switch(self):
        p = empty_provisional(target_id="target_001", video_id="v")
        p = append_provisional_from_confirm(
            p,
            decision_id="dec_1",
            event_id="evt_1",
            segment_id="EXT_SEG_001",
            raw_track_id="7",
            start_frame=10,
            end_frame=50,
            fps=30.0,
        )
        self.assertEqual(len(p["intervals"]), 2)
        self.assertFalse(p["analysis_eligible"])
        p = mark_unresolved(
            p,
            start_frame=51,
            end_frame=80,
            fps=30.0,
            previous_segment_id="EXT_SEG_001",
            previous_raw_track_id="7",
        )
        self.assertEqual(p["unresolved_intervals"][0]["status"], "UNRESOLVED")
        status = target_status_from_selection(
            selected_raw_track_id="7",
            track_end_frame=50,
            current_frame=55,
            has_unresolved=True,
        )
        self.assertEqual(status, "UNRESOLVED")

    def test_tracking_while_on_track(self):
        status = target_status_from_selection(
            selected_raw_track_id="7",
            track_end_frame=50,
            current_frame=20,
            has_unresolved=False,
        )
        self.assertEqual(status, "TRACKING")


class NoGameStateImportTests(unittest.TestCase):
    def test_gate_script_forbids_game_state_clip(self):
        script = (
            _PROJECT
            / "scripts/run_reid_product_new_short_video_preprocess_validation.py"
        )
        text = script.read_text(encoding="utf-8")
        self.assertIn("No Game State", text)
        self.assertIn("forbidden_next", text)

if __name__ == "__main__":
    unittest.main()
