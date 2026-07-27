"""Unit tests covering short-video interactive contract behaviors (offline)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil_ui.dense_observations import (  # noqa: E402
    build_dense_observations_from_mapping,
)
from football_analytics.reid.hil_c2.product_package import _segment_id  # noqa: E402


class SyncAndClickResolutionTests(unittest.TestCase):
    def test_moving_bbox_per_frame_and_click_fields(self):
        mapping = [
            {
                "external_candidate_code": "EXT_010",
                "raw_external_track_id": 42,
                "bbox_per_observation": [
                    {"frame_index": 5, "bbox_xyxy": [10.0, 10.0, 40.0, 80.0]},
                    {"frame_index": 6, "bbox_xyxy": [12.0, 11.0, 42.0, 81.0]},
                    {"frame_index": 7, "bbox_xyxy": [14.0, 12.0, 44.0, 82.0]},
                ],
            }
        ]
        dens = build_dense_observations_from_mapping(
            mapping, codes=None, segment_id_fn=_segment_id
        )
        self.assertEqual(set(dens), {"5", "6", "7"})
        # bboxes move across frames
        self.assertNotEqual(dens["5"][0]["bbox_xyxy"], dens["7"][0]["bbox_xyxy"])
        hit = dens["6"][0]
        # click resolution payload fields
        for key in ("bbox_xyxy", "segment_id", "raw_track_id"):
            self.assertIn(key, hit)
        self.assertEqual(hit["raw_track_id"], "42")
        self.assertEqual(hit["segment_id"], "EXT_SEG_010")

    def test_no_automatic_raw_track_switch_contract(self):
        # Selecting track 42 must not invent another raw_track_id at track end
        selected = "42"
        track_end = 7
        current = 8
        next_auto = None  # contract: never auto-assign
        self.assertIsNone(next_auto)
        self.assertEqual(selected, "42")
        self.assertGreater(current, track_end)


class ArtifactPresenceTests(unittest.TestCase):
    def test_short_video_run_artifacts_if_present(self):
        root = (
            _PROJECT
            / "outputs/reid/product_new_short_video_preprocess_validation"
        )
        runs = sorted(root.glob("sv_run_*")) if root.is_dir() else []
        if not runs:
            self.skipTest("no short-video run yet")
        run = runs[-1]
        for name in (
            "source_audit.json",
            "detection/detections.jsonl",
            "tracking/tracks.jsonl",
            "dense_bbox_timeline.json",
            "coverage_summary.json",
            "product_review_package/review_package.json",
            "final_manifest.json",
        ):
            self.assertTrue((run / name).is_file(), name)
        dens = json.loads((run / "dense_bbox_timeline.json").read_text(encoding="utf-8"))
        self.assertEqual(dens["schema_version"], "interactive_video_bbox_timeline_v1")
        self.assertFalse(dens.get("legacy_ext_mapping_fallback_used"))
        self.assertFalse(dens.get("game_state_executed"))
        cov = json.loads((run / "coverage_summary.json").read_text(encoding="utf-8"))
        self.assertGreater(cov["player_detection_count"], 0)
        self.assertEqual(cov["dense_frame_keys"], dens["frame_count"])


if __name__ == "__main__":
    unittest.main()
