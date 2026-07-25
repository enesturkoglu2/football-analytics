"""Unit tests for Stage 5D-B1 temporal context review helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_target_temporal_context_review_package as tcr  # noqa: E402


class TemporalContextReviewTests(unittest.TestCase):
    def test_expected_git_head_and_candidates(self):
        cfg = tcr.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_temporal_context_review_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "0f24ec7ab1df81f6681b87f901fc458c6cdc62d5",
        )
        self.assertEqual(
            cfg["stage5d_b_package"]["expected_candidate_ids"],
            tcr.EXPECTED_CANDIDATE_IDS,
        )
        self.assertEqual(len(tcr.EXPECTED_CANDIDATE_IDS), 9)

    def test_context_window_clamped_to_segment(self):
        w = tcr.compute_context_window(
            segment_start=10,
            segment_end=20,
            representative=15,
            max_before=60,
            max_after=60,
        )
        self.assertEqual(w["selected_context_start_frame"], 10)
        self.assertEqual(w["selected_context_end_frame"], 20)
        self.assertTrue(w["truncated_left"])
        self.assertTrue(w["truncated_right"])

    def test_context_window_untruncated_when_room(self):
        w = tcr.compute_context_window(
            segment_start=0,
            segment_end=500,
            representative=200,
            max_before=60,
            max_after=60,
        )
        self.assertEqual(w["selected_context_start_frame"], 140)
        self.assertEqual(w["selected_context_end_frame"], 260)
        self.assertFalse(w["truncated_left"])
        self.assertFalse(w["truncated_right"])

    def test_deterministic_frame_selection_and_collapse(self):
        frames = [10, 20, 30, 40, 50]
        selected = tcr.select_sheet_observation_frames(
            frames,
            representative=30,
            ctx_start=10,
            ctx_end=50,
            offsets=[-40, -20, 0, 20, 40],
        )
        self.assertEqual(selected, sorted(selected))
        self.assertLessEqual(len(selected), 7)
        self.assertIn(30, selected)
        # Short segment collapse: single observation.
        one = tcr.select_sheet_observation_frames(
            [5],
            representative=5,
            ctx_start=5,
            ctx_end=5,
            offsets=[-40, -20, 0, 20, 40],
        )
        self.assertEqual(one, [5])

    def test_no_bbox_interpolation_contract(self):
        c = tcr.build_contract(zoom_padding_ratio=0.15)
        self.assertTrue(c["no_new_detection"])
        self.assertTrue(c["no_new_tracking"])
        self.assertTrue(c["no_new_embedding"])
        self.assertTrue(c["no_ocr"])
        self.assertTrue(c["no_similarity"])
        self.assertTrue(c["bbox_interpolation_forbidden"])
        self.assertTrue(c["no_gallery_membership"])
        self.assertEqual(c["manual_decisions"], 0)
        self.assertEqual(c["zoom_padding_ratio"], 0.15)

    def test_allowed_manual_vocabulary(self):
        self.assertIn("target_anchor_yes", tcr.ALLOWED_TEMPORAL_DECISIONS)
        self.assertEqual(set(tcr.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})

    def test_blank_template_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tcr.TEMPLATE_FIELDS))
                writer.writeheader()
                writer.writerow(
                    {
                        "anchor_candidate_id": "target_001_anchor_001",
                        "segment_id": "raw_x",
                        "raw_track_id": 1,
                        "representative_frame": 10,
                        "context_start_frame": 1,
                        "context_end_frame": 20,
                        "temporal_review_decision": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_identity_continuity_observed": "",
                        "manual_human_verified_number_seen_in_context": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["temporal_review_decision"], "")
            self.assertNotIn("similarity_score", row)
            self.assertNotIn("ocr_prediction", row)
            self.assertNotIn("model_identity_prediction", row)

    def test_path_traversal_rejection(self):
        with self.assertRaises(tcr.TemporalContextError):
            tcr.assert_no_path_traversal("../outside")

    def test_zoom_padding_visualization_only(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 30:70] = 200
        zoom = tcr.zoom_from_bbox(frame, [30, 20, 70, 80], padding_ratio=0.15)
        self.assertGreater(zoom.shape[0], 0)
        self.assertGreater(zoom.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
