"""Tests for short-video tracking stabilization audit/compare/one-click contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.short_video.one_click import rank_recovery_candidates  # noqa: E402
from football_analytics.reid.short_video.tracker_compare import (  # noqa: E402
    bounded_variants,
    select_best,
)
from football_analytics.reid.short_video.tracking_audit import continuity_probe  # noqa: E402


class VariantSweepTests(unittest.TestCase):
    def test_bounded_variants_not_random(self):
        variants = bounded_variants()
        ids = [v["variant_id"] for v in variants]
        self.assertIn("A_current_bytetrack", ids)
        self.assertTrue(any(v.startswith("B") for v in ids))
        self.assertIn("C_botsort_gmc_sparseOptFlow", ids)
        # BoT-SORT must disable appearance ReID auto-confirm
        c = next(v for v in variants if v["variant_id"].startswith("C_"))
        self.assertFalse(c["params"]["with_reid"])


class ContinuityProbeTests(unittest.TestCase):
    def test_probe_follows_seed_track(self):
        obs = []
        for fi in range(0, 10):
            obs.append(
                {
                    "frame_index": fi,
                    "raw_track_id": 7,
                    "bbox_xyxy": [10 + fi, 10, 40 + fi, 80],
                }
            )
        # distractor
        obs.append({"frame_index": 5, "raw_track_id": 9, "bbox_xyxy": [200, 200, 240, 280]})
        out = continuity_probe(obs, seed_bbox=[10, 10, 40, 80], seed_frame=0, fps=10.0)
        self.assertEqual(out["matched_raw_track_id"], 7)
        self.assertGreaterEqual(out["uninterrupted_duration_sec"], 0.9)


class RecoveryRankTests(unittest.TestCase):
    def test_rank_does_not_drop_candidates(self):
        cands = [
            {
                "candidate_id": "a",
                "start_frame": 100,
                "display_order": 2,
                "bbox_references": [{"bbox_xyxy": [10, 10, 30, 50]}],
                "team_evidence": {"team_label": "unknown"},
            },
            {
                "candidate_id": "b",
                "start_frame": 200,
                "display_order": 1,
                "bbox_references": [{"bbox_xyxy": [500, 500, 520, 540]}],
                "team_evidence": {"team_label": "red"},
            },
        ]
        ranked = rank_recovery_candidates(
            cands, lost_frame=90, previous_bbox=[12, 12, 32, 52], fps=30.0
        )
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(r["auto_confirm_forbidden"] for r in ranked))
        self.assertEqual(ranked[0]["candidate_id"], "a")


class SelectionSafetyTests(unittest.TestCase):
    def test_select_best_prefers_continuity_when_deterministic(self):
        results = [
            {
                "variant_id": "A_current_bytetrack",
                "determinism_ok": True,
                "continuity_probe": {"uninterrupted_duration_sec": 1.0},
                "short_tracks_lt_1s": 300,
                "fragmentation_index": 10.0,
                "runtime_sec": 1.0,
            },
            {
                "variant_id": "B2_buffer90_match065_new04",
                "determinism_ok": True,
                "continuity_probe": {"uninterrupted_duration_sec": 5.0},
                "short_tracks_lt_1s": 100,
                "fragmentation_index": 4.0,
                "runtime_sec": 1.2,
            },
        ]
        sel = select_best(results)
        self.assertEqual(sel["product_candidate_variant_id"], "B2_buffer90_match065_new04")


class FrontendPauseContractTests(unittest.TestCase):
    def test_frontend_pauses_on_click(self):
        html = (
            _SRC
            / "football_analytics/reid/hil_ui/interactive_video_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("vid.pause()", html)
        self.assertIn("TARGET LOST — RECOVERY REQUIRED", html)
        self.assertIn("target_lost", html)


if __name__ == "__main__":
    unittest.main()
