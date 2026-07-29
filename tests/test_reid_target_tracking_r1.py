"""Tests for Target Tracking R1 persistent state + conservative stitching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.target_tracking_r1.candidates import (  # noqa: E402
    exact_frame_conflict,
    generate_continuation_candidates,
)
from football_analytics.reid.target_tracking_r1.policy import STITCH_POLICY  # noqa: E402
from football_analytics.reid.target_tracking_r1.state import (  # noqa: E402
    apply_human_seed,
    empty_persistent_state,
)
from football_analytics.reid.target_tracking_r1.stitch import decide_stitch  # noqa: E402
from football_analytics.reid.target_tracking_r1.timeline import build_target_timeline  # noqa: E402


class PersistentIdTests(unittest.TestCase):
    def test_persistent_id_independent_from_raw_track(self):
        st = empty_persistent_state(
            match_id="m",
            analysis_run_id="r",
            target_id="target_001",
            source_video_sha256="a" * 64,
        )
        st = apply_human_seed(
            st,
            raw_track_id="10",
            seed_frame=29,
            seed_time=0.97,
            segment_id="S",
            bbox_xyxy=[1, 2, 3, 4],
        )
        self.assertNotEqual(st["persistent_target_id"], "10")
        self.assertTrue(str(st["persistent_target_id"]).startswith("ptarget_"))
        self.assertEqual(st["seed"]["raw_track_id"], "10")
        self.assertEqual(st["observations"][0]["source"], "HUMAN_SEED")


class ConflictAndCandidateTests(unittest.TestCase):
    def test_exact_frame_conflict_and_no_overlap_merge(self):
        a = {"frames": {1, 2, 3}, "first_frame": 1, "last_frame": 3}
        b = {"frames": {3, 4}, "first_frame": 3, "last_frame": 4}
        self.assertTrue(exact_frame_conflict(a, b))
        c = {"frames": {10, 11}, "first_frame": 10, "last_frame": 11}
        self.assertFalse(exact_frame_conflict(a, c))

    def test_deterministic_candidate_generation(self):
        track_index = {
            "10": {
                "raw_track_id": "10",
                "first_frame": 0,
                "last_frame": 10,
                "observation_count": 11,
                "review_eligible": True,
                "frames": set(range(0, 11)),
                "segment_id": "S10",
            },
            "20": {
                "raw_track_id": "20",
                "first_frame": 14,
                "last_frame": 40,
                "observation_count": 27,
                "review_eligible": True,
                "frames": set(range(14, 41)),
                "segment_id": "S20",
            },
            "21": {
                "raw_track_id": "21",
                "first_frame": 15,
                "last_frame": 50,
                "observation_count": 36,
                "review_eligible": True,
                "frames": set(range(15, 51)),
                "segment_id": "S21",
            },
        }
        obs = {}
        for fi in range(0, 11):
            obs[str(fi)] = [{"raw_track_id": "10", "bbox_xyxy": [100, 100, 140, 180]}]
        for fi in range(14, 41):
            obs[str(fi)] = [{"raw_track_id": "20", "bbox_xyxy": [105, 102, 145, 182]}]
        for fi in range(15, 51):
            rows = list(obs.get(str(fi)) or [])
            rows.append({"raw_track_id": "21", "bbox_xyxy": [400, 400, 440, 480]})
            obs[str(fi)] = rows
        c1 = generate_continuation_candidates(
            previous_track_id="10",
            track_index=track_index,
            observations_by_frame=obs,
            frame_width=1326,
            frame_height=750,
        )
        c2 = generate_continuation_candidates(
            previous_track_id="10",
            track_index=track_index,
            observations_by_frame=obs,
            frame_width=1326,
            frame_height=750,
        )
        self.assertEqual(
            [x["candidate_raw_track_id"] for x in c1],
            [x["candidate_raw_track_id"] for x in c2],
        )
        self.assertGreaterEqual(len(c1), 1)
        self.assertEqual(c1[0]["candidate_raw_track_id"], "20")


class StitchGateTests(unittest.TestCase):
    def test_ambiguous_unresolved_and_long_gap(self):
        cands = [
            {
                "candidate_raw_track_id": "a",
                "temporal_gap_frames": 5,
                "center_displacement_px": 20,
                "cost": 0.2,
                "hard_gates_passed": True,
                "bbox_scale_ratio": 1.0,
            },
            {
                "candidate_raw_track_id": "b",
                "temporal_gap_frames": 6,
                "center_displacement_px": 22,
                "cost": 0.21,
                "hard_gates_passed": True,
                "bbox_scale_ratio": 1.0,
            },
        ]
        d = decide_stitch(cands)
        self.assertEqual(d["decision"], "TARGET_UNRESOLVED")
        self.assertIn("margin", d["reason"] or "" + str(d))

        clear = [
            {
                "candidate_raw_track_id": "a",
                "temporal_gap_frames": 5,
                "center_displacement_px": 20,
                "cost": 0.15,
                "hard_gates_passed": True,
                "bbox_scale_ratio": 1.0,
            },
            {
                "candidate_raw_track_id": "b",
                "temporal_gap_frames": 20,
                "center_displacement_px": 90,
                "cost": 0.55,
                "hard_gates_passed": True,
                "bbox_scale_ratio": 1.0,
            },
        ]
        d2 = decide_stitch(clear)
        self.assertEqual(d2["decision"], "AUTO_STITCH")
        self.assertEqual(d2["selected"]["candidate_raw_track_id"], "a")

    def test_unavailable_evidence_in_policy(self):
        self.assertEqual(STITCH_POLICY["appearance_evidence"], "UNAVAILABLE")
        self.assertTrue(STITCH_POLICY["exact_frame_conflict_hard_reject"])


class TimelineAndContractTests(unittest.TestCase):
    def test_multi_raw_track_timeline_and_no_gt_claim(self):
        track_index = {
            "10": {"first_frame": 0, "last_frame": 10, "segment_id": "S10"},
            "20": {"first_frame": 14, "last_frame": 40, "segment_id": "S20"},
        }
        events = [
            {
                "event_id": "e1",
                "previous_raw_track_id": "10",
                "decision": {
                    "decision": "AUTO_STITCH",
                    "selected": {
                        "candidate_raw_track_id": "20",
                        "candidate_start_frame": 14,
                        "temporal_gap_frames": 3,
                        "center_displacement_px": 10,
                        "cost": 0.1,
                        "bbox_scale_ratio": 1.0,
                    },
                    "margin": 0.3,
                },
            }
        ]
        tl = build_target_timeline(
            persistent_target_id="ptarget_x",
            target_id="target_001",
            seed_raw_track_id="10",
            chain_raw_track_ids=["10", "20"],
            track_index=track_index,
            stitch_events=events,
            fps=30.0,
        )
        kinds = {i["kind"] for i in tl["intervals"]}
        self.assertIn("HUMAN_CONFIRMED", kinds)
        self.assertIn("AUTO_STITCHED_CONTINUATION", kinds)
        self.assertEqual(
            tl["full_metrics"]["target_idf1"], "NOT_MEASURABLE_WITHOUT_ACCEPTED_GT"
        )
        self.assertEqual(tl["human_acceptance"], "HUMAN_VISUAL_ACCEPTANCE_PENDING")

    def test_runner_has_no_ui_or_rerun(self):
        script = (_PROJECT / "scripts/run_target_tracking_r1_stitch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("No annotation UI", script)
        self.assertIn('"detection_rerun": False', script)
        self.assertIn('"tracking_rerun": False', script)
        self.assertNotIn("streamlit", script.lower())
        self.assertNotIn("declare_component", script)


if __name__ == "__main__":
    unittest.main()
