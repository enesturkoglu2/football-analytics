"""Tests for Target Tracking R2 purity split + kit-guarded stitching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.target_tracking_r2.changepoint import (  # noqa: E402
    detect_kit_change_points,
)
from football_analytics.reid.target_tracking_r2.evidence import classify_kit_state  # noqa: E402
from football_analytics.reid.target_tracking_r2.policy import R2_POLICY  # noqa: E402
from football_analytics.reid.target_tracking_r2.review import (  # noqa: E402
    build_r1_rejection_record,
)
from football_analytics.reid.target_tracking_r2.segments import (  # noqa: E402
    build_derived_segments,
)
from football_analytics.reid.target_tracking_r2.stitch import (  # noqa: E402
    decide_segment_stitch,
)


class ReviewRecordTests(unittest.TestCase):
    def test_r1_rejection_record(self):
        rec = build_r1_rejection_record(
            match_id="m",
            analysis_run_id="r",
            target_id="target_001",
            persistent_target_id_r1="ptarget_x",
            r1_chain=["10", "365"],
        )
        self.assertEqual(rec["result"], "REJECTED")
        self.assertEqual(rec["r1_correctness"], "NOT_ACCEPTED")
        self.assertIn("CROSS_TEAM_IDENTITY_SWITCH", rec["observed_errors"])
        self.assertTrue(rec["r1_artifacts_not_overwritten"])


class KitAndChangePointTests(unittest.TestCase):
    def test_unknown_kit_when_unreliable(self):
        out = classify_kit_state(None, quality_ok=False)
        self.assertEqual(out["kit_state"], "UNKNOWN")

    def test_no_one_frame_noisy_split(self):
        rows = []
        for fi in range(0, 80):
            if fi < 50:
                kit, y, w = "YELLOW", 0.35, 0.05
            elif fi == 50:
                kit, y, w = "WHITE", 0.02, 0.4  # single frame spike
            else:
                kit, y, w = "YELLOW", 0.3, 0.05
            rows.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": fi / 30.0,
                    "kit_state": kit,
                    "yellow_evidence": y,
                    "white_evidence": w,
                    "crop_quality_ok": True,
                    "reliable": True,
                    "contamination": {"union_other_person_crop_coverage": 0.2},
                    "nearby_player_overlap": 1,
                }
            )
        cp = detect_kit_change_points(rows, seed_frame=29)
        # single-frame white must not create change point
        self.assertIsNone(cp["algorithmic_change_point"])

    def test_sustained_cross_team_change(self):
        rows = []
        for fi in range(0, 100):
            if fi < 48:
                kit, y, w = "YELLOW", 0.35, 0.04
            else:
                kit, y, w = "WHITE", 0.02, 0.45
            rows.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": fi / 30.0,
                    "kit_state": kit,
                    "yellow_evidence": y,
                    "white_evidence": w,
                    "crop_quality_ok": True,
                    "reliable": True,
                    "contamination": {"union_other_person_crop_coverage": 0.15},
                    "nearby_player_overlap": 1,
                }
            )
        cp = detect_kit_change_points(rows, seed_frame=29)
        self.assertIsNotNone(cp["algorithmic_change_point"])
        self.assertGreaterEqual(cp["algorithmic_change_point"]["change_point_frame"], 48)
        self.assertFalse(cp["human_reported_transition_is_algorithm_truth"])


class SegmentTests(unittest.TestCase):
    def test_immutable_parent_and_seed_segment(self):
        cp = {
            "change_point_frame": 48,
            "change_point_time_sec": 1.6,
            "kind": "YELLOW_TO_WHITE_CROSS_TEAM",
        }
        derived = build_derived_segments(
            parent_raw_track_id="10",
            parent_first_frame=0,
            parent_last_frame=312,
            seed_frame=29,
            change_point=cp,
            evidence_rows=[],
            fps=30.0,
        )
        self.assertTrue(derived["parent_immutable"])
        seed = next(s for s in derived["segments"] if s["target_eligibility"] == "TARGET_SEED_SEGMENT")
        conflict = next(
            s for s in derived["segments"] if s["target_eligibility"] == "TARGET_INELIGIBLE_IDENTITY_CONFLICT"
        )
        self.assertTrue(seed["contains_seed_frame"])
        self.assertFalse(conflict.get("analysis_eligible", True))
        self.assertEqual(conflict["reason"], "CROSS_TEAM_IDENTITY_CONFLICT")
        self.assertLess(seed["end_frame"], conflict["start_frame"])


class StitchGuardTests(unittest.TestCase):
    def test_cross_team_hard_rejection_and_unknown_ok(self):
        white = {
            "candidate_raw_track_id": "99",
            "temporal_gap_frames": 5,
            "center_displacement_px": 20,
            "cost": 0.1,
            "hard_gates_passed": False,
            "hard_rejects": ["CROSS_TEAM_KIT_MISMATCH"],
            "kit": {"kit_state": "WHITE", "reliable": True},
        }
        unknown = {
            "candidate_raw_track_id": "88",
            "temporal_gap_frames": 5,
            "center_displacement_px": 25,
            "cost": 0.2,
            "hard_gates_passed": True,
            "hard_rejects": [],
            "kit": {"kit_state": "UNKNOWN", "reliable": False},
        }
        yellow = {
            "candidate_raw_track_id": "77",
            "temporal_gap_frames": 4,
            "center_displacement_px": 15,
            "cost": 0.12,
            "hard_gates_passed": True,
            "hard_rejects": [],
            "kit": {"kit_state": "YELLOW", "reliable": True},
        }
        # only cross-team rejected → unresolved
        d = decide_segment_stitch([white])
        self.assertEqual(d["decision"], "TARGET_UNRESOLVED")
        self.assertGreaterEqual(d["rejected_cross_team_count"], 1)
        # unknown kit alone may stitch (does not hard-reject)
        d_u = decide_segment_stitch([unknown])
        self.assertEqual(d_u["decision"], "AUTO_STITCH")
        # clear yellow with margin over weak unknown
        d2 = decide_segment_stitch(
            [
                yellow,
                {
                    **unknown,
                    "cost": 0.5,
                    "center_displacement_px": 90,
                    "temporal_gap_frames": 20,
                },
            ]
        )
        self.assertEqual(d2["decision"], "AUTO_STITCH")
        self.assertEqual(d2["selected"]["candidate_raw_track_id"], "77")
        self.assertTrue(R2_POLICY["unknown_kit_does_not_hard_reject"])
        self.assertTrue(R2_POLICY["cross_team_hard_reject"])


class ContractTests(unittest.TestCase):
    def test_runner_contract(self):
        script = (
            _PROJECT / "scripts/run_target_tracking_r2_purity_split.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Does NOT overwrite R1", script)
        self.assertIn('"detection_rerun": False', script)
        self.assertIn('"tracking_rerun": False', script)
        self.assertNotIn("streamlit", script.lower())
        self.assertIn("NOT_MEASURABLE_WITHOUT_ACCEPTED_GT", script)
        self.assertIn(
            "change_confirm_frames",
            (_SRC / "football_analytics/reid/target_tracking_r2/policy.py").read_text(
                encoding="utf-8"
            ),
        )


if __name__ == "__main__":
    unittest.main()
