"""Unit tests for Stage 5D-B1A manual seed selection helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_target_manual_seed_selection_package as mss  # noqa: E402


class ManualSeedSelectionTests(unittest.TestCase):
    def test_expected_git_and_seed_window(self):
        cfg = mss.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_manual_seed_selection_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "d6ef4af989e05f961bc6b81dbdc9540bc13f60ad",
        )
        self.assertEqual(cfg["seed_window"]["representative_frame"], 290)
        self.assertEqual(cfg["seed_window"]["start_frame"], 280)
        self.assertEqual(cfg["seed_window"]["end_frame"], 310)

    def test_neutral_codes_stable_and_deterministic(self):
        chains = {
            (10, "raw_10_full"): [
                {"frame_index": 282},
                {"frame_index": 290},
            ],
            (5, "raw_5_full"): [{"frame_index": 280}],
            (10, "raw_10_s02"): [{"frame_index": 300}],
        }
        a = mss.assign_neutral_seed_codes(chains)
        b = mss.assign_neutral_seed_codes(chains)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertTrue(all(v.startswith("SEED_CANDIDATE_") for v in a.values()))
        # Same track different segment => different codes.
        self.assertNotEqual(a[(10, "raw_10_full")], a[(10, "raw_10_s02")])

    def test_sheet_frame_selection_collapse(self):
        selected = mss.select_sheet_frames(
            [280, 285, 290, 295, 300, 305, 310],
            [280, 290, 310],
        )
        self.assertEqual(selected, [280, 290, 310])
        self.assertEqual(len(selected), len(set(selected)))

    def test_contract_forbids_auto_and_gallery(self):
        c = mss.build_contract()
        self.assertTrue(c["human_click_box_selection_required"])
        self.assertTrue(c["no_new_detection"])
        self.assertTrue(c["no_new_tracking"])
        self.assertTrue(c["no_ocr"])
        self.assertTrue(c["no_similarity"])
        self.assertTrue(c["no_automatic_selection"])
        self.assertTrue(c["no_gallery_membership"])
        self.assertTrue(c["no_anchor_freeze"])
        self.assertEqual(c["manual_selection"], 0)
        self.assertIn("STAGE5D-B1B", c["exact_next_gate"])

    def test_blank_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(mss.TEMPLATE_FIELDS))
                writer.writeheader()
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "representative_frame": 290,
                        "selected_neutral_seed_code": "",
                        "manual_target_confirmed": "",
                        "manual_human_verified_number_seen": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["selected_neutral_seed_code"], "")
            self.assertNotIn("similarity_score", row)
            self.assertNotIn("ocr_prediction", row)
            self.assertNotIn("raw_track_id", row)

    def test_path_traversal_rejection(self):
        with self.assertRaises(mss.ManualSeedError):
            mss.assert_no_path_traversal("../x")

    def test_tristate_vocabulary(self):
        self.assertEqual(set(mss.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})


if __name__ == "__main__":
    unittest.main()
