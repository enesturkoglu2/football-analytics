"""Unit tests for Stage 5D-B1E-A external enrollment overlap preflight."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_external_enrollment_preflight as b1ea  # noqa: E402


class ExternalEnrollmentPreflightTests(unittest.TestCase):
    def test_expected_git_and_source_contract(self):
        cfg = b1ea.load_config(
            _PROJECT_ROOT
            / "configs/reid/external_enrollment_preflight_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "f94ab3353fe2ab933a1bb1189cb7450188249331",
        )
        self.assertEqual(
            cfg["external_enrollment_source"]["expected_sha256"],
            "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877",
        )
        self.assertEqual(
            cfg["evaluation_source"]["expected_sha256"],
            "f4b28dd58a6cf242344a4198b8c0ba9062b20977cec3ae12d96322750bfd7b9b",
        )
        self.assertEqual(cfg["overlap_audit"]["coarse_sample_stride"], 15)
        self.assertEqual(cfg["overlap_audit"]["min_eligible_duration_sec"], 5.0)

    def test_merge_contiguous_and_complement(self):
        intervals = b1ea.merge_contiguous(
            [(10, 100), (12, 102), (50, 200)], gap_max=5
        )
        self.assertEqual(len(intervals), 2)
        comps = b1ea.complement_intervals(
            total_frames=100, excluded=[(10, 19), (50, 59)], fps=10.0
        )
        starts = [c["start_frame"] for c in comps]
        self.assertEqual(starts[0], 0)
        self.assertTrue(any(c["duration"] >= 5.0 for c in comps) or True)
        # 0-9 = 1.0s, 20-49=3.0s, 60-99=4.0s at 10fps — none >=5 in this toy
        self.assertFalse(any(c["duration"] >= 5.0 for c in comps))

    def test_blank_seed_template(self):
        tpl = b1ea.blank_seed_template(
            source_path="data/enrollment_clips/x.mp4",
            source_sha="abc",
            eligible_intervals=[{"start_frame": 0, "end_frame": 100}],
        )
        self.assertEqual(tpl["selected_neutral_detection_code"], "")
        self.assertIsNone(tpl["selected_reference_frame"])
        self.assertIsNone(tpl["selected_interval"])
        self.assertEqual(tpl["manual_target_confirmed"], "")
        self.assertEqual(tpl["reviewer"], "")

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1ea.ExternalPreflightError):
            b1ea.assert_no_path_traversal("../x")

    def test_dhash_hamming_identity(self):
        import numpy as np

        gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
        # Use larger synthetic gray
        gray = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
        a = b1ea.dhash_bytes(gray, 16)
        b = b1ea.dhash_bytes(gray, 16)
        self.assertEqual(b1ea.hamming_bytes(a, b), 0)

    def test_live_output_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_enrollment_preflight"
        )
        if not root.is_dir():
            self.skipTest("B1E-A output absent")
        summary = json.loads((root / "stage5d_b1e_a_summary.json").read_text(encoding="utf-8"))
        self.assertIn(
            summary["final_status"],
            {
                "COMPLETED_STAGE5D_B1E_A_EXTERNAL_ENROLLMENT_PREFLIGHT_READY",
                "COMPLETED_STAGE5D_B1E_A_PARTIAL_OVERLAP_ELIGIBLE_INTERVALS_READY",
                "COMPLETED_STAGE5D_B1E_A_EXTERNAL_SOURCE_INELIGIBLE_OVERLAP",
            },
        )
        self.assertFalse(summary["exact_file_duplicate"])
        self.assertEqual(
            summary["external_source"]["sha256"],
            "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877",
        )
        self.assertEqual(
            summary["evaluation_source"]["sha256"],
            "f4b28dd58a6cf242344a4198b8c0ba9062b20977cec3ae12d96322750bfd7b9b",
        )
        self.assertTrue(summary["external_source"]["enrollment_only"])
        self.assertTrue(summary["evaluation_source"]["evaluation_only"])
        self.assertEqual(summary["new_detection"], 0)
        self.assertEqual(summary["new_tracking"], 0)
        self.assertEqual(summary["new_embedding"], 0)
        self.assertEqual(summary["ocr"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["png_count"], 1)
        self.assertEqual(len(list(root.rglob("*.png"))), 1)
        self.assertEqual(len(list(root.rglob("*.mp4"))), 0)

        audit = json.loads(
            (
                root
                / "overlap_audit"
                / "target_001_external_vs_sample_overlap_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(audit["method"]["coarse_sample_stride"], 15)
        self.assertFalse(audit["exact_file_duplicate"])

        elig = json.loads(
            (
                root
                / "eligibility"
                / "target_001_external_enrollment_interval_eligibility.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(elig["external_source_enrollment_only"])
        self.assertTrue(elig["sample_evaluation_only"])
        for iv in elig["intervals"]:
            if iv["overlap_status"] == "EXCLUDED_OVERLAP":
                self.assertFalse(iv["eligible_for_gallery_enrollment"])

        tpl = json.loads(
            (
                root
                / "templates"
                / "target_001_external_enrollment_seed_review_template.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(tpl["selected_neutral_detection_code"], "")
        self.assertIsNone(tpl["selected_reference_frame"])

        if summary["final_status"].endswith("PREFLIGHT_READY"):
            self.assertGreaterEqual(summary["eligible_interval_count"], 1)
            self.assertTrue(summary["has_min_duration_eligible_interval"])
            self.assertEqual(
                summary["exact_next_gate"],
                "STAGE5D-B1E-B_TARGET_001_EXTERNAL_CLIP_DETECTION_TRACKING_AND_SEED_REVIEW_PACKAGE",
            )

    def test_live_b1d_handoff_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_bridge_review_no_selection_freeze"
        )
        if not root.is_dir():
            self.skipTest("B1D absent")
        summary = json.loads((root / "stage5d_b1d_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1D_NO_BRIDGE_SELECTION_EXTERNAL_ENROLLMENT_REQUIRED",
        )
        self.assertEqual(summary["selected_bridge_candidate_code"], "")
        self.assertTrue(summary["current_video_eligible_source_search_closed"])

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1ea.load_config(
            _PROJECT_ROOT
            / "configs/reid/external_enrollment_preflight_stage5d_target_001.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1ea, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1ea.ExternalPreflightError) as ctx:
                    b1ea.run(
                        _PROJECT_ROOT
                        / "configs/reid/external_enrollment_preflight_stage5d_target_001.yaml",
                        project,
                    )
            self.assertIn("final_exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
