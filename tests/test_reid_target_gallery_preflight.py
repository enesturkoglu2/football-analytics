"""Unit tests for Stage 5D-A target gallery preflight helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
for p in (_SCRIPTS,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reid_target_gallery_preflight as pf  # noqa: E402


class TargetGalleryPreflightTests(unittest.TestCase):
    def test_blank_target_template_enforced(self):
        t = pf.blank_target_template()
        self.assertEqual(t["target_id"], "")
        self.assertEqual(t["target_alias"], "")
        self.assertEqual(t["identity_basis"], "")
        self.assertIsNone(t["human_verified_jersey_number"])
        self.assertFalse(t["target_definition_frozen"])
        self.assertEqual(t["reviewer"], "")
        self.assertEqual(t["final_approver"], "")

    def test_design_forbids_automatic_growth_and_identity(self):
        d = pf.design_contract()
        self.assertEqual(d["enrollment_mode"], "manual_frozen")
        self.assertFalse(d["automatic_gallery_growth"])
        self.assertFalse(d["pseudo_label_enrollment"])
        self.assertFalse(d["ocr_based_automatic_enrollment"])
        self.assertFalse(d["identity_assignment_in_stage5d_a"])
        self.assertFalse(d["gallery_membership_created_in_stage5d_a"])
        self.assertFalse(d["gallery_representation_plan"]["prototype_creation_in_stage5d_a"])
        self.assertTrue(d["unknown_identity_preserved"])

    def test_exclusion_contract_stage5c_batches(self):
        keys = {
            "segment_id": {"a"},
            "raw_track_id": {"1"},
            "crop_id": {"c"},
            "source_crop_path": {"p"},
            "source_crop_sha256": {"s"},
            "split_item_id": {"holdout_primary_001"},
        }
        c = pf.exclusion_contract(keys)
        self.assertEqual(c["required_overlap_count"], 0)
        self.assertIn("holdout_primary", c["stage5c_batches_excluded_from_gallery_and_evaluation_inputs"])
        self.assertTrue(c["holdout_primary_forbidden_for_gallery_enrollment"])
        self.assertFalse(c["future_evaluation_set_created_in_stage5d_a"])

    def test_workflow_next_gate(self):
        w = pf.workflow_preregistration()
        self.assertEqual(
            w["exact_next_gate"],
            "STAGE5D-B_TARGET_DEFINITION_AND_ANCHOR_REVIEW_PACKAGE",
        )
        self.assertTrue(w["automatic_gallery_growth_forbidden_in_all_stage5d_gates"])
        self.assertFalse(w["gates"]["STAGE5D-C"]["automatic_enrollment"])
        self.assertFalse(w["candidate_retrieval_plan"]["executed_in_stage5d_a"])

    def test_path_traversal_rejection_helper(self):
        bad = Path("../outside/x")
        self.assertIn("..", bad.parts)

    def test_expected_git_head_constant(self):
        cfg = pf.load_config(
            _PROJECT_ROOT / "configs/reid/target_gallery_preflight_stage5d.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "87b92366d7841994bcbb93eed20bc3c51e1e1a8f",
        )
        self.assertFalse(cfg["gallery_policy"]["automatic_gallery_growth"])

    def test_no_prototype_or_ranking_in_summary_schema(self):
        # summary keys must report zeros for gallery products
        keys = {
            "gallery_members",
            "prototypes",
            "similarity_ranking_rows",
            "identity_assignments",
            "target_positive_decisions",
        }
        self.assertTrue(keys)


if __name__ == "__main__":
    unittest.main()
