"""Unit tests for Stage 5D-F3B retrieval-error manual review and refinement design."""

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

import run_reid_retrieval_error_manual_review_and_refinement_design as f3b  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/retrieval_error_manual_review_and_refinement_design_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_retrieval_error_manual_review_and_refinement_design"
)
_HEAD = "1385b5ba6f82b6d48d4917b3724523388261486b"


class RetrievalErrorManualReviewRefinementDesignTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3b.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3b.HIGH_POSITIVES), 3)
        self.assertEqual(len(f3b.VISIBLE_JERSEY), 12)
        self.assertEqual(len(f3b.CONFLICT_FINDINGS), 4)
        self.assertTrue(cfg["policy"]["new_independent_holdout_required"])
        self.assertFalse(cfg["policy"]["refinement_applied"])

    def test_visible_jersey_and_conflicts(self):
        self.assertEqual(f3b.VISIBLE_JERSEY["SAMPLE_EVAL_111"], "20")
        self.assertEqual(f3b.VISIBLE_JERSEY["SAMPLE_EVAL_127"], "3")
        ids = [c["evaluation_component_id"] for c in f3b.CONFLICT_FINDINGS]
        self.assertEqual(
            ids,
            [
                "SAMPLE_COMPONENT_005",
                "SAMPLE_COMPONENT_018",
                "SAMPLE_COMPONENT_032",
                "SAMPLE_COMPONENT_055",
            ],
        )
        for row in f3b.CONFLICT_FINDINGS:
            self.assertEqual(row["cause"], "component_grouping_overmerge_candidate")

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3b.RefinementDesignError):
            f3b.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3b.RefinementDesignError):
                f3b.atomic_publish(tmp, final)

    def test_manual_findings_builder_with_live_fp(self):
        f3a = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_retrieval_error_analysis_and_diagnostics"
        )
        if not f3a.is_dir():
            self.skipTest("F3A absent")
        fp = f3b.load_jsonl(
            f3a / "analysis" / "target_001_top_false_positive_analysis.jsonl"
        )
        findings = f3b.build_manual_findings(fp)
        self.assertEqual(len(findings["top_false_positives_exact_codes"]), 24)
        self.assertEqual(len(findings["false_positive_findings"]), 24)
        self.assertEqual(len(findings["high_ranked_positives"]), 3)
        self.assertEqual(len(findings["low_ranked_positives"]), 5)
        for a in findings["gallery_anchor_manual_findings"]:
            self.assertFalse(a["removal_authorized"])
        visible = [
            r
            for r in findings["false_positive_findings"]
            if r.get("visible_jersey_number") not in (None, "unknown")
        ]
        self.assertEqual(len(visible), 12)
        for r in visible:
            self.assertEqual(
                r["false_positive_root_cause"], "same_uniform_confusion_candidate"
            )

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3B root absent")
        s = json.loads((_FINAL / "stage5d_f3b_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["final_status"], f3b.FINAL_STATUS)
        self.assertTrue(s["official_f3_outcome_unchanged"])
        self.assertTrue(s["reviewed_diagnostic_findings_frozen"])
        self.assertTrue(s["external_only_refinement_design_ready"])
        self.assertFalse(s["refinement_applied"])
        self.assertFalse(s["gallery_mutation"])
        self.assertEqual(s["gallery_members"], 7)
        self.assertEqual(s["new_embeddings"], 0)
        self.assertFalse(s["threshold_selected"])
        self.assertEqual(s["identity_assignments"], 0)
        self.assertTrue(s["new_independent_holdout_required"])
        plan = json.loads(
            (
                _FINAL
                / "refinement_design"
                / "target_001_external_only_refinement_plan.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(plan["applied_in_f3b"])
        self.assertEqual(len(plan["priorities"]), 5)
        hard = json.loads(
            (
                _FINAL
                / "refinement_design"
                / "target_001_hard_negative_gallery_design.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(hard["sample_crops_forbidden"])
        policy = json.loads(
            (
                _FINAL / "validation_policy" / "target_001_anti_overfit_policy.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(policy["sample_crops_forbidden_for_gallery_enrollment"])
        self.assertTrue(policy["new_independent_holdout_required"])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(len(list(_FINAL.rglob("*.csv"))), 1)

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3B root absent")
        with mock.patch.object(f3b, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3b.RefinementDesignError):
                f3b.run(_CFG)


if __name__ == "__main__":
    unittest.main()
