"""Unit tests for Stage 5D-F3H scoring contract and holdout design."""

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

import run_reid_target_distractor_scoring_contract_and_holdout_design as f3h  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/target_distractor_scoring_contract_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_target_distractor_scoring_contract_and_holdout_design"
)
_HEAD = "0e467c1b7b7203f4e52d186c93eadbafb20c3f9e"
_F3G_SNAP = "dafb77147ce0c7a72b9eed43ba4b9223f1150709da667f655116b8249644f28c"


class TargetDistractorScoringContractTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3h.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(f3h.PRIMARY_FORMULA, "TARGET_DISTRACTOR_MAX_MARGIN")
        self.assertEqual(len(f3h.GT_VOCAB), 6)

    def test_primary_formula_contract(self):
        primary = f3h.build_primary_contract()
        self.assertTrue(primary["preregistered"])
        self.assertEqual(primary["formula_id"], "TARGET_DISTRACTOR_MAX_MARGIN")
        self.assertEqual(primary["T_max"]["aggregation"], "max")
        self.assertEqual(primary["D_max"]["aggregation"], "max")
        self.assertEqual(primary["T_max"]["target_top_k"], 1)
        self.assertEqual(primary["D_max"]["distractor_top_k"], 1)
        self.assertEqual(primary["S_primary"]["formula"], "T_max(q) - D_max(q)")
        self.assertEqual(
            primary["S_primary"]["subtraction_order"], "target_minus_distractor"
        )
        self.assertFalse(primary["T_max"]["member_weighting"])
        self.assertIn("jersey_metadata", primary["forbidden_in_score"])
        self.assertEqual(
            primary["tie_break"],
            [
                "primary_score_descending",
                "T_max_descending",
                "D_max_ascending",
                "query_stable_id_ascending",
            ],
        )

    def test_secondary_top3_and_formulas(self):
        secondary = f3h.build_secondary_contract()
        self.assertTrue(secondary["preregistered"])
        self.assertTrue(secondary["diagnostic_only"])
        self.assertEqual(secondary["formulas"]["S_top3_margin"]["k"], 3)
        self.assertTrue(secondary["formulas"]["S_top3_margin"]["k_frozen"])
        self.assertIn("S_target_centroid_margin", secondary["formulas"])
        self.assertIn("S_target_medoid_margin", secondary["formulas"])
        self.assertIn("S_mean_margin", secondary["formulas"])
        self.assertTrue(secondary["cannot_replace_primary_on_same_holdout"])

    def test_gt_metric_outcome_policies(self):
        gt = f3h.build_gt_policy()
        self.assertEqual(gt["clean_positive"], ["target_occurrence_yes"])
        self.assertEqual(
            set(gt["clean_negative"]), {"target_occurrence_no", "non_player"}
        )
        self.assertTrue(gt["unreviewed_is_not_negative"])
        self.assertIn("uncertain", gt["metric_excluded"])
        metrics = f3h.build_metric_contract()
        self.assertEqual(metrics["minimum_support"]["segment_clean_positive_ge"], 5)
        self.assertEqual(metrics["minimum_support"]["segment_clean_negative_ge"], 20)
        self.assertTrue(
            metrics["recall_at_k_ceiling_policy"][
                "if_positive_count_gt_k_report_mathematical_ceiling"
            ]
        )
        outcomes = f3h.build_outcome_rules()
        self.assertIn(
            "INDEPENDENT_TARGET_DISTRACTOR_STRONG_SIGNAL", outcomes["outcomes"]
        )
        self.assertFalse(outcomes["threshold_and_abstention"]["threshold_selected"])
        self.assertEqual(
            outcomes["threshold_and_abstention"]["threshold_candidate_count"], 0
        )

    def test_holdout_requirements_forbid_sample_external(self):
        cfg = f3h.load_config(_CFG)
        req = f3h.build_holdout_requirements(cfg, pending=True)
        self.assertTrue(req["new_independent_holdout_required"])
        self.assertTrue(req["holdout_input_pending"])
        self.assertTrue(req["missing_file_is_not_blocker_in_f3h"])
        self.assertIn("sample.mp4", req["forbidden_as_holdout"])
        self.assertIn(
            "data/test_clips/sample.mp4", req["forbidden_source_paths"]
        )
        self.assertIn(
            "data/enrollment_clips/target_001_external_enrollment_v1.mp4",
            req["forbidden_source_paths"],
        )

    def test_f3g_validation_if_present(self):
        cfg = f3h.load_config(_CFG)
        root = _PROJECT_ROOT / cfg["stage5d_f3g_package"]["path"]
        if not root.is_dir():
            self.skipTest("F3G absent")
        out = f3h.validate_f3g(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3G_SNAP)
        self.assertEqual(out["summary"]["target_gallery_v2_members"], 13)
        self.assertEqual(out["summary"]["distractor_members"], 23)
        self.assertEqual(
            out["target_manifest"]["shape"], [13, 512]
        )
        self.assertEqual(
            out["distractor_manifest"]["shape"], [23, 512]
        )

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3h.ScoringDesignError):
            f3h.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3h.ScoringDesignError):
                f3h.atomic_publish(tmp, final)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3H root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3h_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3h.FINAL_STATUS)
        self.assertEqual(summary["exact_next_gate"], f3h.NEXT_GATE)
        self.assertEqual(summary["target_gallery_v2_members"], 13)
        self.assertEqual(summary["distractor_gallery_v1_members"], 23)
        self.assertEqual(summary["primary_formula"], f3h.PRIMARY_FORMULA)
        self.assertEqual(summary["primary_target_top_k"], 1)
        self.assertEqual(summary["primary_distractor_top_k"], 1)
        self.assertEqual(summary["secondary_top_k"], 3)
        self.assertTrue(summary["primary_preregistered"])
        self.assertTrue(summary["secondary_preregistered"])
        self.assertEqual(summary["query_scoring_rows"], 0)
        self.assertEqual(summary["rankings"], 0)
        self.assertEqual(summary["metrics"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertEqual(summary["new_embeddings"], 0)
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_score_row_read_count"], 0)
        self.assertEqual(summary["query_embedding_read_count"], 0)
        self.assertTrue(summary["new_independent_holdout_required"])
        self.assertTrue(summary["holdout_input_pending"])
        self.assertTrue(summary["refinement_sample_not_independent_revalidation"])

        primary = json.loads(
            (
                _FINAL
                / "scoring"
                / "target_001_target_distractor_primary_scoring_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(primary["formula_id"], f3h.PRIMARY_FORMULA)
        secondary = json.loads(
            (
                _FINAL
                / "scoring"
                / "target_001_target_distractor_secondary_scoring_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(secondary["formulas"]["S_top3_margin"]["k"], 3)
        agg = json.loads(
            (
                _FINAL
                / "scoring"
                / "target_001_target_distractor_tie_break_and_aggregation_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            agg["component_primary_aggregation"], "maximum_segment_primary_score"
        )
        access = json.loads(
            (_FINAL / "runtime" / "target_001_f3h_access_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(access["score_computation_count"], 0)
        self.assertFalse(access["gallery_npy_content_inspected_for_scoring"])
        self.assertTrue(access["gallery_npy_sha_verified_only"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3H root absent")
        with mock.patch.object(f3h, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3h.ScoringDesignError):
                f3h.run(_CFG)


if __name__ == "__main__":
    unittest.main()
