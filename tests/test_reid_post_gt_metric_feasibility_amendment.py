"""Unit tests for Stage 5D-F2B post-GT metric feasibility amendment."""

from __future__ import annotations

import hashlib
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

import run_reid_post_gt_metric_feasibility_amendment as f2b  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/post_gt_metric_feasibility_amendment_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_post_gt_metric_feasibility_amendment"
)
_F2A = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_sample_ground_truth_freeze"
)
_HEAD = "3245ce3313518d025e12c38ddc934df819a2dc2e"
_F2A_SNAP = "d5db0e4d978ed7677f196f2516fc894f8b4803a09f14f59960a78097ec601504"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PostGtMetricFeasibilityAmendmentTests(unittest.TestCase):
    def test_expected_git_contract_and_ceilings(self):
        cfg = f2b.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Freeze target 001 sample ground truth",
        )
        self.assertEqual(f2b.SEGMENT_POS, 8)
        self.assertEqual(f2b.SEGMENT_NEG, 110)
        self.assertEqual(f2b.SEGMENT_EXCL, 32)
        self.assertEqual(f2b.COMP_POS, 4)
        self.assertEqual(f2b.COMP_NEG, 95)
        self.assertEqual(f2b.COMP_EXCL, 26)
        self.assertEqual(f2b.COMP_CONFLICT, 4)
        self.assertAlmostEqual(f2b.recall_ceiling(8, 1), 0.125)
        self.assertAlmostEqual(f2b.recall_ceiling(8, 3), 0.375)
        self.assertAlmostEqual(f2b.recall_ceiling(8, 5), 0.625)
        self.assertAlmostEqual(f2b.recall_ceiling(8, 10), 1.0)
        feas = f2b.segment_recall_feasibility()
        self.assertFalse(feas["segment_Recall@5_equals_1_0_mathematically_attainable"])
        self.assertFalse(feas["original_strong_signal_criterion_feasible"])
        self.assertEqual(feas["infeasibility_reason"], "positive_count_exceeds_k")
        self.assertTrue(
            feas["component_Recall@5_equals_1_0_mathematically_attainable"]
        )
        self.assertAlmostEqual(
            feas["component_mathematical_ceilings"]["Recall@5"], 1.0
        )

    def test_exact_positive_ids_and_conflicts(self):
        self.assertEqual(
            f2b.POSITIVE_IDS,
            (
                "SAMPLE_EVAL_003",
                "SAMPLE_EVAL_024",
                "SAMPLE_EVAL_028",
                "SAMPLE_EVAL_042",
                "SAMPLE_EVAL_046",
                "SAMPLE_EVAL_069",
                "SAMPLE_EVAL_100",
                "SAMPLE_EVAL_102",
            ),
        )
        ids = [c["evaluation_component_id"] for c in f2b.CONFLICTING_COMPONENTS]
        self.assertEqual(
            ids,
            [
                "SAMPLE_COMPONENT_005",
                "SAMPLE_COMPONENT_018",
                "SAMPLE_COMPONENT_032",
                "SAMPLE_COMPONENT_055",
            ],
        )
        policy = f2b.conflict_policy()
        self.assertTrue(policy["policy"]["excluded_from_component_level_metrics"])
        self.assertTrue(
            policy["policy"]["clean_segment_members_retain_segment_metric_eligibility"]
        )
        self.assertIn("SAMPLE_EVAL_003", policy["retained_segment_level_member_codes"])
        self.assertIn("SAMPLE_EVAL_004", policy["retained_segment_level_member_codes"])
        self.assertEqual(policy["component_metric_universe"]["metric_component_total"], 99)

    def test_amended_strong_outcome_exact(self):
        amended = f2b.amended_outcome_contract()
        strong = amended["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"]
        self.assertEqual(strong["ranking"]["segment_Recall@5"], 0.625)
        self.assertEqual(strong["ranking"]["segment_Recall@10"], 1.0)
        self.assertEqual(strong["ranking"]["component_Recall@5"], 1.0)
        self.assertEqual(strong["quality"]["segment_AP_ge"], 0.80)
        self.assertEqual(strong["quality"]["component_AP_ge"], 0.80)
        self.assertTrue(
            strong["quality"][
                "min_positive_segment_score_gt_max_negative_segment_score"
            ]
        )
        self.assertFalse(amended["insufficient_ground_truth_expected"])
        self.assertEqual(amended["primary_retrieval_score_unchanged"], "max_individual_cosine")
        self.assertEqual(
            amended["secondary_diagnostic_scores_unchanged"],
            list(f2b.SECONDARY_SCORES),
        )

    def test_path_traversal_rejection(self):
        with self.assertRaises(f2b.MetricFeasibilityError):
            f2b.assert_no_path_traversal("../escape")
        with self.assertRaises(f2b.MetricFeasibilityError):
            f2b.assert_no_path_traversal("/abs/path")

    def test_atomic_finalization_rejects_existing_final(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f2b.MetricFeasibilityError):
                f2b.atomic_publish(tmp, final)

    def test_f2a_package_if_present(self):
        if not _F2A.is_dir():
            self.skipTest("F2A freeze root absent")
        cfg = f2b.load_config(_CFG)
        f2a = f2b.validate_f2a(_PROJECT_ROOT, cfg)
        self.assertEqual(
            f2a["summary"]["final_status"],
            "COMPLETED_STAGE5D_F2A_TARGET_001_SAMPLE_GROUND_TRUTH_FROZEN",
        )
        self.assertEqual(f2a["snapshot_sha256"], _F2A_SNAP)
        self.assertEqual(f2a["summary"]["clean_positive_metric_items"], 8)
        self.assertEqual(f2a["summary"]["clean_negative_metric_items"], 110)
        self.assertEqual(f2a["summary"]["excluded_metric_items"], 32)

    def test_live_amendment_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F2B amendment root absent")
        s = json.loads((_FINAL / "stage5d_f2b_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["final_status"], f2b.FINAL_STATUS)
        self.assertEqual(s["segment_positives"], 8)
        self.assertEqual(s["segment_negatives"], 110)
        self.assertEqual(s["segment_excluded"], 32)
        self.assertEqual(s["clean_positive_components"], 4)
        self.assertEqual(s["conflicting_components"], 4)
        self.assertFalse(s["original_strong_signal_feasible"])
        self.assertAlmostEqual(s["segment_recall_ceilings"]["Recall@5"], 0.625)
        self.assertEqual(s["similarity_rows"], 0)
        self.assertEqual(s["ranking_rows"], 0)
        self.assertFalse(s["gallery_vectors_read"])
        self.assertFalse(s["sample_embedding_vectors_read"])
        self.assertFalse(s["threshold_selected"])
        self.assertEqual(s["identity_assignments"], 0)
        self.assertTrue(s["original_scoring_formulas_unchanged"])
        amended = json.loads(
            (
                _FINAL
                / "metric_feasibility"
                / "target_001_amended_retrieval_outcome_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            amended["INDEPENDENT_RETRIEVAL_STRONG_SIGNAL"]["ranking"]["segment_Recall@5"],
            0.625,
        )
        for pattern in ("*.npy", "*.csv", "*.png", "*.mp4"):
            self.assertEqual(list(_FINAL.rglob(pattern)), [])

    def test_run_rejects_when_final_exists(self):
        if not _FINAL.is_dir():
            self.skipTest("F2B amendment root absent")
        with mock.patch.object(f2b, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f2b.MetricFeasibilityError):
                f2b.run(_CFG)


if __name__ == "__main__":
    unittest.main()
