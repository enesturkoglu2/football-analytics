"""Unit tests for Stage 5D-F3A retrieval error analysis."""

from __future__ import annotations

import csv
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

import run_reid_target_retrieval_error_analysis as f3a  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/target_retrieval_error_analysis_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_retrieval_error_analysis_and_diagnostics"
)
_F3 = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_retrieval_evaluation"
)
_HEAD = "27542937d39b80372f4d0f6c92e59d4f168e0e15"


class RetrievalErrorAnalysisTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3a.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3a.POSITIVE_IDS), 8)
        self.assertEqual(len(f3a.CONFLICT_COMPONENTS), 4)
        self.assertTrue(cfg["policy"]["sample_used_for_error_analysis_only"])
        self.assertFalse(cfg["policy"]["sample_authorized_for_gallery_optimization"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3a.ErrorAnalysisError):
            f3a.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3a.ErrorAnalysisError):
                f3a.atomic_publish(tmp, final)

    def test_f3_contract_if_present(self):
        if not _F3.is_dir():
            self.skipTest("F3 root absent")
        cfg = f3a.load_config(_CFG)
        f3 = f3a.validate_f3(_PROJECT_ROOT, cfg)
        self.assertEqual(
            f3["summary"]["descriptive_outcome"],
            "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING",
        )
        pos = [r for r in f3["ranking"] if r["binary_label"] == "1"]
        self.assertEqual(len(pos), 8)
        high = [r for r in pos if int(r["rank"]) <= 10]
        low = [r for r in pos if int(r["rank"]) > 10]
        self.assertEqual(len(high), 3)
        self.assertEqual(len(low), 5)
        neg = [r for r in f3["ranking"] if r["binary_label"] == "0"]
        self.assertEqual(len(neg[:24]), 24)
        min_pos = min(float(r["max_individual_cosine"]) for r in pos)
        overlap = [r for r in neg if float(r["max_individual_cosine"]) >= min_pos]
        self.assertGreaterEqual(len(overlap), 1)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3A root absent")
        s = json.loads((_FINAL / "stage5d_f3a_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["final_status"], f3a.FINAL_STATUS)
        self.assertTrue(s["official_f3_outcome_unchanged"])
        self.assertEqual(s["high_ranked_positives"], 3)
        self.assertEqual(s["low_ranked_positives"], 5)
        self.assertEqual(s["top_false_positive_cohort"], 24)
        self.assertEqual(s["excluded_review_cohort"], 12)
        self.assertEqual(s["conflict_components"], 4)
        self.assertEqual(s["contact_sheets"], 5)
        self.assertEqual(s["diagnostic_videos"], 2)
        self.assertEqual(s["gallery_members"], 7)
        self.assertFalse(s["gallery_mutation"])
        self.assertFalse(s["threshold_selected"])
        self.assertEqual(s["identity_assignments"], 0)
        self.assertFalse(s["score_recompute"])
        self.assertEqual(s["new_embeddings"], 0)
        review = (
            _FINAL
            / "review_packages"
            / "target_001_retrieval_error_review"
        )
        pngs = sorted(review.glob("*.png"))
        self.assertEqual(len(pngs), 5)
        for p in pngs:
            import cv2

            img = cv2.imread(str(p))
            self.assertIsNotNone(img)
            self.assertGreaterEqual(img.shape[1], 3600)
        videos = sorted((_FINAL / "videos").glob("*.mp4"))
        self.assertEqual(len(videos), 2)
        hyp = json.loads(
            (
                _FINAL
                / "gallery_diagnostics"
                / "target_001_gallery_refinement_hypotheses.json"
            ).read_text(encoding="utf-8")
        )
        for h in hyp["hypotheses"]:
            self.assertTrue(h["diagnostic_only"])
            self.assertFalse(h["action_authorized"])
            self.assertTrue(h["same_sample_revalidation_forbidden"])
        with (
            _FINAL / "templates" / "target_001_retrieval_error_manual_review_template.csv"
        ).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for field in (
                "manual_visible_team",
                "manual_visible_jersey_number",
                "manual_notes",
                "reviewer",
                "final_approver",
                "reviewed_at",
            ):
                self.assertEqual(row[field], "")
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        anchors = json.loads(
            (
                _FINAL
                / "gallery_diagnostics"
                / "target_001_anchor_discrimination_diagnostics.json"
            ).read_text(encoding="utf-8")
        )
        by_id = {a["anchor_id"]: a for a in anchors["anchors"]}
        self.assertEqual(by_id["target_001_ext_anchor_006"]["best_match_count_positive"], 5)
        self.assertEqual(by_id["target_001_ext_anchor_008"]["best_match_count_all"], 77)
        self.assertEqual(by_id["target_001_ext_anchor_014"]["best_match_count_positive"], 0)
        for a in anchors["anchors"]:
            self.assertFalse(a["removal_authorized"])

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3A root absent")
        with mock.patch.object(f3a, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3a.ErrorAnalysisError):
                f3a.run(_CFG)


if __name__ == "__main__":
    unittest.main()
