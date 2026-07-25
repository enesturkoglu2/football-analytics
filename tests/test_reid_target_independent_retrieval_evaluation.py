"""Unit tests for Stage 5D-F3 independent retrieval evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_target_independent_retrieval_evaluation as f3  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/target_independent_retrieval_evaluation_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_retrieval_evaluation"
)
_HEAD = "380a3cbe87dddb8a759870f00c8fb38f1e4d16f5"


class IndependentRetrievalEvaluationTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Amend target 001 retrieval metrics before scoring",
        )
        self.assertEqual(len(f3.GALLERY_IDS), 7)
        self.assertEqual(f3.GALLERY_IDS[-1], "target_001_ext_anchor_014")
        self.assertEqual(len(f3.POSITIVE_IDS), 8)
        self.assertEqual(len(f3.CONFLICT_COMPONENTS), 4)

    def test_recall_and_ap_helpers(self):
        labels = [1, 0, 1, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(f3.recall_at_k(labels, 1, 2), 0.5)
        self.assertAlmostEqual(f3.recall_at_k(labels, 3, 2), 1.0)
        ap = f3.average_precision_from_ranks([1, 3], 2)
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0)
        mets = f3.ranking_metrics(labels, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], n_pos=2)
        self.assertAlmostEqual(mets["MRR"], 1.0)
        self.assertAlmostEqual(mets["Recall@5"], 1.0)
        self.assertGreater(mets["Average_Precision"], 0.0)

    def test_score_formulas_on_toy(self):
        gallery = np.eye(7, 512, dtype=np.float32)
        for i in range(7):
            gallery[i] = gallery[i] / np.linalg.norm(gallery[i])
        q = gallery[2].copy()
        queries = [
            {
                "sample_eval_code": "SAMPLE_EVAL_001",
                "segment_id": "s1",
                "evaluation_component_id": "C1",
                "embedding_row": 0,
                "embedding_vector_sha256": "x",
                "vector": q,
            }
        ]
        centroid = gallery.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        medoid = gallery[2]
        matrix, rows = f3.score_queries(
            queries, gallery, centroid.astype(np.float32), medoid, f3.GALLERY_IDS
        )
        self.assertEqual(matrix.shape, (1, 7))
        self.assertAlmostEqual(rows[0]["max_individual_cosine"], 1.0, places=5)
        self.assertEqual(rows[0]["best_gallery_anchor_id"], f3.GALLERY_IDS[2])
        top3 = sorted(rows[0]["individual_cosine_vector"], reverse=True)[:3]
        self.assertAlmostEqual(
            rows[0]["top3_mean_individual_cosine"], sum(top3) / 3.0, places=6
        )
        self.assertAlmostEqual(
            rows[0]["mean_individual_cosine"],
            sum(rows[0]["individual_cosine_vector"]) / 7.0,
            places=6,
        )

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3.RetrievalEvalError):
            f3.assert_no_path_traversal("../x")
        with self.assertRaises(f3.RetrievalEvalError):
            f3.assert_no_path_traversal("/abs")

    def test_atomic_finalization_rejects_existing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3.RetrievalEvalError):
                f3.atomic_publish(tmp, final)

    def test_outcome_strong_and_insufficient(self):
        strong = {
            "support": {
                "clean_positive_segment_count_ge": 2,
                "clean_negative_segment_count_ge": 20,
                "clean_positive_component_count_ge": 2,
                "clean_negative_component_count_ge": 20,
            },
            "ranking": {
                "segment_Recall@5": 0.625,
                "segment_Recall@10": 1.0,
                "component_Recall@5": 1.0,
            },
            "quality": {"segment_AP_ge": 0.80, "component_AP_ge": 0.80},
        }
        seg = {
            "positive_count": 8,
            "negative_count": 110,
            "Recall@5": 0.625,
            "Recall@10": 1.0,
            "Average_Precision": 0.90,
            "separation_margin_min_pos_minus_max_neg": 0.1,
            "every_positive_rank": [1, 2, 3, 4, 5, 6, 7, 8],
        }
        comp = {
            "positive_count": 4,
            "negative_count": 95,
            "Recall@5": 1.0,
            "Average_Precision": 0.95,
            "separation_margin_min_pos_minus_max_neg": 0.05,
            "every_positive_rank": [1, 2, 3, 4],
        }
        out = f3.select_outcome(
            segment_metrics=seg, component_metrics=comp, amended_strong=strong
        )
        self.assertEqual(out["descriptive_outcome"], "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL")
        seg2 = dict(seg)
        seg2["positive_count"] = 1
        out2 = f3.select_outcome(
            segment_metrics=seg2, component_metrics=comp, amended_strong=strong
        )
        self.assertEqual(
            out2["descriptive_outcome"],
            "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH",
        )

    def test_upstream_contracts_if_present(self):
        cfg = f3.load_config(_CFG)
        f2b = f3.validate_f2b(_PROJECT_ROOT, cfg)
        self.assertEqual(f2b["summary"]["segment_positives"], 8)
        f2a = f3.validate_f2a(_PROJECT_ROOT, cfg)
        self.assertEqual(f2a["summary"]["eligible_total"], 118)
        # Do not call validate_gallery/sample here in a way that is heavy if absent;
        # live final test covers scoring when present.

    def test_live_evaluation_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3 evaluation root absent")
        s = json.loads((_FINAL / "stage5d_f3_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["final_status"], f3.FINAL_STATUS)
        self.assertEqual(s["queries_scored"], 150)
        self.assertEqual(s["individual_cosine_shape"], [150, 7])
        self.assertEqual(s["official_segment_metric_rows"], 118)
        self.assertEqual(s["official_component_metric_rows"], 99)
        self.assertEqual(s["gallery_members"], 7)
        self.assertFalse(s["threshold_selected"])
        self.assertEqual(s["identity_assignments"], 0)
        self.assertFalse(s["gallery_mutation"])
        self.assertEqual(s["new_embeddings"], 0)
        self.assertTrue(s["two_pass_deterministic"])
        self.assertIn(
            s["descriptive_outcome"],
            {
                "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL",
                "INDEPENDENT_RETRIEVAL_PROMISING_BUT_OVERLAPPING",
                "INDEPENDENT_RETRIEVAL_WEAK",
                "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH",
            },
        )
        matrix = np.load(_FINAL / "scores" / "target_001_sample_individual_cosine.npy")
        self.assertEqual(matrix.shape, (150, 7))
        pre = json.loads(
            (
                _FINAL / "runtime" / "target_001_f3_scoring_contract_pre_score.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(pre["scores_seen_at_contract_freeze"])
        self.assertFalse(pre["threshold_selected"])
        # conflict components excluded from component ranking
        import csv

        with (_FINAL / "rankings" / "target_001_component_primary_ranking.csv").open(
            encoding="utf-8"
        ) as handle:
            comps = list(csv.DictReader(handle))
        self.assertEqual(len(comps), 99)
        ids = {r["evaluation_component_id"] for r in comps}
        for cid in f3.CONFLICT_COMPONENTS:
            self.assertNotIn(cid, ids)
        with (_FINAL / "rankings" / "target_001_segment_primary_ranking.csv").open(
            encoding="utf-8"
        ) as handle:
            segs = list(csv.DictReader(handle))
        self.assertEqual(len(segs), 118)
        self.assertEqual(sum(1 for r in segs if r["binary_label"] == "1"), 8)
        self.assertEqual(sum(1 for r in segs if r["binary_label"] == "0"), 110)
        # ranking deterministic: score desc then code asc
        for a, b in zip(segs, segs[1:]):
            sa, sb = float(a["max_individual_cosine"]), float(b["max_individual_cosine"])
            self.assertGreaterEqual(sa, sb)
            if sa == sb:
                self.assertLessEqual(a["sample_eval_code"], b["sample_eval_code"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [
            _FINAL / "scores" / "target_001_sample_individual_cosine.npy"
        ])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3 evaluation root absent")
        with mock.patch.object(f3, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3.RetrievalEvalError):
                f3.run(_CFG)


if __name__ == "__main__":
    unittest.main()
