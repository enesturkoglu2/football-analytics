"""Unit tests for ReID-R2D SportsReID domain ablation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.model_registry import get_reid_model_spec  # noqa: E402
from football_analytics.reid.r2d_domain_ablation import (  # noqa: E402
    R2DError,
    candidate_review_outcome,
    classify_football_domain_outcome,
    evaluate_joined,
    next_gate_for_outcome,
    outcome_to_final_status,
    rank_primary,
    score_queries_against_galleries,
    validate_embedding_matrix,
)


class R2DHelperTests(unittest.TestCase):
    def test_model_id_and_sha_from_registry(self) -> None:
        spec = get_reid_model_spec("osnet_x1_0_sportsreid_soccernet")
        self.assertEqual(spec["model_id"], "osnet_x1_0_sportsreid_soccernet")
        self.assertEqual(
            spec["sha256"],
            "c61e0da2007f7c7f4d889cb68774dfeecf8c4c433e0bfe3858b48b8655f83e91",
        )
        self.assertTrue(spec["weights_only"])
        self.assertTrue(spec["safe_load_required"])

    def test_no_market1501_fallback_on_bad_id(self) -> None:
        from football_analytics.reid.embedding import EmbeddingError, load_reid_osnet_by_model_id

        with self.assertRaises(EmbeddingError) as ctx:
            load_reid_osnet_by_model_id("not_registered")
        self.assertIn("no fallback", str(ctx.exception))

    def test_scoring_formula_and_tiebreak(self) -> None:
        Q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        T = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        D = np.asarray([[0.2, 0.0], [0.0, 1.0]], dtype=np.float32)
        # L2 normalize for cosine-like behavior
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        rows = score_queries_against_galleries(
            Q=Q,
            T=T,
            D=D,
            target_ids=["t_b", "t_a"],
            distractor_ids=["d1", "d0"],
            query_meta=[
                {"stable_query_id": "q1"},
                {"stable_query_id": "q0"},
            ],
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["S_primary"], row["T_max"] - row["D_max"], places=5)
        ranked = rank_primary(rows)
        self.assertEqual([r["rank"] for r in ranked], [1, 2])
        # ranks are unique and cover both queries
        self.assertEqual({r["stable_query_id"] for r in ranked}, {"q0", "q1"})

    def test_equal_score_prefers_lexicographically_smaller_member_id(self) -> None:
        from football_analytics.reid.r2d_domain_ablation import argmax_with_id_tiebreak

        scores = np.asarray([0.5, 0.5], dtype=np.float32)
        val, mid = argmax_with_id_tiebreak(scores, ["z_id", "a_id"])
        self.assertEqual(mid, "a_id")
        self.assertAlmostEqual(val, 0.5)

    def test_evaluate_and_outcome_rules(self) -> None:
        joined = []
        # 2 positives at ranks via scores
        for i in range(10):
            joined.append(
                {
                    "stable_query_id": f"q{i:02d}",
                    "S_primary": 1.0 - 0.01 * i,
                    "T_max": 0.9,
                    "D_max": 0.1,
                    "rank": i + 1,
                    "binary_clean_player_label": 1 if i in (0, 1) else 0,
                    "same_team_negative_cohort": i in (2, 3, 4),
                    "other_team_negative_cohort": i >= 5,
                }
            )
        metrics = evaluate_joined(joined)
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["Recall@3"], 1.0)
        weak_a = {
            "AP": 0.1,
            "AUROC": 0.5,
            "same_team_AUROC": 0.5,
            "Recall@10": 0.2,
            "Recall@5": 0.0,
            "positive_median_rank": 50.0,
            "margin": -0.1,
        }
        strong_c = {
            "AP": 0.4,
            "AUROC": 0.7,
            "same_team_AUROC": 0.65,
            "Recall@10": 0.7,
            "Recall@5": 0.4,
            "positive_median_rank": 5.0,
            "margin": 0.1,
        }
        outcome = classify_football_domain_outcome(
            metrics_a=weak_a, metrics_c=strong_c, query_drop=0, deterministic=True
        )
        self.assertEqual(outcome, "FOOTBALL_DOMAIN_STRONG_IMPROVEMENT")
        self.assertTrue(
            outcome_to_final_status(outcome).startswith("COMPLETED_R2D_")
        )
        self.assertTrue(next_gate_for_outcome(outcome).startswith("REID_R2E_"))

    def test_candidate_review_buckets(self) -> None:
        self.assertEqual(
            candidate_review_outcome({"Recall@5": 0.8, "Recall@10": 0.9}),
            "CANDIDATE_REVIEW_PROMISING",
        )
        self.assertEqual(
            candidate_review_outcome({"Recall@5": 0.5, "Recall@10": 0.75}),
            "CANDIDATE_REVIEW_LIMITED",
        )
        self.assertEqual(
            candidate_review_outcome({"Recall@5": 0.2, "Recall@10": 0.4}),
            "CANDIDATE_REVIEW_NOT_USEFUL",
        )

    def test_embedding_validation_and_r2b_source_exclusion(self) -> None:
        mat = np.random.randn(3, 512).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        report = validate_embedding_matrix(mat, expected_rows=3, max_abs_diff=0.0)
        self.assertEqual(report["rows"], 3)
        with self.assertRaises(R2DError):
            validate_embedding_matrix(mat, expected_rows=3, max_abs_diff=1e-3)
        import football_analytics.reid.r2d_domain_ablation as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("multiframe_r2b", source)


if __name__ == "__main__":
    unittest.main()
