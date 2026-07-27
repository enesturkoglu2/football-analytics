"""Unit tests for ReID-R2B multi-frame helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
_src_str = str(_SRC)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.multiframe_r2b import (
    aggregate_with_fallback,
    classify_outcome,
    embedding_medoid,
    evaluate_joined,
    l2_mean_aggregate,
    rank_primary,
    score_queries_against_galleries,
    select_development_candidate,
    select_quality_temporal_diversity,
    select_uniform_temporal,
    uniform_temporal_indices,
    validate_and_clamp_bbox,
)


class TestUniformTemporal(unittest.TestCase):
    def test_indices_cover_span(self):
        idxs = uniform_temporal_indices(100, 12)
        self.assertEqual(len(idxs), 12)
        self.assertEqual(idxs[0], 0)
        self.assertEqual(idxs[-1], 99)
        self.assertEqual(idxs, sorted(set(idxs)))

    def test_n_less_than_k(self):
        self.assertEqual(uniform_temporal_indices(5, 12), list(range(5)))

    def test_select_uniform_skips_ineligible(self):
        cands = [
            {"eligible": False, "frame_index": 0, "crop_id": "a"},
            {"eligible": True, "frame_index": 1, "crop_id": "b"},
            {"eligible": True, "frame_index": 5, "crop_id": "c"},
            {"eligible": True, "frame_index": 9, "crop_id": "d"},
        ]
        sel = select_uniform_temporal(cands, max_frames=12)
        self.assertEqual(len(sel), 3)
        self.assertTrue(all(s["eligible"] for s in sel))


class TestBboxValidation(unittest.TestCase):
    def test_clamp_ok(self):
        m = validate_and_clamp_bbox([10.2, 20.8, 40.1, 80.9], frame_width=100, frame_height=100)
        self.assertTrue(m["eligible"])
        self.assertEqual(m["bbox_xyxy_int"][0], 10)

    def test_non_finite(self):
        m = validate_and_clamp_bbox([0, float("nan"), 10, 20], frame_width=100, frame_height=100)
        self.assertFalse(m["eligible"])

    def test_empty_after_clamp(self):
        m = validate_and_clamp_bbox([50, 50, 50, 50], frame_width=100, frame_height=100)
        self.assertFalse(m["eligible"])


class TestQualityTemporal(unittest.TestCase):
    def _cands(self, n=20):
        rows = []
        for i in range(n):
            rows.append(
                {
                    "eligible": True,
                    "frame_index": i * 3,
                    "crop_id": f"c{i:03d}",
                    "frame_edge_contact_count": 0 if i % 4 else 1,
                    "max_other_person_iou": 0.1 * (i % 5),
                    "bbox_area": 5000 + i * 10,
                    "laplacian_variance": 100.0 + i,
                }
            )
        return rows

    def test_respects_max_and_min(self):
        sel = select_quality_temporal_diversity(self._cands(20), max_frames=12, min_frames=3)
        self.assertGreaterEqual(len(sel), 3)
        self.assertLessEqual(len(sel), 12)
        frames = [r["frame_index"] for r in sel]
        self.assertEqual(frames, sorted(frames))

    def test_near_duplicate_suppression_spreads(self):
        sel = select_quality_temporal_diversity(self._cands(30), max_frames=8, min_frames=3)
        frames = [r["frame_index"] for r in sel]
        if len(frames) >= 2:
            gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
            self.assertTrue(min(gaps) >= 1)

    def test_deterministic(self):
        a = select_quality_temporal_diversity(self._cands(25), max_frames=12)
        b = select_quality_temporal_diversity(self._cands(25), max_frames=12)
        self.assertEqual([x["crop_id"] for x in a], [x["crop_id"] for x in b])


class TestAggregation(unittest.TestCase):
    def test_l2_mean_unit(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(5, 512)).astype(np.float32)
        out = l2_mean_aggregate(v)
        self.assertEqual(out.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)

    def test_short_fallback_one(self):
        v = np.ones((1, 512), dtype=np.float32)
        agg = aggregate_with_fallback(v, mode="l2_mean")
        self.assertEqual(agg["fallback_reason"], "single_frame")

    def test_short_fallback_two(self):
        v = np.stack([np.ones(512), np.arange(512)], axis=0).astype(np.float32)
        agg = aggregate_with_fallback(v, mode="embedding_medoid", crop_ids=["a", "b"], frame_indices=[1, 2], quality_ranks=[0, 1])
        self.assertEqual(agg["fallback_reason"], "two_frame_l2_mean")

    def test_medoid_tiebreak_deterministic(self):
        # three identical vectors -> pick earliest frame / lex crop
        v = np.tile(np.arange(512, dtype=np.float32), (3, 1))
        vec, cid, idx = embedding_medoid(
            v,
            crop_ids=["c", "a", "b"],
            frame_indices=[5, 5, 5],
            quality_ranks=[1, 1, 1],
        )
        self.assertEqual(cid, "a")
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)


class TestScoringMetrics(unittest.TestCase):
    def test_rank_and_score(self):
        Q = np.eye(3, 512, dtype=np.float32)
        T = np.eye(2, 512, dtype=np.float32)
        D = np.eye(2, 512, dtype=np.float32)
        # shift D rows
        D[0] = Q[2]
        rows = score_queries_against_galleries(
            Q=Q,
            T=T,
            D=D,
            target_ids=["t0", "t1"],
            distractor_ids=["d0", "d1"],
            query_meta=[
                {"stable_query_id": "q0", "segment_id": "s0"},
                {"stable_query_id": "q1", "segment_id": "s1"},
                {"stable_query_id": "q2", "segment_id": "s2"},
            ],
        )
        ranked = rank_primary(rows)
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3])

    def test_a_not_rescored_flag_in_outcome_paths(self):
        base = {
            "AP": 0.1,
            "AUROC": 0.5,
            "same_team_AUROC": 0.6,
            "Recall@10": 0.2,
            "positive_median_rank": 50,
            "margin": -0.1,
            "MRR": 0.1,
        }
        strong = {
            **base,
            "AP": 0.3,
            "AUROC": 0.65,
            "same_team_AUROC": 0.7,
            "Recall@10": 0.6,
            "positive_median_rank": 5,
            "margin": 0.1,
        }
        self.assertEqual(
            classify_outcome(metrics_a=base, metrics_b=strong, query_drop=0),
            "MULTIFRAME_STRONG_IMPROVEMENT",
        )
        directional = {
            **base,
            "AP": 0.12,
            "AUROC": 0.55,
            "same_team_AUROC": 0.6,
            "positive_median_rank": 40,
        }
        self.assertEqual(
            classify_outcome(metrics_a=base, metrics_b=directional, query_drop=0),
            "MULTIFRAME_DIRECTIONAL_IMPROVEMENT",
        )

    def test_select_candidate_tie_prefers_b1(self):
        m = {
            "AP": 0.2,
            "AUROC": 0.6,
            "same_team_AUROC": 0.7,
            "positive_median_rank": 10,
            "margin": 0.0,
            "Recall@10": 0.3,
            "MRR": 0.2,
        }
        self.assertEqual(
            select_development_candidate({"B1": m, "B2": m, "B3": m}),
            "B1",
        )

    def test_evaluate_joined_counts(self):
        rows = []
        for i in range(115):
            rows.append(
                {
                    "rank": i + 1,
                    "S_primary": 1.0 - i * 0.01,
                    "binary_clean_player_label": 1 if i < 10 else 0,
                    "same_team_negative_cohort": 10 <= i < 65,
                    "other_team_negative_cohort": i >= 65,
                }
            )
        m = evaluate_joined(rows)
        self.assertEqual(m["query_count"], 115)
        self.assertEqual(m["n_pos"], 10)


class TestNoGtLeakageIntoSelection(unittest.TestCase):
    def test_selection_ignores_gt_fields(self):
        cands = []
        for i in range(15):
            cands.append(
                {
                    "eligible": True,
                    "frame_index": i,
                    "crop_id": f"x{i:02d}",
                    "frame_edge_contact_count": 0,
                    "max_other_person_iou": 0.0,
                    "bbox_area": 8000,
                    "laplacian_variance": 50 + i,
                    "binary_clean_player_label": 1 if i == 0 else 0,  # must be ignored
                }
            )
        sel = select_quality_temporal_diversity(cands, max_frames=5)
        # If GT leaked as preference for label=1 only first frame, still ok either way —
        # ensure function does not require/read that field by deleting it.
        for c in cands:
            c.pop("binary_clean_player_label", None)
        sel2 = select_quality_temporal_diversity(cands, max_frames=5)
        self.assertEqual([s["crop_id"] for s in sel], [s["crop_id"] for s in sel2])


if __name__ == "__main__":
    unittest.main()
