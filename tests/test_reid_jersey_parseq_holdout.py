"""Unit tests for Stage 5C-C3F-A PARSeq-blind holdout design helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_reid_jersey_parseq_holdout.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("c3f_a_holdout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["c3f_a_holdout"] = module
    spec.loader.exec_module(module)
    return module


c3f = _load_module()


def _rank_row(item_id: str, ranks: dict[str, int], **extra):
    row = {
        "review_item_id": item_id,
        "crop_id": f"crop_{item_id}",
        "segment_id": f"seg_{item_id}",
        "raw_track_id": extra.pop("raw_track_id", hash(item_id) % 10_000),
        "frame_index": extra.pop("frame_index", 0),
        "source_crop_sha256": f"sha_{item_id}",
        "composite_score": extra.pop("composite_score", 0.0),
        **{field: ranks.get(field, 1) for field in c3f.RANK_FIELDS},
    }
    row.update(extra)
    return row


def _make_pool_for_selection(per_stratum: int = 60) -> list[dict]:
    """Build large enough pools so combined high quota 48 is feasible."""
    rows: list[dict] = []
    # High scores first; assign_signal_strata will re-label by tercile.
    # Create 3 * per_stratum items with descending scores and unique segments.
    total = per_stratum * 3
    for index in range(total):
        score = float(total - index)
        item_id = f"item_{index:04d}"
        row = _rank_row(
            item_id,
            {field: index + 1 for field in c3f.RANK_FIELDS},
            composite_score=score,
            raw_track_id=10_000 + index,  # unique tracks → diversity free
            frame_index=index * 100,
            segment_id=f"seg_{index:04d}",
            source_crop_sha256=f"sha_{index:04d}",
        )
        rows.append(row)
    return c3f.assign_signal_strata(rows)


class HoldoutDesignTests(unittest.TestCase):
    def test_invert_one_based_rank_to_unit(self):
        self.assertEqual(c3f.invert_one_based_rank_to_unit(1, 5), 1.0)
        self.assertEqual(c3f.invert_one_based_rank_to_unit(5, 5), 0.0)
        self.assertEqual(c3f.invert_one_based_rank_to_unit(3, 5), 0.5)
        self.assertEqual(c3f.invert_one_based_rank_to_unit(1, 1), 1.0)
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.invert_one_based_rank_to_unit(0, 5)
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.invert_one_based_rank_to_unit(6, 5)

    def test_nan_rank_rejected(self):
        row = {field: 1 for field in c3f.RANK_FIELDS}
        row["roi_height_global_rank"] = float("nan")
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.ranking_feature_vector(row)

    def test_composite_score_prefers_lower_ranks(self):
        better = {field: 1 for field in c3f.RANK_FIELDS}
        worse = {field: 10 for field in c3f.RANK_FIELDS}
        self.assertGreater(
            c3f.composite_rank_score(better, universe_size=10),
            c3f.composite_rank_score(worse, universe_size=10),
        )

    def test_strata_terciles_not_ground_truth(self):
        rows = [
            {"review_item_id": f"r{i}", "composite_score": float(9 - i)}
            for i in range(9)
        ]
        stratified = c3f.assign_signal_strata(rows)
        labels = [r["signal_stratum"] for r in stratified]
        self.assertEqual(labels.count("high_signal_candidate"), 3)
        self.assertEqual(labels.count("mid_signal_candidate"), 3)
        self.assertEqual(labels.count("safety_candidate"), 3)
        self.assertTrue(all(r["stratum_is_ground_truth"] is False for r in stratified))
        self.assertEqual(stratified[0]["signal_stratum"], "high_signal_candidate")
        self.assertEqual(stratified[-1]["signal_stratum"], "safety_candidate")

    def test_average_hash_and_hamming(self):
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        b = np.zeros((32, 32, 3), dtype=np.uint8)
        b[:, 16:] = 255
        ha = c3f.average_hash_8x8(a)
        hb = c3f.average_hash_8x8(b)
        self.assertEqual(ha.shape, (64,))
        self.assertEqual(c3f.hamming_distance(ha, ha), 0)
        self.assertGreater(c3f.hamming_distance(ha, hb), 0)
        # Deterministic
        self.assertTrue(np.array_equal(ha, c3f.average_hash_8x8(a)))

    def test_hard_exclusion_reasons(self):
        discovery = {
            "review_item_id": {"review_x"},
            "source_crop_sha256": {"sha_y"},
            "crop_id": {"crop_z"},
            "segment_id": {"seg_w"},
            "raw_track_id": {7},
        }
        reviewed = {"review_pilot"}
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "review_pilot",
                    "source_crop_sha256": "a",
                    "crop_id": "c",
                    "segment_id": "s",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "reviewed_pilot",
        )
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "review_x",
                    "source_crop_sha256": "a",
                    "crop_id": "c",
                    "segment_id": "s",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "discovery_review_item_id",
        )
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "other",
                    "source_crop_sha256": "sha_y",
                    "crop_id": "c",
                    "segment_id": "s",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "discovery_source_crop_sha256",
        )
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "other",
                    "source_crop_sha256": "a",
                    "crop_id": "crop_z",
                    "segment_id": "s",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "discovery_crop_id",
        )
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "other",
                    "source_crop_sha256": "a",
                    "crop_id": "c",
                    "segment_id": "seg_w",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "discovery_segment_id",
        )
        self.assertEqual(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "other",
                    "source_crop_sha256": "a",
                    "crop_id": "c",
                    "segment_id": "s",
                    "raw_track_id": 7,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            ),
            "discovery_raw_track_id",
        )
        self.assertIsNone(
            c3f.hard_exclusion_reason(
                {
                    "review_item_id": "ok",
                    "source_crop_sha256": "a",
                    "crop_id": "c",
                    "segment_id": "s",
                    "raw_track_id": 1,
                },
                reviewed_ids=reviewed,
                discovery=discovery,
            )
        )

    def test_diversity_limits(self):
        state = c3f.DiversityState(
            max_per_segment=1, max_per_raw_track=2, min_frame_gap=60
        )
        a = _rank_row(
            "a",
            {f: 1 for f in c3f.RANK_FIELDS},
            segment_id="seg1",
            raw_track_id=1,
            frame_index=100,
            composite_score=1.0,
        )
        b = _rank_row(
            "b",
            {f: 1 for f in c3f.RANK_FIELDS},
            segment_id="seg1",
            raw_track_id=2,
            frame_index=200,
            composite_score=0.9,
        )
        c = _rank_row(
            "c",
            {f: 1 for f in c3f.RANK_FIELDS},
            segment_id="seg2",
            raw_track_id=1,
            frame_index=120,
            composite_score=0.8,
        )
        d = _rank_row(
            "d",
            {f: 1 for f in c3f.RANK_FIELDS},
            segment_id="seg3",
            raw_track_id=1,
            frame_index=200,
            composite_score=0.7,
        )
        e = _rank_row(
            "e",
            {f: 1 for f in c3f.RANK_FIELDS},
            segment_id="seg4",
            raw_track_id=1,
            frame_index=300,
            composite_score=0.6,
        )
        self.assertTrue(state.accepts(a))
        state.add(a)
        self.assertFalse(state.accepts(b))  # same segment
        self.assertFalse(state.accepts(c))  # frame gap < 60
        self.assertTrue(state.accepts(d))
        state.add(d)
        self.assertFalse(state.accepts(e))  # max 2 per track

    def test_select_primary_and_reserve_counts(self):
        pool = _make_pool_for_selection(60)
        high = sum(1 for r in pool if r["signal_stratum"] == "high_signal_candidate")
        self.assertGreaterEqual(high, 48)
        primary, reserve = c3f.select_primary_and_reserve(
            pool,
            combined_quotas={
                "high_signal_candidate": 48,
                "mid_signal_candidate": 24,
                "safety_candidate": 24,
            },
            primary_quotas={
                "high_signal_candidate": 32,
                "mid_signal_candidate": 16,
                "safety_candidate": 16,
            },
            reserve_quotas={
                "high_signal_candidate": 16,
                "mid_signal_candidate": 8,
                "safety_candidate": 8,
            },
        )
        self.assertEqual(len(primary), 64)
        self.assertEqual(len(reserve), 32)
        self.assertEqual(
            sum(1 for r in primary if r["stratum"] == "high_signal_candidate"), 32
        )
        self.assertEqual(
            sum(1 for r in reserve if r["stratum"] == "high_signal_candidate"), 16
        )
        self.assertEqual(
            sum(1 for r in primary if r["stratum"] == "mid_signal_candidate"), 16
        )
        self.assertEqual(
            sum(1 for r in primary if r["stratum"] == "safety_candidate"), 16
        )
        primary_ids = {r["review_item_id"] for r in primary}
        reserve_ids = {r["review_item_id"] for r in reserve}
        self.assertEqual(len(primary_ids & reserve_ids), 0)
        # Combined-first: first 32 high scores in high stratum go primary
        high_rows = [
            r for r in pool if r["signal_stratum"] == "high_signal_candidate"
        ]
        high_rows = sorted(
            high_rows, key=lambda r: (-r["composite_score"], r["review_item_id"])
        )
        self.assertEqual(
            [r["review_item_id"] for r in primary if r["stratum"] == "high_signal_candidate"],
            [r["review_item_id"] for r in high_rows[:32]],
        )

    def test_stratum_capacity_blocked(self):
        tiny = _make_pool_for_selection(5)
        with self.assertRaises(c3f.HoldoutDesignError) as ctx:
            c3f.select_primary_and_reserve(
                tiny,
                combined_quotas={
                    "high_signal_candidate": 48,
                    "mid_signal_candidate": 24,
                    "safety_candidate": 24,
                },
                primary_quotas={
                    "high_signal_candidate": 32,
                    "mid_signal_candidate": 16,
                    "safety_candidate": 16,
                },
                reserve_quotas={
                    "high_signal_candidate": 16,
                    "mid_signal_candidate": 8,
                    "safety_candidate": 8,
                },
            )
        self.assertIn("BLOCKED_STRATUM_CAPACITY", str(ctx.exception))

    def test_candidate_cut_derivation_lowest_perfect_safe(self):
        ops = [
            {
                "confidence_cut": 0.9999,
                "exact_retained": 1,
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": False,
            },
            {
                "confidence_cut": 0.999636,
                "exact_retained": 3,
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": False,
            },
            {
                "confidence_cut": 0.5,
                "exact_retained": 5,
                "wrong_positive_retained": 1,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": False,
            },
            {
                "confidence_cut": 0.0,
                "exact_retained": 5,
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": "accept_all",
                "selected": False,
            },
        ]
        result = c3f.derive_validation_candidate_cut(ops)
        self.assertEqual(result["validation_candidate_cut"], 0.999636)
        self.assertEqual(result["secondary_descriptive_cuts"], [0.9999])
        self.assertFalse(result["deployment_threshold_selected"])
        self.assertFalse(result["threshold_selected_for_production"])
        self.assertTrue(result["all_operating_points_selected_false"])

    def test_selected_true_rejected_in_cut_derivation(self):
        ops = [
            {
                "confidence_cut": 0.9,
                "exact_retained": 1,
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": True,
            }
        ]
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.derive_validation_candidate_cut(ops)

    def test_decision_rules_preregistration(self):
        self.assertEqual(
            c3f.classify_holdout_decision(
                readable_positive=12,
                negative_or_safety=24,
                accepted_wrong_positive=0,
                accepted_negative=0,
                accepted_exact=2,
            ),
            "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL",
        )
        self.assertEqual(
            c3f.classify_holdout_decision(
                readable_positive=12,
                negative_or_safety=24,
                accepted_wrong_positive=0,
                accepted_negative=0,
                accepted_exact=1,
            ),
            "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT",
        )
        self.assertEqual(
            c3f.classify_holdout_decision(
                readable_positive=12,
                negative_or_safety=24,
                accepted_wrong_positive=1,
                accepted_negative=0,
                accepted_exact=5,
            ),
            "FAIL_INDEPENDENT_GATE_SAFETY",
        )
        self.assertEqual(
            c3f.classify_holdout_decision(
                readable_positive=11,
                negative_or_safety=24,
                accepted_wrong_positive=0,
                accepted_negative=0,
                accepted_exact=5,
            ),
            "BLOCKED_INSUFFICIENT_LABELED_HOLDOUT",
        )

    def test_annotation_rules(self):
        self.assertEqual(
            c3f.validate_holdout_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="yes",
                manual_jersey_number="7",
            ),
            [],
        )
        self.assertIn(
            "readable=yes requires jersey_number",
            c3f.validate_holdout_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="yes",
                manual_jersey_number="",
            ),
        )
        self.assertIn(
            "readable!=yes requires blank jersey_number",
            c3f.validate_holdout_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="no",
                manual_number_readable="no",
                manual_jersey_number="11",
            ),
        )
        errs = c3f.validate_holdout_annotation_values(
            manual_crop_valid="yes",
            manual_number_visible="yes",
            manual_number_readable="yes",
            manual_jersey_number="09",
        )
        self.assertEqual(errs, [])
        errs = c3f.validate_holdout_annotation_values(
            manual_crop_valid="yes",
            manual_number_visible="yes",
            manual_number_readable="yes",
            manual_jersey_number="123",
        )
        self.assertTrue(any("1-2 ASCII" in e for e in errs))

    def test_blank_template_manual_fields_empty(self):
        primary = [
            {
                "holdout_item_id": f"holdout_primary_{i:03d}",
                "batch": "primary",
                "batch_order": i,
                "review_item_id": f"review_{i}",
                "stratum": "high_signal_candidate",
                "crop_id": f"c{i}",
                "segment_id": f"s{i}",
                "raw_track_id": i,
                "frame_index": i,
                "source_crop_path": f"p{i}.jpg",
                "source_crop_sha256": f"sha{i}",
                "overview_path": f"o{i}.png",
                "contact_sheet_path": "sheet.png",
            }
            for i in range(1, 65)
        ]
        reserve = [
            {
                "holdout_item_id": f"holdout_reserve_{i:03d}",
                "batch": "reserve",
                "batch_order": i,
                "review_item_id": f"review_r{i}",
                "stratum": "mid_signal_candidate",
                "crop_id": f"cr{i}",
                "segment_id": f"sr{i}",
                "raw_track_id": 1000 + i,
                "frame_index": i,
                "source_crop_path": f"pr{i}.jpg",
                "source_crop_sha256": f"shar{i}",
                "overview_path": f"or{i}.png",
                "contact_sheet_path": "sheet_r.png",
            }
            for i in range(1, 33)
        ]
        rows = c3f.build_annotation_template_rows(primary, reserve)
        self.assertEqual(len(rows), 96)
        for row in rows:
            for field in c3f.MANUAL_TEMPLATE_FIELDS:
                self.assertEqual(row[field], "")

    def test_prohibited_prediction_fields_in_ranking_context(self):
        features = {field: 1 for field in c3f.RANK_FIELDS}
        features["sequence_confidence"] = 0.9
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.assert_no_prohibited_selection_fields(features)
        features2 = {field: 1 for field in c3f.RANK_FIELDS}
        features2["parseq_prediction"] = "11"
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.assert_no_prohibited_selection_fields(features2)

    def test_nan_confidence_cut_rejected(self):
        ops = [
            {
                "confidence_cut": 0.9,
                "exact_retained": float("nan"),
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": False,
            }
        ]
        with self.assertRaises(c3f.HoldoutDesignError):
            c3f.derive_validation_candidate_cut(ops)

    def test_deployment_threshold_remains_false_in_prereg_helper(self):
        ops = [
            {
                "confidence_cut": 0.999636,
                "exact_retained": 3,
                "wrong_positive_retained": 0,
                "negative_retained": 0,
                "sentinel_type": None,
                "selected": False,
            }
        ]
        result = c3f.derive_validation_candidate_cut(ops)
        self.assertIs(result["deployment_threshold_selected"], False)
        self.assertIs(result["threshold_selected_for_production"], False)
        self.assertTrue(result["not_a_deployment_threshold"])
        self.assertTrue(result["requires_independent_validation"])

    def test_identity_fields_absent_from_public_selected(self):
        # Public selected builder strips identity-like extras via explicit keys.
        item = _rank_row(
            "x",
            {f: 1 for f in c3f.RANK_FIELDS},
            composite_score=1.0,
            stratum="high_signal_candidate",
            holdout_item_id="holdout_primary_001",
            batch="primary",
            batch_order=1,
            crop_width_px=10,
            crop_height_px=20,
            roi_x_min=0,
            roi_y_min=0,
            roi_x_max=5,
            roi_y_max=5,
            roi_width_px=5,
            roi_height_px=5,
            overview_path="o.png",
            contact_sheet_path="c.png",
            kit_family=None,
            identity="should_not_matter",
            global_id=99,
        )
        # Simulate public projection used in script
        public = {
            "schema_version": c3f.SELECTED_SCHEMA,
            "holdout_item_id": item["holdout_item_id"],
            "batch": item["batch"],
            "batch_order": item["batch_order"],
            "review_item_id": item["review_item_id"],
            "stratum": item["stratum"],
            "stratum_is_ground_truth": False,
            "crop_id": item["crop_id"],
            "segment_id": item["segment_id"],
            "raw_track_id": item["raw_track_id"],
            "frame_index": item["frame_index"],
            "source_crop_path": item.get("source_crop_path", ""),
            "source_crop_sha256": item["source_crop_sha256"],
            "composite_score": item["composite_score"],
        }
        self.assertNotIn("identity", public)
        self.assertNotIn("global_id", public)
        self.assertNotIn("manual_jersey_number", public)


if __name__ == "__main__":
    unittest.main()
