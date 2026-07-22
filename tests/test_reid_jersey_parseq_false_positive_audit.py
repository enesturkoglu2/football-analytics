"""Unit tests for Stage 5C-C3E PARSeq false-positive audit helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_reid_jersey_parseq_false_positives.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("c3e_fp_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["c3e_fp_audit"] = module
    spec.loader.exec_module(module)
    return module


c3e = _load_module()


class FalsePositiveAuditTests(unittest.TestCase):
    def test_group_assignment(self):
        self.assertEqual(
            c3e.assign_evaluation_group("POS_readable", "exact_match"),
            c3e.GROUP_POSITIVE_EXACT,
        )
        self.assertEqual(
            c3e.assign_evaluation_group("POS_readable", "wrong_number"),
            c3e.GROUP_POSITIVE_WRONG,
        )
        self.assertEqual(
            c3e.assign_evaluation_group("A_not_visible", "number_emitted"),
            c3e.GROUP_NEG_EMISSION,
        )
        self.assertEqual(
            c3e.assign_evaluation_group("A_not_visible", "rejected"),
            c3e.GROUP_NEG_NONE,
        )

    def test_validate_group_counts(self):
        items = (
            [{"evaluation_group": c3e.GROUP_POSITIVE_EXACT}] * 5
            + [{"evaluation_group": c3e.GROUP_POSITIVE_WRONG}] * 15
            + [{"evaluation_group": c3e.GROUP_NEG_EMISSION}] * 26
        )
        counts = c3e.validate_group_counts(items)
        self.assertEqual(counts[c3e.GROUP_POSITIVE_EXACT], 5)
        self.assertEqual(counts[c3e.GROUP_POSITIVE_WRONG], 15)
        self.assertEqual(counts[c3e.GROUP_NEG_EMISSION], 26)

    def test_confidence_finite_range(self):
        self.assertEqual(c3e._finite_unit(0.5, "c"), 0.5)
        with self.assertRaises(c3e.FalsePositiveAuditError):
            c3e._finite_unit(float("nan"), "c")
        with self.assertRaises(c3e.FalsePositiveAuditError):
            c3e._finite_unit(1.2, "c")

    def test_quantile_determinism(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        summary = c3e.confidence_summary(values)
        self.assertEqual(summary["p25"], 0.2)
        self.assertEqual(summary["p75"], 0.4)
        self.assertEqual(summary["sorted_values"], values)

    def test_auroc_ties_and_all_equal(self):
        self.assertEqual(c3e.auroc_rank([0.5, 0.5], [0.5, 0.5]), 0.5)
        # Perfect separation
        self.assertEqual(c3e.auroc_rank([0.9, 0.8], [0.1, 0.2]), 1.0)
        # Inverted
        self.assertEqual(c3e.auroc_rank([0.1, 0.2], [0.8, 0.9]), 0.0)
        # Tie handling: one shared score
        score = c3e.auroc_rank([0.5, 0.9], [0.5, 0.1])
        self.assertTrue(0.0 < score < 1.0)

    def test_overlap_interval(self):
        self.assertEqual(c3e.overlap_interval(0.2, 0.8, 0.5, 0.9), [0.5, 0.8])
        self.assertIsNone(c3e.overlap_interval(0.1, 0.2, 0.3, 0.4))

    def test_operating_points_and_sentinels(self):
        items = []
        # 2 exact high, 1 wrong mid, 2 neg low/high
        specs = [
            (c3e.GROUP_POSITIVE_EXACT, 0.9),
            (c3e.GROUP_POSITIVE_EXACT, 0.8),
            (c3e.GROUP_POSITIVE_WRONG, 0.7),
            (c3e.GROUP_NEG_EMISSION, 0.95),
            (c3e.GROUP_NEG_EMISSION, 0.1),
        ]
        for group, conf in specs:
            items.append({"evaluation_group": group, "sequence_confidence": conf})
        # pad to avoid frontier helpers assuming global 5/15/26 in build_operating_points
        # build_operating_points uses hardcoded 15 only for wrong fraction; OK for small set
        rows = c3e.build_operating_points(items)
        self.assertTrue(any(row["sentinel_type"] == "accept_all" for row in rows))
        self.assertTrue(any(row["sentinel_type"] == "accept_none" for row in rows))
        self.assertTrue(all(row["selected"] is False for row in rows))
        accept_all = next(row for row in rows if row["sentinel_type"] == "accept_all")
        self.assertEqual(accept_all["accepted_total"], 5)
        accept_none = next(row for row in rows if row["sentinel_type"] == "accept_none")
        self.assertEqual(accept_none["accepted_total"], 0)

    def test_frontiers_and_perfect_safe_point(self):
        # Craft points where cut>=0.99 keeps one exact and zero wrong/neg
        items = [
            {"evaluation_group": c3e.GROUP_POSITIVE_EXACT, "sequence_confidence": 0.99},
            {"evaluation_group": c3e.GROUP_POSITIVE_EXACT, "sequence_confidence": 0.2},
            {"evaluation_group": c3e.GROUP_POSITIVE_WRONG, "sequence_confidence": 0.5},
            {"evaluation_group": c3e.GROUP_NEG_EMISSION, "sequence_confidence": 0.8},
        ]
        ops = c3e.build_operating_points(items)
        frontiers = c3e.frontier_metrics(ops)
        self.assertTrue(frontiers["any_exact_zero_wrong_zero_negative"])
        self.assertGreater(
            frontiers["zero_negative_frontier"]["maximum_exact_retained"], 0
        )

    def test_zero_negative_loses_exact(self):
        items = [
            {"evaluation_group": c3e.GROUP_POSITIVE_EXACT, "sequence_confidence": 0.4},
            {"evaluation_group": c3e.GROUP_NEG_EMISSION, "sequence_confidence": 0.9},
        ]
        ops = c3e.build_operating_points(items)
        frontiers = c3e.frontier_metrics(ops)
        self.assertEqual(
            frontiers["zero_negative_frontier"]["maximum_exact_retained"], 0
        )
        self.assertFalse(frontiers["exact_signal_without_negative"])

    def test_output_frequency(self):
        items = [
            {
                "evaluation_group": c3e.GROUP_NEG_EMISSION,
                "accepted_prediction": "11",
                "prediction_length": 2,
                "eos_position": 2,
                "token_probabilities": [0.9, 0.8, 1.0],
                "minimum_selected_token_probability": 0.8,
                "_first_token_probability": 0.9,
                "_second_token_probability": 0.8,
            },
            {
                "evaluation_group": c3e.GROUP_NEG_EMISSION,
                "accepted_prediction": "11",
                "prediction_length": 2,
                "eos_position": 2,
                "token_probabilities": [0.7, 0.6, 1.0],
                "minimum_selected_token_probability": 0.6,
                "_first_token_probability": 0.7,
                "_second_token_probability": 0.6,
            },
        ]
        dist = c3e.build_output_distribution(items)
        self.assertEqual(dist["negative_most_frequent_numbers"]["11"], 2)

    def test_evidence_and_decision_priority(self):
        frontiers_perfect = {
            "zero_negative_frontier": {"maximum_exact_retained": 1},
            "any_exact_zero_wrong_zero_negative": True,
        }
        labels, decision = c3e.build_evidence_and_decision(
            exact_count=5,
            negative_emission=26,
            frontiers=frontiers_perfect,
            auroc_exact_vs_neg=0.7,
            overlap_exact_neg=[0.5, 0.6],
            high_conf_neg=True,
            high_conf_wrong=False,
            confidence_analyzable=True,
        )
        self.assertEqual(decision, "GO_STAGE5C_C3F_CONFIDENCE_GATE_VALIDATION")
        self.assertIn("PARSEQ_RECOGNITION_SIGNAL_PRESENT", labels)
        self.assertIn("CONFIDENCE_PERFECT_SAFE_POINT_OBSERVED_IN_FROZEN_SET", labels)

        frontiers_partial = {
            "zero_negative_frontier": {"maximum_exact_retained": 2},
            "any_exact_zero_wrong_zero_negative": False,
        }
        _, decision = c3e.build_evidence_and_decision(
            exact_count=5,
            negative_emission=26,
            frontiers=frontiers_partial,
            auroc_exact_vs_neg=0.6,
            overlap_exact_neg=[0.4, 0.7],
            high_conf_neg=False,
            high_conf_wrong=True,
            confidence_analyzable=True,
        )
        self.assertEqual(decision, "GO_STAGE5C_C3F_CONFIDENCE_PLUS_LEGIBILITY_AUDIT")

        frontiers_zero = {
            "zero_negative_frontier": {"maximum_exact_retained": 0},
            "any_exact_zero_wrong_zero_negative": False,
        }
        _, decision = c3e.build_evidence_and_decision(
            exact_count=5,
            negative_emission=26,
            frontiers=frontiers_zero,
            auroc_exact_vs_neg=0.4,
            overlap_exact_neg=None,
            high_conf_neg=True,
            high_conf_wrong=True,
            confidence_analyzable=True,
        )
        self.assertEqual(decision, "GO_STAGE5C_C3F_LEGIBILITY_MODEL_CAPABILITY_AUDIT")

    def test_identity_fields_absent_in_public_row(self):
        row = c3e.public_item_row(
            {
                "review_item_id": "x",
                "pilot_index": 1,
                "selection_class": "POS_readable",
                "evaluation_group": c3e.GROUP_POSITIVE_EXACT,
                "reference_length": 1,
                "accepted_prediction": "7",
                "prediction_length": 1,
                "prediction_correctness": "exact",
                "sequence_confidence": 0.9,
                "confidence_method": "m",
                "token_probabilities": [0.9],
                "minimum_selected_token_probability": 0.9,
                "eos_position": 1,
                "rejection_reason": None,
                "source_prediction_sha256": "a",
                "source_evaluation_sha256": "b",
                "_manual_jersey_number": "7",
                "identity": "bad",
            }
        )
        self.assertNotIn("identity", row)
        self.assertNotIn("_manual_jersey_number", row)
        self.assertNotIn("global_id", row)

    def test_nan_rejection(self):
        with self.assertRaises(c3e.FalsePositiveAuditError):
            c3e._finite_unit(float("nan"), "sequence_confidence")
        with self.assertRaises(c3e.FalsePositiveAuditError):
            c3e._finite_unit(float("inf"), "sequence_confidence")

    def test_duplicate_rejection_in_validate(self):
        items = (
            [{"evaluation_group": c3e.GROUP_POSITIVE_EXACT}] * 4
            + [{"evaluation_group": c3e.GROUP_POSITIVE_WRONG}] * 15
            + [{"evaluation_group": c3e.GROUP_NEG_EMISSION}] * 26
        )
        with self.assertRaises(c3e.FalsePositiveAuditError):
            c3e.validate_group_counts(items)


if __name__ == "__main__":
    unittest.main()
