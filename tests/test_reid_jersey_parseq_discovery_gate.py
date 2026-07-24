"""Unit tests for Stage 5C-R6 discovery-primary PARSeq gate helpers."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
_SRC = _PROJECT_ROOT / "src"
for p in (_SCRIPTS, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_reid_jersey_parseq_discovery_gate as dg  # noqa: E402
from football_analytics.reid import jersey_parseq as jp  # noqa: E402


class DiscoveryGateUnitTests(unittest.TestCase):
    def test_leading_zero_preserved_and_valid_regex(self):
        self.assertEqual(jp.extract_digit_candidate("07"), ("07", None))
        self.assertTrue(dg.DIGIT_RE.fullmatch("07"))
        self.assertTrue(dg.DIGIT_RE.fullmatch("0"))
        self.assertFalse(dg.DIGIT_RE.fullmatch("123"))
        self.assertFalse(dg.DIGIT_RE.fullmatch(""))
        self.assertFalse(dg.DIGIT_RE.fullmatch("7a"))

    def test_annotation_class_mapping(self):
        self.assertEqual(dg.annotation_class("yes"), "readable_positive")
        self.assertEqual(dg.annotation_class("no"), "non_readable_negative")
        self.assertEqual(dg.annotation_class("uncertain"), "uncertain_excluded")

    def test_evaluate_outcomes_matrix(self):
        items = [
            {
                "split_item_id": "a",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
            },
            {
                "split_item_id": "b",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
            },
            {
                "split_item_id": "c",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
            },
            {
                "split_item_id": "d",
                "annotation_class": "non_readable_negative",
                "manual_jersey_number": "",
            },
            {
                "split_item_id": "e",
                "annotation_class": "non_readable_negative",
                "manual_jersey_number": "",
            },
            {
                "split_item_id": "f",
                "annotation_class": "uncertain_excluded",
                "manual_jersey_number": "",
            },
        ]
        preds = [
            {
                "split_item_id": "a",
                "valid_jersey_string": True,
                "accepted_prediction": "8",
                "confidence": 0.9,
                "normalized_prediction": "8",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "b",
                "valid_jersey_string": True,
                "accepted_prediction": "9",
                "confidence": 0.8,
                "normalized_prediction": "9",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "c",
                "valid_jersey_string": False,
                "accepted_prediction": None,
                "confidence": None,
                "normalized_prediction": "",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "d",
                "valid_jersey_string": True,
                "accepted_prediction": "1",
                "confidence": 0.7,
                "normalized_prediction": "1",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "e",
                "valid_jersey_string": False,
                "accepted_prediction": None,
                "confidence": None,
                "normalized_prediction": "",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "f",
                "valid_jersey_string": True,
                "accepted_prediction": "2",
                "confidence": 0.6,
                "normalized_prediction": "2",
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
        ]
        rows = dg.evaluate_items(items, preds)
        by = {r["split_item_id"]: r["outcome"] for r in rows}
        self.assertEqual(by["a"], "exact")
        self.assertEqual(by["b"], "wrong")
        self.assertEqual(by["c"], "no_prediction")
        self.assertEqual(by["d"], "negative_digit_emission")
        self.assertEqual(by["e"], "negative_no_prediction")
        self.assertEqual(by["f"], "uncertain_digit_emission")

    def test_operating_points_ge_boundary_and_sentinel(self):
        eval_rows = [
            {
                "split_item_id": "p1",
                "annotation_class": "readable_positive",
                "outcome": "exact",
                "valid_jersey_string": True,
                "confidence": 0.5,
            },
            {
                "split_item_id": "p2",
                "annotation_class": "readable_positive",
                "outcome": "wrong",
                "valid_jersey_string": True,
                "confidence": 0.4,
            },
            {
                "split_item_id": "n1",
                "annotation_class": "non_readable_negative",
                "outcome": "negative_digit_emission",
                "valid_jersey_string": True,
                "confidence": 0.3,
            },
        ]
        ops = dg.build_operating_points(eval_rows)
        self.assertTrue(any(o["sentinel_type"] == "no_acceptance" for o in ops))
        # cut 0.5 accepts only p1 (>=)
        op05 = next(o for o in ops if o["confidence_cut"] == 0.5)
        self.assertEqual(op05["accepted_exact"], 1)
        self.assertEqual(op05["accepted_wrong_positive"], 0)
        self.assertEqual(op05["accepted_negative"], 0)
        self.assertTrue(op05["safety_eligible"])
        # cut 0.4 includes wrong positive
        op04 = next(o for o in ops if o["confidence_cut"] == 0.4)
        self.assertEqual(op04["accepted_wrong_positive"], 1)
        self.assertFalse(op04["safety_eligible"])

    def test_minimum_cut_selection_among_safe(self):
        ops = [
            {
                "operating_point_id": "op_b",
                "confidence_cut": 0.8,
                "confidence_cut_exact_decimal": "0.8",
                "confidence_cut_float64_hex": dg.float64_hex(0.8),
                "sentinel_type": None,
                "accepted_exact": 2,
                "accepted_wrong_positive": 0,
                "accepted_negative": 0,
                "accepted_uncertain": 0,
                "accepted_item_ids": ["a", "b"],
            },
            {
                "operating_point_id": "op_a",
                "confidence_cut": 0.5,
                "confidence_cut_exact_decimal": "0.5",
                "confidence_cut_float64_hex": dg.float64_hex(0.5),
                "sentinel_type": None,
                "accepted_exact": 1,
                "accepted_wrong_positive": 0,
                "accepted_negative": 0,
                "accepted_uncertain": 0,
                "accepted_item_ids": ["a"],
            },
            {
                "operating_point_id": "op_sentinel_no_acceptance",
                "confidence_cut": 1.5,
                "confidence_cut_exact_decimal": "1.5",
                "confidence_cut_float64_hex": dg.float64_hex(1.5),
                "sentinel_type": "no_acceptance",
                "accepted_exact": 0,
                "accepted_wrong_positive": 0,
                "accepted_negative": 0,
                "accepted_uncertain": 0,
                "accepted_item_ids": [],
            },
        ]
        gate = dg.derive_candidate_gate(ops)
        self.assertEqual(gate["gate_status"], "DISCOVERY_SAFE_CANDIDATE_GATE_DERIVED")
        self.assertEqual(gate["candidate_confidence_cut"], 0.5)
        self.assertTrue(gate["threshold_selected"])
        self.assertFalse(gate["deployment_threshold_selected"])

    def test_no_safe_gate_path(self):
        ops = [
            {
                "operating_point_id": "op_x",
                "confidence_cut": 0.9,
                "confidence_cut_exact_decimal": "0.9",
                "confidence_cut_float64_hex": dg.float64_hex(0.9),
                "sentinel_type": None,
                "accepted_exact": 1,
                "accepted_wrong_positive": 1,
                "accepted_negative": 0,
                "accepted_uncertain": 0,
                "accepted_item_ids": ["a", "b"],
            }
        ]
        gate = dg.derive_candidate_gate(ops)
        self.assertEqual(gate["gate_status"], "NO_SAFE_DISCOVERY_CANDIDATE_GATE")
        self.assertFalse(gate["threshold_selected"])
        self.assertTrue(gate["holdout_should_not_open"])

    def test_nan_confidence_rejected_in_operating_points(self):
        eval_rows = [
            {
                "split_item_id": "p1",
                "annotation_class": "readable_positive",
                "outcome": "exact",
                "valid_jersey_string": True,
                "confidence": float("nan"),
            }
        ]
        with self.assertRaises(dg.DiscoveryGateError):
            dg.build_operating_points(eval_rows)

    def test_determinism_audit_detects_mismatch(self):
        a = [
            {
                "split_item_id": f"discovery_primary_{i:03d}",
                "normalized_prediction": "1",
                "valid_jersey_string": True,
                "confidence": 0.1,
            }
            for i in range(1, 41)
        ]
        b = [dict(x) for x in a]
        b[0]["confidence"] = 0.2
        with self.assertRaises(dg.DiscoveryGateError):
            dg.assert_deterministic(a, b)

    def test_config_schema_and_forbidden_flags(self):
        cfg = dg.load_config(
            _PROJECT_ROOT
            / "configs/reid/jersey_parseq_discovery_gate_stage5c_rebuild_r2.yaml"
        )
        self.assertEqual(cfg["schema_version"], dg.CONFIG_SCHEMA)
        self.assertTrue(cfg["gate_derivation"]["historical_c3e_threshold_forbidden"])
        self.assertTrue(
            cfg["gate_derivation"]["discovery_reserve_forbidden_for_threshold_search"]
        )
        self.assertEqual(cfg["expected_item_count"], 40)


if __name__ == "__main__":
    unittest.main()
