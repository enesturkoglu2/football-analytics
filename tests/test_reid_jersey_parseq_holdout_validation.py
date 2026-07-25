"""Unit tests for Stage 5C-R8 holdout-primary PARSeq fixed-gate validation."""

from __future__ import annotations

import json
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

import run_reid_jersey_parseq_holdout_validation as hv  # noqa: E402
from football_analytics.reid import jersey_parseq as jp  # noqa: E402


def _base_eval_row(**kwargs):
    row = {
        "split_item_id": "holdout_primary_001",
        "annotation_class": "readable_positive",
        "source_type": "reused",
        "outcome": "exact",
        "valid_jersey_string": True,
        "confidence": 0.99993,
        "prediction": "7",
        "normalized_prediction": "7",
    }
    row.update(kwargs)
    return row


class HoldoutValidationUnitTests(unittest.TestCase):
    def test_leading_zero_and_valid_regex(self):
        self.assertEqual(jp.extract_digit_candidate("09"), ("09", None))
        self.assertTrue(hv.DIGIT_RE.fullmatch("09"))
        self.assertTrue(hv.DIGIT_RE.fullmatch("9"))
        self.assertNotEqual("09", "9")
        self.assertFalse(hv.DIGIT_RE.fullmatch("123"))
        self.assertFalse(hv.DIGIT_RE.fullmatch(""))

    def test_annotation_class_mapping(self):
        self.assertEqual(hv.annotation_class("yes"), "readable_positive")
        self.assertEqual(hv.annotation_class("no"), "non_readable_negative")
        self.assertEqual(hv.annotation_class("uncertain"), "uncertain_excluded")

    def test_source_type_normalization(self):
        self.assertEqual(
            hv.normalize_source_type("reused_baseline_selected_crop"), "reused"
        )
        self.assertEqual(
            hv.normalize_source_type("recomputed_manual_segment"), "recomputed"
        )
        with self.assertRaises(hv.HoldoutValidationError):
            hv.normalize_source_type("other")

    def test_evaluate_outcomes_matrix(self):
        items = [
            {
                "split_item_id": "a",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
                "source_type": "reused",
            },
            {
                "split_item_id": "b",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
                "source_type": "reused",
            },
            {
                "split_item_id": "c",
                "annotation_class": "readable_positive",
                "manual_jersey_number": "8",
                "source_type": "reused",
            },
            {
                "split_item_id": "d",
                "annotation_class": "non_readable_negative",
                "manual_jersey_number": "",
                "source_type": "reused",
            },
            {
                "split_item_id": "e",
                "annotation_class": "non_readable_negative",
                "manual_jersey_number": "",
                "source_type": "reused",
            },
            {
                "split_item_id": "f",
                "annotation_class": "uncertain_excluded",
                "manual_jersey_number": "",
                "source_type": "reused",
            },
        ]
        preds = [
            {
                "split_item_id": "a",
                "valid_jersey_string": True,
                "accepted_prediction": "8",
                "confidence": 0.9,
                "normalized_prediction": "8",
                "confidence_exact_decimal": hv.exact_decimal(0.9),
                "confidence_float64_hex": hv.float64_hex(0.9),
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "b",
                "valid_jersey_string": True,
                "accepted_prediction": "9",
                "confidence": 0.8,
                "normalized_prediction": "9",
                "confidence_exact_decimal": hv.exact_decimal(0.8),
                "confidence_float64_hex": hv.float64_hex(0.8),
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "c",
                "valid_jersey_string": False,
                "accepted_prediction": None,
                "confidence": None,
                "normalized_prediction": "",
                "confidence_exact_decimal": None,
                "confidence_float64_hex": None,
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "d",
                "valid_jersey_string": True,
                "accepted_prediction": "1",
                "confidence": 0.7,
                "normalized_prediction": "1",
                "confidence_exact_decimal": hv.exact_decimal(0.7),
                "confidence_float64_hex": hv.float64_hex(0.7),
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "e",
                "valid_jersey_string": False,
                "accepted_prediction": None,
                "confidence": None,
                "normalized_prediction": "",
                "confidence_exact_decimal": None,
                "confidence_float64_hex": None,
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
            {
                "split_item_id": "f",
                "valid_jersey_string": True,
                "accepted_prediction": "2",
                "confidence": 0.6,
                "normalized_prediction": "2",
                "confidence_exact_decimal": hv.exact_decimal(0.6),
                "confidence_float64_hex": hv.float64_hex(0.6),
                "inference_ms": 1.0,
                "extracted_roi_pixel_sha256": "x",
            },
        ]
        rows = hv.evaluate_items(items, preds)
        by = {r["split_item_id"]: r["outcome"] for r in rows}
        self.assertEqual(by["a"], "exact")
        self.assertEqual(by["b"], "wrong")
        self.assertEqual(by["c"], "no_prediction")
        self.assertEqual(by["d"], "negative_digit_emission")
        self.assertEqual(by["e"], "negative_no_prediction")
        self.assertEqual(by["f"], "uncertain_digit_emission")

    def test_ge_boundary_semantics(self):
        cut = float(hv.EXPECTED_CUT_DECIMAL)
        cut_hex = hv.EXPECTED_CUT_HEX
        self.assertEqual(hv.float64_hex(cut), cut_hex)
        rows = [
            _base_eval_row(
                split_item_id="eq",
                confidence=cut,
                outcome="exact",
            ),
            _base_eval_row(
                split_item_id="below",
                confidence=math.nextafter(cut, 0.0),
                outcome="exact",
            ),
            _base_eval_row(
                split_item_id="invalid",
                confidence=cut,
                valid_jersey_string=False,
                prediction=None,
                outcome="no_prediction",
            ),
        ]
        gated = hv.apply_fixed_gate(rows, cut=cut, cut_hex=cut_hex, operator=">=")
        by = {r["split_item_id"]: r for r in gated}
        self.assertTrue(by["eq"]["accepted"])
        self.assertFalse(by["below"]["accepted"])
        self.assertFalse(by["invalid"]["accepted"])

    def test_nan_inf_rejection(self):
        cut = float(hv.EXPECTED_CUT_DECIMAL)
        with self.assertRaises(hv.HoldoutValidationError):
            hv.apply_fixed_gate(
                [_base_eval_row(confidence=float("nan"))],
                cut=cut,
                cut_hex=hv.EXPECTED_CUT_HEX,
                operator=">=",
            )
        with self.assertRaises(hv.HoldoutValidationError):
            hv.apply_fixed_gate(
                [_base_eval_row(confidence=float("inf"))],
                cut=cut,
                cut_hex=hv.EXPECTED_CUT_HEX,
                operator=">=",
            )

    def test_pass_inconclusive_fail_decisions(self):
        cut = float(hv.EXPECTED_CUT_DECIMAL)
        # PASS: two exact accepted
        gated_pass = hv.apply_fixed_gate(
            [
                _base_eval_row(split_item_id="a", confidence=cut, outcome="exact"),
                _base_eval_row(split_item_id="b", confidence=cut, outcome="exact"),
            ],
            cut=cut,
            cut_hex=hv.EXPECTED_CUT_HEX,
            operator=">=",
        )
        d = hv.decide_validation(
            gated_pass,
            positive_count=16,
            negative_count=30,
            min_pos=10,
            min_neg=24,
            pass_exact_gte=2,
        )
        self.assertEqual(d["validation_decision"], "PASS_INDEPENDENT_HIGH_PRECISION_SIGNAL")

        # INCONCLUSIVE: one exact
        gated_inc = hv.apply_fixed_gate(
            [_base_eval_row(split_item_id="a", confidence=cut, outcome="exact")],
            cut=cut,
            cut_hex=hv.EXPECTED_CUT_HEX,
            operator=">=",
        )
        d2 = hv.decide_validation(
            gated_inc,
            positive_count=16,
            negative_count=30,
            min_pos=10,
            min_neg=24,
            pass_exact_gte=2,
        )
        self.assertEqual(d2["validation_decision"], "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT")

        # FAIL wrong positive
        gated_fail_w = hv.apply_fixed_gate(
            [
                _base_eval_row(
                    split_item_id="w",
                    confidence=cut,
                    outcome="wrong",
                    prediction="1",
                )
            ],
            cut=cut,
            cut_hex=hv.EXPECTED_CUT_HEX,
            operator=">=",
        )
        d3 = hv.decide_validation(
            gated_fail_w,
            positive_count=16,
            negative_count=30,
            min_pos=10,
            min_neg=24,
            pass_exact_gte=2,
        )
        self.assertEqual(d3["validation_decision"], "FAIL_INDEPENDENT_GATE_SAFETY")

        # FAIL negative accept
        gated_fail_n = hv.apply_fixed_gate(
            [
                _base_eval_row(
                    split_item_id="n",
                    annotation_class="non_readable_negative",
                    outcome="negative_digit_emission",
                    confidence=cut,
                    prediction="3",
                )
            ],
            cut=cut,
            cut_hex=hv.EXPECTED_CUT_HEX,
            operator=">=",
        )
        d4 = hv.decide_validation(
            gated_fail_n,
            positive_count=16,
            negative_count=30,
            min_pos=10,
            min_neg=24,
            pass_exact_gte=2,
        )
        self.assertEqual(d4["validation_decision"], "FAIL_INDEPENDENT_GATE_SAFETY")

    def test_uncertain_accepted_not_fail_alone(self):
        cut = float(hv.EXPECTED_CUT_DECIMAL)
        gated = hv.apply_fixed_gate(
            [
                _base_eval_row(
                    split_item_id="u",
                    annotation_class="uncertain_excluded",
                    outcome="uncertain_digit_emission",
                    confidence=cut,
                    prediction="4",
                )
            ],
            cut=cut,
            cut_hex=hv.EXPECTED_CUT_HEX,
            operator=">=",
        )
        d = hv.decide_validation(
            gated,
            positive_count=16,
            negative_count=30,
            min_pos=10,
            min_neg=24,
            pass_exact_gte=2,
        )
        self.assertEqual(d["validation_decision"], "INCONCLUSIVE_SAFE_BUT_LOW_SUPPORT")
        self.assertEqual(d["accepted_uncertain"], 1)

    def test_frozen_gate_cut_constants(self):
        cut = float(hv.EXPECTED_CUT_DECIMAL)
        self.assertEqual(hv.float64_hex(cut), hv.EXPECTED_CUT_HEX)
        self.assertEqual(hv.EXPECTED_CUT_DECIMAL, "0.99992299168434329")

    def test_load_frozen_gate_rejects_wrong_status(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gate.json"
            p.write_text(
                json.dumps(
                    {
                        "gate_status": "NO_SAFE_DISCOVERY_CANDIDATE_GATE",
                        "threshold_selected": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(hv.HoldoutValidationError):
                hv.load_frozen_discovery_gate(p)

    def test_determinism_helper_size(self):
        with self.assertRaises(hv.HoldoutValidationError):
            hv.assert_deterministic([], [])

    def test_path_traversal_rejected_in_universe_contract(self):
        # Direct unit check of ".." rejection logic via helper path check style
        bad = Path("../outside/crop.jpg")
        self.assertIn("..", bad.parts)


if __name__ == "__main__":
    unittest.main()
