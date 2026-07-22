"""Unit tests for Stage 5C-C3D PARSeq recognizer-only smoke adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid import jersey_parseq as jp


class FakeTokenizer:
    eos_id = 0
    bos_id = 95
    pad_id = 96

    def decode(self, probs):
        import torch

        # probs: [1,3,11]
        ids = probs.argmax(-1)[0].tolist()
        # map 1-10 -> digits; 0 -> EOS
        chars = []
        for i in ids:
            if i == 0:
                break
            if 1 <= i <= 10:
                chars.append(str(i - 1))
        token_probs = probs[0].max(-1).values
        # include EOS position probability like Tokenizer._filter
        try:
            eos_idx = ids.index(0)
            token_probs = token_probs[: eos_idx + 1]
        except ValueError:
            pass
        return ["".join(chars)], [token_probs]


class FakeHParams:
    charset_train = "0123456789abcdefghijklmnopqrstuvwxyz"
    charset_test = "0123456789abcdefghijklmnopqrstuvwxyz"
    max_label_length = 25
    img_size = [32, 128]
    patch_size = [4, 8]
    embed_dim = 384


class FakeModel:
    def __init__(self, text: str = "7"):
        self.hparams = FakeHParams()
        self.tokenizer = FakeTokenizer()
        self.head = mock.Mock(out_features=95)
        self._text = text
        self.training = False

    def eval(self):
        self.training = False
        return self

    def to(self, device):
        return self

    def parameters(self):
        import torch

        p = torch.nn.Parameter(torch.zeros(1))
        return iter([p])

    def __call__(self, tensor):
        import torch

        logits = torch.zeros(1, 26, 95)
        # encode desired digits into first positions of classes 1..10
        for idx, ch in enumerate(self._text[:2]):
            logits[0, idx, int(ch) + 1] = 10.0
        logits[0, len(self._text[:2]), 0] = 10.0  # EOS
        return logits


class JerseyPARSeqUnitTests(unittest.TestCase):
    def test_digit_acceptance_matrix(self):
        self.assertEqual(jp.extract_digit_candidate("7"), ("7", None))
        self.assertEqual(jp.extract_digit_candidate("07"), ("07", None))
        self.assertEqual(jp.extract_digit_candidate("00"), ("00", None))
        self.assertEqual(jp.extract_digit_candidate(""), (None, "empty_text"))
        self.assertEqual(jp.extract_digit_candidate("123"), (None, "digit_count_exceeds_max"))
        self.assertEqual(jp.extract_digit_candidate("7a"), (None, "non_digit_text"))
        self.assertEqual(jp.extract_digit_candidate("7!"), (None, "non_digit_text"))
        # no substitutions
        self.assertEqual(jp.extract_digit_candidate("O"), (None, "non_digit_text"))
        self.assertEqual(jp.extract_digit_candidate("l"), (None, "non_digit_text"))

    def test_bgr_to_rgb(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[0, 0] = [10, 20, 30]
        image = jp.bgr_to_rgb_pil(bgr)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(list(image.getpixel((0, 0))), [30, 20, 10])

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ckpt"
            data = b"abc"
            path.write_bytes(data)
            digest = jp.sha256_file(path)
            out = jp.validate_checkpoint_asset(path, digest, len(data))
            self.assertEqual(out, path.resolve())
            with self.assertRaises(jp.JerseyPARSeqError):
                jp.validate_checkpoint_asset(path, digest, 1)
            bad = Path(tmp) / "parseq-bb5792a6.pt"
            bad.write_bytes(data)
            with self.assertRaises(jp.JerseyPARSeqError):
                jp.validate_checkpoint_asset(bad, jp.sha256_file(bad), len(data))

    def test_prediction_forbidden_fields(self):
        with self.assertRaises(jp.JerseyPARSeqError):
            jp.assert_prediction_blind({"manual_jersey_number": "7"})

    def test_evidence_and_decision_classes(self):
        labels, decision = jp.build_evidence_and_decision(1, 1, 0, True)
        self.assertIn("PARSEQ_EXACT_SIGNAL_PRESENT", labels)
        self.assertEqual(decision, "GO_STAGE5C_C3E_PARSEQ_SIGNAL_FEASIBILITY")
        labels, decision = jp.build_evidence_and_decision(1, 1, 2, True)
        self.assertEqual(decision, "GO_STAGE5C_C3E_PARSEQ_FALSE_POSITIVE_AUDIT")
        labels, decision = jp.build_evidence_and_decision(0, 3, 0, True)
        self.assertEqual(decision, "GO_STAGE5C_C3E_PARSEQ_INPUT_COMPATIBILITY_AUDIT")
        labels, decision = jp.build_evidence_and_decision(0, 0, 0, True)
        self.assertEqual(decision, "CLOSE_CURRENT_PARSEQ_ROI_SMOKE_PATH")
        self.assertIn("NO_NEGATIVE_DIGIT_EMISSION_OBSERVED", labels)

    def test_decode_jersey_logits(self):
        import torch

        model = FakeModel("12")
        logits = model(torch.zeros(1, 3, 32, 128))
        decoded = jp.decode_jersey_logits(model, logits)
        self.assertEqual(decoded["raw_decoded_text"], "12")
        self.assertEqual(decoded["confidence_method"].startswith("product_of_"), True)

    def test_evaluate_metric_sums_and_join(self):
        predictions = []
        references = []
        for i in range(46):
            sel = "POS_readable" if i < 20 else "A_not_visible"
            rid = f"id_{i}"
            predictions.append(
                {
                    "review_item_id": rid,
                    "pilot_index": i + 1,
                    "selection_class": sel,
                    "source_crop_sha256": f"sha{i}",
                    "accepted_prediction": "7" if i < 5 else None,
                    "raw_decoded_text": "7" if i < 5 else "",
                    "normalized_text": "7" if i < 5 else "",
                    "sequence_confidence": 0.5,
                    "rejection_reason": None if i < 5 else "empty_text",
                    "inference_error": None,
                    "inference_ms": 1.0,
                }
            )
            references.append(
                {
                    "review_item_id": rid,
                    "pilot_index": i + 1,
                    "selection_class": sel,
                    "source_crop_sha256": f"sha{i}",
                    "manual_jersey_number": "7" if i < 20 else "",
                    "manual_digit_count": "1" if i < 20 else "0",
                }
            )
        # Force first five positive exact, rest positive no_prediction; negatives rejected
        for i in range(20):
            predictions[i]["accepted_prediction"] = "7" if i < 5 else None
            predictions[i]["normalized_text"] = "7" if i < 5 else ""
            predictions[i]["rejection_reason"] = None if i < 5 else "empty_text"
        counts = {
            "POS_readable": 20,
            "A_not_visible": 26,
            "B_visible_unreadable": 0,
            "C_uncertain_signal": 0,
            "D_uncertain_crop": 0,
            "E_invalid": 0,
        }
        # Adjust class labels for remaining negatives to match counts used only as totals
        # build_results_summary checks sum(expected)=46; use actual distribution:
        counts = {
            "POS_readable": 20,
            "A_not_visible": 26,
        }
        # Expand expected to include zeros for unused? sum must match len
        item_rows, summary = jp.evaluate_predictions(predictions, references, counts)
        self.assertEqual(len(item_rows), 46)
        self.assertEqual(summary["positive_metrics"]["exact_match_count"], 5)
        self.assertEqual(summary["positive_metrics"]["no_prediction_count"], 15)
        self.assertEqual(summary["positive_metrics"]["wrong_number_count"], 0)
        self.assertEqual(summary["negative_metrics"]["accepted_number_emission_count"], 0)
        self.assertEqual(
            summary["decision_class"], "GO_STAGE5C_C3E_PARSEQ_SIGNAL_FEASIBILITY"
        )

    def test_duplicate_and_sha_join_rejection(self):
        predictions = [
            {
                "review_item_id": "a",
                "pilot_index": 1,
                "selection_class": "POS_readable",
                "source_crop_sha256": "x",
                "accepted_prediction": None,
                "inference_error": None,
                "inference_ms": 1,
            }
        ]
        references = [
            {
                "review_item_id": "a",
                "pilot_index": 1,
                "selection_class": "POS_readable",
                "source_crop_sha256": "y",
                "manual_jersey_number": "1",
            }
        ]
        with self.assertRaises(jp.JerseyPARSeqError):
            jp.evaluate_predictions(predictions, references, {"POS_readable": 1})

    def test_network_policy_offline(self):
        audit = jp.parse_network_strace("")
        self.assertEqual(audit["policy_status"], "pass_offline")

    def test_no_identity_fields_in_summary_safety(self):
        labels, decision = jp.build_evidence_and_decision(0, 0, 0, True)
        summary = jp.build_results_summary(
            [
                {
                    "selection_class": "POS_readable",
                    "outcome": "no_prediction",
                    "accepted_prediction": None,
                    "manual_jersey_number": "9",
                    "manual_digit_count": "1",
                    "sequence_confidence": None,
                }
            ]
            * 20
            + [
                {
                    "selection_class": "A_not_visible",
                    "outcome": "rejected",
                    "accepted_prediction": None,
                    "manual_jersey_number": "",
                    "manual_digit_count": "0",
                    "sequence_confidence": None,
                }
            ]
            * 26,
            [{"selection_class": "POS_readable", "normalized_text": "", "inference_error": None, "rejection_reason": "empty_text"}]
            * 20
            + [
                {
                    "selection_class": "A_not_visible",
                    "normalized_text": "",
                    "inference_error": None,
                    "rejection_reason": "empty_text",
                    "raw_output_shape": [1, 26, 95],
                }
            ]
            * 26,
            {"POS_readable": 20, "A_not_visible": 26},
        )
        self.assertFalse(summary["safety_flags"]["identity_assigned"])
        self.assertFalse(summary["safety_flags"]["threshold_selected"])
        self.assertFalse(summary["safety_flags"]["accuracy_claimed"])
        self.assertEqual(decision, "CLOSE_CURRENT_PARSEQ_ROI_SMOKE_PATH")


if __name__ == "__main__":
    unittest.main()
