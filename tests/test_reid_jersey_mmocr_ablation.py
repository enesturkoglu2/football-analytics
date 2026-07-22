"""Unit tests for the Stage 5C-C2 jersey MMOCR ablation additions.

No real checkpoint loading: detector/recognizer are mocks. The frozen C1
adapter contract is exercised separately in ``test_reid_jersey_mmocr.py``.
"""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid import jersey_mmocr as jm

EXPECTED_COUNTS = {
    "POS_readable": 20,
    "A_not_visible": 10,
    "B_visible_unreadable": 2,
    "C_uncertain_signal": 7,
    "D_uncertain_crop": 2,
    "E_invalid": 5,
}

MODEL_META = {
    "detector_model_id": "dbnet",
    "detector_config_sha256": "0" * 64,
    "detector_checkpoint_sha256": "0" * 64,
    "recognizer_model_id": "sar",
    "recognizer_config_sha256": "0" * 64,
    "recognizer_checkpoint_sha256": "0" * 64,
    "device": "cpu",
    "preprocessing_variant": "roi_bgr_no_preprocessing",
}

C1_BASELINE = {
    "variant_id": "c1_dbnet_sar_roi_1x_baseline",
    "positive_exact_match_count": 0,
    "positive_number_emission_count": 0,
    "positive_wrong_number_count": 0,
    "positive_no_prediction_count": 20,
    "negative_number_emission_count": 0,
    "detector_region_item_count": 1,
    "median_runtime_ms": 656.0,
}


class _RecordingDetector:
    """Mock TextDetInferencer recording input shapes."""

    def __init__(self, polygons=None, scores=None):
        self.polygons = polygons or []
        self.scores = scores or []
        self.calls: list[tuple[int, int]] = []

    def __call__(self, image, return_vis=False, progress_bar=False):
        self.calls.append(image.shape[:2])
        return {"predictions": [{"polygons": self.polygons, "scores": self.scores}]}


class _RecordingRecognizer:
    """Mock TextRecInferencer recording input shapes and returning fixed output."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, image, return_vis=False, progress_bar=False):
        self.calls.append(image.shape[:2])
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return {"predictions": [output]}


def make_crop(tmpdir: Path, width: int = 90, height: int = 120) -> tuple[Path, str]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    path = tmpdir / "crop.png"
    assert cv2.imwrite(str(path), image)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def make_item(crop_path: Path, crop_sha: str, pilot_index: int = 1, **overrides) -> dict:
    item = {
        "pilot_index": pilot_index,
        "review_item_id": f"review_item_{pilot_index:03d}",
        "crop_id": f"crop_{pilot_index:03d}",
        "segment_id": f"seg_{pilot_index:03d}",
        "raw_track_id": 100 + pilot_index,
        "frame_index": 1000 + pilot_index,
        "selection_class": "POS_readable",
        "source_crop_path": str(crop_path),
        "source_crop_sha256": crop_sha,
        "roi_x_min": 10,
        "roi_y_min": 10,
        "roi_x_max": 50,
        "roi_y_max": 70,
        "roi_source": "stage5a_number_search_roi",
    }
    item.update(overrides)
    return item


class TestVariantMatrix(unittest.TestCase):
    def test_exact_four_variants_and_order(self) -> None:
        self.assertEqual(
            jm.ABLATION_VARIANT_IDS,
            (
                "direct_sar_roi_1x",
                "direct_sar_roi_2x_cubic",
                "direct_sar_roi_4x_cubic",
                "dbnet_sar_roi_4x_cubic",
            ),
        )
        jm.validate_ablation_variants(jm.ABLATION_VARIANTS)

    def test_reordered_matrix_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "matrix mismatch"):
            jm.validate_ablation_variants(tuple(reversed(jm.ABLATION_VARIANTS)))

    def test_extra_variant_rejected(self) -> None:
        variants = list(jm.ABLATION_VARIANTS) + [
            {"variant_id": "direct_sar_roi_8x", "detector_used": False, "scale_factor": 8}
        ]
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.validate_ablation_variants(variants)


class TestDeterministicResize(unittest.TestCase):
    def setUp(self) -> None:
        self.roi = np.random.default_rng(0).integers(0, 255, (60, 40, 3), dtype=np.uint8)

    def test_1x_dimensions_unchanged_no_resize(self) -> None:
        out, resize_ms = jm.resize_roi_deterministic(self.roi, 1)
        self.assertIs(out, self.roi)
        self.assertEqual(resize_ms, 0.0)

    def test_2x_exact_dimensions(self) -> None:
        out, _ = jm.resize_roi_deterministic(self.roi, 2)
        self.assertEqual(out.shape[:2], (120, 80))

    def test_4x_exact_dimensions(self) -> None:
        out, _ = jm.resize_roi_deterministic(self.roi, 4)
        self.assertEqual(out.shape[:2], (240, 160))

    def test_aspect_ratio_preserved(self) -> None:
        for scale in (2, 4):
            out, _ = jm.resize_roi_deterministic(self.roi, scale)
            self.assertAlmostEqual(
                out.shape[1] / out.shape[0], self.roi.shape[1] / self.roi.shape[0]
            )

    def test_inter_cubic_is_used(self) -> None:
        out, _ = jm.resize_roi_deterministic(self.roi, 2)
        expected = cv2.resize(self.roi, (80, 120), interpolation=cv2.INTER_CUBIC)
        self.assertTrue(np.array_equal(out, expected))
        linear = cv2.resize(self.roi, (80, 120), interpolation=cv2.INTER_LINEAR)
        self.assertFalse(np.array_equal(out, linear))

    def test_bgr_convention_preserved(self) -> None:
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        roi[:, :, 0] = 200  # blue channel
        out, _ = jm.resize_roi_deterministic(roi, 2)
        self.assertEqual(int(out[..., 0].mean()), 200)
        self.assertEqual(int(out[..., 1].mean()), 0)

    def test_empty_roi_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "empty ROI"):
            jm.resize_roi_deterministic(np.zeros((0, 0, 3), dtype=np.uint8), 2)

    def test_invalid_scale_rejected(self) -> None:
        for scale in (0, 3, 8, -1):
            with self.assertRaises(jm.JerseyMMOCRError):
                jm.resize_roi_deterministic(self.roi, scale)

    def test_no_sharpening_or_contrast_helpers_exist(self) -> None:
        forbidden = [
            name
            for name in dir(jm)
            if any(token in name.lower() for token in ("sharpen", "clahe", "contrast", "denoise"))
        ]
        self.assertEqual(forbidden, [])


class TestCoordinateMapping(unittest.TestCase):
    def test_scaled_to_original_mapping(self) -> None:
        self.assertEqual(jm.scale_coords_to_original([40.0, 80.0, 120.0, 160.0], 4), [10.0, 20.0, 30.0, 40.0])

    def test_round_trip_tolerance(self) -> None:
        original = [3.0, 7.0, 21.0, 35.0]
        scaled = [value * 4 for value in original]
        back = jm.scale_coords_to_original(scaled, 4)
        for a, b in zip(original, back):
            self.assertLess(abs(a - b), 1e-9)

    def test_invalid_scale_rejected(self) -> None:
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.scale_coords_to_original([1.0], 3)


class TestDirectSarSelection(unittest.TestCase):
    def _candidate(self, index: int, text: str, score) -> dict:
        normalized = jm.normalize_recognized_text(text)
        digit, rejection = jm.extract_digit_candidate(normalized)
        return {
            "api_index": index,
            "raw_text": text,
            "normalized_text": normalized,
            "recognizer_score": score,
            "accepted_digit_candidate": digit,
            "candidate_rejection_reason": rejection,
        }

    def test_multi_candidate_highest_confidence_wins(self) -> None:
        result = jm.select_direct_sar_digit(
            [self._candidate(0, "7", 0.2), self._candidate(1, "10", 0.9)]
        )
        self.assertEqual(result["digit_string"], "10")

    def test_confidence_tie_resolves_by_api_order(self) -> None:
        result = jm.select_direct_sar_digit(
            [self._candidate(0, "7", 0.5), self._candidate(1, "9", 0.5)]
        )
        self.assertEqual(result["digit_string"], "7")
        self.assertEqual(result["api_index"], 0)

    def test_missing_confidence_deterministic_first_strict(self) -> None:
        result = jm.select_direct_sar_digit(
            [self._candidate(0, "SB", None), self._candidate(1, "8", None), self._candidate(2, "9", None)]
        )
        self.assertEqual(result["digit_string"], "8")

    def test_no_strict_candidate_returns_none(self) -> None:
        self.assertIsNone(
            jm.select_direct_sar_digit([self._candidate(0, "IO", 0.99), self._candidate(1, "", None)])
        )

    def test_letter_conversion_forbidden(self) -> None:
        # 'I' and 'O' must never become 1/0.
        self.assertIsNone(jm.select_direct_sar_digit([self._candidate(0, "IO", 0.99)]))
        self.assertIsNone(jm.select_direct_sar_digit([self._candidate(0, "O", 0.99)]))

    def test_punctuation_and_overlong_rejected(self) -> None:
        self.assertIsNone(jm.select_direct_sar_digit([self._candidate(0, "1-0", 0.9)]))
        self.assertIsNone(jm.select_direct_sar_digit([self._candidate(0, "123", 0.9)]))

    def test_extract_candidates_preserves_api_order(self) -> None:
        candidates = jm.extract_recognition_candidates({"text": ["12", "34"], "scores": [0.1, 0.2]})
        self.assertEqual([c["api_index"] for c in candidates], [0, 1])
        self.assertEqual([c["raw_text"] for c in candidates], ["12", "34"])


class TestDirectSarVariantPrediction(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.crop_path, self.crop_sha = make_crop(Path(self.tmp.name))
        self.item = make_item(self.crop_path, self.crop_sha)

    def test_direct_sar_one_roi_call_and_flags(self) -> None:
        detector = _RecordingDetector()
        recognizer = _RecordingRecognizer([{"text": "10", "scores": 0.9}])
        variant = {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1}
        prediction = jm.predict_item_ablation_variant(
            self.item, variant, detector, recognizer, MODEL_META
        )
        self.assertEqual(len(recognizer.calls), 1)
        self.assertEqual(len(detector.calls), 0)
        self.assertFalse(prediction["detector_used"])
        self.assertIsNone(prediction["detector_region_count"])
        self.assertIsNone(prediction["selected_detector_confidence"])
        self.assertEqual(prediction["recognition_scope"], "full_number_search_roi")
        self.assertEqual(prediction["recognizer_call_count"], 1)
        self.assertEqual(prediction["selected_digit_string"], "10")
        self.assertEqual(prediction["scale_factor"], 1)
        self.assertIsNone(prediction["interpolation"])
        # ROI is 40x60 → processed same at 1x
        self.assertEqual(recognizer.calls[0], (60, 40))
        self.assertEqual(
            (prediction["processed_roi_width"], prediction["processed_roi_height"]), (40, 60)
        )
        jm.json_safe(prediction)

    def test_direct_sar_2x_and_4x_dimensions(self) -> None:
        for scale, variant_id in ((2, "direct_sar_roi_2x_cubic"), (4, "direct_sar_roi_4x_cubic")):
            recognizer = _RecordingRecognizer([{"text": "", "scores": None}])
            variant = {"variant_id": variant_id, "detector_used": False, "scale_factor": scale}
            prediction = jm.predict_item_ablation_variant(
                self.item, variant, _RecordingDetector(), recognizer, MODEL_META
            )
            self.assertEqual(recognizer.calls[0], (60 * scale, 40 * scale))
            self.assertEqual(prediction["processed_roi_width"], 40 * scale)
            self.assertEqual(prediction["processed_roi_height"], 60 * scale)
            self.assertEqual(prediction["interpolation"], "INTER_CUBIC")
            self.assertEqual(prediction["rejection_reason"], "empty_text")

    def test_raw_text_preserved_and_null_confidence(self) -> None:
        recognizer = _RecordingRecognizer([{"text": "SB", "scores": None}])
        variant = {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1}
        prediction = jm.predict_item_ablation_variant(
            self.item, variant, _RecordingDetector(), recognizer, MODEL_META
        )
        self.assertEqual(prediction["raw_recognition_candidates"][0]["raw_text"], "SB")
        self.assertIsNone(prediction["raw_recognition_candidates"][0]["recognizer_score"])
        self.assertIsNone(prediction["selected_digit_string"])
        self.assertEqual(prediction["rejection_reason"], "non_digit_text")

    def test_crop_sha_mismatch_hard_failure(self) -> None:
        item = dict(self.item, source_crop_sha256="0" * 64)
        variant = {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1}
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "crop sha256 mismatch"):
            jm.predict_item_ablation_variant(
                item, variant, _RecordingDetector(), _RecordingRecognizer([{}]), MODEL_META
            )

    def test_invalid_roi_rejected(self) -> None:
        item = dict(self.item, roi_x_min=500, roi_x_max=600)
        variant = {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1}
        prediction = jm.predict_item_ablation_variant(
            item, variant, _RecordingDetector(), _RecordingRecognizer([{}]), MODEL_META
        )
        self.assertEqual(prediction["rejection_reason"], "roi_invalid")
        self.assertIsNone(prediction["selected_digit_string"])


class TestDbnetSar4xVariant(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.crop_path, self.crop_sha = make_crop(Path(self.tmp.name))
        self.item = make_item(self.crop_path, self.crop_sha)
        self.variant = {
            "variant_id": "dbnet_sar_roi_4x_cubic",
            "detector_used": True,
            "scale_factor": 4,
        }

    def test_detector_runs_on_4x_and_coords_mapped_back(self) -> None:
        polygon = [20.0, 20.0, 100.0, 20.0, 100.0, 60.0, 20.0, 60.0]
        detector = _RecordingDetector([polygon], [0.7])
        recognizer = _RecordingRecognizer([{"text": "10", "scores": 0.9}])
        prediction = jm.predict_item_ablation_variant(
            self.item, self.variant, detector, recognizer, MODEL_META
        )
        # ROI 40x60 → 4x = 160x240
        self.assertEqual(detector.calls[0], (240, 160))
        self.assertEqual(prediction["detector_region_count"], 1)
        region = prediction["detected_text_regions"][0]
        self.assertEqual(region["region_bbox_scaled_xyxy"], [20, 20, 100, 60])
        self.assertEqual(region["region_bbox_original_xyxy"], [5.0, 5.0, 25.0, 15.0])
        self.assertEqual(region["polygon_original"][:2], [5.0, 5.0])
        self.assertEqual(prediction["selected_digit_string"], "10")
        self.assertAlmostEqual(prediction["selected_detector_confidence"], 0.7)
        self.assertAlmostEqual(prediction["selected_combined_confidence"], 0.7 * 0.9)
        jm.json_safe(prediction)

    def test_no_region_is_valid_rejection(self) -> None:
        prediction = jm.predict_item_ablation_variant(
            self.item,
            self.variant,
            _RecordingDetector(),
            _RecordingRecognizer([{"text": "10", "scores": 0.9}]),
            MODEL_META,
        )
        self.assertEqual(prediction["detector_region_count"], 0)
        self.assertEqual(prediction["rejection_reason"], "detector_no_region")
        self.assertEqual(prediction["recognizer_call_count"], 0)


class TestAblationRunOrdering(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.crop_path, self.crop_sha = make_crop(Path(self.tmp.name))
        self.items = [
            make_item(self.crop_path, self.crop_sha, pilot_index=5),
            make_item(self.crop_path, self.crop_sha, pilot_index=2),
            make_item(self.crop_path, self.crop_sha, pilot_index=9),
        ]

    def test_variant_major_then_pilot_ascending(self) -> None:
        predictions = jm.run_ablation_inference(
            self.items,
            jm.ABLATION_VARIANTS,
            _RecordingDetector(),
            _RecordingRecognizer([{"text": "", "scores": None}]),
            MODEL_META,
        )
        self.assertEqual(len(predictions), 12)
        self.assertEqual(
            [p["variant_id"] for p in predictions],
            [v for v in jm.ABLATION_VARIANT_IDS for _ in range(3)],
        )
        for offset in range(0, 12, 3):
            self.assertEqual([p["pilot_index"] for p in predictions[offset : offset + 3]], [2, 5, 9])

    def test_same_item_set_for_every_variant(self) -> None:
        predictions = jm.run_ablation_inference(
            self.items,
            jm.ABLATION_VARIANTS,
            _RecordingDetector(),
            _RecordingRecognizer([{"text": "", "scores": None}]),
            MODEL_META,
        )
        id_sets = {
            variant_id: sorted(
                p["review_item_id"] for p in predictions if p["variant_id"] == variant_id
            )
            for variant_id in jm.ABLATION_VARIANT_IDS
        }
        reference = id_sets["direct_sar_roi_1x"]
        for ids in id_sets.values():
            self.assertEqual(ids, reference)

    def test_manual_label_leakage_rejected(self) -> None:
        leaky = [dict(self.items[0], manual_jersey_number="9")]
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "leaked"):
            jm.run_ablation_inference(
                leaky,
                jm.ABLATION_VARIANTS,
                _RecordingDetector(),
                _RecordingRecognizer([{}]),
                MODEL_META,
            )

    def test_wrong_matrix_rejected(self) -> None:
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.run_ablation_inference(
                self.items,
                jm.ABLATION_VARIANTS[:3],
                _RecordingDetector(),
                _RecordingRecognizer([{}]),
                MODEL_META,
            )


class TestAblationEvaluation(unittest.TestCase):
    COUNTS = {
        "POS_readable": 2,
        "A_not_visible": 1,
        "B_visible_unreadable": 0,
        "C_uncertain_signal": 0,
        "D_uncertain_crop": 0,
        "E_invalid": 1,
    }

    def _prediction(self, variant_id: str, pilot_index: int, selection_class: str, digit, reason=None, detector_used=False, region_count=None, error=None) -> dict:
        return {
            "variant_id": variant_id,
            "pilot_index": pilot_index,
            "review_item_id": f"item_{pilot_index:03d}",
            "selection_class": selection_class,
            "selected_digit_string": digit,
            "selected_recognition_confidence": 0.9 if digit else None,
            "selected_detector_confidence": None,
            "rejection_reason": reason,
            "inference_error": error,
            "detector_used": detector_used,
            "detector_region_count": region_count,
            "raw_recognition_candidates": [],
            "total_runtime_ms": 100.0,
        }

    def _reference(self, pilot_index: int, selection_class: str, jersey: str) -> dict:
        return {
            "pilot_index": pilot_index,
            "review_item_id": f"item_{pilot_index:03d}",
            "selection_class": selection_class,
            "manual_jersey_number": jersey,
            "manual_crop_valid": "valid",
            "manual_number_visible": "yes",
            "manual_number_readable": "yes",
        }

    def _build(self):
        references = [
            self._reference(1, "POS_readable", "10"),
            self._reference(2, "POS_readable", "7"),
            self._reference(3, "A_not_visible", ""),
            self._reference(4, "E_invalid", ""),
        ]
        predictions = []
        for variant_id in jm.ABLATION_VARIANT_IDS:
            detector_used = variant_id.startswith("dbnet")
            # Variant-dependent outcomes for metric coverage.
            if variant_id == "direct_sar_roi_4x_cubic":
                predictions.append(self._prediction(variant_id, 1, "POS_readable", "10", detector_used=detector_used))
                predictions.append(self._prediction(variant_id, 2, "POS_readable", "9", detector_used=detector_used))
                predictions.append(self._prediction(variant_id, 3, "A_not_visible", "4", detector_used=detector_used))
            else:
                predictions.append(self._prediction(variant_id, 1, "POS_readable", None, "non_digit_text", detector_used, 0 if detector_used else None))
                predictions.append(self._prediction(variant_id, 2, "POS_readable", None, "empty_text", detector_used, 0 if detector_used else None))
                predictions.append(self._prediction(variant_id, 3, "A_not_visible", None, "non_digit_text", detector_used, 0 if detector_used else None))
            predictions.append(self._prediction(variant_id, 4, "E_invalid", None, "non_digit_text", detector_used, 0 if detector_used else None))
        return predictions, references

    def test_join_and_per_variant_metrics(self) -> None:
        predictions, references = self._build()
        item_rows, summary = jm.evaluate_ablation_predictions(predictions, references, self.COUNTS)
        self.assertEqual(len(item_rows), len(references) * 4)
        v4 = summary["per_variant"]["direct_sar_roi_4x_cubic"]
        self.assertEqual(v4["positive"]["exact_match_count"], 1)
        self.assertEqual(v4["positive"]["wrong_number_count"], 1)
        self.assertEqual(v4["negative"]["number_emission_count"], 1)
        v1 = summary["per_variant"]["direct_sar_roi_1x"]
        self.assertEqual(v1["positive"]["exact_match_count"], 0)
        self.assertEqual(v1["positive"]["no_prediction_count"], 2)
        self.assertEqual(v1["negative"]["number_emission_count"], 0)

    def test_missing_prediction_fails_join(self) -> None:
        predictions, references = self._build()
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "join mismatch"):
            jm.evaluate_ablation_predictions(predictions[:-1], references, self.COUNTS)

    def test_comparison_ranking_and_labels(self) -> None:
        predictions, references = self._build()
        _, summary = jm.evaluate_ablation_predictions(predictions, references, self.COUNTS)
        comparison = jm.build_ablation_comparison_summary(summary, C1_BASELINE)
        self.assertEqual(comparison["descriptive_ranking"][0], "direct_sar_roi_4x_cubic")
        self.assertIn(jm.LABEL_DIRECT_EXACT, comparison["evidence_labels"])
        self.assertIn(jm.LABEL_UPSCALE_RECOGNITION, comparison["evidence_labels"])
        self.assertIn(jm.LABEL_NEGATIVE_RISK, comparison["evidence_labels"])
        self.assertNotIn(jm.LABEL_NO_EXACT, comparison["evidence_labels"])
        self.assertFalse(comparison["ranking_is_deployment_decision"])

    def test_comparison_ranking_deterministic(self) -> None:
        predictions, references = self._build()
        _, summary = jm.evaluate_ablation_predictions(predictions, references, self.COUNTS)
        first = jm.build_ablation_comparison_summary(summary, C1_BASELINE)
        second = jm.build_ablation_comparison_summary(summary, C1_BASELINE)
        self.assertEqual(first["descriptive_ranking"], second["descriptive_ranking"])
        self.assertEqual(first["comparison_table"], second["comparison_table"])

    def test_no_exact_label_when_all_zero(self) -> None:
        references = [self._reference(1, "POS_readable", "10")]
        counts = dict(self.COUNTS, POS_readable=1, A_not_visible=0, E_invalid=0)
        predictions = [
            self._prediction(variant_id, 1, "POS_readable", None, "non_digit_text")
            for variant_id in jm.ABLATION_VARIANT_IDS
        ]
        _, summary = jm.evaluate_ablation_predictions(predictions, references, counts)
        comparison = jm.build_ablation_comparison_summary(summary, C1_BASELINE)
        self.assertIn(jm.LABEL_NO_EXACT, comparison["evidence_labels"])
        self.assertNotIn(jm.LABEL_DIRECT_EXACT, comparison["evidence_labels"])

    def test_upscale_detection_label(self) -> None:
        references = [self._reference(1, "POS_readable", "10")]
        counts = dict(self.COUNTS, POS_readable=1, A_not_visible=0, E_invalid=0)
        predictions = []
        for variant_id in jm.ABLATION_VARIANT_IDS:
            detector_used = variant_id.startswith("dbnet")
            predictions.append(
                self._prediction(
                    variant_id, 1, "POS_readable", None, "recognizer_no_digit",
                    detector_used, 3 if detector_used else None,
                )
            )
        _, summary = jm.evaluate_ablation_predictions(predictions, references, counts)
        baseline_no_regions = dict(C1_BASELINE, detector_region_item_count=0)
        comparison = jm.build_ablation_comparison_summary(summary, baseline_no_regions)
        self.assertIn(jm.LABEL_UPSCALE_DETECTION, comparison["evidence_labels"])
        comparison_same = jm.build_ablation_comparison_summary(summary, C1_BASELINE)
        self.assertNotIn(jm.LABEL_UPSCALE_DETECTION, comparison_same["evidence_labels"])

    def test_inference_error_outcome(self) -> None:
        references = [self._reference(1, "POS_readable", "10")]
        counts = dict(self.COUNTS, POS_readable=1, A_not_visible=0, E_invalid=0)
        predictions = [
            self._prediction(variant_id, 1, "POS_readable", None, None, error="RuntimeError: x")
            for variant_id in jm.ABLATION_VARIANT_IDS
        ]
        item_rows, summary = jm.evaluate_ablation_predictions(predictions, references, counts)
        self.assertTrue(all(row["outcome"] == "inference_error" for row in item_rows))
        self.assertEqual(
            summary["per_variant"]["direct_sar_roi_1x"]["positive"]["inference_error_count"], 1
        )


class TestOutputSafety(unittest.TestCase):
    def test_json_safe_and_nan_rejection(self) -> None:
        jm.json_safe({"a": np.float32(1.5), "b": [np.int64(2)]})
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.json_safe({"a": float("inf")})
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.json_safe({"a": math.nan})

    def test_identity_and_manual_fields_forbidden_in_blind_records(self) -> None:
        for field in ("manual_jersey_number", "identity_label", "team_label", "reviewer"):
            with self.assertRaisesRegex(jm.JerseyMMOCRError, "leaked"):
                jm.assert_blind_records_safe([{"review_item_id": "x", field: "y"}])

    def test_prediction_has_no_manual_or_identity_fields(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        crop_path, crop_sha = make_crop(Path(tmp.name))
        item = make_item(crop_path, crop_sha)
        variant = {"variant_id": "direct_sar_roi_1x", "detector_used": False, "scale_factor": 1}
        prediction = jm.predict_item_ablation_variant(
            item, variant, _RecordingDetector(), _RecordingRecognizer([{"text": "10", "scores": 0.5}]), MODEL_META
        )
        leaked = jm.BLIND_FORBIDDEN_FIELDS.intersection(prediction.keys())
        self.assertEqual(leaked, set())
        for field in ("identity", "team", "global_id", "gallery"):
            self.assertFalse(any(field in key for key in prediction))

    def test_variant_specific_runtime_fields(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        crop_path, crop_sha = make_crop(Path(tmp.name))
        item = make_item(crop_path, crop_sha)
        direct = jm.predict_item_ablation_variant(
            item,
            {"variant_id": "direct_sar_roi_2x_cubic", "detector_used": False, "scale_factor": 2},
            _RecordingDetector(),
            _RecordingRecognizer([{"text": "", "scores": None}]),
            MODEL_META,
        )
        self.assertIsNotNone(direct["resize_runtime_ms"])
        self.assertIsNotNone(direct["recognizer_runtime_ms"])
        self.assertIsNone(direct["detector_runtime_ms"])
        detector_variant = jm.predict_item_ablation_variant(
            item,
            {"variant_id": "dbnet_sar_roi_4x_cubic", "detector_used": True, "scale_factor": 4},
            _RecordingDetector(),
            _RecordingRecognizer([{"text": "", "scores": None}]),
            MODEL_META,
        )
        self.assertIsNotNone(detector_variant["detector_runtime_ms"])
        self.assertIsNotNone(detector_variant["total_runtime_ms"])


class TestBaselineFreezeImmutability(unittest.TestCase):
    FREEZE = _PROJECT_ROOT / "outputs/reid/full_stage4b/jersey_mmocr_smoke_baseline_freeze_stage5c_c1"

    def test_c1_freeze_artifacts_match_manifest(self) -> None:
        if not self.FREEZE.exists():
            self.skipTest("C1 baseline freeze not present in this checkout")
        import json

        manifest = json.loads((self.FREEZE / "baseline_freeze_manifest.json").read_text())
        for artifact in manifest["artifacts"]:
            if artifact.get("self_hash_omitted"):
                continue
            path = self.FREEZE / artifact["filename"]
            self.assertTrue(path.is_file(), artifact["filename"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["filename"])
            self.assertEqual(path.stat().st_size, artifact["byte_size"], artifact["filename"])


if __name__ == "__main__":
    unittest.main()
