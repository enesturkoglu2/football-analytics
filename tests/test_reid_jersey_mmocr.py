"""Unit tests for the Stage 5C-C1 jersey MMOCR smoke adapter (stdlib unittest only).

No MMOCR import and no model init happen here: model-dependent paths are
exercised with mock inferencers, keeping the suite runnable in football-cv.
"""

from __future__ import annotations

import hashlib
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


def make_reviewed_row(
    pilot_index: int,
    crop_valid: str = "valid",
    visible: str = "yes",
    readable: str = "yes",
    jersey: str = "10",
    segment_id: str | None = None,
    contamination: str = "no",
) -> dict:
    return {
        "review_item_id": f"review_item_{pilot_index:03d}",
        "pilot_index": pilot_index,
        "crop_id": f"crop_{pilot_index:03d}",
        "segment_id": segment_id or f"seg_{pilot_index:03d}",
        "raw_track_id": 100 + pilot_index,
        "frame_index": 1000 + pilot_index,
        "source_crop_path": f"/data/crops/crop_{pilot_index:03d}.jpg",
        "source_crop_sha256": f"{pilot_index:064d}",
        "manual_crop_valid": crop_valid,
        "manual_number_visible": visible,
        "manual_number_readable": readable,
        "manual_jersey_number": jersey,
        "manual_digit_count": str(len(jersey)) if jersey else "",
        "manual_contamination_affects_number_region": contamination,
        "manual_back_facing": "yes",
        "manual_notes": "",
        "reviewer": "human",
        "reviewed_at": "2026-07-01T00:00:00+00:00",
    }


def build_population() -> list[dict]:
    """Synthetic population mirroring the frozen pilot distribution."""
    rows: list[dict] = []
    index = 1
    for _ in range(20):  # POS_readable
        rows.append(make_reviewed_row(index, jersey=str((index % 30) + 1)))
        index += 1
    for i in range(36):  # A_not_visible population (only 10 selectable)
        rows.append(
            make_reviewed_row(
                index,
                visible="no",
                readable="no",
                jersey="",
                segment_id=f"seg_shared_{i % 12}",
                contamination="yes" if i % 3 == 0 else "no",
            )
        )
        index += 1
    for _ in range(2):  # B_visible_unreadable (strict readable=no)
        rows.append(make_reviewed_row(index, visible="yes", readable="no", jersey=""))
        index += 1
    for _ in range(4):  # C via readable=uncertain
        rows.append(make_reviewed_row(index, visible="yes", readable="uncertain", jersey=""))
        index += 1
    for _ in range(3):  # C via visible=uncertain
        rows.append(make_reviewed_row(index, visible="uncertain", readable="uncertain", jersey=""))
        index += 1
    for _ in range(5):  # D population (only first 2 selected)
        rows.append(make_reviewed_row(index, crop_valid="uncertain", visible="no", readable="no", jersey=""))
        index += 1
    for _ in range(8):  # E population (only first 5 selected)
        rows.append(make_reviewed_row(index, crop_valid="invalid", visible="no", readable="no", jersey=""))
        index += 1
    return rows


class TestLocalAssetValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.asset = Path(self.tmp.name) / "model.pth"
        self.asset.write_bytes(b"checkpoint-bytes")
        self.sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()

    def test_url_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "URL"):
            jm.validate_local_model_asset("https://example.com/model.pth", self.sha)

    def test_alias_like_relative_name_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "absolute"):
            jm.validate_local_model_asset("dbnet_resnet18_fpnc_1200e_icdar2015", self.sha)

    def test_missing_file_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "not found"):
            jm.validate_local_model_asset(str(Path(self.tmp.name) / "absent.pth"), self.sha)

    def test_sha_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "sha256 mismatch"):
            jm.validate_local_model_asset(str(self.asset), "0" * 64)

    def test_byte_size_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "byte size mismatch"):
            jm.validate_local_model_asset(str(self.asset), self.sha, expected_byte_size=1)

    def test_valid_asset_accepted(self) -> None:
        path = jm.validate_local_model_asset(str(self.asset), self.sha, expected_byte_size=16)
        self.assertEqual(path, self.asset)


class TestConfigValidation(unittest.TestCase):
    def _base_config(self) -> dict:
        entry = {
            "model_id": "m",
            "config_path": "/a/cfg.py",
            "config_sha256": "0" * 64,
            "checkpoint_path": "/a/ckpt.pth",
            "checkpoint_sha256": "0" * 64,
            "checkpoint_byte_size": 1,
        }
        return {
            "schema_version": "v1",
            "device": "cpu",
            "max_items": 46,
            "detector": dict(entry),
            "recognizer": dict(entry),
            "digit_policy": {
                "max_digits": 2,
                "letter_to_digit_conversion": False,
                "confidence_threshold": None,
            },
        }

    def test_valid_config_passes(self) -> None:
        jm.validate_smoke_config(self._base_config())

    def test_non_cpu_device_rejected(self) -> None:
        config = self._base_config()
        config["device"] = "cuda"
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "cpu"):
            jm.validate_smoke_config(config)

    def test_threshold_must_be_null(self) -> None:
        config = self._base_config()
        config["digit_policy"]["confidence_threshold"] = 0.5
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "threshold"):
            jm.validate_smoke_config(config)

    def test_letter_conversion_must_be_false(self) -> None:
        config = self._base_config()
        config["digit_policy"]["letter_to_digit_conversion"] = True
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.validate_smoke_config(config)


class TestSelectionTaxonomy(unittest.TestCase):
    def test_positive_class(self) -> None:
        row = make_reviewed_row(1, jersey="7")
        self.assertEqual(jm.classify_selection_class(row), "POS_readable")

    def test_b_class_strict_readable_no_only(self) -> None:
        row = make_reviewed_row(1, visible="yes", readable="no", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "B_visible_unreadable")

    def test_readable_uncertain_goes_to_c_not_b(self) -> None:
        row = make_reviewed_row(1, visible="yes", readable="uncertain", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "C_uncertain_signal")

    def test_visible_uncertain_goes_to_c(self) -> None:
        row = make_reviewed_row(1, visible="uncertain", readable="uncertain", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "C_uncertain_signal")

    def test_not_visible_goes_to_a(self) -> None:
        row = make_reviewed_row(1, visible="no", readable="no", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "A_not_visible")

    def test_uncertain_crop_goes_to_d(self) -> None:
        row = make_reviewed_row(1, crop_valid="uncertain", visible="no", readable="no", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "D_uncertain_crop")

    def test_invalid_crop_goes_to_e(self) -> None:
        row = make_reviewed_row(1, crop_valid="invalid", visible="no", readable="no", jersey="")
        self.assertEqual(jm.classify_selection_class(row), "E_invalid")

    def test_readable_without_jersey_number_raises(self) -> None:
        row = make_reviewed_row(1, jersey="")
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.classify_selection_class(row)


class TestBuildSelection(unittest.TestCase):
    def test_expected_counts_and_uniqueness(self) -> None:
        selected = jm.build_selection(build_population(), EXPECTED_COUNTS)
        self.assertEqual(len(selected), 46)
        counts: dict[str, int] = {}
        for item in selected:
            counts[item["selection_class"]] = counts.get(item["selection_class"], 0) + 1
        self.assertEqual(counts, EXPECTED_COUNTS)
        ids = [item["review_item_id"] for item in selected]
        self.assertEqual(len(set(ids)), len(ids))

    def test_deterministic(self) -> None:
        population = build_population()
        first = jm.build_selection(population, EXPECTED_COUNTS)
        second = jm.build_selection(list(reversed(population)), EXPECTED_COUNTS)
        self.assertEqual(
            [item["review_item_id"] for item in first],
            [item["review_item_id"] for item in second],
        )

    def test_a_class_segment_diversity_and_contamination_priority(self) -> None:
        selected = jm.build_selection(build_population(), EXPECTED_COUNTS)
        a_items = [item for item in selected if item["selection_class"] == "A_not_visible"]
        segments = [item["segment_id"] for item in a_items]
        self.assertEqual(len(set(segments)), len(segments), "max 1 A item per segment")
        # All contamination=yes candidates occupy distinct segments first.
        contaminated = [i for i in a_items if i["manual_contamination_affects_number_region"] == "yes"]
        self.assertGreater(len(contaminated), 0)

    def test_d_and_e_first_n_by_pilot_index(self) -> None:
        population = build_population()
        selected = jm.build_selection(population, EXPECTED_COUNTS)
        d_pop = sorted(
            (int(r["pilot_index"]) for r in population if r["manual_crop_valid"] == "uncertain")
        )
        e_pop = sorted(
            (int(r["pilot_index"]) for r in population if r["manual_crop_valid"] == "invalid")
        )
        d_sel = sorted(
            int(i["pilot_index"]) for i in selected if i["selection_class"] == "D_uncertain_crop"
        )
        e_sel = sorted(int(i["pilot_index"]) for i in selected if i["selection_class"] == "E_invalid")
        self.assertEqual(d_sel, d_pop[:2])
        self.assertEqual(e_sel, e_pop[:5])

    def test_count_mismatch_raises(self) -> None:
        population = [row for row in build_population() if jm.classify_selection_class(row) != "E_invalid"]
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "E_invalid"):
            jm.build_selection(population, EXPECTED_COUNTS)


class TestBlindManifest(unittest.TestCase):
    def _selected_and_canonical(self) -> tuple[list[dict], dict[str, dict]]:
        selected = jm.build_selection(build_population(), EXPECTED_COUNTS)
        canonical = {
            item["review_item_id"]: {
                "review_item_id": item["review_item_id"],
                "source_crop_path": item["source_crop_path"],
                "source_crop_sha256": item["source_crop_sha256"],
                "crop_width_px": 90,
                "crop_height_px": 120,
                "roi_x_min": 10,
                "roi_y_min": 10,
                "roi_x_max": 80,
                "roi_y_max": 80,
            }
            for item in selected
        }
        return selected, canonical

    def test_blind_manifest_has_no_manual_fields(self) -> None:
        selected, canonical = self._selected_and_canonical()
        records = jm.build_blind_manifest(selected, canonical)
        self.assertEqual(len(records), 46)
        for record in records:
            self.assertFalse(jm.BLIND_FORBIDDEN_FIELDS.intersection(record.keys()))

    def test_leaked_manual_field_detected(self) -> None:
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "leaked"):
            jm.assert_blind_records_safe([{"review_item_id": "x", "manual_jersey_number": "9"}])

    def test_missing_canonical_row_raises(self) -> None:
        selected, canonical = self._selected_and_canonical()
        canonical.pop(selected[0]["review_item_id"])
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "canonical"):
            jm.build_blind_manifest(selected, canonical)

    def test_frozen_canonical_sha_mismatch_raises(self) -> None:
        selected, canonical = self._selected_and_canonical()
        canonical[selected[0]["review_item_id"]]["source_crop_sha256"] = "f" * 64
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "sha mismatch"):
            jm.build_blind_manifest(selected, canonical)

    def test_manifest_pair_join_and_order(self) -> None:
        selected, canonical = self._selected_and_canonical()
        blind = jm.build_blind_manifest(selected, canonical)
        reference = jm.build_evaluation_reference(selected)
        jm.validate_manifest_pair(blind, reference, 46)
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "join"):
            jm.validate_manifest_pair(blind, list(reversed(reference)), 46)
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "row counts"):
            jm.validate_manifest_pair(blind[:-1], reference[:-1], 46)


class TestDigitPolicy(unittest.TestCase):
    def test_nfkc_normalization_of_fullwidth_digits(self) -> None:
        self.assertEqual(jm.normalize_recognized_text("\uff11\uff10"), "10")
        self.assertEqual(jm.normalize_recognized_text("  7 "), "7")
        self.assertEqual(jm.normalize_recognized_text(None), "")

    def test_single_and_double_digits_accepted(self) -> None:
        self.assertEqual(jm.extract_digit_candidate("7"), ("7", None))
        self.assertEqual(jm.extract_digit_candidate("10"), ("10", None))

    def test_three_or_more_digits_rejected(self) -> None:
        self.assertEqual(jm.extract_digit_candidate("123"), (None, "digit_count_exceeds_max"))

    def test_letters_rejected_without_conversion(self) -> None:
        # 'O' and 'l' must NOT be converted to digits.
        self.assertEqual(jm.extract_digit_candidate("O"), (None, "non_digit_text"))
        self.assertEqual(jm.extract_digit_candidate("1O"), (None, "non_digit_text"))
        self.assertEqual(jm.extract_digit_candidate("SB"), (None, "non_digit_text"))
        self.assertEqual(jm.extract_digit_candidate("7a"), (None, "non_digit_text"))

    def test_empty_text_rejected(self) -> None:
        self.assertEqual(jm.extract_digit_candidate(""), (None, "empty_text"))

    def test_punctuation_rejected(self) -> None:
        self.assertEqual(jm.extract_digit_candidate("1-0"), (None, "non_digit_text"))


class TestDigitSelection(unittest.TestCase):
    def _region(self, index: int, candidate: str | None, rec: float | None, det: float = 0.8) -> dict:
        return {
            "region_index": index,
            "accepted_digit_candidate": candidate,
            "recognizer_score": rec,
            "detector_score": det,
        }

    def test_highest_confidence_wins_without_threshold(self) -> None:
        result = jm.select_digit_from_regions(
            [self._region(0, "7", 0.10), self._region(1, "10", 0.95)]
        )
        self.assertEqual(result["digit_string"], "10")
        self.assertAlmostEqual(result["combined_score"], 0.8 * 0.95)

    def test_tie_breaks_to_lowest_region_index(self) -> None:
        result = jm.select_digit_from_regions(
            [self._region(0, "7", 0.5), self._region(1, "9", 0.5)]
        )
        self.assertEqual(result["region_index"], 0)
        self.assertEqual(result["digit_string"], "7")

    def test_no_accepted_candidate_returns_none(self) -> None:
        self.assertIsNone(
            jm.select_digit_from_regions([self._region(0, None, 0.9), self._region(1, None, None)])
        )

    def test_none_score_is_lowest_not_fabricated(self) -> None:
        result = jm.select_digit_from_regions(
            [self._region(0, "8", None), self._region(1, "9", 0.01)]
        )
        self.assertEqual(result["digit_string"], "9")


class TestJsonSafe(unittest.TestCase):
    def test_numpy_scalars_and_arrays_converted(self) -> None:
        payload = {"a": np.float32(0.5), "b": np.int64(3), "c": np.array([1.0, 2.0])}
        result = jm.json_safe(payload)
        self.assertEqual(result, {"a": 0.5, "b": 3, "c": [1.0, 2.0]})

    def test_non_finite_float_rejected(self) -> None:
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.json_safe({"a": float("nan")})


class TestGeometry(unittest.TestCase):
    def test_clamp_roi_within_bounds(self) -> None:
        self.assertEqual(jm.clamp_roi(-5, -5, 200, 300, 100, 120), (0, 0, 100, 120))

    def test_degenerate_roi_returns_none(self) -> None:
        self.assertIsNone(jm.clamp_roi(50, 50, 50, 80, 100, 100))
        self.assertIsNone(jm.clamp_roi(150, 0, 200, 40, 100, 100))

    def test_polygon_to_bbox(self) -> None:
        self.assertEqual(jm.polygon_to_bbox([1.2, 2.7, 10.1, 2.7, 10.1, 9.2, 1.2, 9.2]), (1, 2, 11, 10))


class _MockDetector:
    def __init__(self, polygons: list[list[float]], scores: list[float]) -> None:
        self.polygons = polygons
        self.scores = scores

    def __call__(self, image, return_vis=False, progress_bar=False):
        return {"predictions": [{"polygons": self.polygons, "scores": self.scores}]}


class _MockRecognizer:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def __call__(self, image, return_vis=False, progress_bar=False):
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return {"predictions": [output]}


class TestPredictSingleItem(unittest.TestCase):
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

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        image = np.zeros((120, 90, 3), dtype=np.uint8)
        image[:, :, 2] = 200
        self.crop_path = Path(self.tmp.name) / "crop.png"
        self.assertTrue(cv2.imwrite(str(self.crop_path), image))
        self.crop_sha = hashlib.sha256(self.crop_path.read_bytes()).hexdigest()

    def _item(self, **overrides) -> dict:
        item = {
            "pilot_index": 1,
            "review_item_id": "review_item_001",
            "crop_id": "crop_001",
            "segment_id": "seg_001",
            "raw_track_id": 101,
            "frame_index": 1001,
            "selection_class": "POS_readable",
            "source_crop_path": str(self.crop_path),
            "source_crop_sha256": self.crop_sha,
            "roi_x_min": 10,
            "roi_y_min": 10,
            "roi_x_max": 80,
            "roi_y_max": 100,
            "roi_source": "stage5a_number_search_roi",
        }
        item.update(overrides)
        return item

    def test_crop_sha_mismatch_is_hard_failure(self) -> None:
        detector = _MockDetector([], [])
        recognizer = _MockRecognizer([{"text": "10", "scores": 0.9}])
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "crop sha256 mismatch"):
            jm.predict_single_item(
                self._item(source_crop_sha256="0" * 64), detector, recognizer, self.MODEL_META
            )

    def test_detector_no_region_is_valid_rejection(self) -> None:
        detector = _MockDetector([], [])
        recognizer = _MockRecognizer([{"text": "10", "scores": 0.9}])
        prediction = jm.predict_single_item(self._item(), detector, recognizer, self.MODEL_META)
        self.assertIsNone(prediction["selected_digit_string"])
        self.assertEqual(prediction["rejection_reason"], "detector_no_region")
        self.assertEqual(prediction["detected_region_count"], 0)
        self.assertIsNone(prediction["inference_error"])
        # Must be JSON-serializable.
        jm.json_safe(prediction)

    def test_digit_prediction_flow(self) -> None:
        polygon = [5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]
        detector = _MockDetector([polygon], [0.7])
        recognizer = _MockRecognizer([{"text": "10", "scores": 0.9}])
        prediction = jm.predict_single_item(self._item(), detector, recognizer, self.MODEL_META)
        self.assertEqual(prediction["selected_digit_string"], "10")
        self.assertAlmostEqual(prediction["selected_recognition_confidence"], 0.9)
        self.assertAlmostEqual(prediction["selected_detector_confidence"], 0.7)
        self.assertIsNone(prediction["rejection_reason"])
        self.assertEqual(prediction["raw_recognition_candidates"][0]["raw_text"], "10")
        jm.json_safe(prediction)

    def test_letter_output_rejected_not_converted(self) -> None:
        polygon = [5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]
        detector = _MockDetector([polygon], [0.7])
        recognizer = _MockRecognizer([{"text": "IO", "scores": 0.99}])
        prediction = jm.predict_single_item(self._item(), detector, recognizer, self.MODEL_META)
        self.assertIsNone(prediction["selected_digit_string"])
        self.assertEqual(prediction["rejection_reason"], "recognizer_no_digit")
        self.assertEqual(
            prediction["detected_text_regions"][0]["region_rejection_reason"], "non_digit_text"
        )

    def test_invalid_roi_rejected_without_inference(self) -> None:
        detector = _MockDetector([], [])
        recognizer = _MockRecognizer([{"text": "10", "scores": 0.9}])
        prediction = jm.predict_single_item(
            self._item(roi_x_min=200, roi_x_max=300), detector, recognizer, self.MODEL_META
        )
        self.assertEqual(prediction["rejection_reason"], "roi_invalid")
        self.assertIsNone(prediction["selected_digit_string"])

    def test_blind_inference_rejects_manual_fields_and_orders_by_pilot_index(self) -> None:
        detector = _MockDetector([], [])
        recognizer = _MockRecognizer([{"text": "10", "scores": 0.9}])
        items = [
            self._item(pilot_index=5, review_item_id="review_item_005"),
            self._item(pilot_index=2, review_item_id="review_item_002"),
        ]
        predictions = jm.run_blind_inference(items, detector, recognizer, self.MODEL_META)
        self.assertEqual([p["pilot_index"] for p in predictions], [2, 5])
        with self.assertRaisesRegex(jm.JerseyMMOCRError, "leaked"):
            jm.run_blind_inference(
                [self._item(manual_jersey_number="9")], detector, recognizer, self.MODEL_META
            )


class TestEvaluation(unittest.TestCase):
    COUNTS = {
        "POS_readable": 3,
        "A_not_visible": 1,
        "B_visible_unreadable": 0,
        "C_uncertain_signal": 0,
        "D_uncertain_crop": 0,
        "E_invalid": 1,
    }

    def _prediction(self, pilot_index: int, selection_class: str, digit: str | None, reason: str | None) -> dict:
        return {
            "pilot_index": pilot_index,
            "review_item_id": f"review_item_{pilot_index:03d}",
            "selection_class": selection_class,
            "selected_digit_string": digit,
            "selected_recognition_confidence": 0.9 if digit else None,
            "selected_detector_confidence": 0.8 if digit else None,
            "rejection_reason": reason,
            "inference_error": None,
            "detected_region_count": 1 if digit else 0,
            "total_runtime_ms": 100.0,
        }

    def _reference(self, pilot_index: int, selection_class: str, jersey: str) -> dict:
        return {
            "pilot_index": pilot_index,
            "review_item_id": f"review_item_{pilot_index:03d}",
            "selection_class": selection_class,
            "manual_jersey_number": jersey,
            "manual_crop_valid": "valid",
            "manual_number_visible": "yes",
            "manual_number_readable": "yes",
        }

    def test_metrics(self) -> None:
        predictions = [
            self._prediction(1, "POS_readable", "10", None),  # exact match
            self._prediction(2, "POS_readable", "9", None),  # wrong number
            self._prediction(3, "POS_readable", None, "detector_no_region"),  # no prediction
            self._prediction(4, "A_not_visible", None, "recognizer_no_digit"),  # safe rejection
            self._prediction(5, "E_invalid", "4", None),  # false positive emission
        ]
        references = [
            self._reference(1, "POS_readable", "10"),
            self._reference(2, "POS_readable", "23"),
            self._reference(3, "POS_readable", "7"),
            self._reference(4, "A_not_visible", ""),
            self._reference(5, "E_invalid", ""),
        ]
        item_rows, summary = jm.evaluate_predictions(predictions, references, self.COUNTS)
        outcomes = {row["review_item_id"]: row["outcome"] for row in item_rows}
        self.assertEqual(outcomes["review_item_001"], "exact_match")
        self.assertEqual(outcomes["review_item_002"], "wrong_number")
        self.assertEqual(outcomes["review_item_003"], "no_prediction")
        self.assertEqual(outcomes["review_item_004"], "rejected")
        self.assertEqual(outcomes["review_item_005"], "number_emitted")

        positive = summary["positive"]
        self.assertEqual(positive["exact_match_count"], 1)
        self.assertAlmostEqual(positive["exact_match_rate"], 1 / 3)
        self.assertEqual(positive["wrong_number_count"], 1)
        self.assertEqual(positive["no_prediction_count"], 1)
        negative = summary["negative_total"]
        self.assertEqual(negative["false_positive_number_count"], 1)
        self.assertEqual(negative["invalid_crop_number_emission_count"], 1)
        self.assertEqual(summary["counters"]["detector_no_region_count"], 1)
        self.assertEqual(summary["counters"]["recognizer_no_digit_count"], 1)
        # Honest reporting: interpretation limits must be present.
        self.assertTrue(any("benchmark" in limit for limit in summary["interpretation_limits"]))

    def test_missing_reference_raises(self) -> None:
        predictions = [self._prediction(1, "POS_readable", "10", None)]
        with self.assertRaises(jm.JerseyMMOCRError):
            jm.evaluate_predictions(predictions, [], self.COUNTS)


class TestNetworkAudit(unittest.TestCase):
    def test_loopback_only_passes(self) -> None:
        trace = "\n".join(
            [
                '1 socket(AF_INET6, SOCK_STREAM|SOCK_CLOEXEC, IPPROTO_IP) = 5<TCPv6:[1]>',
                '1 bind(5<TCPv6:[1]>, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::1", &sin6_addr), sin6_scope_id=0}, 28) = 0',
                '1 socket(AF_UNIX, SOCK_STREAM, 0) = 6',
            ]
        )
        audit = jm.parse_network_strace(trace)
        self.assertEqual(audit["policy_status"], "pass_loopback_only")
        self.assertTrue(audit["loopback_socket_created"])
        self.assertEqual(audit["loopback_bind_count"], 1)
        self.assertEqual(audit["external_connect_attempt_count"], 0)
        self.assertEqual(audit["wildcard_bind_count"], 0)

    def test_external_connect_fails(self) -> None:
        trace = '1 connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}, 16) = 0'
        audit = jm.parse_network_strace(trace)
        self.assertEqual(audit["external_connect_attempt_count"], 1)
        self.assertEqual(audit["policy_status"], "fail_external_network")

    def test_wildcard_bind_fails(self) -> None:
        trace = '1 bind(3, {sa_family=AF_INET, sin_port=htons(8080), sin_addr=inet_addr("0.0.0.0")}, 16) = 0'
        audit = jm.parse_network_strace(trace)
        self.assertEqual(audit["wildcard_bind_count"], 1)
        self.assertEqual(audit["policy_status"], "fail_external_network")

    def test_dns_port_fails(self) -> None:
        trace = '1 connect(3, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("8.8.8.8")}, 16) = 0'
        audit = jm.parse_network_strace(trace)
        self.assertGreaterEqual(audit["DNS_attempt_count"], 1)
        self.assertEqual(audit["policy_status"], "fail_external_network")

    def test_loopback_connect_allowed(self) -> None:
        trace = '1 connect(3, {sa_family=AF_INET, sin_port=htons(8000), sin_addr=inet_addr("127.0.0.1")}, 16) = 0'
        audit = jm.parse_network_strace(trace)
        self.assertEqual(audit["loopback_connect_count"], 1)
        self.assertEqual(audit["policy_status"], "pass_loopback_only")


if __name__ == "__main__":
    unittest.main()
