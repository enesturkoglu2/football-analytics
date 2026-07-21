"""Synthetic tests for Stage 5C-A jersey visibility measurements."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.jersey_visibility import (
    CROP_SIGNALS_NAME,
    OUTPUT_NAMES,
    SEGMENT_SUMMARY_NAME,
    SUMMARY_NAME,
    JerseyVisibilityError,
    assign_independent_ranks,
    build_crop_provenance_plan,
    build_segment_summaries,
    compute_image_measurements,
    compute_other_person_contamination,
    compute_upper_torso_number_search_roi,
    load_jersey_visibility_config,
    load_segment_view_inputs,
    run_analyze_jersey_visibility,
    validate_jersey_visibility_config,
)
from football_analytics.reid.segments import canonical_observation_json

CONFIG = _PROJECT_ROOT / "configs/reid/jersey_visibility_stage5c.yaml"
SCRIPT = _PROJECT_ROOT / "scripts/analyze_jersey_visibility.py"
SOURCE = _SRC_DIR / "football_analytics/reid/jersey_visibility.py"


def _sha_observation(observation: dict) -> str:
    return hashlib.sha256(
        canonical_observation_json(observation).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False))
            handle.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


def _segment(segment_id: str, raw_id: int, kind: str, index=None) -> dict:
    return {
        "schema_version": "reid_track_segment_v1",
        "segment_id": segment_id,
        "raw_track_id": raw_id,
        "segment_kind": kind,
        "segment_index": index,
    }


def _signal(crop_id: str, segment_id: str, value: float) -> dict:
    return {
        "segment_id": segment_id,
        "frame_index": 3,
        "crop_id": crop_id,
        "roi_height_px": int(value),
        "roi_area_px": int(value * 10),
        "local_contrast": value,
        "edge_density": value / 100,
        "entropy": value / 10,
        "roi_other_person_union_coverage": 0.1,
        "laplacian_variance": value,
        "tenengrad_mean": value * 2,
        "sharpness_size_stratum": "0_23",
    }


class SyntheticFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.segment_view = root / "segment_view"
        self.regression = root / "regression"
        self.baseline = root / "baseline"
        self.output = root / "output"
        self.config = root / "config.yaml"
        self.segment_view.mkdir()
        (self.regression / "crops").mkdir(parents=True)
        (self.baseline / "crops/track_2").mkdir(parents=True)
        shutil.copyfile(CONFIG, self.config)
        self._build()

    def _build(self) -> None:
        segments = [
            _segment("raw_1_s01", 1, "manual_split_segment", 1),
            _segment("raw_2_full", 2, "no_split_control"),
            _segment("raw_3_full", 3, "preserved_full_track"),
        ]
        observations = [
            {
                "bbox_xyxy": [0.0, 0.0, 20.0, 40.0],
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "frame_index": 3,
                "timestamp_sec": 0.1,
                "track_id": 1,
            },
            {
                "bbox_xyxy": [30.0, 0.0, 50.0, 40.0],
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.8,
                "frame_index": 3,
                "timestamp_sec": 0.1,
                "track_id": 2,
            },
            {
                "bbox_xyxy": [60.0, 0.0, 80.0, 40.0],
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.7,
                "frame_index": 3,
                "timestamp_sec": 0.1,
                "track_id": 3,
            },
        ]
        assigned = []
        for source_row, (segment, observation) in enumerate(
            zip(segments, observations)
        ):
            assigned.append(
                {
                    "schema_version": "reid_segment_observation_v1",
                    "segment_id": segment["segment_id"],
                    "raw_track_id": segment["raw_track_id"],
                    "segment_kind": segment["segment_kind"],
                    "segment_index": segment["segment_index"],
                    "frame_index": 3,
                    "source_row_index": source_row,
                    "source_observation_sha256": _sha_observation(observation),
                    "source_observation": observation,
                }
            )
        _write_jsonl(self.segment_view / "track_segments.jsonl", segments)
        _write_jsonl(self.segment_view / "segment_observations.jsonl", assigned)
        _write_jsonl(self.segment_view / "unassigned_observations.jsonl", [])
        _write_json(
            self.segment_view / "segment_view_summary.json", {"status": "ok"}
        )

        manual_image = np.zeros((40, 20, 3), dtype=np.uint8)
        manual_image[:, 10:] = 255
        reused_image = np.full((40, 20, 3), 127, dtype=np.uint8)
        cv2.imwrite(str(self.regression / "crops/manual.jpg"), manual_image)
        cv2.imwrite(
            str(self.baseline / "crops/track_2/reused.jpg"), reused_image
        )
        manual_crop = {
            "schema_version": "reid_segment_crop_manifest_v1",
            "segment_id": "raw_1_s01",
            "raw_track_id": 1,
            "segment_kind": "manual_split_segment",
            "segment_index": 1,
            "representation_source": "recomputed_manual_segment",
            "crop_id": "manual_crop",
            "selection_rank": 1,
            "frame_index": 3,
            "bbox_xyxy": [0, 0, 20, 40],
            "crop_relative_path": "crops/manual.jpg",
            "source_observation_row_index": 0,
            "source_observation_sha256": assigned[0][
                "source_observation_sha256"
            ],
            "ambiguous_observation_used": False,
        }
        _write_jsonl(
            self.regression / "segment_crop_manifest.jsonl", [manual_crop]
        )
        entities = [
            {
                **segments[0],
                "schema_version": "reid_segment_embedding_record_v1",
                "representation_source": "recomputed_manual_segment",
                "representation_status": "recompute_manual_segment",
                "crop_ids": ["manual_crop"],
                "parent_mixed_embedding_retired": True,
            },
            {
                **segments[1],
                "schema_version": "reid_segment_embedding_record_v1",
                "representation_source": "reused_baseline_raw_track_embedding",
                "representation_status": "reuse_baseline_full_track",
                "crop_ids": ["reused_crop"],
                "parent_mixed_embedding_retired": False,
            },
            {
                **segments[2],
                "schema_version": "reid_segment_embedding_record_v1",
                "representation_source": "no_baseline_embedding",
                "representation_status": "no_baseline_embedding",
                "crop_ids": [],
                "parent_mixed_embedding_retired": False,
            },
        ]
        _write_jsonl(self.regression / "segment_embedding_index.jsonl", entities)
        _write_jsonl(
            self.regression / "baseline_to_segment_replacement.jsonl",
            [{"raw_track_id": raw_id} for raw_id in (1, 2, 3)],
        )
        _write_json(
            self.regression / "segmented_reid_regression_summary.json",
            {"status": "ok"},
        )
        _write_jsonl(
            self.baseline / "crops/crop_manifest.jsonl",
            [
                {
                    "schema_version": "reid_crop_manifest_v1",
                    "crop_id": "reused_crop",
                    "track_id": 2,
                    "frame_index": 3,
                    "selection_rank": 1,
                    "bbox_xyxy": [30, 0, 50, 40],
                    "crop_relative_path": "crops/track_2/reused.jpg",
                }
            ],
        )

    def run(self, *, overwrite: bool = False) -> dict:
        return run_analyze_jersey_visibility(
            segment_view_dir=self.segment_view,
            segmented_regression_dir=self.regression,
            baseline_run_dir=self.baseline,
            config=self.config,
            output_dir=self.output,
            overwrite=overwrite,
        )


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_valid_config(self) -> None:
        validated = validate_jersey_visibility_config(self.payload)
        self.assertEqual(validated["schema_version"], "reid_jersey_visibility_config_v1")
        self.assertIsNone(
            validated["ranking"]["automatic_readability_threshold"]
        )
        self.assertEqual(
            load_jersey_visibility_config(CONFIG)["stage_status"],
            "implementation_measurement_baseline",
        )

    def test_rejects_unsafe_input_flags(self) -> None:
        cases = (
            ("allow_video_input", True),
            ("allow_new_crop_extraction", True),
            ("include_retired_mixed_parent_entities", True),
            ("ambiguous_observations_allowed", True),
            ("unassigned_observations_allowed", True),
        )
        for key, value in cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["input"][key] = value
                with self.assertRaises(JerseyVisibilityError):
                    validate_jersey_visibility_config(payload)

    def test_rejects_ranking_decisions(self) -> None:
        for key, value in (
            ("composite_score_enabled", True),
            ("automatic_visibility_threshold", 0.5),
            ("automatic_readability_threshold", 0.5),
        ):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["ranking"][key] = value
                with self.assertRaises(JerseyVisibilityError):
                    validate_jersey_visibility_config(payload)

    def test_rejects_recognition_flags(self) -> None:
        for key in (
            "ocr_enabled",
            "recognizer_enabled",
            "checkpoint_required",
            "jersey_number_candidate_enabled",
            "automatic_jersey_assignment_enabled",
        ):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["recognition"][key] = True
                with self.assertRaises(JerseyVisibilityError):
                    validate_jersey_visibility_config(payload)

    def test_rejects_unsafe_safety_flags(self) -> None:
        changes = {
            "source_crops_immutable": False,
            "segment_view_immutable": False,
            "regression_artifacts_immutable": False,
            "identity_ground_truth_available": True,
            "accuracy_claim_allowed": True,
            "team_assignment_enabled": True,
            "global_id_rewrite_enabled": True,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["safety"][key] = value
                with self.assertRaises(JerseyVisibilityError):
                    validate_jersey_visibility_config(payload)

    def test_rejects_invalid_roi_and_strata(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["number_search_roi"]["x_min_normalized"] = 0.9
        with self.assertRaises(JerseyVisibilityError):
            validate_jersey_visibility_config(payload)
        payload = copy.deepcopy(self.payload)
        payload["ranking"]["roi_height_strata_pixels"] = [0, 40, 24]
        with self.assertRaises(JerseyVisibilityError):
            validate_jersey_visibility_config(payload)


class RoiAndMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roi = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
            "number_search_roi"
        ]

    def test_roi_floor_ceil_clamp_and_tiny_crop(self) -> None:
        self.assertEqual(
            compute_upper_torso_number_search_roi(10, 10, self.roi),
            (1, 1, 9, 7),
        )
        self.assertEqual(
            compute_upper_torso_number_search_roi(1, 1, self.roi),
            (0, 0, 1, 1),
        )

    def test_metrics_are_finite_and_flat_behavior_is_zero(self) -> None:
        flat = np.full((20, 10, 3), 80, dtype=np.uint8)
        metrics = compute_image_measurements(flat)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["grayscale_std"], 0.0)
        self.assertEqual(metrics["laplacian_variance"], 0.0)
        self.assertEqual(metrics["tenengrad_mean"], 0.0)
        self.assertEqual(metrics["edge_density"], 0.0)
        self.assertEqual(metrics["entropy"], -0.0)
        self.assertEqual(metrics["local_contrast"], 0.0)

    def test_sharp_image_exceeds_blurred_image(self) -> None:
        sharp = np.zeros((40, 40, 3), dtype=np.uint8)
        sharp[8:32, 8:32] = 255
        blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
        sharp_metrics = compute_image_measurements(sharp)
        blur_metrics = compute_image_measurements(blurred)
        self.assertGreater(
            sharp_metrics["laplacian_variance"],
            blur_metrics["laplacian_variance"],
        )
        self.assertGreater(
            sharp_metrics["tenengrad_mean"], blur_metrics["tenengrad_mean"]
        )

    def test_roi_analysis_does_not_resize_or_write(self) -> None:
        image = np.zeros((7, 11, 3), dtype=np.uint8)
        before = image.shape
        roi = compute_upper_torso_number_search_roi(11, 7, self.roi)
        compute_image_measurements(image[roi[1] : roi[3], roi[0] : roi[2]])
        self.assertEqual(image.shape, before)


class ContaminationAndRankTests(unittest.TestCase):
    @staticmethod
    def _observation(raw_id: int, bbox: list[int], digest: str) -> dict:
        return {
            "raw_track_id": raw_id,
            "source_observation_sha256": digest,
            "source_observation": {"bbox_xyxy": bbox},
        }

    def test_no_other_person_is_zero_and_own_is_excluded(self) -> None:
        rows = [self._observation(1, [0, 0, 10, 10], "own")]
        result = compute_other_person_contamination(
            target_frame_bbox=[0, 0, 10, 10],
            roi_local_bbox=[2, 2, 8, 8],
            frame_observations=rows,
            own_source_observation_sha256="own",
            own_raw_track_id=1,
        )
        self.assertEqual(result["full_crop_other_person_union_coverage"], 0.0)
        self.assertEqual(result["roi_other_person_union_coverage"], 0.0)
        self.assertEqual(result["roi_other_person_center_inside_count"], 0)

    def test_partial_and_full_roi_overlap_and_center(self) -> None:
        rows = [self._observation(2, [2, 2, 8, 8], "other")]
        result = compute_other_person_contamination(
            target_frame_bbox=[0, 0, 10, 10],
            roi_local_bbox=[2, 2, 8, 8],
            frame_observations=rows,
            own_source_observation_sha256="own",
            own_raw_track_id=1,
        )
        self.assertAlmostEqual(
            result["full_crop_other_person_union_coverage"], 0.36
        )
        self.assertEqual(result["roi_other_person_union_coverage"], 1.0)
        self.assertEqual(result["roi_other_person_center_inside_count"], 1)

    def test_union_does_not_double_count_overlap(self) -> None:
        rows = [
            self._observation(2, [0, 0, 6, 10], "a"),
            self._observation(3, [4, 0, 10, 10], "b"),
        ]
        result = compute_other_person_contamination(
            target_frame_bbox=[0, 0, 10, 10],
            roi_local_bbox=[0, 0, 10, 10],
            frame_observations=rows,
            own_source_observation_sha256="own",
            own_raw_track_id=1,
        )
        self.assertEqual(result["full_crop_other_person_union_coverage"], 1.0)
        self.assertEqual(result["roi_other_person_union_coverage"], 1.0)

    def test_independent_ranks_and_tie_break_are_deterministic(self) -> None:
        rows = [_signal("b", "seg_b", 10), _signal("a", "seg_a", 10)]
        assign_independent_ranks(rows)
        by_id = {row["crop_id"]: row for row in rows}
        self.assertEqual(by_id["a"]["roi_height_global_rank"], 1)
        self.assertEqual(by_id["a"]["laplacian_rank_within_size_stratum"], 1)
        self.assertNotIn("visibility_score", by_id["a"])
        self.assertNotIn("readability_score", by_id["a"])


class ProvenanceAndOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = SyntheticFixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _plan(self) -> dict:
        view = load_segment_view_inputs(self.fixture.segment_view)
        return build_crop_provenance_plan(
            segment_view=view,
            segmented_regression_dir=self.fixture.regression,
            baseline_run_dir=self.fixture.baseline,
        )

    def test_plan_has_manual_reused_and_no_crop_segment(self) -> None:
        plan = self._plan()
        self.assertEqual(len(plan["crops"]), 2)
        sources = {row["crop_source_kind"] for row in plan["crops"]}
        self.assertEqual(
            sources,
            {
                "recomputed_manual_segment",
                "reused_baseline_selected_crop",
            },
        )
        self.assertNotIn("raw_3_full", plan["crops_by_segment"])

    def test_duplicate_crop_assignment_is_rejected(self) -> None:
        path = self.fixture.regression / "segment_embedding_index.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows[1]["crop_ids"] = ["manual_crop"]
        _write_jsonl(path, rows)
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_unknown_segment_reference_is_rejected(self) -> None:
        path = self.fixture.regression / "segment_crop_manifest.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["segment_id"] = "unknown"
        _write_jsonl(path, [row])
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_baseline_parent_mismatch_is_rejected(self) -> None:
        path = self.fixture.baseline / "crops/crop_manifest.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["track_id"] = 99
        _write_jsonl(path, [row])
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_missing_crop_file_is_rejected(self) -> None:
        (self.fixture.regression / "crops/manual.jpg").unlink()
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_source_observation_sha_mismatch_is_rejected(self) -> None:
        path = self.fixture.regression / "segment_crop_manifest.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["source_observation_sha256"] = "0" * 64
        _write_jsonl(path, [row])
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_retired_parent_cannot_be_reused(self) -> None:
        path = self.fixture.regression / "segment_embedding_index.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows[1]["parent_mixed_embedding_retired"] = True
        _write_jsonl(path, rows)
        with self.assertRaises(JerseyVisibilityError):
            self._plan()

    def test_run_writes_only_three_strict_outputs_and_all_segments(self) -> None:
        result = self.fixture.run()
        self.assertEqual(result["total_derived_segment_count"], 3)
        self.assertEqual(result["measured_segment_count"], 2)
        self.assertEqual(result["no_selected_crop_segment_count"], 1)
        self.assertEqual(
            sorted(path.name for path in self.fixture.output.iterdir()),
            sorted(OUTPUT_NAMES),
        )
        for name in OUTPUT_NAMES:
            self.assertTrue((self.fixture.output / name).read_bytes().endswith(b"\n"))
        segments = [
            json.loads(line)
            for line in (self.fixture.output / SEGMENT_SUMMARY_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            [row["segment_id"] for row in segments],
            ["raw_1_s01", "raw_2_full", "raw_3_full"],
        )
        no_crop = segments[-1]
        self.assertEqual(
            no_crop["measurement_status"], "no_selected_crop_provenance"
        )
        self.assertEqual(no_crop["no_crop_reason"], "no_selected_crop_provenance")
        self.assertIsNone(no_crop["laplacian_variance"]["median"])
        self.assertFalse(no_crop["ocr_attempted"])
        self.assertIsNone(no_crop["manual_any_number_readable"])

    def test_segment_summary_has_distributions_and_best_references(self) -> None:
        rows = [_signal("crop_a", "seg", 4), _signal("crop_b", "seg", 8)]
        summaries = build_segment_summaries(
            segments=[_segment("seg", 1, "manual_split_segment", 1)],
            crop_signals=rows,
            entity_by_id={
                "seg": {"representation_status": "recompute_manual_segment"}
            },
        )
        self.assertEqual(summaries[0]["laplacian_variance"]["median"], 6.0)
        self.assertEqual(summaries[0]["largest_roi_crop_id"], "crop_b")
        self.assertEqual(summaries[0]["highest_entropy_crop_id"], "crop_b")

    def test_collision_overwrite_and_source_immutability(self) -> None:
        sources = [
            self.fixture.regression / "crops/manual.jpg",
            self.fixture.baseline / "crops/track_2/reused.jpg",
            self.fixture.segment_view / "track_segments.jsonl",
        ]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        self.fixture.run()
        with self.assertRaises(JerseyVisibilityError):
            self.fixture.run()
        self.fixture.run(overwrite=True)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        self.assertEqual(before, after)
        self.assertFalse(
            list(self.root.glob("_tmp_reid_jersey_visibility_output_*"))
        )
        self.assertFalse(
            list(self.root.glob("_backup_reid_jersey_visibility_output_*"))
        )

    def test_validation_failure_creates_no_final_output(self) -> None:
        crop = self.fixture.regression / "crops/manual.jpg"
        crop.write_bytes(b"not an image")
        with self.assertRaises(JerseyVisibilityError):
            self.fixture.run()
        self.assertFalse(self.fixture.output.exists())
        self.assertFalse(
            list(self.root.glob("_tmp_reid_jersey_visibility_output_*"))
        )

    def test_crop_and_segment_outputs_are_deterministic(self) -> None:
        self.fixture.run()
        first = {
            name: (self.fixture.output / name).read_bytes()
            for name in (CROP_SIGNALS_NAME, SEGMENT_SUMMARY_NAME)
        }
        self.fixture.run(overwrite=True)
        second = {
            name: (self.fixture.output / name).read_bytes()
            for name in (CROP_SIGNALS_NAME, SEGMENT_SUMMARY_NAME)
        }
        self.assertEqual(first, second)


class CliAndSafetyTests(unittest.TestCase):
    def test_help_has_only_approved_inputs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--segment-view-dir",
            "--segmented-regression-dir",
            "--baseline-run-dir",
            "--config",
            "--output-dir",
            "--overwrite",
        ):
            self.assertIn(option, result.stdout)
        for option in (
            "--video",
            "--checkpoint",
            "--model",
            "--ocr-backend",
            "--team-count",
            "--jersey-number",
            "--global-id-map",
            "--similarity-threshold",
            "--manual-label",
        ):
            self.assertNotIn(option, result.stdout)

    def test_product_source_has_no_forbidden_runtime_or_writer(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("VideoCapture", text)
        self.assertNotIn("import torch", text)
        self.assertNotIn("MMOCR", text)
        self.assertNotIn("PaddleOCR", text)
        self.assertNotIn("Tesseract", text)
        self.assertNotIn("EasyOCR", text)
        self.assertNotIn("urllib", text)
        self.assertNotIn("requests.", text)
        self.assertNotIn("cv2.imwrite", text)


if __name__ == "__main__":
    unittest.main()
