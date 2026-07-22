"""Synthetic tests for Stage 5C-A2b jersey review panel generation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
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

from football_analytics.reid.jersey_review import (
    ANNOTATION_COLUMNS,
    ANNOTATION_TEMPLATE_NAME,
    GROUP_MEMBERSHIPS_NAME,
    MANUAL_FIELDS,
    OUTPUT_NAMES,
    PANEL_INDEX_NAME,
    PANELS_DIRNAME,
    REVIEW_ITEMS_NAME,
    SUMMARY_NAME,
    TILE_HEIGHT,
    TILE_WIDTH,
    JerseyReviewError,
    load_measurement_inputs,
    run_build_jersey_review_panels,
    validate_jersey_review_config,
)

REPO_CONFIG = _PROJECT_ROOT / "configs/reid/jersey_review_panels_stage5c.yaml"
SCRIPT = _PROJECT_ROOT / "scripts/build_jersey_review_panels.py"
SOURCE = _SRC_DIR / "football_analytics/reid/jersey_review.py"

CROP_SIGNAL_SCHEMA = "reid_jersey_visibility_crop_signal_v1"
SEGMENT_SUMMARY_SCHEMA = "reid_jersey_visibility_segment_summary_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SyntheticReviewFixture:
    """Small synthetic measurement artifacts + crops for panel generation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "project"
        self.measurement = self.project / "measurement"
        self.measurement_config_path = self.project / "measurement_config.yaml"
        self.output = root / "review_output"
        self.config_path = root / "review_config.yaml"
        self.measurement.mkdir(parents=True)
        self._build_crops()
        self._build_measurement()

    @staticmethod
    def _image(width: int, height: int, seed: int) -> np.ndarray:
        ys = np.arange(height, dtype=np.int32).reshape(-1, 1)
        xs = np.arange(width, dtype=np.int32).reshape(1, -1)
        base = ((ys * 3 + xs * 5 + seed * 17) % 256).astype(np.uint8)
        return np.stack([base, base // 2 + 40, base // 3 + 80], axis=-1)

    def _build_crops(self) -> None:
        # (crop_id, segment, kind, track, frame, rank, width, height, seed)
        self.specs = [
            ("track_1_frame_10_rank_1", "raw_1_s01", "manual_split_segment", 1, 10, 1, 40, 52, 1),
            ("track_1_frame_20_rank_2", "raw_1_s01", "manual_split_segment", 1, 20, 2, 40, 52, 2),
            ("track_2_frame_5_rank_1", "raw_2_full", "preserved_full_track", 2, 5, 1, 36, 52, 3),
            ("track_2_frame_9_rank_2", "raw_2_full", "preserved_full_track", 2, 9, 2, 30, 32, 4),
            ("track_3_frame_7_rank_1", "raw_3_full", "preserved_full_track", 3, 7, 1, 30, 32, 5),
        ]
        self.crop_paths: dict[str, Path] = {}
        for crop_id, _seg, _kind, track, _frame, rank, width, height, seed in self.specs:
            path = self.project / f"crops/track_{track}/{crop_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), self._image(width, height, seed))
            self.crop_paths[crop_id] = path

    def _build_measurement(self) -> None:
        # Per-crop metric values consistent with the rank permutations below.
        metric_values = {
            "track_1_frame_10_rank_1": dict(lc=5.0, ent=5.5, ed=0.25, cont=0.0, lap=900.0, ten=30000.0),
            "track_1_frame_20_rank_2": dict(lc=4.0, ent=6.0, ed=0.22, cont=0.0, lap=800.0, ten=28000.0),
            "track_2_frame_5_rank_1": dict(lc=3.0, ent=6.5, ed=0.19, cont=0.2, lap=700.0, ten=26000.0),
            "track_2_frame_9_rank_2": dict(lc=2.0, ent=7.0, ed=0.16, cont=0.5, lap=950.0, ten=32000.0),
            "track_3_frame_7_rank_1": dict(lc=1.0, ent=7.5, ed=0.13, cont=1.0, lap=480.0, ten=15000.0),
        }
        global_ranks = {
            "roi_height_global_rank": [1, 2, 3, 4, 5],
            "roi_area_global_rank": [1, 2, 3, 4, 5],
            "local_contrast_global_rank": [1, 2, 3, 4, 5],
            "edge_density_global_rank": [1, 2, 3, 4, 5],
            "entropy_global_rank": [5, 4, 3, 2, 1],
            "contamination_low_global_rank": [1, 2, 3, 4, 5],
        }
        stratum_ranks = {
            "track_1_frame_10_rank_1": ("40_63", 1, 1),
            "track_1_frame_20_rank_2": ("40_63", 2, 2),
            "track_2_frame_5_rank_1": ("40_63", 3, 3),
            "track_2_frame_9_rank_2": ("24_39", 1, 1),
            "track_3_frame_7_rank_1": ("24_39", 2, 2),
        }
        rows = []
        for index, spec in enumerate(self.specs):
            crop_id, seg, kind, track, frame, rank, width, height, _seed = spec
            manual = kind == "manual_split_segment"
            values = metric_values[crop_id]
            stratum, lap_rank, ten_rank = stratum_ranks[crop_id]
            path = self.crop_paths[crop_id]
            rows.append(
                {
                    "schema_version": CROP_SIGNAL_SCHEMA,
                    "crop_id": crop_id,
                    "segment_id": seg,
                    "raw_track_id": track,
                    "segment_kind": kind,
                    "representation_source": (
                        "recomputed_manual_segment"
                        if manual
                        else "reused_baseline_raw_track_embedding"
                    ),
                    "crop_source_kind": (
                        "recomputed_manual_segment"
                        if manual
                        else "reused_baseline_selected_crop"
                    ),
                    "frame_index": frame,
                    "selection_rank": rank,
                    "source_crop_path": str(path),
                    "source_crop_sha256": _sha256(path),
                    "crop_width_px": width,
                    "crop_height_px": height,
                    "roi_x_min": 1,
                    "roi_y_min": 1,
                    "roi_x_max": width - 1,
                    "roi_y_max": height - 1,
                    "roi_width_px": width - 2,
                    "roi_height_px": height - 2,
                    "grayscale_mean": 120.0 + index,
                    "grayscale_std": 30.0 + index,
                    "laplacian_variance": values["lap"],
                    "tenengrad_mean": values["ten"],
                    "local_contrast": values["lc"],
                    "entropy": values["ent"],
                    "edge_density": values["ed"],
                    "roi_other_person_union_coverage": values["cont"],
                    "full_crop_other_person_union_coverage": values["cont"] / 2,
                    "roi_other_person_center_inside_count": 0,
                    "sharpness_size_stratum": stratum,
                    "laplacian_rank_within_size_stratum": lap_rank,
                    "tenengrad_rank_within_size_stratum": ten_rank,
                    **{
                        field: ranks[index]
                        for field, ranks in global_ranks.items()
                    },
                }
            )
        self.crop_rows = rows
        _write_jsonl(self.measurement / "jersey_visibility_crop_signals.jsonl", rows)

        segments = [
            ("raw_1_s01", 1, "manual_split_segment", "measured_selected_crops", 2),
            ("raw_2_full", 2, "preserved_full_track", "measured_selected_crops", 2),
            ("raw_3_full", 3, "preserved_full_track", "measured_selected_crops", 1),
            ("raw_4_full", 4, "preserved_full_track", "no_selected_crop_provenance", 0),
        ]
        _write_jsonl(
            self.measurement / "jersey_visibility_segment_summary.jsonl",
            [
                {
                    "schema_version": SEGMENT_SUMMARY_SCHEMA,
                    "segment_id": sid,
                    "raw_track_id": track,
                    "segment_kind": kind,
                    "measurement_status": status,
                    "selected_crop_count": count,
                    "measured_crop_count": count,
                }
                for sid, track, kind, status, count in segments
            ],
        )
        self.measurement_config_path.write_text(
            yaml.safe_dump(
                {"ranking": {"roi_height_strata_pixels": [24, 40, 64]}}
            ),
            encoding="utf-8",
        )
        (self.measurement / "jersey_visibility_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "reid_jersey_visibility_summary_v1",
                    "status": "ok",
                    "total_selected_crop_count": 5,
                    "total_derived_segment_count": 4,
                    "measured_segment_count": 3,
                    "no_selected_crop_segment_count": 1,
                    "source_artifacts": {
                        "config": {
                            "path": str(self.measurement_config_path),
                            "sha256": _sha256(self.measurement_config_path),
                        }
                    },
                },
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def set_declared_strata(self, boundaries: list[int]) -> None:
        self.measurement_config_path.write_text(
            yaml.safe_dump(
                {"ranking": {"roi_height_strata_pixels": boundaries}}
            ),
            encoding="utf-8",
        )
        summary_path = self.measurement / "jersey_visibility_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source_artifacts"]["config"]["sha256"] = _sha256(
            self.measurement_config_path
        )
        summary_path.write_text(
            json.dumps(summary, allow_nan=False) + "\n", encoding="utf-8"
        )

    def config_payload(self) -> dict:
        payload = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
        for name in (
            "roi_height_top",
            "roi_height_bottom",
            "local_contrast_top",
            "local_contrast_bottom",
            "entropy_top",
            "entropy_bottom",
            "roi_contamination_low",
            "roi_contamination_high",
        ):
            payload["groups"][name]["count"] = 2
        payload["groups"]["sharpness_by_roi_height_stratum"].update(
            {
                "laplacian_top_count": 3,
                "laplacian_bottom_count": 2,
                "tenengrad_top_count": 2,
                "tenengrad_bottom_count": 2,
            }
        )
        payload["groups"]["critical_segments"]["segment_ids"] = [
            "raw_1_s01",
            "raw_4_full",
        ]
        payload["panel"]["columns"] = 2
        payload["panel"]["rows"] = 2
        return payload

    def write_config(self, payload: dict | None = None) -> Path:
        self.config_path.write_text(
            yaml.safe_dump(payload or self.config_payload()), encoding="utf-8"
        )
        return self.config_path

    def run(self, *, overwrite: bool = False, output: Path | None = None) -> dict:
        if not self.config_path.exists():
            self.write_config()
        return run_build_jersey_review_panels(
            measurement_dir=self.measurement,
            project_root=self.project,
            config=self.config_path,
            output_dir=output or self.output,
            overwrite=overwrite,
        )


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.payload = self.fixture.config_payload()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_config(self) -> None:
        validated = validate_jersey_review_config(self.payload)
        self.assertEqual(
            validated["schema_version"], "reid_jersey_review_panel_config_v1"
        )
        repo_payload = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
        validate_jersey_review_config(repo_payload)

    def test_rejects_input_and_universe_flags(self) -> None:
        cases = (
            ("input", "require_measurement_status_ok", False),
            ("input", "use_existing_crop_signals_only", False),
            ("input", "allow_video_input", True),
            ("input", "allow_new_crop_extraction", True),
            ("input", "require_source_crop_sha_match", False),
            ("review_universe", "include_all_selected_crops", False),
            ("review_universe", "deduplicate_review_items", False),
            ("review_universe", "preserve_all_group_memberships", False),
        )
        for section, key, value in cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload[section][key] = value
                with self.assertRaises(JerseyReviewError):
                    validate_jersey_review_config(payload)

    def test_rejects_panel_flags(self) -> None:
        cases = (
            ("allow_visual_enhancement", True),
            ("roi_zoom_interpolation", "linear"),
            ("annotate_predicted_number", True),
            ("annotate_visibility_class", True),
            ("output_format", "jpg"),
            ("columns", 0),
            ("rows", -1),
        )
        for key, value in cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["panel"][key] = value
                with self.assertRaises(JerseyReviewError):
                    validate_jersey_review_config(payload)
        payload = copy.deepcopy(self.payload)
        payload["groups"]["roi_height_top"]["count"] = 0
        with self.assertRaises(JerseyReviewError):
            validate_jersey_review_config(payload)
        payload = copy.deepcopy(self.payload)
        payload["groups"]["sharpness_by_roi_height_stratum"][
            "laplacian_top_count"
        ] = 0
        with self.assertRaises(JerseyReviewError):
            validate_jersey_review_config(payload)

    def test_rejects_recognition_and_safety_flags(self) -> None:
        for key in (
            "OCR_enabled",
            "recognizer_enabled",
            "checkpoint_required",
            "jersey_number_candidate_enabled",
            "automatic_jersey_assignment_enabled",
        ):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["recognition"][key] = True
                with self.assertRaises(JerseyReviewError):
                    validate_jersey_review_config(payload)
        safety_cases = (
            ("source_crops_immutable", False),
            ("measurement_artifacts_immutable", False),
            ("panels_are_display_only", False),
            ("panel_images_are_not_recognition_input", False),
            ("identity_ground_truth_available", True),
            ("accuracy_claim_allowed", True),
            ("team_assignment_enabled", True),
            ("global_id_rewrite_enabled", True),
        )
        for key, value in safety_cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["safety"][key] = value
                with self.assertRaises(JerseyReviewError):
                    validate_jersey_review_config(payload)

    def test_rejects_bad_critical_segment_ids(self) -> None:
        for value in ([], ["raw_1_s01", "raw_1_s01"], ["../evil"], "raw_1_s01"):
            with self.subTest(value=value):
                payload = copy.deepcopy(self.payload)
                payload["groups"]["critical_segments"]["segment_ids"] = value
                with self.assertRaises(JerseyReviewError):
                    validate_jersey_review_config(payload)


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.config = validate_jersey_review_config(self.fixture.config_payload())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _load(self, *, project_root: Path | None = None) -> dict:
        return load_measurement_inputs(
            measurement_dir=self.fixture.measurement,
            project_root=project_root or self.fixture.project,
            config=self.config,
        )

    def _crop_signal_path(self) -> Path:
        return self.fixture.measurement / "jersey_visibility_crop_signals.jsonl"

    def test_valid_inputs_load_and_exclude_no_crop_segments(self) -> None:
        inputs = self._load()
        self.assertEqual(len(inputs["crop_rows"]), 5)
        self.assertEqual(len(inputs["segment_rows"]), 4)
        self.assertEqual(len(inputs["images"]), 5)
        crop_segments = {row["segment_id"] for row in inputs["crop_rows"]}
        self.assertNotIn("raw_4_full", crop_segments)

    def test_non_ok_summary_rejected(self) -> None:
        path = self.fixture.measurement / "jersey_visibility_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "failed"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_schema_mismatch_rejected(self) -> None:
        rows = _load_jsonl(self._crop_signal_path())
        rows[0]["schema_version"] = "wrong_schema"
        _write_jsonl(self._crop_signal_path(), rows)
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_count_mismatch_rejected(self) -> None:
        path = self.fixture.measurement / "jersey_visibility_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["total_selected_crop_count"] = 99
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_duplicate_crop_id_rejected(self) -> None:
        rows = _load_jsonl(self._crop_signal_path())
        rows.append(dict(rows[0]))
        _write_jsonl(self._crop_signal_path(), rows)
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_missing_crop_and_sha_mismatch_rejected(self) -> None:
        crop_id = "track_1_frame_10_rank_1"
        path = self.fixture.crop_paths[crop_id]
        original = path.read_bytes()
        with self.subTest(case="sha_mismatch"):
            path.write_bytes(original + b"tamper")
            with self.assertRaises(JerseyReviewError) as ctx:
                self._load()
            self.assertIn("SHA mismatch", str(ctx.exception))
        with self.subTest(case="missing"):
            path.unlink()
            with self.assertRaises(JerseyReviewError) as ctx:
                self._load()
            self.assertIn("missing", str(ctx.exception))

    def test_roi_out_of_bounds_rejected(self) -> None:
        rows = _load_jsonl(self._crop_signal_path())
        rows[0]["roi_x_max"] = rows[0]["crop_width_px"] + 5
        _write_jsonl(self._crop_signal_path(), rows)
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_nan_rejected(self) -> None:
        text = self._crop_signal_path().read_text(encoding="utf-8")
        lines = text.splitlines()
        lines[0] = lines[0].replace(
            f'"laplacian_variance": {self.fixture.crop_rows[0]["laplacian_variance"]}',
            '"laplacian_variance": NaN',
        )
        self.assertIn("NaN", lines[0])
        self._crop_signal_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(JerseyReviewError):
            self._load()

    def test_project_root_escape_rejected(self) -> None:
        with self.assertRaises(JerseyReviewError) as ctx:
            self._load(project_root=self.fixture.measurement)
        self.assertIn("escapes project root", str(ctx.exception))


class ReviewItemAndGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.result = self.fixture.run()
        self.items = _load_jsonl(self.fixture.output / REVIEW_ITEMS_NAME)
        self.memberships = _load_jsonl(
            self.fixture.output / GROUP_MEMBERSHIPS_NAME
        )
        self.summary = json.loads(
            (self.fixture.output / SUMMARY_NAME).read_text(encoding="utf-8")
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _group_crop_ids(self, name: str) -> list[str]:
        rows = sorted(
            (row for row in self.memberships if row["group_name"] == name),
            key=lambda row: row["group_item_rank"],
        )
        return [row["crop_id"] for row in rows]

    def test_canonical_items_order_and_manual_fields(self) -> None:
        expected_order = [
            "track_1_frame_10_rank_1",
            "track_1_frame_20_rank_2",
            "track_2_frame_5_rank_1",
            "track_2_frame_9_rank_2",
            "track_3_frame_7_rank_1",
        ]
        self.assertEqual([row["crop_id"] for row in self.items], expected_order)
        self.assertEqual(
            [row["review_index"] for row in self.items], [1, 2, 3, 4, 5]
        )
        for row in self.items:
            for field in MANUAL_FIELDS:
                self.assertIsNone(row[field])
            for forbidden in (
                "predicted_number",
                "ocr_confidence",
                "visibility_score",
                "readability_score",
                "automatic_label",
            ):
                self.assertNotIn(forbidden, row)

    def test_master_group_and_dedup(self) -> None:
        master = self._group_crop_ids("all_selected_crops")
        self.assertEqual(master, [row["crop_id"] for row in self.items])
        ids = [row["review_item_id"] for row in self.items]
        self.assertEqual(len(set(ids)), len(ids))
        for row in self.items:
            self.assertIn("all_selected_crops", row["group_memberships"])
            self.assertGreaterEqual(len(row["group_memberships"]), 1)
        multi = [
            row
            for row in self.items
            if len(row["group_memberships"]) > 2
        ]
        self.assertTrue(multi, "expected crops in multiple groups")

    def test_metric_group_semantics(self) -> None:
        self.assertEqual(
            self._group_crop_ids("roi_height_top"),
            ["track_1_frame_10_rank_1", "track_1_frame_20_rank_2"],
        )
        self.assertEqual(
            self._group_crop_ids("roi_height_bottom"),
            ["track_3_frame_7_rank_1", "track_2_frame_9_rank_2"],
        )
        self.assertEqual(
            self._group_crop_ids("entropy_top"),
            ["track_3_frame_7_rank_1", "track_2_frame_9_rank_2"],
        )
        self.assertEqual(
            self._group_crop_ids("roi_contamination_low"),
            ["track_1_frame_10_rank_1", "track_1_frame_20_rank_2"],
        )
        self.assertEqual(
            self._group_crop_ids("roi_contamination_high"),
            ["track_3_frame_7_rank_1", "track_2_frame_9_rank_2"],
        )
        manual = self._group_crop_ids("all_manual_segment_crops")
        self.assertEqual(
            manual, ["track_1_frame_10_rank_1", "track_1_frame_20_rank_2"]
        )
        for name in ("roi_height_top", "entropy_top"):
            rows = [r for r in self.memberships if r["group_name"] == name]
            for row in rows:
                self.assertIsNotNone(row["metric_value"])

    def test_stratum_sharpness_groups_and_requested_gt_available(self) -> None:
        self.assertEqual(
            self._group_crop_ids("sharpness_40_63_laplacian_top"),
            [
                "track_1_frame_10_rank_1",
                "track_1_frame_20_rank_2",
                "track_2_frame_5_rank_1",
            ],
        )
        self.assertEqual(
            self._group_crop_ids("sharpness_40_63_laplacian_bottom"),
            ["track_2_frame_5_rank_1", "track_1_frame_20_rank_2"],
        )
        self.assertEqual(
            self._group_crop_ids("sharpness_24_39_laplacian_top"),
            ["track_2_frame_9_rank_2", "track_3_frame_7_rank_1"],
        )
        group_names = {g["group_name"] for g in self.summary["groups"]}
        self.assertNotIn("sharpness_0_23_laplacian_top", group_names)
        short = next(
            g
            for g in self.summary["groups"]
            if g["group_name"] == "sharpness_24_39_laplacian_top"
        )
        self.assertEqual(short["requested_count"], 3)
        self.assertEqual(short["actual_count"], 2)
        self.assertTrue(short["notes"])
        for group in self.summary["groups"]:
            ids = self._group_crop_ids(group["group_name"])
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(group["duplicate_item_count_within_group"], 0)

    def test_critical_group_and_no_crop_note(self) -> None:
        self.assertEqual(
            self._group_crop_ids("critical_segments"),
            ["track_1_frame_10_rank_1", "track_1_frame_20_rank_2"],
        )
        self.assertEqual(
            self.summary["critical_no_crop_segments"], ["raw_4_full"]
        )

    def test_unknown_critical_segment_rejected(self) -> None:
        payload = self.fixture.config_payload()
        payload["groups"]["critical_segments"]["segment_ids"] = ["raw_99_s01"]
        self.fixture.write_config(payload)
        with self.assertRaises(JerseyReviewError):
            self.fixture.run(output=self.fixture.root / "other_output")


class EmptySharpnessStratumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.fixture.set_declared_strata([0, 24, 40, 64])
        self.result = self.fixture.run()
        self.summary = json.loads(
            (self.fixture.output / SUMMARY_NAME).read_text(encoding="utf-8")
        )
        self.memberships = _load_jsonl(
            self.fixture.output / GROUP_MEMBERSHIPS_NAME
        )
        self.panel_index = _load_jsonl(self.fixture.output / PANEL_INDEX_NAME)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _empty_groups(self) -> list[dict]:
        return [
            group
            for group in self.summary["groups"]
            if group["sharpness_size_stratum"] == "0_23"
        ]

    def test_declared_unobserved_stratum_has_four_explicit_groups(self) -> None:
        groups = self._empty_groups()
        self.assertEqual(
            [group["group_name"] for group in groups],
            [
                "sharpness_0_23_laplacian_top",
                "sharpness_0_23_laplacian_bottom",
                "sharpness_0_23_tenengrad_top",
                "sharpness_0_23_tenengrad_bottom",
            ],
        )
        self.assertEqual(
            [group["group_order"] for group in groups],
            list(range(groups[0]["group_order"], groups[0]["group_order"] + 4)),
        )
        expected_requested = [3, 2, 2, 2]
        for group, requested in zip(groups, expected_requested):
            self.assertEqual(group["requested_count"], requested)
            self.assertEqual(group["actual_count"], 0)
            self.assertEqual(group["panel_page_count"], 0)
            self.assertEqual(group["duplicate_item_count_within_group"], 0)
            self.assertEqual(
                group["empty_reason"],
                "no_crop_signals_in_declared_stratum",
            )
            self.assertEqual(group["notes"], [])
            self.assertIsNotNone(group["metric_name"])
            self.assertIsNotNone(group["rank_source"])

    def test_empty_groups_emit_no_membership_panel_index_or_png(self) -> None:
        empty_names = {group["group_name"] for group in self._empty_groups()}
        self.assertFalse(
            [row for row in self.memberships if row["group_name"] in empty_names]
        )
        self.assertFalse(
            [row for row in self.panel_index if row["group_name"] in empty_names]
        )
        for name in empty_names:
            self.assertFalse(
                (self.fixture.output / PANELS_DIRNAME / name).exists()
            )
        self.assertFalse(
            list(
                (self.fixture.output / PANELS_DIRNAME).glob(
                    "sharpness_0_23_*"
                )
            )
        )

    def test_global_counts_include_empty_groups_only_where_applicable(self) -> None:
        counts = self.summary["counts"]
        self.assertEqual(counts["group_count"], 23)
        self.assertEqual(counts["empty_group_count"], 4)
        self.assertEqual(self.summary["empty_strata"], ["0_23"])
        # Empty groups add no pages, PNGs, memberships, or canonical items.
        self.assertEqual(counts["total_panel_page_count"], 20)
        self.assertEqual(counts["total_panel_png_count"], 20)
        self.assertEqual(counts["canonical_review_item_count"], 5)
        self.assertEqual(counts["group_membership_count"], len(self.memberships))
        self.assertEqual(
            sorted(path.name for path in self.fixture.output.iterdir()),
            sorted(list(OUTPUT_NAMES) + [PANELS_DIRNAME]),
        )

    def test_declared_order_is_deterministic_and_observed_selection_unchanged(self) -> None:
        group_names = [group["group_name"] for group in self.summary["groups"]]
        sharpness_names = [
            name for name in group_names if name.startswith("sharpness_")
        ]
        self.assertEqual(
            sharpness_names[:4],
            [
                "sharpness_0_23_laplacian_top",
                "sharpness_0_23_laplacian_bottom",
                "sharpness_0_23_tenengrad_top",
                "sharpness_0_23_tenengrad_bottom",
            ],
        )
        observed_rows = sorted(
            (
                row
                for row in self.memberships
                if row["group_name"] == "sharpness_40_63_laplacian_top"
            ),
            key=lambda row: row["group_item_rank"],
        )
        self.assertEqual(
            [row["crop_id"] for row in observed_rows],
            [
                "track_1_frame_10_rank_1",
                "track_1_frame_20_rank_2",
                "track_2_frame_5_rank_1",
            ],
        )
        second_output = self.fixture.root / "second_output"
        self.fixture.run(output=second_output)
        second_summary = json.loads(
            (second_output / SUMMARY_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [
                (group["group_name"], group["group_order"])
                for group in self.summary["groups"]
            ],
            [
                (group["group_name"], group["group_order"])
                for group in second_summary["groups"]
            ],
        )

    def test_observed_undeclared_stratum_is_rejected_without_output(self) -> None:
        other_output = self.fixture.root / "undeclared_output"
        self.fixture.set_declared_strata([40, 64])
        with self.assertRaises(JerseyReviewError) as ctx:
            self.fixture.run(output=other_output)
        self.assertIn("undeclared sharpness strata", str(ctx.exception))
        self.assertFalse(other_output.exists())
        self.assertFalse(
            list(self.fixture.root.glob("_tmp_reid_jersey_review_*"))
        )
        self.assertFalse(
            list(self.fixture.root.glob("_backup_reid_jersey_review_*"))
        )


class RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.before = {
            crop_id: _sha256(path)
            for crop_id, path in self.fixture.crop_paths.items()
        }
        self.result = self.fixture.run()
        self.panel_index = _load_jsonl(self.fixture.output / PANEL_INDEX_NAME)
        self.summary = json.loads(
            (self.fixture.output / SUMMARY_NAME).read_text(encoding="utf-8")
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_panel_pages_counts_and_dimensions(self) -> None:
        # master 2 pages + manual 1 + 8 metric groups + 8 stratum sharpness
        # groups + critical 1, each single page.
        self.assertEqual(len(self.panel_index), 20)
        self.assertEqual(self.summary["counts"]["group_count"], 19)
        self.assertEqual(self.summary["counts"]["empty_group_count"], 0)
        self.assertEqual(self.summary["empty_strata"], [])
        self.assertEqual(self.summary["counts"]["total_panel_page_count"], 20)
        self.assertEqual(self.summary["counts"]["total_panel_png_count"], 20)
        pngs = sorted(
            (self.fixture.output / PANELS_DIRNAME).rglob("*.png")
        )
        self.assertEqual(len(pngs), 20)
        for png in pngs[:3]:
            image = cv2.imread(str(png), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (2 * TILE_HEIGHT, 2 * TILE_WIDTH))
        master_pages = [
            row
            for row in self.panel_index
            if row["group_name"] == "all_selected_crops"
        ]
        self.assertEqual([row["page_number"] for row in master_pages], [1, 2])
        self.assertEqual(master_pages[0]["tile_count"], 4)
        self.assertEqual(master_pages[1]["tile_count"], 1)

    def test_panel_index_consistency_and_tile_order(self) -> None:
        items = _load_jsonl(self.fixture.output / REVIEW_ITEMS_NAME)
        by_id = {row["review_item_id"]: row for row in items}
        for row in self.panel_index:
            path = self.fixture.output / row["panel_path"]
            self.assertTrue(path.is_file())
            self.assertEqual(_sha256(path), row["panel_sha256"])
            self.assertEqual(path.stat().st_size, row["panel_byte_size"])
            self.assertEqual(
                len(row["review_item_ids"]),
                len(row["source_crop_sha256_values"]),
            )
            for rid, sha in zip(
                row["review_item_ids"], row["source_crop_sha256_values"]
            ):
                self.assertEqual(by_id[rid]["source_crop_sha256"], sha)
        master_first = next(
            row
            for row in self.panel_index
            if row["group_name"] == "all_selected_crops"
            and row["page_number"] == 1
        )
        expected = [row["review_item_id"] for row in items[:4]]
        self.assertEqual(master_first["review_item_ids"], expected)
        for item in items:
            self.assertEqual(
                item["master_panel_path"].startswith("panels/00_all_selected_crops/"),
                True,
            )
            self.assertIsNotNone(item["master_panel_page"])
            self.assertIsNotNone(item["master_panel_tile_index"])

    def test_deterministic_rerun(self) -> None:
        second_out = self.fixture.root / "review_output_second"
        self.fixture.run(output=second_out)
        for name in (REVIEW_ITEMS_NAME, GROUP_MEMBERSHIPS_NAME, PANEL_INDEX_NAME):
            self.assertEqual(
                (self.fixture.output / name).read_bytes(),
                (second_out / name).read_bytes(),
                name,
            )
        self.assertEqual(
            (self.fixture.output / ANNOTATION_TEMPLATE_NAME).read_bytes(),
            (second_out / ANNOTATION_TEMPLATE_NAME).read_bytes(),
        )
        first_shas = {
            row["panel_path"]: row["panel_sha256"] for row in self.panel_index
        }
        second_shas = {
            row["panel_path"]: row["panel_sha256"]
            for row in _load_jsonl(second_out / PANEL_INDEX_NAME)
        }
        self.assertEqual(first_shas, second_shas)

    def test_no_individual_crop_exports_and_sources_unchanged(self) -> None:
        after = {
            crop_id: _sha256(path)
            for crop_id, path in self.fixture.crop_paths.items()
        }
        self.assertEqual(self.before, after)
        for entry in (self.fixture.output / PANELS_DIRNAME).rglob("*"):
            if entry.is_file():
                self.assertEqual(entry.suffix, ".png")
                self.assertTrue(entry.name.startswith("page_"))
        self.assertFalse(list(self.fixture.output.rglob("*.jpg")))
        self.assertFalse(list(self.fixture.output.rglob("*.html")))


class AnnotationTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))
        self.fixture.run()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_annotation_template_contract(self) -> None:
        path = self.fixture.output / ANNOTATION_TEMPLATE_NAME
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        rows = list(csv.reader(text.splitlines()))
        self.assertEqual(rows[0], list(ANNOTATION_COLUMNS))
        self.assertEqual(len(rows) - 1, 5)
        manual_start = len(ANNOTATION_COLUMNS) - len(MANUAL_FIELDS)
        crop_ids = []
        for row in rows[1:]:
            self.assertEqual(row[manual_start:], [""] * len(MANUAL_FIELDS))
            memberships = json.loads(row[ANNOTATION_COLUMNS.index("group_memberships")])
            self.assertIsInstance(memberships, list)
            self.assertIn("all_selected_crops", memberships)
            crop_ids.append(row[ANNOTATION_COLUMNS.index("crop_id")])
        self.assertEqual(
            crop_ids,
            [
                "track_1_frame_10_rank_1",
                "track_1_frame_20_rank_2",
                "track_2_frame_5_rank_1",
                "track_2_frame_9_rank_2",
                "track_3_frame_7_rank_1",
            ],
        )


class AtomicOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticReviewFixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _no_stray_dirs(self) -> None:
        parent = self.fixture.output.parent
        self.assertFalse(list(parent.glob("_tmp_reid_jersey_review_*")))
        self.assertFalse(list(parent.glob("_backup_reid_jersey_review_*")))

    def test_output_contract_collision_overwrite(self) -> None:
        self.fixture.run()
        entries = sorted(p.name for p in self.fixture.output.iterdir())
        self.assertEqual(entries, sorted(list(OUTPUT_NAMES) + [PANELS_DIRNAME]))
        with self.assertRaises(JerseyReviewError):
            self.fixture.run()
        self.fixture.run(overwrite=True)
        self._no_stray_dirs()

    def test_failure_creates_no_final_output(self) -> None:
        crop_path = self.fixture.crop_paths["track_2_frame_5_rank_1"]
        crop_path.write_bytes(crop_path.read_bytes() + b"tamper")
        with self.assertRaises(JerseyReviewError):
            self.fixture.run()
        self.assertFalse(self.fixture.output.exists())
        self._no_stray_dirs()


class CliAndStaticTests(unittest.TestCase):
    def test_help_has_only_approved_inputs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--measurement-dir",
            "--project-root",
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
            "--jersey-number",
            "--team",
            "--global-id",
            "--similarity-threshold",
            "--manual-label",
            "--segment-view-dir",
            "--segmented-regression-dir",
        ):
            self.assertNotIn(option, result.stdout)

    def test_product_source_static_safety(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in (
            "VideoCapture",
            "import torch",
            "torchreid",
            "MMOCR",
            "PaddleOCR",
            "Tesseract",
            "EasyOCR",
            "pytesseract",
            "urllib",
            "requests.",
            "http://",
            "https://",
            "GaussianBlur",
            "equalizeHist",
            "detailEnhance",
            "fastNlMeansDenoising",
            "INTER_LINEAR",
            "INTER_CUBIC",
            "INTER_LANCZOS",
            "visibility_score",
            "readability_score",
        ):
            self.assertNotIn(forbidden, text, forbidden)
        # predicted_number may appear only as the rejected config key.
        self.assertEqual(
            text.count("predicted_number"),
            text.count("annotate_predicted_number"),
        )
        self.assertEqual(text.count("cv2.imwrite"), 1)
        self.assertIn("_write_panel_png", text)
        self.assertIn("INTER_NEAREST", text)


if __name__ == "__main__":
    unittest.main()
