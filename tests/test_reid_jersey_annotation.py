"""Synthetic tests for Stage 5C-A2b3 jersey manual annotation validation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.jersey_annotation import (
    ANNOTATION_COLUMNS,
    CONFIG_SCHEMA,
    DECISION_FIELDS,
    PROVENANCE_COLUMNS,
    REPORT_SCHEMA,
    JerseyAnnotationError,
    load_jersey_manual_review_config,
    run_validate_jersey_review_annotations,
    validate_jersey_manual_review_config,
    validate_jersey_review_annotations,
)
from football_analytics.reid.jersey_review import (
    ANNOTATION_TEMPLATE_NAME,
    MANUAL_FIELDS,
    REVIEW_ITEMS_NAME,
    SUMMARY_NAME,
)

REPO_CONFIG = _PROJECT_ROOT / "configs/reid/jersey_manual_review_stage5c.yaml"
SCRIPT = _PROJECT_ROOT / "scripts/validate_jersey_review_annotations.py"
SOURCE = _SRC_DIR / "football_analytics/reid/jersey_annotation.py"
DOC = _PROJECT_ROOT / "docs/setup/stage5c-jersey-manual-review-protocol.md"

REVIEWED_AT = "2026-07-22T12:00:00+03:00"
REVIEWER = "pilot_reviewer"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            handle.write("\n")


def _base_config() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))


def _make_item(index: int, *, groups: list[str] | None = None) -> dict:
    crop_id = f"track_{index}_frame_{10 * index}_rank_1"
    return {
        "schema_version": "reid_jersey_review_item_v1",
        "review_item_id": f"review_{crop_id}",
        "review_index": index,
        "segment_id": f"raw_{index}_full",
        "raw_track_id": index,
        "crop_id": crop_id,
        "frame_index": 10 * index,
        "master_panel_path": f"panels/all_selected_crops/page_{((index - 1) // 12) + 1:03d}.png",
        "master_panel_page": ((index - 1) // 12) + 1,
        "master_panel_tile_index": (index - 1) % 12,
        "group_memberships": groups
        or ["all_selected_crops", "roi_height_top" if index == 1 else "roi_height_bottom"],
        **{field: None for field in MANUAL_FIELDS},
    }


def _write_template(path: Path, items: list[dict], *, fills: list[dict] | None = None) -> None:
    fills = fills or [{} for _ in items]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(ANNOTATION_COLUMNS)
        for item, fill in zip(items, fills):
            row = [
                item["review_item_id"],
                item["review_index"],
                item["segment_id"],
                item["raw_track_id"],
                item["crop_id"],
                item["frame_index"],
                item["master_panel_path"],
                item["master_panel_page"],
                item["master_panel_tile_index"],
                json.dumps(item["group_memberships"], ensure_ascii=False),
            ]
            for field in MANUAL_FIELDS:
                row.append(fill.get(field, ""))
            writer.writerow(row)


def _build_review_dir(root: Path, *, n_items: int = 3) -> tuple[Path, list[dict], Path]:
    review_dir = root / "review"
    review_dir.mkdir(parents=True)
    (review_dir / "panels").mkdir()
    items = [_make_item(i + 1) for i in range(n_items)]
    _write_jsonl(review_dir / REVIEW_ITEMS_NAME, items)
    summary = {
        "schema_version": "reid_jersey_review_summary_v1",
        "status": "ok",
        "counts": {"canonical_review_item_count": n_items},
    }
    (review_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    template = review_dir / ANNOTATION_TEMPLATE_NAME
    _write_template(template, items)
    return review_dir, items, template


def _valid_visible_no() -> dict:
    return {
        "manual_crop_valid": "valid",
        "manual_back_facing": "yes",
        "manual_number_visible": "no",
        "manual_number_readable": "no",
        "manual_digit_count": "0",
        "manual_jersey_number": "",
        "manual_contamination_affects_number_region": "no",
        "manual_notes": "",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
    }


def _valid_readable_one(digit: str = "7") -> dict:
    return {
        "manual_crop_valid": "valid",
        "manual_back_facing": "yes",
        "manual_number_visible": "yes",
        "manual_number_readable": "yes",
        "manual_digit_count": "1",
        "manual_jersey_number": digit,
        "manual_contamination_affects_number_region": "no",
        "manual_notes": "clear back",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
    }


class AnnotationFixture:
    def __init__(self, root: Path, *, n_items: int = 3) -> None:
        self.root = root
        self.config_path = root / "config.yaml"
        self.config_path.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
        self.review_dir, self.items, self.template_path = _build_review_dir(
            root, n_items=n_items
        )
        self.annotations_path = root / "annotations.csv"
        shutil.copyfile(self.template_path, self.annotations_path)
        self.report_path = root / "report.json"

    def write_annotations(self, fills: list[dict]) -> None:
        _write_template(self.annotations_path, self.items, fills=fills)

    def validate(self, *, overwrite: bool = False) -> dict:
        return run_validate_jersey_review_annotations(
            review_dir=self.review_dir,
            annotations_csv=self.annotations_path,
            config=self.config_path,
            report_path=self.report_path,
            overwrite=overwrite,
        )


class ConfigValidationTests(unittest.TestCase):
    def test_repo_config_valid(self) -> None:
        payload = load_jersey_manual_review_config(REPO_CONFIG)
        self.assertEqual(payload["schema_version"], CONFIG_SCHEMA)

    def test_reject_exact_template_identity_false(self) -> None:
        cfg = _base_config()
        cfg["input"]["require_exact_template_identity"] = False
        with self.assertRaises(JerseyAnnotationError):
            validate_jersey_manual_review_config(cfg)

    def test_reject_missing_extra_duplicate_allows(self) -> None:
        for key in (
            "allow_missing_review_rows",
            "allow_extra_review_rows",
            "allow_duplicate_review_items",
        ):
            cfg = _base_config()
            cfg["input"][key] = True
            with self.assertRaises(JerseyAnnotationError):
                validate_jersey_manual_review_config(cfg)

    def test_reject_immutability_false(self) -> None:
        for key in (
            "review_items_immutable",
            "panels_immutable",
            "source_crops_immutable",
        ):
            cfg = _base_config()
            cfg["safety"][key] = False
            with self.assertRaises(JerseyAnnotationError):
                validate_jersey_manual_review_config(cfg)

    def test_reject_gallery_identity_team_update_true(self) -> None:
        for key in (
            "annotation_does_not_update_gallery",
            "annotation_does_not_assign_identity",
            "annotation_does_not_assign_team",
        ):
            cfg = _base_config()
            cfg["safety"][key] = False
            with self.assertRaises(JerseyAnnotationError):
                validate_jersey_manual_review_config(cfg)

    def test_reject_gt_and_accuracy_true(self) -> None:
        for key in ("identity_ground_truth_available", "accuracy_claim_allowed"):
            cfg = _base_config()
            cfg["safety"][key] = True
            with self.assertRaises(JerseyAnnotationError):
                validate_jersey_manual_review_config(cfg)

    def test_reject_automatic_prediction_and_identity_label_allows(self) -> None:
        for key in ("reject_identity_labels", "reject_automatic_predictions"):
            cfg = _base_config()
            cfg["validation"][key] = False
            with self.assertRaises(JerseyAnnotationError):
                validate_jersey_manual_review_config(cfg)

    def test_reject_unsafe_jersey_number_policy(self) -> None:
        cfg = _base_config()
        cfg["jersey_number"]["allow_non_digit_characters"] = True
        with self.assertRaises(JerseyAnnotationError):
            validate_jersey_manual_review_config(cfg)
        cfg = _base_config()
        cfg["jersey_number"]["allow_more_than_two_digits"] = True
        with self.assertRaises(JerseyAnnotationError):
            validate_jersey_manual_review_config(cfg)

    def test_reject_timezone_required_false(self) -> None:
        cfg = _base_config()
        cfg["timestamp"]["timezone_required"] = False
        with self.assertRaises(JerseyAnnotationError):
            validate_jersey_manual_review_config(cfg)

    def test_reject_invalid_allowed_values(self) -> None:
        cfg = _base_config()
        cfg["allowed_values"]["manual_crop_valid"] = ["valid", "bad"]
        with self.assertRaises(JerseyAnnotationError):
            validate_jersey_manual_review_config(cfg)


class ExactTemplateIdentityTests(unittest.TestCase):
    def test_blank_template_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["reviewed_row_count"], 0)
            self.assertEqual(result["unreviewed_row_count"], len(fx.items))

    def test_changed_provenance_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fills = [{} for _ in fx.items]
            _write_template(fx.annotations_path, fx.items, fills=fills)
            text = fx.annotations_path.read_text(encoding="utf-8")
            text = text.replace(fx.items[0]["crop_id"], "tampered_crop", 1)
            fx.annotations_path.write_text(text, encoding="utf-8")
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertTrue(
                {"provenance_mismatch", "missing_review_item", "extra_review_item"}
                & codes
            )

    def test_reordered_rows_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            reordered = [fx.items[1], fx.items[0], fx.items[2]]
            _write_template(fx.annotations_path, reordered)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = [e["code"] for e in result["report"]["validation"]["errors"]]
            self.assertIn("row_order_mismatch", codes)

    def test_missing_row_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            _write_template(fx.annotations_path, fx.items[:2])
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("row_count_mismatch", codes)
            self.assertIn("missing_review_item", codes)

    def test_extra_row_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            extra = fx.items + [_make_item(99)]
            _write_template(fx.annotations_path, extra)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("extra_review_item", codes)

    def test_duplicate_review_item_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            dup = [fx.items[0], fx.items[0], fx.items[2]]
            _write_template(fx.annotations_path, dup)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("duplicate_review_item", codes)

    def test_group_membership_mismatch_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            tampered = copy.deepcopy(fx.items)
            tampered[0]["group_memberships"] = ["all_selected_crops", "tampered_group"]
            _write_template(fx.annotations_path, tampered)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("provenance_mismatch", codes)


class UnreviewedTests(unittest.TestCase):
    def test_fully_blank_manual_fields_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["unreviewed_row_count"], 3)
            self.assertEqual(result["reviewed_row_count"], 0)

    def test_reviewer_alone_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fills = [{"reviewer": REVIEWER}, {}, {}]
            fx.write_annotations(fills)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("metadata_without_decision", codes)

    def test_timestamp_alone_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fills = [{"reviewed_at": REVIEWED_AT}, {}, {}]
            fx.write_annotations(fills)
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            codes = {e["code"] for e in result["report"]["validation"]["errors"]}
            self.assertIn("metadata_without_decision", codes)


class ValidCombinationTests(unittest.TestCase):
    def test_visible_no_readable_no_digit_0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fills = [_valid_visible_no(), {}, {}]
            fx.write_annotations(fills)
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["reviewed_row_count"], 1)
            self.assertEqual(result["report"]["counts"]["number_visible_no_count"], 1)

    def test_visible_yes_readable_no_digit_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = {
                "manual_crop_valid": "valid",
                "manual_back_facing": "no",
                "manual_number_visible": "yes",
                "manual_number_readable": "no",
                "manual_digit_count": "uncertain",
                "manual_jersey_number": "",
                "manual_contamination_affects_number_region": "yes",
                "manual_notes": "partial left digit",
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            }
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")

    def test_visible_yes_readable_yes_single_digit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fx.write_annotations([_valid_readable_one("7"), {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["report"]["counts"]["readable_number_count"], 1)

    def test_visible_yes_readable_yes_two_digit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = _valid_readable_one()
            fill["manual_digit_count"] = "2"
            fill["manual_jersey_number"] = "10"
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")

    def test_leading_zero_string_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = _valid_readable_one()
            fill["manual_digit_count"] = "2"
            fill["manual_jersey_number"] = "01"
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            with fx.annotations_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["manual_jersey_number"], "01")

    def test_visible_uncertain_readable_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = {
                "manual_crop_valid": "uncertain",
                "manual_back_facing": "uncertain",
                "manual_number_visible": "uncertain",
                "manual_number_readable": "uncertain",
                "manual_digit_count": "uncertain",
                "manual_jersey_number": "",
                "manual_contamination_affects_number_region": "uncertain",
                "manual_notes": "",
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            }
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")

    def test_invalid_crop_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = {
                "manual_crop_valid": "invalid",
                "manual_back_facing": "uncertain",
                "manual_number_visible": "",
                "manual_number_readable": "",
                "manual_digit_count": "",
                "manual_jersey_number": "",
                "manual_contamination_affects_number_region": "",
                "manual_notes": "wrong crop",
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            }
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["report"]["counts"]["invalid_crop_count"], 1)


class InvalidCombinationTests(unittest.TestCase):
    def _invalid(self, fill: dict) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            return [e["code"] for e in result["report"]["validation"]["errors"]]

    def test_visible_no_readable_yes(self) -> None:
        fill = _valid_visible_no()
        fill["manual_number_readable"] = "yes"
        fill["manual_digit_count"] = "1"
        fill["manual_jersey_number"] = "7"
        codes = self._invalid(fill)
        self.assertTrue(
            {"visible_no_readable_mismatch", "readable_yes_requires_visible_yes"}
            & set(codes)
        )

    def test_readable_yes_jersey_blank(self) -> None:
        fill = _valid_readable_one()
        fill["manual_jersey_number"] = ""
        codes = self._invalid(fill)
        self.assertIn("readable_yes_missing_jersey", codes)

    def test_jersey_with_readable_no(self) -> None:
        fill = _valid_readable_one()
        fill["manual_number_readable"] = "no"
        fill["manual_digit_count"] = "1"
        codes = self._invalid(fill)
        self.assertTrue(
            {"readable_no_jersey_present", "jersey_without_readable_yes"} & set(codes)
        )

    def test_digit_count_mismatch(self) -> None:
        fill = _valid_readable_one()
        fill["manual_digit_count"] = "2"
        fill["manual_jersey_number"] = "7"
        codes = self._invalid(fill)
        self.assertIn("jersey_digit_count_mismatch", codes)

    def test_non_digit_jersey(self) -> None:
        fill = _valid_readable_one()
        fill["manual_jersey_number"] = "7a"
        codes = self._invalid(fill)
        self.assertIn("invalid_jersey_number", codes)

    def test_three_digit_jersey(self) -> None:
        fill = _valid_readable_one()
        fill["manual_digit_count"] = "2"
        fill["manual_jersey_number"] = "100"
        codes = self._invalid(fill)
        self.assertIn("invalid_jersey_number", codes)

    def test_partial_reviewed_row(self) -> None:
        fill = {
            "manual_crop_valid": "valid",
            "manual_back_facing": "",
            "manual_number_visible": "",
            "manual_number_readable": "",
            "manual_digit_count": "",
            "manual_jersey_number": "",
            "manual_contamination_affects_number_region": "",
            "manual_notes": "",
            "reviewer": REVIEWER,
            "reviewed_at": REVIEWED_AT,
        }
        codes = self._invalid(fill)
        self.assertIn("partial_reviewed_row", codes)

    def test_missing_reviewer(self) -> None:
        fill = _valid_visible_no()
        fill["reviewer"] = ""
        codes = self._invalid(fill)
        self.assertIn("missing_reviewer", codes)

    def test_timestamp_without_timezone(self) -> None:
        fill = _valid_visible_no()
        fill["reviewed_at"] = "2026-07-22T12:00:00"
        codes = self._invalid(fill)
        self.assertIn("invalid_reviewed_at", codes)

    def test_invalid_crop_with_jersey(self) -> None:
        fill = {
            "manual_crop_valid": "invalid",
            "manual_back_facing": "uncertain",
            "manual_number_visible": "",
            "manual_number_readable": "",
            "manual_digit_count": "",
            "manual_jersey_number": "7",
            "manual_contamination_affects_number_region": "",
            "manual_notes": "",
            "reviewer": REVIEWER,
            "reviewed_at": REVIEWED_AT,
        }
        codes = self._invalid(fill)
        self.assertIn("invalid_crop_field_filled", codes)

    def test_uncertain_visible_with_readable_yes(self) -> None:
        fill = _valid_readable_one()
        fill["manual_number_visible"] = "uncertain"
        codes = self._invalid(fill)
        self.assertTrue(
            {"visible_uncertain_readable_yes", "readable_yes_requires_visible_yes"}
            & set(codes)
        )


class ReportTests(unittest.TestCase):
    def test_deterministic_counts_and_valid_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fills = [_valid_readable_one(), _valid_visible_no(), {}]
            fx.write_annotations(fills)
            result = fx.validate()
            report = result["report"]
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["counts"]["expected_row_count"], 3)
            self.assertEqual(report["counts"]["actual_row_count"], 3)
            self.assertEqual(report["counts"]["reviewed_row_count"], 2)
            self.assertEqual(report["counts"]["unreviewed_row_count"], 1)
            self.assertEqual(report["counts"]["readable_number_count"], 1)
            self.assertEqual(report["counts"]["distinct_reviewer_count"], 1)
            self.assertEqual(report["validation"]["error_count"], 0)
            for key, value in report["safety"].items():
                self.assertIs(value, False, key)
            raw = fx.report_path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            reloaded = json.loads(raw.decode("utf-8"))
            self.assertEqual(reloaded["source_provenance"]["annotation_csv"]["sha256"], _sha256(fx.annotations_path))
            self.assertEqual(
                reloaded["source_provenance"]["review_items"]["sha256"],
                _sha256(fx.review_dir / REVIEW_ITEMS_NAME),
            )

    def test_invalid_status_and_error_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = _valid_readable_one()
            fill["manual_jersey_number"] = ""
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            self.assertGreater(result["error_count"], 0)
            errors = result["report"]["validation"]["errors"]
            self.assertEqual(
                errors,
                sorted(
                    errors,
                    key=lambda row: (
                        row.get("review_index") is None,
                        row.get("review_index") or 0,
                        row.get("code") or "",
                        row.get("field") or "",
                        row.get("message") or "",
                    ),
                ),
            )

    def test_validate_without_writing_still_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            report = validate_jersey_review_annotations(
                review_dir=fx.review_dir,
                annotations_csv=fx.annotations_path,
                config=fx.config_path,
            )
            self.assertEqual(report["status"], "valid")
            self.assertFalse(fx.report_path.exists())


class AtomicOutputTests(unittest.TestCase):
    def test_collision_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fx.validate()
            with self.assertRaises(JerseyAnnotationError):
                fx.validate(overwrite=False)

    def test_overwrite_and_deterministic_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            first = fx.validate()
            first_hash = _sha256(fx.report_path)
            second = fx.validate(overwrite=True)
            self.assertEqual(first["status"], second["status"])
            self.assertEqual(first["reviewed_row_count"], second["reviewed_row_count"])
            # elapsed_sec may differ; compare stable payload slices
            a = json.loads(fx.report_path.read_text(encoding="utf-8"))
            del a["elapsed_sec"]
            fx.validate(overwrite=True)
            b = json.loads(fx.report_path.read_text(encoding="utf-8"))
            del b["elapsed_sec"]
            self.assertEqual(a, b)
            self.assertNotEqual(first_hash, "")  # report written

    def test_validation_failure_still_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            fill = _valid_readable_one()
            fill["manual_jersey_number"] = "abc"
            fx.write_annotations([fill, {}, {}])
            result = fx.validate()
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(fx.report_path.is_file())
            payload = json.loads(fx.report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "invalid")

    def test_source_artifacts_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            hashes = {
                "items": _sha256(fx.review_dir / REVIEW_ITEMS_NAME),
                "summary": _sha256(fx.review_dir / SUMMARY_NAME),
                "template": _sha256(fx.template_path),
                "annotations": _sha256(fx.annotations_path),
            }
            panel = fx.review_dir / "panels" / "marker.png"
            panel.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            hashes["panel"] = _sha256(panel)
            fx.write_annotations([_valid_visible_no(), {}, {}])
            hashes["annotations"] = _sha256(fx.annotations_path)
            fx.validate()
            self.assertEqual(_sha256(fx.review_dir / REVIEW_ITEMS_NAME), hashes["items"])
            self.assertEqual(_sha256(fx.review_dir / SUMMARY_NAME), hashes["summary"])
            self.assertEqual(_sha256(fx.template_path), hashes["template"])
            self.assertEqual(_sha256(fx.annotations_path), hashes["annotations"])
            self.assertEqual(_sha256(panel), hashes["panel"])
            # temp cleanup
            temps = list(fx.report_path.parent.glob("_tmp_reid_jersey_annotation_report_*"))
            self.assertEqual(temps, [])


class CliAndSafetyTests(unittest.TestCase):
    def test_cli_help_options(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        for option in (
            "--review-dir",
            "--annotations-csv",
            "--config",
            "--report-path",
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
            "--manual-label",
            "--gallery",
        ):
            self.assertNotIn(option, result.stdout)

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fx = AnnotationFixture(Path(tmp))
            ok = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--review-dir",
                    str(fx.review_dir),
                    "--annotations-csv",
                    str(fx.annotations_path),
                    "--config",
                    str(fx.config_path),
                    "--report-path",
                    str(fx.report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(ok.returncode, 0)
            fill = _valid_readable_one()
            fill["manual_jersey_number"] = ""
            fx.write_annotations([fill, {}, {}])
            bad = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--review-dir",
                    str(fx.review_dir),
                    "--annotations-csv",
                    str(fx.annotations_path),
                    "--config",
                    str(fx.config_path),
                    "--report-path",
                    str(Path(tmp) / "bad_report.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad.returncode, 1)

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
            "cv2.imwrite",
            "gallery_updated = True",
            "identity_assigned = True",
            "team_assigned = True",
        ):
            self.assertNotIn(forbidden, text, forbidden)
        self.assertIn('"OCR_performed": False', text)
        doc = DOC.read_text(encoding="utf-8")
        self.assertIn("Blank row vs `uncertain`", doc)
        self.assertIn("Pilot review plan", doc)
        self.assertIn("identity ground truth", doc.lower())

    def test_decision_fields_cover_manual_columns(self) -> None:
        self.assertTrue(set(DECISION_FIELDS).issubset(set(MANUAL_FIELDS)))
        self.assertEqual(
            list(PROVENANCE_COLUMNS),
            list(ANNOTATION_COLUMNS[: len(PROVENANCE_COLUMNS)]),
        )


if __name__ == "__main__":
    unittest.main()
