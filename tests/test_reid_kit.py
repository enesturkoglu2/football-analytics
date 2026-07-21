"""Unit tests for Stage 5B1A ReID kit descriptors (no real sample.mp4)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.kit import (  # noqa: E402
    CROP_KIT_NAME,
    KIT_SUMMARY_NAME,
    TRACK_KIT_NAME,
    KitError,
    analyze_one_crop_kit,
    compute_torso_kit_metrics,
    compute_torso_roi,
    load_crop_manifest_for_kit,
    load_kit_descriptor_config,
    load_quality_signals_for_kit,
    run_analyze_reid_kit_descriptors,
    validate_kit_descriptor_config,
)
from football_analytics.reid.schema import build_crop_manifest_row  # noqa: E402

DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "reid" / "kit_descriptor_stage5b.yaml"


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to write jpeg {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bgr_solid(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def _quality_row_from_manifest(row: dict, *, laplacian: float = 10.0) -> dict:
    return {
        "crop_id": row["crop_id"],
        "track_id": row["track_id"],
        "frame_index": row["frame_index"],
        "selection_rank": row["selection_rank"],
        "crop_relative_path": row["crop_relative_path"],
        "laplacian_variance": laplacian,
        "union_other_person_crop_coverage": 0.0,
        "frame_edge_contact_count": 0,
        "quality_decision": "measurement_only",
        "automatic_exclusion_applied": False,
        "quality_threshold": None,
        "contamination_threshold": None,
        "schema_version": "reid_crop_quality_signal_v1",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _make_kit_fixture(
    root: Path,
    *,
    boxes: list[tuple[int, int, list[int]]],
    images: dict[str, np.ndarray] | None = None,
    border_color: tuple[int, int, int] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Create crops + manifest + quality + config copy.

    boxes: (track_id, frame_index, [l,t,r,b])
    """
    crops_root = root / "crops_out"
    crops_dir = crops_root / "crops"
    crops_dir.mkdir(parents=True)
    manifest_rows: list[dict] = []
    for track_id, frame_index, bbox in boxes:
        left, top, right, bottom = bbox
        w, h = right - left, bottom - top
        existing = sum(1 for r in manifest_rows if r["track_id"] == track_id)
        rank = existing + 1
        key = f"{track_id}_{frame_index}_{rank}"
        if images and key in images:
            img = images[key]
        else:
            img = np.full((h, w, 3), 120, dtype=np.uint8)
            if border_color is not None:
                img[:, :] = border_color
                # fill torso-ish center with gray so border color differs
                x0 = int(math.floor(w * 0.20))
                x1 = int(math.ceil(w * 0.80))
                y0 = int(math.floor(h * 0.15))
                y1 = int(math.ceil(h * 0.65))
                img[y0:y1, x0:x1] = (128, 128, 128)
        assert img.shape[0] == h and img.shape[1] == w
        row = build_crop_manifest_row(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_sec=float(frame_index) / 30.0,
            source_video="synthetic.mp4",
            bbox_xyxy=bbox,
            detection_confidence=0.9,
            quality_score=float(w * h) * 0.9,
            selection_rank=rank,
        )
        _write_jpeg(crops_root / row["crop_relative_path"], img)
        manifest_rows.append(row)

    manifest_path = crops_root / "crop_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)

    quality_rows = [
        _quality_row_from_manifest(row, laplacian=float(10 + i))
        for i, row in enumerate(manifest_rows)
    ]
    quality_path = root / "crop_quality_signals.jsonl"
    _write_jsonl(quality_path, quality_rows)

    config_path = root / "kit_config.yaml"
    config_path.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    return crops_root, manifest_path, quality_path, config_path


def _load_default_payload() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


class ConfigValidationTests(unittest.TestCase):
    def test_valid_default_config(self) -> None:
        cfg = load_kit_descriptor_config(DEFAULT_CONFIG)
        self.assertEqual(cfg["stage_status"], "measurement_baseline")
        self.assertFalse(cfg["automatic_team_assignment_enabled"])
        self.assertIsNone(cfg["kit_similarity_threshold"])
        self.assertEqual(cfg["source_region"]["name"], "normalized_center_torso")

    def test_missing_required_and_invalid_bounds(self) -> None:
        payload = _load_default_payload()
        del payload["source_region"]
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["source_region"]["x_min_fraction"] = 0.9
        payload["source_region"]["x_max_fraction"] = 0.2
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["source_region"]["y_min_fraction"] = -0.1
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["source_region"]["y_max_fraction"] = 1.5
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

    def test_invalid_histograms_and_flags(self) -> None:
        payload = _load_default_payload()
        payload["histograms"]["hue_bin_count"] = 0
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["automatic_team_assignment_enabled"] = True
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["forced_two_team_clustering_enabled"] = True
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["kit_similarity_threshold"] = 0.5
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

    def test_hue_family_overlap_gap_and_duplicate(self) -> None:
        payload = _load_default_payload()
        payload["coarse_color_families"]["chromatic_hue_ranges"]["orange"] = [[0, 24]]
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        payload["coarse_color_families"]["chromatic_hue_ranges"]["magenta"] = [[151, 168]]
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)

        payload = _load_default_payload()
        order = list(payload["coarse_color_families"]["family_order"])
        order[0] = "gray"
        payload["coarse_color_families"]["family_order"] = order
        with self.assertRaises(KitError):
            validate_kit_descriptor_config(payload)


class ManifestQualityPathTests(unittest.TestCase):
    def test_valid_join_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops_root, manifest, quality, config = _make_kit_fixture(
                root,
                boxes=[(1, 0, [10, 10, 50, 70]), (2, 1, [20, 20, 60, 80])],
            )
            rows = load_crop_manifest_for_kit(manifest)
            qrows = load_quality_signals_for_kit(quality, manifest_rows=rows)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(qrows), 2)

            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(KitError):
                load_crop_manifest_for_kit(empty)

            bad = root / "bad.jsonl"
            bad.write_text("{not-json\n", encoding="utf-8")
            with self.assertRaises(KitError):
                load_crop_manifest_for_kit(bad)

            # duplicate crop_id
            dup_rows = copy.deepcopy(rows)
            dup_rows[1]["crop_id"] = dup_rows[0]["crop_id"]
            dup_path = root / "dup_manifest.jsonl"
            _write_jsonl(dup_path, dup_rows)
            with self.assertRaises(KitError):
                load_crop_manifest_for_kit(dup_path)

            # absolute path
            abs_rows = copy.deepcopy(rows)
            abs_rows[0]["crop_relative_path"] = "/tmp/abs.jpg"
            abs_path = root / "abs_manifest.jsonl"
            _write_jsonl(abs_path, abs_rows)
            with self.assertRaises(KitError):
                # schema may pass path string; join/analyze rejects on resolve
                run_analyze_reid_kit_descriptors(
                    crop_manifest=abs_path,
                    quality_signals=quality,
                    config=config,
                    output_dir=root / "out_abs",
                )

            # traversal
            trav_rows = copy.deepcopy(rows)
            trav_rows[0]["crop_relative_path"] = "../escape.jpg"
            trav_path = root / "trav_manifest.jsonl"
            # rebuild quality to match for order test later; here just load path resolve
            _write_jsonl(trav_path, trav_rows)
            with self.assertRaises(KitError):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=trav_path,
                    quality_signals=quality,
                    config=config,
                    output_dir=root / "out_trav",
                )

    def test_quality_mismatch_and_policy_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, quality, config = _make_kit_fixture(
                root, boxes=[(1, 0, [10, 10, 50, 70]), (2, 1, [12, 12, 52, 72])]
            )
            rows = load_crop_manifest_for_kit(manifest)
            qrows = [
                _quality_row_from_manifest(r) for r in rows
            ]
            # order mismatch
            qrows = list(reversed(qrows))
            qpath = root / "q_order.jsonl"
            _write_jsonl(qpath, qrows)
            with self.assertRaises(KitError):
                load_quality_signals_for_kit(qpath, manifest_rows=rows)

            # count mismatch
            qpath2 = root / "q_count.jsonl"
            _write_jsonl(qpath2, qrows[:1])
            with self.assertRaises(KitError):
                load_quality_signals_for_kit(qpath2, manifest_rows=rows)

            # exclusion true
            bad = _quality_row_from_manifest(rows[0])
            bad["automatic_exclusion_applied"] = True
            qpath3 = root / "q_excl.jsonl"
            _write_jsonl(qpath3, [bad, _quality_row_from_manifest(rows[1])])
            with self.assertRaises(KitError):
                load_quality_signals_for_kit(qpath3, manifest_rows=rows)

            # threshold non-null
            bad2 = _quality_row_from_manifest(rows[0])
            bad2["quality_threshold"] = 0.1
            qpath4 = root / "q_thr.jsonl"
            _write_jsonl(qpath4, [bad2, _quality_row_from_manifest(rows[1])])
            with self.assertRaises(KitError):
                load_quality_signals_for_kit(qpath4, manifest_rows=rows)

            # NaN reject
            nan_line = (
                json.dumps(_quality_row_from_manifest(rows[0])).replace("10.0", "NaN")
            )
            # force NaN token
            payload = _quality_row_from_manifest(rows[0])
            text = json.dumps(payload).replace('"laplacian_variance": 10.0', '"laplacian_variance": NaN')
            qpath5 = root / "q_nan.jsonl"
            qpath5.write_text(
                text + "\n" + json.dumps(_quality_row_from_manifest(rows[1])) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KitError):
                load_quality_signals_for_kit(qpath5, manifest_rows=rows)

    def test_missing_corrupt_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops_root, manifest, quality, config = _make_kit_fixture(
                root, boxes=[(1, 0, [10, 10, 50, 70])]
            )
            rows = load_crop_manifest_for_kit(manifest)
            jpeg = crops_root / rows[0]["crop_relative_path"]
            jpeg.unlink()
            with self.assertRaises(KitError):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=manifest,
                    quality_signals=quality,
                    config=config,
                    output_dir=root / "out_missing",
                )
            self.assertFalse((root / "out_missing").exists())

            # recreate fixture for corrupt
            crops_root, manifest, quality, config = _make_kit_fixture(
                root / "b", boxes=[(1, 0, [10, 10, 50, 70])]
            )
            rows = load_crop_manifest_for_kit(manifest)
            jpeg = crops_root / rows[0]["crop_relative_path"]
            jpeg.write_bytes(b"not-a-jpeg")
            with self.assertRaises(KitError):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=manifest,
                    quality_signals=quality,
                    config=config,
                    output_dir=root / "out_corrupt",
                )

            crops_root, manifest, quality, config = _make_kit_fixture(
                root / "c", boxes=[(1, 0, [10, 10, 50, 70])]
            )
            rows = load_crop_manifest_for_kit(manifest)
            jpeg = crops_root / rows[0]["crop_relative_path"]
            _write_jpeg(jpeg, np.full((20, 20, 3), 50, dtype=np.uint8))
            with self.assertRaises(KitError):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=manifest,
                    quality_signals=quality,
                    config=config,
                    output_dir=root / "out_size",
                )


class TorsoRoiTests(unittest.TestCase):
    def test_floor_ceil_clamp_and_border_exclusion(self) -> None:
        cfg = load_kit_descriptor_config(DEFAULT_CONFIG)
        roi = compute_torso_roi(width=100, height=200, config=cfg)
        self.assertEqual(roi["torso_x0"], 20)
        self.assertEqual(roi["torso_x1"], 80)
        self.assertEqual(roi["torso_y0"], 30)
        self.assertEqual(roi["torso_y1"], 130)
        self.assertTrue(roi["torso_region_valid"])

        tiny = compute_torso_roi(width=1, height=1, config=cfg)
        self.assertGreaterEqual(tiny["torso_width"], 1)
        self.assertGreaterEqual(tiny["torso_height"], 1)

        # border color must not dominate when torso is filled differently
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w, h = 60, 100
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = (0, 0, 255)  # red border (BGR)
            x0 = int(math.floor(w * 0.20))
            x1 = int(math.ceil(w * 0.80))
            y0 = int(math.floor(h * 0.15))
            y1 = int(math.ceil(h * 0.65))
            img[y0:y1, x0:x1] = (255, 0, 0)  # blue torso
            before = img.copy()
            metrics = compute_torso_kit_metrics(img, config=cfg)
            self.assertEqual(metrics["dominant_color_family"], "blue")
            self.assertTrue(np.array_equal(img, before))


class ColorDescriptorTests(unittest.TestCase):
    def _assert_pure(self, bgr: tuple[int, int, int], family: str) -> None:
        cfg = load_kit_descriptor_config(DEFAULT_CONFIG)
        img = _bgr_solid(80, 60, bgr)
        metrics = compute_torso_kit_metrics(img, config=cfg)
        self.assertEqual(metrics["dominant_color_family"], family)
        self.assertGreater(metrics["dominant_color_family_fraction"], 0.9)
        self.assertAlmostEqual(
            sum(metrics["color_family_fractions"].values()), 1.0, places=6
        )
        self.assertEqual(len(metrics["hue_histogram_chromatic"]), 18)
        self.assertEqual(len(metrics["saturation_histogram"]), 8)
        self.assertEqual(len(metrics["value_histogram"]), 8)
        self.assertAlmostEqual(sum(metrics["saturation_histogram"]), 1.0, places=6)
        self.assertAlmostEqual(sum(metrics["value_histogram"]), 1.0, places=6)
        for value in metrics["hue_histogram_chromatic"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            self.assertTrue(math.isfinite(value))

    def test_pure_and_mixed_colors(self) -> None:
        self._assert_pure((0, 0, 0), "black")
        self._assert_pure((255, 255, 255), "white")
        self._assert_pure((128, 128, 128), "gray")
        self._assert_pure((0, 0, 255), "red")
        self._assert_pure((0, 255, 255), "yellow")
        self._assert_pure((0, 255, 0), "green")
        self._assert_pure((255, 0, 0), "blue")

        cfg = load_kit_descriptor_config(DEFAULT_CONFIG)
        img = _bgr_solid(80, 60, (0, 0, 0))
        # make achromatic → hue hist all zero
        metrics = compute_torso_kit_metrics(img, config=cfg)
        self.assertEqual(metrics["hue_histogram_chromatic"], [0.0] * 18)
        self.assertEqual(metrics["chromatic_pixel_count"], 0)

        # half white / half blue in torso region by painting full image halves
        mixed = np.zeros((100, 80, 3), dtype=np.uint8)
        mixed[:, :40] = (255, 255, 255)
        mixed[:, 40:] = (255, 0, 0)
        m = compute_torso_kit_metrics(mixed, config=cfg)
        self.assertAlmostEqual(sum(m["color_family_fractions"].values()), 1.0, places=6)
        self.assertIn(m["dominant_color_family"], {"white", "blue"})
        self.assertEqual(m["top_color_families"][0]["family"], m["dominant_color_family"])

    def test_red_circular_and_chromatic_mask(self) -> None:
        cfg = load_kit_descriptor_config(DEFAULT_CONFIG)
        # OpenCV red near hue 0
        red = _bgr_solid(80, 60, (0, 0, 255))
        m = compute_torso_kit_metrics(red, config=cfg)
        self.assertEqual(m["dominant_color_family"], "red")
        # first hue bin should dominate chromatic hist
        self.assertEqual(m["hue_histogram_chromatic"].index(max(m["hue_histogram_chromatic"])), 0)

        # low saturation gray-reddish should be achromatic
        grayish = _bgr_solid(80, 60, (40, 40, 45))
        m2 = compute_torso_kit_metrics(grayish, config=cfg)
        self.assertIn(m2["dominant_color_family"], {"black", "gray", "white"})


class AggregationAndOutputTests(unittest.TestCase):
    def test_pipeline_aggregation_output_overwrite_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = {
                "1_0_1": _bgr_solid(60, 40, (0, 0, 255)),  # red
                "1_1_2": _bgr_solid(60, 40, (255, 0, 0)),  # blue
                "2_2_1": _bgr_solid(60, 40, (0, 0, 0)),  # black / zero hue
            }
            crops_root, manifest, quality, config = _make_kit_fixture(
                root,
                boxes=[
                    (1, 0, [0, 0, 40, 60]),
                    (1, 1, [0, 0, 40, 60]),
                    (2, 2, [0, 0, 40, 60]),
                ],
                images=images,
            )
            jpeg_paths = sorted(crops_root.rglob("*.jpg"))
            hashes_before = {p: _sha256(p) for p in jpeg_paths}
            quality_before = quality.read_bytes()

            out = root / "kit_out"
            result1 = run_analyze_reid_kit_descriptors(
                crop_manifest=manifest,
                quality_signals=quality,
                config=config,
                output_dir=out,
            )
            self.assertEqual(result1["crop_count"], 3)
            self.assertEqual(result1["track_count"], 2)
            self.assertFalse(result1["automatic_team_assignment_performed"])
            self.assertIsNone(result1["kit_similarity_threshold"])

            crop_lines = (out / CROP_KIT_NAME).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(crop_lines), 3)
            self.assertTrue((out / CROP_KIT_NAME).read_text(encoding="utf-8").endswith("\n"))
            crop_rows = [json.loads(line) for line in crop_lines]
            self.assertEqual(
                [r["crop_id"] for r in crop_rows],
                [json.loads(l)["crop_id"] for l in manifest.read_text().splitlines() if l.strip()],
            )
            for row in crop_rows:
                self.assertIsNone(row["team_assignment"])
                self.assertFalse(row["automatic_link_applied"])
                self.assertFalse(row["quality_weight_applied"])
                self.assertFalse(row["quality_exclusion_applied"])

            track_rows = [
                json.loads(line)
                for line in (out / TRACK_KIT_NAME).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([r["track_id"] for r in track_rows], [1, 2])
            track1 = track_rows[0]
            self.assertEqual(track1["crop_ids"], [crop_rows[0]["crop_id"], crop_rows[1]["crop_id"]])
            self.assertEqual(track1["aggregation_method"], "equal_weight_mean")
            self.assertEqual(track1["excluded_crop_count"], 0)
            self.assertFalse(track1["forced_two_team_clustering_applied"])
            # equal-weight mean of red+blue family fractions
            mean_red = (
                crop_rows[0]["color_family_fractions"]["red"]
                + crop_rows[1]["color_family_fractions"]["red"]
            ) / 2.0
            self.assertAlmostEqual(
                track1["mean_color_family_fractions"]["red"], mean_red, places=6
            )
            # achromatic zero hue included in mean and not renormalized
            track2 = track_rows[1]
            self.assertEqual(track2["mean_hue_histogram_chromatic"], [0.0] * 18)

            summary = json.loads((out / KIT_SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["crop_count"], 3)
            self.assertFalse(summary["accuracy_claimed"])
            self.assertFalse(summary["resize_performed"])
            names = sorted(p.name for p in out.iterdir() if p.is_file())
            self.assertEqual(names, sorted([CROP_KIT_NAME, TRACK_KIT_NAME, KIT_SUMMARY_NAME]))

            # collision without overwrite
            with self.assertRaises(Exception):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=manifest,
                    quality_signals=quality,
                    config=config,
                    output_dir=out,
                    overwrite=False,
                )

            result2 = run_analyze_reid_kit_descriptors(
                crop_manifest=manifest,
                quality_signals=quality,
                config=config,
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(result2["crop_count"], 3)
            leftovers = list(out.parent.glob(f"_tmp_reid_kit_{out.name}_*")) + list(
                out.parent.glob(f"_backup_reid_kit_{out.name}_*")
            )
            self.assertEqual(leftovers, [])

            for path, digest in hashes_before.items():
                self.assertEqual(_sha256(path), digest)
            self.assertEqual(quality.read_bytes(), quality_before)

            # deterministic repeat
            out2 = root / "kit_out2"
            run_analyze_reid_kit_descriptors(
                crop_manifest=manifest,
                quality_signals=quality,
                config=config,
                output_dir=out2,
            )
            self.assertEqual(
                (out / CROP_KIT_NAME).read_text(encoding="utf-8"),
                (out2 / CROP_KIT_NAME).read_text(encoding="utf-8"),
            )

    def test_validation_failure_leaves_no_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops_root, manifest, quality, config = _make_kit_fixture(
                root, boxes=[(1, 0, [0, 0, 40, 60])]
            )
            # stale quality path mismatch
            rows = load_crop_manifest_for_kit(manifest)
            bad_q = _quality_row_from_manifest(rows[0])
            bad_q["crop_relative_path"] = "crops/track_1/does_not_match.jpg"
            qpath = root / "bad_q.jsonl"
            _write_jsonl(qpath, [bad_q])
            out = root / "no_final"
            with self.assertRaises(KitError):
                run_analyze_reid_kit_descriptors(
                    crop_manifest=manifest,
                    quality_signals=qpath,
                    config=config,
                    output_dir=out,
                )
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob("_tmp_reid_kit_*")), [])

    def test_no_forbidden_imports_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, quality, config = _make_kit_fixture(
                root,
                boxes=[(1, 0, [0, 0, 40, 60])],
                images={"1_0_1": _bgr_solid(60, 40, (0, 255, 0))},
            )
            out = root / "kit_safe"
            with mock.patch.dict(sys.modules, {"sklearn": None, "sklearn.cluster": None}):
                with mock.patch("cv2.kmeans", side_effect=AssertionError("kmeans called")):
                    run_analyze_reid_kit_descriptors(
                        crop_manifest=manifest,
                        quality_signals=quality,
                        config=config,
                        output_dir=out,
                    )


class HelpTests(unittest.TestCase):
    def test_cli_help_no_inputs(self) -> None:
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "analyze_reid_kit_descriptors.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--crop-manifest", proc.stdout)
        self.assertIn("--quality-signals", proc.stdout)
        self.assertIn("--config", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)
        self.assertIn("--overwrite", proc.stdout)
        self.assertNotIn("--video", proc.stdout)
        self.assertNotIn("--tracks", proc.stdout)


if __name__ == "__main__":
    unittest.main()
