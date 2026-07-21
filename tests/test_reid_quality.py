"""Unit tests for Stage 5A2A ReID crop quality analysis (no real sample.mp4)."""

from __future__ import annotations

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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.quality import (  # noqa: E402
    CROP_QUALITY_NAME,
    QUALITY_SUMMARY_NAME,
    TRACK_QUALITY_NAME,
    QualityError,
    compute_edge_contacts,
    compute_image_metrics,
    compute_tracking_bbox_contamination,
    infer_frame_size,
    load_crop_manifest_for_quality,
    load_person_track_observations,
    run_analyze_reid_crop_quality,
)
from football_analytics.reid.schema import build_crop_manifest_row  # noqa: E402


FRAME_W = 100
FRAME_H = 80


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to write jpeg {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _obs(
    *,
    track_id: int,
    frame_index: int,
    bbox: list[float],
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) / 30.0,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


def _make_fixture(
    root: Path,
    *,
    boxes: list[tuple[int, int, list[int]]],
    extra_tracks: list[dict] | None = None,
    images: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path, Path]:
    """Create crop dir + manifest + tracks.

    boxes: list of (track_id, frame_index, [l,t,r,b] int bbox)
    """
    crops_root = root / "crops_out"
    crops_dir = crops_root / "crops"
    crops_dir.mkdir(parents=True)
    manifest_rows = []
    track_rows = list(extra_tracks or [])
    for rank_offset, (track_id, frame_index, bbox) in enumerate(boxes, start=1):
        left, top, right, bottom = bbox
        w, h = right - left, bottom - top
        key = f"{track_id}_{frame_index}"
        if images and key in images:
            img = images[key]
        else:
            img = np.full((h, w, 3), 120, dtype=np.uint8)
        assert img.shape[0] == h and img.shape[1] == w
        row = build_crop_manifest_row(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_sec=float(frame_index) / 30.0,
            source_video="synthetic.mp4",
            bbox_xyxy=bbox,
            detection_confidence=0.9,
            quality_score=float(w * h) * 0.9,
            selection_rank=1,
        )
        # unique selection ranks when same track appears twice
        if any(r["track_id"] == track_id for r in manifest_rows):
            existing = sum(1 for r in manifest_rows if r["track_id"] == track_id)
            row = build_crop_manifest_row(
                track_id=track_id,
                frame_index=frame_index,
                timestamp_sec=float(frame_index) / 30.0,
                source_video="synthetic.mp4",
                bbox_xyxy=bbox,
                detection_confidence=0.9,
                quality_score=float(w * h) * 0.9,
                selection_rank=existing + 1,
            )
        rel = row["crop_relative_path"]
        _write_jpeg(crops_root / rel, img)
        manifest_rows.append(row)
        track_rows.append(
            _obs(
                track_id=track_id,
                frame_index=frame_index,
                bbox=[float(v) for v in bbox],
            )
        )

    manifest_path = crops_root / "crop_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")

    tracks_path = root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for row in track_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
    return crops_root, manifest_path, tracks_path


class ManifestPathTests(unittest.TestCase):
    def test_valid_and_empty_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, _ = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 30, 40])]
            )
            rows = load_crop_manifest_for_quality(manifest)
            self.assertEqual(len(rows), 1)

            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(QualityError):
                load_crop_manifest_for_quality(empty)

            bad = root / "bad.jsonl"
            bad.write_text("{not-json\n", encoding="utf-8")
            with self.assertRaises(QualityError):
                load_crop_manifest_for_quality(bad)

            text = manifest.read_text(encoding="utf-8")
            dup_id = root / "dup_id.jsonl"
            row = json.loads(text.splitlines()[0])
            row2 = dict(row)
            row2["crop_relative_path"] = "crops/track_1/other.jpg"
            dup_id.write_text(
                json.dumps(row) + "\n" + json.dumps(row2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(QualityError):
                load_crop_manifest_for_quality(dup_id)

            dup_rel = root / "dup_rel.jsonl"
            row3 = dict(row)
            row3["crop_id"] = "other_id"
            dup_rel.write_text(
                json.dumps(row) + "\n" + json.dumps(row3) + "\n", encoding="utf-8"
            )
            with self.assertRaises(QualityError):
                load_crop_manifest_for_quality(dup_rel)

    def test_path_and_jpeg_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops_root, manifest, tracks = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 30, 40])]
            )
            out = root / "out"

            # absolute path
            rows = [json.loads(manifest.read_text().splitlines()[0])]
            rows[0]["crop_relative_path"] = str((crops_root / rows[0]["crop_relative_path"]).resolve())
            abs_manifest = crops_root / "abs.jsonl"
            abs_manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=abs_manifest, tracks=tracks, output_dir=out
                )

            # traversal
            trav = dict(json.loads(manifest.read_text().splitlines()[0]))
            trav["crop_relative_path"] = "../secret.jpg"
            trav_manifest = crops_root / "trav.jsonl"
            trav_manifest.write_text(json.dumps(trav) + "\n", encoding="utf-8")
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=trav_manifest, tracks=tracks, output_dir=out
                )

            # missing jpeg
            miss = dict(json.loads(manifest.read_text().splitlines()[0]))
            miss["crop_relative_path"] = "crops/track_1/missing.jpg"
            miss_manifest = crops_root / "miss.jsonl"
            miss_manifest.write_text(json.dumps(miss) + "\n", encoding="utf-8")
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=miss_manifest, tracks=tracks, output_dir=out
                )

            # corrupt jpeg
            bad_path = crops_root / "crops" / "track_1" / "crop_0_1.jpg"
            bad_path.write_bytes(b"not-a-jpeg")
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest, tracks=tracks, output_dir=out
                )

    def test_decoded_size_mismatch_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops_root, manifest, tracks = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 30, 40])]
            )
            # rewrite jpeg with wrong size
            wrong = np.zeros((10, 10, 3), dtype=np.uint8)
            _write_jpeg(crops_root / "crops/track_1/crop_0_1.jpg", wrong)
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest,
                    tracks=tracks,
                    output_dir=root / "out",
                )

            nan_manifest = crops_root / "nan.jsonl"
            row = json.loads(manifest.read_text().splitlines()[0])
            row["detection_confidence"] = float("nan")
            nan_manifest.write_text(
                json.dumps(row).replace("NaN", "NaN") + "\n", encoding="utf-8"
            )
            # json.dumps converts nan to NaN token; parse_constant should reject
            with self.assertRaises(QualityError):
                load_crop_manifest_for_quality(nan_manifest)


class TrackingTests(unittest.TestCase):
    def test_target_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, tracks = _make_fixture(
                root,
                boxes=[(1, 0, [10, 10, 30, 40])],
                extra_tracks=[
                    _obs(track_id=2, frame_index=0, bbox=[50, 10, 70, 40]),
                    _obs(
                        track_id=9,
                        frame_index=0,
                        bbox=[1, 1, 5, 5],
                        class_id=2,
                        class_name="ball",
                    ),
                ],
            )
            obs = load_person_track_observations(tracks)
            self.assertEqual(len(obs), 2)  # ball filtered

            # missing target
            tracks2 = root / "tracks_missing.jsonl"
            tracks2.write_text(
                json.dumps(_obs(track_id=99, frame_index=0, bbox=[10, 10, 30, 40]))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest,
                    tracks=tracks2,
                    output_dir=root / "out_missing",
                )

            # duplicate target
            tracks3 = root / "tracks_dup.jsonl"
            row = _obs(track_id=1, frame_index=0, bbox=[10, 10, 30, 40])
            tracks3.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaises(QualityError):
                load_person_track_observations(tracks3)

            # mismatched converted bbox
            tracks4 = root / "tracks_mismatch.jsonl"
            tracks4.write_text(
                json.dumps(_obs(track_id=1, frame_index=0, bbox=[11, 10, 30, 40]))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest,
                    tracks=tracks4,
                    output_dir=root / "out_mismatch",
                )

            # invalid bbox / ids
            with self.assertRaises(QualityError):
                load_person_track_observations_from_rows(
                    [_obs(track_id=1, frame_index=0, bbox=[10, 10, 10, 40])]
                )


def load_person_track_observations_from_rows(rows: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, allow_nan=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        return load_person_track_observations(path)


class ImageMetricTests(unittest.TestCase):
    def test_uniform_vs_checker_and_extremes(self) -> None:
        uniform = np.full((40, 20, 3), 128, dtype=np.uint8)
        u = compute_image_metrics(uniform)
        self.assertAlmostEqual(u["grayscale_std"], 0.0, places=5)
        self.assertAlmostEqual(u["laplacian_variance"], 0.0, places=5)
        self.assertTrue(math.isfinite(u["grayscale_entropy_bits"]))

        checker = np.zeros((40, 20, 3), dtype=np.uint8)
        checker[0::2, 0::2] = 255
        checker[1::2, 1::2] = 255
        c = compute_image_metrics(checker)
        self.assertGreater(c["laplacian_variance"], u["laplacian_variance"])
        self.assertTrue(0.0 <= c["dark_pixel_ratio"] <= 1.0)
        self.assertTrue(0.0 <= c["bright_pixel_ratio"] <= 1.0)

        dark = np.zeros((16, 16, 3), dtype=np.uint8)
        d = compute_image_metrics(dark)
        self.assertAlmostEqual(d["dark_pixel_ratio"], 1.0, places=5)

        bright = np.full((16, 16, 3), 255, dtype=np.uint8)
        b = compute_image_metrics(bright)
        self.assertAlmostEqual(b["bright_pixel_ratio"], 1.0, places=5)

        # entropy deterministic
        e1 = compute_image_metrics(checker)["grayscale_entropy_bits"]
        e2 = compute_image_metrics(checker)["grayscale_entropy_bits"]
        self.assertEqual(e1, e2)


class EdgeContactTests(unittest.TestCase):
    def test_contacts(self) -> None:
        none = compute_edge_contacts([10, 10, 30, 40], frame_width=100, frame_height=80)
        self.assertEqual(none["frame_edge_contact_count"], 0)
        left_top = compute_edge_contacts([0, 0, 20, 20], frame_width=100, frame_height=80)
        self.assertTrue(left_top["touches_left_edge"])
        self.assertTrue(left_top["touches_top_edge"])
        self.assertEqual(left_top["frame_edge_contact_count"], 2)
        right_bottom = compute_edge_contacts(
            [80, 60, 100, 80], frame_width=100, frame_height=80
        )
        self.assertTrue(right_bottom["touches_right_edge"])
        self.assertTrue(right_bottom["touches_bottom_edge"])
        full = compute_edge_contacts([0, 0, 100, 80], frame_width=100, frame_height=80)
        self.assertEqual(full["frame_edge_contact_count"], 4)


class ContaminationTests(unittest.TestCase):
    def test_cases(self) -> None:
        target = [10, 10, 50, 50]
        # no others
        empty = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[_obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50])],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(empty["other_person_observation_count_in_frame"], 0)
        self.assertEqual(empty["other_person_overlap_count"], 0)
        self.assertEqual(empty["union_other_person_crop_coverage"], 0.0)

        # other present, no overlap
        no_ov = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[
                _obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50]),
                _obs(track_id=2, frame_index=0, bbox=[60, 10, 80, 40]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(no_ov["other_person_observation_count_in_frame"], 1)
        self.assertEqual(no_ov["other_person_overlap_count"], 0)

        # partial overlap with center clearly inside target
        partial = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[
                _obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50]),
                _obs(track_id=2, frame_index=0, bbox=[20, 20, 45, 45]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(partial["other_person_overlap_count"], 1)
        self.assertGreater(partial["max_other_person_crop_coverage"], 0.0)
        self.assertGreater(partial["max_other_person_iou"], 0.0)
        self.assertGreaterEqual(partial["other_person_center_inside_count"], 1)

        # two overlaps — union must not double-count
        two = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[
                _obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50]),
                _obs(track_id=2, frame_index=0, bbox=[10, 10, 30, 30]),
                _obs(track_id=3, frame_index=0, bbox=[20, 20, 40, 40]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertLessEqual(two["union_other_person_crop_coverage"], 1.0)
        self.assertGreater(two["union_other_person_crop_coverage"], 0.0)
        # union area of two overlapping regions is less than sum of intersections
        inter1 = 20 * 20
        inter2 = 20 * 20
        self.assertLess(
            two["union_other_person_crop_coverage"] * 1600,
            inter1 + inter2,
        )
        self.assertEqual(two["other_person_observation_count_in_frame"], 2)

        # full cover
        full = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[
                _obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50]),
                _obs(track_id=2, frame_index=0, bbox=[0, 0, 100, 80]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertAlmostEqual(full["max_other_person_crop_coverage"], 1.0, places=5)
        self.assertAlmostEqual(full["union_other_person_crop_coverage"], 1.0, places=5)

        # target excluded; other frame ignored by caller (only same-frame list passed)
        only_target = compute_tracking_bbox_contamination(
            target_bbox=target,
            target_track_id=1,
            frame_observations=[
                _obs(track_id=1, frame_index=0, bbox=[10, 10, 50, 50]),
            ],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        self.assertEqual(only_target["other_person_observation_count_in_frame"], 0)


class OutputPipelineTests(unittest.TestCase):
    def test_pipeline_outputs_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sharp = np.zeros((30, 20, 3), dtype=np.uint8)
            sharp[0::2, 0::2] = 255
            crops_root, manifest, tracks = _make_fixture(
                root,
                boxes=[
                    (1, 0, [10, 10, 30, 40]),
                    (2, 0, [25, 15, 45, 45]),
                    (1, 5, [0, 0, 20, 30]),
                ],
                images={
                    "1_0": np.full((30, 20, 3), 128, dtype=np.uint8),
                    "2_0": sharp,
                    "1_5": np.full((30, 20, 3), 10, dtype=np.uint8),
                },
            )
            # hashes before
            jpeg_paths = sorted(crops_root.rglob("*.jpg"))
            before = {p: (_sha256(p), p.stat().st_size, p.read_bytes()) for p in jpeg_paths}

            out = root / "quality_out"
            result = run_analyze_reid_crop_quality(
                crop_manifest=manifest, tracks=tracks, output_dir=out
            )
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["quality_threshold"])
            self.assertFalse(result["automatic_exclusion_performed"])

            files = sorted(p.name for p in out.iterdir() if p.is_file())
            self.assertEqual(
                files,
                [CROP_QUALITY_NAME, QUALITY_SUMMARY_NAME, TRACK_QUALITY_NAME],
            )

            crop_rows = [
                json.loads(line)
                for line in (out / CROP_QUALITY_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue((out / CROP_QUALITY_NAME).read_text().endswith("\n"))
            self.assertEqual(len(crop_rows), 3)
            manifest_ids = [
                json.loads(line)["crop_id"]
                for line in manifest.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual([r["crop_id"] for r in crop_rows], manifest_ids)
            for row in crop_rows:
                self.assertEqual(row["quality_decision"], "measurement_only")
                self.assertIsNone(row["quality_threshold"])
                self.assertFalse(row["automatic_exclusion_applied"])
                self.assertEqual(row["contamination_source"], "tracking_bbox_overlap")

            track_rows = [
                json.loads(line)
                for line in (out / TRACK_QUALITY_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual([r["track_id"] for r in track_rows], [1, 2])
            track1 = track_rows[0]
            self.assertEqual(track1["crop_count"], len(track1["crop_ids"]))
            self.assertEqual(track1["automatic_excluded_crop_count"], 0)

            summary = json.loads((out / QUALITY_SUMMARY_NAME).read_text())
            self.assertEqual(summary["crop_count"], 3)
            self.assertEqual(summary["track_count"], 2)
            self.assertIsNone(summary["quality_threshold"])
            self.assertFalse(summary["composite_quality_score_created"])
            self.assertFalse(summary["automatic_exclusion_performed"])
            self.assertFalse(summary["embedding_aggregation_modified"])
            self.assertFalse(summary["crop_files_modified"])
            self.assertGreaterEqual(summary["crops_with_other_person_overlap"], 1)

            # jpeg unchanged
            after = {p: (_sha256(p), p.stat().st_size, p.read_bytes()) for p in jpeg_paths}
            self.assertEqual(before, after)

            # collision without overwrite
            with self.assertRaises(Exception):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest, tracks=tracks, output_dir=out
                )

            # overwrite ok + deterministic
            result2 = run_analyze_reid_crop_quality(
                crop_manifest=manifest,
                tracks=tracks,
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(result2["crop_count"], 3)
            crop_rows2 = [
                json.loads(line)
                for line in (out / CROP_QUALITY_NAME).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(crop_rows, crop_rows2)

            # no tmp leftovers
            leftovers = list(out.parent.glob("_tmp_reid_quality_*")) + list(
                out.parent.glob("_backup_reid_quality_*")
            )
            self.assertEqual(leftovers, [])

            # validation failure leaves no final when new dir
            out2 = root / "quality_fail"
            bad_tracks = root / "bad_tracks.jsonl"
            bad_tracks.write_text("{bad\n", encoding="utf-8")
            with self.assertRaises(QualityError):
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest, tracks=bad_tracks, output_dir=out2
                )
            self.assertFalse(out2.exists())

    def test_no_model_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, tracks = _make_fixture(
                root, boxes=[(1, 0, [10, 10, 30, 40])]
            )
            with mock.patch("football_analytics.reid.quality.cv2.imread", wraps=cv2.imread) as _:
                run_analyze_reid_crop_quality(
                    crop_manifest=manifest,
                    tracks=tracks,
                    output_dir=root / "out",
                )
            # ensure torchreid / build_model not imported by quality module
            import football_analytics.reid.quality as quality_mod

            self.assertFalse(hasattr(quality_mod, "torchreid"))
            self.assertFalse(hasattr(quality_mod, "build_model"))


class HelpAndInferTests(unittest.TestCase):
    def test_infer_frame_size(self) -> None:
        w, h = infer_frame_size(
            observations=[_obs(track_id=1, frame_index=0, bbox=[0, 0, 90.2, 70.8])],
            manifest_rows=[{"bbox_xyxy": [10, 10, 30, 40]}],
        )
        self.assertGreaterEqual(w, 91)
        self.assertGreaterEqual(h, 71)


if __name__ == "__main__":
    unittest.main()
