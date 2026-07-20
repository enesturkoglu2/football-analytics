"""Unit tests for ReID crop extraction and atomic writers (no real sample.mp4)."""

from __future__ import annotations

import json
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

from football_analytics.reid.crop_extract import (
    CropExtractError,
    crop_frame_region,
    extract_crops_single_pass,
    run_select_reid_crops,
    write_crop_jpeg,
)
from football_analytics.reid.crop_select import float_bbox_to_int_crop
from football_analytics.reid.schema import build_crop_manifest_row
from football_analytics.reid.writers import (
    MANIFEST_NAME,
    ReIDWritersError,
    check_output_collision,
    cleanup_dir,
    create_temp_output_dir,
    finalize_output_dir,
    validate_manifest_disk_consistency,
    write_manifest_jsonl,
)

STAGE4B_CONFIG = _PROJECT_ROOT / "configs" / "reid" / "crop_selection_stage4b.yaml"


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0
        self.opened = True
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True
        self.opened = False


class BBoxConvertTests(unittest.TestCase):
    def test_floor_ceil(self) -> None:
        self.assertEqual(
            float_bbox_to_int_crop([1.2, 3.8, 10.1, 20.9], video_width=100, video_height=100),
            [1, 3, 11, 21],
        )

    def test_negative_and_oob_clamp(self) -> None:
        box = float_bbox_to_int_crop(
            [-5.0, -2.0, 50.5, 80.2], video_width=40, video_height=60
        )
        self.assertEqual(box, [0, 0, 40, 60])

    def test_empty_crop_rejected(self) -> None:
        with self.assertRaises(Exception):
            float_bbox_to_int_crop([10.0, 10.0, 10.2, 10.2], video_width=5, video_height=5)


class CropArrayTests(unittest.TestCase):
    def test_synthetic_crop_size(self) -> None:
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        frame[10:30, 5:25] = (0, 0, 255)
        crop = crop_frame_region(frame, [5, 10, 25, 30])
        self.assertEqual(crop.shape, (20, 20, 3))
        self.assertTrue(np.all(crop[:, :, 2] == 255))

    def test_negative_index_rejected(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(CropExtractError):
            crop_frame_region(frame, [-1, 0, 10, 10])


class ExtractPassTests(unittest.TestCase):
    def test_single_pass_multi_crop_same_frame(self) -> None:
        frames = [np.full((60, 60, 3), fill_value=i, dtype=np.uint8) for i in range(5)]
        rows = [
            build_crop_manifest_row(
                track_id=1,
                frame_index=2,
                timestamp_sec=0.2,
                source_video="synthetic.mp4",
                bbox_xyxy=[0, 0, 10, 10],
                detection_confidence=0.8,
                quality_score=100.0,
                selection_rank=1,
            ),
            build_crop_manifest_row(
                track_id=2,
                frame_index=2,
                timestamp_sec=0.2,
                source_video="synthetic.mp4",
                bbox_xyxy=[10, 10, 20, 20],
                detection_confidence=0.7,
                quality_score=90.0,
                selection_rank=1,
            ),
            build_crop_manifest_row(
                track_id=1,
                frame_index=4,
                timestamp_sec=0.4,
                source_video="synthetic.mp4",
                bbox_xyxy=[0, 0, 8, 8],
                detection_confidence=0.9,
                quality_score=110.0,
                selection_rank=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fake = FakeCapture(frames)
            written = extract_crops_single_pass(
                video_path="ignored.mp4",
                selected_rows=rows,
                output_dir=out,
                open_capture=lambda _p: fake,
            )
            self.assertEqual(len(written), 3)
            self.assertTrue(fake.released)
            self.assertEqual(fake.index, 5)  # read through frame 4 inclusive => 5 reads
            for row in rows:
                path = out / row["crop_relative_path"]
                self.assertTrue(path.is_file())
                img = cv2.imread(str(path))
                self.assertIsNotNone(img)
                self.assertEqual(
                    img.shape[0],
                    int(row["bbox_height"]),
                )
                self.assertEqual(
                    img.shape[1],
                    int(row["bbox_width"]),
                )

    def test_frame_read_failure(self) -> None:
        frames = [np.zeros((20, 20, 3), dtype=np.uint8)]
        rows = [
            build_crop_manifest_row(
                track_id=1,
                frame_index=3,
                timestamp_sec=0.1,
                source_video="synthetic.mp4",
                bbox_xyxy=[0, 0, 5, 5],
                detection_confidence=0.8,
                quality_score=100.0,
                selection_rank=1,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CropExtractError):
                extract_crops_single_pass(
                    video_path="ignored.mp4",
                    selected_rows=rows,
                    output_dir=Path(tmp),
                    open_capture=lambda _p: FakeCapture(frames),
                )


class WritersTests(unittest.TestCase):
    def test_output_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reid_out"
            out.mkdir()
            with self.assertRaises(ReIDWritersError):
                check_output_collision(out, overwrite=False)
            check_output_collision(out, overwrite=True)

    def test_overwrite_temp_final_and_cleanup_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            final_dir = parent / "crops_out"
            final_dir.mkdir()
            (final_dir / "old.txt").write_text("old", encoding="utf-8")

            temp_dir = create_temp_output_dir(final_dir)
            self.assertTrue(temp_dir.is_dir())
            (temp_dir / "crop_manifest.jsonl").write_text("{}\n", encoding="utf-8")

            finalized = finalize_output_dir(
                temp_dir=temp_dir, final_dir=final_dir, overwrite=True
            )
            self.assertEqual(finalized, final_dir.resolve())
            self.assertTrue((final_dir / "crop_manifest.jsonl").is_file())
            self.assertFalse((final_dir / "old.txt").exists())
            leftovers = list(parent.glob("_tmp_reid_crops_*")) + list(
                parent.glob("_backup_reid_crops_*")
            )
            self.assertEqual(leftovers, [])

            # Error cleanup path
            temp2 = create_temp_output_dir(final_dir)
            cleanup_dir(temp2)
            self.assertFalse(temp2.exists())

    def test_manifest_jpeg_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = build_crop_manifest_row(
                track_id=3,
                frame_index=1,
                timestamp_sec=0.0,
                source_video="v.mp4",
                bbox_xyxy=[0, 0, 4, 4],
                detection_confidence=0.8,
                quality_score=50.0,
                selection_rank=1,
            )
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            crop = crop_frame_region(frame, row["bbox_xyxy"])
            write_crop_jpeg(out / row["crop_relative_path"], crop)
            write_manifest_jsonl(out / MANIFEST_NAME, [row])
            validate_manifest_disk_consistency(out, [row])


class PipelineMockTests(unittest.TestCase):
    def test_run_select_with_mock_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "tiny.mp4"
            tracks = root / "tracks.jsonl"
            out = root / "reid_smoke"

            # Tiny real mp4 for probe_video_size; extraction uses mock capture.
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (80, 120),
            )
            self.assertTrue(writer.isOpened())
            for i in range(20):
                frame = np.zeros((120, 80, 3), dtype=np.uint8)
                frame[:, :] = (i * 10) % 255
                writer.write(frame)
            writer.release()

            # Large enough boxes to pass balanced thresholds on 80x120 video.
            # area needs >= 2211.6, short_side >= 29.77, conf >= 0.567
            observations = []
            for frame_index in (0, 7, 14):
                observations.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": frame_index / 10.0,
                        "track_id": 463,
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.8,
                        "bbox_xyxy": [0.0, 0.0, 50.0, 100.0],
                    }
                )
            with tracks.open("w", encoding="utf-8") as handle:
                for row in observations:
                    handle.write(json.dumps(row) + "\n")

            frames = [np.full((120, 80, 3), fill_value=i, dtype=np.uint8) for i in range(20)]
            fake = FakeCapture(frames)

            with mock.patch(
                "football_analytics.reid.crop_extract.probe_video_size",
                return_value=(80, 120, 10.0, 20),
            ):
                result = run_select_reid_crops(
                    video=video,
                    tracks=tracks,
                    config=STAGE4B_CONFIG,
                    output_dir=out,
                    track_ids=[463],
                    overwrite=False,
                    open_capture=lambda _p: fake,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["crops_written"], 3)
            manifest = Path(result["manifest_path"])
            self.assertTrue(manifest.is_file())
            lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(Path(result["output_dir"], row["crop_relative_path"]).is_file() for row in lines))
            # No leftover temp dirs
            self.assertEqual(list(root.glob("_tmp_reid_crops_*")), [])


if __name__ == "__main__":
    unittest.main()
