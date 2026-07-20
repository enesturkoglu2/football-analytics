"""Unit tests for Stage 3 person tracking (stdlib unittest only)."""

from __future__ import annotations

import json
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

from football_analytics.tracking.annotate import draw_tracks
from football_analytics.tracking.pipeline import (
    TrackingError,
    load_tracker_config,
    run_tracking,
)
from football_analytics.tracking.writers import (
    WritersError,
    check_output_collision,
    cleanup_partial_outputs,
    finalize_outputs,
    output_paths,
    write_json_file,
    write_jsonl_lines,
)

STAGE3_TRACKER = _PROJECT_ROOT / "configs" / "tracking" / "bytetrack_stage3.yaml"


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, *, is_track=True, ids=None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.is_track = is_track
        self.id = ids

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None):
        self.boxes = boxes


class FakeModel:
    def __init__(self, names=None, boxes_per_frame=None):
        self.names = names if names is not None else {0: "person"}
        self.boxes_per_frame = boxes_per_frame or []
        self.track_calls: list[dict] = []

    def track(self, **kwargs):
        self.track_calls.append(kwargs)
        idx = len(self.track_calls) - 1
        if idx < len(self.boxes_per_frame):
            boxes = self.boxes_per_frame[idx]
        else:
            boxes = FakeBoxes([], [], [], is_track=False, ids=None)
        return [FakeResult(boxes if len(boxes) else None)]

    def predict(self, **kwargs):
        raise AssertionError("predict must not be called during tracking")


def _write_tiny_mp4(path: Path, frames: int = 3, size=(64, 48), fps: float = 10.0) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create test video")
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 40) % 255
        writer.write(frame)
    writer.release()


class TrackerConfigTests(unittest.TestCase):
    def test_stage3_yaml_defaults(self) -> None:
        self.assertTrue(STAGE3_TRACKER.is_file())
        params = load_tracker_config(STAGE3_TRACKER)
        self.assertEqual(params["tracker_type"], "bytetrack")
        self.assertEqual(params["track_high_thresh"], 0.25)
        self.assertEqual(params["track_low_thresh"], 0.1)
        self.assertEqual(params["new_track_thresh"], 0.25)
        self.assertEqual(params["track_buffer"], 30)
        self.assertEqual(params["match_thresh"], 0.8)
        self.assertTrue(params["fuse_score"])

    def test_missing_tracker_config(self) -> None:
        with self.assertRaises(TrackingError):
            load_tracker_config(Path("/tmp/missing-bytetrack-stage3.yaml"))


class AnnotateTests(unittest.TestCase):
    def test_draw_with_and_without_id(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out = draw_tracks(
            frame,
            [
                {
                    "bbox_xyxy": [2.0, 2.0, 20.0, 20.0],
                    "confidence": 0.9,
                    "track_id": 7,
                },
                {
                    "bbox_xyxy": [30.0, 10.0, 50.0, 40.0],
                    "confidence": 0.8,
                    "track_id": None,
                },
            ],
        )
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(out, frame))


class WritersTests(unittest.TestCase):
    def test_overwrite_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "tracked.mp4").write_bytes(b"x")
            with self.assertRaises(WritersError):
                check_output_collision(out, overwrite=False)
            check_output_collision(out, overwrite=True)

    def test_finalize_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = output_paths(out)
            paths["tmp_tracked"].write_bytes(b"video")
            write_jsonl_lines(paths["tmp_tracks"], [{"frame_index": 0}])
            write_json_file(paths["tmp_summary"], {"status": "ok"})
            finals = finalize_outputs(out)
            self.assertTrue(finals["tracked"].is_file())
            self.assertFalse(paths["tmp_tracked"].exists())
            cleanup_partial_outputs(out)
            self.assertTrue(finals["summary"].is_file())


class PipelineTests(unittest.TestCase):
    def test_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with self.assertRaises(TrackingError):
                run_tracking(
                    video=Path(tmp) / "missing.mp4",
                    output_dir=out,
                    tracker_path=STAGE3_TRACKER,
                    model=FakeModel(),
                )

    def test_person_name_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            with self.assertRaises(TrackingError):
                run_tracking(
                    video=video,
                    output_dir=root / "out",
                    tracker_path=STAGE3_TRACKER,
                    model=FakeModel(names={0: "car"}),
                )

    def test_overwrite_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            out = root / "out"
            out.mkdir()
            (out / "tracks.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(TrackingError):
                run_tracking(
                    video=video,
                    output_dir=out,
                    tracker_path=STAGE3_TRACKER,
                    model=FakeModel(),
                    overwrite=False,
                )

    def test_is_track_false_yields_null_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            boxes = FakeBoxes(
                xyxy=[[2.0, 2.0, 20.0, 20.0]],
                conf=[0.9],
                cls=[0.0],
                is_track=False,
                ids=None,
            )
            model = FakeModel(boxes_per_frame=[boxes])
            result = run_tracking(
                video=video,
                output_dir=root / "out",
                tracker_path=STAGE3_TRACKER,
                model_path=root / "no-weights.pt",
                model=model,
            )
            rows = [
                json.loads(line)
                for line in Path(result["tracks_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["track_id"])
            metrics = result["summary"]["metrics"]
            self.assertEqual(metrics["box_observations_without_track_id"], 1)
            self.assertEqual(metrics["box_observations_with_track_id"], 0)
            self.assertEqual(metrics["unique_track_ids"], 0)

    def test_ids_none_even_if_is_track_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            boxes = FakeBoxes(
                xyxy=[[2.0, 2.0, 20.0, 20.0]],
                conf=[0.9],
                cls=[0.0],
                is_track=True,
                ids=None,
            )
            model = FakeModel(boxes_per_frame=[boxes])
            result = run_tracking(
                video=video,
                output_dir=root / "out",
                tracker_path=STAGE3_TRACKER,
                model_path=root / "no-weights.pt",
                model=model,
            )
            rows = [
                json.loads(line)
                for line in Path(result["tracks_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIsNone(rows[0]["track_id"])

    def test_track_ids_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=3, size=(64, 48))
            # Frame0: id=1 valid. Frame1: id=1 + invalid bbox. Frame2: id=2.
            boxes0 = FakeBoxes(
                xyxy=[[2.0, 2.0, 20.0, 20.0]],
                conf=[0.91],
                cls=[0.0],
                is_track=True,
                ids=[1.0],
            )
            boxes1 = FakeBoxes(
                xyxy=[[5.0, 5.0, 5.0, 10.0], [1.0, 1.0, 15.0, 15.0]],
                conf=[0.8, 0.7],
                cls=[0.0, 0.0],
                is_track=True,
                ids=[1.0, 1.0],
            )
            boxes2 = FakeBoxes(
                xyxy=[[3.0, 3.0, 18.0, 18.0]],
                conf=[0.85],
                cls=[0.0],
                is_track=True,
                ids=[2.0],
            )
            model = FakeModel(boxes_per_frame=[boxes0, boxes1, boxes2])
            result = run_tracking(
                video=video,
                output_dir=root / "benchmark_100",
                tracker_path=STAGE3_TRACKER,
                model_path=root / "no-weights.pt",
                model=model,
            )
            summary = result["summary"]
            metrics = summary["metrics"]
            self.assertEqual(metrics["frames_processed"], 3)
            # valid rows: frame0 one, frame1 one (invalid skipped), frame2 one => 3
            self.assertEqual(metrics["total_box_observations"], 3)
            self.assertEqual(metrics["skipped_invalid_detections"], 1)
            self.assertEqual(metrics["unique_track_ids"], 2)
            self.assertEqual(metrics["box_observations_with_track_id"], 3)
            self.assertEqual(metrics["box_observations_without_track_id"], 0)
            # id1 observed on frames 0 and 1 => count 2, span 2
            # id2 observed on frame 2 => count 1, span 1
            self.assertEqual(metrics["track_observation_count"]["min"], 1.0)
            self.assertEqual(metrics["track_observation_count"]["max"], 2.0)
            self.assertEqual(metrics["track_span_frames"]["min"], 1.0)
            self.assertEqual(metrics["track_span_frames"]["max"], 2.0)

            self.assertEqual(summary["parameters"]["detection_iou"], 0.70)
            self.assertEqual(summary["parameters"]["iou"], 0.70)
            self.assertEqual(summary["tracker"]["parameters"]["match_thresh"], 0.8)
            self.assertIsNotNone(summary["tracker"]["sha256"])
            self.assertIsNotNone(summary["source"]["sha256"])
            self.assertIn("ultralytics_version", summary["environment"])
            self.assertIn("torch_version", summary["environment"])
            self.assertTrue(
                any("detection_iou" in note for note in summary["notes"])
            )
            self.assertTrue(any("ID-switch" in note for note in summary["notes"]))

            # track kwargs
            for call in model.track_calls:
                self.assertEqual(call.get("device"), "cpu")
                self.assertEqual(call.get("classes"), [0])
                self.assertEqual(call.get("conf"), 0.25)
                self.assertEqual(call.get("iou"), 0.70)
                self.assertEqual(call.get("imgsz"), 640)
                self.assertTrue(call.get("persist"))
                self.assertFalse(call.get("save"))
                self.assertFalse(call.get("verbose"))
                self.assertTrue(str(call.get("tracker")).endswith("bytetrack_stage3.yaml"))

            leftover = list((root / "benchmark_100").glob("_tmp_*"))
            self.assertEqual(leftover, [])

    def test_failure_cleans_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=2)
            out = root / "out"

            class BoomModel(FakeModel):
                def track(self, **kwargs):
                    raise RuntimeError("boom")

            with self.assertRaises(TrackingError):
                run_tracking(
                    video=video,
                    output_dir=out,
                    tracker_path=STAGE3_TRACKER,
                    model=BoomModel(),
                )
            paths = output_paths(out)
            self.assertFalse(paths["tmp_tracked"].exists())
            self.assertFalse(paths["tmp_tracks"].exists())
            self.assertFalse(paths["tmp_summary"].exists())
            self.assertFalse(paths["tracked"].exists())

    def test_default_loader_not_used_when_model_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            model = FakeModel()
            with mock.patch(
                "football_analytics.tracking.pipeline.load_yolo_model"
            ) as loader:
                run_tracking(
                    video=video,
                    output_dir=root / "out",
                    tracker_path=STAGE3_TRACKER,
                    model=model,
                )
                loader.assert_not_called()

    def test_model_loader_di(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            fake = FakeModel()

            def factory(path: Path):
                self.assertTrue(str(path).endswith(".pt"))
                return fake

            result = run_tracking(
                video=video,
                output_dir=root / "out",
                tracker_path=STAGE3_TRACKER,
                model_path=root / "injected.pt",
                model_loader=factory,
            )
            self.assertEqual(result["summary"]["status"], "ok")
            self.assertEqual(len(fake.track_calls), 1)


if __name__ == "__main__":
    unittest.main()
