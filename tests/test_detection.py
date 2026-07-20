"""Unit tests for Stage 2 person detection (stdlib unittest only)."""

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

from football_analytics.detection.annotate import draw_detections, sanitize_detection
from football_analytics.detection.pipeline import DetectionError, load_yolo_model, run_detection
from football_analytics.detection.writers import (
    WritersError,
    check_output_collision,
    cleanup_partial_outputs,
    finalize_outputs,
    output_paths,
    write_json_file,
    write_jsonl_lines,
)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None):
        self.boxes = boxes


class FakeModel:
    def __init__(self, names=None, boxes_per_frame=None):
        self.names = names if names is not None else {0: "person"}
        self.boxes_per_frame = boxes_per_frame or []
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        idx = len(self.predict_calls) - 1
        if idx < len(self.boxes_per_frame):
            boxes = self.boxes_per_frame[idx]
        else:
            boxes = FakeBoxes([], [], [])
        return [FakeResult(boxes if len(boxes) else None)]


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


class SanitizeTests(unittest.TestCase):
    def test_clips_and_keeps_valid(self) -> None:
        result = sanitize_detection(
            bbox_xyxy=[-10, -5, 70, 50],
            confidence=0.9,
            frame_width=64,
            frame_height=48,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["bbox_xyxy"], [0.0, 0.0, 64.0, 48.0])
        self.assertEqual(result["confidence"], 0.9)

    def test_rejects_bad_geometry(self) -> None:
        self.assertIsNone(
            sanitize_detection(
                bbox_xyxy=[10, 10, 10, 20],
                confidence=0.5,
                frame_width=64,
                frame_height=48,
            )
        )

    def test_rejects_bad_confidence(self) -> None:
        self.assertIsNone(
            sanitize_detection(
                bbox_xyxy=[1, 1, 10, 10],
                confidence=1.5,
                frame_width=64,
                frame_height=48,
            )
        )


class AnnotateTests(unittest.TestCase):
    def test_draw_detections_no_crash(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out = draw_detections(
            frame,
            [{"bbox_xyxy": [2.0, 2.0, 20.0, 20.0], "confidence": 0.8}],
        )
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(out, frame))


class WritersTests(unittest.TestCase):
    def test_overwrite_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "annotated.mp4").write_bytes(b"x")
            with self.assertRaises(WritersError):
                check_output_collision(out, overwrite=False)
            check_output_collision(out, overwrite=True)

    def test_finalize_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = output_paths(out)
            paths["tmp_annotated"].write_bytes(b"video")
            write_jsonl_lines(paths["tmp_detections"], [{"frame_index": 0}])
            write_json_file(paths["tmp_summary"], {"status": "ok"})
            finals = finalize_outputs(out)
            self.assertTrue(finals["annotated"].is_file())
            self.assertFalse(paths["tmp_annotated"].exists())
            cleanup_partial_outputs(out)
            # finals remain; tmps gone
            self.assertTrue(finals["summary"].is_file())


class LoadModelTests(unittest.TestCase):
    def test_missing_model_file_errors_without_download(self) -> None:
        missing = Path("/tmp/football-analytics-missing-weights-does-not-exist.pt")
        with self.assertRaises(DetectionError) as ctx:
            load_yolo_model(missing)
        self.assertIn("not found", str(ctx.exception).lower())
        self.assertIn("not automatic", str(ctx.exception).lower())


class PipelineTests(unittest.TestCase):
    def test_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            model = FakeModel()
            with self.assertRaises(DetectionError):
                run_detection(
                    video=Path(tmp) / "missing.mp4",
                    output_dir=out,
                    model=model,
                )

    def test_person_name_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            out = root / "out"
            bad = FakeModel(names={0: "car"})
            with self.assertRaises(DetectionError) as ctx:
                run_detection(video=video, output_dir=out, model=bad)
            self.assertIn("person", str(ctx.exception))

    def test_overwrite_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            out = root / "out"
            out.mkdir()
            (out / "detections.jsonl").write_text("", encoding="utf-8")
            model = FakeModel()
            with self.assertRaises(DetectionError):
                run_detection(video=video, output_dir=out, model=model, overwrite=False)

    def test_empty_detections_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=2)
            out = root / "benchmark_100"
            model = FakeModel(boxes_per_frame=[])
            # Fake weights path for sha256 skip
            result = run_detection(
                video=video,
                output_dir=out,
                model_path=root / "missing.pt",
                model=model,
                max_frames=2,
            )
            self.assertTrue(Path(result["annotated_path"]).is_file())
            self.assertEqual(Path(result["detections_path"]).read_text(encoding="utf-8"), "")
            summary = result["summary"]
            self.assertEqual(summary["metrics"]["total_detections"], 0)
            self.assertEqual(summary["metrics"]["frames_processed"], 2)
            self.assertEqual(summary["parameters"]["iou"], 0.70)
            self.assertFalse(summary["parameters"]["save"])
            self.assertFalse(summary["parameters"]["verbose"])
            # predict flags
            for call in model.predict_calls:
                self.assertEqual(call.get("device"), "cpu")
                self.assertEqual(call.get("classes"), [0])
                self.assertEqual(call.get("save"), False)
                self.assertEqual(call.get("verbose"), False)
                self.assertEqual(call.get("iou"), 0.70)

    def test_normal_detection_and_invalid_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=2, size=(64, 48))
            out = root / "full"
            # Frame 0: one valid box. Frame 1: invalid geometry + valid.
            boxes0 = FakeBoxes(
                xyxy=[[2.0, 2.0, 20.0, 20.0]],
                conf=[0.91],
                cls=[0.0],
            )
            boxes1 = FakeBoxes(
                xyxy=[[5.0, 5.0, 5.0, 10.0], [1.0, 1.0, 15.0, 15.0]],
                conf=[0.8, 0.7],
                cls=[0.0, 0.0],
            )
            model = FakeModel(boxes_per_frame=[boxes0, boxes1])
            result = run_detection(
                video=video,
                output_dir=out,
                model_path=root / "no-weights.pt",
                model=model,
            )
            lines = [
                json.loads(line)
                for line in Path(result["detections_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["class_name"], "person")
            self.assertEqual(lines[0]["class_id"], 0)
            self.assertIn("bbox_xyxy", lines[0])
            summary = result["summary"]
            self.assertEqual(summary["metrics"]["total_detections"], 2)
            self.assertEqual(summary["metrics"]["skipped_invalid_detections"], 1)
            self.assertEqual(summary["metrics"]["frames_with_detections"], 2)
            self.assertIn("sha256", summary["source"])
            self.assertIsNone(summary["model"]["sha256"])
            self.assertIn("ultralytics_version", summary["environment"])
            self.assertIn("torch_version", summary["environment"])
            # no leftover temps
            leftover = list(out.glob("_tmp_*")) + list(out.glob("*.tmp"))
            self.assertEqual(leftover, [])

    def test_failure_cleans_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=2)
            out = root / "out"

            class BoomModel(FakeModel):
                def predict(self, **kwargs):
                    raise RuntimeError("boom")

            with self.assertRaises(DetectionError):
                run_detection(video=video, output_dir=out, model=BoomModel())
            paths = output_paths(out)
            self.assertFalse(paths["tmp_annotated"].exists())
            self.assertFalse(paths["tmp_detections"].exists())
            self.assertFalse(paths["tmp_summary"].exists())
            self.assertFalse(paths["annotated"].exists())
            self.assertFalse(paths["summary"].exists())

    def test_default_loader_not_used_when_model_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            out = root / "out"
            model = FakeModel()
            with mock.patch(
                "football_analytics.detection.pipeline.load_yolo_model"
            ) as loader:
                run_detection(video=video, output_dir=out, model=model)
                loader.assert_not_called()

    def test_model_loader_di(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_mp4(video, frames=1)
            out = root / "out"
            fake = FakeModel()

            def factory(path: Path):
                self.assertTrue(str(path).endswith(".pt"))
                return fake

            result = run_detection(
                video=video,
                output_dir=out,
                model_path=root / "injected.pt",
                model_loader=factory,
            )
            self.assertEqual(result["summary"]["status"], "ok")
            self.assertEqual(len(fake.predict_calls), 1)


if __name__ == "__main__":
    unittest.main()
