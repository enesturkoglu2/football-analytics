"""Unit tests for Stage 1 video ingest (stdlib unittest only)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.ingest.checksum import sha256_file
from football_analytics.ingest.ffprobe import (
    FfprobeError,
    extract_ffprobe_fields,
    parse_frame_rate,
    select_video_stream,
)
from football_analytics.ingest.manifest import build_manifest, write_json_atomic
from football_analytics.ingest.pipeline import IngestError, run_ingest
from football_analytics.ingest.validate import (
    ValidationError,
    check_output_collision,
    resolve_metadata,
    validate_video_path,
)


class ParseFrameRateTests(unittest.TestCase):
    def test_fraction(self) -> None:
        self.assertAlmostEqual(parse_frame_rate("30000/1001") or 0.0, 29.97002997, places=5)

    def test_integer_string(self) -> None:
        self.assertEqual(parse_frame_rate("25/1"), 25.0)

    def test_invalid_values(self) -> None:
        self.assertIsNone(parse_frame_rate(None))
        self.assertIsNone(parse_frame_rate("0/0"))
        self.assertIsNone(parse_frame_rate("N/A"))
        self.assertIsNone(parse_frame_rate("not-a-rate"))


class ChecksumTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self) -> None:
        payload = b"football-analytics-ingest-test"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(sha256_file(path), expected)


class ValidateVideoPathTests(unittest.TestCase):
    def test_missing_path(self) -> None:
        with self.assertRaises(ValidationError):
            validate_video_path(Path("/tmp/does-not-exist-football-analytics.mp4"))

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.mp4"
            path.write_bytes(b"")
            with self.assertRaises(ValidationError):
                validate_video_path(path)

    def test_bad_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.txt"
            path.write_bytes(b"not-empty")
            with self.assertRaises(ValidationError):
                validate_video_path(path)

    def test_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(b"not-empty")
            resolved = validate_video_path(path)
            self.assertTrue(resolved.is_file())


class OverwriteCollisionTests(unittest.TestCase):
    def test_existing_without_overwrite_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "video_manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValidationError) as ctx:
                check_output_collision(out, overwrite=False)
            self.assertIn("--overwrite", str(ctx.exception))

    def test_existing_with_overwrite_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "ffprobe.json").write_text("{}", encoding="utf-8")
            check_output_collision(out, overwrite=True)


class ResolveMetadataTests(unittest.TestCase):
    def test_matching_sources(self) -> None:
        resolved = resolve_metadata(
            ffprobe_fields={
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": 25.0,
                "r_frame_rate": 25.0,
                "nb_frames": 250,
                "duration_sec": 10.0,
            },
            opencv_fields={
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "frame_count": 250,
            },
        )
        self.assertEqual(resolved["width"], 1920)
        self.assertEqual(resolved["height"], 1080)
        self.assertEqual(resolved["fps"], 25.0)
        self.assertEqual(resolved["frame_count"], 250)
        self.assertEqual(resolved["duration_sec"], 10.0)

    def test_dimension_mismatch_fails(self) -> None:
        with self.assertRaises(ValidationError):
            resolve_metadata(
                ffprobe_fields={
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": 25.0,
                    "duration_sec": 10.0,
                    "nb_frames": 250,
                },
                opencv_fields={
                    "width": 1280,
                    "height": 720,
                    "fps": 25.0,
                    "frame_count": 250,
                },
            )

    def test_fps_mismatch_leaves_null_and_fails(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            resolve_metadata(
                ffprobe_fields={
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": 30.0,
                    "duration_sec": 5.0,
                    "nb_frames": 150,
                },
                opencv_fields={
                    "width": 640,
                    "height": 360,
                    "fps": 25.0,
                    "frame_count": 150,
                },
            )
        self.assertIn("fps", str(ctx.exception).lower())

    def test_frame_count_mismatch_becomes_null(self) -> None:
        resolved = resolve_metadata(
            ffprobe_fields={
                "width": 640,
                "height": 360,
                "avg_frame_rate": 25.0,
                "duration_sec": 4.0,
                "nb_frames": 100,
            },
            opencv_fields={
                "width": 640,
                "height": 360,
                "fps": 25.0,
                "frame_count": 99,
            },
        )
        self.assertIsNone(resolved["frame_count"])
        self.assertTrue(any("frame_count" in note for note in resolved["notes"]))


class FfprobeFieldTests(unittest.TestCase):
    def test_select_video_stream(self) -> None:
        payload = {
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 360},
            ]
        }
        stream = select_video_stream(payload)
        self.assertEqual(stream["codec_name"], "h264")

    def test_no_video_stream(self) -> None:
        with self.assertRaises(FfprobeError):
            select_video_stream({"streams": [{"codec_type": "audio"}]})

    def test_extract_fields(self) -> None:
        payload = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "2.5",
                "bit_rate": "1000000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "nb_frames": "62",
                }
            ],
        }
        fields = extract_ffprobe_fields(payload)
        self.assertEqual(fields["width"], 640)
        self.assertEqual(fields["height"], 360)
        self.assertEqual(fields["avg_frame_rate"], 25.0)
        self.assertEqual(fields["nb_frames"], 62)
        self.assertEqual(fields["duration_sec"], 2.5)


class AtomicWriteTests(unittest.TestCase):
    def test_write_json_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            write_json_atomic(path, {"status": "ok"})
            self.assertTrue(path.is_file())
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "ok")


class BuildManifestTests(unittest.TestCase):
    def test_schema_keys(self) -> None:
        manifest = build_manifest(
            video_path=Path("/tmp/clip.mp4"),
            size_bytes=10,
            sha256="abc",
            ffprobe_fields={
                "format_name": "mp4",
                "duration_sec": 1.0,
                "bit_rate": 100,
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "avg_frame_rate_raw": "25/1",
                "r_frame_rate_raw": "25/1",
                "nb_frames": 25,
            },
            opencv_fields={
                "opened": True,
                "first_frame_ok": True,
                "width": 640,
                "height": 360,
                "fps": 25.0,
                "frame_count": 25,
            },
            resolved={
                "width": 640,
                "height": 360,
                "fps": 25.0,
                "frame_count": 25,
                "duration_sec": 1.0,
                "notes": [],
            },
        )
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["source"]["sha256"], "abc")
        self.assertEqual(manifest["resolved"]["width"], 640)


class PipelineTests(unittest.TestCase):
    def test_run_ingest_success_with_mocks(self) -> None:
        ffprobe_payload = {
            "format": {
                "format_name": "mp4",
                "duration": "2.0",
                "bit_rate": "500000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 320,
                    "height": 240,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "nb_frames": "50",
                }
            ],
        }
        opencv_fields = {
            "opened": True,
            "first_frame_ok": True,
            "width": 320,
            "height": 240,
            "fps": 25.0,
            "frame_count": 50,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"fake-video-bytes")
            out_dir = root / "outputs" / "ingest"

            with mock.patch(
                "football_analytics.ingest.pipeline.run_ffprobe",
                return_value=ffprobe_payload,
            ), mock.patch(
                "football_analytics.ingest.pipeline.read_opencv_metadata",
                return_value=opencv_fields,
            ):
                result = run_ingest(video, out_dir, overwrite=False)

            manifest_path = Path(result["manifest_path"])
            ffprobe_path = Path(result["ffprobe_path"])
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(ffprobe_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["resolved"]["width"], 320)
            self.assertEqual(
                manifest["source"]["sha256"],
                hashlib.sha256(b"fake-video-bytes").hexdigest(),
            )

            # Without --overwrite, a second run must fail and leave files unchanged.
            stamp = manifest_path.read_text(encoding="utf-8")
            with mock.patch(
                "football_analytics.ingest.pipeline.run_ffprobe",
                return_value=ffprobe_payload,
            ), mock.patch(
                "football_analytics.ingest.pipeline.read_opencv_metadata",
                return_value=opencv_fields,
            ):
                with self.assertRaises(IngestError) as ctx:
                    run_ingest(video, out_dir, overwrite=False)
            self.assertIn("--overwrite", str(ctx.exception))
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), stamp)

            # With --overwrite, rewrite succeeds.
            with mock.patch(
                "football_analytics.ingest.pipeline.run_ffprobe",
                return_value=ffprobe_payload,
            ), mock.patch(
                "football_analytics.ingest.pipeline.read_opencv_metadata",
                return_value=opencv_fields,
            ):
                result2 = run_ingest(video, out_dir, overwrite=True)
            self.assertEqual(result2["manifest"]["status"], "ok")

    def test_run_ingest_fails_before_writing_on_opencv_error(self) -> None:
        from football_analytics.ingest.opencv_meta import OpenCVMetaError

        ffprobe_payload = {
            "format": {"format_name": "mp4", "duration": "1.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 320,
                    "height": 240,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "nb_frames": "25",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"x")
            out_dir = root / "out"

            with mock.patch(
                "football_analytics.ingest.pipeline.run_ffprobe",
                return_value=ffprobe_payload,
            ), mock.patch(
                "football_analytics.ingest.pipeline.read_opencv_metadata",
                side_effect=OpenCVMetaError("cannot read first frame"),
            ):
                with self.assertRaises(IngestError):
                    run_ingest(video, out_dir, overwrite=False)

            self.assertFalse((out_dir / "video_manifest.json").exists())
            self.assertFalse((out_dir / "ffprobe.json").exists())

if __name__ == "__main__":
    unittest.main()
