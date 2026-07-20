"""Unit tests for ReID crop selection (no real video / model)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.crop_select import (
    CropSelectError,
    filter_candidate_observations,
    float_bbox_to_int_crop,
    load_crop_selection_config,
    load_track_observations,
    select_crops_for_tracks,
    select_crops_from_tracks_file,
    validate_crop_selection_config,
)
from football_analytics.reid.schema import CROP_MANIFEST_SCHEMA_VERSION

STAGE4B_CONFIG = _PROJECT_ROOT / "configs" / "reid" / "crop_selection_stage4b.yaml"


def _base_config(**overrides):
    cfg = load_crop_selection_config(STAGE4B_CONFIG)
    # Deep-ish copy via yaml roundtrip for mutation safety
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    for key, value in overrides.items():
        if key in ("filters", "selection") and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return validate_crop_selection_config(cfg)


def _obs(
    *,
    track_id=1,
    frame_index=0,
    confidence=0.8,
    bbox=(10.0, 10.0, 70.0, 130.0),
    class_id=0,
    class_name="person",
    timestamp_sec=None,
):
    x1, y1, x2, y2 = bbox
    return {
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) / 30.0 if timestamp_sec is None else timestamp_sec,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": [x1, y1, x2, y2],
    }


class ConfigTests(unittest.TestCase):
    def test_stage4b_config_loads(self) -> None:
        cfg = load_crop_selection_config(STAGE4B_CONFIG)
        self.assertEqual(cfg["filters"]["min_bbox_area"], 2211.6)
        self.assertEqual(cfg["filters"]["min_short_side"], 29.77)
        self.assertEqual(cfg["filters"]["min_confidence"], 0.567)
        self.assertEqual(cfg["selection"]["max_crops_per_track"], 5)
        self.assertEqual(cfg["selection"]["min_frame_gap_within_track"], 7)

    def test_missing_required_field(self) -> None:
        raw = yaml.safe_load(STAGE4B_CONFIG.read_text(encoding="utf-8"))
        del raw["filters"]["min_bbox_area"]
        with self.assertRaises(CropSelectError):
            validate_crop_selection_config(raw)

    def test_negative_threshold_rejected(self) -> None:
        raw = yaml.safe_load(STAGE4B_CONFIG.read_text(encoding="utf-8"))
        raw["filters"]["min_short_side"] = -1
        with self.assertRaises(CropSelectError):
            validate_crop_selection_config(raw)


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _base_config()
        self.video_w = 200
        self.video_h = 200

    def test_threshold_equality_accepted(self) -> None:
        # Construct bbox with area == min and short_side == min after clamp.
        min_area = self.cfg["filters"]["min_bbox_area"]
        min_short = self.cfg["filters"]["min_short_side"]
        min_conf = self.cfg["filters"]["min_confidence"]
        width = min_short
        height = min_area / width
        obs = [
            _obs(
                confidence=min_conf,
                bbox=(0.0, 0.0, width, height),
            )
        ]
        kept, reasons = filter_candidate_observations(
            obs, config=self.cfg, video_width=self.video_w, video_height=self.video_h
        )
        self.assertEqual(len(kept), 1, reasons)
        self.assertAlmostEqual(kept[0]["bbox_area"], min_area, places=4)
        self.assertAlmostEqual(kept[0]["short_side"], min_short, places=4)
        self.assertAlmostEqual(kept[0]["detection_confidence"], min_conf, places=6)

    def test_area_filter(self) -> None:
        obs = [_obs(bbox=(0.0, 0.0, 20.0, 20.0), confidence=0.9)]
        kept, reasons = filter_candidate_observations(
            obs, config=self.cfg, video_width=self.video_w, video_height=self.video_h
        )
        self.assertEqual(kept, [])
        self.assertGreater(reasons.get("area_below_min", 0) + reasons.get("short_side_below_min", 0), 0)

    def test_confidence_filter(self) -> None:
        obs = [_obs(confidence=0.5)]
        kept, reasons = filter_candidate_observations(
            obs, config=self.cfg, video_width=self.video_w, video_height=self.video_h
        )
        self.assertEqual(kept, [])
        self.assertEqual(reasons.get("confidence_below_min"), 1)

    def test_null_track_id(self) -> None:
        row = _obs()
        row["track_id"] = None
        kept, reasons = filter_candidate_observations(
            [row], config=self.cfg, video_width=self.video_w, video_height=self.video_h
        )
        self.assertEqual(kept, [])
        self.assertEqual(reasons["null_track_id"], 1)

    def test_non_person_class(self) -> None:
        kept, reasons = filter_candidate_observations(
            [_obs(class_id=1, class_name="ball")],
            config=self.cfg,
            video_width=self.video_w,
            video_height=self.video_h,
        )
        self.assertEqual(kept, [])
        self.assertEqual(reasons["non_person_class"], 1)

    def test_nan_bbox_rejected(self) -> None:
        kept, reasons = filter_candidate_observations(
            [_obs(bbox=(0.0, 0.0, math.nan, 100.0))],
            config=self.cfg,
            video_width=self.video_w,
            video_height=self.video_h,
        )
        self.assertEqual(kept, [])
        self.assertEqual(reasons["non_finite_or_invalid_fields"], 1)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _base_config()

    def _cand(self, track_id, frame_index, quality, conf=0.8):
        return {
            "track_id": track_id,
            "frame_index": frame_index,
            "timestamp_sec": frame_index / 30.0,
            "detection_confidence": conf,
            "float_bbox_xyxy": [0.0, 0.0, 40.0, 80.0],
            "bbox_xyxy": [0, 0, 40, 80],
            "bbox_area": 3200.0,
            "short_side": 40.0,
            "quality_score": quality,
        }

    def test_quality_ordering(self) -> None:
        cands = [
            self._cand(1, 0, 1000.0),
            self._cand(1, 20, 3000.0),
            self._cand(1, 40, 2000.0),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual([r["frame_index"] for r in rows], [20, 40, 0])
        self.assertEqual([r["selection_rank"] for r in rows], [1, 2, 3])

    def test_confidence_tiebreak(self) -> None:
        cands = [
            self._cand(1, 10, 2000.0, conf=0.7),
            self._cand(1, 0, 2000.0, conf=0.9),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual(rows[0]["frame_index"], 0)
        self.assertEqual(rows[0]["detection_confidence"], 0.9)

    def test_frame_index_tiebreak(self) -> None:
        cands = [
            self._cand(1, 8, 2000.0, conf=0.8),
            self._cand(1, 0, 2000.0, conf=0.8),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual(rows[0]["frame_index"], 0)

    def test_max_crops_five(self) -> None:
        cands = [self._cand(1, i * 10, 5000.0 - i) for i in range(10)]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual(len(rows), 5)

    def test_gap_six_reject_seven_accept(self) -> None:
        # First pick frame 0 (highest quality). Frame 6 is gap 6 -> reject.
        # Frame 7 is gap 7 -> accept.
        cands = [
            self._cand(1, 0, 5000.0),
            self._cand(1, 6, 4000.0),
            self._cand(1, 7, 3000.0),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        frames = [r["frame_index"] for r in rows]
        self.assertEqual(frames, [0, 7])

    def test_duplicate_frame_blocked(self) -> None:
        cands = [
            self._cand(1, 0, 5000.0, conf=0.9),
            self._cand(1, 0, 4000.0, conf=0.8),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual(len(rows), 1)

    def test_deterministic_track_order(self) -> None:
        cands = [
            self._cand(5, 0, 3000.0),
            self._cand(2, 0, 3000.0),
            self._cand(9, 0, 3000.0),
        ]
        rows = select_crops_for_tracks(
            cands, config=self.cfg, source_video="v.mp4"
        )
        self.assertEqual([r["track_id"] for r in rows], [2, 5, 9])
        self.assertTrue(all(r["schema_version"] == CROP_MANIFEST_SCHEMA_VERSION for r in rows))


class TracksFileTests(unittest.TestCase):
    def test_track_id_filter_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            rows = [
                _obs(track_id=1, frame_index=0),
                _obs(track_id=2, frame_index=0),
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            cfg = load_crop_selection_config(STAGE4B_CONFIG)
            with self.assertRaises(CropSelectError):
                select_crops_from_tracks_file(
                    tracks_path=path,
                    config=cfg,
                    video_width=200,
                    video_height=200,
                    source_video="v.mp4",
                    track_ids=[999],
                )

    def test_bad_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            path.write_text("{bad\n", encoding="utf-8")
            with self.assertRaises(CropSelectError):
                load_track_observations(path)

    def test_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            path.write_text(json.dumps({"frame_index": 0}) + "\n", encoding="utf-8")
            with self.assertRaises(CropSelectError):
                load_track_observations(path)

    def test_int_bbox_conversion(self) -> None:
        box = float_bbox_to_int_crop(
            [1.2, 3.8, 10.1, 20.9], video_width=100, video_height=100
        )
        self.assertEqual(box, [1, 3, 11, 21])


if __name__ == "__main__":
    unittest.main()
