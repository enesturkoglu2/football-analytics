"""Unit tests for Stage 5C clean label-blind review universe builder."""

from __future__ import annotations

import hashlib
import json
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
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_reid_jersey_clean_review_universe as cu  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_crop(
    path: Path, *, w: int = 40, h: int = 80, color: tuple[int, int, int] = (20, 40, 60)
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    # Unique noise so distinct fixtures do not share SHA-256.
    img[0, 0] = (color[0] % 250, color[1] % 250, color[2] % 250)
    img[1, 1] = ((color[0] + 17) % 255, (color[1] + 31) % 255, (color[2] + 53) % 255)
    cv2.imwrite(str(path), img)
    return _sha(path)


def _vis_row(
    *,
    crop_id: str,
    path: Path,
    sha: str,
    track: int,
    segment: str,
    frame: int,
    rank: int,
    w: int = 40,
    h: int = 80,
) -> dict:
    return {
        "schema_version": "reid_jersey_visibility_crop_signal_v1",
        "crop_id": crop_id,
        "segment_id": segment,
        "raw_track_id": track,
        "frame_index": frame,
        "selection_rank": rank,
        "crop_source_kind": "reused_baseline_selected_crop",
        "source_crop_path": str(path),
        "source_crop_sha256": sha,
        "crop_width_px": w,
        "crop_height_px": h,
        "roi_x_min": 5,
        "roi_y_min": 8,
        "roi_x_max": 30,
        "roi_y_max": 45,
        "roi_width_px": 25,
        "roi_height_px": 37,
        "roi_area_px": 25 * 37,
        "laplacian_variance": 100.0,
        "tenengrad_mean": 10.0,
        "local_contrast": 1.0,
        "entropy": 4.0,
        "edge_density": 0.1,
        "roi_other_person_union_coverage": 0.0,
        "roi_other_person_center_inside_count": 0,
        "manual_number_visible": None,
        "manual_number_readable": None,
        "manual_jersey_number": None,
        "manual_notes": None,
        "manual_back_facing": None,
        "manual_digit_count": None,
    }


class CleanUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.vis = self.root / "visibility"
        self.vis.mkdir()
        self.cfg = self.root / "config.yaml"
        self.cfg.write_text(
            (_PROJECT_ROOT / "configs/reid/jersey_clean_review_universe_stage5c_rebuild_r2.yaml").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _prepare(self, rows: list[dict]) -> None:
        (self.vis / "jersey_visibility_crop_signals.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (self.vis / "jersey_visibility_summary.json").write_text(
            json.dumps({"status": "ok", "total_selected_crop_count": len(rows)}) + "\n",
            encoding="utf-8",
        )

    def test_deterministic_id_order_and_manual_blankness(self) -> None:
        c1 = self.project / "c1.jpg"
        c2 = self.project / "c2.jpg"
        s1 = _write_crop(c1, color=(20, 40, 60))
        s2 = _write_crop(c2, color=(90, 10, 200))
        rows = [
            _vis_row(
                crop_id="track_2_frame_10_rank_1",
                path=c2,
                sha=s2,
                track=2,
                segment="raw_2_full",
                frame=10,
                rank=1,
            ),
            _vis_row(
                crop_id="track_1_frame_5_rank_2",
                path=c1,
                sha=s1,
                track=1,
                segment="raw_1_s02",
                frame=5,
                rank=2,
            ),
        ]
        self._prepare(rows)
        out = self.root / "out"
        summary = cu.run_build_clean_review_universe(
            visibility_dir=self.vis,
            config_path=self.cfg,
            project_root=self.project,
            output_dir=out,
        )
        items = [
            json.loads(line)
            for line in (out / "clean_review_items.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["canonical_item_count"], 2)
        self.assertEqual(items[0]["review_item_id"], "review_track_1_frame_5_rank_2")
        self.assertEqual(items[1]["review_item_id"], "review_track_2_frame_10_rank_1")
        self.assertEqual([i["canonical_order"] for i in items], [1, 2])
        for item in items:
            for field in cu.MANUAL_NULL_FIELDS:
                self.assertIsNone(item[field])
            for field in cu.OCR_FORBIDDEN_FIELDS:
                self.assertIsNone(item[field])
            self.assertEqual(item["annotation_status"], "unreviewed")
            self.assertTrue(item["review_universe_label_blind"])
            self.assertFalse(item["selection_performed"])

    def test_duplicate_crop_sha_rejected(self) -> None:
        c1 = self.project / "a.jpg"
        sha = _write_crop(c1)
        # second path same bytes
        c2 = self.project / "b.jpg"
        c2.write_bytes(c1.read_bytes())
        rows = [
            _vis_row(
                crop_id="track_1_frame_1_rank_1",
                path=c1,
                sha=sha,
                track=1,
                segment="raw_1_full",
                frame=1,
                rank=1,
            ),
            _vis_row(
                crop_id="track_1_frame_2_rank_1",
                path=c2,
                sha=sha,
                track=1,
                segment="raw_1_full",
                frame=2,
                rank=1,
            ),
        ]
        self._prepare(rows)
        with self.assertRaisesRegex(cu.CleanUniverseError, "duplicate source crop SHA"):
            cu.run_build_clean_review_universe(
                visibility_dir=self.vis,
                config_path=self.cfg,
                project_root=self.project,
                output_dir=self.root / "out_dup",
            )

    def test_missing_source_and_sha_mismatch(self) -> None:
        c1 = self.project / "ok.jpg"
        sha = _write_crop(c1)
        missing = self.project / "missing.jpg"
        rows = [
            _vis_row(
                crop_id="track_1_frame_1_rank_1",
                path=missing,
                sha="0" * 64,
                track=1,
                segment="raw_1_full",
                frame=1,
                rank=1,
            )
        ]
        self._prepare(rows)
        with self.assertRaisesRegex(cu.CleanUniverseError, "missing source crop"):
            cu.run_build_clean_review_universe(
                visibility_dir=self.vis,
                config_path=self.cfg,
                project_root=self.project,
                output_dir=self.root / "out_miss",
            )
        rows = [
            _vis_row(
                crop_id="track_1_frame_1_rank_1",
                path=c1,
                sha="1" * 64,
                track=1,
                segment="raw_1_full",
                frame=1,
                rank=1,
            )
        ]
        self._prepare(rows)
        with self.assertRaisesRegex(cu.CleanUniverseError, "SHA mismatch"):
            cu.run_build_clean_review_universe(
                visibility_dir=self.vis,
                config_path=self.cfg,
                project_root=self.project,
                output_dir=self.root / "out_sha",
            )
        # ensure good sha still works for control
        self.assertEqual(len(sha), 64)

    def test_manual_field_nonzero_rejected(self) -> None:
        c1 = self.project / "m.jpg"
        sha = _write_crop(c1)
        row = _vis_row(
            crop_id="track_1_frame_1_rank_1",
            path=c1,
            sha=sha,
            track=1,
            segment="raw_1_full",
            frame=1,
            rank=1,
        )
        row["manual_jersey_number"] = "9"
        self._prepare([row])
        with self.assertRaisesRegex(cu.CleanUniverseError, "manual field must be null"):
            cu.run_build_clean_review_universe(
                visibility_dir=self.vis,
                config_path=self.cfg,
                project_root=self.project,
                output_dir=self.root / "out_manual",
            )

    def test_invalid_roi_rejected(self) -> None:
        c1 = self.project / "r.jpg"
        sha = _write_crop(c1)
        row = _vis_row(
            crop_id="track_1_frame_1_rank_1",
            path=c1,
            sha=sha,
            track=1,
            segment="raw_1_full",
            frame=1,
            rank=1,
        )
        row["roi_x_max"] = 3
        row["roi_width_px"] = 10
        self._prepare([row])
        with self.assertRaisesRegex(cu.CleanUniverseError, "invalid ROI"):
            cu.run_build_clean_review_universe(
                visibility_dir=self.vis,
                config_path=self.cfg,
                project_root=self.project,
                output_dir=self.root / "out_roi",
            )

    def test_config_schema_present(self) -> None:
        cfg = yaml.safe_load(
            (
                _PROJECT_ROOT
                / "configs/reid/jersey_clean_review_universe_stage5c_rebuild_r2.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["schema_version"], cu.CONFIG_SCHEMA)
        self.assertEqual(cfg["review_item_id"]["format"], "review_{crop_id}")
        self.assertFalse(cfg["output"]["png_jpeg_mp4_allowed"])


if __name__ == "__main__":
    unittest.main()
