"""Unit tests for Stage 5D-B1E-D external anchor review package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_external_anchor_review_package as b1ed  # noqa: E402

_CFG = _PROJECT_ROOT / "configs/reid/external_anchor_review_stage5d_target_001.yaml"
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_review_package"
)
_B1EC = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_occurrence_freeze"
)
_HEAD = "845cfab623a5f58ae31672d505575360593733e0"
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ExternalAnchorReviewTests(unittest.TestCase):
    def test_expected_git_and_selected_occurrences(self):
        cfg = b1ed.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(b1ed.SELECTED_CODES, ("EXT_004", "EXT_183", "EXT_198"))
        codes = [s["external_candidate_code"] for s in cfg["selected_occurrences"]]
        self.assertEqual(tuple(codes), b1ed.SELECTED_CODES)
        self.assertEqual(cfg["selection"]["max_candidates_per_occurrence"], 6)
        self.assertEqual(cfg["selection"]["max_total_candidates"], 18)
        self.assertEqual(cfg["selection"]["min_frame_gap"], 12)
        self.assertEqual(cfg["crop_extraction"]["padding_fraction"], 0.05)

    def test_hard_quality_thresholds_preregistered(self):
        cfg = b1ed.load_config(_CFG)
        hq = cfg["hard_quality"]
        self.assertEqual(hq["min_crop_height_px"], 64)
        self.assertEqual(hq["min_crop_width_px"], 20)
        self.assertEqual(hq["min_bbox_area_px2"], 1500)
        self.assertEqual(hq["max_edge_clipping_fraction"], 0.20)
        self.assertEqual(hq["max_person_iou"], 0.35)

    def test_edge_clip_and_iou_and_padding(self):
        clip = b1ed.edge_clipping_fraction([-10, 0, 10, 20], width=100, height=100)
        self.assertGreater(clip, 0.0)
        self.assertLessEqual(clip, 1.0)
        self.assertEqual(b1ed.iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 25.0 / 175.0)
        padded = b1ed.pad_bbox([10, 10, 30, 50], width=100, height=100, fraction=0.05)
        self.assertLessEqual(padded[0], 10)
        self.assertGreaterEqual(padded[2], 30)

    def test_temporal_buckets_deterministic(self):
        buckets = [
            "early",
            "early-middle",
            "middle",
            "late-middle",
            "late",
            "representative-support",
        ]
        a = b1ed.temporal_bucket_name(
            frame_index=0,
            first=0,
            last=100,
            buckets=buckets,
            representative_frame=50,
            rep_half_window=5,
        )
        b = b1ed.temporal_bucket_name(
            frame_index=50,
            first=0,
            last=100,
            buckets=buckets,
            representative_frame=50,
            rep_half_window=5,
        )
        self.assertEqual(a, "early")
        self.assertEqual(b, "representative-support")

    def test_dhash_and_hard_exclude(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[:, 32:] = 255
        h = b1ed.dhash_hex(gray, 8)
        self.assertEqual(b1ed.hamming_hex(h, h), 0)
        reasons = b1ed.hard_exclude(
            crop_w=10,
            crop_h=80,
            bbox_area_px=2000,
            edge_clip=0.01,
            max_iou=0.1,
            hq={
                "min_crop_height_px": 64,
                "min_crop_width_px": 20,
                "min_bbox_area_px2": 1500,
                "max_edge_clipping_fraction": 0.2,
                "max_person_iou": 0.35,
            },
            decode_ok=True,
            lineage_ok=True,
            sha_ok=True,
        )
        self.assertIn("crop_width_below_min", reasons)

    def test_contract_forbids_auto_and_embeddings(self):
        c = b1ed.build_contract()
        self.assertTrue(c["frozen_positive_occurrences_only"])
        self.assertFalse(c["unreviewed_occurrences_read"])
        self.assertTrue(c["no_osnet"])
        self.assertTrue(c["no_ocr"])
        self.assertTrue(c["manual_approval_required"])
        self.assertEqual(c["approved_anchor_crops"], 0)
        self.assertEqual(c["gallery_members"], 0)
        self.assertIn("target_anchor_yes", b1ed.ALLOWED_ANCHOR_DECISIONS)
        self.assertEqual(set(b1ed.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1ed.AnchorReviewError):
            b1ed.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1ed.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1ed, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1ed.AnchorReviewError) as ctx:
                    b1ed.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_live_b1e_c_and_b1e_d_if_present(self):
        if not _B1EC.is_dir():
            self.skipTest("B1E-C absent")
        s = json.loads((_B1EC / "stage5d_b1e_c_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            s["final_status"],
            "COMPLETED_STAGE5D_B1E_C_TARGET_001_EXTERNAL_OCCURRENCES_FROZEN",
        )
        self.assertEqual(s["selected_external_candidate_codes"], list(b1ed.SELECTED_CODES))
        self.assertEqual(s["selected_positive_count"], 3)

        if not _FINAL.is_dir():
            self.skipTest("B1E-D output absent")
        summary = json.loads(
            (_FINAL / "stage5d_b1e_d_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], b1ed.FINAL_STATUS)
        self.assertEqual(summary["source_observation_count"], 501)
        self.assertEqual(summary["unreviewed_ext_source_reads"], 0)
        self.assertEqual(summary["frozen_positive_occurrences"], 3)
        self.assertLessEqual(summary["total_candidate_count"], 18)
        for code in b1ed.SELECTED_CODES:
            self.assertLessEqual(summary["candidate_counts_per_occurrence"][code], 6)
        self.assertEqual(summary["contact_sheet_png_count"], 3)
        self.assertEqual(summary["reviewed_candidate_crops"], 0)
        self.assertEqual(summary["approved_anchor_crops"], 0)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["osnet_inference"], 0)
        self.assertEqual(summary["ocr"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["new_detection"], 0)
        self.assertEqual(summary["new_tracking"], 0)
        self.assertEqual(len(list(_FINAL.rglob("*.mp4"))), 0)
        sheets = list(
            (
                _FINAL / "review_packages" / "target_001_external_anchor_crop_review"
            ).glob("anchor_review_EXT_*.png")
        )
        self.assertEqual(len(sheets), 3)
        crops = list((_FINAL / "crops").rglob("*.png"))
        self.assertEqual(len(crops), summary["total_candidate_count"])
        self.assertLessEqual(len(crops), 18)
        inv = [
            json.loads(line)
            for line in (
                _FINAL / "inventory" / "target_001_external_anchor_candidate_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        ids = [r["anchor_candidate_id"] for r in inv]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(i.startswith("target_001_ext_anchor_") for i in ids))
        self.assertTrue(all(r["manual_fields_blank"] for r in inv))
        # Ordering: EXT_004 then EXT_183 then EXT_198, frames ascending within.
        order_codes = [r["source_occurrence_code"] for r in inv]
        self.assertEqual(
            order_codes,
            sorted(order_codes, key=lambda c: (b1ed.SELECTED_CODES.index(c),)),
        )
        qrows = [
            json.loads(line)
            for line in (
                _FINAL
                / "inventory"
                / "target_001_external_tracklet_observation_quality.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(qrows), 501)
        tpl = (
            _FINAL
            / "templates"
            / "target_001_external_anchor_crop_review_template.csv"
        )
        with tpl.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(inv))
        self.assertTrue(all(r["manual_anchor_decision"] == "" for r in rows))
        self.assertTrue(all(r["reviewer"] == "" for r in rows))
        # Source immutability.
        ext = _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
        self.assertEqual(_sha256(ext), _EXT_SHA)
        runtime = json.loads((_FINAL / "runtime" / "runtime.json").read_text(encoding="utf-8"))
        self.assertFalse(runtime["osnet_loaded"])
        self.assertFalse(runtime["yolo_loaded"])


if __name__ == "__main__":
    unittest.main()
