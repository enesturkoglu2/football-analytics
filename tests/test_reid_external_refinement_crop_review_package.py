"""Unit tests for Stage 5D-F3E external refinement crop review package."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_external_refinement_crop_review_package as f3e  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_refinement_crop_review_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_crop_review_package"
)
_F3D = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_manual_freeze"
)
_HEAD = "8ee00b848da5f24af27cccc01a8adea731941e91"
_F3D_SNAP = "6c37869b6daa04484dff89afe81280bb1e3e5f7b60e47c09b54784e7dedcbd5f"
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


class ExternalRefinementCropReviewPackageTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3e.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3e.SAME_TEAM_DISTRACTORS), 35)
        self.assertEqual(cfg["selection"]["ext_161_max_candidates"], 4)
        self.assertEqual(cfg["selection"]["min_frame_gap"], 12)
        self.assertEqual(cfg["hard_quality"]["min_crop_height_px"], 64)
        self.assertEqual(cfg["contact_sheets"]["min_width_px"], 3600)
        self.assertTrue(cfg["evaluation_source"]["decode_forbidden"])

    def test_f3d_freeze_and_snapshot_validation(self):
        cfg = f3e.load_config(_CFG)
        if not _F3D.is_dir():
            self.skipTest("F3D absent")
        out = f3e.validate_f3d(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3D_SNAP)
        self.assertEqual(out["freeze"]["new_frozen_target_occurrence"], "EXT_161")
        corr = [(c["from"], c["to"]) for c in out["freeze"]["transcription_corrections"]]
        self.assertEqual(corr, list(f3e.TRANSCRIPTION_CORRECTIONS))
        self.assertEqual(
            [r["external_candidate_code"] for r in out["distractors"]],
            list(f3e.SAME_TEAM_DISTRACTORS),
        )

    def test_external_source_contract(self):
        cfg = f3e.load_config(_CFG)
        ext = f3e.validate_external_and_tracking(_PROJECT_ROOT, cfg)
        self.assertEqual(ext["sha256"], _EXT_SHA)
        self.assertEqual(ext["frames"], 784)
        self.assertIn("EXT_161", ext["mapping"])
        self.assertEqual(ext["mapping"]["EXT_161"]["raw_external_track_id"], 347)

    def test_hard_quality_and_rank_helpers(self):
        reasons = f3e.hard_exclude(
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
        )
        self.assertIn("crop_width_below_min", reasons)
        a = {
            "hard_quality_pass": True,
            "frame_index": 10,
            "quality": {
                "max_person_iou": 0.1,
                "edge_clipping_fraction": 0.05,
                "laplacian_variance": 100.0,
                "bbox_area": 2000.0,
            },
        }
        b = dict(a)
        b["hard_quality_pass"] = False
        self.assertLess(
            f3e.observation_rank_key(a, midpoint=50)[0],
            f3e.observation_rank_key(b, midpoint=50)[0],
        )

    def test_contract_and_vocab(self):
        c = f3e.build_contract()
        self.assertTrue(c["external_only"])
        self.assertFalse(c["sample_read"])
        self.assertTrue(c["existing_bbox_lineage_only"])
        self.assertEqual(c["target_source_references"], 9)
        self.assertEqual(c["distractor_candidate_count"], 35)
        self.assertTrue(c["one_candidate_per_distractor_source"])
        self.assertEqual(c["hard_negative_approvals"], 0)
        self.assertIn("target_crop_yes", f3e.ALLOWED_TARGET_DECISIONS)
        self.assertIn("hard_negative_crop_yes", f3e.ALLOWED_HN_DECISIONS)

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3e.CropReviewError):
            f3e.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3e.CropReviewError):
                f3e.atomic_publish(tmp, final)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3E root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3e_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3e.FINAL_STATUS)
        self.assertEqual(summary["existing_frozen_target_occurrences"], 4)
        self.assertEqual(summary["official_gallery_v1_members"], 7)
        self.assertEqual(summary["approved_target_source_count"], 9)
        self.assertEqual(summary["immutable_approved_expansion_sources"], 2)
        self.assertGreaterEqual(summary["ext_161_target_crop_candidates"], 1)
        self.assertLessEqual(summary["ext_161_target_crop_candidates"], 4)
        self.assertEqual(summary["same_team_distractor_sources"], 35)
        self.assertEqual(summary["distractor_crop_candidates"], 35)
        self.assertEqual(summary["source_silently_dropped"], 0)
        self.assertEqual(summary["manual_crop_decisions"], 0)
        self.assertEqual(summary["approved_new_target_crops"], 0)
        self.assertEqual(summary["approved_hard_negative_crops"], 0)
        self.assertEqual(summary["target_gallery_members"], 7)
        self.assertEqual(summary["hard_negative_gallery_members"], 0)
        self.assertEqual(summary["new_embeddings"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_crop_read_count"], 0)
        self.assertEqual(summary["hard_negative_sheets"], 3)
        self.assertEqual(summary["hard_negative_sheet_distribution"], [12, 12, 11])
        self.assertEqual(summary["diagnostic_mp4"], 1)
        self.assertEqual(summary["hard_negative_candidate_crop_copies"], 35)

        pkg = (
            _FINAL
            / "review_packages"
            / "target_001_external_refinement_crop_review"
        )
        self.assertTrue((pkg / "target_001_existing_target_source_reference.png").is_file())
        self.assertTrue((pkg / "target_001_EXT_161_target_crop_candidates.png").is_file())
        for i in range(1, 4):
            self.assertTrue(
                (pkg / f"target_001_hard_negative_crop_candidates_{i:02d}.png").is_file()
            )
        self.assertTrue(
            (
                _FINAL / "videos" / "target_001_external_hard_negative_crop_review.mp4"
            ).is_file()
        )

        with (
            _FINAL / "templates" / "target_001_EXT_161_target_crop_review_template.csv"
        ).open(encoding="utf-8") as handle:
            trows = list(csv.DictReader(handle))
        self.assertEqual(len(trows), summary["ext_161_target_crop_candidates"])
        for row in trows:
            self.assertEqual(row["manual_target_crop_decision"], "")
            self.assertEqual(row["source_occurrence_code"], "EXT_161")

        with (
            _FINAL
            / "templates"
            / "target_001_external_hard_negative_crop_review_template.csv"
        ).open(encoding="utf-8") as handle:
            hrows = list(csv.DictReader(handle))
        self.assertEqual(len(hrows), 35)
        codes = [r["source_external_code"] for r in hrows]
        self.assertEqual(codes, list(f3e.SAME_TEAM_DISTRACTORS))
        for row in hrows:
            self.assertEqual(row["manual_hard_negative_crop_decision"], "")
            self.assertEqual(row["frozen_source_decision"], "same_team_distractor_yes")

        inv = [
            json.loads(line)
            for line in (
                _FINAL
                / "inventory"
                / "target_001_external_hard_negative_crop_candidate_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(inv), 35)
        for item in inv:
            self.assertFalse(item["hard_negative_gallery_member"])
            self.assertFalse(item["embedding_input"])
            self.assertTrue(item["manual_crop_approval_pending"])

        audit = json.loads(
            (
                _FINAL
                / "quality"
                / "target_001_external_hard_negative_crop_selection_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(audit["source_silently_dropped"], 0)
        self.assertFalse(audit["jersey_metadata_used_for_selection"])
        self.assertFalse(audit["identity_or_similarity_used"])

        runtime = json.loads(
            (
                _FINAL
                / "runtime"
                / "target_001_external_refinement_crop_access_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(runtime["sample_video_read"])
        self.assertEqual(runtime["sample_score_row_read_count"], 0)
        self.assertEqual(runtime["sample_rank_row_read_count"], 0)
        self.assertFalse(runtime["sample_used_for_candidate_selection"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("sample.mp4")), [])

        # Target near-duplicate / diversity basics.
        t_inv = [
            json.loads(line)
            for line in (
                _FINAL
                / "inventory"
                / "target_001_EXT_161_target_crop_candidate_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        frames = [int(r["frame_index"]) for r in t_inv]
        self.assertEqual(frames, sorted(frames))
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                self.assertGreaterEqual(abs(frames[i] - frames[j]), 12)

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3E root absent")
        with mock.patch.object(f3e, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3e.CropReviewError):
                f3e.run(_CFG)


if __name__ == "__main__":
    unittest.main()
