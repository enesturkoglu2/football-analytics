"""Unit tests for Stage 5D-F3F external refinement crop manual freeze."""

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

import run_reid_external_refinement_crop_manual_freeze as f3f  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_refinement_crop_manual_freeze_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_crop_manual_freeze"
)
_F3E = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_crop_review_package"
)
_HEAD = "f986c7e34a4c3272a7fdfd892454d5d87780e98a"
_F3E_SNAP = "c86a5f23c80bc9e4e4dc3527babe25faf692c561ec247f0f265eee211a94b78a"


class ExternalRefinementCropManualFreezeTests(unittest.TestCase):
    def test_expected_git_and_exact_sets(self):
        cfg = f3f.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3f.TARGET_APPROVED), 4)
        self.assertEqual(len(f3f.HN_YES), 23)
        self.assertEqual(len(f3f.HN_NO_WRONG_TEAM), 4)
        self.assertEqual(len(f3f.HN_INVALID), 5)
        self.assertEqual(len(f3f.HN_AMBIGUOUS), 3)
        all_seq = set(f3f.HN_YES) | set(f3f.HN_NO_WRONG_TEAM) | set(f3f.HN_INVALID) | set(
            f3f.HN_AMBIGUOUS
        )
        self.assertEqual(all_seq, set(range(1, 36)))

    def test_hn_decision_map_distribution(self):
        m = f3f.hn_decision_map()
        self.assertEqual(len(m), 35)
        dist = {}
        for v in m.values():
            d = v["manual_hard_negative_crop_decision"]
            dist[d] = dist.get(d, 0) + 1
        self.assertEqual(
            dist,
            {
                "hard_negative_crop_yes": 23,
                "hard_negative_crop_no": 4,
                "invalid": 5,
                "multi_person_ambiguous": 3,
            },
        )
        self.assertEqual(
            m[f3f.hn_id(4)]["decision_reason"], f3f.REASON_WRONG_TEAM
        )
        self.assertEqual(m[f3f.hn_id(1)]["decision_reason"], f3f.REASON_INVALID)
        self.assertEqual(m[f3f.hn_id(3)]["decision_reason"], f3f.REASON_AMBIGUOUS)

    def test_f3e_validation_if_present(self):
        cfg = f3f.load_config(_CFG)
        if not _F3E.is_dir():
            self.skipTest("F3E absent")
        out = f3f.validate_f3e(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3E_SNAP)
        self.assertEqual(len(out["target_inv"]), 4)
        self.assertEqual(len(out["hn_inv"]), 35)

    def test_contract_forbids_gallery_and_inference(self):
        c = f3f.build_contract()
        self.assertTrue(c["decisions_frozen"])
        self.assertFalse(c["sample_read"])
        self.assertFalse(c["gallery_mutation"])
        self.assertFalse(c["hard_negative_gallery_built"])
        self.assertFalse(c["threshold_selected"])
        self.assertFalse(c["identity_assignment"])
        self.assertEqual(c["gallery_members"], 7)
        self.assertEqual(c["approved_new_target_crops"], 4)
        self.assertEqual(c["approved_hard_negative_crops"], 23)

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3f.CropFreezeError):
            f3f.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3f.CropFreezeError):
                f3f.atomic_publish(tmp, final)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3F root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3f_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3f.FINAL_STATUS)
        self.assertEqual(summary["reviewed_target_crop_candidates"], 4)
        self.assertEqual(summary["approved_new_target_crops"], 4)
        self.assertEqual(summary["approved_target_crop_ids"], list(f3f.TARGET_APPROVED))
        self.assertEqual(summary["reviewed_hard_negative_crop_candidates"], 35)
        self.assertEqual(summary["approved_hard_negative_crops"], 23)
        self.assertEqual(summary["hard_negative_crop_no_wrong_team"], 4)
        self.assertEqual(summary["invalid"], 5)
        self.assertEqual(summary["multi_person_ambiguous"], 3)
        self.assertEqual(
            summary["approved_hard_negative_crop_ids"],
            [f3f.hn_id(i) for i in f3f.HN_YES],
        )
        self.assertEqual(
            summary["wrong_team_rejected_ids"],
            [f3f.hn_id(i) for i in f3f.HN_NO_WRONG_TEAM],
        )
        self.assertEqual(summary["invalid_ids"], [f3f.hn_id(i) for i in f3f.HN_INVALID])
        self.assertEqual(
            summary["multi_person_ambiguous_ids"],
            [f3f.hn_id(i) for i in f3f.HN_AMBIGUOUS],
        )
        self.assertEqual(summary["official_gallery_v1_members"], 7)
        self.assertEqual(summary["hard_negative_gallery_members"], 0)
        self.assertEqual(summary["target_crops_current_gallery_members"], 0)
        self.assertEqual(summary["new_embeddings"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_crop_read_count"], 0)

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_EXT_161_target_crop_decisions_frozen.csv"
        ).open(encoding="utf-8") as handle:
            trows = list(csv.DictReader(handle))
        self.assertEqual(len(trows), 4)
        for row in trows:
            self.assertEqual(row["manual_target_crop_decision"], "target_crop_yes")
            self.assertEqual(row["current_gallery_member"], "false")
            self.assertEqual(row["embedding_input"], "false")

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_hard_negative_crop_decisions_frozen.csv"
        ).open(encoding="utf-8") as handle:
            hrows = list(csv.DictReader(handle))
        self.assertEqual(len(hrows), 35)
        by = {r["hard_negative_candidate_id"]: r for r in hrows}
        self.assertEqual(
            by[f3f.hn_id(2)]["manual_hard_negative_crop_decision"],
            "hard_negative_crop_yes",
        )
        self.assertEqual(
            by[f3f.hn_id(4)]["manual_hard_negative_crop_decision"],
            "hard_negative_crop_no",
        )
        self.assertEqual(by[f3f.hn_id(4)]["manual_same_team_confirmed"], "no")
        self.assertEqual(by[f3f.hn_id(4)]["decision_reason"], f3f.REASON_WRONG_TEAM)
        self.assertEqual(by[f3f.hn_id(1)]["manual_hard_negative_crop_decision"], "invalid")
        self.assertEqual(
            by[f3f.hn_id(3)]["manual_hard_negative_crop_decision"],
            "multi_person_ambiguous",
        )
        for row in hrows:
            self.assertEqual(row["hard_negative_gallery_member"], "false")
            self.assertEqual(row["embedding_input"], "false")

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_approved_hard_negative_crops_frozen.csv"
        ).open(encoding="utf-8") as handle:
            approved = list(csv.DictReader(handle))
        self.assertEqual(len(approved), 23)

        freeze = json.loads(
            (
                _FINAL
                / "manual_freeze"
                / "target_001_external_refinement_crop_manual_freeze.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(freeze["decisions_frozen"])
        self.assertFalse(freeze["gallery_mutation"])
        self.assertFalse(freeze["hard_negative_gallery_built"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

        runtime = json.loads(
            (_FINAL / "runtime" / "stage5d_f3f_runtime_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(runtime["sample_video_read"])
        self.assertEqual(runtime["new_embeddings"], 0)

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3F root absent")
        with mock.patch.object(f3f, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3f.CropFreezeError):
                f3f.run(_CFG)


if __name__ == "__main__":
    unittest.main()
