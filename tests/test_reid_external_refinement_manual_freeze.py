"""Unit tests for Stage 5D-F3D external refinement manual freeze."""

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

import run_reid_external_refinement_manual_freeze as f3d  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_refinement_manual_freeze_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_manual_freeze"
)
_F3C = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_review_package"
)
_HEAD = "3258df10cda7ec1db2f5f24acec7a8ee4abeeb7b"
_F3C_SNAP = "dc22ef152471eccc157ec634fb8a27d136817f45b59cafd9612402774c518113"


class ExternalRefinementManualFreezeTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3d.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3d.VIEW_ALL), 11)
        self.assertEqual(f3d.VIEW_EXPANSION_YES, (
            "target_001_ext_view_candidate_001",
            "target_001_ext_view_candidate_002",
        ))
        self.assertEqual(len(f3d.VIEW_EXPANSION_NO), 7)
        self.assertEqual(
            f3d.VIEW_AMBIGUOUS,
            (
                "target_001_ext_view_candidate_005",
                "target_001_ext_view_candidate_011",
            ),
        )
        self.assertEqual(f3d.ADDITIONAL_TARGET, ("EXT_161",))
        self.assertEqual(len(f3d.SAME_TEAM_DISTRACTORS), 35)
        self.assertEqual(len(f3d.OTHER_TEAM), 52)
        self.assertEqual(len(f3d.NON_PLAYER), 9)
        self.assertEqual(len(f3d.UNCERTAIN), 4)
        self.assertEqual(len(f3d.INVALID), 5)
        self.assertEqual(len(f3d.AMBIGUOUS), 29)
        self.assertIn("EXT_019", f3d.OTHER_TEAM)
        self.assertIn("EXT_082", f3d.OTHER_TEAM)
        self.assertIn("EXT_226", f3d.OTHER_TEAM)
        self.assertNotIn("EXT_016", f3d.OTHER_TEAM)
        self.assertNotIn("EXT_080", f3d.OTHER_TEAM)
        self.assertNotIn("EXT_228", f3d.OTHER_TEAM)
        self.assertIn("EXT_049", f3d.AMBIGUOUS)
        self.assertIn("EXT_066", f3d.AMBIGUOUS)
        self.assertNotIn("EXT_048", f3d.AMBIGUOUS)
        self.assertNotIn("EXT_065", f3d.AMBIGUOUS)
        self.assertIn("EXT_213", f3d.AMBIGUOUS)

    def test_decision_code_set_coverage_and_distribution(self):
        mapping = f3d.decision_code_set()
        self.assertEqual(len(mapping), 135)
        if _F3C.is_dir():
            inv = {
                json.loads(line)["external_candidate_code"]
                for line in (
                    _F3C
                    / "inventory"
                    / "target_001_external_unreviewed_occurrence_inventory.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            }
            self.assertEqual(inv, set(mapping))
        counts = {}
        for v in mapping.values():
            counts[v] = counts.get(v, 0) + 1
        self.assertEqual(
            counts,
            {
                "additional_target_occurrence_yes": 1,
                "same_team_distractor_yes": 35,
                "other_team_player": 52,
                "non_player": 9,
                "uncertain": 4,
                "invalid": 5,
                "multi_person_ambiguous": 29,
            },
        )

    def test_f3c_package_and_snapshot_validation(self):
        cfg = f3d.load_config(_CFG)
        if not _F3C.is_dir():
            self.skipTest("F3C absent")
        out = f3d.validate_f3c(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3C_SNAP)
        self.assertEqual(len(out["view_inv"]), 11)
        self.assertEqual(len(out["occ_inv"]), 135)

    def test_visible_jersey_and_ext213_special(self):
        self.assertEqual(f3d.DISTRACTOR_JERSEY["EXT_158"], "30")
        self.assertEqual(f3d.DISTRACTOR_JERSEY["EXT_245"], "3")
        self.assertEqual(len(f3d.DISTRACTOR_JERSEY), 9)
        self.assertNotIn("EXT_007", f3d.DISTRACTOR_JERSEY)
        self.assertNotIn("EXT_010", f3d.DISTRACTOR_JERSEY)
        self.assertEqual(len(f3d.TRANSCRIPTION_CORRECTIONS), 5)

    def test_contract_forbids_gallery_and_inference(self):
        c = f3d.build_contract()
        self.assertTrue(c["decisions_frozen"])
        self.assertFalse(c["sample_read"])
        self.assertFalse(c["similarity_used"])
        self.assertFalse(c["automated_ocr_used"])
        self.assertFalse(c["gallery_mutation"])
        self.assertFalse(c["hard_negative_gallery_built"])
        self.assertFalse(c["threshold_selected"])
        self.assertFalse(c["identity_assignment"])
        self.assertEqual(c["gallery_members"], 7)
        self.assertEqual(c["hard_negative_gallery_members"], 0)

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3d.RefinementFreezeError):
            f3d.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3d.RefinementFreezeError):
                f3d.atomic_publish(tmp, final)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3D root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3d_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3d.FINAL_STATUS)
        self.assertEqual(summary["reviewed_target_view"], 11)
        self.assertEqual(summary["target_anchor_expansion_yes"], 2)
        self.assertEqual(summary["target_anchor_expansion_no"], 7)
        self.assertEqual(summary["target_view_multi_person_ambiguous"], 2)
        self.assertEqual(summary["reviewed_external_occurrence"], 135)
        self.assertEqual(summary["additional_target_occurrence_yes"], 1)
        self.assertEqual(summary["same_team_distractor_yes"], 35)
        self.assertEqual(summary["other_team_player"], 52)
        self.assertEqual(summary["non_player"], 9)
        self.assertEqual(summary["uncertain"], 4)
        self.assertEqual(summary["invalid"], 5)
        self.assertEqual(summary["multi_person_ambiguous"], 29)
        self.assertEqual(summary["total_frozen_target_occurrences_after_f3d"], 4)
        self.assertEqual(
            summary["total_human_approved_target_anchor_sources_after_f3d"], 9
        )
        self.assertEqual(summary["official_gallery_v1_members"], 7)
        self.assertEqual(summary["hard_negative_gallery_members"], 0)
        self.assertEqual(summary["new_embeddings"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_crop_read_count"], 0)
        self.assertEqual(summary["transcription_corrections_applied"], 5)
        self.assertEqual(summary["inventory_coverage_missing"], 0)
        self.assertEqual(summary["inventory_coverage_extra"], 0)
        self.assertEqual(
            summary["approved_expansion_candidate_ids"],
            list(f3d.VIEW_EXPANSION_YES),
        )
        self.assertEqual(summary["additional_target_occurrence_code"], "EXT_161")

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_target_view_decisions_frozen.csv"
        ).open(encoding="utf-8") as handle:
            view_rows = list(csv.DictReader(handle))
        self.assertEqual(len(view_rows), 11)
        yes = [
            r["target_view_candidate_id"]
            for r in view_rows
            if r["manual_anchor_expansion_decision"] == "target_anchor_expansion_yes"
        ]
        self.assertEqual(yes, list(f3d.VIEW_EXPANSION_YES))
        for r in view_rows:
            if r["manual_anchor_expansion_decision"] == "target_anchor_expansion_no":
                self.assertEqual(r["identity_negative"], "false")
                self.assertEqual(r["hard_negative_eligible"], "false")
            self.assertEqual(r["current_gallery_member"], "false")

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_occurrence_refinement_decisions_frozen.csv"
        ).open(encoding="utf-8") as handle:
            occ_rows = list(csv.DictReader(handle))
        self.assertEqual(len(occ_rows), 135)
        by = {r["external_candidate_code"]: r for r in occ_rows}
        self.assertEqual(
            by["EXT_161"]["manual_refinement_decision"],
            "additional_target_occurrence_yes",
        )
        self.assertEqual(by["EXT_161"]["manual_visible_jersey_number"], "5")
        self.assertEqual(by["EXT_213"]["manual_same_target_as_target_001"], "yes")
        self.assertEqual(by["EXT_213"]["target_present_but_contaminated"], "true")
        self.assertEqual(by["EXT_213"]["additional_target_occurrence_frozen"], "false")
        self.assertEqual(by["EXT_213"]["hard_negative_eligible"], "false")
        for code in ("EXT_019", "EXT_082", "EXT_226"):
            self.assertEqual(by[code]["manual_refinement_decision"], "other_team_player")
        for code in ("EXT_049", "EXT_066"):
            self.assertEqual(
                by[code]["manual_refinement_decision"], "multi_person_ambiguous"
            )
            self.assertEqual(by[code]["manual_same_target_as_target_001"], "no")
        for code in f3d.UNCERTAIN + f3d.INVALID:
            self.assertEqual(by[code]["hard_negative_eligible"], "false")
            self.assertEqual(by[code]["automatic_negative"], "false")
            self.assertEqual(by[code]["target_negative"], "false")
        for code in f3d.SAME_TEAM_DISTRACTORS:
            self.assertEqual(by[code]["hard_negative_gallery_member"], "false")
            self.assertEqual(by[code]["embedding_input"], "false")
            self.assertEqual(
                by[code]["future_hard_negative_crop_review_source_eligible"], "true"
            )
        for code in ("EXT_007", "EXT_010"):
            self.assertEqual(by[code]["manual_visible_jersey_number"], "")
        for code, num in f3d.DISTRACTOR_JERSEY.items():
            self.assertEqual(by[code]["manual_visible_jersey_number"], num)
            self.assertEqual(
                by[code]["jersey_number_provenance"], f3d.JERSEY_PROVENANCE
            )

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_same_team_distractor_sources_frozen.csv"
        ).open(encoding="utf-8") as handle:
            dist = list(csv.DictReader(handle))
        self.assertEqual(len(dist), 35)
        with (
            _FINAL
            / "manual_freeze"
            / "target_001_external_target_anchor_expansion_sources_frozen.csv"
        ).open(encoding="utf-8") as handle:
            exp = list(csv.DictReader(handle))
        self.assertEqual(len(exp), 2)
        for r in exp:
            self.assertEqual(r["current_gallery_member"], "false")
            self.assertEqual(r["future_target_embedding_source_eligible"], "true")

        freeze = json.loads(
            (
                _FINAL
                / "manual_freeze"
                / "target_001_external_refinement_manual_freeze.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(freeze["total_frozen_target_occurrences_after_f3d"], 4)
        self.assertEqual(
            freeze["total_human_approved_target_anchor_sources_after_f3d"], 9
        )
        self.assertFalse(freeze["gallery_mutation"])
        self.assertFalse(freeze["hard_negative_gallery_built"])
        self.assertEqual(len(freeze["transcription_corrections"]), 5)
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

        runtime = json.loads(
            (_FINAL / "runtime" / "stage5d_f3d_runtime_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(runtime["sample_video_read"])
        self.assertEqual(runtime["sample_score_row_read_count"], 0)
        self.assertFalse(runtime["sample_used_for_decision"])
        self.assertEqual(runtime["new_embeddings"], 0)

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3D root absent")
        with mock.patch.object(f3d, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3d.RefinementFreezeError):
                f3d.run(_CFG)

    def test_upstream_immutability_markers(self):
        cfg = f3d.load_config(_CFG)
        if not _F3C.is_dir():
            self.skipTest("F3C absent")
        up = f3d.validate_upstream(_PROJECT_ROOT, cfg)
        self.assertEqual(up["gallery"]["individual_gallery_members"], 7)
        self.assertEqual(up["b1ec"]["selected_positive_count"], 3)
        self.assertEqual(up["b1ee"]["frozen_approved_anchors"], 7)


if __name__ == "__main__":
    unittest.main()
