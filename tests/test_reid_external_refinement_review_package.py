"""Unit tests for Stage 5D-F3C external refinement review package."""

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

import run_reid_external_refinement_review_package as f3c  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_refinement_review_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_refinement_review_package"
)
_HEAD = "f98215a22b272fbbb6c591a9014b33a35d4f62ee"
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"
_F3B_SNAP = (
    "4e1dbf18dbc95e9fd84b0b6e05cabf3ff0d3fb7358729338a987d68b75f85c54"
)


class ExternalRefinementReviewPackageTests(unittest.TestCase):
    def test_expected_git_contract_constants(self):
        cfg = f3c.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Add target 001 retrieval refinement design",
        )
        self.assertEqual(f3c.FROZEN_CODES, ("EXT_004", "EXT_183", "EXT_198"))
        self.assertEqual(cfg["selection"]["max_candidates_per_occurrence"], 4)
        self.assertEqual(cfg["selection"]["max_total_candidates"], 12)
        self.assertEqual(cfg["selection"]["min_frame_gap"], 12)
        self.assertTrue(cfg["evaluation_source"]["decode_forbidden"])
        self.assertTrue(cfg["gallery_v1"]["npy_load_forbidden"])

    def test_f3b_refinement_design_validation(self):
        cfg = f3c.load_config(_CFG)
        f3b_root = _PROJECT_ROOT / cfg["stage5d_f3b_refinement_design"]["path"]
        if not f3b_root.is_dir():
            self.skipTest("F3B absent")
        out = f3c.validate_f3b(_PROJECT_ROOT, cfg)
        self.assertEqual(
            out["summary"]["final_status"],
            "COMPLETED_STAGE5D_F3B_TARGET_001_REFINEMENT_DESIGN_READY",
        )
        self.assertEqual(out["snapshot_sha256"], _F3B_SNAP)
        self.assertEqual(out["summary"]["gallery_members"], 7)

    def test_external_source_and_tracking_contract(self):
        cfg = f3c.load_config(_CFG)
        ext = f3c.validate_external_source(_PROJECT_ROOT, cfg)
        self.assertEqual(ext["sha256"], _EXT_SHA)
        self.assertEqual(ext["bytes"], 14366504)
        self.assertEqual(ext["frames"], 784)
        b1eb = f3c.validate_b1eb(_PROJECT_ROOT, cfg)
        self.assertEqual(b1eb["summary"]["detection_total"], 8355)
        self.assertEqual(b1eb["summary"]["review_eligible_candidate_count"], 138)
        self.assertTrue(b1eb["summary"]["two_replay_determinism"])
        b1ec = f3c.validate_b1ec(_PROJECT_ROOT, cfg)
        self.assertEqual(b1ec["summary"]["selected_positive_count"], 3)
        unreviewed = f3c.unreviewed_eligible(b1eb["eligible"])
        self.assertEqual(len(unreviewed), 135)
        self.assertTrue(
            all(r["external_candidate_code"] not in f3c.FROZEN_CODES for r in unreviewed)
        )

    def test_frozen_occurrences_and_prior_frames(self):
        cfg = f3c.load_config(_CFG)
        frozen = {
            s["external_candidate_code"]: s for s in cfg["frozen_target_occurrences"]
        }
        self.assertEqual(
            (
                frozen["EXT_004"]["raw_track_id"],
                frozen["EXT_004"]["first_frame"],
                frozen["EXT_004"]["last_frame"],
                frozen["EXT_004"]["observation_count"],
            ),
            (11, 0, 186, 185),
        )
        self.assertEqual(
            (
                frozen["EXT_183"]["raw_track_id"],
                frozen["EXT_183"]["first_frame"],
                frozen["EXT_183"]["last_frame"],
                frozen["EXT_183"]["observation_count"],
            ),
            (388, 456, 499, 44),
        )
        self.assertEqual(
            (
                frozen["EXT_198"]["raw_track_id"],
                frozen["EXT_198"]["first_frame"],
                frozen["EXT_198"]["last_frame"],
                frozen["EXT_198"]["observation_count"],
            ),
            (450, 511, 783, 272),
        )
        prior = cfg["prior_anchor_candidate_frames"]
        self.assertEqual(prior["EXT_004"], [37, 54, 95, 123, 143, 155])
        self.assertEqual(prior["EXT_183"], [463, 475, 493])
        self.assertEqual(prior["EXT_198"], [522, 594, 610, 638, 684, 731])
        self.assertEqual(sum(len(v) for v in prior.values()), 15)

    def test_view_candidate_selection_excludes_prior_and_anchor_dups(self):
        cfg = f3c.load_config(_CFG)
        b1ed = f3c.validate_b1ed_b1ee(_PROJECT_ROOT, cfg)
        selected, audit = f3c.select_view_candidates(
            b1ed["quality_rows"],
            prior_frames=cfg["prior_anchor_candidate_frames"],
            approved_candidates=b1ed["candidates"],
            approved_ids=cfg["approved_anchor_ids"],
            max_per_occ=4,
            max_total=12,
            min_gap=12,
            dup_ham=8,
        )
        self.assertLessEqual(len(selected), 12)
        for code in f3c.FROZEN_CODES:
            n = sum(1 for s in selected if s["source_occurrence_code"] == code)
            self.assertLessEqual(n, 4)
        prior = {
            c: set(cfg["prior_anchor_candidate_frames"][c]) for c in f3c.FROZEN_CODES
        }
        for s in selected:
            self.assertNotIn(
                int(s["frame_index"]), prior[s["source_occurrence_code"]]
            )
        self.assertFalse(audit["f3_item_score_based_selection"])
        self.assertFalse(audit["sample_used_for_candidate_selection"])
        ids = [s["target_view_candidate_id"] for s in selected]
        self.assertEqual(ids, sorted(ids))
        # Deterministic ordering by occurrence then frame.
        frames = [
            (f3c.FROZEN_CODES.index(s["source_occurrence_code"]), int(s["frame_index"]))
            for s in selected
        ]
        self.assertEqual(frames, sorted(frames))

    def test_allowed_decision_vocabularies(self):
        self.assertIn("target_anchor_expansion_yes", f3c.ALLOWED_VIEW_DECISIONS)
        self.assertIn("same_team_distractor_yes", f3c.ALLOWED_OCC_DECISIONS)
        self.assertIn("additional_target_occurrence_yes", f3c.ALLOWED_OCC_DECISIONS)
        self.assertEqual(set(f3c.ALLOWED_SCALE), {"small", "medium", "large", "unknown"})
        contract = f3c.build_contract()
        self.assertTrue(contract["external_only"])
        self.assertFalse(contract["sample_read"])
        self.assertFalse(contract["sample_candidate_use"])
        self.assertTrue(contract["existing_detection_tracking_only"])
        self.assertFalse(contract["automated_team_classification"])
        self.assertFalse(contract["automated_ocr"])
        self.assertFalse(contract["osnet_similarity"])
        self.assertTrue(contract["manual_approval_required"])
        self.assertTrue(contract["candidate_is_not_anchor_until_freeze"])
        self.assertTrue(
            contract["distractor_is_not_hard_negative_member_until_freeze"]
        )
        self.assertFalse(contract["gallery_mutation"])
        self.assertFalse(contract["threshold"])
        self.assertFalse(contract["identity_assignment"])
        self.assertEqual(contract["frozen_target_occurrence_count"], 3)
        self.assertEqual(contract["existing_frozen_anchor_count"], 7)
        self.assertEqual(contract["unreviewed_review_eligible_count"], 135)

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3c.RefinementReviewError):
            f3c.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3c.RefinementReviewError):
                f3c.atomic_publish(tmp, final)

    def test_unreviewed_not_negative_and_blank_templates_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3C root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3c_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3c.FINAL_STATUS)
        self.assertEqual(summary["frozen_target_occurrences"], 3)
        self.assertEqual(summary["existing_frozen_target_anchors"], 7)
        self.assertEqual(summary["unreviewed_review_eligible_external_tracks"], 135)
        self.assertEqual(summary["external_occurrence_manual_decisions"], 0)
        self.assertEqual(summary["new_anchor_approvals"], 0)
        self.assertEqual(summary["hard_negative_approvals"], 0)
        self.assertEqual(summary["new_embeddings"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertEqual(summary["similarity_scoring"], 0)
        self.assertFalse(summary["threshold"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_crop_read_count"], 0)
        self.assertEqual(summary["sample_embedding_read_count"], 0)
        self.assertFalse(summary["sample_used_for_candidate_selection"])
        self.assertEqual(summary["target_view_sheets"], 3)
        self.assertEqual(summary["occurrence_sheets"], 12)
        self.assertEqual(
            summary["occurrence_sheet_item_distribution"], [12] * 11 + [3]
        )
        self.assertLessEqual(summary["target_view_candidate_count"], 12)
        for code in f3c.FROZEN_CODES:
            self.assertLessEqual(
                summary["target_view_candidate_distribution"][code], 4
            )
        self.assertEqual(summary["diagnostic_mp4"], 1)
        self.assertEqual(summary["source_video_copy"], 0)
        self.assertEqual(summary["source_representative_crop_copies"], 0)
        self.assertFalse(summary["automated_team_classification"])
        self.assertFalse(summary["automated_ocr"])
        self.assertFalse(summary["osnet_similarity"])

        view_tpl = (
            _FINAL
            / "templates"
            / "target_001_external_target_view_candidate_review_template.csv"
        )
        occ_tpl = (
            _FINAL
            / "templates"
            / "target_001_external_occurrence_refinement_review_template.csv"
        )
        with view_tpl.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), summary["target_view_candidate_count"])
        self.assertLessEqual(len(rows), 12)
        for row in rows:
            for key in f3c.MANUAL_BLANK_VIEW:
                self.assertEqual(row[key], "")
            self.assertIn(
                row["source_occurrence_code"], f3c.FROZEN_CODES
            )
        with occ_tpl.open(encoding="utf-8") as handle:
            occ_rows = list(csv.DictReader(handle))
        self.assertEqual(len(occ_rows), 135)
        for row in occ_rows:
            for key in f3c.MANUAL_BLANK_OCC:
                self.assertEqual(row[key], "")
            self.assertNotIn(row["external_candidate_code"], f3c.FROZEN_CODES)

        runtime = json.loads(
            (_FINAL / "runtime" / "stage5d_f3c_runtime_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(runtime["sample_video_read"])
        self.assertEqual(runtime["sample_crop_read_count"], 0)
        self.assertEqual(runtime["sample_embedding_read_count"], 0)
        self.assertFalse(runtime["sample_used_for_candidate_selection"])
        self.assertFalse(runtime["f3_item_score_based_selection"])
        self.assertFalse(runtime["new_detection"])
        self.assertFalse(runtime["new_tracking"])
        self.assertEqual(runtime["new_embeddings"], 0)

        review_pkg = (
            _FINAL / "review_packages" / "target_001_external_refinement_review"
        )
        self.assertEqual(
            len(list(review_pkg.glob("target_view_candidates_EXT_*.png"))), 3
        )
        self.assertEqual(
            len(list(review_pkg.glob("external_occurrence_review_sheet_*.png"))), 12
        )
        mp4 = (
            _FINAL
            / "videos"
            / "target_001_external_unreviewed_occurrence_review.mp4"
        )
        self.assertTrue(mp4.is_file())
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.npz")), [])
        self.assertEqual(list(_FINAL.rglob("sample.mp4")), [])
        # No auto approvals / identity.
        contract = json.loads(
            (_FINAL / "stage5d_f3c_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["hard_negative_approvals"], 0)
        self.assertEqual(contract["new_anchor_approvals"], 0)
        self.assertFalse(contract["identity_assignment"])
        # Selection audit excludes prior frames.
        audit = json.loads(
            (
                _FINAL
                / "quality"
                / "target_001_external_target_view_candidate_selection_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(audit["f3_item_score_based_selection"])
        inv = [
            json.loads(line)
            for line in (
                _FINAL
                / "inventory"
                / "target_001_external_target_view_candidate_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        cfg = f3c.load_config(_CFG)
        prior = {
            c: set(cfg["prior_anchor_candidate_frames"][c]) for c in f3c.FROZEN_CODES
        }
        for item in inv:
            self.assertFalse(item["is_anchor"])
            self.assertFalse(item["is_gallery_member"])
            self.assertFalse(item["embedding_input"])
            self.assertFalse(item["f3_sample_score_used"])
            self.assertNotIn(item["frame_index"], prior[item["source_occurrence_code"]])

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3C root absent")
        with mock.patch.object(f3c, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3c.RefinementReviewError):
                f3c.run(_CFG)

    def test_source_upstream_immutability_markers(self):
        cfg = f3c.load_config(_CFG)
        # Upstream packages must remain readable and unchanged in counts.
        b1eb = f3c.validate_b1eb(_PROJECT_ROOT, cfg)
        self.assertEqual(b1eb["summary"]["raw_track_count"], 248)
        b1ed = f3c.validate_b1ed_b1ee(_PROJECT_ROOT, cfg)
        self.assertEqual(len(b1ed["quality_rows"]), 501)
        self.assertEqual(len(b1ed["approved_ids"]), 7)
        gallery = f3c.validate_gallery_v1(_PROJECT_ROOT, cfg)
        self.assertEqual(gallery["members"], 7)


if __name__ == "__main__":
    unittest.main()
