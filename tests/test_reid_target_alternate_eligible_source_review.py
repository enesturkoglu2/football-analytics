"""Unit tests for Stage 5D-B1C alternate eligible anchor source review."""

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

import run_reid_target_alternate_eligible_source_review as b1c  # noqa: E402


class AlternateEligibleSourceReviewTests(unittest.TestCase):
    def test_expected_git_contract_fields(self):
        cfg = b1c.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "d1c7aded068f4d2a5a9dc32e20b8ef4a25873b6e",
        )
        self.assertEqual(cfg["review_window"]["start_frame"], 30)
        self.assertEqual(cfg["review_window"]["end_frame"], 75)
        self.assertEqual(
            cfg["review_window"]["sheet_target_frames"],
            [30, 36, 42, 48, 54, 60, 66, 75],
        )
        self.assertEqual(
            cfg["frozen_seed_forbidden"]["selected_neutral_seed_code"],
            "SEED_CANDIDATE_07",
        )
        self.assertEqual(cfg["frozen_seed_forbidden"]["segment_id"], "raw_222_full")
        self.assertEqual(int(cfg["frozen_seed_forbidden"]["raw_track_id"]), 222)
        self.assertEqual(
            cfg["source_video"]["expected_sha256"],
            "f4b28dd58a6cf242344a4198b8c0ba9062b20977cec3ae12d96322750bfd7b9b",
        )

    def test_frozen_seed_constant_and_contract(self):
        self.assertEqual(b1c.FROZEN_SEED_CODE, "SEED_CANDIDATE_07")
        contract = b1c.build_contract()
        self.assertTrue(contract["frozen_original_seed_preserved"])
        self.assertTrue(
            contract["original_excluded_seed_not_used_for_scoring_or_enrollment"]
        )
        self.assertTrue(contract["existing_eligible_observations_only"])
        self.assertTrue(contract["no_new_detection"])
        self.assertTrue(contract["no_new_tracking"])
        self.assertTrue(contract["no_new_embedding"])
        self.assertTrue(contract["no_ocr"])
        self.assertTrue(contract["no_similarity"])
        self.assertTrue(contract["no_automatic_identity"])
        self.assertTrue(contract["human_selection_required"])
        self.assertTrue(
            contract["alternative_selection_requires_separate_freeze_gate"]
        )
        self.assertTrue(contract["no_gallery_membership"])
        self.assertTrue(contract["unknown_identity_preserved"])
        self.assertEqual(contract["alternate_manual_selection"], 0)
        self.assertEqual(contract["derived_anchors"], 0)
        self.assertEqual(contract["gallery_members"], 0)
        self.assertEqual(contract["prototypes"], 0)
        self.assertEqual(contract["identity_assignments"], 0)

    def test_exact_original_seed_exclusion(self):
        exclusion = {
            "keys": {
                "segment_id": set(),
                "raw_track_id": set(),
                "crop_id": set(),
                "source_crop_sha256": set(),
                "near_duplicate_component": set(),
                "exact_duplicate_group": set(),
                "documented_link_component": set(),
                "temporal_source_window": set(),
                "frame_identity": set(),
            },
            "frozen_component": {
                "segment_id": {"raw_222_full"},
                "raw_track_id": {"222"},
                "crop_id": set(),
                "source_crop_sha256": set(),
                "exact_duplicate_group": set(),
                "near_duplicate_component": set(),
                "documented_link_component": set(),
                "temporal_source_window": set(),
            },
            "universe_by_seg": {},
        }
        reasons = b1c.chain_exclusion_reasons(
            raw_track_id=222,
            segment_id="raw_222_full",
            emb={"embedding_available": True, "crop_ids": ["c1"]},
            crop_catalog={"c1": _PROJECT_ROOT / "README.md"},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("original_frozen_seed_exclusion", reasons)

    def test_stage5c_and_documented_link_exclusion(self):
        exclusion = {
            "keys": {
                "segment_id": {"raw_1_full"},
                "raw_track_id": {"1"},
                "crop_id": set(),
                "source_crop_sha256": set(),
                "near_duplicate_component": set(),
                "exact_duplicate_group": set(),
                "documented_link_component": {"4"},
                "temporal_source_window": set(),
                "frame_identity": set(),
            },
            "frozen_component": {
                "segment_id": {"raw_222_full"},
                "raw_track_id": {"222"},
                "crop_id": set(),
                "source_crop_sha256": set(),
                "exact_duplicate_group": set(),
                "near_duplicate_component": set(),
                "documented_link_component": {"222"},
                "temporal_source_window": set(),
            },
            "universe_by_seg": {
                "raw_4_full": [{"documented_global_candidate_id": 4}],
                "raw_1_full": [{"documented_global_candidate_id": 1}],
            },
        }
        r1 = b1c.chain_exclusion_reasons(
            raw_track_id=1,
            segment_id="raw_1_full",
            emb={"embedding_available": True, "crop_ids": []},
            crop_catalog={},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("stage5c_membership", r1)

        r4 = b1c.chain_exclusion_reasons(
            raw_track_id=4,
            segment_id="raw_4_full",
            emb={"embedding_available": True, "crop_ids": ["c4"]},
            crop_catalog={},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("stage5d_documented_link_component", r4)
        self.assertNotIn("original_frozen_seed_exclusion", r4)

    def test_existing_embedding_eligibility_gate(self):
        exclusion = {
            "keys": {
                "segment_id": set(),
                "raw_track_id": set(),
                "crop_id": set(),
                "source_crop_sha256": set(),
                "near_duplicate_component": set(),
                "exact_duplicate_group": set(),
                "documented_link_component": set(),
                "temporal_source_window": set(),
                "frame_identity": set(),
            },
            "frozen_component": {
                "segment_id": set(),
                "raw_track_id": set(),
                "crop_id": set(),
                "source_crop_sha256": set(),
                "exact_duplicate_group": set(),
                "near_duplicate_component": set(),
                "documented_link_component": set(),
                "temporal_source_window": set(),
            },
            "universe_by_seg": {},
        }
        reasons = b1c.chain_exclusion_reasons(
            raw_track_id=99,
            segment_id="raw_99_full",
            emb={"embedding_available": False},
            crop_catalog={},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("no_existing_embedding", reasons)

    def test_stable_neutral_codes(self):
        chains = {
            (10, "raw_10_full"): [
                {"frame_index": 40},
                {"frame_index": 50},
            ],
            (5, "raw_5_full"): [{"frame_index": 30}],
            (10, "raw_10_s02"): [{"frame_index": 60}],
        }
        keys = list(chains.keys())
        a = b1c.assign_alt_codes(keys, chains)
        b = b1c.assign_alt_codes(keys, chains)
        self.assertEqual(a, b)
        self.assertEqual(a[(5, "raw_5_full")], "ALT_SEED_CANDIDATE_01")
        self.assertTrue(all(v.startswith("ALT_SEED_CANDIDATE_") for v in a.values()))
        self.assertNotEqual(a[(10, "raw_10_full")], a[(10, "raw_10_s02")])

    def test_sheet_frames_deterministic_no_dupes(self):
        selected = b1c.select_sheet_frames(
            [30, 36, 42, 48, 54, 60, 66, 75],
            [30, 42, 75],
        )
        self.assertEqual(selected, [30, 42, 75])
        self.assertEqual(len(selected), len(set(selected)))

    def test_blank_manual_template_fields(self):
        self.assertEqual(
            set(b1c.TEMPLATE_FIELDS),
            {
                "target_id",
                "review_window_start_frame",
                "review_window_end_frame",
                "selected_alternate_neutral_seed_code",
                "manual_target_confirmed",
                "manual_human_verified_number_seen",
                "manual_crop_valid",
                "manual_target_dominant",
                "manual_identity_continuity_observed",
                "manual_notes",
                "reviewer",
                "final_approver",
                "reviewed_at",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(b1c.TEMPLATE_FIELDS))
                writer.writeheader()
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "review_window_start_frame": 30,
                        "review_window_end_frame": 75,
                        "selected_alternate_neutral_seed_code": "",
                        "manual_target_confirmed": "",
                        "manual_human_verified_number_seen": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_identity_continuity_observed": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["selected_alternate_neutral_seed_code"], "")
            self.assertNotIn("similarity_score", row)
            self.assertNotIn("ocr_prediction", row)
            self.assertNotIn("raw_track_id", row)
            self.assertNotIn("segment_id", row)

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1c.AlternateSourceError):
            b1c.assert_no_path_traversal("../x")
        with self.assertRaises(b1c.AlternateSourceError):
            b1c.assert_no_path_traversal("/abs/path")

    def test_tristate_vocabulary(self):
        self.assertEqual(set(b1c.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})

    def test_no_automatic_alternate_selection_in_contract(self):
        c = b1c.build_contract()
        self.assertEqual(c["alternate_manual_selection"], 0)
        self.assertTrue(c["human_selection_required"])
        self.assertTrue(c["alternative_selection_requires_separate_freeze_gate"])

    def test_output_summary_counts_zero_gallery(self):
        # Guardrail constants used by summary writer expectations.
        self.assertEqual(b1c.build_contract()["gallery_members"], 0)
        self.assertEqual(b1c.build_contract()["derived_anchors"], 0)
        self.assertEqual(b1c.build_contract()["approved_anchors"], 0)
        self.assertEqual(b1c.build_contract()["identity_assignments"], 0)

    def test_live_frozen_seed_package_contract_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_seed_freeze_anchor_derivation"
        )
        if not root.is_dir():
            self.skipTest("B1B output root absent")
        summary = json.loads((root / "stage5d_b1b_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1B_SEED_FROZEN_SOURCE_INELIGIBLE",
        )
        self.assertEqual(summary["selected_neutral_seed_code"], "SEED_CANDIDATE_07")
        self.assertEqual(summary["resolved_segment_id"], "raw_222_full")
        self.assertEqual(int(summary["resolved_raw_track_id"]), 222)
        self.assertEqual(
            summary["selected_seed_source_eligibility"],
            "frozen_identity_seed_only_stage5c_excluded",
        )
        self.assertEqual(int(summary.get("derived_anchor_candidate_count") or 0), 0)
        self.assertEqual(int(summary.get("gallery_members") or 0), 0)

    def test_live_b1c_output_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_alternate_eligible_source_review"
        )
        if not root.is_dir():
            self.skipTest("B1C output root absent")
        summary = json.loads((root / "stage5d_b1c_summary.json").read_text(encoding="utf-8"))
        self.assertIn(
            summary["final_status"],
            {
                "COMPLETED_STAGE5D_B1C_ALTERNATE_ELIGIBLE_SOURCE_REVIEW_READY",
                "COMPLETED_STAGE5D_B1C_NO_ELIGIBLE_SOURCE_IN_EARLY_WINDOW",
            },
        )
        self.assertEqual(summary["original_frozen_seed_code"], "SEED_CANDIDATE_07")
        self.assertFalse(summary["original_seed_embedding_used"])
        self.assertEqual(summary["review_window_frames"], [30, 75])
        self.assertEqual(summary["alternate_manual_selection"], 0)
        self.assertEqual(summary["derived_anchors"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["similarity_ranking_rows"], 0)
        self.assertEqual(summary["new_detection"], 0)
        self.assertEqual(summary["new_tracking"], 0)
        self.assertEqual(summary["new_embedding"], 0)
        self.assertEqual(summary["ocr"], 0)
        png = list(root.rglob("*.png"))
        mp4 = list(root.rglob("*.mp4"))
        if summary["final_status"].endswith("REVIEW_READY"):
            self.assertEqual(len(png), 1)
            self.assertEqual(len(mp4), 1)
            self.assertGreater(summary["eligible_candidate_count"], 0)
        else:
            self.assertEqual(len(png), 0)
            self.assertEqual(len(mp4), 0)
            self.assertEqual(summary["eligible_candidate_count"], 0)
            self.assertEqual(
                summary["exact_next_gate"],
                "STAGE5D-B1C2_TARGET_001_ADDITIONAL_HUMAN_WINDOW_REVIEW",
            )
        tpl = (
            root
            / "templates"
            / "target_001_alternate_eligible_source_review_template.csv"
        )
        with tpl.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["selected_alternate_neutral_seed_code"], "")
        mapping = root / "inventory" / "target_001_alternate_eligible_source_mapping.jsonl"
        lines = [ln for ln in mapping.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), summary["neutral_alternate_candidate_count"])
        for line in lines:
            row = json.loads(line)
            self.assertTrue(row["alternate_neutral_seed_code"].startswith("ALT_SEED_CANDIDATE_"))
            self.assertNotEqual(row["segment_id"], "raw_222_full")
            self.assertNotEqual(int(row["raw_track_id"]), 222)
            self.assertTrue(row["eligible_for_anchor_derivation"])
            self.assertEqual(row["manual_target_confirmed"], "")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1c.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1c, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1c.AlternateSourceError) as ctx:
                    b1c.run(
                        _PROJECT_ROOT
                        / "configs/reid/target_alternate_eligible_source_review_stage5d_target_001.yaml",
                        project,
                    )
            self.assertIn("final_exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
