"""Unit tests for Stage 5D-F3L holdout v2 ground-truth manual freeze."""

from __future__ import annotations

import csv
import hashlib
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

import run_reid_independent_holdout_v2_ground_truth_manual_freeze as f3l  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_ground_truth_manual_freeze_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ground_truth_manual_freeze"
)
_F3K = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ground_truth_review_package"
)
_F3J = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
)
_HEAD = "d2c93831f6282cf984352ad9bf1f5e9e7c28d7e4"
_F3K_SNAP = "63af341120a9e2d6003cce89c9b14fecb68ca300f2d5fc3fa18debc2f02cfa2b"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HoldoutGroundTruthManualFreezeTests(unittest.TestCase):
    def test_expected_git_and_decision_set_sizes(self):
        cfg = f3l.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Add target 001 holdout v2 ground truth review package",
        )
        f3l.validate_decision_sets()
        self.assertEqual(len(f3l.POSITIVE_IDS), 10)
        self.assertEqual(len(f3l.SAME_TEAM_NEGATIVE_IDS), 55)
        self.assertEqual(len(f3l.OTHER_TEAM_NEGATIVE_IDS), 50)
        self.assertEqual(len(f3l.NON_PLAYER_IDS), 5)
        self.assertEqual(len(f3l.INVALID_IDS), 7)
        self.assertEqual(len(f3l.AMBIGUOUS_IDS), 14)

    def test_exact_positive_ids(self):
        self.assertEqual(
            f3l.POSITIVE_IDS,
            (
                "H2_GT_REVIEW_000010",
                "H2_GT_REVIEW_000030",
                "H2_GT_REVIEW_000061",
                "H2_GT_REVIEW_000090",
                "H2_GT_REVIEW_000094",
                "H2_GT_REVIEW_000104",
                "H2_GT_REVIEW_000124",
                "H2_GT_REVIEW_000129",
                "H2_GT_REVIEW_000135",
                "H2_GT_REVIEW_000136",
            ),
        )

    def test_decision_templates(self):
        pos = f3l.build_decision_for_review_id("H2_GT_REVIEW_000010")
        self.assertEqual(pos["manual_ground_truth_decision"], "target_occurrence_yes")
        self.assertEqual(pos["manual_visible_jersey_number"], "5")
        self.assertEqual(pos["jersey_number_provenance"], "human_visual_review_by_Furkan")
        self.assertTrue(pos["clean_positive"])
        self.assertTrue(pos["query_score_eligibility"])

        stn = f3l.build_decision_for_review_id("H2_GT_REVIEW_000003")
        self.assertEqual(stn["manual_ground_truth_decision"], "target_occurrence_no")
        self.assertEqual(stn["manual_same_team_as_target"], "yes")
        self.assertTrue(stn["clean_same_team_negative"])

        otn = f3l.build_decision_for_review_id("H2_GT_REVIEW_000001")
        self.assertEqual(otn["manual_same_team_as_target"], "no")
        self.assertFalse(otn["clean_same_team_negative"])

        np_row = f3l.build_decision_for_review_id("H2_GT_REVIEW_000004")
        self.assertEqual(np_row["manual_ground_truth_decision"], "non_player")
        self.assertFalse(np_row["query_score_eligibility"])
        self.assertEqual(
            np_row["exclusion_from_reid_query_reason"], "human_reviewed_non_player"
        )

        inv = f3l.build_decision_for_review_id("H2_GT_REVIEW_000062")
        self.assertEqual(inv["manual_ground_truth_decision"], "invalid")
        self.assertEqual(inv["manual_crop_valid"], "no")
        self.assertFalse(inv["metric_inclusion"])

        amb = f3l.build_decision_for_review_id("H2_GT_REVIEW_000035")
        self.assertEqual(amb["manual_ground_truth_decision"], "multi_person_ambiguous")
        self.assertEqual(amb["manual_single_person"], "no")
        self.assertEqual(amb["manual_track_impurity_observed"], "yes")
        self.assertFalse(amb["metric_inclusion"])

    def test_invalid_and_ambiguous_not_clean_negative(self):
        for rid in f3l.INVALID_IDS:
            d = f3l.build_decision_for_review_id(rid)
            self.assertFalse(d["clean_negative"])
            self.assertFalse(d["clean_positive"])
            self.assertFalse(d["metric_inclusion"])
        for rid in f3l.AMBIGUOUS_IDS:
            d = f3l.build_decision_for_review_id(rid)
            self.assertFalse(d["clean_negative"])
            self.assertFalse(d["clean_positive"])
            self.assertFalse(d["metric_inclusion"])

    def test_component_policy_one_segment_per_component(self):
        self.assertEqual(
            f3l.reviewed_component_id("H2_GT_REVIEW_000010"),
            "H2_GT_COMPONENT_000010",
        )
        self.assertEqual(
            f3l.ineligible_component_id("H2_SEG_000017"),
            "H2_GT_COMPONENT_INELIG_000017",
        )
        self.assertNotEqual(
            f3l.reviewed_component_id("H2_GT_REVIEW_000010"),
            f3l.ineligible_component_id("H2_SEG_000010"),
        )
        self.assertEqual(
            f3l.COMPONENT_POLICY,
            "ONE_FROZEN_SEGMENT_PER_COMPONENT_NO_CROSS_TRACK_LINK_EVIDENCE",
        )

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3l.GroundTruthFreezeError):
            f3l.assert_no_path_traversal("../x")

    def test_access_audit_zeros(self):
        audit = f3l.build_access_audit()
        self.assertFalse(audit["sample_video_read"])
        self.assertFalse(audit["external_video_read"])
        self.assertFalse(audit["holdout_video_read"])
        self.assertEqual(audit["gallery_embedding_read_count"], 0)
        self.assertEqual(audit["similarity_row_read_count"], 0)
        self.assertEqual(audit["metric_computation_count"], 0)
        self.assertEqual(audit["crop_png_bytes_read"], 0)
        self.assertEqual(audit["contact_sheet_bytes_read"], 0)
        self.assertEqual(audit["video_bytes_read"], 0)

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = f3l.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(f3l, "assert_git_contract", return_value="deadbeef"):
                with mock.patch.object(f3l, "validate_decision_sets"):
                    with self.assertRaises(f3l.GroundTruthFreezeError) as ctx:
                        f3l.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_f3k_package_if_present(self):
        if not _F3K.is_dir():
            self.skipTest("F3K absent")
        cfg = f3l.load_config(_CFG)
        out = f3l.validate_f3k(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3K_SNAP)
        self.assertEqual(len(out["inventory"]), 141)
        self.assertEqual(len(out["ineligible"]), 102)
        summary = out["summary"]
        self.assertEqual(summary["manual_ground_truth_decisions"], 0)
        self.assertEqual(summary["review_item_count"], 141)

    def test_f3j_summary_if_present(self):
        if not _F3J.is_dir():
            self.skipTest("F3J absent")
        cfg = f3l.load_config(_CFG)
        out = f3l.validate_f3j(_PROJECT_ROOT, cfg)
        self.assertEqual(out["summary"]["segment_count"], 243)
        self.assertEqual(out["summary"]["raw_track_count"], 243)
        self.assertEqual(out["summary"]["pass_through_segment_count"], 243)
        self.assertEqual(out["summary"]["split_segment_count"], 0)

    def test_live_freeze_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3L freeze absent")
        summary = json.loads((_FINAL / "stage5d_f3l_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_status"], f3l.STATUS)
        self.assertEqual(summary["readiness"], f3l.READINESS)
        self.assertEqual(summary["exact_next_gate"], f3l.NEXT_GATE)
        self.assertEqual(summary["reviewed_total"], 141)
        self.assertEqual(summary["target_occurrence_yes"], 10)
        self.assertEqual(summary["target_occurrence_no"], 105)
        self.assertEqual(summary["same_team_target_occurrence_no"], 55)
        self.assertEqual(summary["other_team_target_occurrence_no"], 50)
        self.assertEqual(summary["non_player"], 5)
        self.assertEqual(summary["invalid"], 7)
        self.assertEqual(summary["multi_person_ambiguous"], 14)
        self.assertEqual(summary["uncertain"], 0)
        self.assertEqual(summary["clean_positive_segments"], 10)
        self.assertEqual(summary["clean_negative_segments"], 110)
        self.assertEqual(summary["clean_same_team_negative_segments"], 55)
        self.assertEqual(summary["reviewed_metric_excluded"], 21)
        self.assertEqual(summary["unreviewed_ineligible"], 102)
        self.assertEqual(summary["complete_universe"], 243)
        self.assertEqual(summary["component_conflict_count"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_vectors_read"])
        self.assertEqual(list(summary["positive_exact_ids"]), list(f3l.POSITIVE_IDS))

        with (
            _FINAL
            / "manual_freeze"
            / "target_001_holdout_v2_ground_truth_decisions_frozen.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 141)
        ids = [r["review_item_id"] for r in rows]
        self.assertEqual(ids, [f"H2_GT_REVIEW_{i:06d}" for i in range(1, 142)])
        by = {r["review_item_id"]: r for r in rows}
        self.assertEqual(by["H2_GT_REVIEW_000010"]["manual_visible_jersey_number"], "5")
        self.assertEqual(by["H2_GT_REVIEW_000003"]["manual_same_team_as_target"], "yes")
        self.assertEqual(by["H2_GT_REVIEW_000004"]["manual_ground_truth_decision"], "non_player")
        self.assertEqual(by["H2_GT_REVIEW_000062"]["manual_ground_truth_decision"], "invalid")
        self.assertEqual(
            by["H2_GT_REVIEW_000035"]["manual_ground_truth_decision"], "multi_person_ambiguous"
        )

        pos_lines = (
            _FINAL / "manual_freeze" / "target_001_holdout_v2_clean_positive_inventory.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        neg_lines = (
            _FINAL / "manual_freeze" / "target_001_holdout_v2_clean_negative_inventory.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        stn_lines = (
            _FINAL
            / "manual_freeze"
            / "target_001_holdout_v2_clean_same_team_negative_inventory.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        excl_lines = (
            _FINAL
            / "manual_freeze"
            / "target_001_holdout_v2_reviewed_metric_exclusion_inventory.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(pos_lines), 10)
        self.assertEqual(len(neg_lines), 110)
        self.assertEqual(len(stn_lines), 55)
        self.assertEqual(len(excl_lines), 21)

        comp_lines = (
            _FINAL / "components" / "target_001_holdout_v2_ground_truth_component_mapping.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(comp_lines), 243)

        coverage = json.loads(
            (
                _FINAL
                / "manual_freeze"
                / "target_001_holdout_v2_complete_segment_ground_truth_coverage.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(coverage["reviewed_eligible"], 141)
        self.assertEqual(coverage["ineligible"], 102)
        self.assertEqual(coverage["complete"], 243)
        self.assertEqual(coverage["silent_drop"], 0)
        self.assertEqual(coverage["duplicate"], 0)

        audit = json.loads(
            (_FINAL / "runtime" / "target_001_f3l_access_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit["crop_png_bytes_read"], 0)
        self.assertEqual(audit["video_bytes_read"], 0)
        self.assertFalse(audit["holdout_video_read"])

        freeze = json.loads(
            (
                _FINAL
                / "manual_freeze"
                / "target_001_holdout_v2_ground_truth_decisions_frozen.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(freeze["schema_version"], "reid_independent_holdout_ground_truth_freeze_v1")
        self.assertTrue(freeze["manual_decisions_frozen"])
        self.assertFalse(freeze["gallery_vectors_read"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
