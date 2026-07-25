"""Unit tests for Stage 5D-B1C2 seed-to-eligible bridge review helpers."""

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

import run_reid_target_seed_to_eligible_bridge_review as b1c2  # noqa: E402


class SeedToEligibleBridgeReviewTests(unittest.TestCase):
    def test_expected_git_and_bridge_window(self):
        cfg = b1c2.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "5112e2912b7272fc2c876a26d59d601faf462d28",
        )
        self.assertEqual(cfg["bridge_window"]["start_frame"], 280)
        self.assertEqual(cfg["bridge_window"]["end_frame"], 420)
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

    def test_frozen_visual_only_contract(self):
        self.assertEqual(b1c2.FROZEN_SEED_CODE, "SEED_CANDIDATE_07")
        self.assertEqual(b1c2.FROZEN_VISUAL_LABEL, "FROZEN_HUMAN_SEED_REF")
        self.assertIn("NOT ENROLLABLE", b1c2.FROZEN_VISUAL_WARNING)
        c = b1c2.build_contract()
        self.assertTrue(c["frozen_seed_used_for_human_visual_continuity_only"])
        self.assertFalse(c["frozen_seed_embedding_used"])
        self.assertFalse(c["frozen_seed_enrollment_allowed"])
        self.assertTrue(c["no_new_detection"])
        self.assertTrue(c["no_new_tracking"])
        self.assertTrue(c["no_new_embedding"])
        self.assertTrue(c["no_ocr"])
        self.assertTrue(c["no_similarity"])
        self.assertTrue(c["no_automatic_continuation_assignment"])
        self.assertTrue(c["human_decision_required"])
        self.assertTrue(c["no_anchor_derivation"])
        self.assertTrue(c["no_gallery_membership"])
        self.assertEqual(c["eligible_bridge_manual_selection"], 0)
        self.assertEqual(c["derived_anchors"], 0)
        self.assertEqual(c["gallery_members"], 0)
        self.assertEqual(c["frozen_target_seed_count"], 1)

    def test_frozen_seed_embedding_read_rejected(self):
        cfg = b1c2.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml"
        )
        with self.assertRaises(b1c2.BridgeReviewError) as ctx:
            b1c2.validate_embedding_non_frozen(
                _PROJECT_ROOT,
                cfg,
                "raw_222_full",
                forbidden_segment="raw_222_full",
            )
        self.assertIn("frozen seed embedding", str(ctx.exception))

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
        reasons = b1c2.chain_exclusion_reasons(
            raw_track_id=222,
            segment_id="raw_222_full",
            emb={"embedding_available": False},
            crop_catalog={},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("original_frozen_seed_exclusion", reasons)

    def test_stage5c_exclusion(self):
        exclusion = {
            "keys": {
                "segment_id": {"raw_1_full"},
                "raw_track_id": {"1"},
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
        reasons = b1c2.chain_exclusion_reasons(
            raw_track_id=1,
            segment_id="raw_1_full",
            emb={"embedding_available": True, "crop_ids": []},
            crop_catalog={},
            exclusion=exclusion,
            forbidden_segment="raw_222_full",
            forbidden_track=222,
        )
        self.assertIn("stage5c_membership", reasons)

    def test_stable_bridge_codes(self):
        chains = {
            (472, "raw_472_full"): [{"frame_index": 394}],
            (241, "raw_241_full"): [{"frame_index": 280}, {"frame_index": 290}],
            (474, "raw_474_full"): [{"frame_index": 395}],
        }
        a = b1c2.assign_bridge_codes(list(chains.keys()), chains)
        b = b1c2.assign_bridge_codes(list(chains.keys()), chains)
        self.assertEqual(a, b)
        self.assertEqual(a[(241, "raw_241_full")], "BRIDGE_CANDIDATE_01")
        self.assertTrue(all(v.startswith("BRIDGE_CANDIDATE_") for v in a.values()))

    def test_sheet_frames_deterministic(self):
        selected = b1c2.select_sheet_frames(
            [280, 290, 300, 310, 320],
            [280, 290, 310],
        )
        self.assertEqual(selected, [280, 290, 310])
        self.assertEqual(len(selected), len(set(selected)))

    def test_blank_manual_template_fields(self):
        self.assertIn("selected_as_target_continuation", b1c2.TEMPLATE_FIELDS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(b1c2.TEMPLATE_FIELDS))
                writer.writeheader()
                writer.writerow(
                    {
                        "target_id": "target_001",
                        "bridge_candidate_code": "BRIDGE_CANDIDATE_01",
                        "segment_id": "raw_241_full",
                        "raw_track_id": 241,
                        "first_frame": 280,
                        "last_frame": 294,
                        "selected_as_target_continuation": "",
                        "manual_target_confirmed": "",
                        "manual_identity_continuity_observed": "",
                        "manual_crop_valid": "",
                        "manual_target_dominant": "",
                        "manual_human_verified_number_seen": "",
                        "manual_notes": "",
                        "reviewer": "",
                        "final_approver": "",
                        "reviewed_at": "",
                    }
                )
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["selected_as_target_continuation"], "")
            self.assertEqual(row["manual_target_confirmed"], "")
            self.assertNotIn("similarity_score", row)
            self.assertNotIn("ocr_prediction", row)

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1c2.BridgeReviewError):
            b1c2.assert_no_path_traversal("../x")

    def test_tristate_vocabulary(self):
        self.assertEqual(set(b1c2.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})

    def test_live_b1c_no_eligible_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_alternate_eligible_source_review"
        )
        if not root.is_dir():
            self.skipTest("B1C output absent")
        summary = json.loads((root / "stage5d_b1c_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1C_NO_ELIGIBLE_SOURCE_IN_EARLY_WINDOW",
        )
        self.assertEqual(summary["review_window_frames"], [30, 75])
        self.assertEqual(summary["eligible_candidate_count"], 0)
        self.assertEqual(summary["ineligible_candidate_count"], 25)
        self.assertEqual(summary["gallery_members"], 0)

    def test_live_b1c2_output_if_present(self):
        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_seed_to_eligible_bridge_review"
        )
        if not root.is_dir():
            self.skipTest("B1C2 output absent")
        summary = json.loads((root / "stage5d_b1c2_summary.json").read_text(encoding="utf-8"))
        self.assertIn(
            summary["final_status"],
            {
                "COMPLETED_STAGE5D_B1C2_SEED_TO_ELIGIBLE_BRIDGE_REVIEW_READY",
                "COMPLETED_STAGE5D_B1C2_NO_ELIGIBLE_BRIDGE_SOURCE",
            },
        )
        self.assertEqual(summary["original_frozen_seed_code"], "SEED_CANDIDATE_07")
        self.assertFalse(summary["frozen_seed_embedding_used"])
        self.assertTrue(summary["frozen_seed_visual_only"])
        self.assertEqual(summary["bridge_window_frames"], [280, 420])
        self.assertEqual(summary["eligible_bridge_manual_selection"], 0)
        self.assertEqual(summary["derived_anchors"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["png_count"], 3)
        self.assertEqual(summary["mp4_count"], 1)
        self.assertEqual(len(list(root.rglob("*.png"))), 3)
        self.assertEqual(len(list(root.rglob("*.mp4"))), 1)
        self.assertEqual(len(list(root.rglob("*.jpg"))), 0)
        mapping = root / "inventory" / "target_001_seed_to_eligible_bridge_mapping.jsonl"
        lines = [ln for ln in mapping.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), summary["eligible_bridge_candidate_count"])
        for line in lines:
            row = json.loads(line)
            self.assertTrue(row["bridge_candidate_code"].startswith("BRIDGE_CANDIDATE_"))
            self.assertNotEqual(row["segment_id"], "raw_222_full")
            self.assertNotEqual(int(row["raw_track_id"]), 222)
            self.assertFalse(row["embedding_lineage"]["is_original_frozen_seed_embedding"])
            self.assertEqual(row["selected_as_target_continuation"], "")
        tpl = (
            root
            / "templates"
            / "target_001_seed_to_eligible_bridge_review_template.csv"
        )
        with tpl.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), summary["eligible_bridge_candidate_count"])
        for row in rows:
            self.assertEqual(row["selected_as_target_continuation"], "")
            self.assertEqual(row["manual_target_confirmed"], "")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1c2.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1c2, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1c2.BridgeReviewError) as ctx:
                    b1c2.run(
                        _PROJECT_ROOT
                        / "configs/reid/target_seed_to_eligible_bridge_review_stage5d_target_001.yaml",
                        project,
                    )
            self.assertIn("final_exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
