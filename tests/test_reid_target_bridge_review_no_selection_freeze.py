"""Unit tests for Stage 5D-B1D bridge no-selection freeze."""

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

import run_reid_target_bridge_review_no_selection_freeze as b1d  # noqa: E402


class BridgeNoSelectionFreezeTests(unittest.TestCase):
    def test_expected_git_and_human_decisions(self):
        cfg = b1d.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "32bd5cef513d726ca60f195f6ac3f113c62b1491",
        )
        human = cfg["human_no_selection_freeze"]
        self.assertEqual(human["selected_bridge_candidate_code"], "")
        self.assertEqual(human["manual_target_continuation_found"], "no")
        self.assertEqual(
            human["manual_review_result"],
            "NO_ELIGIBLE_BRIDGE_CONTINUATION_SELECTED",
        )
        self.assertEqual(
            human["candidate_decisions"],
            {
                "BRIDGE_CANDIDATE_01": "non_player",
                "BRIDGE_CANDIDATE_02": "target_anchor_no",
                "BRIDGE_CANDIDATE_03": "target_anchor_no",
                "BRIDGE_CANDIDATE_04": "target_anchor_no",
                "BRIDGE_CANDIDATE_05": "target_anchor_no",
            },
        )

    def test_exact_five_bridge_codes_constant(self):
        self.assertEqual(len(b1d.EXPECTED_BRIDGE_CODES), 5)
        self.assertEqual(b1d.FROZEN_SEED_CODE, "SEED_CANDIDATE_07")

    def test_force_selection_rejected(self):
        cfg = b1d.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml"
        )
        bad = dict(cfg)
        human = dict(cfg["human_no_selection_freeze"])
        human["selected_bridge_candidate_code"] = "BRIDGE_CANDIDATE_02"
        bad["human_no_selection_freeze"] = human
        with self.assertRaises(b1d.BridgeNoSelectionError) as ctx:
            b1d.validate_human_freeze(bad)
        self.assertIn("FORCE_SELECTION", str(ctx.exception))

    def test_target_anchor_yes_rejected(self):
        cfg = b1d.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml"
        )
        bad = dict(cfg)
        human = dict(cfg["human_no_selection_freeze"])
        decisions = dict(human["candidate_decisions"])
        decisions["BRIDGE_CANDIDATE_03"] = "target_anchor_yes"
        human["candidate_decisions"] = decisions
        bad["human_no_selection_freeze"] = human
        with self.assertRaises(b1d.BridgeNoSelectionError):
            b1d.validate_human_freeze(bad)

    def test_contract_and_external_handoff(self):
        c = b1d.build_contract()
        self.assertTrue(c["original_frozen_seed_preserved"])
        self.assertFalse(c["frozen_seed_enrollment_allowed"])
        self.assertIsNone(c["selected_eligible_bridge_source"])
        self.assertTrue(c["force_selection_forbidden"])
        self.assertTrue(c["current_video_eligible_source_search_closed"])
        self.assertEqual(c["next_source_type"], "external_enrollment_clip")
        self.assertEqual(c["gallery_members"], 0)
        self.assertEqual(c["derived_anchors"], 0)
        self.assertFalse(c["frozen_seed_embedding_used"])

        cfg = b1d.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml"
        )
        req = b1d.build_external_requirements(cfg)
        self.assertEqual(req["human_verified_jersey_number"], 5)
        self.assertFalse(req["automated_ocr_used"])
        self.assertTrue(req["clip_must_be_enrollment_only"])
        self.assertTrue(req["clip_must_not_be_future_evaluation_input"])
        self.assertTrue(req["human_seed_selection_required"])
        self.assertTrue(req["manual_frozen_enrollment"])
        self.assertFalse(req["automatic_gallery_growth"])
        self.assertTrue(req["unknown_identity_preserved"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1d.BridgeNoSelectionError):
            b1d.assert_no_path_traversal("../x")

    def test_live_b1c2_and_b1d_if_present(self):
        b1c2 = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_seed_to_eligible_bridge_review"
        )
        if not b1c2.is_dir():
            self.skipTest("B1C2 absent")
        s = json.loads((b1c2 / "stage5d_b1c2_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            s["final_status"],
            "COMPLETED_STAGE5D_B1C2_SEED_TO_ELIGIBLE_BRIDGE_REVIEW_READY",
        )
        self.assertEqual(s["eligible_bridge_candidate_count"], 5)
        self.assertFalse(s["frozen_seed_embedding_used"])

        root = (
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_bridge_review_no_selection_freeze"
        )
        if not root.is_dir():
            self.skipTest("B1D output absent")
        summary = json.loads((root / "stage5d_b1d_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1D_NO_BRIDGE_SELECTION_EXTERNAL_ENROLLMENT_REQUIRED",
        )
        self.assertEqual(summary["selected_bridge_candidate_code"], "")
        self.assertIsNone(summary["selected_eligible_bridge_source"])
        self.assertEqual(
            summary["manual_review_result"],
            "NO_ELIGIBLE_BRIDGE_CONTINUATION_SELECTED",
        )
        self.assertEqual(summary["original_frozen_seed_code"], "SEED_CANDIDATE_07")
        self.assertFalse(summary["frozen_seed_embedding_used"])
        self.assertEqual(summary["derived_anchors"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["new_detection"], 0)
        self.assertEqual(
            summary["exact_next_gate"],
            "STAGE5D-B1E_TARGET_001_EXTERNAL_ENROLLMENT_CLIP_DESIGN_AND_INGEST",
        )
        self.assertEqual(
            summary["candidate_decisions"]["BRIDGE_CANDIDATE_01"], "non_player"
        )
        for code in (
            "BRIDGE_CANDIDATE_02",
            "BRIDGE_CANDIDATE_03",
            "BRIDGE_CANDIDATE_04",
            "BRIDGE_CANDIDATE_05",
        ):
            self.assertEqual(summary["candidate_decisions"][code], "target_anchor_no")

        csv_path = (
            root
            / "bridge_review_freeze"
            / "target_001_bridge_review_decisions_frozen.csv"
        )
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["selected_as_target_continuation"] == "no" for r in rows))
        self.assertEqual(rows[0]["manual_bridge_decision"], "non_player")

        no_sel = json.loads(
            (
                root
                / "bridge_review_freeze"
                / "target_001_bridge_review_no_selection.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(no_sel["selected_bridge_candidate_code"], "")
        self.assertTrue(no_sel["bridge_review_frozen"])

        handoff = json.loads(
            (
                root
                / "external_enrollment_handoff"
                / "target_001_external_enrollment_requirements.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(handoff["automated_ocr_used"])
        self.assertTrue(handoff["clip_must_be_enrollment_only"])
        self.assertEqual(len(list(root.rglob("*.png"))), 0)
        self.assertEqual(len(list(root.rglob("*.mp4"))), 0)

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1d.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1d, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1d.BridgeNoSelectionError) as ctx:
                    b1d.run(
                        _PROJECT_ROOT
                        / "configs/reid/target_bridge_review_no_selection_stage5d_target_001.yaml",
                        project,
                    )
            self.assertIn("final_exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
