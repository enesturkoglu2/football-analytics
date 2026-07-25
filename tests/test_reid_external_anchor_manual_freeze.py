"""Unit tests for Stage 5D-B1E-E external anchor manual freeze."""

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

import run_reid_external_anchor_manual_freeze as b1ee  # noqa: E402

_CFG = _PROJECT_ROOT / "configs/reid/external_anchor_manual_freeze_stage5d_target_001.yaml"
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_freeze"
)
_B1ED = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_review_package"
)
_HEAD = "143ce0880324119be8ffab97b8f50a9b3b984485"
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ExternalAnchorManualFreezeTests(unittest.TestCase):
    def test_expected_git_and_exact_ids(self):
        cfg = b1ee.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(b1ee.APPROVED_IDS, tuple(cfg["human_anchor_freeze"]["approved_exact_ids"]))
        self.assertEqual(len(b1ee.CANDIDATE_IDS), 15)
        self.assertEqual(len(b1ee.APPROVED_IDS), 7)
        self.assertEqual(len(b1ee.REDUNDANT_IDS), 5)

    def test_validate_human_decisions_distribution(self):
        cfg = b1ee.load_config(_CFG)
        human = b1ee.validate_human_decisions(cfg)
        self.assertEqual(human["redundant_nonselected_reason"], b1ee.REDUNDANT_REASON)
        self.assertFalse(human["automated_ocr_used"])

    def test_decision_reason_semantics(self):
        self.assertEqual(
            b1ee.decision_reason(
                "target_001_ext_anchor_002",
                "target_anchor_no",
                b1ee.REDUNDANT_REASON,
            ),
            b1ee.REDUNDANT_REASON,
        )
        self.assertEqual(
            b1ee.decision_reason(
                "target_001_ext_anchor_001", "target_anchor_yes", b1ee.REDUNDANT_REASON
            ),
            "approved_frozen_anchor",
        )
        with self.assertRaises(b1ee.AnchorFreezeError):
            b1ee.decision_reason(
                "target_001_ext_anchor_001",
                "target_anchor_no",
                b1ee.REDUNDANT_REASON,
            )

    def test_contract_forbids_gallery_and_negatives(self):
        c = b1ee.build_contract()
        self.assertFalse(c["target_anchor_no_is_identity_negative"])
        self.assertFalse(c["frozen_anchors_are_gallery_members"])
        self.assertFalse(c["embedding_generated"])
        self.assertFalse(c["osnet_used"])
        self.assertEqual(c["frozen_approved_anchors"], 7)
        self.assertEqual(c["redundant_valid_non_selected"], 5)
        self.assertEqual(c["crop_copies"], 0)

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1ee.AnchorFreezeError):
            b1ee.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1ee.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1ee, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1ee.AnchorFreezeError) as ctx:
                    b1ee.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_live_b1e_d_and_b1e_e_if_present(self):
        if not _B1ED.is_dir():
            self.skipTest("B1E-D absent")
        s = json.loads((_B1ED / "stage5d_b1e_d_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            s["final_status"],
            "COMPLETED_STAGE5D_B1E_D_TARGET_001_EXTERNAL_ANCHOR_REVIEW_READY",
        )
        self.assertEqual(s["total_candidate_count"], 15)

        if not _FINAL.is_dir():
            self.skipTest("B1E-E output absent")
        summary = json.loads(
            (_FINAL / "stage5d_b1e_e_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], b1ee.FINAL_STATUS)
        self.assertEqual(summary["reviewed_candidate_crops"], 15)
        self.assertEqual(summary["frozen_approved_anchors"], 7)
        self.assertEqual(summary["redundant_valid_non_selected"], 5)
        self.assertEqual(summary["multi_person_ambiguous"], 2)
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(summary["approved_exact_ids"], list(b1ee.APPROVED_IDS))
        self.assertEqual(
            summary["occurrence_distribution"],
            {"EXT_004": 4, "EXT_183": 1, "EXT_198": 2},
        )
        self.assertEqual(
            summary["view_distribution"],
            {
                "front": 1,
                "front_oblique": 1,
                "right_side": 1,
                "rear_oblique": 2,
                "rear": 2,
            },
        )
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["crop_copies"], 0)
        self.assertEqual(summary["osnet_inference"], 0)
        self.assertEqual(len(list(_FINAL.rglob("*.png"))), 0)
        self.assertEqual(len(list(_FINAL.rglob("*.mp4"))), 0)

        with (
            _FINAL
            / "anchor_freeze"
            / "target_001_external_anchor_review_decisions_frozen.csv"
        ).open(encoding="utf-8", newline="") as handle:
            decisions = list(csv.DictReader(handle))
        self.assertEqual(len(decisions), 15)
        self.assertTrue(all(r["is_identity_negative"] == "False" for r in decisions))
        no_rows = [r for r in decisions if r["manual_anchor_decision"] == "target_anchor_no"]
        self.assertEqual(len(no_rows), 5)
        self.assertTrue(
            all(r["decision_reason"] == b1ee.REDUNDANT_REASON for r in no_rows)
        )

        with (
            _FINAL
            / "anchor_freeze"
            / "target_001_external_approved_anchors_frozen.csv"
        ).open(encoding="utf-8", newline="") as handle:
            approved = list(csv.DictReader(handle))
        self.assertEqual(len(approved), 7)
        self.assertEqual(
            [r["anchor_candidate_id"] for r in approved], list(b1ee.APPROVED_IDS)
        )
        self.assertTrue(all(r["manual_crop_valid"] == "yes" for r in approved))
        self.assertTrue(all(r["manual_target_dominant"] == "yes" for r in approved))
        self.assertTrue(all(r["manual_single_person"] == "yes" for r in approved))
        self.assertTrue(all(r["manual_identity_confirmed"] == "yes" for r in approved))
        self.assertTrue(all(r["is_gallery_member"] == "False" for r in approved))
        self.assertTrue(all(r["embedding_generated"] == "False" for r in approved))

        freeze = json.loads(
            (
                _FINAL / "anchor_freeze" / "target_001_external_anchor_freeze.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(freeze["schema_version"], "reid_target_external_anchor_freeze_v1")
        self.assertTrue(freeze["manual_decisions_frozen"])
        self.assertFalse(freeze["frozen_anchors_are_gallery_members"])
        self.assertFalse(freeze["target_anchor_no_is_identity_negative"])

        # Source + crop immutability (references only).
        ext = _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
        self.assertEqual(_sha256(ext), _EXT_SHA)
        for row in approved:
            crop = _B1ED / row["crop_path"]
            self.assertTrue(crop.is_file())
            self.assertEqual(_sha256(crop), row["crop_sha256"])
        runtime = json.loads((_FINAL / "runtime" / "runtime.json").read_text(encoding="utf-8"))
        self.assertFalse(runtime["osnet_loaded"])
        self.assertFalse(runtime["yolo_loaded"])


if __name__ == "__main__":
    unittest.main()
