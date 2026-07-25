"""Unit tests for Stage 5D-F2A sample ground-truth manual freeze."""

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

import run_reid_sample_ground_truth_manual_freeze as f2a  # noqa: E402

_CFG = (
    _PROJECT_ROOT / "configs/reid/sample_ground_truth_manual_freeze_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_sample_ground_truth_freeze"
)
_F2 = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_sample_ground_truth_review_package"
)
_HEAD = "5381128da24378a99c91f4076390380917a7fd18"
_F2_SNAP = "225e189d51008e83c6bb2c8889c9f66292bccd3aac17e6501faa2cbbecb0774b"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SampleGroundTruthManualFreezeTests(unittest.TestCase):
    def test_expected_git_and_decision_set_sizes(self):
        cfg = f2a.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Add target 001 sample ground-truth review package",
        )
        f2a.validate_decision_sets()
        self.assertEqual(len(f2a.POSITIVE_IDS), 8)
        self.assertEqual(len(f2a.UNCERTAIN_IDS), 8)
        self.assertEqual(len(f2a.NON_PLAYER_IDS), 7)
        self.assertEqual(len(f2a.AMBIGUOUS_IDS), 24)
        self.assertEqual(len(f2a.NEGATIVE_IDS), 103)

    def test_exact_positive_and_special_ambiguous(self):
        self.assertEqual(
            f2a.POSITIVE_IDS,
            (
                "SAMPLE_EVAL_003",
                "SAMPLE_EVAL_024",
                "SAMPLE_EVAL_028",
                "SAMPLE_EVAL_042",
                "SAMPLE_EVAL_046",
                "SAMPLE_EVAL_069",
                "SAMPLE_EVAL_100",
                "SAMPLE_EVAL_102",
            ),
        )
        d108 = f2a.build_decision_for_code("SAMPLE_EVAL_108")
        self.assertEqual(d108["manual_occurrence_decision"], "multi_person_ambiguous")
        self.assertTrue(d108["target_present"])
        self.assertFalse(d108["retrieval_metric_eligible"])
        d148 = f2a.build_decision_for_code("SAMPLE_EVAL_148")
        self.assertTrue(d148["target_present"])
        self.assertEqual(d148["manual_human_verified_number_seen"], "yes")
        d146 = f2a.build_decision_for_code("SAMPLE_EVAL_146")
        d150 = f2a.build_decision_for_code("SAMPLE_EVAL_150")
        self.assertFalse(d146["target_present"])
        self.assertFalse(d150["target_present"])

    def test_uncertain_and_ambiguous_not_negative(self):
        for code in f2a.UNCERTAIN_IDS:
            d = f2a.build_decision_for_code(code)
            self.assertFalse(d["clean_negative"])
            self.assertFalse(d["retrieval_metric_eligible"])
        for code in f2a.AMBIGUOUS_IDS:
            d = f2a.build_decision_for_code(code)
            self.assertFalse(d["clean_negative"])
            self.assertFalse(d["clean_positive"])
            self.assertFalse(d["retrieval_metric_eligible"])

    def test_component_label_conflict_policy(self):
        rows = [
            {
                "sample_eval_code": "A",
                "evaluation_component_id": "C1",
                "clean_positive": True,
                "clean_negative": False,
                "retrieval_metric_eligible": True,
            },
            {
                "sample_eval_code": "B",
                "evaluation_component_id": "C1",
                "clean_positive": False,
                "clean_negative": True,
                "retrieval_metric_eligible": True,
            },
            {
                "sample_eval_code": "C",
                "evaluation_component_id": "C2",
                "clean_positive": True,
                "clean_negative": False,
                "retrieval_metric_eligible": True,
            },
            {
                "sample_eval_code": "D",
                "evaluation_component_id": "C2",
                "clean_positive": False,
                "clean_negative": False,
                "retrieval_metric_eligible": False,
            },
            {
                "sample_eval_code": "E",
                "evaluation_component_id": "C3",
                "clean_positive": False,
                "clean_negative": False,
                "retrieval_metric_eligible": False,
            },
        ]
        labels, stats = f2a.label_components(rows)
        self.assertEqual(labels["C1"], "conflicting_component")
        self.assertEqual(labels["C2"], "positive_component")
        self.assertEqual(labels["C3"], "excluded_component")
        self.assertEqual(stats["conflicting_component_count"], 1)

    def test_path_traversal_rejection(self):
        with self.assertRaises(f2a.GroundTruthFreezeError):
            f2a.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = f2a.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(f2a, "assert_git_contract", return_value="deadbeef"):
                with mock.patch.object(f2a, "validate_decision_sets"):
                    with self.assertRaises(f2a.GroundTruthFreezeError) as ctx:
                        f2a.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_f2_package_if_present(self):
        if not _F2.is_dir():
            self.skipTest("F2 absent")
        summary = json.loads((_F2 / "stage5d_f2_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["final_status"],
            f2a.load_config(_CFG)["stage5d_f2_package"]["expected_final_status"],
        )
        self.assertEqual(summary["scoreable_evaluation_items"], 150)
        self.assertEqual(summary["manual_ground_truth_decisions"], 0)
        snap = Path(f2a.load_config(_CFG)["stage5d_f2_package"]["snapshot_path"])
        if snap.is_file():
            self.assertEqual(_sha256(snap), _F2_SNAP)

    def test_live_freeze_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F2A freeze absent")
        summary = json.loads((_FINAL / "stage5d_f2a_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_status"], f2a.FINAL_STATUS)
        self.assertEqual(summary["exact_next_gate"], f2a.NEXT_GATE)
        self.assertEqual(summary["reviewed_total"], 150)
        self.assertEqual(summary["target_occurrence_yes"], 8)
        self.assertEqual(summary["target_occurrence_no"], 103)
        self.assertEqual(summary["non_player"], 7)
        self.assertEqual(summary["uncertain"], 8)
        self.assertEqual(summary["multi_person_ambiguous"], 24)
        self.assertEqual(summary["invalid"], 0)
        self.assertEqual(summary["clean_positive_metric_items"], 8)
        self.assertEqual(summary["clean_negative_metric_items"], 110)
        self.assertEqual(summary["excluded_metric_items"], 32)
        self.assertEqual(summary["eligible_total"], 118)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertEqual(summary["ranking_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["gallery_members"], 7)
        self.assertFalse(summary["gallery_vectors_read"])
        self.assertFalse(summary["sample_embedding_vectors_read"])
        self.assertEqual(list(summary["positive_exact_ids"]), list(f2a.POSITIVE_IDS))

        with (
            _FINAL
            / "ground_truth_freeze"
            / "target_001_sample_ground_truth_decisions_frozen.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 150)
        codes = [r["sample_eval_code"] for r in rows]
        self.assertEqual(codes, [f"SAMPLE_EVAL_{i:03d}" for i in range(1, 151)])
        by = {r["sample_eval_code"]: r for r in rows}
        self.assertEqual(by["SAMPLE_EVAL_108"]["manual_occurrence_decision"], "multi_person_ambiguous")
        self.assertEqual(by["SAMPLE_EVAL_108"]["target_present"], "True")
        self.assertEqual(by["SAMPLE_EVAL_148"]["manual_human_verified_number_seen"], "yes")
        self.assertEqual(by["SAMPLE_EVAL_146"]["target_present"], "False")
        self.assertEqual(by["SAMPLE_EVAL_150"]["target_present"], "False")

        with (
            _FINAL
            / "ground_truth_freeze"
            / "target_001_sample_metric_eligible_ground_truth.csv"
        ).open(encoding="utf-8", newline="") as handle:
            eligible = list(csv.DictReader(handle))
        self.assertEqual(len(eligible), 118)
        self.assertEqual(sum(1 for r in eligible if r["clean_positive"] == "True"), 8)
        self.assertEqual(sum(1 for r in eligible if r["clean_negative"] == "True"), 110)

        freeze = json.loads(
            (
                _FINAL / "ground_truth_freeze" / "target_001_sample_ground_truth_freeze.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(freeze["manual_decisions_frozen"])
        self.assertFalse(freeze["similarity_observed_before_freeze"])
        self.assertFalse(freeze["gallery_vectors_read"])
        self.assertFalse(freeze["sample_embedding_vectors_read"])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
