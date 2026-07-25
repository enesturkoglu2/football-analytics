"""Unit tests for Stage 5D-F2 sample ground-truth review package."""

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

import run_reid_sample_ground_truth_review_package as f2  # noqa: E402

_CFG = _PROJECT_ROOT / "configs/reid/sample_ground_truth_review_stage5d_target_001.yaml"
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_sample_ground_truth_review_package"
)
_F1 = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_validation_design"
)
_HEAD = "011d732ec0822238f5042257cd924a88fc459c8f"
_F1_SNAP = "2acdeb4e94294bcff4c658c88b0555ae427fc0df52b780676cb6ed9a9b6f48e2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SampleGroundTruthReviewTests(unittest.TestCase):
    def test_expected_git_contract_constants(self):
        cfg = f2.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Add target 001 independent retrieval validation design",
        )
        self.assertTrue(cfg["policy"]["similarity_blind"])
        self.assertTrue(cfg["policy"]["label_blind"])
        self.assertFalse(cfg["policy"]["gallery_vectors_read"])
        self.assertFalse(cfg["policy"]["sample_embedding_vectors_read"])
        self.assertFalse(cfg["policy"]["ground_truth_decisions"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(f2.GroundTruthReviewError):
            f2.assert_no_path_traversal("../x")

    def test_select_context_frames_deterministic_and_dedup(self):
        frames = [10, 20, 30, 40, 50]
        chosen = f2.select_context_frames(
            frames, start_frame=10, end_frame=50, representative_frame=30
        )
        self.assertEqual(chosen, [("START", 10), ("REP", 30), ("END", 50)])
        # When all targets collapse to one observation.
        chosen2 = f2.select_context_frames(
            [7], start_frame=0, end_frame=100, representative_frame=50
        )
        self.assertEqual(chosen2, [("START", 7)])

    def test_component_assignment_union_and_unique(self):
        rows = [
            {
                "segment_id": "a",
                "raw_track_id": 1,
                "frame_range": [0, 10],
                "documented_link_component_id": "doc_1",
                "exact_crop_sha_group": "c1",
                "exact_duplicate_embedding_group": "e1",
                "source_observation_component": "obs_1",
                "near_duplicate_cluster_id": None,
                "leakage_group_id": None,
            },
            {
                "segment_id": "b",
                "raw_track_id": 1,
                "frame_range": [11, 20],
                "documented_link_component_id": "doc_1",
                "exact_crop_sha_group": "c2",
                "exact_duplicate_embedding_group": "e2",
                "source_observation_component": "obs_1",
                "near_duplicate_cluster_id": None,
                "leakage_group_id": None,
            },
            {
                "segment_id": "c",
                "raw_track_id": 9,
                "frame_range": [0, 5],
                "documented_link_component_id": "doc_9",
                "exact_crop_sha_group": "c3",
                "exact_duplicate_embedding_group": "e3",
                "source_observation_component": "obs_9",
                "near_duplicate_cluster_id": None,
                "leakage_group_id": None,
            },
        ]
        codes, stats = f2.assign_evaluation_components(rows)
        self.assertEqual(codes[0], codes[1])
        self.assertNotEqual(codes[0], codes[2])
        self.assertEqual(stats["duplicate_membership_conflict"], 0)
        self.assertEqual(stats["component_count"], 2)
        ordered = f2.order_and_code_items(rows, codes)
        self.assertEqual(
            [r["sample_eval_code"] for r in ordered],
            ["SAMPLE_EVAL_001", "SAMPLE_EVAL_002", "SAMPLE_EVAL_003"],
        )

    def test_vocab_and_template_fields(self):
        self.assertIn("target_occurrence_yes", f2.OCCURRENCE_VOCAB)
        self.assertIn("non_player", f2.OCCURRENCE_VOCAB)
        self.assertIn("manual_occurrence_decision", f2.TEMPLATE_FIELDS)
        self.assertEqual(f2.SHEET_COUNT, 13)
        self.assertEqual(f2.ITEMS_PER_SHEET, 12)
        self.assertEqual(f2.LAST_SHEET_ITEMS, 6)

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = f2.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(f2, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(f2.GroundTruthReviewError) as ctx:
                    f2.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_f1_contract_if_present(self):
        if not _F1.is_dir():
            self.skipTest("F1 absent")
        summary = json.loads((_F1 / "stage5d_f1_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_status"], f2.load_config(_CFG)["stage5d_f1_package"]["expected_final_status"])
        self.assertEqual(summary["scoreable_sample_units"], 150)
        self.assertEqual(summary["no_embedding_sample_units"], 141)
        self.assertEqual(summary["sample_similarity_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        snap = Path(f2.load_config(_CFG)["stage5d_f1_package"]["snapshot_path"])
        if snap.is_file():
            self.assertEqual(_sha256(snap), _F1_SNAP)

    def test_live_f2_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F2 output absent")
        summary = json.loads((_FINAL / "stage5d_f2_summary.json").read_text(encoding="utf-8"))
        contract = json.loads((_FINAL / "stage5d_f2_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_status"], f2.FINAL_STATUS)
        self.assertEqual(summary["exact_next_gate"], f2.NEXT_GATE)
        self.assertEqual(summary["scoreable_evaluation_items"], 150)
        self.assertEqual(summary["unscoreable_no_embedding_items"], 141)
        self.assertEqual(summary["contact_sheets"], 13)
        self.assertEqual(summary["sheet_item_counts"], [12] * 12 + [6])
        self.assertEqual(summary["manual_ground_truth_decisions"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertEqual(summary["retrieval_ranking_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["gallery_members"], 7)
        self.assertTrue(summary["gallery_unchanged"])
        self.assertFalse(summary["gallery_vectors_read"])
        self.assertFalse(summary["sample_embedding_vectors_read"])
        self.assertFalse(summary["existing_stage5c_labels_prefilled"])
        self.assertFalse(contract["gallery_vectors_read"])

        mapping = [
            json.loads(line)
            for line in (
                _FINAL / "inventory" / "target_001_sample_evaluation_mapping.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(mapping), 150)
        codes = [r["sample_eval_code"] for r in mapping]
        self.assertEqual(codes, [f"SAMPLE_EVAL_{i:03d}" for i in range(1, 151)])
        self.assertEqual(len(set(codes)), 150)
        self.assertTrue(all(r["all_manual_fields_blank"] for r in mapping))
        self.assertTrue(all(r["scoreable"] for r in mapping))
        self.assertTrue(all(r["similarity_computed"] is False for r in mapping))

        unscoreable = [
            json.loads(line)
            for line in (
                _FINAL
                / "inventory"
                / "target_001_sample_unscoreable_no_embedding_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(unscoreable), 141)
        self.assertTrue(all(r["automatic_negative"] is False for r in unscoreable))
        self.assertTrue(all(r["recompute_authorized"] is False for r in unscoreable))
        self.assertTrue(all(r["include_in_contact_sheet"] is False for r in unscoreable))

        with (
            _FINAL / "templates" / "target_001_sample_ground_truth_review_template.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 150)
        self.assertTrue(all(r["manual_occurrence_decision"] == "" for r in rows))
        self.assertTrue(all(r["reviewer"] == "" for r in rows))

        sheets = sorted(
            (
                _FINAL / "review_packages" / "target_001_sample_ground_truth_review"
            ).glob("sample_ground_truth_sheet_*.png")
        )
        self.assertEqual(len(sheets), 13)
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])

        runtime = json.loads((_FINAL / "runtime" / "runtime.json").read_text(encoding="utf-8"))
        self.assertFalse(runtime["osnet_loaded"])
        self.assertFalse(runtime["yolo_loaded"])
        self.assertFalse(runtime["sample_mp4_inference"])
        self.assertGreater(runtime["sample_mp4_context_decode_frames"], 0)


if __name__ == "__main__":
    unittest.main()
