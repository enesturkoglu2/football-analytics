"""Unit tests for Stage 5D-F1 independent sample retrieval validation design."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_target_independent_validation_design as f1  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/target_independent_validation_design_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_validation_design"
)
_GALLERY = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1"
)
_HEAD = "e43a768648d9c76e5d65ad1827ba7b06d8835efc"
_SNAP_SHA = "bc448e6ea61331cd864268b687cb9b6c84048a9f9295737db43624b851b8468e"
_SAMPLE_SHA = "f4b28dd58a6cf242344a4198b8c0ba9062b20977cec3ae12d96322750bfd7b9b"
_EMB_SHA = "355676d7e017c3b3b5397bf82e9a088686740fbc495df35598fc90b897d28d97"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class IndependentValidationDesignTests(unittest.TestCase):
    def test_expected_git_contract_constants(self):
        cfg = f1.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Build target 001 frozen OSNet gallery",
        )
        self.assertEqual(tuple(cfg["approved_exact_ids"]), f1.APPROVED_IDS)
        self.assertFalse(cfg["policy"]["sample_similarity_computation"])
        self.assertFalse(cfg["policy"]["threshold_selection"])
        self.assertFalse(cfg["policy"]["identity_assignment"])
        self.assertFalse(cfg["policy"]["automatic_gallery_growth"])

    def test_exact_seven_gallery_member_ids(self):
        self.assertEqual(len(f1.APPROVED_IDS), 7)
        self.assertEqual(
            f1.APPROVED_IDS[3],
            "target_001_ext_anchor_006",
        )

    def test_path_traversal_rejection(self):
        with self.assertRaises(f1.ValidationDesignError):
            f1.assert_no_path_traversal("../x")
        with self.assertRaises(f1.ValidationDesignError):
            f1.assert_no_path_traversal("/abs")

    def test_ground_truth_label_blind_and_unreviewed_not_negative(self):
        c = f1.ground_truth_contract()
        self.assertTrue(c["label_blind"])
        self.assertTrue(c["policy"]["unreviewed_is_not_negative"])
        self.assertFalse(c["similarity_scores_shown_to_reviewer"])
        self.assertEqual(c["manual_decisions_in_f1"], 0)
        self.assertIn("gallery_similarity", c["forbidden_visible_fields"])
        self.assertFalse(c["f2_coverage"]["no_embedding_automatic_negative"])

    def test_primary_and_secondary_scoring_formulas(self):
        s = f1.scoring_contract()
        self.assertEqual(s["primary_retrieval_score"]["name"], "max_individual_cosine")
        self.assertTrue(s["primary_retrieval_score"]["used_for_primary_ranking"])
        names = [x["name"] for x in s["secondary_diagnostic_scores"]]
        self.assertEqual(
            names,
            [
                "top3_mean_individual_cosine",
                "centroid_cosine",
                "medoid_cosine",
                "mean_individual_cosine",
            ],
        )
        self.assertEqual(s["similarity_rows_in_f1"], 0)
        self.assertEqual(s["ranking_rows_in_f1"], 0)
        self.assertFalse(s["executed_in_f1"])

    def test_component_grouping_keys_and_aggregation(self):
        g = f1.leakage_grouping_contract()
        self.assertIn("documented_link_component", g["grouping_keys"])
        self.assertIn("exact_crop_sha", g["grouping_keys"])
        self.assertEqual(
            g["component_score_aggregation"]["primary"],
            "max_segment_retrieval_score_within_component",
        )
        self.assertTrue(g["lineage_design_only_in_f1"])
        self.assertFalse(g["computed_with_scores_in_f1"])

    def test_preregistered_metrics_and_support_gates(self):
        m = f1.metric_contract()
        self.assertIn("Recall@5", m["segment_level_primary_metrics"])
        self.assertIn("Average_Precision", m["segment_level_primary_metrics"])
        self.assertEqual(m["secondary_diagnostics"]["AUROC"]["support_gate"]["negative_ge"], 20)
        self.assertEqual(m["secondary_diagnostics"]["AUPRC"]["support_gate"]["positive_ge"], 2)
        self.assertFalse(m["threshold_policy"]["acceptance_threshold_selected"])
        self.assertIn(
            "INDEPENDENT_RETRIEVAL_STRONG_SIGNAL",
            m["descriptive_outcomes"],
        )
        self.assertIn(
            "INDEPENDENT_RETRIEVAL_INSUFFICIENT_GROUND_TRUTH",
            m["descriptive_outcomes"],
        )
        self.assertIn("deployment_approval", m["outcome_is_not"])

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = f1.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(f1, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(f1.ValidationDesignError) as ctx:
                    f1.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_gallery_v1_contract_if_present(self):
        if not _GALLERY.is_dir():
            self.skipTest("gallery-v1 absent")
        cfg = f1.load_config(_CFG)
        summary = json.loads(
            (_GALLERY / "stage5d_b1e_f_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], cfg["gallery_v1"]["expected_final_status"])
        self.assertEqual(summary["readiness"], cfg["gallery_v1"]["expected_readiness"])
        self.assertEqual(summary["individual_gallery_members"], 7)
        self.assertEqual(summary["medoid_anchor_candidate_id"], "target_001_ext_anchor_004")
        emb = np.load(_GALLERY / "embeddings" / "target_001_anchor_embeddings.npy")
        self.assertEqual(emb.shape, (7, 512))
        self.assertEqual(str(emb.dtype), "float32")
        self.assertTrue(np.isfinite(emb).all())
        self.assertFalse(np.any(np.all(emb == 0, axis=1)))
        self.assertTrue(np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4))
        snap = Path(cfg["gallery_v1"]["snapshot_path"])
        if snap.is_file():
            self.assertEqual(_sha256(snap), _SNAP_SHA)

    def test_live_f1_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F1 design output absent")
        summary = json.loads((_FINAL / "stage5d_f1_summary.json").read_text(encoding="utf-8"))
        contract = json.loads((_FINAL / "stage5d_f1_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["final_status"], f1.FINAL_STATUS)
        self.assertEqual(summary["exact_next_gate"], f1.NEXT_GATE)
        self.assertEqual(summary["gallery_members"], 7)
        self.assertEqual(summary["sample_ground_truth_decisions"], 0)
        self.assertEqual(summary["sample_similarity_rows"], 0)
        self.assertEqual(summary["retrieval_rankings"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["automatic_gallery_growth"])
        self.assertEqual(summary["scoreable_sample_units"], 150)
        self.assertEqual(summary["no_embedding_sample_units"], 141)
        self.assertEqual(summary["sample_embedding_dimension"], 512)
        self.assertTrue(summary["enrollment_evaluation_independent"])
        self.assertEqual(summary["verified_overlapping_frame_pairs"], 0)
        self.assertEqual(summary["sample_sha256"], _SAMPLE_SHA)
        self.assertEqual(summary["sample_embeddings_sha256"], _EMB_SHA)
        self.assertEqual(contract["primary_score"], "max_individual_cosine")
        self.assertTrue(contract["label_blind_ground_truth"])

        scoreable = [
            json.loads(line)
            for line in (
                _FINAL / "sample_universe" / "target_001_sample_scoreable_universe.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(scoreable), 150)
        self.assertTrue(all(r["scoreable"] for r in scoreable))
        self.assertTrue(all(r["similarity_computed"] is False for r in scoreable))
        self.assertTrue(all(r["manual_ground_truth_decision"] is None for r in scoreable))
        self.assertTrue(all(r["embedding_recompute_planned"] is False for r in scoreable))

        no_emb = [
            json.loads(line)
            for line in (
                _FINAL / "sample_universe" / "target_001_sample_no_embedding_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(no_emb), 141)
        self.assertTrue(all(not r["scoreable"] for r in no_emb))
        self.assertTrue(all(r["automatic_negative"] is False for r in no_emb))

        gt = json.loads(
            (
                _FINAL / "ground_truth_design" / "target_001_sample_ground_truth_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(gt["label_blind"])
        scoring = json.loads(
            (_FINAL / "scoring_design" / "target_001_retrieval_scoring_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(scoring["primary_retrieval_score"]["name"], "max_individual_cosine")
        metrics = json.loads(
            (_FINAL / "metric_design" / "target_001_retrieval_metric_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(metrics["threshold_policy"]["acceptance_threshold_selected"])

        runtime = json.loads((_FINAL / "runtime" / "runtime.json").read_text(encoding="utf-8"))
        self.assertFalse(runtime["sample_mp4_decoded"])
        self.assertFalse(runtime["osnet_loaded"])
        self.assertFalse(runtime["yolo_loaded"])
        self.assertEqual(runtime["sample_similarity_rows"], 0)
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])


if __name__ == "__main__":
    unittest.main()
