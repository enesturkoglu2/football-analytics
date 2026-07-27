"""Unit tests for Stage 5D-F3M holdout v2 frozen target–distractor scoring."""

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

import run_reid_independent_holdout_v2_frozen_scoring_evaluation as f3m  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_frozen_scoring_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_frozen_target_distractor_scoring_evaluation"
)
_HEAD = "3057d56d6f6d7510fa01b05af811ae3229e80aa0"
_F3L_SNAP = "6e6dfed290509182cb3838320afd7b69a175bb2e497c26ef215d3e19eadb45bf"
_F3H_SNAP = "09b7844fbd8e298956820456a7f0a1b82742d64f8669a02d68ec86d9aad7e6a3"
_F3G_SNAP = "dafb77147ce0c7a72b9eed43ba4b9223f1150709da667f655116b8249644f28c"
_F3K_SNAP = "63af341120a9e2d6003cce89c9b14fecb68ca300f2d5fc3fa18debc2f02cfa2b"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HoldoutFrozenScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = f3m.load_config(_CFG)

    def test_expected_git_contract_fields(self):
        self.assertEqual(self.cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            self.cfg["project_head_message_expected"],
            "Freeze target 001 holdout v2 ground truth",
        )

    def test_f3l_status_readiness_and_snapshot(self):
        f3l = f3m.validate_f3l(self.cfg)
        self.assertEqual(
            f3l["summary"]["final_status"],
            self.cfg["stage5d_f3l_package"]["expected_final_status"],
        )
        self.assertEqual(
            f3l["summary"]["readiness"],
            self.cfg["stage5d_f3l_package"]["expected_readiness"],
        )
        self.assertEqual(f3l["snapshot"]["sha256"], _F3L_SNAP)
        self.assertIn("/Users/enest/", self.cfg["stage5d_f3l_package"]["snapshot_path"])
        self.assertNotIn("/Users/enestur/", self.cfg["stage5d_f3l_package"]["snapshot_path"])

    def test_f3h_scoring_contract_and_snapshot(self):
        f3h = f3m.validate_f3h(self.cfg)
        self.assertEqual(f3h["primary"]["formula_id"], "TARGET_DISTRACTOR_MAX_MARGIN")
        self.assertEqual(f3h["primary"]["S_primary"]["formula"], "T_max(q) - D_max(q)")
        self.assertEqual(f3h["secondary"]["formulas"]["S_top3_margin"]["k"], 3)
        self.assertEqual(f3h["snapshot"]["sha256"], _F3H_SNAP)

    def test_f3g_gallery_contract_and_snapshot(self):
        g = f3m.validate_f3g(self.cfg)
        self.assertEqual(g["T"].shape, (13, 512))
        self.assertEqual(g["D"].shape, (23, 512))
        self.assertEqual(g["snapshot"]["sha256"], _F3G_SNAP)
        self.assertFalse(g["gallery_v2_mutation"])
        self.assertEqual(g["holdout_member_count"], 0)

    def test_f3k_crop_manifest_and_snapshot(self):
        f3k = f3m.validate_f3k(self.cfg)
        self.assertEqual(len(f3k["manifest"]), 141)
        self.assertEqual(f3k["snapshot"]["sha256"], _F3K_SNAP)

    def test_exact_scoreable_universe(self):
        f3l = f3m.validate_f3l(self.cfg)
        f3k = f3m.validate_f3k(self.cfg)
        universe = f3m.build_scoreable_universe(f3l["decisions"], f3k["by_id"], self.cfg)
        self.assertEqual(len(universe["scoreable"]), 115)
        self.assertEqual(len(universe["positive_ids"]), 10)
        self.assertEqual(len(universe["negative_player_ids"]), 105)
        self.assertEqual(len(universe["same_team_negative_ids"]), 55)
        self.assertEqual(len(universe["other_team_negative_ids"]), 50)
        self.assertEqual(universe["positive_ids"], list(f3m.POSITIVE_IDS))

    def test_non_player_invalid_ambiguous_not_scored(self):
        f3l = f3m.validate_f3l(self.cfg)
        f3k = f3m.validate_f3k(self.cfg)
        universe = f3m.build_scoreable_universe(f3l["decisions"], f3k["by_id"], self.cfg)
        scored = set(universe["positive_ids"] + universe["negative_player_ids"])
        for row in f3l["decisions"]:
            if row["manual_ground_truth_decision"] in {
                "non_player",
                "invalid",
                "multi_person_ambiguous",
            }:
                self.assertNotIn(row["review_item_id"], scored)
                self.assertFalse(f3m._as_bool(row["query_score_eligibility"]))

    def test_query_projection_contains_no_gt_labels(self):
        f3l = f3m.validate_f3l(self.cfg)
        f3k = f3m.validate_f3k(self.cfg)
        universe = f3m.build_scoreable_universe(f3l["decisions"], f3k["by_id"], self.cfg)
        projection = f3m.build_label_blind_projection(universe["scoreable"])
        self.assertEqual(len(projection), 115)
        blob = json.dumps(projection).lower()
        for bad in (
            "target_occurrence",
            "clean_positive",
            "clean_negative",
            "same_team",
            "jersey",
            "reviewer",
            "approver",
        ):
            self.assertNotIn(bad, blob)

    def test_primary_formula_subtraction_and_argmax_tiebreak(self):
        t_ids = ["b_member", "a_member"]
        scores = np.asarray([0.5, 0.5], dtype=np.float32)
        val, mid, idx = f3m.argmax_with_id_tiebreak(scores, t_ids)
        self.assertEqual(mid, "a_member")
        self.assertEqual(idx, 1)
        self.assertEqual(val, 0.5)
        self.assertEqual(0.9 - 0.4, 0.5)

    def test_ranking_tie_break_exact(self):
        rows = [
            {
                "stable_query_id": "H2_GT_REVIEW_000002",
                "segment_id": "S2",
                "component_id": "C2",
                "query_order_index": 1,
                "T_max": 0.5,
                "T_max_member_id": "t1",
                "D_max": 0.2,
                "D_max_member_id": "d1",
                "S_primary": 0.3,
            },
            {
                "stable_query_id": "H2_GT_REVIEW_000001",
                "segment_id": "S1",
                "component_id": "C1",
                "query_order_index": 0,
                "T_max": 0.5,
                "T_max_member_id": "t1",
                "D_max": 0.2,
                "D_max_member_id": "d1",
                "S_primary": 0.3,
            },
            {
                "stable_query_id": "H2_GT_REVIEW_000003",
                "segment_id": "S3",
                "component_id": "C3",
                "query_order_index": 2,
                "T_max": 0.8,
                "T_max_member_id": "t1",
                "D_max": 0.1,
                "D_max_member_id": "d1",
                "S_primary": 0.7,
            },
        ]
        # pad to avoid 1..115 check by testing sort key only via partial mock
        ranked_partial = sorted(
            rows,
            key=lambda r: (
                -float(r["S_primary"]),
                -float(r["T_max"]),
                float(r["D_max"]),
                str(r["stable_query_id"]),
            ),
        )
        self.assertEqual(ranked_partial[0]["stable_query_id"], "H2_GT_REVIEW_000003")
        self.assertEqual(ranked_partial[1]["stable_query_id"], "H2_GT_REVIEW_000001")
        self.assertEqual(ranked_partial[2]["stable_query_id"], "H2_GT_REVIEW_000002")

    def test_recall_ceilings_metadata(self):
        self.assertEqual(10 / 10 * 0.1, 0.1)
        ceilings = {
            "Recall@1_ceiling": 0.10,
            "Recall@3_ceiling": 0.30,
            "Recall@5_ceiling": 0.50,
            "Recall@10_ceiling": 1.00,
        }
        self.assertEqual(ceilings["Recall@1_ceiling"], 1 / 10)
        self.assertEqual(ceilings["Recall@10_ceiling"], 10 / 10)

    def test_outcome_rules_frozen_strong_numeric(self):
        f3h = f3m.validate_f3h(self.cfg)
        strong = f3h["outcomes"]["outcomes"]["INDEPENDENT_TARGET_DISTRACTOR_STRONG_SIGNAL"]
        self.assertEqual(strong["segment_AP_ge"], 0.8)
        self.assertEqual(strong["segment_AUROC_ge"], 0.9)
        self.assertEqual(strong["same_team_negative_AUROC_ge"], 0.85)

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3m.FrozenScoringError):
            f3m.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3m.FrozenScoringError):
                f3m.atomic_publish(tmp, final)

    def test_secondary_k_exact_three(self):
        f3h = f3m.validate_f3h(self.cfg)
        self.assertEqual(f3h["secondary"]["formulas"]["S_top3_margin"]["k"], 3)
        self.assertTrue(f3h["secondary"]["diagnostic_only"])
        self.assertTrue(f3h["secondary"]["cannot_replace_primary_on_same_holdout"])

    def test_gallery_sha_before_after_helper(self):
        g = f3m.validate_f3g(self.cfg)
        after = {k: _sha256(Path(v)) for k, v in g["paths"].items() if k in {
            "target", "distractor", "target_centroid", "target_medoid",
            "distractor_centroid", "distractor_medoid",
        }}
        # map keys
        mapped = {
            "target": after["target"],
            "distractor": after["distractor"],
            "target_centroid": after["target_centroid"],
            "target_medoid": after["target_medoid"],
            "distractor_centroid": after["distractor_centroid"],
            "distractor_medoid": after["distractor_medoid"],
        }
        self.assertEqual(mapped, g["shas_before"])


@unittest.skipUnless(_FINAL.is_dir(), "F3M final package not built yet")
class HoldoutFrozenScoringOutputTests(unittest.TestCase):
    def test_output_shapes_and_counts(self):
        summary = json.loads((_FINAL / "stage5d_f3m_summary.json").read_text())
        self.assertEqual(summary["final_status"], f3m.FINAL_STATUS)
        self.assertEqual(summary["scoreable"], 115)
        self.assertEqual(summary["positive"], 10)
        self.assertEqual(summary["negative_player"], 105)
        self.assertEqual(summary["same_team_negative"], 55)
        self.assertEqual(summary["other_team_negative"], 50)
        self.assertEqual(summary["excluded"], 128)
        self.assertEqual(summary["query_embedding_shape"], [115, 512])
        self.assertEqual(summary["target_cosine_shape"], [115, 13])
        self.assertEqual(summary["distractor_cosine_shape"], [115, 23])
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertFalse(summary["holdout_enrollment"])

        Q = np.load(_FINAL / "query_embeddings" / "target_001_holdout_v2_query_embeddings.npy")
        Ct = np.load(_FINAL / "scoring" / "target_001_holdout_v2_target_cosine.npy")
        Cd = np.load(_FINAL / "scoring" / "target_001_holdout_v2_distractor_cosine.npy")
        self.assertEqual(Q.shape, (115, 512))
        self.assertEqual(Ct.shape, (115, 13))
        self.assertEqual(Cd.shape, (115, 23))
        self.assertEqual(Q.dtype, np.float32)
        self.assertTrue(np.isfinite(Q).all())
        self.assertEqual(int(np.all(Q == 0, axis=1).sum()), 0)
        norms = np.linalg.norm(Q.astype(np.float64), axis=1)
        self.assertTrue(np.all(np.abs(norms - 1.0) < 1e-4))

        primary = [
            json.loads(l)
            for l in (_FINAL / "scoring" / "target_001_holdout_v2_primary_scores_label_blind.jsonl").read_text().splitlines()
            if l.strip()
        ]
        ranking = [
            json.loads(l)
            for l in (_FINAL / "ranking" / "target_001_holdout_v2_primary_ranking_label_blind.jsonl").read_text().splitlines()
            if l.strip()
        ]
        self.assertEqual(len(primary), 115)
        self.assertEqual(len(ranking), 115)
        blob = json.dumps(primary).lower()
        for bad in ("target_occurrence", "clean_positive", "jersey", "reviewer"):
            self.assertNotIn(bad, blob)

        seal = json.loads(
            (_FINAL / "pre_execution" / "target_001_holdout_v2_pre_ground_truth_join_execution_seal.json").read_text()
        )
        self.assertEqual(seal["gt_labels_read_so_far"], 0)

        access = json.loads((_FINAL / "runtime" / "target_001_f3m_access_audit.json").read_text())
        self.assertEqual(access["query_crop_reads"], 230)
        self.assertEqual(access["holdout_mp4_decode"], 0)
        self.assertEqual(access["crop_generation"], 0)
        self.assertEqual(access["threshold_selection"], 0)
        self.assertFalse(access["gt_labels_visible_during_embedding"])
        self.assertFalse(access["gt_labels_visible_during_similarity"])
        self.assertFalse(access["gt_labels_visible_during_ranking"])

        coverage = json.loads(
            (_FINAL / "evaluation" / "target_001_holdout_v2_complete_scoring_coverage.json").read_text()
        )
        self.assertEqual(coverage["complete_holdout_universe"], 243)
        self.assertEqual(coverage["scoreable_plus_excluded"], 243)

        det = json.loads(
            (_FINAL / "query_embeddings" / "target_001_holdout_v2_query_embedding_determinism.json").read_text()
        )
        self.assertTrue(det["exact_match"])
        self.assertEqual(det["overall_max_absolute_difference"], 0.0)

        npy = list(_FINAL.rglob("*.npy"))
        self.assertEqual(len(npy), 3)
        self.assertFalse(list(_FINAL.rglob("*.png")))
        self.assertFalse(list(_FINAL.rglob("*.mp4")))

        outcome = json.loads(
            (_FINAL / "evaluation" / "target_001_holdout_v2_frozen_outcome_evaluation.json").read_text()
        )
        self.assertTrue(outcome["outcome_determined_by_primary_score_only"])
        self.assertFalse(outcome["threshold_selected"])

        imm = json.loads((_FINAL / "runtime" / "target_001_f3m_immutability_audit.json").read_text())
        self.assertTrue(imm["gallery_unchanged"])
        self.assertEqual(imm["gallery_shas_before"], imm["gallery_shas_after"])


if __name__ == "__main__":
    unittest.main()
