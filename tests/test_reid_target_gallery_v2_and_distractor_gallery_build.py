"""Unit tests for Stage 5D-F3G gallery-v2 + distractor gallery build."""

from __future__ import annotations

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

import run_reid_target_gallery_v2_and_distractor_gallery_build as f3g  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/target_gallery_v2_and_distractor_gallery_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v2_and_distractor_gallery_v1"
)
_HEAD = "a71c12287a8f9ad857716d3ed685135c024dd4fe"
_F3F_SNAP = "c05637f2e047c10f9306ab241a2c073ffb0890bbb412391822283237f6db8a10"
_G1_SNAP = "bc448e6ea61331cd864268b687cb9b6c84048a9f9295737db43624b851b8468e"
_G1_EMB = "ec5a8380a90100f306cde40d98d1d5757456208a88ec438f5cb6a855778d62aa"
_OSNET = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"


class GalleryV2AndDistractorBuildTests(unittest.TestCase):
    def test_expected_git_and_exact_sets(self):
        cfg = f3g.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(len(f3g.GALLERY_V1_IDS), 7)
        self.assertEqual(len(f3g.NEW_TARGET_IDS), 6)
        self.assertEqual(len(f3g.GALLERY_V2_IDS), 13)
        self.assertEqual(len(f3g.HN_YES_IDS), 23)
        self.assertEqual(len(f3g.HN_EXCLUDED_IDS), 12)
        self.assertEqual(f3g.GALLERY_V2_IDS[:7], f3g.GALLERY_V1_IDS)
        self.assertEqual(f3g.GALLERY_V2_IDS[7:], f3g.NEW_TARGET_IDS)

    def test_osnet_checkpoint_sha_in_config(self):
        cfg = f3g.load_config(_CFG)
        self.assertEqual(cfg["osnet_checkpoint"]["expected_sha256"], _OSNET)
        self.assertEqual(cfg["osnet_checkpoint"]["embedding_dimension"], 512)

    def test_f3f_and_gallery_v1_contracts_if_present(self):
        cfg = f3g.load_config(_CFG)
        f3f_root = _PROJECT_ROOT / cfg["stage5d_f3f_package"]["path"]
        g1_root = _PROJECT_ROOT / cfg["gallery_v1"]["path"]
        if not f3f_root.is_dir() or not g1_root.is_dir():
            self.skipTest("upstream packages absent")
        f3f = f3g.validate_f3f(_PROJECT_ROOT, cfg)
        self.assertEqual(f3f["snapshot_sha256"], _F3F_SNAP)
        self.assertEqual(len(f3f["target_rows"]), 4)
        self.assertEqual(len(f3f["hn_rows"]), 23)
        g1 = f3g.validate_gallery_v1(_PROJECT_ROOT, cfg)
        self.assertEqual(g1["snapshot_sha256"], _G1_SNAP)
        self.assertEqual(g1["embedding_sha256"], _G1_EMB)
        self.assertEqual(g1["vectors"].shape, (7, 512))

    def test_centroid_medoid_helpers(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(5, 512)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        ids = [f"m{i}" for i in range(5)]
        c = f3g.compute_centroid(raw)
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
        m = f3g.compute_medoid(raw, ids)
        self.assertIn(m["member_id"], ids)
        identical = np.tile(raw[0], (3, 1))
        tie = f3g.compute_medoid(identical, ["c", "a", "b"])
        self.assertEqual(tie["member_id"], "a")

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3g.GalleryV2Error):
            f3g.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3g.GalleryV2Error):
                f3g.atomic_publish(tmp, final)

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3G root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3g_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3g.FINAL_STATUS)
        self.assertEqual(summary["readiness"], f3g.READINESS)
        self.assertEqual(summary["exact_next_gate"], f3g.NEXT_GATE)
        self.assertEqual(summary["reused_target_members"], 7)
        self.assertEqual(summary["new_target_members"], 6)
        self.assertEqual(summary["target_gallery_v2_members"], 13)
        self.assertEqual(summary["distractor_members"], 23)
        self.assertEqual(summary["new_embeddings"], 29)
        self.assertEqual(summary["embedding_dimension"], 512)
        self.assertEqual(summary["target_gallery_v2_ids"], list(f3g.GALLERY_V2_IDS))
        self.assertEqual(summary["distractor_member_ids"], list(f3g.HN_YES_IDS))
        self.assertFalse(summary["gallery_v1_mutation"])
        self.assertFalse(summary["sample_video_read"])
        self.assertEqual(summary["sample_crop_read_count"], 0)
        self.assertEqual(summary["excluded_hard_negative_crop_read_count"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["automatic_gallery_growth"])
        self.assertTrue(summary["two_pass_determinism"]["deterministic"])
        self.assertTrue(summary["two_pass_determinism"]["ordering_match"])

        target = np.load(
            _FINAL
            / "target_gallery_v2"
            / "target_001_gallery_v2_individual_embeddings.npy"
        )
        self.assertEqual(target.shape, (13, 512))
        self.assertEqual(target.dtype, np.float32)
        g1 = np.load(
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1"
            / "embeddings"
            / "target_001_anchor_embeddings.npy"
        )
        self.assertTrue(np.array_equal(target[:7], g1))
        self.assertEqual(f3g.sha256_bytes(g1.tobytes())[:8], f3g.sha256_bytes(g1.tobytes())[:8])
        # Prefer file-bytes equality via saved reused prefix SHA.
        man = json.loads(
            (
                _FINAL
                / "target_gallery_v2"
                / "target_001_gallery_v2_embedding_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(man["reused_prefix_sha256"], _G1_EMB)
        norms = np.linalg.norm(target.astype(np.float64), axis=1)
        self.assertTrue(np.all(np.abs(norms - 1.0) < 1e-4))
        self.assertEqual(int(np.isnan(target).sum()), 0)
        self.assertEqual(int(np.isinf(target).sum()), 0)
        self.assertEqual(int(np.all(target == 0, axis=1).sum()), 0)

        centroid = np.load(
            _FINAL / "target_gallery_v2" / "target_001_gallery_v2_centroid.npy"
        )
        medoid = np.load(
            _FINAL / "target_gallery_v2" / "target_001_gallery_v2_medoid.npy"
        )
        self.assertEqual(centroid.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)
        diag = json.loads(
            (
                _FINAL
                / "target_gallery_v2"
                / "target_001_gallery_v2_internal_diagnostics.json"
            ).read_text(encoding="utf-8")
        )
        mid = diag["medoid_member_id"]
        idx = list(f3g.GALLERY_V2_IDS).index(mid)
        self.assertTrue(np.allclose(medoid, target[idx]))
        self.assertFalse(diag["automatic_member_removal"])
        pw = np.load(
            _FINAL / "target_gallery_v2" / "target_001_gallery_v2_pairwise_cosine.npy"
        )
        self.assertEqual(pw.shape, (13, 13))
        self.assertTrue(np.allclose(np.diag(pw), 1.0, atol=1e-4))

        dist = np.load(
            _FINAL
            / "distractor_gallery_v1"
            / "target_001_same_team_distractor_individual_embeddings.npy"
        )
        self.assertEqual(dist.shape, (23, 512))
        d_norms = np.linalg.norm(dist.astype(np.float64), axis=1)
        self.assertTrue(np.all(np.abs(d_norms - 1.0) < 1e-4))
        d_pw = np.load(
            _FINAL
            / "distractor_gallery_v1"
            / "target_001_same_team_distractor_pairwise_cosine.npy"
        )
        self.assertEqual(d_pw.shape, (23, 23))
        jersey = json.loads(
            (
                _FINAL
                / "distractor_gallery_v1"
                / "target_001_same_team_distractor_human_jersey_inventory.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(jersey["aggregation_prototypes_created"])
        self.assertTrue(jersey["unknown_identities_not_merged"])
        self.assertIn("unknown", jersey["groups"])

        cross = np.load(
            _FINAL
            / "cross_diagnostics"
            / "target_001_target_distractor_cross_cosine.npy"
        )
        self.assertEqual(cross.shape, (13, 23))
        cross_sum = json.loads(
            (
                _FINAL
                / "cross_diagnostics"
                / "target_001_target_distractor_cross_similarity_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(
            cross_sum["descriptive_internal_enrollment_diagnostic_only"]
        )
        self.assertTrue(cross_sum["same_source_cross_similarity_not_independent"])
        self.assertFalse(cross_sum["threshold_selected"])
        self.assertFalse(cross_sum["score_formula_applied"])

        pre = json.loads(
            (
                _FINAL
                / "runtime"
                / "target_001_gallery_v2_preprocessing_contract_pre_inference.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(
            pre["predictions_embeddings_not_inspected_before_contract_freeze"]
        )
        self.assertEqual(pre["total_new_inference_crops"], 29)
        self.assertEqual(pre["existing_reused_vectors"], 7)

        access = json.loads(
            (
                _FINAL / "runtime" / "target_001_gallery_v2_access_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(access["excluded_hard_negative_crop_read_count"], 0)
        self.assertFalse(access["sample_video_read"])
        self.assertFalse(access["gallery_v1_mutation"])
        self.assertFalse(access["target_distractor_score_formula_applied"])

        self.assertEqual(len(list(_FINAL.rglob("*.npy"))), 9)
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

        # gallery-v1 still unchanged
        g1_sha = f3g.sha256_file(
            _PROJECT_ROOT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1"
            / "embeddings"
            / "target_001_anchor_embeddings.npy"
        )
        self.assertEqual(g1_sha, _G1_EMB)

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3G root absent")
        with mock.patch.object(f3g, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3g.GalleryV2Error):
                f3g.run(_CFG)


if __name__ == "__main__":
    unittest.main()
