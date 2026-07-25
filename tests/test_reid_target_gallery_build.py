"""Unit tests for Stage 5D-B1E-F target 001 frozen OSNet gallery build."""

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

import run_reid_target_gallery_build as b1ef  # noqa: E402

_CFG = _PROJECT_ROOT / "configs/reid/target_gallery_build_stage5d_target_001.yaml"
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_gallery_v1"
)
_B1EE = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_anchor_freeze"
)
_HEAD = "3d56f0901fda4f96f030724793a5b6159e7912f2"
_OSNET_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
_SNAP_SHA = "439eb03e600252ece3f980e3d47257a3a2327599b761cd9ba52ccdfae04a17e6"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TargetGalleryBuildTests(unittest.TestCase):
    def test_expected_git_contract_constants(self):
        cfg = b1ef.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Freeze target 001 external anchor selections",
        )
        self.assertEqual(tuple(cfg["approved_exact_ids"]), b1ef.APPROVED_IDS)
        self.assertEqual(len(b1ef.APPROVED_IDS), 7)
        self.assertEqual(len(b1ef.EXCLUDED_IDS), 8)
        self.assertFalse(cfg["gallery_policy"]["automatic_gallery_growth"])
        self.assertFalse(cfg["gallery_policy"]["threshold_selection"])
        self.assertFalse(cfg["gallery_policy"]["sample_identity_assignment"])

    def test_exact_approved_and_excluded_ids(self):
        self.assertEqual(
            b1ef.APPROVED_IDS,
            (
                "target_001_ext_anchor_001",
                "target_001_ext_anchor_003",
                "target_001_ext_anchor_004",
                "target_001_ext_anchor_006",
                "target_001_ext_anchor_008",
                "target_001_ext_anchor_011",
                "target_001_ext_anchor_014",
            ),
        )
        for cid in b1ef.EXCLUDED_IDS:
            self.assertNotIn(cid, b1ef.APPROVED_IDS)

    def test_osnet_checkpoint_sha_in_config(self):
        cfg = b1ef.load_config(_CFG)
        self.assertEqual(cfg["osnet_checkpoint"]["expected_sha256"], _OSNET_SHA)
        self.assertEqual(cfg["osnet_checkpoint"]["embedding_dimension"], 512)

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1ef.GalleryBuildError):
            b1ef.assert_no_path_traversal("../x")
        with self.assertRaises(b1ef.GalleryBuildError):
            b1ef.assert_no_path_traversal("/abs/x")

    def test_centroid_and_medoid_helpers(self):
        # orthonormal-ish rows
        g = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        # pad to 512 for helper that assumes gallery rows used with full dim
        # Use compute on 3-d manually via functions after patching? Instead test
        # medoid/centroid with synthetic 512-d.
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(7, 512)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        gallery = (raw / norms).astype(np.float32)
        centroid = b1ef.compute_centroid(gallery)
        self.assertEqual(centroid.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)
        ids = list(b1ef.APPROVED_IDS)
        medoid = b1ef.compute_medoid(gallery, ids)
        self.assertIn(medoid["anchor_candidate_id"], ids)
        # Deterministic tie-break: identical rows -> first ID ascending
        same = np.tile(gallery[0], (7, 1))
        med_tie = b1ef.compute_medoid(same, ids)
        self.assertEqual(med_tie["anchor_candidate_id"], ids[0])

    def test_pairwise_diagonal_and_no_removal(self):
        rng = np.random.default_rng(1)
        raw = rng.normal(size=(7, 512)).astype(np.float32)
        gallery = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
        ids = list(b1ef.APPROVED_IDS)
        approved = [
            {
                "source_occurrence_code": ["EXT_004", "EXT_183", "EXT_198"][i % 3],
                "view_category": ["front", "rear", "right_side"][i % 3],
            }
            for i in range(7)
        ]
        centroid = b1ef.compute_centroid(gallery)
        medoid = b1ef.compute_medoid(gallery, ids)
        audit = b1ef.pairwise_audit(
            gallery,
            ids,
            approved,
            centroid,
            medoid,
            {"mad_multiplier": 3.0, "removal_allowed": False},
        )
        diag = np.diag(audit["pairwise"])
        self.assertTrue(np.allclose(diag, 1.0, atol=1e-5))
        self.assertFalse(audit["metrics"]["threshold_selected"])
        self.assertFalse(audit["metrics"]["automatic_anchor_removal"])
        for flag in audit["metrics"]["outlier_diagnostics"]["flagged"]:
            self.assertFalse(flag["removal_allowed"])

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1ef.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1ef, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1ef.GalleryBuildError) as ctx:
                    b1ef.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_b1e_e_freeze_contract_if_present(self):
        if not _B1EE.is_dir():
            self.skipTest("B1E-E absent")
        cfg = b1ef.load_config(_CFG)
        summary = json.loads(
            (_B1EE / "stage5d_b1e_e_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["final_status"],
            cfg["stage5d_b1e_e_package"]["expected_final_status"],
        )
        self.assertEqual(summary["frozen_approved_anchors"], 7)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(list(summary["approved_exact_ids"]), list(b1ef.APPROVED_IDS))
        snap = Path(cfg["stage5d_b1e_e_package"]["snapshot_path"])
        if snap.is_file():
            self.assertEqual(_sha256(snap), _SNAP_SHA)

    def test_live_gallery_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("B1E-F gallery output absent")
        summary = json.loads(
            (_FINAL / "stage5d_b1e_f_summary.json").read_text(encoding="utf-8")
        )
        contract = json.loads(
            (_FINAL / "stage5d_b1e_f_contract.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            summary["final_status"],
            {b1ef.STATUS_READY, b1ef.STATUS_INTERNAL},
        )
        self.assertEqual(summary["frozen_anchor_count"], 7)
        self.assertEqual(summary["embedding_count"], 7)
        self.assertEqual(summary["embedding_dimension"], 512)
        self.assertEqual(summary["individual_gallery_members"], 7)
        self.assertEqual(summary["centroid_count"], 1)
        self.assertEqual(summary["medoid_count"], 1)
        self.assertEqual(summary["target_assignments_on_sample_mp4"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertFalse(summary["automatic_gallery_growth"])
        self.assertFalse(summary["sample_mp4_inference"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(contract["excluded_crop_source_reads"], 0)
        self.assertEqual(contract["crop_copies"], 0)

        emb = np.load(_FINAL / "embeddings" / "target_001_anchor_embeddings.npy")
        self.assertEqual(emb.shape, (7, 512))
        self.assertTrue(np.isfinite(emb).all())
        self.assertFalse(np.any(np.all(emb == 0, axis=1)))
        norms = np.linalg.norm(emb, axis=1)
        self.assertTrue(np.all(norms > 0))
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))

        gallery = np.load(_FINAL / "gallery" / "target_001_individual_gallery.npy")
        self.assertEqual(gallery.shape, (7, 512))
        self.assertTrue(np.allclose(gallery, emb))
        centroid = np.load(_FINAL / "gallery" / "target_001_gallery_centroid.npy")
        self.assertEqual(centroid.shape, (512,))
        expected_centroid = emb.mean(axis=0)
        expected_centroid = expected_centroid / np.linalg.norm(expected_centroid)
        self.assertTrue(np.allclose(centroid, expected_centroid, atol=1e-5))
        medoid = np.load(_FINAL / "gallery" / "target_001_gallery_medoid.npy")
        medoid_id = summary["medoid_anchor_candidate_id"]
        idx = list(b1ef.APPROVED_IDS).index(medoid_id)
        self.assertTrue(np.allclose(medoid, gallery[idx]))

        members = [
            json.loads(line)
            for line in (
                _FINAL / "gallery" / "target_001_gallery_members.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(members), 7)
        self.assertTrue(all(m["human_approved"] for m in members))
        self.assertTrue(all(not m["automatic_enrollment"] for m in members))
        self.assertTrue(all(m["target_id"] == "target_001" for m in members))

        pairwise = np.load(_FINAL / "audit" / "target_001_gallery_pairwise_cosine.npy")
        self.assertEqual(pairwise.shape, (7, 7))
        self.assertTrue(np.allclose(np.diag(pairwise), 1.0, atol=1e-5))
        consistency = json.loads(
            (_FINAL / "audit" / "target_001_gallery_internal_consistency.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(consistency["is_identity_threshold"])
        self.assertFalse(consistency["threshold_selected"])
        self.assertFalse(consistency["automatic_anchor_removal"])

        pre = json.loads(
            (
                _FINAL
                / "runtime"
                / "target_001_gallery_preprocessing_contract_pre_inference.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(pre["predictions_embeddings_seen_at_freeze"])
        self.assertEqual(pre["crop_count"], 7)
        self.assertEqual(pre["expected_dimension"], 512)
        self.assertEqual(pre["checkpoint_sha256"], _OSNET_SHA)

        meta = [
            json.loads(line)
            for line in (
                _FINAL / "embeddings" / "target_001_anchor_embedding_metadata.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual([m["anchor_candidate_id"] for m in meta], list(b1ef.APPROVED_IDS))
        self.assertEqual(len(list(_FINAL.rglob("*.npy"))), 5)
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])

        # Source immutability: freeze and crops unchanged relative to freeze SHA fields
        freeze_csv = (
            _B1EE / "anchor_freeze" / "target_001_external_approved_anchors_frozen.csv"
        )
        self.assertTrue(freeze_csv.is_file())
        runtime = json.loads((_FINAL / "runtime" / "runtime.json").read_text())
        self.assertEqual(runtime["excluded_crop_source_reads"], 0)
        self.assertFalse(runtime["sample_mp4_decoded"])
        self.assertFalse(runtime["yolo_loaded"])


if __name__ == "__main__":
    unittest.main()
