"""Unit tests for Stage 5D-F3I independent holdout v2 ingestion/preflight."""

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

import run_reid_independent_holdout_v2_ingestion_and_preflight as f3i  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_ingestion_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ingestion_and_preflight"
)
_HEAD = "13d33d7f177c02ccf0f1a3392b7569bd92996444"
_F3H_SNAP = "09b7844fbd8e298956820456a7f0a1b82742d64f8669a02d68ec86d9aad7e6a3"


class IndependentHoldoutV2PreflightTests(unittest.TestCase):
    def test_expected_git_and_constants(self):
        cfg = f3i.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["holdout_source"]["exact_filename"],
            "target_001_independent_holdout_v2.mp4",
        )
        self.assertEqual(cfg["fingerprint"]["sampling_rate_fps"], 2.0)

    def test_fingerprint_helpers_deterministic(self):
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 256, size=(36, 64), dtype=np.uint8)
        a = f3i.dhash64(gray)
        b = f3i.dhash64(gray.copy())
        self.assertEqual(a, b)
        e1 = f3i.edge_dhash64(gray)
        e2 = f3i.edge_dhash64(gray.copy())
        self.assertEqual(e1, e2)
        self.assertEqual(f3i.hamming64(a, a), 0)
        other = np.zeros((36, 64), dtype=np.uint8)
        self.assertIsInstance(f3i.dhash64(other), int)

    def test_pair_match_exact_and_perceptual(self):
        gray = np.full((36, 64), 40, dtype=np.uint8)
        a = f3i.FramePrint(0, 0, 0.0, f3i.sha256_bytes(gray.tobytes()), f3i.dhash64(gray), f3i.edge_dhash64(gray), gray)
        b = f3i.FramePrint(1, 1, 0.5, a.sha256, a.dhash, a.edge_dhash, gray.copy())
        m = f3i.pair_match(a, b, dhash_max=6, edge_max=8, mad_max=12)
        self.assertIsNotNone(m)
        self.assertTrue(m["exact_hash_match"])
        noise = gray.copy()
        noise[0, 0] = 255
        c = f3i.FramePrint(2, 2, 1.0, f3i.sha256_bytes(noise.tobytes()), f3i.dhash64(noise), f3i.edge_dhash64(noise), noise)
        # Highly different frame should not match.
        far = np.full((36, 64), 200, dtype=np.uint8)
        d = f3i.FramePrint(3, 3, 1.5, f3i.sha256_bytes(far.tobytes()), f3i.dhash64(far), f3i.edge_dhash64(far), far)
        self.assertIsNone(f3i.pair_match(a, d, dhash_max=6, edge_max=8, mad_max=12))
        self.assertIsNotNone(c)

    def test_classify_incidental_vs_confirmed(self):
        cfg = {
            "sampling_rate_fps": 2.0,
            "dhash_hamming_max": 6,
            "edge_dhash_hamming_max": 8,
            "mad_max": 12,
            "confirmed_min_consecutive": 8,
            "confirmed_median_dhash_max": 4,
            "confirmed_median_mad_max": 8,
            "ambiguous_min_consecutive": 4,
            "ambiguous_max_consecutive": 7,
            "near_duplicate_coverage_ge": 0.90,
            "offset_drift_max_samples": 1,
        }
        hold = []
        ref = []
        for i in range(12):
            g = np.full((36, 64), 10 + i, dtype=np.uint8)
            # Make hold and ref identical sequences → near/confirmed overlap.
            hold.append(
                f3i.FramePrint(
                    i,
                    i,
                    i * 0.5,
                    f3i.sha256_bytes(g.tobytes()),
                    f3i.dhash64(g),
                    f3i.edge_dhash64(g),
                    g,
                )
            )
            ref.append(
                f3i.FramePrint(
                    i,
                    i,
                    i * 0.5,
                    hold[-1].sha256,
                    hold[-1].dhash,
                    hold[-1].edge_dhash,
                    g.copy(),
                )
            )
        out = f3i.classify_overlap(hold, ref, fp_cfg=cfg)
        self.assertIn(
            out["final_classification"],
            {"confirmed_temporal_overlap", "near_duplicate_sequence"},
        )
        self.assertGreaterEqual(out["maximum_matched_run_length"], 8)

        # Unrelated sequences → no_overlap / incidental
        ref2 = []
        for i in range(12):
            g = np.full((36, 64), 200 - i, dtype=np.uint8)
            g[i % 36, :] = 0
            ref2.append(
                f3i.FramePrint(
                    i,
                    i,
                    i * 0.5,
                    f3i.sha256_bytes(g.tobytes()),
                    f3i.dhash64(g),
                    f3i.edge_dhash64(g),
                    g,
                )
            )
        out2 = f3i.classify_overlap(hold, ref2, fp_cfg=cfg)
        self.assertIn(
            out2["final_classification"],
            {"no_overlap", "incidental_similarity"},
        )

    def test_sample_timestamps_and_downsample(self):
        times = f3i.sample_timestamps(5.0, 2.0, 3600)
        self.assertEqual(times[0], 0.0)
        self.assertLess(times[-1], 5.0)
        self.assertEqual(len(times), 10)
        long_times = list(range(5000))
        down, meta = f3i.maybe_downsample_uniform([float(x) for x in long_times], 100)
        self.assertTrue(meta["downsampled"])
        self.assertLessEqual(len(down), 100)

    def test_path_traversal_and_atomic(self):
        with self.assertRaises(f3i.HoldoutPreflightError):
            f3i.assert_no_path_traversal("../x")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3i.HoldoutPreflightError):
                f3i.atomic_publish(tmp, final)

    def test_f3h_validation_if_present(self):
        cfg = f3i.load_config(_CFG)
        root = _PROJECT_ROOT / cfg["stage5d_f3h_package"]["path"]
        if not root.is_dir():
            self.skipTest("F3H absent")
        out = f3i.validate_f3h(_PROJECT_ROOT, cfg)
        self.assertEqual(out["snapshot_sha256"], _F3H_SNAP)
        self.assertEqual(out["summary"]["primary_formula"], "TARGET_DISTRACTOR_MAX_MARGIN")

    def test_fingerprint_contract_frozen_fields(self):
        cfg = f3i.load_config(_CFG)
        contract = f3i.build_fingerprint_contract(cfg)
        self.assertTrue(contract["frozen_before_frame_decode"])
        self.assertEqual(contract["sampling_rate_fps"], 2.0)
        self.assertEqual(contract["normalize_width"], 64)
        self.assertEqual(contract["normalize_height"], 36)
        self.assertTrue(contract["no_frames_retained_after_finalization"])

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3I root absent")
        summary = json.loads(
            (_FINAL / "stage5d_f3i_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["final_status"], f3i.FINAL_STATUS)
        self.assertEqual(summary["readiness"], f3i.READINESS)
        self.assertTrue(summary["accepted_as_independent_holdout"])
        self.assertFalse(summary["exact_duplicate_sample"])
        self.assertFalse(summary["exact_duplicate_external"])
        self.assertFalse(summary["gallery_overlap"])
        self.assertTrue(summary["full_decode_pass"])
        self.assertEqual(summary["detections"], 0)
        self.assertEqual(summary["tracks"], 0)
        self.assertEqual(summary["crops"], 0)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["score_rows"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertIn(
            summary["sample_temporal_classification"],
            {
                "no_overlap",
                "incidental_similarity",
            },
        )
        self.assertIn(
            summary["external_temporal_classification"],
            {
                "no_overlap",
                "incidental_similarity",
            },
        )
        decision = json.loads(
            (
                _FINAL
                / "independence"
                / "target_001_holdout_independence_decision.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(decision["accepted_as_independent_holdout"])
        self.assertFalse(decision["enrollment_eligible"])
        self.assertFalse(decision["gallery_growth_eligible"])
        self.assertFalse(decision["threshold_calibration_eligible"])
        self.assertTrue(decision["scoring_evaluation_eligible"])
        access = json.loads(
            (_FINAL / "runtime" / "target_001_f3i_access_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(access["osnet_model_loads"], 0)
        self.assertEqual(access["embeddings"], 0)
        self.assertEqual(access["fingerprint_raw_frames_retained"], 0)
        self.assertEqual(list(_FINAL.rglob("*.npy")), [])
        self.assertEqual(list(_FINAL.rglob("*.mp4")), [])
        self.assertEqual(list(_FINAL.rglob("*.png")), [])
        pre = json.loads(
            (
                _FINAL
                / "independence"
                / "target_001_holdout_frame_fingerprint_contract_pre_decode.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(pre["frozen_before_frame_decode"])

    def test_run_rejects_existing_final(self):
        if not _FINAL.is_dir():
            self.skipTest("F3I root absent")
        with mock.patch.object(f3i, "assert_git_contract", return_value=_HEAD):
            with self.assertRaises(f3i.HoldoutPreflightError):
                f3i.run(_CFG)


if __name__ == "__main__":
    unittest.main()
