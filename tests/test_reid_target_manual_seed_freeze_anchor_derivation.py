"""Unit tests for Stage 5D-B1B seed freeze and anchor derivation helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_target_manual_seed_freeze_anchor_derivation as sfd  # noqa: E402


class SeedFreezeAnchorDerivationTests(unittest.TestCase):
    def test_exact_selected_code_constant(self):
        cfg = sfd.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_manual_seed_freeze_anchor_derivation_stage5d_target_001.yaml"
        )
        self.assertEqual(cfg["human_selection"]["selected_neutral_seed_code"], "SEED_CANDIDATE_07")
        self.assertEqual(sfd.SELECTED_CODE, "SEED_CANDIDATE_07")
        self.assertEqual(
            cfg["project_head_expected"],
            "9ab561ce3784a08e0b30c92715810d0016bf7b9a",
        )

    def test_resolve_selected_seed_found_and_unique(self):
        rows = [
            {
                "neutral_seed_code": "SEED_CANDIDATE_06",
                "raw_track_id": 1,
                "segment_id": "raw_1_full",
                "observation_frames": [280],
                "first_frame": 280,
                "last_frame": 280,
                "bbox_per_frame": [],
            },
            {
                "neutral_seed_code": "SEED_CANDIDATE_07",
                "raw_track_id": 222,
                "segment_id": "raw_222_full",
                "observation_frames": [280, 290, 310],
                "first_frame": 280,
                "last_frame": 310,
                "bbox_per_frame": [],
            },
        ]
        resolved = sfd.resolve_selected_seed(
            rows,
            selected_code="SEED_CANDIDATE_07",
            review_window=[280, 310],
            representative_frame=290,
        )
        self.assertEqual(resolved["raw_track_id"], 222)
        self.assertEqual(resolved["segment_id"], "raw_222_full")
        self.assertEqual(resolved["representative_observation_frame"], 290)

    def test_duplicate_mapping_rejected(self):
        rows = [
            {
                "neutral_seed_code": "SEED_CANDIDATE_07",
                "raw_track_id": 222,
                "segment_id": "raw_222_full",
                "observation_frames": [290],
                "first_frame": 290,
                "last_frame": 290,
                "bbox_per_frame": [],
            },
            {
                "neutral_seed_code": "SEED_CANDIDATE_07",
                "raw_track_id": 223,
                "segment_id": "raw_223_full",
                "observation_frames": [290],
                "first_frame": 290,
                "last_frame": 290,
                "bbox_per_frame": [],
            },
        ]
        with self.assertRaises(sfd.SeedFreezeError) as ctx:
            sfd.resolve_selected_seed(
                rows,
                selected_code="SEED_CANDIDATE_07",
                review_window=[280, 310],
                representative_frame=290,
            )
        self.assertIn("AMBIGUOUS", str(ctx.exception))

    def test_missing_seed_rejected(self):
        with self.assertRaises(sfd.SeedFreezeError) as ctx:
            sfd.resolve_selected_seed(
                [],
                selected_code="SEED_CANDIDATE_07",
                review_window=[280, 310],
                representative_frame=290,
            )
        self.assertIn("NOT_FOUND", str(ctx.exception))

    def test_eligibility_stage5c_excluded(self):
        exclusion = {
            "keys": {
                "segment_id": {"raw_222_full"},
                "raw_track_id": {"222"},
                "crop_id": set(),
                "source_crop_sha256": set(),
                "near_duplicate_component": set(),
                "exact_duplicate_group": set(),
                "documented_link_component": set(),
                "temporal_source_window": set(),
                "frame_identity": set(),
            },
            "batches": {
                "discovery_primary": [],
                "discovery_reserve": [],
                "holdout_primary": [
                    {"split_item_id": "holdout_primary_010", "segment_id": "raw_222_full", "raw_track_id": 222}
                ],
                "holdout_reserve": [],
            },
            "universe_by_seg": {},
        }
        result = sfd.evaluate_eligibility(
            segment_id="raw_222_full",
            raw_track_id=222,
            emb={"embedding_available": True},
            exclusion=exclusion,
            crop_ids=[],
            crop_shas=[],
        )
        self.assertEqual(
            result["selected_seed_source_eligibility"],
            "frozen_identity_seed_only_stage5c_excluded",
        )
        self.assertFalse(result["eligible_for_anchor_derivation"])

    def test_eligibility_ok_when_clean(self):
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
            "batches": {
                "discovery_primary": [],
                "discovery_reserve": [],
                "holdout_primary": [],
                "holdout_reserve": [],
            },
            "universe_by_seg": {},
        }
        result = sfd.evaluate_eligibility(
            segment_id="raw_x_full",
            raw_track_id=999,
            emb={"embedding_available": True},
            exclusion=exclusion,
            crop_ids=[],
            crop_shas=[],
        )
        self.assertEqual(
            result["selected_seed_source_eligibility"],
            "eligible_for_anchor_derivation",
        )

    def test_diverse_anchor_derivation_and_near_dup_suppression(self):
        crops = []
        for i, frame in enumerate([100, 101, 150, 200, 250, 300, 350, 400]):
            img = np.zeros((80, 40, 3), dtype=np.uint8)
            crops.append(
                {
                    "crop_id": f"c{i}",
                    "frame_index": frame,
                    "bbox_area": 5000 + i,
                    "short_side": 40,
                    "quality_score": 100,
                    "width": 40,
                    "height": 80,
                    "source_crop_path": f"/tmp/{i}.jpg",
                    "source_crop_sha256": f"{i:064d}",
                    "bbox_xyxy": [0, 0, 40, 80],
                    "image": img,
                }
            )
        emb = {
            "embedding_artifact_path": "x.npz",
            "embedding_artifact_sha256": "a" * 64,
            "embedding_sha256": "b" * 64,
        }
        selected = sfd.derive_diverse_anchors(
            crops,
            segment_id="raw_x",
            raw_track_id=1,
            seed_freeze_sha="c" * 64,
            emb=emb,
            max_candidates=8,
        )
        self.assertGreaterEqual(len(selected), 3)
        frames = [c["frame_index"] for c in selected]
        # Near-duplicate gap: no adjacent frames within 2.
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                self.assertGreater(abs(frames[i] - frames[j]), 2)
        self.assertTrue(all(c["manual_anchor_decision"] == "" for c in selected))
        self.assertTrue(all(c["gallery_member"] is False for c in selected))

    def test_blank_annotation_vocabulary(self):
        self.assertIn("target_anchor_yes", sfd.ALLOWED_ANCHOR_DECISIONS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sfd.ANNOTATION_FIELDS))
                writer.writeheader()
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_path_traversal_rejection(self):
        with self.assertRaises(sfd.SeedFreezeError):
            sfd.assert_no_path_traversal("../x")

    def test_human_jersey_provenance_not_ocr(self):
        cfg = sfd.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_manual_seed_freeze_anchor_derivation_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["human_selection"]["jersey_number_provenance"],
            "human_visual_verification_not_automated_ocr",
        )
        self.assertEqual(cfg["human_selection"]["manual_target_confirmed"], "yes")


if __name__ == "__main__":
    unittest.main()
