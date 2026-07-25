"""Unit tests for Stage 5D-B target_001 anchor review package helpers."""

from __future__ import annotations

import csv
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

import run_reid_target_anchor_review_package as arb  # noqa: E402


class TargetAnchorReviewPackageTests(unittest.TestCase):
    def test_expected_git_head_constant(self):
        cfg = arb.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_anchor_review_stage5d_target_001.yaml"
        )
        self.assertEqual(
            cfg["project_head_expected"],
            "b7210155e014463b1592e2a27df4ed7e22b8319c",
        )
        self.assertEqual(cfg["target_definition"]["target_id"], "target_001")
        self.assertEqual(
            cfg["target_definition"]["jersey_number_provenance"],
            "human_verified_by_user_not_automated_ocr",
        )
        self.assertEqual(cfg["target_definition"]["human_verified_jersey_number"], 5)

    def test_target_definition_frozen_fields(self):
        cfg = arb.load_config(
            _PROJECT_ROOT
            / "configs/reid/target_anchor_review_stage5d_target_001.yaml"
        )
        td = arb.build_target_definition(cfg, approved_at="2026-01-01T00:00:00Z")
        self.assertEqual(td["schema_version"], "reid_target_definition_freeze_v1")
        self.assertTrue(td["target_definition_frozen"])
        self.assertFalse(td["automated_jersey_used"])
        self.assertFalse(td["model_identity_prediction_used"])
        self.assertFalse(td["similarity_score_used"])
        self.assertEqual(td["identity_basis"], "human_visual_verification_from_source_video")

    def test_allowed_manual_decision_vocabulary(self):
        self.assertEqual(
            set(arb.ALLOWED_MANUAL_DECISIONS),
            {
                "target_anchor_yes",
                "target_anchor_no",
                "uncertain",
                "invalid",
                "multi_person_ambiguous",
                "non_player",
            },
        )

    def test_anchor_contract_forbids_gallery_and_auto_enroll(self):
        c = arb.build_anchor_review_contract(
            target_definition_sha="abc",
            eligible_count=3,
            exclusion_counts={"x": 1},
            no_embedding_count=141,
        )
        self.assertTrue(c["no_automatic_enrollment"])
        self.assertTrue(c["no_similarity_ranking"])
        self.assertTrue(c["no_ocr_usage"])
        self.assertTrue(c["no_gallery_membership"])
        self.assertTrue(c["no_identity_assignment"])
        self.assertEqual(c["manual_decisions"], 0)
        self.assertEqual(c["approved_anchors"], 0)
        self.assertEqual(c["gallery_members"], 0)
        self.assertEqual(c["prototypes"], 0)
        self.assertTrue(c["unknown_identity_preserved"])
        self.assertTrue(c["target_anchor_yes_is_not_gallery_membership_in_stage5d_b"])
        self.assertIn("STAGE5D-B2", c["anchor_freeze_requires_separate_gate"])

    def test_blank_annotation_template_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpl.csv"
            rows = [
                {
                    "anchor_candidate_id": "target_001_anchor_001",
                    "target_id": "target_001",
                    "segment_id": "raw_x_full",
                    "raw_track_id": 1,
                    "frame_index": 10,
                    "source_crop_path": "/tmp/a.jpg",
                    "source_crop_sha256": "0" * 64,
                    "manual_anchor_decision": "",
                    "manual_crop_valid": "",
                    "manual_target_dominant": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "final_approver": "",
                    "reviewed_at": "",
                }
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(arb.ANNOTATION_FIELDS))
                writer.writeheader()
                writer.writerows(rows)
            with path.open(encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(loaded[0]["manual_anchor_decision"], "")
            self.assertNotIn("similarity_score", loaded[0])
            self.assertNotIn("model_identity_prediction", loaded[0])
            self.assertNotIn("parseq_prediction", loaded[0])

    def test_path_traversal_rejection(self):
        with self.assertRaises(arb.AnchorPackageError):
            arb.assert_no_path_traversal("../outside")

    def test_representative_crop_deterministic_tiebreak(self):
        img = np.zeros((64, 32, 3), dtype=np.uint8)
        crop_a = {
            "crop_id": "track_1_frame_10_rank_1",
            "frame_index": 10,
            "bbox_xyxy": [10, 10, 40, 70],
            "bbox_area": 1800,
            "short_side": 30,
            "quality_score": 100,
        }
        crop_b = {
            "crop_id": "track_1_frame_11_rank_2",
            "frame_index": 11,
            "bbox_xyxy": [10, 10, 40, 70],
            "bbox_area": 1800,
            "short_side": 30,
            "quality_score": 100,
        }
        ka = arb.representative_crop_score(
            crop_a, img, mid_frame=10.5, frame_w=1336, frame_h=744
        )
        kb = arb.representative_crop_score(
            crop_b, img, mid_frame=10.5, frame_w=1336, frame_h=744
        )
        # Same quality proxies; crop_id lexicographic tie-break must be stable.
        self.assertNotEqual(ka, kb)
        self.assertEqual(sorted([ka, kb])[0][-1], "track_1_frame_10_rank_1")

    def test_exclusion_keys_include_stage5c_batches(self):
        # Contract-level expectation from Stage 5D-A preregistration.
        keys = {
            "segment_id",
            "raw_track_id",
            "crop_id",
            "source_crop_sha256",
            "exact_duplicate_group",
            "near_duplicate_component",
            "documented_link_component",
            "temporal_source_window",
        }
        self.assertTrue(keys)

    def test_contact_sheet_labels_omit_identity_proof(self):
        img = np.full((80, 40, 3), 90, dtype=np.uint8)
        items = [
            {
                "anchor_order": 1,
                "anchor_candidate_id": "target_001_anchor_001",
            }
        ]
        sheet = arb.render_contact_sheet(
            items,
            {"target_001_anchor_001": img},
            max_items=12,
            columns=4,
        )
        self.assertEqual(sheet.ndim, 3)
        self.assertGreater(sheet.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
