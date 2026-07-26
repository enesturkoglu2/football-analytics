"""Unit tests for Stage 5D-F3K holdout v2 ground-truth review package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_independent_holdout_v2_ground_truth_review_package as f3k  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_ground_truth_review_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ground_truth_review_package"
)
_F3J = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
)
_HEAD = "09f604fa1feceab3e91d1fcc0bf69cec93dd18a3"
_HOLDOUT_SHA = "bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f"
_F3J_SNAP = "5af2ff6d0549d6f136d8bcfe96fbe63505c6517a4e9b469378ca869dac762022"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HoldoutGroundTruthReviewTests(unittest.TestCase):
    def test_expected_git_contract_fields(self):
        cfg = f3k.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Build target 001 holdout v2 label-blind universe",
        )

    def test_f3j_status_readiness_and_snapshot(self):
        cfg = f3k.load_config(_CFG)
        self.assertEqual(
            cfg["stage5d_f3j_package"]["expected_final_status"],
            "COMPLETED_STAGE5D_F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_UNIVERSE_BUILT",
        )
        self.assertEqual(
            cfg["stage5d_f3j_package"]["expected_snapshot_sha256"], _F3J_SNAP
        )
        if _F3J.is_dir():
            out = f3k.validate_f3j(_PROJECT_ROOT, cfg)
            self.assertEqual(out["snapshot_sha256"], _F3J_SNAP)
            self.assertEqual(out["summary"]["complete_universe_count"], 243)
            self.assertEqual(out["summary"]["review_eligible_count"], 141)
            self.assertEqual(out["summary"]["review_ineligible_count"], 102)

    def test_holdout_sha_metadata(self):
        cfg = f3k.load_config(_CFG)
        h = cfg["holdout_source"]
        self.assertEqual(h["expected_sha256"], _HOLDOUT_SHA)
        self.assertEqual(h["expected_frames"], 1058)
        path = _PROJECT_ROOT / h["path"]
        if path.is_file():
            self.assertEqual(_sha256(path), _HOLDOUT_SHA)

    def test_universe_counts_frozen(self):
        self.assertEqual(f3k.UNIVERSE_N, 243)
        self.assertEqual(f3k.ELIGIBLE_N, 141)
        self.assertEqual(f3k.INELIGIBLE_N, 102)
        self.assertEqual(f3k.SHEET_COUNT, 12)
        self.assertEqual(f3k.ITEMS_PER_FULL_SHEET, 12)
        self.assertEqual(f3k.LAST_SHEET_ITEMS, 9)
        self.assertEqual(f3k.VIDEO_PARTS, 3)
        self.assertEqual(f3k.ITEMS_PER_PART, 47)
        self.assertEqual(11 * 12 + 9, 141)
        self.assertEqual(3 * 47, 141)

    def test_stable_review_id_ordering(self):
        self.assertEqual(f"H2_GT_REVIEW_{1:06d}", "H2_GT_REVIEW_000001")
        self.assertEqual(f"H2_GT_REVIEW_{141:06d}", "H2_GT_REVIEW_000141")

    def test_context_frames_are_real_observations(self):
        frames = [10, 12, 15, 20, 30]
        ctx = f3k.select_context_frames(frames, start_frame=10, end_frame=30)
        roles = [r for r, _ in ctx]
        self.assertEqual(roles, ["START", "MIDDLE", "END"])
        self.assertEqual(ctx[0][1], 10)
        self.assertEqual(ctx[2][1], 30)
        self.assertIn(ctx[1][1], frames)

    def test_clip_window_bounds(self):
        a, b = f3k.compute_clip_window(
            start_frame=100,
            end_frame=105,
            representative_frame=102,
            total_frames=1058,
            fps=30.0,
            pre_pad_sec=0.5,
            post_pad_sec=0.5,
            min_clip_sec=1.0,
            max_clip_sec=4.0,
        )
        self.assertGreaterEqual(b - a + 1, 30)
        self.assertLessEqual(b - a + 1, 120)
        c, d = f3k.compute_clip_window(
            start_frame=0,
            end_frame=400,
            representative_frame=200,
            total_frames=1058,
            fps=30.0,
            pre_pad_sec=0.5,
            post_pad_sec=0.5,
            min_clip_sec=1.0,
            max_clip_sec=4.0,
        )
        self.assertEqual(d - c + 1, 120)

    def test_decision_vocabulary_exact(self):
        self.assertEqual(
            set(f3k.GT_DECISION_VOCAB),
            {
                "target_occurrence_yes",
                "target_occurrence_no",
                "uncertain",
                "invalid",
                "multi_person_ambiguous",
                "non_player",
            },
        )
        self.assertEqual(set(f3k.SAME_TEAM_VOCAB), {"yes", "no", "uncertain"})

    def test_template_fields_exact(self):
        self.assertIn("manual_ground_truth_decision", f3k.TEMPLATE_FIELDS)
        self.assertIn("manual_same_team_as_target", f3k.TEMPLATE_FIELDS)
        self.assertIn("manual_visible_jersey_number", f3k.TEMPLATE_FIELDS)
        self.assertIn("jersey_number_provenance", f3k.TEMPLATE_FIELDS)
        self.assertIn("tracker_native_id", f3k.TEMPLATE_FIELDS)
        self.assertNotIn("manual_view_category", f3k.TEMPLATE_FIELDS)

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3k.GroundTruthReviewError):
            f3k.assert_no_path_traversal("../x")

    def test_access_audit_zeros(self):
        a = f3k.build_access_audit(holdout_decode_frames=10)
        self.assertFalse(a["sample_video_read"])
        self.assertFalse(a["external_video_read"])
        self.assertEqual(a["gallery_embedding_read_count"], 0)
        self.assertEqual(a["osnet_model_loads"], 0)
        self.assertEqual(a["ocr_calls"], 0)
        self.assertEqual(a["detection_inference_passes"], 0)
        self.assertEqual(a["tracker_passes"], 0)
        self.assertEqual(a["segmentation_passes"], 0)

    def test_forbidden_fields_absent_probe(self):
        self.assertEqual(f3k.forbidden_field_audit({"label_blind": True}), [])
        hits = f3k.forbidden_field_audit({"target_probability": 0.1, "rank": 2})
        self.assertIn("target_probability", hits)

    def test_crop_padding_fraction_frozen(self):
        cfg = f3k.load_config(_CFG)
        self.assertEqual(cfg["crop_extraction"]["padding_fraction"], 0.05)
        self.assertEqual(cfg["crop_extraction"]["max_padding_fraction"], 0.05)

    def test_live_final_package_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3K final package not built yet")
        summary = json.loads((_FINAL / "stage5d_f3k_summary.json").read_text())
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_F3K_TARGET_001_NEW_INDEPENDENT_HOLDOUT_GROUND_TRUTH_REVIEW_PACKAGE_READY",
        )
        self.assertEqual(summary["complete_universe"], 243)
        self.assertEqual(summary["review_eligible"], 141)
        self.assertEqual(summary["review_ineligible"], 102)
        self.assertEqual(summary["review_item_count"], 141)
        self.assertEqual(summary["representative_crop_count"], 141)
        self.assertEqual(summary["contact_sheets"], 12)
        self.assertEqual(summary["review_videos"], 3)
        self.assertEqual(summary["manual_decision_count"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["gallery_reads"], 0)
        self.assertEqual(summary["similarity_rows"], 0)

        crops = list((_FINAL / "crops").glob("H2_GT_REVIEW_*__*__rep.png"))
        self.assertEqual(len(crops), 141)
        sheets = list(
            (
                _FINAL
                / "review_packages"
                / "target_001_holdout_v2_ground_truth_review"
            ).glob("target_001_holdout_v2_gt_review_sheet_*.png")
        )
        self.assertEqual(len(sheets), 12)
        videos = list((_FINAL / "videos").glob("target_001_holdout_v2_gt_review_part_*.mp4"))
        self.assertEqual(len(videos), 3)

        with (
            _FINAL / "templates" / "target_001_holdout_v2_ground_truth_review_template.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 141)
        for field in (
            "manual_ground_truth_decision",
            "reviewer",
            "final_approver",
            "reviewed_at",
        ):
            self.assertEqual(sum(1 for r in rows if (r.get(field) or "").strip()), 0)

        inelig = list(
            (
                _FINAL
                / "exclusions"
                / "target_001_holdout_v2_review_ineligible_segment_inventory.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(len(inelig), 102)
        for line in inelig:
            row = json.loads(line)
            self.assertFalse(row["automatic_negative"])
            self.assertFalse(row["metric_inclusion"])

        holdout = _PROJECT_ROOT / "data/test_clips/target_001_independent_holdout_v2.mp4"
        self.assertEqual(_sha256(holdout), _HOLDOUT_SHA)
        npy = list(_FINAL.rglob("*.npy"))
        self.assertEqual(npy, [])


if __name__ == "__main__":
    unittest.main()
