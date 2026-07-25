"""Unit tests for Stage 5D-B1E-B external tracking seed review helpers."""

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

import run_reid_external_tracking_seed_review_package as b1eb  # noqa: E402

_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
)
_PREFLIGHT = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_enrollment_preflight"
)
_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_tracking_seed_review_stage5d_target_001.yaml"
)
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"
_YOLO_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
_HEAD = "62f8120be63a66adc2627fa178e8de18c10de2fc"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ExternalTrackingSeedReviewTests(unittest.TestCase):
    def test_expected_git_contract(self):
        cfg = b1eb.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)

    def test_b1e_a_preflight_validation_fields(self):
        cfg = b1eb.load_config(_CFG)
        self.assertEqual(
            cfg["stage5d_b1e_a_package"]["expected_final_status"],
            "COMPLETED_STAGE5D_B1E_A_EXTERNAL_ENROLLMENT_PREFLIGHT_READY",
        )
        self.assertEqual(
            cfg["stage5d_b1e_a_package"]["expected_snapshot_sha256"],
            "0a5b53167b5972cf1a92e873c219d4c2a96eb8397017a3c89ff462c25c4fd2a5",
        )

    def test_external_source_sha_and_eligible_interval(self):
        cfg = b1eb.load_config(_CFG)
        self.assertEqual(cfg["external_enrollment_source"]["expected_sha256"], _EXT_SHA)
        self.assertEqual(cfg["external_enrollment_source"]["expected_bytes"], 14366504)
        self.assertEqual(cfg["eligible_interval"]["start_frame"], 0)
        self.assertEqual(cfg["eligible_interval"]["end_frame"], 783)

    def test_canonical_yolo_config_and_checkpoint_sha(self):
        cfg = b1eb.load_config(_CFG)
        self.assertEqual(cfg["detection"]["conf"], 0.25)
        self.assertEqual(cfg["detection"]["iou"], 0.70)
        self.assertEqual(cfg["detection"]["imgsz"], 640)
        self.assertEqual(cfg["detection"]["classes"], [0])
        self.assertEqual(cfg["detection"]["device"], "cpu")
        self.assertEqual(cfg["yolo_checkpoint"]["expected_sha256"], _YOLO_SHA)

    def test_canonical_bytetrack_config_resolution(self):
        cfg = b1eb.load_config(_CFG)
        tracker = _PROJECT_ROOT / cfg["tracking"]["tracker_path"]
        self.assertTrue(tracker.is_file())
        self.assertEqual(
            cfg["tracking"]["expected_tracker_sha256"],
            "b951014a9ef48b14eb4c13003d2b83c579b260abc9079156c986818898c8549b",
        )

    def test_stable_unique_ext_codes_and_ordering(self):
        tracks = {
            10: [
                {"frame_index": 20, "bbox_xyxy": [100, 0, 120, 10]},
                {"frame_index": 21, "bbox_xyxy": [100, 0, 120, 10]},
            ],
            3: [
                {"frame_index": 5, "bbox_xyxy": [50, 0, 70, 10]},
                {"frame_index": 6, "bbox_xyxy": [50, 0, 70, 10]},
            ],
            7: [
                {"frame_index": 5, "bbox_xyxy": [10, 0, 30, 10]},
                {"frame_index": 8, "bbox_xyxy": [10, 0, 30, 10]},
            ],
        }
        a = b1eb.assign_ext_codes(tracks)
        b = b1eb.assign_ext_codes(tracks)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a.values())), 3)
        self.assertTrue(all(v.startswith("EXT_") for v in a.values()))
        self.assertEqual(a[7], "EXT_001")
        self.assertEqual(a[3], "EXT_002")
        self.assertEqual(a[10], "EXT_003")

    def test_observations_fingerprint_stable(self):
        obs = [
            {
                "frame_index": 1,
                "raw_track_id": 2,
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.5,
            }
        ]
        self.assertEqual(
            b1eb.observations_fingerprint(obs), b1eb.observations_fingerprint(obs)
        )

    def test_contract_forbids_auto_embeddings_team_filter(self):
        c = b1eb.build_contract()
        self.assertTrue(c["no_osnet"])
        self.assertTrue(c["no_ocr"])
        self.assertTrue(c["no_similarity"])
        self.assertTrue(c["no_automatic_target_selection"])
        self.assertTrue(c["no_team_or_jersey_filtering"])
        self.assertTrue(c["no_bbox_interpolation"])
        self.assertTrue(c["two_replay_determinism_required"])
        self.assertEqual(c["manual_selections"], 0)
        self.assertEqual(c["gallery_members"], 0)
        self.assertEqual(c["anchors"], 0)
        self.assertEqual(c["prototypes"], 0)
        self.assertEqual(c["identity_assignments"], 0)
        self.assertTrue(c["multiple_positive_occurrences_allowed_later"])

    def test_blank_template_vocab_and_multiple_positive_support(self):
        self.assertIn("target_occurrence_yes", b1eb.ALLOWED_OCCURRENCE)
        self.assertIn("target_occurrence_no", b1eb.ALLOWED_OCCURRENCE)
        self.assertEqual(set(b1eb.ALLOWED_TRISTATE), {"yes", "no", "uncertain"})
        self.assertTrue(b1eb.build_contract()["multiple_positive_occurrences_allowed_later"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1eb.ExternalTrackingError):
            b1eb.assert_no_path_traversal("../x")

    def test_live_preflight_if_present(self):
        if not _PREFLIGHT.is_dir():
            self.skipTest("B1E-A absent")
        summary = json.loads(
            (_PREFLIGHT / "stage5d_b1e_a_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1E_A_EXTERNAL_ENROLLMENT_PREFLIGHT_READY",
        )
        self.assertEqual(summary.get("target_id", "target_001"), "target_001")

    def test_live_output_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("B1E-B output absent")
        summary = json.loads(
            (_FINAL / "stage5d_b1e_b_summary.json").read_text(encoding="utf-8")
        )
        contract = json.loads(
            (_FINAL / "stage5d_b1e_b_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY",
        )
        self.assertEqual(summary["eligible_interval"], [0, 783])
        self.assertTrue(summary["two_replay_determinism"])
        self.assertEqual(summary["detection_frames_with_boxes"], 784)
        self.assertEqual(summary["manual_selections"], 0)
        self.assertEqual(summary["approved_target_tracklets"], 0)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["osnet_inference"], 0)
        self.assertEqual(summary["ocr"], 0)
        self.assertEqual(summary["similarity_ranking_rows"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["derived_anchors"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["frozen_seed_used_for_tracking_or_ranking"])
        self.assertEqual(summary["temporal_overview_png_count"], 4)
        self.assertEqual(summary["annotated_mp4_count"], 1)
        self.assertEqual(contract["manual_selections"], 0)
        self.assertEqual(len(list(_FINAL.rglob("*.mp4"))), 1)
        review = _FINAL / "review_packages" / "target_001_external_seed_review"
        temporal = list(review.glob("temporal_overview_sheet_*.png"))
        self.assertEqual(len(temporal), 4)
        mapping = [
            json.loads(line)
            for line in (
                _FINAL / "inventory" / "target_001_external_track_candidate_mapping.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        codes = [m["external_candidate_code"] for m in mapping]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(codes), summary["ext_candidate_count"])
        self.assertTrue(all(c.startswith("EXT_") for c in codes))
        self.assertTrue(all(m["manual_fields_blank"] for m in mapping))
        # Stable EXT per raw track; unique codes.
        raw_ids = [m["raw_external_track_id"] for m in mapping]
        self.assertEqual(len(raw_ids), len(set(raw_ids)))
        # Deterministic ordering by first_frame then x-center then id.
        for i in range(1, len(mapping)):
            a, b = mapping[i - 1], mapping[i]
            self.assertLessEqual(a["first_frame"], b["first_frame"])
        eligible = [m for m in mapping if m["review_eligible"]]
        self.assertEqual(len(eligible), summary["review_eligible_candidate_count"])
        index_sheets = list((review / "candidate_index").glob("candidate_index_sheet_*.png"))
        expected_sheets = (len(eligible) + 11) // 12
        self.assertEqual(len(index_sheets), expected_sheets)
        self.assertEqual(len(index_sheets), summary["candidate_index_png_count"])
        # No raw track id / target suggestion in review filenames.
        for path in review.rglob("*"):
            name = path.name.lower()
            self.assertNotIn("track_id", name)
            self.assertNotIn("raw_", name)
            self.assertNotIn("jersey", name)
            self.assertNotIn("osnet", name)
        mp4_manifest = json.loads(
            (review / "target_001_external_tracking_seed_review_mp4_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(mp4_manifest["frame_count"], 784)
        self.assertEqual(mp4_manifest["fps"], 30.0)
        self.assertIn("sha256", mp4_manifest)
        self.assertIn("candidate_code_coverage", mp4_manifest)
        tpl = _FINAL / "templates" / "target_001_external_seed_manual_review_template.csv"
        with tpl.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(mapping))
        self.assertTrue(all(r["manual_occurrence_decision"] == "" for r in rows))
        self.assertTrue(all(r["manual_same_target_as_target_001"] == "" for r in rows))
        self.assertTrue(all(r["manual_human_verified_number_seen"] == "" for r in rows))
        self.assertTrue(all(r["reviewer"] == "" for r in rows))
        # No source video copy under output.
        for path in _FINAL.rglob("*.mp4"):
            self.assertIn("review", path.name.lower())
        # External source immutability.
        ext = _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
        self.assertEqual(ext.stat().st_size, 14366504)
        self.assertEqual(_sha256(ext), _EXT_SHA)
        yolo = _PROJECT_ROOT / "models/yolo11n.pt"
        self.assertEqual(_sha256(yolo), _YOLO_SHA)

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1eb.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1eb, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1eb.ExternalTrackingError) as ctx:
                    b1eb.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
