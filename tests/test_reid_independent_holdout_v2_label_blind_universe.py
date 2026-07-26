"""Unit tests for Stage 5D-F3J holdout v2 label-blind universe."""

from __future__ import annotations

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

import run_reid_independent_holdout_v2_label_blind_universe as f3j  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_label_blind_universe_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_label_blind_universe"
)
_F3I = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_ingestion_and_preflight"
)
_B1EB = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
)
_HEAD = "e9123b1f5780716dda770b880c92983ab4aec5f6"
_HOLDOUT_SHA = "bbfe3669cfbf39534f71a80131401bb4d9f931c7a4ea485404ab2fa207a6231f"
_YOLO_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
_TRACKER_SHA = "b951014a9ef48b14eb4c13003d2b83c579b260abc9079156c986818898c8549b"
_F3I_SNAP = (
    "fbdcd4e55fc11dbbbf6f4226b145433fc874b06db272b4753703d3e598cf7de4"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HoldoutLabelBlindUniverseTests(unittest.TestCase):
    def test_expected_git_contract_fields(self):
        cfg = f3j.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            cfg["project_head_message_expected"],
            "Ingest target 001 independent holdout v2",
        )

    def test_f3i_status_readiness_and_snapshot(self):
        cfg = f3j.load_config(_CFG)
        self.assertEqual(
            cfg["stage5d_f3i_package"]["expected_final_status"],
            "COMPLETED_STAGE5D_F3I_TARGET_001_NEW_INDEPENDENT_HOLDOUT_INGESTED_AND_PREFLIGHT_PASSED",
        )
        self.assertEqual(
            cfg["stage5d_f3i_package"]["expected_readiness"],
            "TARGET_001_INDEPENDENT_HOLDOUT_V2_READY_FOR_LABEL_BLIND_UNIVERSE_BUILD",
        )
        self.assertEqual(
            cfg["stage5d_f3i_package"]["expected_snapshot_sha256"], _F3I_SNAP
        )
        if _F3I.is_dir():
            out = f3j.validate_f3i(_PROJECT_ROOT, cfg)
            self.assertEqual(out["f3i_sha"], _HOLDOUT_SHA)
            self.assertTrue(out["decision"]["label_blind_universe_build_eligible"])

    def test_holdout_path_sha_metadata(self):
        cfg = f3j.load_config(_CFG)
        h = cfg["holdout_source"]
        self.assertEqual(
            h["path"], "data/test_clips/target_001_independent_holdout_v2.mp4"
        )
        self.assertEqual(h["expected_sha256"], _HOLDOUT_SHA)
        self.assertEqual(h["expected_bytes"], 19569991)
        self.assertEqual(h["expected_frames"], 1058)
        self.assertEqual(h["expected_width"], 1336)
        self.assertEqual(h["expected_height"], 754)
        self.assertEqual(h["expected_fps"], 30.0)
        path = _PROJECT_ROOT / h["path"]
        if path.is_file():
            self.assertEqual(_sha256(path), _HOLDOUT_SHA)
            self.assertEqual(path.stat().st_size, 19569991)

    def test_frozen_detector_checkpoint_sha(self):
        cfg = f3j.load_config(_CFG)
        self.assertEqual(cfg["yolo_checkpoint"]["expected_sha256"], _YOLO_SHA)
        self.assertEqual(cfg["yolo_checkpoint"]["expected_bytes"], 5613764)
        model = _PROJECT_ROOT / cfg["yolo_checkpoint"]["path"]
        if model.is_file():
            self.assertEqual(_sha256(model), _YOLO_SHA)

    def test_upstream_detector_tracker_config_resolved(self):
        cfg = f3j.load_config(_CFG)
        self.assertEqual(cfg["detection"]["conf"], 0.25)
        self.assertEqual(cfg["detection"]["iou"], 0.70)
        self.assertEqual(cfg["detection"]["imgsz"], 640)
        self.assertEqual(cfg["detection"]["classes"], [0])
        self.assertEqual(cfg["detection"]["device"], "cpu")
        self.assertEqual(cfg["tracking"]["expected_tracker_sha256"], _TRACKER_SHA)
        if _B1EB.is_dir():
            out = f3j.validate_upstream_b1eb(_PROJECT_ROOT, cfg)
            self.assertEqual(
                out["summary"]["final_status"],
                "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY",
            )
            self.assertEqual(out["tracking"]["tracker_sha256"], _TRACKER_SHA)

    def test_segmentation_contract_resolved_objective_only(self):
        cfg = f3j.load_config(_CFG)
        seg = f3j.resolve_segmentation_contract(_PROJECT_ROOT, cfg)
        self.assertFalse(seg["automatic_track_split_enabled"])
        self.assertIsNone(seg["change_threshold"])
        self.assertIsNone(seg["split_threshold"])
        self.assertEqual(seg["mode"], "pass_through_raw_track")
        self.assertFalse(seg["cross_track_merge_enabled"])
        self.assertIn("raw_track_lineage", seg["allowed_objective_inputs"])
        self.assertIn("osnet_embedding", seg["forbidden_inputs"])

    def test_stable_detection_id_ordering(self):
        items = [
            {"bbox_xyxy": [10.0, 1.0, 20.0, 2.0], "confidence": 0.5, "detector_index": 0},
            {"bbox_xyxy": [1.0, 1.0, 2.0, 2.0], "confidence": 0.9, "detector_index": 1},
            {"bbox_xyxy": [1.0, 1.0, 2.0, 2.0], "confidence": 0.8, "detector_index": 2},
        ]
        ordered = sorted(enumerate(items), key=f3j.detection_sort_key)
        ranks = [items[i]["detector_index"] for i, _ in ordered]
        self.assertEqual(ranks[0], 1)  # same xy, higher conf first among equals? 
        # sort: x1,y1,x2,y2 asc then conf desc — both (1,1,2,2) before (10,...); among equals conf 0.9 then 0.8
        self.assertEqual([items[i]["confidence"] for i, _ in ordered], [0.9, 0.8, 0.5])

    def test_raw_track_and_segment_id_formats(self):
        self.assertEqual(f"H2_RAW_{7:06d}", "H2_RAW_000007")
        self.assertEqual(f"H2_OBS_{12:06d}_{7:06d}", "H2_OBS_000012_000007")
        self.assertEqual(f"H2_DET_{3:06d}_{1:03d}", "H2_DET_000003_001")
        self.assertEqual(f"H2_SEG_{1:06d}", "H2_SEG_000001")

    def test_pass_through_segments_cover_all_observations(self):
        observations = [
            {
                "observation_id": "H2_OBS_000001_000001",
                "raw_tracker_id": 1,
                "raw_track_code": "H2_RAW_000001",
                "frame_index": 1,
                "timestamp_sec": 1 / 30.0,
                "bbox_xyxy": [10.0, 10.0, 40.0, 50.0],
                "source_detection_confidence": 0.9,
                "edge_clipping": {
                    "left": False,
                    "top": False,
                    "right": False,
                    "bottom": False,
                    "any": False,
                },
            },
            {
                "observation_id": "H2_OBS_000002_000001",
                "raw_tracker_id": 1,
                "raw_track_code": "H2_RAW_000001",
                "frame_index": 2,
                "timestamp_sec": 2 / 30.0,
                "bbox_xyxy": [12.0, 10.0, 42.0, 50.0],
                "source_detection_confidence": 0.8,
                "edge_clipping": {
                    "left": False,
                    "top": False,
                    "right": False,
                    "bottom": False,
                    "any": False,
                },
            },
            {
                "observation_id": "H2_OBS_000001_000002",
                "raw_tracker_id": 2,
                "raw_track_code": "H2_RAW_000002",
                "frame_index": 1,
                "timestamp_sec": 1 / 30.0,
                "bbox_xyxy": [100.0, 10.0, 140.0, 50.0],
                "source_detection_confidence": 0.7,
                "edge_clipping": {
                    "left": False,
                    "top": False,
                    "right": False,
                    "bottom": False,
                    "any": False,
                },
            },
        ]
        inv = f3j.build_raw_track_inventory(observations, width=200, height=100)
        seg_obs, seg_inv, track_map, audit = f3j.build_segments(
            observations, inv, fps=30.0, width=200, height=100
        )
        self.assertEqual(len(seg_inv), 2)
        self.assertEqual(audit["pass_through_segment_count"], 2)
        self.assertEqual(audit["split_segment_count"], 0)
        self.assertEqual(audit["cross_track_merge_count"], 0)
        self.assertEqual(len(seg_obs), 3)
        self.assertEqual(len({r["observation_id"] for r in seg_obs}), 3)
        self.assertEqual(set(track_map.keys()), {"H2_RAW_000001", "H2_RAW_000002"})
        self.assertTrue(all(s["label_blind"] and s["target_label_absent"] for s in seg_inv))
        # segment IDs ordered by raw track id
        self.assertEqual(seg_inv[0]["segment_id"], "H2_SEG_000001")
        self.assertEqual(seg_inv[0]["raw_track_code"], "H2_RAW_000001")
        self.assertEqual(seg_inv[1]["raw_track_code"], "H2_RAW_000002")

    def test_review_eligibility_not_identity_and_retains_ineligible(self):
        segments = [
            {
                "segment_id": "H2_SEG_000001",
                "observation_count": 1,
            },
            {
                "segment_id": "H2_SEG_000002",
                "observation_count": 5,
            },
        ]
        seg_obs = [
            {
                "segment_id": "H2_SEG_000001",
                "observation_id": "a",
                "frame_index": 0,
                "bbox_xyxy": [1.0, 1.0, 10.0, 10.0],
                "confidence": 0.5,
            },
            {
                "segment_id": "H2_SEG_000002",
                "observation_id": "b",
                "frame_index": 1,
                "bbox_xyxy": [1.0, 1.0, 10.0, 10.0],
                "confidence": 0.9,
            },
            {
                "segment_id": "H2_SEG_000002",
                "observation_id": "c",
                "frame_index": 2,
                "bbox_xyxy": [1.0, 1.0, 12.0, 12.0],
                "confidence": 0.8,
            },
            {
                "segment_id": "H2_SEG_000002",
                "observation_id": "d",
                "frame_index": 3,
                "bbox_xyxy": [1.0, 1.0, 11.0, 11.0],
                "confidence": 0.7,
            },
        ]
        elig, reps, summary = f3j.build_review_eligibility(
            segments, seg_obs, width=100, height=100, min_obs=3, max_reps=3
        )
        self.assertEqual(summary["complete_universe_count"], 2)
        self.assertEqual(summary["review_eligible_count"], 1)
        self.assertEqual(summary["review_ineligible_count"], 1)
        self.assertTrue(summary["review_eligibility_is_not_identity"])
        self.assertTrue(all(e["complete_universe_member"] for e in elig))
        self.assertTrue(all(e["identity_absent"] for e in elig))
        self.assertTrue(all(r["crop_exported"] is False for r in reps))
        self.assertLessEqual(len(reps[1]["representative_frame_candidates"]), 3)

    def test_forbidden_identity_score_fields_absent(self):
        payload = {
            "label_blind": True,
            "confidence": 0.5,
            "segment_id": "H2_SEG_000001",
        }
        self.assertEqual(f3j.forbidden_field_audit(payload), [])
        bad = {"target_probability": 0.9, "rank": 1}
        hits = f3j.forbidden_field_audit(bad)
        self.assertIn("target_probability", hits)
        self.assertIn("rank", hits)

    def test_compare_passes_exact(self):
        base = {
            "detection_count": 1,
            "observation_count": 1,
            "raw_track_count": 1,
            "segment_count": 1,
            "detection_rows_sha": "a",
            "observation_rows_sha": "b",
            "segment_inventory_sha": "c",
            "eligibility_sha": "d",
            "rep_candidates_sha": "e",
            "detections": [
                {
                    "detection_id": "H2_DET_000000_000",
                    "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                    "confidence": 0.5,
                }
            ],
        }
        out = f3j.compare_passes(base, base, bbox_atol=1e-6, conf_atol=1e-7)
        self.assertTrue(out["exact_match"])
        self.assertEqual(out["maximum_numeric_absolute_difference"], 0.0)

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3j.HoldoutUniverseError):
            f3j.assert_no_path_traversal("../x")

    def test_access_audit_zeros(self):
        a = f3j.build_access_audit()
        self.assertFalse(a["sample_video_read"])
        self.assertFalse(a["external_video_read"])
        self.assertEqual(a["gallery_embedding_read_count"], 0)
        self.assertEqual(a["osnet_model_loads"], 0)
        self.assertEqual(a["parseq_model_loads"], 0)
        self.assertEqual(a["ocr_calls"], 0)
        self.assertEqual(a["team_classifier_calls"], 0)
        self.assertEqual(a["crop_exports"], 0)
        self.assertEqual(a["embeddings"], 0)
        self.assertEqual(a["identity_assignments"], 0)
        self.assertEqual(a["holdout_video_decode_passes"], 2)
        self.assertEqual(a["detector_inference_frames_total"], 2116)

    def test_live_final_package_if_present(self):
        if not _FINAL.is_dir():
            self.skipTest("F3J final package not built yet")
        summary = json.loads((_FINAL / "stage5d_f3j_summary.json").read_text())
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_F3J_TARGET_001_NEW_INDEPENDENT_HOLDOUT_LABEL_BLIND_UNIVERSE_BUILT",
        )
        self.assertEqual(
            summary["readiness"],
            "TARGET_001_INDEPENDENT_HOLDOUT_V2_LABEL_BLIND_UNIVERSE_READY_FOR_GROUND_TRUTH_REVIEW_PACKAGE",
        )
        self.assertEqual(summary["frames_processed"], 1058)
        self.assertEqual(summary["frames_expected"], 1058)
        self.assertEqual(summary["detector_input_source_count"], 1)
        self.assertGreater(summary["person_detection_count"], 0)
        self.assertGreater(summary["raw_track_count"], 0)
        self.assertGreater(summary["segment_count"], 0)
        self.assertTrue(summary["replay_deterministic"])
        self.assertEqual(summary["label_blind_forbidden_fields"], 0)
        self.assertEqual(summary["gallery_reads"], 0)
        self.assertEqual(summary["osnet_loads"], 0)
        self.assertEqual(summary["ocr_calls"], 0)
        self.assertEqual(summary["team_classifier_calls"], 0)
        self.assertEqual(summary["crop_exports"], 0)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["similarity_rows"], 0)
        self.assertEqual(summary["rankings"], 0)
        self.assertEqual(summary["gt_labels"], 0)
        self.assertEqual(summary["metrics"], 0)
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertEqual(
            summary["assigned_detection_count"] + summary["unassigned_detection_count"],
            summary["person_detection_count"],
        )
        self.assertEqual(summary["complete_universe_count"], summary["segment_count"])
        self.assertEqual(
            summary["review_eligible_count"] + summary["review_ineligible_count"],
            summary["segment_count"],
        )
        # no media artifacts
        for p in _FINAL.rglob("*"):
            if p.is_file():
                self.assertNotIn(p.suffix.lower(), {".mp4", ".png", ".jpg", ".jpeg", ".npy"})
        # contracts pre-inference present
        self.assertTrue(
            (
                _FINAL
                / "runtime"
                / "target_001_holdout_v2_detector_tracker_contract_pre_inference.json"
            ).is_file()
        )
        self.assertTrue(
            (
                _FINAL
                / "runtime"
                / "target_001_holdout_v2_label_blind_segmentation_contract_pre_build.json"
            ).is_file()
        )
        replay = json.loads(
            (_FINAL / "runtime" / "pass2_replay_comparison.json").read_text()
        )
        self.assertTrue(replay["exact_match"])
        sha_ck = json.loads(
            (_FINAL / "runtime" / "source_sha_checkpoints.json").read_text()
        )
        self.assertTrue(sha_ck["unchanged"])
        self.assertEqual(sha_ck["before_inference"], _HOLDOUT_SHA)
        # holdout unchanged
        holdout = _PROJECT_ROOT / "data/test_clips/target_001_independent_holdout_v2.mp4"
        self.assertEqual(_sha256(holdout), _HOLDOUT_SHA)
        # detection ID coverage sample
        det_path = _FINAL / "detections" / "target_001_holdout_v2_person_detections.jsonl"
        with det_path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        self.assertTrue(first["detection_id"].startswith("H2_DET_"))
        self.assertEqual(first["class_name"], "person")
        self.assertTrue(first["label_blind"])
        # segmentation no identity
        seg_path = (
            _FINAL
            / "segmentation"
            / "target_001_holdout_v2_label_blind_segment_inventory.jsonl"
        )
        with seg_path.open(encoding="utf-8") as handle:
            seg = json.loads(handle.readline())
        self.assertTrue(seg["label_blind"])
        self.assertTrue(seg["target_label_absent"])
        self.assertTrue(seg["gallery_score_absent"])
        for key in f3j.FORBIDDEN_FIELDS:
            self.assertNotIn(key, seg)


if __name__ == "__main__":
    unittest.main()
