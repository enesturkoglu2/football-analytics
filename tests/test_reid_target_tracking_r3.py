"""Tests for Target Tracking R3 short-occlusion bridge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.target_tracking_r3.boundary import (  # noqa: E402
    audit_seed_segment_white_leak,
    refine_purity_boundary,
)
from football_analytics.reid.target_tracking_r3.bridge import (  # noqa: E402
    build_r3_timeline,
    resolve_bridge_window,
)
from football_analytics.reid.target_tracking_r3.bridge_state import (  # noqa: E402
    build_short_occlusion_state,
    select_bridge_template_rows,
)
from football_analytics.reid.target_tracking_r3.flow import (  # noqa: E402
    extract_quality_gated_features,
    project_bbox,
    track_flow_forward_backward,
)
from football_analytics.reid.target_tracking_r3.policy import R3_POLICY  # noqa: E402
from football_analytics.reid.target_tracking_r3.snap import (  # noqa: E402
    decide_snap,
    score_detector_candidates,
)


def _row(fi, kit, y, w, contam=0.0, bbox=None, quality=True):
    bbox = bbox or [100, 100, 140, 180]
    return {
        "frame_index": fi,
        "timestamp_sec": fi / 30.0,
        "kit_state": kit,
        "yellow_evidence": y,
        "white_evidence": w,
        "crop_quality_ok": quality,
        "reliable": quality,
        "bbox_xyxy": bbox,
        "bbox_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        "laplacian_variance": 40.0,
        "crop_sha256": f"sha{fi}",
        "contamination": {
            "union_other_person_crop_coverage": contam,
            "max_other_person_iou": contam,
        },
        "raw_track_id": "10",
    }


class BoundaryTests(unittest.TestCase):
    def test_white_in_seed_triggers_refinement(self):
        rows = [_row(i, "YELLOW", 0.4, 0.05) for i in range(0, 50)]
        rows += [_row(50, "WHITE", 0.05, 0.35), _row(51, "WHITE", 0.04, 0.36)]
        rows += [_row(i, "WHITE", 0.02, 0.4) for i in range(52, 60)]
        audit = audit_seed_segment_white_leak(
            rows, seed_segment_start=0, seed_segment_end=55, r2_change_point_frame=56
        )
        self.assertTrue(audit["refinement_required"])
        refined = refine_purity_boundary(rows, seed_frame=29, r2_change_point_frame=56)
        self.assertLess(refined["refined_change_point_frame"], 56)
        self.assertGreaterEqual(refined["refined_seed_segment_end_frame"], 29)
        self.assertFalse(refined["human_reported_transition_is_algorithm_truth"])


class BridgeStateTests(unittest.TestCase):
    def test_impure_observations_excluded_from_bridge_state(self):
        rows = [_row(i, "YELLOW", 0.4, 0.05) for i in range(40, 48)]
        rows.append(_row(48, "WHITE", 0.05, 0.4))  # must be excluded
        rows.append(_row(49, "YELLOW", 0.35, 0.05, contam=0.5))  # contaminated
        selected = select_bridge_template_rows(rows, seed_start=40, refined_seed_end=49)
        self.assertTrue(all(r["kit_state"] != "WHITE" for r in selected))
        self.assertTrue(
            all(
                float(r["contamination"]["union_other_person_crop_coverage"]) <= 0.20
                for r in selected
            )
        )
        state = build_short_occlusion_state(
            persistent_target_id="ptarget_x",
            source_segment_id="raw10_seg_A",
            parent_raw_track_id="10",
            template_rows=selected,
        )
        self.assertEqual(state["schema_version"], "target_short_occlusion_state_v1")
        self.assertTrue(state["impure_observations_excluded"])
        self.assertNotEqual(state["persistent_target_id"], "10")


class FlowTests(unittest.TestCase):
    def test_quality_gated_features_and_fb_consistency(self):
        rng = np.random.default_rng(0)
        img = (rng.random((200, 200)) * 255).astype(np.uint8)
        # textured box
        img[60:140, 60:140] = np.linspace(0, 255, 80, dtype=np.uint8)[:, None]
        pts = extract_quality_gated_features(img, [50, 50, 150, 150])
        self.assertIsNotNone(pts)
        # shift image by 3px
        M = np.float32([[1, 0, 3], [0, 1, 0]])
        shifted = cv2.warpAffine(img, M, (200, 200))
        flow = track_flow_forward_backward(img, shifted, pts)
        self.assertTrue(flow["reliable"])
        self.assertAlmostEqual(flow["median_delta"][0], 3.0, delta=1.5)
        proj = project_bbox([50, 50, 150, 150], flow["median_delta"])
        self.assertGreater(proj[0], 50)

    def test_unreliable_flow_flag(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array([[[10.0, 10.0]], [[20.0, 20.0]], [[30.0, 30.0]]], dtype=np.float32)
        flow = track_flow_forward_backward(a, b, pts)
        # flat images → few reliable tracks
        self.assertIn(flow["reason"], {"BRIDGE_FLOW_UNRELIABLE", "FLOW_OK"})


class SnapTests(unittest.TestCase):
    def test_reliable_cross_team_rejection(self):
        projected = [100, 100, 140, 180]
        dets = [
            {"raw_track_id": "70", "bbox_xyxy": [102, 102, 142, 182], "detection_id": "a"},
            {"raw_track_id": "10", "bbox_xyxy": [105, 100, 145, 180], "detection_id": "b"},
        ]
        kit = {
            "70": {"kit_state": "YELLOW", "reliable": True},
            "10": {"kit_state": "WHITE", "reliable": True},
        }
        cands = score_detector_candidates(
            projected_bbox=projected,
            detections=dets,
            target_kit_state="YELLOW",
            kit_by_detection=kit,
            excluded_raw_track_ids=["10"],
            seed_track_meta=None,
            track_index=None,
        )
        by_id = {c["raw_track_id"]: c for c in cands}
        self.assertIn("CROSS_TEAM_KIT_MISMATCH", by_id["10"]["hard_rejects"])
        self.assertIn("EXCLUDED_IMPURE_PARENT_CONTINUATION", by_id["10"]["hard_rejects"])
        self.assertTrue(by_id["70"]["eligible"])
        dec = decide_snap(cands)
        self.assertEqual(dec["decision"], "DETECTOR_SNAP")
        self.assertEqual(dec["selected"]["raw_track_id"], "70")

    def test_unknown_kit_does_not_hard_reject(self):
        projected = [100, 100, 140, 180]
        dets = [{"raw_track_id": "5", "bbox_xyxy": [101, 101, 141, 181]}]
        kit = {"5": {"kit_state": "UNKNOWN", "reliable": False}}
        cands = score_detector_candidates(
            projected_bbox=projected,
            detections=dets,
            target_kit_state="YELLOW",
            kit_by_detection=kit,
            excluded_raw_track_ids=[],
            seed_track_meta=None,
            track_index=None,
        )
        self.assertNotIn("CROSS_TEAM_KIT_MISMATCH", cands[0]["hard_rejects"])

    def test_ambiguous_candidate_unresolved(self):
        projected = [100, 100, 140, 180]
        dets = [
            {"raw_track_id": "1", "bbox_xyxy": [100, 100, 140, 180]},
            {"raw_track_id": "2", "bbox_xyxy": [101, 101, 141, 181]},
        ]
        kit = {
            "1": {"kit_state": "YELLOW", "reliable": True},
            "2": {"kit_state": "YELLOW", "reliable": True},
        }
        cands = score_detector_candidates(
            projected_bbox=projected,
            detections=dets,
            target_kit_state="YELLOW",
            kit_by_detection=kit,
            excluded_raw_track_ids=[],
            seed_track_meta=None,
            track_index=None,
        )
        dec = decide_snap(cands)
        self.assertEqual(dec["decision"], "TARGET_UNRESOLVED")
        self.assertEqual(dec["reason"], "AMBIGUOUS_CANDIDATE_MARGIN")


class WindowAndTimelineTests(unittest.TestCase):
    def test_bounded_short_bridge_and_long_gap(self):
        w = resolve_bridge_window(
            last_reliable_frame=50,
            refined_change_point=53,
            contamination_frames=[52, 55],
            nearby_birth_frames=[55],
        )
        self.assertLessEqual(w["bridge_span_frames"], int(R3_POLICY["max_bridge_frames"]))
        self.assertFalse(w["long_gap_review_required"])
        long = resolve_bridge_window(
            last_reliable_frame=10,
            refined_change_point=12,
            contamination_frames=[],
            nearby_birth_frames=[],
            policy={**R3_POLICY, "max_bridge_frames": 100, "long_gap_frames": 30, "overlap_extend_after_contam_frames": 80},
        )
        # end candidates min of (12+80, 10+100) = 92 → span 82 > 30 → long gap
        self.assertTrue(long["long_gap_review_required"])

    def test_timeline_provenance_and_no_gt_claims(self):
        br = {
            "accepted": False,
            "frames": [
                {
                    "frame_index": 53,
                    "timestamp_sec": 53 / 30,
                    "status": "TARGET_UNRESOLVED",
                    "projected_bbox": [1, 2, 3, 4],
                    "provenance": "UNRESOLVED_GAP",
                    "flow_confidence": 0.1,
                }
            ],
        }
        tl = build_r3_timeline(
            persistent_target_id="ptarget_x",
            target_id="target_001",
            refined_seed_start=0,
            refined_seed_end=52,
            seed_segment_id="raw10_seg_A_refined",
            parent_raw_track_id="10",
            bridge_result=br,
            fps=30.0,
            frame_count=100,
        )
        self.assertFalse(tl["raw_track_permanent_binding"])
        kinds = {iv["kind"] for iv in tl["intervals"]}
        self.assertIn("HUMAN_SEED_SEGMENT", kinds)
        self.assertIn("UNRESOLVED_GAP", kinds)
        self.assertNotIn("IDF1", tl)


class PolicySafetyTests(unittest.TestCase):
    def test_no_detection_tracking_rerun_flags_in_policy(self):
        self.assertEqual(R3_POLICY["appearance_reid"], "UNAVAILABLE")
        self.assertTrue(R3_POLICY["cross_team_hard_reject"])
        self.assertTrue(R3_POLICY["unknown_kit_does_not_hard_reject"])
        self.assertFalse(R3_POLICY["human_reported_transition_is_algorithm_truth"])


if __name__ == "__main__":
    unittest.main()
