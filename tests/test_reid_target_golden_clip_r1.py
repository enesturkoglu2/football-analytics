"""Tests for Target Golden Clip R1 ground-truth annotation + metrics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.golden_clip import NOT_MEASURABLE_WITHOUT_GT  # noqa: E402
from football_analytics.reid.golden_clip.annotation_log import (  # noqa: E402
    AnnotationLog,
    assert_not_product_log,
)
from football_analytics.reid.golden_clip.intervals import (  # noqa: E402
    can_merge_intervals,
    merge_intervals,
    rebuild_gt_from_events,
    split_interval_at_frame,
)
from football_analytics.reid.golden_clip.metric_validity import (  # noqa: E402
    build_metric_validity_manifest,
)
from football_analytics.reid.golden_clip.metrics import (  # noqa: E402
    evaluate_variant_against_gt,
    metrics_without_gt,
    select_best_variant,
)
from football_analytics.reid.golden_clip.overlay import render_gt_overlay_video  # noqa: E402
from football_analytics.reid.golden_clip.schema import (  # noqa: E402
    build_annotation_event,
    build_annotation_interval,
    empty_ground_truth,
)
from football_analytics.reid.golden_clip.validate import validate_ground_truth  # noqa: E402


SHA = "f2f6d8a077ca2908bbd661753724cd6d457c62cffe39d3e9d5322a1a146897f5"


def _base_gt(**kwargs):
    return empty_ground_truth(
        source_video_sha256=SHA,
        match_id="match_short_video_f2f6d8a077ca",
        analysis_run_id="sv_run_20260727T234854Z",
        target_id="target_001",
        annotation_session_id="gtsess_test",
        reviewer="tester",
        frame_count=kwargs.get("frame_count", 30),
        fps=10.0,
    )


class AppendOnlyAnnotationTests(unittest.TestCase):
    def test_append_only_and_refuse_product_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation_log.jsonl"
            log = AnnotationLog(path)
            iv = build_annotation_interval(
                start_frame=0,
                end_frame=9,
                fps=10.0,
                target_state="TARGET_VISIBLE_ASSOCIATED",
                associated_raw_track_ids=["7"],
            )
            ev = build_annotation_event(
                action="APPEND_INTERVAL",
                interval=iv,
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            )
            log.append(ev)
            self.assertEqual(len(log.read_raw()), 1)
            with self.assertRaises(Exception):
                assert_not_product_log(Path(tmp) / "decision_log.jsonl")


class SupersedeSplitMergeTests(unittest.TestCase):
    def test_supersede_split_merge(self):
        fps = 10.0
        base = _base_gt(frame_count=20)
        iv = build_annotation_interval(
            start_frame=0,
            end_frame=19,
            fps=fps,
            target_state="TARGET_UNCERTAIN",
            associated_raw_track_ids=["1"],
        )
        events = [
            build_annotation_event(
                action="APPEND_INTERVAL",
                interval=iv,
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            )
        ]
        gt = rebuild_gt_from_events(events, base_gt=base, fps=fps)
        self.assertEqual(len([x for x in gt["intervals"] if x["active"]]), 1)

        left, right = split_interval_at_frame(iv, split_frame=10, fps=fps)
        split_ev = build_annotation_event(
            action="SPLIT_INTERVAL",
            interval=iv,
            reviewer="t",
            annotation_session_id="s",
            source_video_sha256=SHA,
            match_id="m",
            analysis_run_id="r",
            target_id="target_001",
        )
        split_ev["split"] = {"left": left, "right": right}
        gt = rebuild_gt_from_events(events + [split_ev], base_gt=base, fps=fps)
        actives = [x for x in gt["intervals"] if x["active"]]
        self.assertEqual(len(actives), 2)

        # label both sides same and merge
        left2 = dict(left)
        left2["target_state"] = "TARGET_OCCLUDED"
        right2 = dict(right)
        right2["target_state"] = "TARGET_OCCLUDED"
        self.assertTrue(can_merge_intervals(left2, right2))
        merged = merge_intervals(left2, right2, fps=fps)
        merge_ev = build_annotation_event(
            action="MERGE_INTERVALS",
            interval=merged,
            reviewer="t",
            annotation_session_id="s",
            source_video_sha256=SHA,
            match_id="m",
            analysis_run_id="r",
            target_id="target_001",
        )
        merge_ev["merge"] = {
            "deactivate_ids": [left["annotation_id"], right["annotation_id"]]
        }
        gt = rebuild_gt_from_events(events + [split_ev, merge_ev], base_gt=base, fps=fps)
        actives = [x for x in gt["intervals"] if x["active"]]
        self.assertEqual(len(actives), 1)
        self.assertEqual(actives[0]["start_frame"], 0)
        self.assertEqual(actives[0]["end_frame"], 19)


class ValidationTests(unittest.TestCase):
    def test_no_overlap_uncertain_oof_coverage_sha(self):
        fps = 10.0
        frame_count = 10
        base = _base_gt(frame_count=frame_count)
        a = build_annotation_interval(
            start_frame=0,
            end_frame=4,
            fps=fps,
            target_state="TARGET_VISIBLE_ASSOCIATED",
            associated_raw_track_ids=["1"],
            associated_detection_ids=["det_0"],
            bbox_observations=[{"frame_index": 0, "bbox_xyxy": [0, 0, 10, 10]}],
        )
        b = build_annotation_interval(
            start_frame=5,
            end_frame=9,
            fps=fps,
            target_state="TARGET_OUT_OF_FRAME",
        )
        u = build_annotation_interval(
            start_frame=3,
            end_frame=6,
            fps=fps,
            target_state="TARGET_UNCERTAIN",
        )
        events = [
            build_annotation_event(
                action="APPEND_INTERVAL",
                interval=x,
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            )
            for x in (a, b)
        ]
        gt = rebuild_gt_from_events(events, base_gt=base, fps=fps)
        ok = validate_ground_truth(
            gt, expected_source_sha256=SHA, frame_count=frame_count, fps=fps
        )
        self.assertTrue(ok["ok"])
        self.assertTrue(ok["full_coverage"])

        # overlapping active fails
        bad_events = events + [
            build_annotation_event(
                action="APPEND_INTERVAL",
                interval=u,
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            )
        ]
        gt_bad = rebuild_gt_from_events(bad_events, base_gt=base, fps=fps)
        bad = validate_ground_truth(
            gt_bad, expected_source_sha256=SHA, frame_count=frame_count, fps=fps
        )
        self.assertFalse(bad["ok"])

        # SHA mismatch
        bad_sha = validate_ground_truth(
            gt, expected_source_sha256="0" * 64, frame_count=frame_count, fps=fps
        )
        self.assertFalse(bad_sha["ok"])


class MetricsTests(unittest.TestCase):
    def test_no_gt_not_measurable_and_placeholder_excluded(self):
        m = metrics_without_gt()
        self.assertEqual(m["false_target_identity_switch"], NOT_MEASURABLE_WITHOUT_GT)
        validity = build_metric_validity_manifest(
            prior_stabilization={
                "false_target_identity_switch": 0,
                "baseline_summary": {
                    "continuity_probe": {"uninterrupted_duration_sec": 25.3}
                },
            },
            accepted_gt=False,
        )
        names = [c["old_name"] for c in validity["corrections"]]
        self.assertIn("false_target_identity_switch", names)
        self.assertIn("continuity_probe", names)
        self.assertEqual(
            validity["corrections"][1]["new_name"], "seed_iou_continuity_proxy_seconds"
        )
        self.assertFalse(validity["corrections"][1]["is_target_accuracy"])

    def test_metric_correctness_and_multi_variant(self):
        fps = 10.0
        frame_count = 10
        base = _base_gt(frame_count=frame_count)
        iv = build_annotation_interval(
            start_frame=0,
            end_frame=9,
            fps=fps,
            target_state="TARGET_VISIBLE_ASSOCIATED",
            associated_raw_track_ids=["7"],
            bbox_observations=[
                {"frame_index": i, "bbox_xyxy": [10, 10, 30, 50]} for i in range(10)
            ],
        )
        events = [
            build_annotation_event(
                action="APPEND_INTERVAL",
                interval=iv,
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            ),
            build_annotation_event(
                action="ACCEPT_GROUND_TRUTH",
                interval=build_annotation_interval(
                    start_frame=0,
                    end_frame=0,
                    fps=fps,
                    target_state="TARGET_UNCERTAIN",
                    active=False,
                    provenance={"acceptance_marker": True},
                ),
                reviewer="t",
                annotation_session_id="s",
                source_video_sha256=SHA,
                match_id="m",
                analysis_run_id="r",
                target_id="target_001",
            ),
        ]
        gt = rebuild_gt_from_events(events, base_gt=base, fps=fps)
        self.assertTrue(gt["accepted"])

        good_obs = [
            {
                "frame_index": i,
                "raw_track_id": 7,
                "bbox_xyxy": [10, 10, 30, 50],
            }
            for i in range(10)
        ]
        bad_obs = [
            {
                "frame_index": i,
                "raw_track_id": 99,
                "bbox_xyxy": [10, 10, 30, 50],
            }
            for i in range(10)
        ]
        good = evaluate_variant_against_gt(
            ground_truth=gt,
            variant_observations=good_obs,
            fps=fps,
            variant_id="A_current_bytetrack",
            runtime_sec=1.0,
            seed_iou_continuity_proxy_seconds=25.3,
        )
        bad = evaluate_variant_against_gt(
            ground_truth=gt,
            variant_observations=bad_obs,
            fps=fps,
            variant_id="B2_aggressive",
            runtime_sec=2.0,
            seed_iou_continuity_proxy_seconds=0.1,
        )
        self.assertEqual(good["status"], "ok")
        self.assertGreater(good["target_recall"], 0.9)
        self.assertGreater(bad["false_target_assignment_count"], 0)
        self.assertTrue(good["seed_iou_continuity_proxy_is_not_target_accuracy"])
        sel = select_best_variant([good, bad])
        self.assertEqual(sel["selected_variant_id"], "A_current_bytetrack")


class OverlayAndNoRerunContractTests(unittest.TestCase):
    def test_gt_overlay_labels(self):
        # Synthetic tiny video via numpy + cv2 if available
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("cv2 not available in test env")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            video = tmp_p / "tiny.mp4"
            w, h, n, fps = 64, 48, 5, 5.0
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            for _ in range(n):
                writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            writer.release()
            gt = _base_gt(frame_count=n)
            gt["fps"] = fps
            gt["intervals"] = [
                build_annotation_interval(
                    start_frame=0,
                    end_frame=n - 1,
                    fps=fps,
                    target_state="TARGET_VISIBLE_ASSOCIATED",
                    associated_raw_track_ids=["1"],
                    bbox_observations=[
                        {"frame_index": i, "bbox_xyxy": [5, 5, 20, 30]} for i in range(n)
                    ],
                )
            ]
            out = render_gt_overlay_video(
                video_path=video,
                ground_truth=gt,
                output_path=tmp_p / "overlay.mp4",
            )
            self.assertTrue(Path(out["overlay_path"]).is_file())
            self.assertGreater(out["frames_written"], 0)

    def test_scripts_forbid_detection_tracking_rerun(self):
        boot = (_PROJECT / "scripts/run_target_golden_clip_r1_bootstrap.py").read_text()
        ev = (_PROJECT / "scripts/run_target_golden_clip_r1_evaluate.py").read_text()
        self.assertIn("no detection/tracking rerun", boot.lower())
        self.assertIn("detection_rerun", boot)
        self.assertIn("VARIANT_ARTIFACT_NOT_PRESENT", ev)
        self.assertNotIn("run_yolo_detection", boot)
        self.assertNotIn("replay_bytetrack", ev)


if __name__ == "__main__":
    unittest.main()
