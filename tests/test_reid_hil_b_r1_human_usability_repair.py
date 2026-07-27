"""Tests for HIL-B-R1 human usability repair."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil_ui.geometry import letterbox_params, resolve_click_to_bbox  # noqa: E402
from football_analytics.reid.hil_ui.package_builders import build_fixture_review_package  # noqa: E402
from football_analytics.reid.hil_ui.session import open_review_session  # noqa: E402
from football_analytics.reid.hil_ui.visualization import (  # noqa: E402
    RANK_UNAVAILABLE_LABEL,
    SPARSE_PACKAGE_NOTICE,
    SUBMIT_SUCCESS_MESSAGE,
    TRACKING_NOT_RUN_NOTICE,
    action_requires_selection,
    appearance_rank_label,
    build_selection_from_bbox_hit,
    confirmation_user_summary,
    is_sparse_observations,
    selection_visibility,
)


class VisualizationRepairTests(unittest.TestCase):
    def test_rank_null_friendly(self):
        self.assertEqual(
            appearance_rank_label({"appearance_rank": None}),
            RANK_UNAVAILABLE_LABEL,
        )
        self.assertNotIn("None", appearance_rank_label({"appearance_rank": None}))

    def test_listed_and_direct_selection_flags(self):
        cands = [
            {
                "candidate_id": "c1",
                "segment_id": "s1",
                "raw_track_id": "r1",
                "bbox_references": [{"frame_index": 0, "bbox_xyxy": [10.0, 10.0, 40.0, 40.0]}],
                "appearance_rank": None,
                "S": None,
                "T_max": None,
                "D_max": None,
                "sportsreid_model_id": None,
                "sportsreid_checkpoint_sha256": None,
            }
        ]
        listed = build_selection_from_bbox_hit(
            frame_index=0,
            bbox_xyxy=[10.0, 10.0, 40.0, 40.0],
            candidates=cands,
        )
        self.assertTrue(listed["listed_selection"])
        self.assertFalse(listed["direct_bbox_selection"])
        direct = build_selection_from_bbox_hit(
            frame_index=0,
            bbox_xyxy=[100.0, 100.0, 140.0, 140.0],
            candidates=cands,
            hit_meta={"segment_id": "EXTRA", "raw_track_id": "raw_x"},
        )
        self.assertTrue(direct["direct_bbox_selection"])
        self.assertFalse(direct["listed_selection"])

    def test_image_click_produces_direct_selection(self):
        params = letterbox_params(frame_w=200, frame_h=100, display_w=200, display_h=100)
        boxes = [
            {
                "bbox_id": "extra",
                "bbox_xyxy": [20.0, 20.0, 60.0, 60.0],
                "provenance": {"segment_id": "DIRECT", "raw_track_id": "raw_d"},
            }
        ]
        hit = resolve_click_to_bbox(ui_x=30, ui_y=30, params=params, bboxes=boxes)
        self.assertEqual(hit.status, "hit")
        sel = build_selection_from_bbox_hit(
            frame_index=7,
            bbox_xyxy=hit.selected.bbox_xyxy,
            candidates=[],
            hit_meta=hit.selected.provenance,
        )
        self.assertTrue(sel["direct_bbox_selection"])
        preview = confirmation_user_summary(
            action="CONFIRM_TARGET", selection=sel, confidence="unknown"
        )
        self.assertTrue(preview["direct_bbox_selection"])
        self.assertTrue(preview["tracking_calistirilmayacak"])

    def test_visibility_frame0_and_sparse_missing(self):
        cands = [
            {
                "candidate_id": "c1",
                "segment_id": "H2_SEG_000003",
                "raw_track_id": "H2_RAW_000003",
                "bbox_references": [
                    {"frame_index": 0, "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]},
                ],
                "start_frame": 0,
                "middle_frame": 0,
                "end_frame": 5,
            }
        ]
        sel = {
            "selected_segment_id": "H2_SEG_000003",
            "selected_candidate_id": "c1",
        }
        v0 = selection_visibility(selection=sel, frame_index=0, candidates=cands)
        self.assertTrue(v0["visible"])
        self.assertEqual(v0["label"], "SELECTED TARGET CANDIDATE")
        v3 = selection_visibility(selection=sel, frame_index=3, candidates=cands)
        self.assertFalse(v3["visible"])
        self.assertIn("not visible", v3["message"].lower())
        lookup = {("H2_SEG_000003", 5): [10.0, 10.0, 20.0, 20.0]}
        v5 = selection_visibility(
            selection=sel,
            frame_index=5,
            candidates=cands,
            observation_lookup=lookup,
        )
        self.assertTrue(v5["visible"])
        self.assertTrue(
            is_sparse_observations(cands, review_window_start=0, review_window_end=5)
        )
        self.assertTrue(SPARSE_PACKAGE_NOTICE.startswith("This review package"))

    def test_confirm_requires_selection_and_notices(self):
        self.assertTrue(action_requires_selection("Confirm Target"))
        self.assertFalse(action_requires_selection("Defer"))
        self.assertFalse(action_requires_selection("None of These"))
        self.assertIn("takip etmez", TRACKING_NOT_RUN_NOTICE)
        self.assertIn("Tracking was not run", SUBMIT_SUCCESS_MESSAGE)

    def test_selection_persists_across_rerun_dict(self):
        # Simulate Streamlit session_state persistence semantics.
        state = {"selection": {}}
        state["selection"] = build_selection_from_bbox_hit(
            frame_index=0,
            bbox_xyxy=[1, 2, 3, 4],
            candidates=[
                {
                    "candidate_id": "c1",
                    "segment_id": "s1",
                    "raw_track_id": "r1",
                    "bbox_references": [{"frame_index": 0, "bbox_xyxy": [1, 2, 3, 4]}],
                }
            ],
        )
        # "rerun" keeps state dict
        self.assertEqual(state["selection"]["selected_candidate_id"], "c1")
        self.assertTrue(state["selection"]["listed_selection"])

    def test_prior_revision2_log_unchanged_bytes(self):
        log = (
            _PROJECT_ROOT
            / "outputs/reid/target_001_reid_hil_b_offline_review_ui/packages/existing_artifact/session/decision_log.jsonl"
        )
        if not log.is_file():
            self.skipTest("acceptance decision log missing")
        data = log.read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "9ace4d2177bbf8993ec2a0093bd694c3bff6f07e2a606c87c9e318110b886623",
        )
        rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
        rev2 = [r for r in rows if r["revision"] == 2][0]
        self.assertEqual(rev2["decision_id"], "dec_6fbbcc997aff")
        self.assertFalse(rev2["direct_bbox_selection"])
        self.assertEqual(rev2["selected_candidate_id"], "real_cand_003")

    def test_streamlit_app_uses_image_coordinates_not_plotly_primary(self):
        src = (_SRC / "football_analytics/reid/hil_ui/streamlit_app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("streamlit_image_coordinates", src)
        self.assertIn("Select this player", src)
        self.assertIn("Advanced / debug: manual pixel coordinates", src)
        self.assertIn("Advanced technical details", src)
        self.assertNotIn("rank={cand.get('appearance_rank')}", src)
        self.assertIn("disabled=submit_disabled", src)

    def test_fixture_package_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            built = build_fixture_review_package(Path(tmp))
            session = open_review_session(built["package_file"])
            self.assertGreaterEqual(session.package_summary()["event_count"], 1)


class SecurityImportTests(unittest.TestCase):
    def test_app_import_no_torch(self):
        import football_analytics.reid.hil_ui.streamlit_app as app

        src = Path(app.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", src)
        self.assertIn("streamlit_image_coordinates", src)


if __name__ == "__main__":
    unittest.main()
