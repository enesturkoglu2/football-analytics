"""MEHIL-R2 tests: dense observations, gallery quality, component presence."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil_c2.product_package import _segment_id  # noqa: E402
from football_analytics.reid.hil_ui.dense_observations import (  # noqa: E402
    build_dense_observations_from_mapping,
    load_mapping_jsonl,
)
from football_analytics.reid.hil_ui.gallery_quality import (  # noqa: E402
    audit_gallery_candidates,
)


class DenseObsTests(unittest.TestCase):
    def test_dense_frames_for_ext004(self):
        path = (
            _PROJECT
            / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
            / "inventory/target_001_external_track_candidate_mapping.jsonl"
        )
        if not path.is_file():
            self.skipTest("mapping missing")
        dens = build_dense_observations_from_mapping(
            load_mapping_jsonl(path),
            codes=["EXT_004"],
            segment_id_fn=_segment_id,
        )
        self.assertGreaterEqual(len(dens), 100)
        sample = dens[sorted(dens, key=int)[0]][0]
        self.assertEqual(sample["segment_id"], "EXT_SEG_004")
        self.assertEqual(sample["raw_track_id"], "11")


class GalleryQualityTests(unittest.TestCase):
    def test_quality_gate_classifies_and_disables_bad(self):
        man = (
            _PROJECT
            / "outputs/reid/target_001_multi_event_hil_review_package/gallery_crop_candidates"
            / "enrollment_crop_candidates.json"
        )
        if not man.is_file():
            self.skipTest("crops missing")
        crop_man = json.loads(man.read_text(encoding="utf-8"))
        audit = audit_gallery_candidates(
            crop_man["candidates"], source_frame_size=(1332, 746)
        )
        self.assertEqual(audit["candidate_count"], 5)
        self.assertFalse(audit["fixed_reid_threshold_invented"])
        self.assertFalse(audit["upscaling_used"])
        for row in audit["audits"]:
            self.assertIn(row["quality_class"], {
                "USABLE_GALLERY_CANDIDATE",
                "LOW_RESOLUTION",
                "BLURRED",
                "CLIPPED",
                "OCCLUDED",
                "CONTAMINATED",
                "INVALID",
            })
            if row["quality_class"] != "USABLE_GALLERY_CANDIDATE":
                self.assertFalse(row["approval_enabled"])


class ComponentPresenceTests(unittest.TestCase):
    def test_frontend_index_exists(self):
        p = (
            _SRC
            / "football_analytics/reid/hil_ui/interactive_video_component/frontend/index.html"
        )
        self.assertTrue(p.is_file())
        text = p.read_text(encoding="utf-8")
        self.assertIn("TARGET TRACK ENDED", text)
        self.assertIn("Selected target is not visible", text)
        self.assertIn("bbox_click", text)


if __name__ == "__main__":
    unittest.main()
