"""Tests for multi-event HIL scaffold (no live SportsReID required for unit tests)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalError,
    GalleryApprovalLog,
    build_gallery_approval,
    resolve_active_gallery_approvals,
)
from football_analytics.reid.multi_event_hil.identity import build_identity  # noqa: E402
from football_analytics.reid.multi_event_hil.package_build import (  # noqa: E402
    build_multi_event_review_package,
)
from football_analytics.reid.multi_event_hil.source_audit import (  # noqa: E402
    audit_multi_event_sources,
)


class MultiEventSourceTests(unittest.TestCase):
    def test_source_prefers_external_not_holdout(self):
        audit = audit_multi_event_sources(_PROJECT_ROOT)
        if audit.get("blocked_status"):
            self.skipTest("no preprocessed source")
        self.assertEqual(audit["selected_video_id"], "target_001_external_enrollment_v1")
        self.assertFalse(
            audit["selected_source"].get("preferred_duration_band_1_to_5_min")
        )
        holdout = next(
            s for s in audit["sources"] if s["video_id"] == "target_001_independent_holdout_v2"
        )
        self.assertEqual(holdout["classification"], "DEVELOPMENT_HOLDOUT_ONLY")


class GalleryApprovalTests(unittest.TestCase):
    def test_append_only_and_reject_dev_gallery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gal.jsonl"
            log = GalleryApprovalLog(path)
            crop = {
                "crop_id": "c1",
                "segment_id": "EXT_SEG_004",
                "raw_track_id": "11",
                "frame_index": 0,
                "crop_path": "crops/c1.jpg",
                "crop_sha256": "a" * 64,
            }
            rec = build_gallery_approval(
                approval_id="gal_001",
                crop=crop,
                match_id="m1",
                analysis_run_id="r1",
                target_id="target_001",
                product_package_id="pkg",
                reviewer="t",
            )
            log.append(rec)
            self.assertEqual(len(resolve_active_gallery_approvals(log.read_raw())), 1)
            with self.assertRaises(GalleryApprovalError):
                bad = dict(rec)
                bad["approval_id"] = "gal_002"
                bad["provenance"] = {
                    "from_development_gallery": True,
                    "automatic_gallery_expansion": False,
                }
                from football_analytics.reid.multi_event_hil.gallery_approvals import (
                    validate_gallery_approval,
                )

                validate_gallery_approval(bad)


class PackageBuildTests(unittest.TestCase):
    def test_isolated_package_has_recovery_events(self):
        audit = audit_multi_event_sources(_PROJECT_ROOT)
        if audit.get("blocked_status"):
            self.skipTest("no preprocessed source")
        with tempfile.TemporaryDirectory() as tmp:
            identity = build_identity("mehil_test_run")
            pkg = build_multi_event_review_package(
                Path(tmp), project_root=_PROJECT_ROOT, identity=identity
            )
            self.assertEqual(pkg["identity"]["match_id"], identity.match_id)
            self.assertGreaterEqual(pkg["recovery_event_count"], 2)
            package = json.loads(Path(pkg["package_file"]).read_text(encoding="utf-8"))
            self.assertEqual(package["provenance"]["package_mode"], "product")
            self.assertFalse(package["provenance"]["old_development_gallery"])
            self.assertFalse(package["provenance"]["gt_prefill"])
            # decision log empty / separate from HIL-C2
            self.assertEqual(Path(pkg["decision_log_path"]).read_text(), "")


if __name__ == "__main__":
    unittest.main()
