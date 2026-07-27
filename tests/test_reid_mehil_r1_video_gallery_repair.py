"""MEHIL-R1 tests: Streamlit image compat + crop fail-closed + clips."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil_ui.compat import (  # noqa: E402
    streamlit_image,
    streamlit_image_api_report,
)
from football_analytics.reid.hil_ui.gallery_view import (  # noqa: E402
    validate_gallery_crop_for_display,
)
from football_analytics.reid.multi_event_hil.gallery_approvals import (  # noqa: E402
    GalleryApprovalLog,
    build_gallery_approval,
    resolve_active_gallery_approvals,
)


class CompatTests(unittest.TestCase):
    def test_streamlit_image_uses_column_width_on_137_signature(self):
        calls = {}

        def fake_image(image, caption=None, use_column_width=None, **kwargs):
            calls["kwargs"] = {
                "caption": caption,
                "use_column_width": use_column_width,
                **kwargs,
            }
            return "ok"

        fake_image.__signature__ = None  # replaced below
        import inspect

        def image(self_or_image, image=None, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto"):
            # support both module-style and bound
            pass

        mod = types.SimpleNamespace()

        def image_fn(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto"):
            calls["use_column_width"] = use_column_width
            calls["caption"] = caption
            calls["forbidden"] = "use_container_width" in (inspect.signature(image_fn).parameters)
            return "ok"

        mod.image = image_fn
        mod.__version__ = "1.37.1"
        report = streamlit_image_api_report(mod)
        self.assertTrue(report["supports_use_column_width"])
        self.assertFalse(report["supports_use_container_width"])
        streamlit_image(mod, "img", caption="c", use_container_width=True)
        self.assertTrue(calls["use_column_width"])


class CropValidationTests(unittest.TestCase):
    def test_real_crop_decode_and_broken_disables_approval(self):
        crop_path = (
            _PROJECT_ROOT
            / "outputs/reid/target_001_multi_event_hil_review_package/gallery_crop_candidates"
            / "mehil_crop_EXT_SEG_004_000000.jpg"
        )
        if not crop_path.is_file():
            self.skipTest("crop artifact missing")
        man = json.loads(
            (
                _PROJECT_ROOT
                / "outputs/reid/target_001_multi_event_hil_review_package/gallery_crop_candidates"
                / "enrollment_crop_candidates.json"
            ).read_text(encoding="utf-8")
        )
        crop = next(c for c in man["candidates"] if c["crop_id"].endswith("000000"))
        ok = validate_gallery_crop_for_display(crop)
        self.assertTrue(ok["visible"])
        self.assertTrue(ok["approval_enabled"])
        self.assertEqual(crop["segment_id"], "EXT_SEG_004")
        self.assertEqual(str(crop["raw_track_id"]), "11")

        broken = dict(crop)
        broken["crop_path"] = str(crop_path.parent / "missing.jpg")
        bad = validate_gallery_crop_for_display(broken)
        self.assertFalse(bad["approval_enabled"])
        self.assertIsNotNone(bad["error"])

        mismatch = dict(crop)
        mismatch["crop_sha256"] = "0" * 64
        bad2 = validate_gallery_crop_for_display(mismatch)
        self.assertFalse(bad2["approval_enabled"])


class GalleryAppendTests(unittest.TestCase):
    def test_multiple_approvals_append_only_no_dev_gallery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.jsonl"
            log = GalleryApprovalLog(path)
            for i in range(2):
                crop = {
                    "crop_id": f"c{i}",
                    "segment_id": "EXT_SEG_004",
                    "raw_track_id": "11",
                    "frame_index": i,
                    "crop_path": f"c{i}.jpg",
                    "crop_sha256": f"{i:064d}"[:64].replace(" ", "0"),
                }
                # fix sha to valid hex
                crop["crop_sha256"] = (f"{i}" * 64)[:64]
                log.append(
                    build_gallery_approval(
                        approval_id=f"a{i}",
                        crop=crop,
                        match_id="m",
                        analysis_run_id="r",
                        target_id="target_001",
                        product_package_id="pkg",
                        reviewer="t",
                    )
                )
            self.assertEqual(len(resolve_active_gallery_approvals(log.read_raw())), 2)
            before = path.read_bytes()
            log.append(
                build_gallery_approval(
                    approval_id="a2",
                    crop={
                        "crop_id": "c0",
                        "segment_id": "EXT_SEG_004",
                        "raw_track_id": "11",
                        "frame_index": 0,
                        "crop_path": "c0.jpg",
                        "crop_sha256": ("0" * 64),
                    },
                    match_id="m",
                    analysis_run_id="r",
                    target_id="target_001",
                    product_package_id="pkg",
                    reviewer="t",
                    approval_status="revoked",
                    supersedes_approval_id="a0",
                )
            )
            self.assertTrue(path.read_bytes().startswith(before))


if __name__ == "__main__":
    unittest.main()
