"""Unit tests for Stage 5D-B1E-C external positive occurrence freeze."""

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

import run_reid_external_positive_occurrence_freeze as b1ec  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/external_positive_occurrence_freeze_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_occurrence_freeze"
)
_B1EB = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_external_tracking_seed_review_package"
)
_HEAD = "7cf2ba3e9aaf90c4778d5cb6a8bf4a8cb899c0a6"
_EXT_SHA = "ab1c622cd0243f99a0ee9ae6317d8e11132216d9cd34970eb59448593073e877"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ExternalPositiveOccurrenceFreezeTests(unittest.TestCase):
    def test_expected_git_and_selected_codes(self):
        cfg = b1ec.load_config(_CFG)
        self.assertEqual(cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            tuple(cfg["human_positive_occurrence_freeze"]["selected_external_candidate_codes"]),
            b1ec.SELECTED_CODES,
        )
        self.assertEqual(b1ec.SELECTED_CODES, ("EXT_004", "EXT_183", "EXT_198"))
        self.assertEqual(
            cfg["human_positive_occurrence_freeze"]["review_scope"],
            "positive_occurrence_selection_only",
        )

    def test_b1e_b_package_expected_counts(self):
        cfg = b1ec.load_config(_CFG)
        pkg = cfg["stage5d_b1e_b_package"]
        self.assertEqual(
            pkg["expected_final_status"],
            "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY",
        )
        self.assertEqual(pkg["expected_ext_candidate_count"], 248)
        self.assertEqual(pkg["expected_review_eligible_count"], 138)
        self.assertEqual(cfg["external_enrollment_source"]["expected_sha256"], _EXT_SHA)

    def test_validate_selected_codes_happy_and_not_found(self):
        mapping = [
            {
                "external_candidate_code": "EXT_004",
                "raw_external_track_id": 11,
                "first_frame": 0,
                "last_frame": 2,
                "observation_count": 3,
                "observation_frames": [0, 1, 2],
                "bbox_per_observation": [{}, {}, {}],
                "detection_lineage": [{}, {}, {}],
                "representative_frame": 1,
                "source_video_sha256": _EXT_SHA,
            },
            {
                "external_candidate_code": "EXT_183",
                "raw_external_track_id": 388,
                "first_frame": 10,
                "last_frame": 12,
                "observation_count": 3,
                "observation_frames": [10, 11, 12],
                "bbox_per_observation": [{}, {}, {}],
                "detection_lineage": [{}, {}, {}],
                "representative_frame": 11,
                "source_video_sha256": _EXT_SHA,
            },
            {
                "external_candidate_code": "EXT_198",
                "raw_external_track_id": 450,
                "first_frame": 20,
                "last_frame": 22,
                "observation_count": 3,
                "observation_frames": [20, 21, 22],
                "bbox_per_observation": [{}, {}, {}],
                "detection_lineage": [{}, {}, {}],
                "representative_frame": 21,
                "source_video_sha256": _EXT_SHA,
            },
        ]
        resolved = b1ec.validate_selected_codes(
            mapping, b1ec.SELECTED_CODES, expected_source_sha=_EXT_SHA
        )
        self.assertEqual(len(resolved), 3)
        with self.assertRaises(b1ec.OccurrenceFreezeError) as ctx:
            b1ec.validate_selected_codes(
                mapping[:2], b1ec.SELECTED_CODES, expected_source_sha=_EXT_SHA
            )
        self.assertIn("SELECTED_CODE_NOT_FOUND", str(ctx.exception))

    def test_duplicate_code_and_ambiguous_raw_rejected(self):
        mapping = [
            {
                "external_candidate_code": "EXT_004",
                "raw_external_track_id": 11,
                "first_frame": 0,
                "last_frame": 1,
                "observation_count": 2,
                "observation_frames": [0, 1],
                "bbox_per_observation": [{}, {}],
                "detection_lineage": [{}, {}],
                "representative_frame": 0,
                "source_video_sha256": _EXT_SHA,
            },
            {
                "external_candidate_code": "EXT_183",
                "raw_external_track_id": 11,
                "first_frame": 2,
                "last_frame": 3,
                "observation_count": 2,
                "observation_frames": [2, 3],
                "bbox_per_observation": [{}, {}],
                "detection_lineage": [{}, {}],
                "representative_frame": 2,
                "source_video_sha256": _EXT_SHA,
            },
            {
                "external_candidate_code": "EXT_198",
                "raw_external_track_id": 450,
                "first_frame": 4,
                "last_frame": 5,
                "observation_count": 2,
                "observation_frames": [4, 5],
                "bbox_per_observation": [{}, {}],
                "detection_lineage": [{}, {}],
                "representative_frame": 4,
                "source_video_sha256": _EXT_SHA,
            },
        ]
        with self.assertRaises(b1ec.OccurrenceFreezeError) as ctx:
            b1ec.validate_selected_codes(
                mapping, b1ec.SELECTED_CODES, expected_source_sha=_EXT_SHA
            )
        self.assertIn("LINEAGE_AMBIGUOUS", str(ctx.exception))
        with self.assertRaises(b1ec.OccurrenceFreezeError):
            b1ec.validate_selected_codes(
                mapping,
                ("EXT_004", "EXT_004", "EXT_198"),
                expected_source_sha=_EXT_SHA,
            )

    def test_contract_counts_and_forbidden_ops(self):
        c = b1ec.build_contract()
        self.assertEqual(c["selected_positive_count"], 3)
        self.assertEqual(c["reviewed_negative_count"], 0)
        self.assertEqual(c["unreviewed_count"], 245)
        self.assertTrue(c["unreviewed_not_converted_to_negative"])
        self.assertEqual(c["review_scope"], "positive_occurrence_selection_only")
        self.assertFalse(c["automated_ocr_used"])
        self.assertFalse(c["similarity_used"])
        self.assertFalse(c["model_identity_prediction_used"])
        self.assertFalse(c["external_occurrences_are_gallery_members"])
        self.assertFalse(c["external_occurrences_are_final_anchors"])
        self.assertEqual(c["embeddings"], 0)
        self.assertEqual(c["gallery_members"], 0)
        self.assertEqual(c["prototypes"], 0)
        self.assertEqual(c["identity_assignments"], 0)
        self.assertEqual(c["new_detection"], 0)
        self.assertEqual(c["new_tracking"], 0)

    def test_path_traversal_rejection(self):
        with self.assertRaises(b1ec.OccurrenceFreezeError):
            b1ec.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing_final(self):
        cfg = b1ec.load_config(_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            final = project / cfg["output"]["final_dir"]
            final.mkdir(parents=True)
            with mock.patch.object(b1ec, "assert_git_contract", return_value="deadbeef"):
                with self.assertRaises(b1ec.OccurrenceFreezeError) as ctx:
                    b1ec.run(_CFG, project)
            self.assertIn("final_exists", str(ctx.exception))

    def test_live_b1e_b_and_b1e_c_if_present(self):
        if not _B1EB.is_dir():
            self.skipTest("B1E-B absent")
        s = json.loads((_B1EB / "stage5d_b1e_b_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            s["final_status"],
            "COMPLETED_STAGE5D_B1E_B_EXTERNAL_TRACKING_SEED_REVIEW_READY",
        )
        self.assertEqual(s["ext_candidate_count"], 248)
        self.assertTrue(s["two_replay_determinism"])
        self.assertEqual(s["manual_selections"], 0)

        if not _FINAL.is_dir():
            self.skipTest("B1E-C output absent")
        summary = json.loads(
            (_FINAL / "stage5d_b1e_c_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["final_status"],
            "COMPLETED_STAGE5D_B1E_C_TARGET_001_EXTERNAL_OCCURRENCES_FROZEN",
        )
        self.assertEqual(
            summary["selected_external_candidate_codes"],
            ["EXT_004", "EXT_183", "EXT_198"],
        )
        self.assertEqual(summary["selected_positive_count"], 3)
        self.assertEqual(summary["reviewed_negative_count"], 0)
        self.assertEqual(summary["unreviewed_count"], 245)
        self.assertEqual(summary["review_scope"], "positive_occurrence_selection_only")
        self.assertFalse(summary["automated_ocr_used"])
        self.assertFalse(summary["similarity_used"])
        self.assertFalse(summary["model_identity_prediction_used"])
        self.assertEqual(summary["approved_anchor_crops"], 0)
        self.assertEqual(summary["embeddings"], 0)
        self.assertEqual(summary["gallery_members"], 0)
        self.assertEqual(summary["prototypes"], 0)
        self.assertEqual(summary["identity_assignments"], 0)
        self.assertEqual(summary["new_detection"], 0)
        self.assertEqual(summary["new_tracking"], 0)
        self.assertEqual(len(list(_FINAL.rglob("*.png"))), 0)
        self.assertEqual(len(list(_FINAL.rglob("*.mp4"))), 0)

        csv_path = (
            _FINAL
            / "occurrence_freeze"
            / "target_001_external_positive_occurrences_frozen.csv"
        )
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [r["external_candidate_code"] for r in rows],
            ["EXT_004", "EXT_183", "EXT_198"],
        )
        self.assertEqual(len({r["resolved_raw_track_id"] for r in rows}), 3)
        self.assertTrue(
            all(r["manual_occurrence_decision"] == "target_occurrence_yes" for r in rows)
        )
        self.assertTrue(all(r["reviewer"] == "Furkan" for r in rows))
        freeze = json.loads(
            (
                _FINAL
                / "occurrence_freeze"
                / "target_001_external_positive_occurrence_freeze.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(freeze["schema_version"], "reid_target_external_positive_occurrence_freeze_v1")
        self.assertTrue(freeze["target_occurrence_freeze"])
        self.assertFalse(freeze["external_occurrences_are_gallery_members"])
        lineage = json.loads(
            (
                _FINAL
                / "validation"
                / "target_001_external_occurrence_lineage_validation.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(lineage["unreviewed_policy"]["count"], 245)
        self.assertTrue(
            lineage["unreviewed_policy"]["not_counted_as_target_occurrence_no"]
        )
        # Source immutability.
        ext = _PROJECT_ROOT / "data/enrollment_clips/target_001_external_enrollment_v1.mp4"
        self.assertEqual(_sha256(ext), _EXT_SHA)
        self.assertEqual(summary["external_source_sha256"], _EXT_SHA)


if __name__ == "__main__":
    unittest.main()
