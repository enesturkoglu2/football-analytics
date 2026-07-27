"""Unit tests for Stage 5D-F3N holdout v2 result audit package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_reid_independent_holdout_v2_result_audit_and_error_analysis as f3n  # noqa: E402

_CFG = (
    _PROJECT_ROOT
    / "configs/reid/independent_holdout_v2_result_audit_stage5d_target_001.yaml"
)
_FINAL = (
    _PROJECT_ROOT
    / "outputs/reid/full_stage4b_rebuild_r2_stage5d_target_001_independent_holdout_v2_result_audit_and_error_analysis_package"
)
_HEAD = "edd31c6aad1c7b697d9faee44e57fd074fbaa6d2"
_F3M_SNAP = "d13c6e5de1ff254e487ba7737a03521019cf704b3028766e6a55103a76a00cb7"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class HoldoutResultAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = f3n.load_config(_CFG)

    def test_expected_git_contract_fields(self):
        self.assertEqual(self.cfg["project_head_expected"], _HEAD)
        self.assertEqual(
            self.cfg["project_head_message_expected"],
            "Evaluate target 001 holdout v2 with frozen scoring",
        )

    def test_f3m_status_snapshot_and_frozen_metrics(self):
        f3m = f3n.validate_f3m(self.cfg)
        self.assertEqual(f3m["summary"]["final_status"], self.cfg["stage5d_f3m_package"]["expected_final_status"])
        self.assertEqual(f3m["summary"]["readiness"], self.cfg["stage5d_f3m_package"]["expected_readiness"])
        self.assertEqual(
            f3m["summary"]["performance_outcome"],
            "MANUAL_RULE_INTERPRETATION_REQUIRED",
        )
        self.assertEqual(f3m["snapshot"]["sha256"], _F3M_SNAP)
        self.assertEqual(f3m["seal"]["gt_labels_read_so_far"], 0)
        self.assertFalse(f3m["scores_recomputed"])
        self.assertEqual(f3m["summary"]["positive_ranks"], [6, 9, 27, 31, 46, 53, 65, 70, 83, 100])
        self.assertAlmostEqual(f3m["seg"]["Recall@10"], 0.2)
        self.assertAlmostEqual(f3m["seg"]["Average_Precision"], 0.12813472143260216)
        self.assertAlmostEqual(f3m["seg"]["AUROC"], 0.5857142857142857)

    def test_f3l_f3k_f3g_lineage(self):
        f3m = f3n.validate_f3m(self.cfg)
        f3l = f3n.validate_f3l(self.cfg, f3m["joined"])
        f3k = f3n.validate_f3k(self.cfg)
        f3g = f3n.validate_f3g(self.cfg)
        self.assertEqual(len(f3l["decisions"]), 141)
        self.assertEqual(len(f3k["crops_by_id"]), 141)
        self.assertEqual(len(f3g["target_ids"]), 13)
        self.assertEqual(len(f3g["distractor_ids"]), 23)
        self.assertEqual(f3g["holdout_member_count"], 0)

    def test_exact_positive_ids(self):
        self.assertEqual(f3n.POSITIVE_IDS, tuple(self.cfg["exact_positive_ids"]))

    def test_audit_cohorts_exact(self):
        f3m = f3n.validate_f3m(self.cfg)
        f3l = f3n.validate_f3l(self.cfg, f3m["joined"])
        f3k = f3n.validate_f3k(self.cfg)
        f3g = f3n.validate_f3g(self.cfg)
        items = f3n.build_audit_items(f3m=f3m, f3l=f3l, f3k=f3k, f3g=f3g)
        self.assertEqual(len(items), 50)
        pos = [i for i in items if i["cohort"] == "positive"]
        st = [i for i in items if i["cohort"] == "same_team_fp"]
        ot = [i for i in items if i["cohort"] == "other_team_fp"]
        self.assertEqual(len(pos), 10)
        self.assertEqual(len(st), 20)
        self.assertEqual(len(ot), 20)
        self.assertEqual(len({i["query_id"] for i in items}), 50)
        self.assertEqual([i["audit_item_id"] for i in pos], [f"F3N_POS_{i:03d}" for i in range(1, 11)])
        self.assertEqual([i["audit_item_id"] for i in st], [f"F3N_STFP_{i:03d}" for i in range(1, 21)])
        self.assertEqual([i["audit_item_id"] for i in ot], [f"F3N_OTFP_{i:03d}" for i in range(1, 21)])
        for i in items:
            self.assertTrue(i["automatic_diagnosis_absent"])
            self.assertTrue(i["human_diagnosis_pending"])
            self.assertIn(i["T_max_member_id"], f3g["members"])
            self.assertIn(i["D_max_member_id"], f3g["members"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(f3n.ResultAuditError):
            f3n.assert_no_path_traversal("../x")

    def test_atomic_finalization_rejects_existing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp"
            final = Path(td) / "final"
            tmp.mkdir()
            final.mkdir()
            with self.assertRaises(f3n.ResultAuditError):
                f3n.atomic_publish(tmp, final)

    def test_blank_manual_fields_constant(self):
        self.assertIn("manual_primary_failure_category", f3n.BLANK_MANUAL_FIELDS)
        self.assertIn("reviewer", f3n.BLANK_MANUAL_FIELDS)
        self.assertNotIn("query_id", f3n.BLANK_MANUAL_FIELDS)


@unittest.skipUnless(_FINAL.is_dir(), "F3N final package not built yet")
class HoldoutResultAuditOutputTests(unittest.TestCase):
    def test_output_counts_and_no_mutation(self):
        summary = json.loads((_FINAL / "stage5d_f3n_summary.json").read_text())
        self.assertEqual(summary["final_status"], f3n.FINAL_STATUS)
        self.assertEqual(summary["total_audit_items"], 50)
        self.assertEqual(summary["contact_sheets"], 5)
        self.assertEqual(summary["diagnostic_videos"], 3)
        self.assertEqual(summary["manual_root_cause_decisions"], 0)
        self.assertFalse(summary["scores_recomputed"])
        self.assertEqual(summary["embeddings_generated"], 0)
        self.assertFalse(summary["gallery_mutation"])
        self.assertEqual(summary["deleted_files"], 0)
        self.assertTrue(summary["holdout_retired_from_future_independent_testing"])
        self.assertTrue(summary["cleanup_deferred"])

        inv = [
            json.loads(l)
            for l in (_FINAL / "inventory" / "target_001_holdout_v2_f3n_error_audit_item_inventory.jsonl")
            .read_text()
            .splitlines()
            if l.strip()
        ]
        self.assertEqual(len(inv), 50)

        sheets = list(
            (_FINAL / "review_packages" / "target_001_holdout_v2_frozen_result_error_analysis").glob("*.png")
        )
        self.assertEqual(len(sheets), 5)
        videos = list((_FINAL / "videos").glob("*.mp4"))
        self.assertEqual(len(videos), 3)
        self.assertFalse(list(_FINAL.rglob("*.npy")))

        with (_FINAL / "templates" / "target_001_holdout_v2_error_analysis_manual_review_template.csv").open(
            encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 50)
        for row in rows:
            for field in f3n.BLANK_MANUAL_FIELDS:
                self.assertEqual(row[field], "")

        quant = json.loads(
            (_FINAL / "audit" / "target_001_holdout_v2_frozen_result_quantitative_audit.json").read_text()
        )
        self.assertFalse(quant["metrics_recomputed"])
        integrity = json.loads(
            (_FINAL / "audit" / "target_001_holdout_v2_frozen_result_contract_integrity_audit.json").read_text()
        )
        self.assertTrue(integrity["gallery_sha_unchanged"])
        self.assertFalse(integrity["scores_recomputed"])

        retirement = json.loads(
            (_FINAL / "governance" / "target_001_holdout_v2_post_evaluation_retirement.json").read_text()
        )
        self.assertFalse(retirement["future_independent_test_eligible"])
        self.assertTrue(retirement["new_method_requires_new_holdout"])

        cleanup = json.loads(
            (_FINAL / "governance" / "target_001_stage5d_cleanup_deferred_inventory.json").read_text()
        )
        self.assertEqual(cleanup["deleted_files"], 0)
        self.assertTrue(cleanup["deletion_approvals_all_false"])
        for cat in cleanup["categories"].values():
            self.assertFalse(cat["delete_approved"])

        access = json.loads((_FINAL / "runtime" / "target_001_f3n_access_audit.json").read_text())
        self.assertEqual(access["new_embeddings"], 0)
        self.assertEqual(access["score_rows_recomputed"], 0)
        self.assertEqual(access["detection_inference"], 0)
        self.assertEqual(access["deleted_files"], 0)


if __name__ == "__main__":
    unittest.main()
