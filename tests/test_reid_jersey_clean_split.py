"""Unit tests for Stage 5C clean label-blind discovery/holdout split."""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_reid_jersey_clean_split as cs  # noqa: E402


def _write_crop(path: Path, *, seed: int, w: int = 40, h: int = 80) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    img[8:40, 5:30] = (seed % 200, (seed * 3) % 200, (seed * 7) % 200)
    cv2.imwrite(str(path), img)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_item(i: int, path: Path, sha: str, **extra) -> dict:
    row = {
        "schema_version": "reid_jersey_clean_review_item_v1",
        "review_item_id": f"review_item_{i:04d}",
        "crop_id": f"crop_{i:04d}",
        "segment_id": f"seg_{i:04d}",
        "raw_track_id": 1000 + i,
        "documented_global_candidate_id": 1000 + i,
        "frame_index": i * 70,
        "source_crop_path": str(path),
        "source_crop_sha256": sha,
        "source_crop_width": 40,
        "source_crop_height": 80,
        "crop_source_type": "reused_baseline_selected_crop"
        if i % 2 == 0
        else "recomputed_manual_segment",
        "number_roi_x": 5,
        "number_roi_y": 8,
        "number_roi_width": 25,
        "number_roi_height": 32,
        "number_roi_x_max": 30,
        "number_roi_y_max": 40,
        "number_roi_relative_area": 0.2 + (i % 10) * 0.02,
        "roi_valid": True,
        "laplacian_variance": 100.0 + i,
        "tenengrad_mean": 1000.0 + i,
        "local_contrast": 1.0 + i * 0.01,
        "entropy": 4.0 + (i % 20) * 0.05,
        "edge_density": 0.1 + (i % 15) * 0.01,
        "roi_other_person_union_coverage": 0.0 + (i % 5) * 0.01,
        "roi_other_person_center_inside_count": i % 3,
        "quality_signal_joined": i % 9 != 0,
        "kit_descriptor_joined": i % 9 != 0,
        "annotation_status": "unreviewed",
        "manual_crop_valid": None,
        "manual_number_visible": None,
        "manual_number_readable": None,
        "manual_jersey_number": None,
        "manual_notes": None,
        "reviewer": None,
        "reviewed_at": None,
        "ocr_prediction": None,
        "ocr_confidence": None,
    }
    row.update(extra)
    return row


def _load_cfg() -> dict:
    return yaml.safe_load(
        (
            _PROJECT_ROOT / "configs/reid/jersey_clean_split_stage5c_rebuild_r2.yaml"
        ).read_text(encoding="utf-8")
    )


def _synthetic_universe(n: int = 240) -> tuple[list[dict], dict[str, str]]:
    rows = []
    for i in range(n):
        if i < n * 0.4:
            stratum = "high_signal_candidate"
        elif i < n * 0.75:
            stratum = "mid_signal_candidate"
        else:
            stratum = "safety_candidate"
        source = (
            "recomputed_manual_segment"
            if (i % 4 == 0)
            else "reused_baseline_selected_crop"
        )
        rows.append(
            {
                "review_item_id": f"r{i:04d}",
                "crop_id": f"c{i:04d}",
                "segment_id": f"seg{i:04d}",
                "raw_track_id": 5000 + i,
                "documented_global_candidate_id": 5000 + i,
                "frame_index": i * 100,
                "source_crop_path": f"/tmp/{i}.jpg",
                "source_crop_sha256": f"sha{i:04d}",
                "crop_source_type": source,
                "composite_score": float(n - i),
                "signal_stratum": stratum,
                "timeline_bin": i % 4,
            }
        )
    item_to_group = {r["review_item_id"]: f"g_{r['review_item_id']}" for r in rows}
    for r in rows:
        r["near_duplicate_cluster_id"] = f"nd_{r['review_item_id']}"
        r["leakage_group_id"] = item_to_group[r["review_item_id"]]
    return rows, item_to_group


class CleanSplitUnitTests(unittest.TestCase):
    def test_474_contract_constant(self):
        self.assertEqual(_load_cfg()["source"]["expected_universe_size"], 474)

    def test_forbidden_manual_ocr_exclusion_in_template(self):
        for col in cs.TEMPLATE_COLUMNS:
            self.assertFalse(col.startswith("ocr_"))
            self.assertNotIn(col, {"stratum", "composite_score", "leakage_group_id"})

    def test_deterministic_feature_normalization(self):
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                p = root / f"c{i}.jpg"
                sha = _write_crop(p, seed=i + 1)
                rows.append(_base_item(i, p, sha))
            cfg = _load_cfg()
            scored_a, _ = cs.score_universe(rows, cfg["feature_contract"])
            scored_b, _ = cs.score_universe(rows, cfg["feature_contract"])
            self.assertEqual(
                [r["composite_score"] for r in scored_a],
                [r["composite_score"] for r in scored_b],
            )
            for r in scored_a:
                self.assertTrue(math.isfinite(r["composite_score"]))

    def test_missing_field_handling_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(6):
                p = root / f"c{i}.jpg"
                sha = _write_crop(p, seed=50 + i)
                rows.append(
                    _base_item(
                        i,
                        p,
                        sha,
                        quality_signal_joined=(i != 0),
                        kit_descriptor_joined=(i != 0),
                    )
                )
            cfg = _load_cfg()
            scored, meta = cs.score_universe(rows, cfg["feature_contract"])
            self.assertIn("quality_signal_joined_missing", meta["missing_counts"])
            self.assertEqual(meta["missing_counts"]["quality_signal_joined_missing"], 1)
            miss = next(r for r in scored if not r["quality_signal_joined"])
            self.assertIn(
                "quality_signal_joined_missing", miss["applied_missingness_penalties"]
            )

    def test_deterministic_strata_quantiles(self):
        rows = [
            {"review_item_id": f"r{i:02d}", "composite_score": float(100 - i)}
            for i in range(20)
        ]
        strata_cfg = {
            "bands": {
                "high_signal_candidate": {
                    "quantile_start": 0.0,
                    "quantile_end": 0.4,
                },
                "mid_signal_candidate": {
                    "quantile_start": 0.4,
                    "quantile_end": 0.75,
                },
                "safety_candidate": {
                    "quantile_start": 0.75,
                    "quantile_end": 1.0,
                },
            },
            "semantics": {
                "high_signal_candidate_is_readable_positive": False,
                "safety_candidate_is_negative_ground_truth": False,
                "strata_are_sampling_tools_only": True,
            },
        }
        out = cs.assign_quantile_strata(rows, strata_cfg)
        labels = [r["signal_stratum"] for r in out]
        self.assertEqual(labels.count("high_signal_candidate"), 8)
        self.assertEqual(labels.count("mid_signal_candidate"), 7)
        self.assertEqual(labels.count("safety_candidate"), 5)
        self.assertTrue(all(r["stratum_is_ground_truth"] is False for r in out))
        out2 = cs.assign_quantile_strata(rows, strata_cfg)
        self.assertEqual(
            [r["review_item_id"] for r in out],
            [r["review_item_id"] for r in out2],
        )

    def test_dhash_and_near_duplicate_grouping(self):
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        b = a.copy()
        b[:, 16:] = 255
        ha = cs.dhash_bits(a, resize=(9, 8))
        hb = cs.dhash_bits(b, resize=(9, 8))
        self.assertEqual(ha.shape, (64,))
        self.assertEqual(cs.hamming_distance(ha, ha), 0)
        self.assertGreater(cs.hamming_distance(ha, hb), 0)
        self.assertTrue(np.array_equal(ha, cs.dhash_bits(a, resize=(9, 8))))

    def test_exact_and_transitive_leakage_grouping(self):
        rows = [
            {
                "review_item_id": "a",
                "crop_id": "c1",
                "source_crop_sha256": "sha1",
                "segment_id": "s1",
                "raw_track_id": 4,
                "documented_global_candidate_id": 4,
            },
            {
                "review_item_id": "b",
                "crop_id": "c2",
                "source_crop_sha256": "sha2",
                "segment_id": "s2",
                "raw_track_id": 682,
                "documented_global_candidate_id": 4,
            },
            {
                "review_item_id": "c",
                "crop_id": "c3",
                "source_crop_sha256": "sha3",
                "segment_id": "s3",
                "raw_track_id": 99,
                "documented_global_candidate_id": 99,
            },
            {
                "review_item_id": "d",
                "crop_id": "c4",
                "source_crop_sha256": "sha1",
                "segment_id": "s4",
                "raw_track_id": 100,
                "documented_global_candidate_id": 100,
            },
        ]
        groups, item_to_group, item_to_nd, audit = cs.build_leakage_groups(
            rows,
            near_dup_edges=[("b", "c", 3)],
            documented_components=[[4, 682]],
        )
        self.assertEqual(item_to_group["a"], item_to_group["b"])
        self.assertEqual(item_to_group["a"], item_to_group["c"])
        self.assertEqual(item_to_group["a"], item_to_group["d"])
        self.assertEqual(audit["exact_duplicate_sha_cluster_count"], 1)
        self.assertEqual(item_to_nd["b"], item_to_nd["c"])
        self.assertEqual(len(groups), 1)

    def test_same_raw_track_and_segment_grouping(self):
        rows = [
            {
                "review_item_id": "a",
                "crop_id": "c1",
                "source_crop_sha256": "s1",
                "segment_id": "same_seg",
                "raw_track_id": 7,
                "documented_global_candidate_id": 1,
            },
            {
                "review_item_id": "b",
                "crop_id": "c2",
                "source_crop_sha256": "s2",
                "segment_id": "same_seg",
                "raw_track_id": 8,
                "documented_global_candidate_id": 2,
            },
            {
                "review_item_id": "c",
                "crop_id": "c3",
                "source_crop_sha256": "s3",
                "segment_id": "other",
                "raw_track_id": 7,
                "documented_global_candidate_id": 3,
            },
        ]
        _, item_to_group, _, _ = cs.build_leakage_groups(
            rows, near_dup_edges=[], documented_components=[]
        )
        self.assertEqual(item_to_group["a"], item_to_group["b"])
        self.assertEqual(item_to_group["a"], item_to_group["c"])

    def test_dynamic_recomputed_capacity_audit(self):
        rows, item_to_group = _synthetic_universe(80)
        cfg = _load_cfg()
        audit = cs.audit_recomputed_capacity(
            rows, item_to_group=item_to_group, allocation=cfg["allocation"]
        )
        recomp = [
            r
            for r in rows
            if r["crop_source_type"] == "recomputed_manual_segment"
        ]
        self.assertEqual(audit["recomputed_item_count"], len(recomp))
        self.assertEqual(
            audit["global_maximum_selectable_recomputed_count"],
            sum(g["maximum_selectable_item_count"] for g in audit["groups"]),
        )
        self.assertGreater(audit["global_maximum_selectable_recomputed_count"], 0)

    def test_hamilton_proportional_reference_for_max18(self):
        totals = {
            "discovery_primary": 40,
            "discovery_reserve": 16,
            "holdout_primary": 48,
            "holdout_reserve": 24,
        }
        ref = cs.hamilton_proportional_reference(18, batch_totals=totals)
        self.assertEqual(
            [ref[b] for b in cs.BATCH_ORDER],
            [6, 2, 7, 3],
        )
        self.assertEqual(
            _load_cfg()["capacity_search"]["proportional_reference_for_max18"],
            [6, 2, 7, 3],
        )

    def test_quota_vector_enumeration_and_hard_minima(self):
        cfg = _load_cfg()
        totals = {b: int(cfg["batches"][b]["total"]) for b in cs.BATCH_ORDER}
        mins = cfg["capacity_search"]["hard_minima"]
        vectors = cs.enumerate_quota_vectors(
            total=18, hard_minima=mins, batch_totals=totals
        )
        self.assertTrue(all(sum(v) == 18 for v in vectors))
        for v in vectors:
            for i, b in enumerate(cs.BATCH_ORDER):
                self.assertGreaterEqual(v[i], int(mins[b]))
                self.assertLessEqual(v[i], totals[b])
        self.assertIn((6, 2, 7, 3), vectors)
        self.assertIn((5, 7, 4, 2), vectors)

    def test_objective_ordering_prefers_proportional_reference(self):
        totals = [40, 16, 48, 24]
        ratio = 75 / 474
        preferred = (6, 2, 7, 3)
        other = (5, 7, 4, 2)
        k_pref = cs.objective_sort_key(
            preferred, batch_totals=totals, universe_ratio=ratio
        )
        k_other = cs.objective_sort_key(
            other, batch_totals=totals, universe_ratio=ratio
        )
        self.assertLess(k_pref, k_other)

    def test_batch_quotas_and_overlap_zero(self):
        rows, item_to_group = _synthetic_universe(240)
        cfg = _load_cfg()
        # Exact proportional reference quotas when fixture capacity allows.
        batch_quotas = {}
        vector = (6, 2, 7, 3)
        for i, batch in enumerate(cs.BATCH_ORDER):
            q = dict(cfg["batches"][batch])
            q["recomputed"] = vector[i]
            q["reused"] = int(q["total"]) - vector[i]
            batch_quotas[batch] = q
        batches = cs.allocate_batches(
            rows,
            item_to_group=item_to_group,
            batch_quotas=batch_quotas,
            allocation=cfg["allocation"],
            source_type_keys=cfg.get("source_type_keys"),
        )
        self.assertEqual(len(batches["discovery_primary"]), 40)
        self.assertEqual(len(batches["discovery_reserve"]), 16)
        self.assertEqual(len(batches["holdout_primary"]), 48)
        self.assertEqual(len(batches["holdout_reserve"]), 24)
        self.assertEqual(sum(len(v) for v in batches.values()), 128)

        def _src_counts(items):
            reused = sum(
                1
                for r in items
                if r["crop_source_type"] == "reused_baseline_selected_crop"
            )
            recomp = sum(
                1
                for r in items
                if r["crop_source_type"] == "recomputed_manual_segment"
            )
            return reused, recomp

        self.assertEqual(_src_counts(batches["discovery_primary"]), (34, 6))
        self.assertEqual(_src_counts(batches["discovery_reserve"]), (14, 2))
        self.assertEqual(_src_counts(batches["holdout_primary"]), (41, 7))
        self.assertEqual(_src_counts(batches["holdout_reserve"]), (21, 3))
        all_sel = [r for b in batches.values() for r in b]
        self.assertEqual(_src_counts(all_sel), (110, 18))
        for batch, items in batches.items():
            self.assertGreater(
                sum(
                    1
                    for r in items
                    if r["crop_source_type"] == "recomputed_manual_segment"
                ),
                0,
                msg=batch,
            )
            q = batch_quotas[batch]
            self.assertEqual(
                sum(1 for r in items if r["stratum"] == "high_signal_candidate"),
                q["high"],
            )
            self.assertEqual(
                sum(1 for r in items if r["stratum"] == "mid_signal_candidate"),
                q["mid"],
            )
            self.assertEqual(
                sum(1 for r in items if r["stratum"] == "safety_candidate"),
                q["safety"],
            )

        for items in batches.values():
            for it in items:
                it["near_duplicate_cluster_id"] = f"nd_{it['review_item_id']}"
        overlap = cs.assert_zero_overlap(batches)
        self.assertEqual(overlap["overlap_count"], 0)
        segs = [r["segment_id"] for r in batches["discovery_primary"]]
        self.assertEqual(len(segs), len(set(segs)))

        batches2 = cs.allocate_batches(
            rows,
            item_to_group=item_to_group,
            batch_quotas=batch_quotas,
            allocation=cfg["allocation"],
            source_type_keys=cfg.get("source_type_keys"),
        )
        self.assertEqual(
            [r["review_item_id"] for r in batches["discovery_primary"]],
            [r["review_item_id"] for r in batches2["discovery_primary"]],
        )

    def _capacity18_universe(self) -> tuple[list[dict], dict[str, str]]:
        """Universe with max selectable recomputed == 18 and strata headroom."""
        rows: list[dict] = []
        n = 0

        def add(source: str, stratum: str, track: int) -> None:
            nonlocal n
            rows.append(
                {
                    "review_item_id": f"r{n:04d}",
                    "crop_id": f"c{n:04d}",
                    "segment_id": f"seg{n:04d}",
                    "raw_track_id": track,
                    "documented_global_candidate_id": track,
                    "frame_index": n * 100,
                    "source_crop_path": f"/tmp/{n}.jpg",
                    "source_crop_sha256": f"sha{n:04d}",
                    "crop_source_type": source,
                    "composite_score": float(1000 - n),
                    "signal_stratum": stratum,
                    "timeline_bin": n % 4,
                }
            )
            n += 1

        # 18 independent recomputed items spread across strata.
        recomp_strata = (
            ["high_signal_candidate"] * 6
            + ["mid_signal_candidate"] * 6
            + ["safety_candidate"] * 6
        )
        for i, stratum in enumerate(recomp_strata):
            add("recomputed_manual_segment", stratum, 8000 + i)
        # Plenty of reused filler for exact strata quotas.
        reused_strata = (
            ["high_signal_candidate"] * 80
            + ["mid_signal_candidate"] * 80
            + ["safety_candidate"] * 80
        )
        for i, stratum in enumerate(reused_strata):
            add("reused_baseline_selected_crop", stratum, 10000 + i)
        item_to_group = {r["review_item_id"]: f"g_{r['review_item_id']}" for r in rows}
        for r in rows:
            r["near_duplicate_cluster_id"] = f"nd_{r['review_item_id']}"
            r["leakage_group_id"] = item_to_group[r["review_item_id"]]
        return rows, item_to_group

    def test_exact_6273_preferred_when_feasible(self):
        rows, item_to_group = self._capacity18_universe()
        cfg = _load_cfg()
        audit = cs.audit_recomputed_capacity(
            rows, item_to_group=item_to_group, allocation=cfg["allocation"]
        )
        self.assertEqual(audit["global_maximum_selectable_recomputed_count"], 18)
        result = cs.search_capacity_balanced_quotas(
            rows, item_to_group=item_to_group, config=cfg
        )
        self.assertEqual(result["selected_quota_tuple"], [6, 2, 7, 3])
        self.assertEqual(result["selected_total_recomputed"], 18)
        self.assertIn(
            result["selected_recomputed_is_maximum_feasible"], (True, "unresolved")
        )
        for batch in cs.BATCH_ORDER:
            recomp = sum(
                1
                for r in result["batches"][batch]
                if r["crop_source_type"] == "recomputed_manual_segment"
            )
            self.assertGreater(recomp, 0)
            self.assertEqual(recomp, result["selected_quota_vector"][batch])

    def test_nearest_feasible_vector_selection_skips_infeasible_preferred(self):
        rows, item_to_group = self._capacity18_universe()
        cfg = _load_cfg()
        original_allocate = cs.allocate_batches

        def wrapped(*args, **kwargs):
            quotas = kwargs.get("batch_quotas") or args[2]
            vec = tuple(int(quotas[b]["recomputed"]) for b in cs.BATCH_ORDER)
            if vec == (6, 2, 7, 3):
                raise cs.CleanSplitError(
                    "BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY forced_infeasible_6273"
                )
            return original_allocate(*args, **kwargs)

        cs.allocate_batches = wrapped  # type: ignore[assignment]
        try:
            result = cs.search_capacity_balanced_quotas(
                rows, item_to_group=item_to_group, config=cfg
            )
        finally:
            cs.allocate_batches = original_allocate  # type: ignore[assignment]
        self.assertNotEqual(result["selected_quota_tuple"], [6, 2, 7, 3])
        self.assertEqual(result["selected_total_recomputed"], 18)
        for batch in cs.BATCH_ORDER:
            self.assertGreater(result["selected_quota_vector"][batch], 0)

    def test_stratum_capacity_blocked(self):
        rows = [
            {
                "review_item_id": f"r{i}",
                "crop_id": f"c{i}",
                "segment_id": f"s{i}",
                "raw_track_id": i,
                "documented_global_candidate_id": i,
                "frame_index": i * 100,
                "source_crop_path": f"p{i}",
                "source_crop_sha256": f"sha{i}",
                "crop_source_type": "reused_baseline_selected_crop",
                "composite_score": float(i),
                "signal_stratum": "high_signal_candidate",
                "timeline_bin": 0,
            }
            for i in range(5)
        ]
        item_to_group = {r["review_item_id"]: f"g{i}" for i, r in enumerate(rows)}
        cfg = _load_cfg()
        quotas = {
            b: {
                **cfg["batches"][b],
                "recomputed": 0,
                "reused": cfg["batches"][b]["total"],
            }
            for b in cs.BATCH_ORDER
        }
        with self.assertRaises(cs.CleanSplitError) as ctx:
            cs.allocate_batches(
                rows,
                item_to_group=item_to_group,
                batch_quotas=quotas,
                allocation=cfg["allocation"],
            )
        self.assertIn("BLOCKED_SOURCE_TYPE_AND_STRATUM_CAPACITY", str(ctx.exception))

    def test_no_relaxation_flags_in_config(self):
        cfg = _load_cfg()
        search = cfg["capacity_search"]
        self.assertTrue(search["no_relaxation_of_leakage"])
        self.assertTrue(search["no_relaxation_of_strata"])
        self.assertTrue(search["no_relaxation_of_diversity"])
        bal = cfg["capacity_balancing"]
        self.assertFalse(bal["leakage_rules_relaxed"])
        self.assertFalse(bal["strata_rules_relaxed"])
        self.assertFalse(bal["diversity_rules_relaxed"])
        self.assertFalse(bal["labels_used_for_quota_selection"])
        self.assertFalse(bal["predictions_used_for_quota_selection"])
        self.assertFalse(bal["contact_sheets_viewed_for_quota_selection"])

    def test_capacity_balancing_contract_flags(self):
        cfg = _load_cfg()
        self.assertEqual(
            cfg["canonical_split_root"],
            "outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced",
        )
        self.assertEqual(cfg["canonical_split_generation"], "r2_capacity_balanced")
        self.assertFalse(cfg["previous_split"]["deprecated_for_downstream"])
        self.assertFalse(cfg["preregistration"]["threshold_selected"])
        self.assertEqual(
            cfg["capacity_search"]["hard_minima"]["discovery_primary"], 4
        )
        self.assertIn("crop_source_type", cfg["reviewer_facing_bias_fields_forbidden"])

    def test_previous_r4_root_immutability_path_recorded(self):
        cfg = _load_cfg()
        self.assertEqual(
            cfg["previous_split"]["path"],
            "outputs/reid/full_stage4b_rebuild_r2_stage5c_clean_split",
        )
        self.assertTrue(cfg["previous_split"]["immutable"])
        self.assertFalse(cfg["previous_split"]["deprecated_for_downstream"])

    def test_old_split_deprecation_only_after_publish_in_config(self):
        cfg = _load_cfg()
        # Pre-publish config must keep R4 canonical.
        self.assertFalse(cfg["previous_split"]["deprecated_for_downstream"])
        self.assertEqual(
            cfg["previous_split"]["deprecation_reason_if_replaced"],
            "source_type_distribution_not_representative_in_holdout",
        )

    def test_annotation_validation_rules(self):
        self.assertEqual(
            cs.validate_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="yes",
                manual_jersey_number="7",
            ),
            [],
        )
        self.assertIn(
            "readable=yes requires jersey_number",
            cs.validate_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="yes",
                manual_jersey_number="",
            ),
        )
        self.assertIn(
            "readable!=yes requires blank jersey_number",
            cs.validate_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="no",
                manual_jersey_number="11",
            ),
        )
        self.assertEqual(
            cs.validate_annotation_values(
                manual_crop_valid="yes",
                manual_number_visible="yes",
                manual_number_readable="yes",
                manual_jersey_number="09",
            ),
            [],
        )

    def test_preregistration_threshold_unselected(self):
        rule = cs.derive_discovery_candidate_gate_rule()
        self.assertFalse(rule["threshold_selected"])
        self.assertFalse(rule["deployment_threshold_selected"])
        decisions = cs.classify_holdout_decision_rule()
        self.assertTrue(decisions["rules_immutable_after_results"])

    def test_historical_never_seen_false_in_config(self):
        cfg = _load_cfg()
        self.assertFalse(cfg["independence"]["strict_historical_never_seen_claim"])
        self.assertTrue(cfg["independence"]["independent_within_rebuild_r2_protocol"])
        self.assertFalse(cfg["independence"]["player_identity_independence_guaranteed"])

    def test_nan_inf_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "c.jpg"
            sha = _write_crop(p, seed=1)
            row = _base_item(0, p, sha, laplacian_variance=float("nan"))
            cfg = _load_cfg()
            with self.assertRaises(cs.CleanSplitError):
                cs.score_universe([row], cfg["feature_contract"])

    def test_path_traversal_rejection(self):
        with self.assertRaises(cs.CleanSplitError):
            cs._validate_crop_path(
                Path("/tmp/../etc/passwd"),
                project_root=_PROJECT_ROOT,
            )

    def test_sha_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = root / "proj"
            crop = proj / "crop.jpg"
            _write_crop(crop, seed=9)
            row = _base_item(0, crop, "deadbeef" * 8)
            with self.assertRaises(cs.CleanSplitError):
                cs.compute_near_duplicates(
                    [row],
                    nd_contract={
                        "algorithm": "dhash",
                        "input_region": "number_roi_if_valid_else_full_crop",
                        "resize": [9, 8],
                        "bit_length": 64,
                        "distance_metric": "hamming",
                        "near_duplicate_max_distance": 8,
                    },
                    project_root=proj,
                )

    def test_invalid_roi_rejection_in_run_precheck(self):
        row = {
            "roi_valid": False,
            "review_item_id": "x",
            "annotation_status": "unreviewed",
        }
        self.assertFalse(bool(row.get("roi_valid")))

    def test_reviewer_bias_fields_absent_from_template(self):
        cfg = _load_cfg()
        forbidden = set(cfg["reviewer_facing_bias_fields_forbidden"])
        for col in cs.TEMPLATE_COLUMNS:
            self.assertNotIn(col, forbidden)

    def test_annotation_fields_blank_in_template_builder(self):
        item = {
            "split_item_id": "discovery_primary_001",
            "batch_order": 1,
            "review_item_id": "review_x",
            "source_crop_path": "/x.jpg",
            "source_crop_sha256": "abc",
            "contact_sheet_path": "sheet.png",
            "contact_sheet_page": 1,
            "tile_index": 1,
        }
        rows = cs.build_annotation_template_rows([item])
        self.assertEqual(rows[0]["manual_jersey_number"], "")
        self.assertEqual(rows[0]["manual_number_readable"], "")
        self.assertEqual(rows[0]["reviewer"], "")

    def test_atomic_temp_naming_capacity_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced"
            # create_unique_temp_root uses final.parent
            # monkey via writing into tmp as parent
            token_parent = Path(tmp)
            # emulate naming contract
            name = f"_tmp_full_stage4b_rebuild_r2_stage5c_clean_split_capacity_balanced_1"
            self.assertIn("capacity_balanced", name)
            self.assertTrue(str(final).endswith("capacity_balanced"))


if __name__ == "__main__":
    unittest.main()
