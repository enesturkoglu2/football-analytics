"""Synthetic unit tests for Stage 5B3G segmented ReID regression."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.embedding import EMBEDDING_DIM
from football_analytics.reid.segment_regression import (
    CONFIG_SCHEMA,
    PAIR_DELTAS_NAME,
    REPLACEMENT_MAP_NAME,
    SEGMENT_CANDIDATES_NAME,
    SEGMENT_CROP_MANIFEST_NAME,
    SEGMENT_EMB_INDEX_NAME,
    SEGMENT_EMB_NPZ_NAME,
    STATUS_NO_BASELINE,
    STATUS_NO_CROP,
    STATUS_RECOMPUTE,
    STATUS_REUSE,
    SUMMARY_NAME,
    SegmentRegressionError,
    build_pair_deltas,
    build_representation_plan,
    build_segment_candidates,
    embedding_vector_sha256,
    load_baseline_artifacts,
    load_regression_config,
    load_segment_view,
    run_segmented_reid_regression,
    select_segment_crops,
    validate_regression_config,
)
from football_analytics.reid.crop_select import load_crop_selection_config

CROP_CFG = _PROJECT_ROOT / "configs/reid/crop_selection_stage4b.yaml"
REG_CFG = _PROJECT_ROOT / "configs/reid/segmented_reid_regression_stage5b3.yaml"
POLICY = _PROJECT_ROOT / "configs/reid/manual_track_segmentation_policy_stage5b3.yaml"
REID_CFG = _PROJECT_ROOT / "configs/reid/benchmark_stage4b.yaml"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(EMBEDDING_DIM,)).astype(np.float32)
    v /= float(np.linalg.norm(v))
    return v.astype(np.float32)


def _obs(track_id: int, frame: int, *, conf: float = 0.9) -> dict:
    return {
        "frame_index": frame,
        "timestamp_sec": float(frame) / 25.0,
        "track_id": track_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": conf,
        "bbox_xyxy": [10.0, 10.0, 90.0, 110.0],
    }


def _canonical(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _mini_decisions() -> dict:
    return {
        "schema_version": "reid_manual_track_segment_decisions_v1",
        "status": "visually_reviewed_plan_not_applied",
        "automatic_application_enabled": False,
        "manual_visual_review_is_ground_truth": False,
        "raw_tracks_mutated": False,
        "global_id_map_changed": False,
        "tracks": [
            {
                "raw_track_id": 7,
                "decision": "manual_split_candidate",
                "manual_split_candidate": True,
                "probable_switch_event_count": 1,
                "ambiguous_existing_observation_frames": [5],
                "unobserved_gap_ranges": [],
                "segments": [
                    {
                        "segment_id": "raw_7_s01",
                        "raw_track_id": 7,
                        "frame_min": 0,
                        "frame_max": 4,
                        "include_existing_observations_only": True,
                        "proven_physical_identity": False,
                        "team_assignment": None,
                        "global_id": None,
                    },
                    {
                        "segment_id": "raw_7_s02",
                        "raw_track_id": 7,
                        "frame_min": 6,
                        "frame_max": 12,
                        "include_existing_observations_only": True,
                        "proven_physical_identity": False,
                        "team_assignment": None,
                        "global_id": None,
                    },
                ],
                "boundaries": [
                    {
                        "last_segment_frame": 4,
                        "next_segment_frame": 6,
                        "boundary_type": "overlap_ambiguous",
                        "exact_real_world_switch_frame_known": False,
                        "automatic_split_decision": False,
                    }
                ],
            },
            {
                "raw_track_id": 268,
                "decision": "no_split_contamination_control",
                "manual_split_candidate": False,
                "raw_track_preserved": True,
                "probable_switch_event_count": 0,
                "ambiguous_existing_observation_frames": [],
                "unobserved_gap_ranges": [],
                "segments": [],
                "boundaries": [],
            },
        ],
    }


def _segment_summary(**overrides) -> dict:
    base = {
        "status": "ok",
        "schema_version": "reid_manual_segment_view_summary_v1",
        "raw_track_count": 4,
        "raw_observation_count": 20,
        "derived_view_only": True,
        "segment_view_is_raw_track_replacement": False,
        "raw_tracks_mutated": False,
        "source_observations_preserved": True,
        "created_observation_count": 0,
        "interpolated_observation_count": 0,
        "deleted_observation_count": 0,
        "duplicated_source_observation_count": 0,
        "uncovered_source_observation_count": 0,
        "automatic_split_performed": False,
        "automatic_merge_performed": False,
        "automatic_link_performed": False,
        "global_id_rewrite_performed": False,
        "team_assignment_performed": False,
        "proven_physical_identity": False,
        "accuracy_claimed": False,
        "existing_component_audit": {
            "raw_component": [231, 635],
            "modified": False,
            "segment_component_assignment_performed": False,
        },
    }
    base.update(overrides)
    return base


def _build_segment_view(root: Path, *, with_conflict_track: bool = False) -> None:
    """Build a tiny segment view: split 7, control 268, embedded 10, non-embedded 99.

    With ``with_conflict_track`` an extra preserved track 12 is added whose
    frames exactly overlap track 10 on frame 24 (segment-level exact conflict).
    """
    segs = [
        {
            "schema_version": "reid_track_segment_v1",
            "segment_id": "raw_7_s01",
            "raw_track_id": 7,
            "segment_kind": "manual_split_segment",
            "segment_index": 1,
            "decision": "manual_split_candidate",
            "source_decision_status": "visually_reviewed_plan_not_applied",
            "configured_frame_min": 0,
            "configured_frame_max": 4,
            "first_observation_frame": 0,
            "last_observation_frame": 4,
            "observation_count": 5,
            "ambiguous_observation_count_excluded": 1,
            "include_existing_observations_only": True,
            "source_observations_preserved": True,
            "proven_physical_identity": False,
            "team_assignment": None,
            "global_id": None,
            "automatic_split_applied": False,
            "automatic_merge_applied": False,
            "automatic_link_applied": False,
            "global_id_rewrite_applied": False,
        },
        {
            "schema_version": "reid_track_segment_v1",
            "segment_id": "raw_7_s02",
            "raw_track_id": 7,
            "segment_kind": "manual_split_segment",
            "segment_index": 2,
            "decision": "manual_split_candidate",
            "source_decision_status": "visually_reviewed_plan_not_applied",
            "configured_frame_min": 6,
            "configured_frame_max": 12,
            "first_observation_frame": 6,
            "last_observation_frame": 12,
            "observation_count": 4,
            "ambiguous_observation_count_excluded": 1,
            "include_existing_observations_only": True,
            "source_observations_preserved": True,
            "proven_physical_identity": False,
            "team_assignment": None,
            "global_id": None,
            "automatic_split_applied": False,
            "automatic_merge_applied": False,
            "automatic_link_applied": False,
            "global_id_rewrite_applied": False,
        },
        {
            "schema_version": "reid_track_segment_v1",
            "segment_id": "raw_10_full",
            "raw_track_id": 10,
            "segment_kind": "preserved_full_track",
            "segment_index": None,
            "decision": "preserved_full_track",
            "source_decision_status": "visually_reviewed_plan_not_applied",
            "configured_frame_min": 20,
            "configured_frame_max": 28,
            "first_observation_frame": 20,
            "last_observation_frame": 28,
            "observation_count": 3,
            "ambiguous_observation_count_excluded": 0,
            "include_existing_observations_only": True,
            "source_observations_preserved": True,
            "proven_physical_identity": False,
            "team_assignment": None,
            "global_id": None,
            "automatic_split_applied": False,
            "automatic_merge_applied": False,
            "automatic_link_applied": False,
            "global_id_rewrite_applied": False,
        },
        {
            "schema_version": "reid_track_segment_v1",
            "segment_id": "raw_99_full",
            "raw_track_id": 99,
            "segment_kind": "preserved_full_track",
            "segment_index": None,
            "decision": "preserved_full_track",
            "source_decision_status": "visually_reviewed_plan_not_applied",
            "configured_frame_min": 30,
            "configured_frame_max": 32,
            "first_observation_frame": 30,
            "last_observation_frame": 32,
            "observation_count": 2,
            "ambiguous_observation_count_excluded": 0,
            "include_existing_observations_only": True,
            "source_observations_preserved": True,
            "proven_physical_identity": False,
            "team_assignment": None,
            "global_id": None,
            "automatic_split_applied": False,
            "automatic_merge_applied": False,
            "automatic_link_applied": False,
            "global_id_rewrite_applied": False,
        },
        {
            "schema_version": "reid_track_segment_v1",
            "segment_id": "raw_268_full",
            "raw_track_id": 268,
            "segment_kind": "no_split_control",
            "segment_index": None,
            "decision": "no_split_contamination_control",
            "source_decision_status": "visually_reviewed_plan_not_applied",
            "configured_frame_min": 40,
            "configured_frame_max": 49,
            "first_observation_frame": 40,
            "last_observation_frame": 49,
            "observation_count": 3,
            "ambiguous_observation_count_excluded": 0,
            "include_existing_observations_only": True,
            "source_observations_preserved": True,
            "proven_physical_identity": False,
            "team_assignment": None,
            "global_id": None,
            "automatic_split_applied": False,
            "automatic_merge_applied": False,
            "automatic_link_applied": False,
            "global_id_rewrite_applied": False,
        },
    ]
    if with_conflict_track:
        segs.append(
            {
                "schema_version": "reid_track_segment_v1",
                "segment_id": "raw_12_full",
                "raw_track_id": 12,
                "segment_kind": "preserved_full_track",
                "segment_index": None,
                "decision": "preserved_full_track",
                "source_decision_status": "visually_reviewed_plan_not_applied",
                "configured_frame_min": 22,
                "configured_frame_max": 26,
                "first_observation_frame": 22,
                "last_observation_frame": 26,
                "observation_count": 3,
                "ambiguous_observation_count_excluded": 0,
                "include_existing_observations_only": True,
                "source_observations_preserved": True,
                "proven_physical_identity": False,
                "team_assignment": None,
                "global_id": None,
                "automatic_split_applied": False,
                "automatic_merge_applied": False,
                "automatic_link_applied": False,
                "global_id_rewrite_applied": False,
            }
        )
    # sort by raw_track_id then manual first
    segs.sort(key=lambda r: (r["raw_track_id"], 0 if r["segment_kind"] == "manual_split_segment" else 1, r["segment_index"] or 0))

    assigned_specs = [
        ("raw_7_s01", 7, [0, 1, 2, 3, 4]),
        ("raw_7_s02", 7, [6, 8, 10, 12]),
        ("raw_10_full", 10, [20, 24, 28]),
        ("raw_99_full", 99, [30, 32]),
        ("raw_268_full", 268, [40, 44, 49]),
    ]
    if with_conflict_track:
        # frame 24 exactly overlaps raw_10_full
        assigned_specs.insert(3, ("raw_12_full", 12, [22, 24, 26]))
    assigned = []
    row_idx = 0
    for sid, tid, frames in assigned_specs:
        seg = next(s for s in segs if s["segment_id"] == sid)
        for frame in frames:
            src = _obs(tid, frame)
            digest = hashlib.sha256(_canonical(src).encode()).hexdigest()
            assigned.append(
                {
                    "schema_version": "reid_segment_observation_v1",
                    "segment_id": sid,
                    "raw_track_id": tid,
                    "segment_kind": seg["segment_kind"],
                    "segment_index": seg["segment_index"],
                    "frame_index": frame,
                    "source_row_index": row_idx,
                    "source_observation_sha256": digest,
                    "source_observation": src,
                }
            )
            row_idx += 1
    assigned.sort(key=lambda r: (
        r["raw_track_id"],
        0 if r["segment_kind"] == "manual_split_segment" else 1,
        -1 if r["segment_index"] is None else r["segment_index"],
        r["frame_index"],
        r["source_row_index"],
    ))

    amb = _obs(7, 5)
    unassigned = [
        {
            "schema_version": "reid_unassigned_segment_observation_v1",
            "raw_track_id": 7,
            "frame_index": 5,
            "reason": "manual_ambiguous_existing_observation",
            "decision_source": "manual_track_segment_decisions",
            "source_row_index": 999,
            "source_observation_sha256": hashlib.sha256(_canonical(amb).encode()).hexdigest(),
            "source_observation": amb,
            "deleted": False,
            "interpolated": False,
            "assigned_to_segment": False,
            "team_assignment": None,
            "global_id": None,
        }
    ]
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "track_segments.jsonl", segs)
    _write_jsonl(root / "segment_observations.jsonl", assigned)
    _write_jsonl(root / "unassigned_observations.jsonl", unassigned)
    _write_json(root / "segment_view_summary.json", _segment_summary())


def _build_baseline(
    root: Path,
    *,
    vectors: dict[int, np.ndarray],
    conflict_pairs: set[tuple[int, int]] | None = None,
) -> None:
    conflict_pairs = conflict_pairs or set()
    (root / "crops").mkdir(parents=True)
    (root / "aggregation").mkdir(parents=True)
    (root / "candidates").mkdir(parents=True)
    (root / "embeddings").mkdir(parents=True)
    # minimal crop manifest
    crop_rows = []
    for tid in sorted(vectors):
        crop_rows.append(
            {
                "crop_id": f"track_{tid}_frame_0_rank_1",
                "track_id": tid,
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "source_video": "fake.mp4",
                "bbox_xyxy": [10, 10, 90, 110],
                "bbox_width": 80.0,
                "bbox_height": 100.0,
                "bbox_area": 8000.0,
                "short_side": 80.0,
                "detection_confidence": 0.9,
                "quality_score": 7200.0,
                "crop_relative_path": f"crops/track_{tid}/crop_0_1.jpg",
                "selection_rank": 1,
                "schema_version": "reid_crop_manifest_v1",
            }
        )
    _write_jsonl(root / "crops" / "crop_manifest.jsonl", crop_rows)

    tids = sorted(vectors)
    mat = np.stack([vectors[t] for t in tids], axis=0).astype(np.float32)
    np.savez(
        root / "aggregation" / "track_embeddings.npz",
        vectors=mat,
        track_ids=np.asarray(tids, dtype=np.int64),
        crop_counts=np.ones(len(tids), dtype=np.int64),
        first_frames=np.zeros(len(tids), dtype=np.int64),
        last_frames=np.full(len(tids), 10, dtype=np.int64),
        observation_counts=np.full(len(tids), 3, dtype=np.int64),
    )
    index_rows = []
    for i, tid in enumerate(tids):
        index_rows.append(
            {
                "track_id": tid,
                "crop_ids": [f"track_{tid}_frame_0_rank_1"],
                "crop_count": 1,
                "embedding_row": i,
                "aggregation": "l2_mean",
                "embedding_shape": [EMBEDDING_DIM],
                "embedding_dtype": "float32",
                "l2_norm": 1.0,
                "observation_count": 3,
                "first_frame": 0,
                "last_frame": 10,
                "observed_frame_count": 3,
                "schema_version": "reid_track_embedding_v1",
            }
        )
    _write_jsonl(root / "aggregation" / "track_embeddings.jsonl", index_rows)

    pairs = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            sim = float(np.dot(vectors[a], vectors[b]))
            conflict = (a, b) in conflict_pairs
            pairs.append(
                {
                    "track_id_a": a,
                    "track_id_b": b,
                    "cosine_similarity": sim,
                    "temporal_gap_frames": 0,
                    "exact_frame_overlap_count": 1 if conflict else 0,
                    "exact_frame_conflict": conflict,
                    "span_interval_overlap": conflict,
                    "decision": "rejected" if conflict else "eligible_unthresholded",
                    "decision_reason": (
                        "exact_frame_conflict"
                        if conflict
                        else "similarity_threshold_pending"
                    ),
                    "schema_version": "reid_candidate_pair_v1",
                }
            )
    _write_jsonl(root / "candidates" / "candidate_pairs.jsonl", pairs)
    _write_json(
        root / "candidates" / "candidate_summary.json",
        {
            "status": "ok",
            "track_count": len(tids),
            "total_pairs": len(pairs),
            "similarity_threshold": None,
            "automatic_linking_performed": False,
            "schema_version": "reid_candidate_summary_v1",
        },
    )
    ckpt = b"fake-checkpoint-bytes"
    _write_json(
        root / "embeddings" / "embedding_summary.json",
        {
            "status": "ok",
            "checkpoint_path": "/tmp/fake_osnet.pth",
            "checkpoint_sha256": _sha(ckpt),
            "sn_reid_commit": "a" * 40,
            "schema_version": "reid_embedding_summary_v1",
        },
    )
    (root / "fake_checkpoint.bin").write_bytes(ckpt)


class DeterministicModel:
    def eval(self):
        return self

    def cpu(self):
        return self

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        b = batch.shape[0]
        base = torch.arange(EMBEDDING_DIM, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
        offsets = batch.reshape(b, -1).mean(dim=1, keepdim=True)
        return base + offsets + 1.0


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._i = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def release(self):
        self._opened = False


class ConfigValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        cfg = load_regression_config(REG_CFG)
        self.assertEqual(cfg["schema_version"], CONFIG_SCHEMA)
        bad = copy.deepcopy(cfg)
        bad["representation"]["reuse_manual_split_raw_track_embedding"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["representation"]["expand_baseline_embedding_coverage_for_unaffected_tracks"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["crop_selection"]["ambiguous_observations_allowed"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["crop_selection"]["unassigned_observations_allowed"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["crop_selection"]["fallback_to_parent_raw_track_crops"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["crop_selection"]["missing_frames_interpolated"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["candidate_generation"]["similarity_threshold"] = 0.8
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        for key in ("automatic_link_enabled", "automatic_reject_enabled", "component_building_enabled"):
            bad = copy.deepcopy(cfg)
            bad["candidate_generation"][key] = True
            with self.assertRaises(SegmentRegressionError):
                validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["global_identity"]["global_id_rewrite_enabled"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["global_identity"]["component_assignment_enabled"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)
        bad = copy.deepcopy(cfg)
        bad["regression"]["accuracy_claim_allowed"] = True
        with self.assertRaises(SegmentRegressionError):
            validate_regression_config(bad)


class SegmentViewValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "view"
            _build_segment_view(root)
            view = load_segment_view(root)
            self.assertEqual(len(view["segments"]), 5)
            # duplicate segment id
            text = (root / "track_segments.jsonl").read_text(encoding="utf-8")
            (root / "track_segments.jsonl").write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaises(SegmentRegressionError):
                load_segment_view(root)
            _build_segment_view(root)
            summary = json.loads((root / "segment_view_summary.json").read_text(encoding="utf-8"))
            summary["created_observation_count"] = 1
            _write_json(root / "segment_view_summary.json", summary)
            with self.assertRaises(SegmentRegressionError):
                load_segment_view(root)
            _build_segment_view(root)
            summary = json.loads((root / "segment_view_summary.json").read_text(encoding="utf-8"))
            summary["raw_tracks_mutated"] = True
            _write_json(root / "segment_view_summary.json", summary)
            with self.assertRaises(SegmentRegressionError):
                load_segment_view(root)
            _build_segment_view(root)
            rows = [json.loads(l) for l in (root / "unassigned_observations.jsonl").read_text().splitlines() if l.strip()]
            rows[0]["segment_id"] = "raw_7_s01"
            _write_jsonl(root / "unassigned_observations.jsonl", rows)
            with self.assertRaises(SegmentRegressionError):
                load_segment_view(root)


class BaselineValidationTests(unittest.TestCase):
    def test_valid_and_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base"
            vectors = {7: _unit(1), 10: _unit(2), 268: _unit(3)}
            _build_baseline(root, vectors=vectors)
            base = load_baseline_artifacts(root)
            self.assertEqual(len(base["vectors_by_track"]), 3)
            # threshold non-null
            summary = json.loads((root / "candidates" / "candidate_summary.json").read_text())
            summary["similarity_threshold"] = 0.5
            _write_json(root / "candidates" / "candidate_summary.json", summary)
            with self.assertRaises(SegmentRegressionError):
                load_baseline_artifacts(root)


class RepresentationPlanTests(unittest.TestCase):
    def test_plan_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = Path(tmp) / "view"
            base_dir = Path(tmp) / "base"
            _build_segment_view(view_dir)
            vectors = {7: _unit(1), 10: _unit(2), 268: _unit(3)}
            _build_baseline(base_dir, vectors=vectors)
            view = load_segment_view(view_dir)
            baseline = load_baseline_artifacts(base_dir)
            decisions = _mini_decisions()
            # inject validated segment indexes like loader would
            from football_analytics.reid.segments import validate_segment_decisions

            decisions = validate_segment_decisions(decisions)
            plan = build_representation_plan(
                segments=view["segments"], decisions=decisions, baseline=baseline
            )
            by = {p["segment_id"]: p for p in plan}
            self.assertEqual(by["raw_7_s01"]["status"], STATUS_RECOMPUTE)
            self.assertEqual(by["raw_7_s02"]["status"], STATUS_RECOMPUTE)
            self.assertTrue(by["raw_7_s01"]["parent_mixed_embedding_retired"])
            self.assertEqual(by["raw_268_full"]["status"], STATUS_REUSE)
            self.assertEqual(by["raw_10_full"]["status"], STATUS_REUSE)
            self.assertEqual(by["raw_99_full"]["status"], STATUS_NO_BASELINE)


class EndToEndRegressionTests(unittest.TestCase):
    def test_end_to_end_atomic_and_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            view_dir = tmp_path / "view"
            base_dir = tmp_path / "base"
            out_dir = tmp_path / "out"
            video = tmp_path / "fake.mp4"
            video.write_bytes(b"not-a-real-video")
            ckpt = tmp_path / "ckpt.bin"
            ckpt.write_bytes(b"fake-checkpoint-bytes")
            sn_root = tmp_path / "sn-reid"
            sn_root.mkdir()
            (sn_root / "torchreid").mkdir()
            subprocess.run(["git", "init"], cwd=sn_root, check=True, capture_output=True)
            (sn_root / "torchreid" / "__init__.py").write_text("#\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=sn_root, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
                cwd=sn_root,
                check=True,
                capture_output=True,
            )
            commit = subprocess.check_output(["git", "-C", str(sn_root), "rev-parse", "HEAD"], text=True).strip()

            _build_segment_view(view_dir)
            vectors = {7: _unit(1), 10: _unit(2), 268: _unit(3)}
            _build_baseline(base_dir, vectors=vectors)
            # align checkpoint hash/commit in baseline summary
            emb_sum = json.loads((base_dir / "embeddings" / "embedding_summary.json").read_text())
            emb_sum["checkpoint_sha256"] = _sha(ckpt.read_bytes())
            emb_sum["sn_reid_commit"] = commit
            emb_sum["checkpoint_path"] = str(ckpt)
            _write_json(base_dir / "embeddings" / "embedding_summary.json", emb_sum)

            decisions_path = tmp_path / "decisions.yaml"
            decisions_path.write_text(yaml.safe_dump(_mini_decisions()), encoding="utf-8")

            frames = [np.full((160, 160, 3), fill_value=(i * 3) % 200, dtype=np.uint8) for i in range(60)]

            def open_capture(_path: str):
                return FakeCapture(frames)

            before_view = (view_dir / "segment_view_summary.json").read_bytes()
            before_base = (base_dir / "aggregation" / "track_embeddings.npz").read_bytes()

            result = run_segmented_reid_regression(
                video=video,
                segment_view_dir=view_dir,
                baseline_run_dir=base_dir,
                segmentation_policy=POLICY,
                segment_decisions=decisions_path,
                crop_config=CROP_CFG,
                reid_config=REID_CFG,
                regression_config=REG_CFG,
                sn_reid_root=sn_root,
                checkpoint=ckpt,
                output_dir=out_dir,
                overwrite=False,
                open_capture=open_capture,
                video_size=(160, 160),
                model_builder=lambda model_name=None: DeterministicModel(),
                weight_loader=lambda model, path: None,
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue((out_dir / SUMMARY_NAME).is_file())
            self.assertTrue((out_dir / SEGMENT_EMB_NPZ_NAME).is_file())
            self.assertTrue((out_dir / SEGMENT_CANDIDATES_NAME).is_file())
            self.assertTrue((out_dir / REPLACEMENT_MAP_NAME).is_file())
            self.assertTrue((out_dir / PAIR_DELTAS_NAME).is_file())
            self.assertTrue((out_dir / SEGMENT_CROP_MANIFEST_NAME).is_file())
            self.assertTrue((out_dir / SEGMENT_EMB_INDEX_NAME).is_file())
            self.assertEqual((view_dir / "segment_view_summary.json").read_bytes(), before_view)
            self.assertEqual((base_dir / "aggregation" / "track_embeddings.npz").read_bytes(), before_base)

            entities = [json.loads(l) for l in (out_dir / SEGMENT_EMB_INDEX_NAME).read_text().splitlines() if l.strip()]
            by = {e["segment_id"]: e for e in entities}
            self.assertEqual(by["raw_7_s01"]["representation_status"], STATUS_RECOMPUTE)
            self.assertEqual(by["raw_7_s02"]["representation_status"], STATUS_RECOMPUTE)
            self.assertEqual(by["raw_10_full"]["representation_status"], STATUS_REUSE)
            self.assertEqual(by["raw_268_full"]["representation_status"], STATUS_REUSE)
            self.assertEqual(by["raw_99_full"]["representation_status"], STATUS_NO_BASELINE)
            self.assertTrue(by["raw_7_s01"]["parent_mixed_embedding_retired"])
            # reused vector SHA matches baseline
            self.assertEqual(
                by["raw_10_full"]["embedding_sha256"],
                embedding_vector_sha256(vectors[10]),
            )
            # ambiguous frame not in crop manifest
            crops = [json.loads(l) for l in (out_dir / SEGMENT_CROP_MANIFEST_NAME).read_text().splitlines() if l.strip()]
            self.assertTrue(all(c["frame_index"] != 5 for c in crops))
            self.assertTrue(all(c["ambiguous_observation_used"] is False for c in crops))
            self.assertTrue(all(c["parent_mixed_raw_embedding_reused"] is False for c in crops))
            # no mixed parent entity
            self.assertNotIn("7", {str(e["segment_id"]) for e in entities})
            self.assertFalse(any(e["segment_id"] == "raw_7_full" for e in entities))

            # collision without overwrite
            with self.assertRaises(SegmentRegressionError):
                run_segmented_reid_regression(
                    video=video,
                    segment_view_dir=view_dir,
                    baseline_run_dir=base_dir,
                    segmentation_policy=POLICY,
                    segment_decisions=decisions_path,
                    crop_config=CROP_CFG,
                    reid_config=REID_CFG,
                    regression_config=REG_CFG,
                    sn_reid_root=sn_root,
                    checkpoint=ckpt,
                    output_dir=out_dir,
                    overwrite=False,
                    open_capture=open_capture,
                    video_size=(160, 160),
                    model_builder=lambda model_name=None: DeterministicModel(),
                    weight_loader=lambda model, path: None,
                )

            # overwrite works
            result2 = run_segmented_reid_regression(
                video=video,
                segment_view_dir=view_dir,
                baseline_run_dir=base_dir,
                segmentation_policy=POLICY,
                segment_decisions=decisions_path,
                crop_config=CROP_CFG,
                reid_config=REID_CFG,
                regression_config=REG_CFG,
                sn_reid_root=sn_root,
                checkpoint=ckpt,
                output_dir=out_dir,
                overwrite=True,
                open_capture=open_capture,
                video_size=(160, 160),
                model_builder=lambda model_name=None: DeterministicModel(),
                weight_loader=lambda model, path: None,
            )
            self.assertEqual(result2["status"], "ok")
            # no tmp leftovers
            leftovers = list(out_dir.parent.glob("_tmp_reid_segreg_*")) + list(
                out_dir.parent.glob("_backup_reid_segreg_*")
            )
            self.assertEqual(leftovers, [])

            summary = json.loads((out_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["similarity_threshold"], None)
            self.assertFalse(summary["accuracy_claimed"])
            self.assertFalse(summary["global_id_rewrite_performed"])
            self.assertEqual(summary["ambiguous_observation_crop_count"], 0)
            self.assertEqual(summary["unaffected_pair_similarity_mismatch_count"], 0)
            self.assertEqual(summary["existing_component_audit"]["raw_component"], [231, 635])
            self.assertFalse(summary["existing_component_audit"]["raw_231_s01_inherits_component"])
            self.assertFalse(summary["existing_component_audit"]["raw_231_s02_automatically_links_to_635"])

            # failure leaves no final when writing to new dir
            bad_out = tmp_path / "bad_out"
            with mock.patch(
                "football_analytics.reid.segment_regression.build_segment_candidates",
                side_effect=SegmentRegressionError("boom"),
            ):
                with self.assertRaises(SegmentRegressionError):
                    run_segmented_reid_regression(
                        video=video,
                        segment_view_dir=view_dir,
                        baseline_run_dir=base_dir,
                        segmentation_policy=POLICY,
                        segment_decisions=decisions_path,
                        crop_config=CROP_CFG,
                        reid_config=REID_CFG,
                        regression_config=REG_CFG,
                        sn_reid_root=sn_root,
                        checkpoint=ckpt,
                        output_dir=bad_out,
                        overwrite=False,
                        open_capture=open_capture,
                        video_size=(160, 160),
                        model_builder=lambda model_name=None: DeterministicModel(),
                        weight_loader=lambda model, path: None,
                    )
            self.assertFalse(bad_out.exists())


class CandidateHelperTests(unittest.TestCase):
    def test_exact_overlap_and_ranking(self) -> None:
        entities = [
            {
                "segment_id": "raw_a_s01",
                "raw_track_id": 1,
                "embedding_available": True,
                "representation_source": "recomputed_manual_segment",
            },
            {
                "segment_id": "raw_a_s02",
                "raw_track_id": 1,
                "embedding_available": True,
                "representation_source": "recomputed_manual_segment",
            },
            {
                "segment_id": "raw_b_full",
                "raw_track_id": 2,
                "embedding_available": True,
                "representation_source": "reused_baseline_raw_track_embedding",
            },
        ]
        vectors = {
            "raw_a_s01": _unit(10),
            "raw_a_s02": _unit(11),
            "raw_b_full": _unit(12),
        }
        frames = {
            "raw_a_s01": {1, 2, 3},
            "raw_a_s02": {10, 11},
            "raw_b_full": {2, 20},  # exact overlap with s01 on frame 2
        }
        pairs = build_segment_candidates(
            entities=entities, vectors_by_segment=vectors, frames_by_segment=frames
        )
        overlap = next(
            p for p in pairs if {p["segment_id_a"], p["segment_id_b"]} == {"raw_a_s01", "raw_b_full"}
        )
        self.assertTrue(overlap["exact_same_frame_overlap"])
        self.assertIsNone(overlap["cosine_similarity"])
        same_parent = next(
            p for p in pairs if {p["segment_id_a"], p["segment_id_b"]} == {"raw_a_s01", "raw_a_s02"}
        )
        self.assertTrue(same_parent["same_parent_raw_track"])
        self.assertIsNotNone(same_parent["cosine_similarity"])
        self.assertIsNotNone(same_parent["rank"])
        # span overlap alone: disjoint exact frames but overlapping spans
        frames2 = {
            "raw_a_s01": {1, 5},
            "raw_a_s02": {2, 4},
            "raw_b_full": {10},
        }
        pairs2 = build_segment_candidates(
            entities=entities, vectors_by_segment=vectors, frames_by_segment=frames2
        )
        span_only = next(
            p for p in pairs2 if {p["segment_id_a"], p["segment_id_b"]} == {"raw_a_s01", "raw_a_s02"}
        )
        self.assertTrue(span_only["span_overlap"])
        self.assertFalse(span_only["exact_same_frame_overlap"])
        self.assertIsNotNone(span_only["cosine_similarity"])


class PairDeltaConflictTests(unittest.TestCase):
    """Unaffected reused pairs: exact-frame-conflict audit semantics."""

    @staticmethod
    def _entity(sid: str, rid: int) -> dict:
        return {
            "segment_id": sid,
            "raw_track_id": rid,
            "representation_source": "reused_baseline_raw_track_embedding",
            "embedding_available": True,
        }

    @staticmethod
    def _cand(
        sid_a: str,
        sid_b: str,
        rid_a: int,
        rid_b: int,
        *,
        sim: float | None,
        overlap: bool,
        overlap_count: int = 0,
        rank: int | None = None,
    ) -> dict:
        return {
            "segment_id_a": sid_a,
            "segment_id_b": sid_b,
            "raw_track_id_a": rid_a,
            "raw_track_id_b": rid_b,
            "same_parent_raw_track": False,
            "cosine_similarity": sim,
            "rank": rank,
            "exact_same_frame_overlap": overlap,
            "exact_frame_overlap_count": overlap_count,
            "span_overlap": overlap,
            "similarity_threshold": None,
            "automatic_link_decision": None,
            "automatic_reject_decision": None,
            "component_assignment": None,
            "manual_review_required": True,
        }

    def _fixture(self):
        v10, v12, v268 = _unit(2), _unit(4), _unit(3)
        vectors_by_segment = {
            "raw_10_full": v10,
            "raw_12_full": v12,
            "raw_268_full": v268,
        }
        baseline_vectors = {10: v10, 12: v12, 268: v268}
        entity_by_segment = {
            "raw_10_full": self._entity("raw_10_full", 10),
            "raw_12_full": self._entity("raw_12_full", 12),
            "raw_268_full": self._entity("raw_268_full", 268),
        }
        sim_10_12 = float(np.dot(v10, v12))
        sim_10_268 = float(np.dot(v10, v268))
        sim_12_268 = float(np.dot(v12, v268))
        baseline_pairs = {
            (10, 12): {
                "cosine_similarity": sim_10_12,
                "exact_frame_conflict": True,
                "exact_frame_overlap_count": 1,
                "decision": "rejected",
            },
            (10, 268): {
                "cosine_similarity": sim_10_268,
                "exact_frame_conflict": False,
                "exact_frame_overlap_count": 0,
                "decision": "eligible_unthresholded",
            },
            (12, 268): {
                "cosine_similarity": sim_12_268,
                "exact_frame_conflict": False,
                "exact_frame_overlap_count": 0,
                "decision": "eligible_unthresholded",
            },
        }
        candidates = [
            self._cand(
                "raw_10_full", "raw_12_full", 10, 12,
                sim=None, overlap=True, overlap_count=1,
            ),
            self._cand(
                "raw_10_full", "raw_268_full", 10, 268,
                sim=sim_10_268, overlap=False, rank=1,
            ),
            self._cand(
                "raw_12_full", "raw_268_full", 12, 268,
                sim=sim_12_268, overlap=False, rank=2,
            ),
        ]
        return (
            baseline_pairs,
            candidates,
            entity_by_segment,
            vectors_by_segment,
            baseline_vectors,
        )

    def _run(self, baseline_pairs, candidates, entities, seg_vecs, base_vecs):
        return build_pair_deltas(
            baseline_pairs=baseline_pairs,
            segment_candidates=candidates,
            entity_by_segment=entities,
            split_parents=set(),
            vectors_by_segment=seg_vecs,
            baseline_vectors_by_track=base_vecs,
        )

    def test_exact_conflict_audit_success(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        deltas = self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)
        conflict = [d for d in deltas if d["delta_kind"] == "unaffected_exact_frame_conflict"]
        normal = [d for d in deltas if d["delta_kind"] == "unaffected_reused_pair"]
        self.assertEqual(len(conflict), 1)
        self.assertEqual(len(normal), 2)
        row = conflict[0]
        self.assertEqual((row["baseline_track_id_a"], row["baseline_track_id_b"]), (10, 12))
        self.assertTrue(row["baseline_exact_frame_conflict"])
        self.assertTrue(row["segmented_exact_same_frame_overlap"])
        self.assertTrue(row["hard_rejected"])
        self.assertEqual(row["hard_reject_reason"], "exact_same_frame_overlap")
        self.assertFalse(row["segmented_ranked_candidate_available"])
        self.assertIsNone(row["segmented_rank"])
        self.assertTrue(row["similarity_match"])
        self.assertAlmostEqual(
            row["reused_vector_audit_cosine"],
            row["baseline_cosine_similarity"],
            places=5,
        )
        self.assertIsNone(row["automatic_link_decision"])
        self.assertIsNone(row["automatic_reject_decision"])
        self.assertIsNone(row["component_assignment"])
        self.assertFalse(row["accuracy_claimed"])
        # hard-rejected candidate itself stays unranked and similarity-null
        cand = candidates[0]
        self.assertIsNone(cand["rank"])
        self.assertIsNone(cand["cosine_similarity"])

    def test_conflict_audit_cosine_mismatch_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        baseline_pairs[(10, 12)]["cosine_similarity"] += 0.01
        with self.assertRaisesRegex(SegmentRegressionError, "audit cosine mismatch"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_baseline_conflict_segmented_nonconflict_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        candidates[0]["exact_same_frame_overlap"] = False
        candidates[0]["cosine_similarity"] = float(
            baseline_pairs[(10, 12)]["cosine_similarity"]
        )
        with self.assertRaisesRegex(SegmentRegressionError, "conflict flag mismatch"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_baseline_nonconflict_segmented_conflict_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        candidates[1]["exact_same_frame_overlap"] = True
        candidates[1]["cosine_similarity"] = None
        with self.assertRaisesRegex(SegmentRegressionError, "conflict flag mismatch"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_undetermined_hard_reject_provenance_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        candidates[0]["exact_same_frame_overlap"] = None
        with self.assertRaisesRegex(SegmentRegressionError, "provenance is undetermined"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_conflict_reused_vector_mismatch_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        base_vecs = dict(base_vecs)
        base_vecs[10] = _unit(77)
        with self.assertRaisesRegex(SegmentRegressionError, "reused vector differs"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_nonconflict_missing_candidate_still_fatal(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        candidates = [
            c
            for c in candidates
            if {c["segment_id_a"], c["segment_id_b"]} != {"raw_10_full", "raw_268_full"}
        ]
        with self.assertRaisesRegex(SegmentRegressionError, "missing segmented candidate"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_nonconflict_similarity_mismatch_still_fatal(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        candidates[1]["cosine_similarity"] = float(candidates[1]["cosine_similarity"]) + 0.01
        with self.assertRaisesRegex(SegmentRegressionError, "similarity mismatch"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)

    def test_missing_mapped_full_segment_rejected(self) -> None:
        baseline_pairs, candidates, entities, seg_vecs, base_vecs = self._fixture()
        entities = {k: v for k, v in entities.items() if k != "raw_12_full"}
        with self.assertRaisesRegex(SegmentRegressionError, "no mapped full segments"):
            self._run(baseline_pairs, candidates, entities, seg_vecs, base_vecs)


class ConflictEndToEndTests(unittest.TestCase):
    def test_summary_counts_with_conflict_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            view_dir = tmp_path / "view"
            base_dir = tmp_path / "base"
            out_dir = tmp_path / "out"
            video = tmp_path / "fake.mp4"
            video.write_bytes(b"not-a-real-video")
            ckpt = tmp_path / "ckpt.bin"
            ckpt.write_bytes(b"fake-checkpoint-bytes")
            sn_root = tmp_path / "sn-reid"
            sn_root.mkdir()
            (sn_root / "torchreid").mkdir()
            subprocess.run(["git", "init"], cwd=sn_root, check=True, capture_output=True)
            (sn_root / "torchreid" / "__init__.py").write_text("#\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=sn_root, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
                cwd=sn_root,
                check=True,
                capture_output=True,
            )

            _build_segment_view(view_dir, with_conflict_track=True)
            vectors = {7: _unit(1), 10: _unit(2), 12: _unit(4), 268: _unit(3)}
            _build_baseline(base_dir, vectors=vectors, conflict_pairs={(10, 12)})
            emb_sum = json.loads(
                (base_dir / "embeddings" / "embedding_summary.json").read_text()
            )
            emb_sum["checkpoint_sha256"] = _sha(ckpt.read_bytes())
            emb_sum["sn_reid_commit"] = subprocess.check_output(
                ["git", "-C", str(sn_root), "rev-parse", "HEAD"], text=True
            ).strip()
            emb_sum["checkpoint_path"] = str(ckpt)
            _write_json(base_dir / "embeddings" / "embedding_summary.json", emb_sum)

            decisions_path = tmp_path / "decisions.yaml"
            decisions_path.write_text(yaml.safe_dump(_mini_decisions()), encoding="utf-8")
            frames = [
                np.full((160, 160, 3), fill_value=(i * 3) % 200, dtype=np.uint8)
                for i in range(60)
            ]

            result = run_segmented_reid_regression(
                video=video,
                segment_view_dir=view_dir,
                baseline_run_dir=base_dir,
                segmentation_policy=POLICY,
                segment_decisions=decisions_path,
                crop_config=CROP_CFG,
                reid_config=REID_CFG,
                regression_config=REG_CFG,
                sn_reid_root=sn_root,
                checkpoint=ckpt,
                output_dir=out_dir,
                overwrite=False,
                open_capture=lambda _path: FakeCapture(frames),
                video_size=(160, 160),
                model_builder=lambda model_name=None: DeterministicModel(),
                weight_loader=lambda model, path: None,
            )
            self.assertEqual(result["status"], "ok")
            summary = json.loads((out_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
            # unaffected pairs among reused {10, 12, 268}: (10,12) conflict,
            # (10,268) and (12,268) rank-eligible
            self.assertEqual(summary["unaffected_baseline_pair_count"], 3)
            self.assertEqual(summary["unaffected_rank_eligible_pair_count"], 2)
            self.assertEqual(summary["unaffected_exact_frame_conflict_pair_count"], 1)
            self.assertEqual(summary["unaffected_pair_similarity_match_count"], 3)
            self.assertEqual(summary["unaffected_pair_similarity_mismatch_count"], 0)
            self.assertEqual(summary["unaffected_missing_nonconflict_candidate_count"], 0)
            self.assertEqual(summary["similarity_threshold"], None)
            self.assertEqual(summary["automatic_link_count"], 0)
            self.assertEqual(summary["automatic_reject_count"], 0)
            self.assertEqual(summary["component_building_count"], 0)

            candidates = [
                json.loads(l)
                for l in (out_dir / SEGMENT_CANDIDATES_NAME).read_text().splitlines()
                if l.strip()
            ]
            conflict_cand = next(
                c
                for c in candidates
                if {c["segment_id_a"], c["segment_id_b"]} == {"raw_10_full", "raw_12_full"}
            )
            self.assertTrue(conflict_cand["exact_same_frame_overlap"])
            self.assertIsNone(conflict_cand["cosine_similarity"])
            self.assertIsNone(conflict_cand["rank"])
            ranked_pairs = {
                (c["segment_id_a"], c["segment_id_b"])
                for c in candidates
                if c["rank"] is not None
            }
            self.assertNotIn(("raw_10_full", "raw_12_full"), ranked_pairs)
            self.assertEqual(summary["ranked_candidate_count"], len(ranked_pairs))

            deltas = [
                json.loads(l)
                for l in (out_dir / PAIR_DELTAS_NAME).read_text().splitlines()
                if l.strip()
            ]
            conflict_deltas = [
                d for d in deltas if d["delta_kind"] == "unaffected_exact_frame_conflict"
            ]
            self.assertEqual(len(conflict_deltas), 1)
            row = conflict_deltas[0]
            self.assertTrue(row["hard_rejected"])
            self.assertIsNone(row["segmented_rank"])
            self.assertFalse(row["segmented_ranked_candidate_available"])
            self.assertTrue(row["similarity_match"])
            self.assertIsNone(row["automatic_link_decision"])
            self.assertIsNone(row["component_assignment"])
            self.assertFalse(row["accuracy_claimed"])
            # normal unaffected pairs unchanged
            normal = [d for d in deltas if d["delta_kind"] == "unaffected_reused_pair"]
            self.assertEqual(len(normal), 2)
            self.assertTrue(all(d["similarity_match"] for d in normal))
            # component audit policy unchanged
            audit = summary["existing_component_audit"]
            self.assertEqual(audit["raw_component"], [231, 635])
            self.assertFalse(audit["raw_231_s01_inherits_component"])
            self.assertFalse(audit["raw_231_s02_automatically_links_to_635"])


class ForbiddenImportTests(unittest.TestCase):
    def test_help_args_and_static(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_segmented_reid_regression.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        help_text = proc.stdout
        for token in (
            "--video",
            "--segment-view-dir",
            "--baseline-run-dir",
            "--segmentation-policy",
            "--segment-decisions",
            "--crop-config",
            "--reid-config",
            "--regression-config",
            "--sn-reid-root",
            "--checkpoint",
            "--output-dir",
            "--overwrite",
        ):
            self.assertIn(token, help_text)
        for bad in ("--team-count", "--global-id-map", "--similarity-threshold", "--auto-merge"):
            self.assertNotIn(bad, help_text)

        src = (_SRC_DIR / "football_analytics" / "reid" / "segment_regression.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("urllib.request", src)
        self.assertNotIn("requests.", src)
        self.assertNotIn("global_id_map", src)
        self.assertNotIn("sklearn", src)
        self.assertNotIn("allow_pickle=True", src)


if __name__ == "__main__":
    unittest.main()
