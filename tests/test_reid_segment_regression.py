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


def _build_segment_view(root: Path) -> None:
    """Build a tiny segment view: split 7, control 268, embedded 10, non-embedded 99."""
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
    # sort by raw_track_id then manual first
    segs.sort(key=lambda r: (r["raw_track_id"], 0 if r["segment_kind"] == "manual_split_segment" else 1, r["segment_index"] or 0))

    assigned_specs = [
        ("raw_7_s01", 7, [0, 1, 2, 3, 4]),
        ("raw_7_s02", 7, [6, 8, 10, 12]),
        ("raw_10_full", 10, [20, 24, 28]),
        ("raw_99_full", 99, [30, 32]),
        ("raw_268_full", 268, [40, 44, 49]),
    ]
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


def _build_baseline(root: Path, *, vectors: dict[int, np.ndarray]) -> None:
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
            pairs.append(
                {
                    "track_id_a": a,
                    "track_id_b": b,
                    "cosine_similarity": sim,
                    "temporal_gap_frames": 0,
                    "exact_frame_overlap_count": 0,
                    "exact_frame_conflict": False,
                    "span_interval_overlap": False,
                    "decision": "eligible_unthresholded",
                    "decision_reason": "similarity_threshold_pending",
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
