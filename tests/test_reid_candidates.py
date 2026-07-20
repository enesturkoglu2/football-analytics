"""Unit tests for ReID candidate pair generation (no real smoke outputs)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.aggregate import (
    EMBEDDING_DIM,
    run_aggregate_reid_tracks,
)
from football_analytics.reid.candidates import (
    DECISION_ELIGIBLE,
    DECISION_REJECTED,
    CandidateError,
    build_candidate_pairs,
    cosine_similarity,
    run_build_reid_candidates,
    summarize_candidate_pairs,
    temporal_gap_frames,
)
from football_analytics.reid.writers import ReIDWritersError, write_manifest_jsonl


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).astype(np.float32)


def _write_tracks(path: Path, observations: list[tuple[int, int]]) -> None:
    rows = []
    for track_id, frame_index in observations:
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / 25.0,
                "track_id": track_id,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox_xyxy": [0.0, 0.0, 10.0, 20.0],
            }
        )
    write_manifest_jsonl(path, rows)


def _write_track_bundle(
    root: Path,
    *,
    vectors: np.ndarray,
    track_ids: list[int],
    first_frames: list[int],
    last_frames: list[int],
    observation_counts: list[int],
    observed_frame_counts: list[int],
    crop_counts: list[int] | None = None,
) -> tuple[Path, Path]:
    t = vectors.shape[0]
    if crop_counts is None:
        crop_counts = [1] * t
    npz = root / "track_embeddings.npz"
    np.savez(
        npz,
        vectors=vectors.astype(np.float32),
        track_ids=np.asarray(track_ids, dtype=np.int64),
        crop_counts=np.asarray(crop_counts, dtype=np.int64),
        first_frames=np.asarray(first_frames, dtype=np.int64),
        last_frames=np.asarray(last_frames, dtype=np.int64),
        observation_counts=np.asarray(observation_counts, dtype=np.int64),
    )
    rows = []
    for i in range(t):
        rows.append(
            {
                "track_id": track_ids[i],
                "crop_ids": [f"c{track_ids[i]}"],
                "crop_count": crop_counts[i],
                "embedding_row": i,
                "aggregation": "l2_mean",
                "embedding_shape": [EMBEDDING_DIM],
                "embedding_dtype": "float32",
                "l2_norm": float(np.linalg.norm(vectors[i])),
                "observation_count": observation_counts[i],
                "first_frame": first_frames[i],
                "last_frame": last_frames[i],
                "observed_frame_count": observed_frame_counts[i],
                "schema_version": "reid_track_embedding_v1",
            }
        )
    index = root / "track_embeddings.jsonl"
    write_manifest_jsonl(index, rows)
    return npz, index


class CosineAndGapTests(unittest.TestCase):
    def test_cosine_same_vector(self) -> None:
        v = _unit(np.arange(EMBEDDING_DIM, dtype=np.float32) + 1)
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_cosine_opposite(self) -> None:
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[0] = 1.0
        self.assertAlmostEqual(cosine_similarity(v, -v), -1.0, places=6)

    def test_temporal_gap_disjoint(self) -> None:
        self.assertEqual(
            temporal_gap_frames(first_a=0, last_a=10, first_b=15, last_b=20), 4
        )

    def test_temporal_gap_adjacent(self) -> None:
        self.assertEqual(
            temporal_gap_frames(first_a=0, last_a=10, first_b=11, last_b=20), 0
        )

    def test_temporal_gap_overlap(self) -> None:
        self.assertEqual(
            temporal_gap_frames(first_a=0, last_a=10, first_b=5, last_b=20), 0
        )


class CandidateLogicTests(unittest.TestCase):
    def test_pair_order_and_counts(self) -> None:
        vectors = np.stack(
            [
                _unit(np.full(EMBEDDING_DIM, 1.0)),
                _unit(np.full(EMBEDDING_DIM, 2.0)),
                _unit(np.full(EMBEDDING_DIM, 3.0)),
            ]
        )
        stats = {
            1: {
                "observation_count": 1,
                "first_frame": 0,
                "last_frame": 0,
                "observed_frames": {0},
                "observed_frame_count": 1,
            },
            2: {
                "observation_count": 1,
                "first_frame": 10,
                "last_frame": 10,
                "observed_frames": {10},
                "observed_frame_count": 1,
            },
            3: {
                "observation_count": 1,
                "first_frame": 20,
                "last_frame": 20,
                "observed_frames": {20},
                "observed_frame_count": 1,
            },
        }
        pairs = build_candidate_pairs(
            vectors=vectors, track_ids=[1, 2, 3], track_stats=stats
        )
        self.assertEqual(len(pairs), 3)
        self.assertEqual(
            [(p["track_id_a"], p["track_id_b"]) for p in pairs],
            [(1, 2), (1, 3), (2, 3)],
        )

    def test_t1_zero_pairs(self) -> None:
        vectors = _unit(np.ones(EMBEDDING_DIM))[None, :]
        stats = {
            1: {
                "observation_count": 1,
                "first_frame": 0,
                "last_frame": 0,
                "observed_frames": {0},
                "observed_frame_count": 1,
            }
        }
        pairs = build_candidate_pairs(
            vectors=vectors, track_ids=[1], track_stats=stats
        )
        self.assertEqual(pairs, [])

    def test_exact_frame_hard_reject(self) -> None:
        vectors = np.stack(
            [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
        )
        stats = {
            1: {
                "observation_count": 2,
                "first_frame": 5,
                "last_frame": 10,
                "observed_frames": {5, 10},
                "observed_frame_count": 2,
            },
            2: {
                "observation_count": 2,
                "first_frame": 10,
                "last_frame": 15,
                "observed_frames": {10, 15},
                "observed_frame_count": 2,
            },
        }
        pairs = build_candidate_pairs(
            vectors=vectors, track_ids=[1, 2], track_stats=stats
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["exact_frame_conflict"])
        self.assertEqual(pairs[0]["exact_frame_overlap_count"], 1)
        self.assertEqual(pairs[0]["decision"], DECISION_REJECTED)
        self.assertEqual(pairs[0]["decision_reason"], "exact_frame_conflict")

    def test_span_overlap_without_exact_is_eligible(self) -> None:
        vectors = np.stack(
            [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
        )
        # Spans overlap [0..10] and [5..15], but exact frames {0,10} vs {5,15} disjoint.
        stats = {
            1: {
                "observation_count": 2,
                "first_frame": 0,
                "last_frame": 10,
                "observed_frames": {0, 10},
                "observed_frame_count": 2,
            },
            2: {
                "observation_count": 2,
                "first_frame": 5,
                "last_frame": 15,
                "observed_frames": {5, 15},
                "observed_frame_count": 2,
            },
        }
        pairs = build_candidate_pairs(
            vectors=vectors, track_ids=[1, 2], track_stats=stats
        )
        self.assertTrue(pairs[0]["span_interval_overlap"])
        self.assertFalse(pairs[0]["exact_frame_conflict"])
        self.assertEqual(pairs[0]["decision"], DECISION_ELIGIBLE)
        self.assertEqual(pairs[0]["decision_reason"], "similarity_threshold_pending")

    def test_no_accepted_link_and_threshold_null(self) -> None:
        summary = summarize_candidate_pairs(
            [],
            track_count=1,
            elapsed_sec=0.01,
            track_npz_path=Path("x.npz"),
            track_index_path=Path("x.jsonl"),
            tracks_path=Path("t.jsonl"),
        )
        self.assertIsNone(summary["similarity_threshold"])
        self.assertFalse(summary["automatic_linking_performed"])
        self.assertIsNone(summary["cosine_min"])
        self.assertIsNone(summary["eligible_cosine_min"])

    def test_duplicate_track_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = np.stack(
                [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
            )
            npz, index = _write_track_bundle(
                root,
                vectors=vectors,
                track_ids=[1, 1],
                first_frames=[0, 1],
                last_frames=[0, 1],
                observation_counts=[1, 1],
                observed_frame_counts=[1, 1],
            )
            tracks = root / "tracks.jsonl"
            _write_tracks(tracks, [(1, 0), (1, 1)])
            with self.assertRaises(CandidateError):
                run_build_reid_candidates(
                    track_embeddings=npz,
                    track_embeddings_index=index,
                    tracks=tracks,
                    output_dir=root / "out",
                )

    def test_nan_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.stack(
                [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
            )
            npz, index = _write_track_bundle(
                root,
                vectors=clean,
                track_ids=[1, 2],
                first_frames=[0, 10],
                last_frames=[0, 10],
                observation_counts=[1, 1],
                observed_frame_counts=[1, 1],
            )
            # Inject NaN into NPZ after index was written with finite norms.
            data = np.load(npz, allow_pickle=False)
            vectors = np.array(data["vectors"], copy=True)
            vectors[0, 0] = np.nan
            np.savez(
                npz,
                vectors=vectors,
                track_ids=data["track_ids"],
                crop_counts=data["crop_counts"],
                first_frames=data["first_frames"],
                last_frames=data["last_frames"],
                observation_counts=data["observation_counts"],
            )
            tracks = root / "tracks.jsonl"
            _write_tracks(tracks, [(1, 0), (2, 10)])
            with self.assertRaises(CandidateError):
                run_build_reid_candidates(
                    track_embeddings=npz,
                    track_embeddings_index=index,
                    tracks=tracks,
                    output_dir=root / "out",
                )

    def test_npz_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = np.stack(
                [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
            )
            npz, index = _write_track_bundle(
                root,
                vectors=vectors,
                track_ids=[1, 2],
                first_frames=[0, 10],
                last_frames=[0, 10],
                observation_counts=[1, 1],
                observed_frame_counts=[1, 1],
            )
            rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
            rows[1]["track_id"] = 99
            write_manifest_jsonl(index, rows)
            tracks = root / "tracks.jsonl"
            _write_tracks(tracks, [(1, 0), (2, 10)])
            with self.assertRaises(CandidateError):
                run_build_reid_candidates(
                    track_embeddings=npz,
                    track_embeddings_index=index,
                    tracks=tracks,
                    output_dir=root / "out",
                )


class CandidatePipelineTests(unittest.TestCase):
    def _build_from_aggregate(self, tmp: Path) -> tuple[Path, Path, Path]:
        # Create via aggregate pipeline for realistic NPZ/JSONL.
        from football_analytics.reid.embedding import l2_normalize_rows

        vectors = l2_normalize_rows(
            np.stack(
                [
                    np.ones(EMBEDDING_DIM, dtype=np.float32),
                    np.full(EMBEDDING_DIM, 2.0, dtype=np.float32),
                    np.arange(EMBEDDING_DIM, dtype=np.float32) + 1,
                ]
            )
        )
        crop_npz = tmp / "crop_embeddings.npz"
        np.savez(
            crop_npz,
            vectors=vectors,
            crop_ids=np.asarray(["a", "b", "c"], dtype=np.str_),
            track_ids=np.asarray([1, 2, 3], dtype=np.int64),
            frame_indices=np.asarray([0, 10, 20], dtype=np.int64),
        )
        crop_index = tmp / "crop_embeddings_index.jsonl"
        write_manifest_jsonl(
            crop_index,
            [
                {
                    "crop_id": cid,
                    "track_id": tid,
                    "frame_index": fr,
                    "embedding_row": i,
                    "embedding_shape": [EMBEDDING_DIM],
                    "embedding_dtype": "float32",
                    "l2_norm": 1.0,
                    "model_name": "osnet_x1_0",
                    "checkpoint_sha256": "a" * 64,
                    "preprocessing": {},
                    "schema_version": "reid_crop_embedding_index_v1",
                }
                for i, (cid, tid, fr) in enumerate(
                    [("a", 1, 0), ("b", 2, 10), ("c", 3, 20)]
                )
            ],
        )
        tracks = tmp / "tracks.jsonl"
        # 1 and 2: span-only style disjoint exact frames; 1 and 3 disjoint; 2 and 3 disjoint.
        # Also add exact conflict between none of them for baseline eligible pairs.
        _write_tracks(
            tracks,
            [
                (1, 0),
                (1, 5),
                (2, 10),
                (2, 15),
                (3, 20),
                (3, 25),
            ],
        )
        agg_out = tmp / "agg"
        run_aggregate_reid_tracks(
            crop_embeddings=crop_npz,
            crop_embeddings_index=crop_index,
            tracks=tracks,
            output_dir=agg_out,
        )
        return (
            agg_out / "track_embeddings.npz",
            agg_out / "track_embeddings.jsonl",
            tracks,
        )

    def test_pipeline_summary_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._build_from_aggregate(tmp_path)
            out = tmp_path / "cand_out"
            result = run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
            )
            self.assertEqual(result["track_count"], 3)
            self.assertEqual(result["total_pairs"], 3)
            self.assertEqual(result["exact_frame_conflict_pairs"], 0)
            self.assertEqual(result["eligible_unthresholded_pairs"], 3)
            self.assertIsNone(result["similarity_threshold"])
            self.assertFalse(result["automatic_linking_performed"])

            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(names, ["candidate_pairs.jsonl", "candidate_summary.json"])
            pairs = [
                json.loads(l)
                for l in (out / "candidate_pairs.jsonl").read_text().splitlines()
                if l.strip()
            ]
            self.assertTrue(all(p["decision"] != "accepted_link" for p in pairs))
            self.assertTrue(
                all(p["schema_version"] == "reid_candidate_pair_v1" for p in pairs)
            )
            summary = json.loads((out / "candidate_summary.json").read_text())
            self.assertEqual(summary["schema_version"], "reid_candidate_summary_v1")
            self.assertIsNone(summary["similarity_threshold"])

    def test_exact_conflict_in_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = np.stack(
                [_unit(np.ones(EMBEDDING_DIM)), _unit(np.full(EMBEDDING_DIM, 2.0))]
            )
            npz, index = _write_track_bundle(
                root,
                vectors=vectors,
                track_ids=[1, 2],
                first_frames=[0, 10],
                last_frames=[10, 15],
                observation_counts=[2, 2],
                observed_frame_counts=[2, 2],
            )
            tracks = root / "tracks.jsonl"
            _write_tracks(tracks, [(1, 0), (1, 10), (2, 10), (2, 15)])
            out = root / "out"
            result = run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
            )
            self.assertEqual(result["exact_frame_conflict_pairs"], 1)
            self.assertEqual(result["eligible_unthresholded_pairs"], 0)
            self.assertIsNone(result["summary"]["eligible_cosine_min"])
            self.assertTrue(result["summary"]["span_overlap_pairs"] >= 1)

    def test_deterministic_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._build_from_aggregate(tmp_path)
            a = run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=tmp_path / "out_a",
            )
            b = run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=tmp_path / "out_b",
            )
            self.assertEqual(a["pairs"], b["pairs"])

    def test_collision_overwrite_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._build_from_aggregate(tmp_path)
            out = tmp_path / "cand"
            run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
            )
            with self.assertRaises(ReIDWritersError):
                run_build_reid_candidates(
                    track_embeddings=npz,
                    track_embeddings_index=index,
                    tracks=tracks,
                    output_dir=out,
                )
            run_build_reid_candidates(
                track_embeddings=npz,
                track_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(list(tmp_path.glob("_tmp_reid_candidates_*")), [])
            self.assertEqual(list(tmp_path.glob("_backup_reid_candidates_*")), [])

            with mock.patch(
                "football_analytics.reid.candidates.write_candidate_artifacts",
                side_effect=CandidateError("boom"),
            ):
                with self.assertRaises(CandidateError):
                    run_build_reid_candidates(
                        track_embeddings=npz,
                        track_embeddings_index=index,
                        tracks=tracks,
                        output_dir=tmp_path / "cand_fail",
                    )
            self.assertEqual(list(tmp_path.glob("_tmp_reid_candidates_*")), [])


if __name__ == "__main__":
    unittest.main()
