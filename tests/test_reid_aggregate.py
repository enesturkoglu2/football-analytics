"""Unit tests for ReID track embedding aggregation (no real smoke outputs)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.aggregate import (
    AGGREGATION_NAME,
    EMBEDDING_DIM,
    TRACK_JSONL_NAME,
    TRACK_NPZ_NAME,
    AggregateError,
    aggregate_track_embeddings,
    build_track_observation_stats,
    load_crop_embedding_bundle,
    run_aggregate_reid_tracks,
)
from football_analytics.reid.embedding import l2_normalize_rows
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


def _write_crop_bundle(
    root: Path,
    *,
    vectors: np.ndarray,
    crop_ids: list[str],
    track_ids: list[int],
    frame_indices: list[int],
) -> tuple[Path, Path]:
    n = vectors.shape[0]
    npz = root / "crop_embeddings.npz"
    np.savez(
        npz,
        vectors=vectors.astype(np.float32),
        crop_ids=np.asarray(crop_ids, dtype=np.str_),
        track_ids=np.asarray(track_ids, dtype=np.int64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
    )
    index_rows = []
    for i in range(n):
        index_rows.append(
            {
                "crop_id": crop_ids[i],
                "track_id": track_ids[i],
                "frame_index": frame_indices[i],
                "embedding_row": i,
                "embedding_shape": [EMBEDDING_DIM],
                "embedding_dtype": "float32",
                "l2_norm": float(np.linalg.norm(vectors[i])),
                "model_name": "osnet_x1_0",
                "checkpoint_sha256": "a" * 64,
                "preprocessing": {},
                "schema_version": "reid_crop_embedding_index_v1",
            }
        )
    index = root / "crop_embeddings_index.jsonl"
    write_manifest_jsonl(index, index_rows)
    return npz, index


class AggregateMathTests(unittest.TestCase):
    def test_two_crop_l2_mean(self) -> None:
        v1 = _unit(np.ones(EMBEDDING_DIM, dtype=np.float32))
        v2 = _unit(np.concatenate([np.ones(256), -np.ones(256)]).astype(np.float32))
        vectors = np.stack([v1, v2], axis=0)
        stats = {
            7: {
                "observation_count": 2,
                "first_frame": 1,
                "last_frame": 2,
                "observed_frames": {1, 2},
                "observed_frame_count": 2,
            }
        }
        result = aggregate_track_embeddings(
            vectors=vectors,
            crop_ids=["c1", "c2"],
            track_ids=[7, 7],
            track_stats=stats,
        )
        expected = _unit(((v1 + v2) / 2.0).astype(np.float32))
        self.assertEqual(result["vectors"].shape, (1, EMBEDDING_DIM))
        self.assertTrue(np.allclose(result["vectors"][0], expected, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.norm(result["vectors"][0])), 1.0, places=5)
        self.assertEqual(result["rows"][0]["aggregation"], AGGREGATION_NAME)
        self.assertEqual(result["rows"][0]["crop_ids"], ["c1", "c2"])

    def test_single_crop_track(self) -> None:
        v = _unit(np.arange(EMBEDDING_DIM, dtype=np.float32) + 1.0)
        stats = {
            3: {
                "observation_count": 5,
                "first_frame": 10,
                "last_frame": 20,
                "observed_frames": set(range(10, 21)),
                "observed_frame_count": 11,
            }
        }
        result = aggregate_track_embeddings(
            vectors=v[None, :],
            crop_ids=["only"],
            track_ids=[3],
            track_stats=stats,
        )
        self.assertTrue(np.allclose(result["vectors"][0], v, atol=1e-6))
        self.assertEqual(result["rows"][0]["crop_count"], 1)

    def test_track_order_deterministic(self) -> None:
        # Input crop order has track 9 before track 2; output must sort by track_id.
        v2 = _unit(np.full(EMBEDDING_DIM, 2.0, dtype=np.float32))
        v9 = _unit(np.full(EMBEDDING_DIM, 9.0, dtype=np.float32))
        vectors = np.stack([v9, v2], axis=0)
        stats = {
            2: {
                "observation_count": 1,
                "first_frame": 0,
                "last_frame": 0,
                "observed_frames": {0},
                "observed_frame_count": 1,
            },
            9: {
                "observation_count": 1,
                "first_frame": 1,
                "last_frame": 1,
                "observed_frames": {1},
                "observed_frame_count": 1,
            },
        }
        result = aggregate_track_embeddings(
            vectors=vectors,
            crop_ids=["c9", "c2"],
            track_ids=[9, 2],
            track_stats=stats,
        )
        self.assertEqual([r["track_id"] for r in result["rows"]], [2, 9])
        self.assertEqual(result["rows"][0]["crop_ids"], ["c2"])
        self.assertEqual(result["rows"][1]["crop_ids"], ["c9"])

    def test_crop_order_preserved_within_track(self) -> None:
        a = _unit(np.ones(EMBEDDING_DIM, dtype=np.float32))
        b = _unit(np.full(EMBEDDING_DIM, 2.0, dtype=np.float32))
        c = _unit(np.full(EMBEDDING_DIM, 3.0, dtype=np.float32))
        vectors = np.stack([a, b, c], axis=0)
        stats = {
            1: {
                "observation_count": 3,
                "first_frame": 0,
                "last_frame": 2,
                "observed_frames": {0, 1, 2},
                "observed_frame_count": 3,
            }
        }
        result = aggregate_track_embeddings(
            vectors=vectors,
            crop_ids=["first", "second", "third"],
            track_ids=[1, 1, 1],
            track_stats=stats,
        )
        self.assertEqual(result["rows"][0]["crop_ids"], ["first", "second", "third"])

    def test_wrong_vector_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = np.ones((2, 128), dtype=np.float32)
            npz, index = _write_crop_bundle(
                root,
                vectors=bad,
                crop_ids=["a", "b"],
                track_ids=[1, 1],
                frame_indices=[0, 1],
            )
            with self.assertRaises(AggregateError):
                load_crop_embedding_bundle(
                    crop_embeddings=npz, crop_embeddings_index=index
                )

    def test_nan_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
            clean = clean / np.linalg.norm(clean)
            npz, index = _write_crop_bundle(
                root,
                vectors=clean.astype(np.float32),
                crop_ids=["a"],
                track_ids=[1],
                frame_indices=[0],
            )
            data = np.load(npz, allow_pickle=False)
            vectors = np.array(data["vectors"], copy=True)
            vectors[0, 0] = np.nan
            np.savez(
                npz,
                vectors=vectors,
                crop_ids=data["crop_ids"],
                track_ids=data["track_ids"],
                frame_indices=data["frame_indices"],
            )
            with self.assertRaises(AggregateError):
                load_crop_embedding_bundle(
                    crop_embeddings=npz, crop_embeddings_index=index
                )

    def test_zero_norm_mean_rejected(self) -> None:
        # Two opposite unit vectors average to ~0.
        v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v2[0] = -1.0
        stats = {
            1: {
                "observation_count": 2,
                "first_frame": 0,
                "last_frame": 1,
                "observed_frames": {0, 1},
                "observed_frame_count": 2,
            }
        }
        with self.assertRaises(AggregateError):
            aggregate_track_embeddings(
                vectors=np.stack([v1, v2]),
                crop_ids=["a", "b"],
                track_ids=[1, 1],
                track_stats=stats,
            )

    def test_npz_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = l2_normalize_rows(np.ones((2, EMBEDDING_DIM), dtype=np.float32))
            npz, index = _write_crop_bundle(
                root,
                vectors=vectors,
                crop_ids=["a", "b"],
                track_ids=[1, 2],
                frame_indices=[0, 1],
            )
            # Corrupt index crop_id.
            rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
            rows[1]["crop_id"] = "CHANGED"
            write_manifest_jsonl(index, rows)
            with self.assertRaises(AggregateError):
                load_crop_embedding_bundle(
                    crop_embeddings=npz, crop_embeddings_index=index
                )

    def test_duplicate_crop_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors = l2_normalize_rows(np.ones((2, EMBEDDING_DIM), dtype=np.float32))
            npz, index = _write_crop_bundle(
                root,
                vectors=vectors,
                crop_ids=["same", "same"],
                track_ids=[1, 1],
                frame_indices=[0, 1],
            )
            with self.assertRaises(AggregateError):
                load_crop_embedding_bundle(
                    crop_embeddings=npz, crop_embeddings_index=index
                )

    def test_missing_track_metadata(self) -> None:
        vectors = l2_normalize_rows(np.ones((1, EMBEDDING_DIM), dtype=np.float32))
        with self.assertRaises(AggregateError):
            aggregate_track_embeddings(
                vectors=vectors,
                crop_ids=["a"],
                track_ids=[99],
                track_stats={},
            )

    def test_object_dtype_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "crop_embeddings.npz"
            # object arrays require pickle; loading with allow_pickle=False fails.
            np.savez(
                npz,
                vectors=np.ones((1, EMBEDDING_DIM), dtype=np.float32),
                crop_ids=np.array(["x"], dtype=object),
                track_ids=np.asarray([1], dtype=np.int64),
                frame_indices=np.asarray([0], dtype=np.int64),
            )
            index = root / "crop_embeddings_index.jsonl"
            write_manifest_jsonl(
                index,
                [
                    {
                        "crop_id": "x",
                        "track_id": 1,
                        "frame_index": 0,
                        "embedding_row": 0,
                    }
                ],
            )
            with self.assertRaises(AggregateError):
                load_crop_embedding_bundle(
                    crop_embeddings=npz, crop_embeddings_index=index
                )


class AggregatePipelineTests(unittest.TestCase):
    def _make_inputs(self, tmp: Path) -> tuple[Path, Path, Path]:
        v1 = _unit(np.ones(EMBEDDING_DIM, dtype=np.float32))
        v2 = _unit(np.full(EMBEDDING_DIM, 2.0, dtype=np.float32))
        v3 = _unit(np.arange(EMBEDDING_DIM, dtype=np.float32) + 1)
        vectors = np.stack([v1, v2, v3], axis=0)
        npz, index = _write_crop_bundle(
            tmp,
            vectors=vectors,
            crop_ids=["t1_a", "t1_b", "t2_a"],
            track_ids=[1, 1, 2],
            frame_indices=[10, 20, 30],
        )
        tracks = tmp / "tracks.jsonl"
        _write_tracks(
            tracks,
            [
                (1, 10),
                (1, 15),
                (1, 20),
                (2, 30),
                (2, 31),
            ],
        )
        return npz, index, tracks

    def test_pipeline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._make_inputs(tmp_path)
            out = tmp_path / "agg_out"
            result = run_aggregate_reid_tracks(
                crop_embeddings=npz,
                crop_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
            )
            self.assertEqual(result["crop_embedding_count"], 3)
            self.assertEqual(result["track_embedding_count"], 2)
            self.assertEqual(result["tracks_with_single_crop"], 1)
            self.assertEqual(result["tracks_with_multiple_crops"], 1)

            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(
                names,
                [
                    "aggregation_summary.json",
                    "track_embeddings.jsonl",
                    "track_embeddings.npz",
                ],
            )
            data = np.load(out / TRACK_NPZ_NAME, allow_pickle=False)
            self.assertEqual(data["vectors"].shape, (2, EMBEDDING_DIM))
            self.assertEqual(data["vectors"].dtype, np.float32)
            self.assertNotEqual(data["vectors"].dtype, object)
            self.assertEqual(list(data["track_ids"]), [1, 2])

            rows = [
                json.loads(l)
                for l in (out / TRACK_JSONL_NAME).read_text().splitlines()
                if l.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["embedding_row"], 0)
            self.assertEqual(rows[1]["embedding_row"], 1)
            self.assertEqual(rows[0]["crop_ids"], ["t1_a", "t1_b"])
            self.assertEqual(rows[0]["schema_version"], "reid_track_embedding_v1")

            summary = json.loads((out / "aggregation_summary.json").read_text())
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["aggregation"], "l2_mean")
            self.assertEqual(summary["schema_version"], "reid_aggregation_summary_v1")
            self.assertAlmostEqual(summary["l2_norm_min"], 1.0, places=5)

    def test_collision_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._make_inputs(tmp_path)
            out = tmp_path / "agg_out"
            run_aggregate_reid_tracks(
                crop_embeddings=npz,
                crop_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
            )
            with self.assertRaises(ReIDWritersError):
                run_aggregate_reid_tracks(
                    crop_embeddings=npz,
                    crop_embeddings_index=index,
                    tracks=tracks,
                    output_dir=out,
                    overwrite=False,
                )
            second = run_aggregate_reid_tracks(
                crop_embeddings=npz,
                crop_embeddings_index=index,
                tracks=tracks,
                output_dir=out,
                overwrite=True,
            )
            self.assertEqual(second["status"], "ok")
            self.assertEqual(list(tmp_path.glob("_tmp_reid_aggregate_*")), [])
            self.assertEqual(list(tmp_path.glob("_backup_reid_aggregate_*")), [])

    def test_temp_cleanup_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz, index, tracks = self._make_inputs(tmp_path)
            out = tmp_path / "agg_out"
            # Break tracks so aggregation fails after inputs load... actually
            # missing track fails before temp. Force failure during write by
            # using a track set that loads then patch finalize.
            from unittest import mock

            with mock.patch(
                "football_analytics.reid.aggregate.write_aggregation_artifacts",
                side_effect=AggregateError("boom"),
            ):
                with self.assertRaises(AggregateError):
                    run_aggregate_reid_tracks(
                        crop_embeddings=npz,
                        crop_embeddings_index=index,
                        tracks=tracks,
                        output_dir=out,
                    )
            self.assertFalse(out.exists())
            self.assertEqual(list(tmp_path.glob("_tmp_reid_aggregate_*")), [])

    def test_observation_stats(self) -> None:
        stats = build_track_observation_stats(
            [
                {"track_id": 5, "frame_index": 3},
                {"track_id": 5, "frame_index": 1},
                {"track_id": 5, "frame_index": 3},
                {"track_id": 8, "frame_index": 9},
            ]
        )
        self.assertEqual(stats[5]["observation_count"], 3)
        self.assertEqual(stats[5]["first_frame"], 1)
        self.assertEqual(stats[5]["last_frame"], 3)
        self.assertEqual(stats[5]["observed_frame_count"], 2)
        self.assertEqual(stats[8]["observed_frames"], {9})


if __name__ == "__main__":
    unittest.main()
