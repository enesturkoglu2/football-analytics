"""Track-level ReID embedding aggregation (Stage 4B-4).

Baseline: per-track arithmetic mean of L2-normalized crop embeddings,
then L2-normalize the mean again (``l2_mean``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from football_analytics.reid.embedding import EMBEDDING_DIM, l2_normalize_rows
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

AGGREGATION_NAME = "l2_mean"
TRACK_EMBEDDING_SCHEMA_VERSION = "reid_track_embedding_v1"
AGGREGATION_SUMMARY_SCHEMA_VERSION = "reid_aggregation_summary_v1"

TRACK_NPZ_NAME = "track_embeddings.npz"
TRACK_JSONL_NAME = "track_embeddings.jsonl"
AGGREGATION_SUMMARY_NAME = "aggregation_summary.json"

_NORM_ATOL = 1e-4


class AggregateError(RuntimeError):
    """Raised when track embedding aggregation fails."""


def _reject_non_finite_json(value: str) -> None:
    raise AggregateError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AggregateError(f"{field} must be finite, got {value!r}")
    return number


def load_tracks_jsonl(tracks_path: str | Path) -> list[dict[str, Any]]:
    path = Path(tracks_path).expanduser().resolve()
    if not path.is_file():
        raise AggregateError(f"tracks JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text, parse_constant=_reject_non_finite_json)
            except AggregateError:
                raise
            except json.JSONDecodeError as exc:
                raise AggregateError(
                    f"invalid JSON on tracks line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise AggregateError(f"tracks line {line_no} must be a JSON object")
            if "track_id" not in obj or "frame_index" not in obj:
                raise AggregateError(
                    f"tracks line {line_no} missing track_id or frame_index"
                )
            rows.append(obj)
    if not rows:
        raise AggregateError("tracks JSONL is empty")
    return rows


def build_track_observation_stats(
    observations: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Build per-track observation metadata from tracks.jsonl rows."""
    frames_by_track: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)

    for row in observations:
        track_id = row.get("track_id")
        frame_index = row.get("frame_index")
        if track_id is None:
            continue
        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id <= 0:
            raise AggregateError(f"invalid track_id in tracks.jsonl: {track_id!r}")
        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index < 0
        ):
            raise AggregateError(
                f"invalid frame_index for track {track_id}: {frame_index!r}"
            )
        frames_by_track[track_id].add(frame_index)
        counts[track_id] += 1

    stats: dict[int, dict[str, Any]] = {}
    for track_id, frames in frames_by_track.items():
        ordered = sorted(frames)
        stats[track_id] = {
            "track_id": track_id,
            "observation_count": int(counts[track_id]),
            "first_frame": int(ordered[0]),
            "last_frame": int(ordered[-1]),
            "observed_frames": set(ordered),
            "observed_frame_count": len(ordered),
        }
    return stats


def _load_index_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AggregateError(f"crop embeddings index not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise AggregateError("crop embeddings index is empty")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_non_finite_json)
        except AggregateError:
            raise
        except json.JSONDecodeError as exc:
            raise AggregateError(
                f"invalid JSON on crop index line {line_no}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise AggregateError(f"crop index line {line_no} must be a JSON object")
        for key in ("crop_id", "track_id", "frame_index", "embedding_row"):
            if key not in obj:
                raise AggregateError(
                    f"crop index line {line_no} missing field {key}"
                )
        rows.append(obj)
    if not rows:
        raise AggregateError("crop embeddings index is empty")
    return rows


def load_crop_embedding_bundle(
    *,
    crop_embeddings: str | Path,
    crop_embeddings_index: str | Path,
) -> dict[str, Any]:
    npz_path = Path(crop_embeddings).expanduser().resolve()
    index_path = Path(crop_embeddings_index).expanduser().resolve()
    if not npz_path.is_file():
        raise AggregateError(f"crop embeddings NPZ not found: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=False)
    except ValueError as exc:
        raise AggregateError(f"failed loading crop NPZ (pickle?): {exc}") from exc

    required = ("vectors", "crop_ids", "track_ids", "frame_indices")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise AggregateError(f"crop NPZ missing arrays: {missing}")

    try:
        vectors = np.asarray(data["vectors"])
        crop_ids = np.asarray(data["crop_ids"])
        track_ids = np.asarray(data["track_ids"])
        frame_indices = np.asarray(data["frame_indices"])
    except ValueError as exc:
        raise AggregateError(f"failed reading crop NPZ arrays: {exc}") from exc

    for name, array in (
        ("vectors", vectors),
        ("crop_ids", crop_ids),
        ("track_ids", track_ids),
        ("frame_indices", frame_indices),
    ):
        if array.dtype == object:
            raise AggregateError(f"crop NPZ array {name} must not use object dtype")

    n = int(vectors.shape[0])
    if n <= 0:
        raise AggregateError("crop embeddings NPZ has zero rows")
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise AggregateError(
            f"crop vectors must have shape (N, {EMBEDDING_DIM}), got {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    if not np.isfinite(vectors).all():
        raise AggregateError("crop embedding vectors contain NaN or Infinity")

    if crop_ids.shape != (n,) or track_ids.shape != (n,) or frame_indices.shape != (n,):
        raise AggregateError("crop NPZ array lengths do not match vectors rows")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise AggregateError("crop embedding rows must have positive finite L2 norms")

    index_rows = _load_index_jsonl(index_path)
    if len(index_rows) != n:
        raise AggregateError(
            f"crop index length {len(index_rows)} != NPZ rows {n}"
        )

    seen_ids: set[str] = set()
    embedding_rows: list[int] = []
    for i, row in enumerate(index_rows):
        crop_id = str(row["crop_id"])
        if crop_id in seen_ids:
            raise AggregateError(f"duplicate crop_id in index: {crop_id}")
        seen_ids.add(crop_id)
        embedding_row = int(row["embedding_row"])
        embedding_rows.append(embedding_row)
        if embedding_row != i:
            raise AggregateError(
                f"embedding_row must equal line order index; "
                f"expected {i}, got {embedding_row}"
            )
        npz_crop_id = str(crop_ids[i])
        if npz_crop_id != crop_id:
            raise AggregateError(
                f"NPZ/index crop_id mismatch at row {i}: "
                f"{npz_crop_id!r} vs {crop_id!r}"
            )
        if int(track_ids[i]) != int(row["track_id"]):
            raise AggregateError(f"NPZ/index track_id mismatch at row {i}")
        if int(frame_indices[i]) != int(row["frame_index"]):
            raise AggregateError(f"NPZ/index frame_index mismatch at row {i}")

    if embedding_rows != list(range(n)):
        raise AggregateError("embedding_row values must be exactly 0..N-1 in order")
    if len(seen_ids) != n:
        raise AggregateError("crop_id values in index are not unique")

    # Also ensure NPZ crop_ids are unique.
    npz_id_list = [str(x) for x in crop_ids.tolist()]
    if len(set(npz_id_list)) != n:
        raise AggregateError("duplicate crop_id in crop NPZ")

    return {
        "npz_path": npz_path,
        "index_path": index_path,
        "vectors": vectors.astype(np.float32, copy=False),
        "crop_ids": npz_id_list,
        "track_ids": [int(x) for x in track_ids.tolist()],
        "frame_indices": [int(x) for x in frame_indices.tolist()],
        "index_rows": index_rows,
        "n": n,
    }


def aggregate_track_embeddings(
    *,
    vectors: np.ndarray,
    crop_ids: Sequence[str],
    track_ids: Sequence[int],
    track_stats: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate crop embeddings into one L2-normalized vector per track."""
    n = int(vectors.shape[0])
    if n != len(crop_ids) or n != len(track_ids):
        raise AggregateError("aggregate input lengths mismatch")

    # Preserve crop input order within each track.
    crops_by_track: dict[int, list[int]] = defaultdict(list)
    for row_idx, track_id in enumerate(track_ids):
        crops_by_track[int(track_id)].append(row_idx)

    missing = sorted(tid for tid in crops_by_track if tid not in track_stats)
    if missing:
        raise AggregateError(
            f"tracks present in embeddings but missing from tracks.jsonl: {missing}"
        )

    ordered_track_ids = sorted(crops_by_track.keys())
    out_vectors = np.zeros((len(ordered_track_ids), EMBEDDING_DIM), dtype=np.float32)
    rows: list[dict[str, Any]] = []

    for emb_row, track_id in enumerate(ordered_track_ids):
        indices = crops_by_track[track_id]
        crop_id_list = [str(crop_ids[i]) for i in indices]
        crop_matrix = vectors[np.asarray(indices, dtype=np.int64)]
        try:
            normalized_crops = l2_normalize_rows(crop_matrix.astype(np.float32, copy=False))
        except Exception as exc:  # EmbeddingError from l2_normalize_rows
            raise AggregateError(
                f"failed L2-normalizing crops for track {track_id}: {exc}"
            ) from exc

        mean_vec = np.mean(normalized_crops, axis=0).astype(np.float32, copy=False)
        if not np.isfinite(mean_vec).all():
            raise AggregateError(f"non-finite mean embedding for track {track_id}")
        mean_norm = float(np.linalg.norm(mean_vec))
        if not math.isfinite(mean_norm) or mean_norm <= 0.0:
            raise AggregateError(f"zero or non-finite mean norm for track {track_id}")
        unit = (mean_vec / mean_norm).astype(np.float32, copy=False)
        final_norm = float(np.linalg.norm(unit))
        if abs(final_norm - 1.0) > _NORM_ATOL:
            raise AggregateError(
                f"aggregated embedding for track {track_id} has L2={final_norm}"
            )

        meta = track_stats[track_id]
        out_vectors[emb_row] = unit
        rows.append(
            {
                "track_id": int(track_id),
                "crop_ids": crop_id_list,
                "crop_count": len(crop_id_list),
                "embedding_row": emb_row,
                "aggregation": AGGREGATION_NAME,
                "embedding_shape": [EMBEDDING_DIM],
                "embedding_dtype": "float32",
                "l2_norm": _ensure_finite_float(final_norm, field="l2_norm"),
                "observation_count": int(meta["observation_count"]),
                "first_frame": int(meta["first_frame"]),
                "last_frame": int(meta["last_frame"]),
                "observed_frame_count": int(meta["observed_frame_count"]),
                "schema_version": TRACK_EMBEDDING_SCHEMA_VERSION,
            }
        )

    return {
        "vectors": out_vectors,
        "rows": rows,
        "track_ids": ordered_track_ids,
    }


def create_temp_aggregate_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_aggregate_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise AggregateError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_aggregate_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise AggregateError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise AggregateError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_aggregate_{final_path.name}_{uuid.uuid4().hex[:8]}"
            )
            os.rename(final_path, backup_path)

        os.rename(temp_path, final_path)

        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=False)
            backup_path = None
    except Exception:
        if backup_path is not None and backup_path.exists() and not final_path.exists():
            try:
                os.rename(backup_path, final_path)
                backup_path = None
            except OSError:
                pass
        raise

    parent = final_path.parent
    for stray in parent.glob(f"_tmp_reid_aggregate_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_aggregate_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def write_aggregation_artifacts(
    *,
    output_dir: Path,
    vectors: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    crop_npz_path: Path,
    crop_index_path: Path,
    tracks_path: Path,
    crop_embedding_count: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    t = len(rows)
    if vectors.shape != (t, EMBEDDING_DIM):
        raise AggregateError("track vectors/rows length mismatch")
    if vectors.dtype != np.float32:
        raise AggregateError("track vectors must be float32")

    track_ids = np.asarray([int(r["track_id"]) for r in rows], dtype=np.int64)
    crop_counts = np.asarray([int(r["crop_count"]) for r in rows], dtype=np.int64)
    first_frames = np.asarray([int(r["first_frame"]) for r in rows], dtype=np.int64)
    last_frames = np.asarray([int(r["last_frame"]) for r in rows], dtype=np.int64)
    observation_counts = np.asarray(
        [int(r["observation_count"]) for r in rows], dtype=np.int64
    )

    for name, array in (
        ("vectors", vectors),
        ("track_ids", track_ids),
        ("crop_counts", crop_counts),
        ("first_frames", first_frames),
        ("last_frames", last_frames),
        ("observation_counts", observation_counts),
    ):
        if array.dtype == object:
            raise AggregateError(f"NPZ array {name} must not use object dtype")

    npz_path = output_dir / TRACK_NPZ_NAME
    np.savez(
        npz_path,
        vectors=vectors,
        track_ids=track_ids,
        crop_counts=crop_counts,
        first_frames=first_frames,
        last_frames=last_frames,
        observation_counts=observation_counts,
    )
    write_manifest_jsonl(output_dir / TRACK_JSONL_NAME, rows)

    norms = np.asarray([float(r["l2_norm"]) for r in rows], dtype=np.float64)
    single = int(sum(1 for r in rows if int(r["crop_count"]) == 1))
    multi = int(sum(1 for r in rows if int(r["crop_count"]) > 1))
    summary = {
        "status": "ok",
        "crop_embedding_count": int(crop_embedding_count),
        "track_embedding_count": t,
        "tracks_with_single_crop": single,
        "tracks_with_multiple_crops": multi,
        "aggregation": AGGREGATION_NAME,
        "embedding_shape": [t, EMBEDDING_DIM],
        "embedding_dtype": "float32",
        "l2_norm_min": _ensure_finite_float(float(np.min(norms)), field="l2_norm_min"),
        "l2_norm_max": _ensure_finite_float(float(np.max(norms)), field="l2_norm_max"),
        "crop_embeddings_npz": str(crop_npz_path),
        "crop_embeddings_index": str(crop_index_path),
        "tracks_jsonl": str(tracks_path),
        "elapsed_sec": _ensure_finite_float(float(elapsed_sec), field="elapsed_sec"),
        "schema_version": AGGREGATION_SUMMARY_SCHEMA_VERSION,
    }
    (output_dir / AGGREGATION_SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )

    names = sorted(p.name for p in output_dir.iterdir())
    expected = sorted([TRACK_NPZ_NAME, TRACK_JSONL_NAME, AGGREGATION_SUMMARY_NAME])
    if names != expected:
        raise AggregateError(f"unexpected aggregation output files: {names}")

    return {"summary": summary, "npz_path": str(npz_path)}


def run_aggregate_reid_tracks(
    *,
    crop_embeddings: str | Path,
    crop_embeddings_index: str | Path,
    tracks: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    final_dir = Path(output_dir).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    check_output_collision(final_dir, overwrite=overwrite)

    started = time.perf_counter()
    bundle = load_crop_embedding_bundle(
        crop_embeddings=crop_embeddings,
        crop_embeddings_index=crop_embeddings_index,
    )
    observations = load_tracks_jsonl(tracks_path)
    track_stats = build_track_observation_stats(observations)

    aggregated = aggregate_track_embeddings(
        vectors=bundle["vectors"],
        crop_ids=bundle["crop_ids"],
        track_ids=bundle["track_ids"],
        track_stats=track_stats,
    )

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_aggregate_dir(final_dir)
        elapsed = time.perf_counter() - started
        artifacts = write_aggregation_artifacts(
            output_dir=temp_dir,
            vectors=aggregated["vectors"],
            rows=aggregated["rows"],
            crop_npz_path=bundle["npz_path"],
            crop_index_path=bundle["index_path"],
            tracks_path=tracks_path,
            crop_embedding_count=bundle["n"],
            elapsed_sec=elapsed,
        )
        finalized = finalize_aggregate_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    summary = artifacts["summary"]
    return {
        "status": "ok",
        "output_dir": str(finalized),
        "npz_path": str(finalized / TRACK_NPZ_NAME),
        "jsonl_path": str(finalized / TRACK_JSONL_NAME),
        "summary_path": str(finalized / AGGREGATION_SUMMARY_NAME),
        "crop_embedding_count": summary["crop_embedding_count"],
        "track_embedding_count": summary["track_embedding_count"],
        "tracks_with_single_crop": summary["tracks_with_single_crop"],
        "tracks_with_multiple_crops": summary["tracks_with_multiple_crops"],
        "embedding_shape": summary["embedding_shape"],
        "embedding_dtype": summary["embedding_dtype"],
        "l2_norm_min": summary["l2_norm_min"],
        "l2_norm_max": summary["l2_norm_max"],
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
        "rows": aggregated["rows"],
        "vectors": aggregated["vectors"],
    }
