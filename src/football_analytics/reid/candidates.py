"""ReID track candidate pair generation (Stage 4B-4).

Produces unordered track pairs with cosine similarity and temporal conflict
metadata. No similarity threshold and no linking / global_candidate_id.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from football_analytics.reid.aggregate import (
    AggregateError,
    build_track_observation_stats,
    load_tracks_jsonl,
)
from football_analytics.reid.embedding import EMBEDDING_DIM
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

CANDIDATE_PAIR_SCHEMA_VERSION = "reid_candidate_pair_v1"
CANDIDATE_SUMMARY_SCHEMA_VERSION = "reid_candidate_summary_v1"

PAIRS_NAME = "candidate_pairs.jsonl"
SUMMARY_NAME = "candidate_summary.json"

DECISION_REJECTED = "rejected"
DECISION_ELIGIBLE = "eligible_unthresholded"
REASON_EXACT_FRAME = "exact_frame_conflict"
REASON_THRESHOLD_PENDING = "similarity_threshold_pending"

_NORM_ATOL = 1e-4
_COSINE_CLAMP = 1e-6


class CandidateError(RuntimeError):
    """Raised when candidate pair generation fails."""


def _reject_non_finite_json(value: str) -> None:
    raise CandidateError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CandidateError(f"{field} must be finite, got {value!r}")
    return number


def _nullable_stat(values: Sequence[float], reducer) -> float | None:
    if not values:
        return None
    result = float(reducer(values))
    return _ensure_finite_float(result, field="stat")


def temporal_gap_frames(
    *,
    first_a: int,
    last_a: int,
    first_b: int,
    last_b: int,
) -> int:
    """Return empty frames between disjoint spans; 0 if overlapping or adjacent."""
    if first_a > last_a or first_b > last_b:
        raise CandidateError("invalid span: first_frame > last_frame")
    # Overlap (inclusive intervals).
    if first_a <= last_b and first_b <= last_a:
        return 0
    if last_a < first_b:
        return max(0, first_b - last_a - 1)
    return max(0, first_a - last_b - 1)


def span_interval_overlap(
    *,
    first_a: int,
    last_a: int,
    first_b: int,
    last_b: int,
) -> bool:
    return first_a <= last_b and first_b <= last_a


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.shape != (EMBEDDING_DIM,) or vec_b.shape != (EMBEDDING_DIM,):
        raise CandidateError(
            f"cosine expects shape ({EMBEDDING_DIM},), got {vec_a.shape} / {vec_b.shape}"
        )
    a = vec_a.astype(np.float32, copy=False)
    b = vec_b.astype(np.float32, copy=False)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise CandidateError("cosine inputs contain NaN or Infinity")
    value = float(np.dot(a, b))
    if not math.isfinite(value):
        raise CandidateError(f"non-finite cosine similarity: {value}")
    if value > 1.0 + _COSINE_CLAMP or value < -1.0 - _COSINE_CLAMP:
        raise CandidateError(f"cosine similarity out of range: {value}")
    return float(max(-1.0, min(1.0, value)))


def load_track_embedding_bundle(
    *,
    track_embeddings: str | Path,
    track_embeddings_index: str | Path,
) -> dict[str, Any]:
    npz_path = Path(track_embeddings).expanduser().resolve()
    index_path = Path(track_embeddings_index).expanduser().resolve()
    if not npz_path.is_file():
        raise CandidateError(f"track embeddings NPZ not found: {npz_path}")
    if not index_path.is_file():
        raise CandidateError(f"track embeddings JSONL not found: {index_path}")

    try:
        data = np.load(npz_path, allow_pickle=False)
    except ValueError as exc:
        raise CandidateError(f"failed loading track NPZ (pickle?): {exc}") from exc

    required = (
        "vectors",
        "track_ids",
        "crop_counts",
        "first_frames",
        "last_frames",
        "observation_counts",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise CandidateError(f"track NPZ missing arrays: {missing}")

    try:
        vectors = np.asarray(data["vectors"])
        track_ids = np.asarray(data["track_ids"])
        crop_counts = np.asarray(data["crop_counts"])
        first_frames = np.asarray(data["first_frames"])
        last_frames = np.asarray(data["last_frames"])
        observation_counts = np.asarray(data["observation_counts"])
    except ValueError as exc:
        raise CandidateError(f"failed reading track NPZ arrays: {exc}") from exc

    for name, array in (
        ("vectors", vectors),
        ("track_ids", track_ids),
        ("crop_counts", crop_counts),
        ("first_frames", first_frames),
        ("last_frames", last_frames),
        ("observation_counts", observation_counts),
    ):
        if array.dtype == object:
            raise CandidateError(f"track NPZ array {name} must not use object dtype")

    t = int(vectors.shape[0])
    if t <= 0:
        raise CandidateError("track embeddings NPZ has zero rows")
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise CandidateError(
            f"track vectors must have shape (T, {EMBEDDING_DIM}), got {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    if not np.isfinite(vectors).all():
        raise CandidateError("track embedding vectors contain NaN or Infinity")

    if (
        track_ids.shape != (t,)
        or crop_counts.shape != (t,)
        or first_frames.shape != (t,)
        or last_frames.shape != (t,)
        or observation_counts.shape != (t,)
    ):
        raise CandidateError("track NPZ array lengths do not match vectors rows")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise CandidateError("track embedding norms contain NaN or Infinity")
    if np.any(np.abs(norms - 1.0) > _NORM_ATOL):
        raise CandidateError("track embeddings must be approximately L2-normalized")

    text = index_path.read_text(encoding="utf-8")
    if not text.strip():
        raise CandidateError("track embeddings JSONL is empty")
    index_rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_non_finite_json)
        except CandidateError:
            raise
        except json.JSONDecodeError as exc:
            raise CandidateError(
                f"invalid JSON on track index line {line_no}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise CandidateError(f"track index line {line_no} must be a JSON object")
        for key in (
            "track_id",
            "embedding_row",
            "first_frame",
            "last_frame",
            "observation_count",
            "observed_frame_count",
            "crop_count",
        ):
            if key not in obj:
                raise CandidateError(
                    f"track index line {line_no} missing field {key}"
                )
        index_rows.append(obj)

    if len(index_rows) != t:
        raise CandidateError(
            f"track index length {len(index_rows)} != NPZ rows {t}"
        )

    seen: set[int] = set()
    for i, row in enumerate(index_rows):
        track_id = int(row["track_id"])
        if track_id in seen:
            raise CandidateError(f"duplicate track_id in track index: {track_id}")
        seen.add(track_id)
        if int(row["embedding_row"]) != i:
            raise CandidateError(
                f"embedding_row must equal line order; expected {i}, "
                f"got {row['embedding_row']}"
            )
        if int(track_ids[i]) != track_id:
            raise CandidateError(f"NPZ/index track_id mismatch at row {i}")
        if int(first_frames[i]) != int(row["first_frame"]):
            raise CandidateError(f"NPZ/index first_frame mismatch at row {i}")
        if int(last_frames[i]) != int(row["last_frame"]):
            raise CandidateError(f"NPZ/index last_frame mismatch at row {i}")
        if int(observation_counts[i]) != int(row["observation_count"]):
            raise CandidateError(f"NPZ/index observation_count mismatch at row {i}")
        if int(crop_counts[i]) != int(row["crop_count"]):
            raise CandidateError(f"NPZ/index crop_count mismatch at row {i}")

    # Track ids must be ascending to match aggregation output contract.
    tid_list = [int(x) for x in track_ids.tolist()]
    if tid_list != sorted(tid_list):
        raise CandidateError("track_ids must be sorted ascending")

    return {
        "npz_path": npz_path,
        "index_path": index_path,
        "vectors": vectors.astype(np.float32, copy=False),
        "track_ids": tid_list,
        "first_frames": [int(x) for x in first_frames.tolist()],
        "last_frames": [int(x) for x in last_frames.tolist()],
        "observation_counts": [int(x) for x in observation_counts.tolist()],
        "crop_counts": [int(x) for x in crop_counts.tolist()],
        "index_rows": index_rows,
        "t": t,
    }


def build_candidate_pairs(
    *,
    vectors: np.ndarray,
    track_ids: Sequence[int],
    track_stats: Mapping[int, Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    t = len(track_ids)
    if vectors.shape != (t, EMBEDDING_DIM):
        raise CandidateError("vectors/track_ids length mismatch")

    for track_id in track_ids:
        if int(track_id) not in track_stats:
            raise CandidateError(
                f"track {track_id} missing from tracks.jsonl observation stats"
            )

    # Optional consistency with index metadata spans.
    if index_rows is not None:
        if len(index_rows) != t:
            raise CandidateError("index_rows length mismatch")
        for i, track_id in enumerate(track_ids):
            meta = track_stats[int(track_id)]
            row = index_rows[i]
            if int(row["first_frame"]) != int(meta["first_frame"]):
                raise CandidateError(
                    f"index first_frame disagrees with tracks.jsonl for {track_id}"
                )
            if int(row["last_frame"]) != int(meta["last_frame"]):
                raise CandidateError(
                    f"index last_frame disagrees with tracks.jsonl for {track_id}"
                )
            if int(row["observation_count"]) != int(meta["observation_count"]):
                raise CandidateError(
                    f"index observation_count disagrees with tracks.jsonl for {track_id}"
                )
            if int(row["observed_frame_count"]) != int(meta["observed_frame_count"]):
                raise CandidateError(
                    f"index observed_frame_count disagrees with tracks.jsonl for {track_id}"
                )

    pairs: list[dict[str, Any]] = []
    for i in range(t):
        for j in range(i + 1, t):
            tid_a = int(track_ids[i])
            tid_b = int(track_ids[j])
            if tid_a >= tid_b:
                raise CandidateError(
                    "track_ids must be ascending so pairs satisfy track_id_a < track_id_b"
                )

            meta_a = track_stats[tid_a]
            meta_b = track_stats[tid_b]
            frames_a: set[int] = meta_a["observed_frames"]
            frames_b: set[int] = meta_b["observed_frames"]
            overlap_count = len(frames_a & frames_b)
            exact_conflict = overlap_count > 0
            first_a = int(meta_a["first_frame"])
            last_a = int(meta_a["last_frame"])
            first_b = int(meta_b["first_frame"])
            last_b = int(meta_b["last_frame"])
            span_overlap = span_interval_overlap(
                first_a=first_a, last_a=last_a, first_b=first_b, last_b=last_b
            )
            gap = temporal_gap_frames(
                first_a=first_a, last_a=last_a, first_b=first_b, last_b=last_b
            )
            cosine = cosine_similarity(vectors[i], vectors[j])

            if exact_conflict:
                decision = DECISION_REJECTED
                reason = REASON_EXACT_FRAME
            else:
                decision = DECISION_ELIGIBLE
                reason = REASON_THRESHOLD_PENDING

            pairs.append(
                {
                    "track_id_a": tid_a,
                    "track_id_b": tid_b,
                    "cosine_similarity": _ensure_finite_float(
                        cosine, field="cosine_similarity"
                    ),
                    "temporal_gap_frames": int(gap),
                    "exact_frame_overlap_count": int(overlap_count),
                    "exact_frame_conflict": bool(exact_conflict),
                    "span_interval_overlap": bool(span_overlap),
                    "decision": decision,
                    "decision_reason": reason,
                    "schema_version": CANDIDATE_PAIR_SCHEMA_VERSION,
                }
            )

    # Lexicographic by (track_id_a, track_id_b) — already implied by nested loops
    # over ascending track_ids, but assert determinism.
    keys = [(p["track_id_a"], p["track_id_b"]) for p in pairs]
    if keys != sorted(keys):
        raise CandidateError("candidate pairs are not lexicographically sorted")
    return pairs


def summarize_candidate_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    track_count: int,
    elapsed_sec: float,
    track_npz_path: Path,
    track_index_path: Path,
    tracks_path: Path,
) -> dict[str, Any]:
    total = len(pairs)
    expected = track_count * (track_count - 1) // 2
    if total != expected:
        raise CandidateError(
            f"pair count {total} != T*(T-1)/2 = {expected} for T={track_count}"
        )

    if any(p.get("decision") == "accepted_link" for p in pairs):
        raise CandidateError("accepted_link decisions are forbidden in Stage 4B-4")

    exact_conflict = [p for p in pairs if p["exact_frame_conflict"]]
    eligible = [p for p in pairs if p["decision"] == DECISION_ELIGIBLE]
    span_overlap = [p for p in pairs if p["span_interval_overlap"]]
    span_only = [
        p
        for p in pairs
        if p["span_interval_overlap"] and not p["exact_frame_conflict"]
    ]
    disjoint = [p for p in pairs if not p["span_interval_overlap"]]

    all_cos = [float(p["cosine_similarity"]) for p in pairs]
    elig_cos = [float(p["cosine_similarity"]) for p in eligible]

    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    summary = {
        "status": "ok",
        "track_count": int(track_count),
        "total_pairs": int(total),
        "exact_frame_conflict_pairs": len(exact_conflict),
        "eligible_unthresholded_pairs": len(eligible),
        "span_overlap_pairs": len(span_overlap),
        "span_only_overlap_pairs": len(span_only),
        "temporally_disjoint_pairs": len(disjoint),
        "cosine_min": _nullable_stat(all_cos, min),
        "cosine_median": _nullable_stat(all_cos, _median),
        "cosine_max": _nullable_stat(all_cos, max),
        "eligible_cosine_min": _nullable_stat(elig_cos, min),
        "eligible_cosine_median": _nullable_stat(elig_cos, _median),
        "eligible_cosine_max": _nullable_stat(elig_cos, max),
        "similarity_threshold": None,
        "automatic_linking_performed": False,
        "track_embeddings_npz": str(track_npz_path),
        "track_embeddings_index": str(track_index_path),
        "tracks_jsonl": str(tracks_path),
        "elapsed_sec": _ensure_finite_float(float(elapsed_sec), field="elapsed_sec"),
        "schema_version": CANDIDATE_SUMMARY_SCHEMA_VERSION,
    }
    return summary


def create_temp_candidate_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_candidates_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise CandidateError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_candidate_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise CandidateError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise CandidateError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_candidates_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_candidates_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_candidates_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def write_candidate_artifacts(
    *,
    output_dir: Path,
    pairs: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    write_manifest_jsonl(output_dir / PAIRS_NAME, pairs)
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(dict(summary), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    names = sorted(p.name for p in output_dir.iterdir())
    expected = sorted([PAIRS_NAME, SUMMARY_NAME])
    if names != expected:
        raise CandidateError(f"unexpected candidate output files: {names}")


def run_build_reid_candidates(
    *,
    track_embeddings: str | Path,
    track_embeddings_index: str | Path,
    tracks: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    final_dir = Path(output_dir).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    check_output_collision(final_dir, overwrite=overwrite)

    started = time.perf_counter()
    try:
        bundle = load_track_embedding_bundle(
            track_embeddings=track_embeddings,
            track_embeddings_index=track_embeddings_index,
        )
        observations = load_tracks_jsonl(tracks_path)
        track_stats = build_track_observation_stats(observations)
    except AggregateError as exc:
        raise CandidateError(str(exc)) from exc

    pairs = build_candidate_pairs(
        vectors=bundle["vectors"],
        track_ids=bundle["track_ids"],
        track_stats=track_stats,
        index_rows=bundle["index_rows"],
    )
    elapsed = time.perf_counter() - started
    summary = summarize_candidate_pairs(
        pairs,
        track_count=bundle["t"],
        elapsed_sec=elapsed,
        track_npz_path=bundle["npz_path"],
        track_index_path=bundle["index_path"],
        tracks_path=tracks_path,
    )

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_candidate_dir(final_dir)
        write_candidate_artifacts(output_dir=temp_dir, pairs=pairs, summary=summary)
        finalized = finalize_candidate_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "pairs_path": str(finalized / PAIRS_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "track_count": summary["track_count"],
        "total_pairs": summary["total_pairs"],
        "exact_frame_conflict_pairs": summary["exact_frame_conflict_pairs"],
        "eligible_unthresholded_pairs": summary["eligible_unthresholded_pairs"],
        "span_only_overlap_pairs": summary["span_only_overlap_pairs"],
        "cosine_min": summary["cosine_min"],
        "cosine_median": summary["cosine_median"],
        "cosine_max": summary["cosine_max"],
        "similarity_threshold": None,
        "automatic_linking_performed": False,
        "elapsed_sec": summary["elapsed_sec"],
        "summary": summary,
        "pairs": pairs,
    }
