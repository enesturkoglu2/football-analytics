"""Stage 5B3G — segmented ReID regression orchestration.

Builds a segment-entity embedding view from the non-destructive segment
view plus Stage 4B baseline artifacts. Manual-split parents recompute
embeddings from segment observations; unaffected full-track segments
reuse baseline track embeddings without expanding coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from football_analytics.reid.aggregate import AGGREGATION_NAME
from football_analytics.reid.candidates import (
    cosine_similarity,
    load_track_embedding_bundle,
    span_interval_overlap,
)
from football_analytics.reid.crop_extract import (
    extract_crops_single_pass,
    probe_video_size,
)
from football_analytics.reid.crop_select import (
    filter_candidate_observations,
    load_crop_selection_config,
    select_crops_for_tracks,
)
from football_analytics.reid.embedding import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PREPROCESSING,
    embed_tensors,
    l2_normalize_rows,
    load_and_preprocess_crop,
    sha256_file,
    verify_checkpoint,
    verify_sn_reid_root,
)
from football_analytics.reid.segments import (
    SEGMENT_KIND_CONTROL,
    SEGMENT_KIND_MANUAL,
    SEGMENT_KIND_PRESERVED,
    load_segment_decisions,
    load_segmentation_policy,
)
from football_analytics.reid.writers import (
    ReIDWritersError,
    check_output_collision,
    cleanup_dir,
    validate_manifest_disk_consistency,
)

SEGMENT_CROP_MANIFEST_NAME = "segment_crop_manifest.jsonl"
SEGMENT_EMB_INDEX_NAME = "segment_embedding_index.jsonl"
SEGMENT_EMB_NPZ_NAME = "segment_embeddings.npz"
SEGMENT_CANDIDATES_NAME = "segment_candidates.jsonl"
REPLACEMENT_MAP_NAME = "baseline_to_segment_replacement.jsonl"
PAIR_DELTAS_NAME = "segmented_reid_pair_deltas.jsonl"
SUMMARY_NAME = "segmented_reid_regression_summary.json"
CROPS_DIRNAME = "crops"

SEGMENT_CROP_SCHEMA = "reid_segment_crop_manifest_v1"
SEGMENT_EMB_SCHEMA = "reid_segment_embedding_record_v1"
SEGMENT_CANDIDATE_SCHEMA = "reid_segment_candidate_v1"
REPLACEMENT_SCHEMA = "reid_baseline_segment_replacement_v1"
SUMMARY_SCHEMA = "reid_segmented_regression_summary_v1"
CONFIG_SCHEMA = "reid_segmented_regression_config_v1"

STATUS_RECOMPUTE = "recompute_manual_segment"
STATUS_REUSE = "reuse_baseline_full_track"
STATUS_NO_BASELINE = "no_baseline_embedding"
STATUS_NO_CROP = "manual_segment_no_eligible_crop"

_NORM_ATOL = 1e-4
_SIM_ATOL = 1e-5


class SegmentRegressionError(RuntimeError):
    """Raised when segmented ReID regression validation or writing fails."""


def _reject_non_finite_json(value: str) -> None:
    raise SegmentRegressionError(f"non-finite JSON constant is not allowed: {value}")


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise SegmentRegressionError(f"{field} must be a bool, got {value!r}")
    if expected is not None and value is not expected:
        raise SegmentRegressionError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_null(value: Any, *, field: str) -> None:
    if value is not None:
        raise SegmentRegressionError(f"{field} must be null, got {value!r}")


def embedding_vector_sha256(vector: np.ndarray) -> str:
    arr = np.asarray(vector, dtype=np.float32)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SegmentRegressionError(f"JSONL not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise SegmentRegressionError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_non_finite_json)
        except SegmentRegressionError:
            raise
        except json.JSONDecodeError as exc:
            raise SegmentRegressionError(
                f"invalid JSON on {path.name} line {line_no}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise SegmentRegressionError(f"{path.name} line {line_no} must be an object")
        rows.append(obj)
    return rows


def create_temp_regression_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_segreg_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise SegmentRegressionError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    (tmp_dir / CROPS_DIRNAME).mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_regression_dir(
    *, temp_dir: Path, final_dir: Path, overwrite: bool
) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise SegmentRegressionError(f"temporary output directory missing: {temp_path}")
    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise SegmentRegressionError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_segreg_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_segreg_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_segreg_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def validate_regression_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SegmentRegressionError("regression config must be a mapping")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise SegmentRegressionError(
            f"regression config schema_version must be {CONFIG_SCHEMA}"
        )
    if payload.get("stage_status") != "implementation_baseline":
        raise SegmentRegressionError(
            "regression config stage_status must be implementation_baseline"
        )

    rep = payload.get("representation")
    if not isinstance(rep, dict):
        raise SegmentRegressionError("representation must be a mapping")
    if rep.get("entity_id") != "segment_id":
        raise SegmentRegressionError("representation.entity_id must be segment_id")
    _require_bool(rep.get("raw_track_id_preserved"), field="raw_track_id_preserved", expected=True)
    if rep.get("recompute_scope") != "manual_split_segments_only":
        raise SegmentRegressionError(
            "representation.recompute_scope must be manual_split_segments_only"
        )
    for key, expected in (
        ("reuse_unaffected_baseline_embeddings", True),
        ("reuse_no_split_control_baseline_embeddings", True),
        ("reuse_pass_through_baseline_embeddings", True),
        ("reuse_manual_split_raw_track_embedding", False),
        ("expand_baseline_embedding_coverage_for_unaffected_tracks", False),
        ("mixed_raw_track_embedding_allowed_for_manual_split_tracks", False),
    ):
        _require_bool(rep.get(key), field=f"representation.{key}", expected=expected)

    crop = payload.get("crop_selection")
    if not isinstance(crop, dict):
        raise SegmentRegressionError("crop_selection must be a mapping")
    for key, expected in (
        ("reuse_existing_stage4b_algorithm", True),
        ("selected_from_assigned_segment_observations_only", True),
        ("ambiguous_observations_allowed", False),
        ("unassigned_observations_allowed", False),
        ("fallback_to_parent_raw_track_crops", False),
        ("missing_frames_interpolated", False),
        ("empty_segment_allowed", True),
        ("empty_segment_requires_reason", True),
    ):
        _require_bool(crop.get(key), field=f"crop_selection.{key}", expected=expected)

    emb = payload.get("embedding")
    if not isinstance(emb, dict):
        raise SegmentRegressionError("embedding must be a mapping")
    if emb.get("backend") != "existing_stage4b_osnet":
        raise SegmentRegressionError("embedding.backend must be existing_stage4b_osnet")
    for key, expected in (
        ("preprocessing_must_match_stage4b", True),
        ("checkpoint_provenance_required", True),
        ("checkpoint_sha256_required", True),
        ("normalized_embedding_required", True),
        ("recomputed_manual_segment_embedding_required_when_crop_available", True),
        ("quality_weighting_enabled", False),
        ("kit_weighting_enabled", False),
    ):
        _require_bool(emb.get(key), field=f"embedding.{key}", expected=expected)

    cand = payload.get("candidate_generation")
    if not isinstance(cand, dict):
        raise SegmentRegressionError("candidate_generation must be a mapping")
    _require_bool(cand.get("cosine_similarity_only"), field="cosine_similarity_only", expected=True)
    _require_null(cand.get("similarity_threshold"), field="similarity_threshold")
    for key, expected in (
        ("exact_same_frame_overlap_hard_reject", True),
        ("span_overlap_alone_hard_reject", False),
        ("same_raw_track_segment_pair_allowed_for_ranking", True),
        ("same_raw_track_segment_pair_is_auto_merge", False),
        ("automatic_link_enabled", False),
        ("automatic_reject_enabled", False),
        ("component_building_enabled", False),
        ("manual_review_required", True),
    ):
        _require_bool(cand.get(key), field=f"candidate_generation.{key}", expected=expected)

    reg = payload.get("regression")
    if not isinstance(reg, dict):
        raise SegmentRegressionError("regression must be a mapping")
    for key, expected in (
        ("compare_to_raw_track_baseline", True),
        ("affected_raw_track_embedding_retired_from_segmented_view", True),
        ("unaffected_pair_similarity_must_match_when_both_embeddings_reused", True),
        ("candidate_rank_delta_is_accuracy", False),
        ("identity_ground_truth_available", False),
        ("accuracy_claim_allowed", False),
    ):
        _require_bool(reg.get(key), field=f"regression.{key}", expected=expected)

    gi = payload.get("global_identity")
    if not isinstance(gi, dict):
        raise SegmentRegressionError("global_identity must be a mapping")
    for key, expected in (
        ("global_id_rewrite_enabled", False),
        ("existing_stage4b_output_preserved", True),
        ("component_assignment_enabled", False),
        ("accepted_components_automatically_modified", False),
    ):
        _require_bool(gi.get(key), field=f"global_identity.{key}", expected=expected)

    audit = payload.get("existing_component_audit")
    if not isinstance(audit, dict):
        raise SegmentRegressionError("existing_component_audit must be a mapping")
    if audit.get("raw_component") != [231, 635]:
        raise SegmentRegressionError(
            f"existing_component_audit.raw_component must be [231, 635], got {audit.get('raw_component')!r}"
        )
    for key, expected in (
        ("unchanged", True),
        ("raw_231_s01_inherits_component", False),
        ("raw_231_s02_automatically_links_to_635", False),
        ("future_manual_review_required", True),
    ):
        _require_bool(audit.get(key), field=f"existing_component_audit.{key}", expected=expected)
    return dict(payload)


def load_regression_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise SegmentRegressionError(f"regression config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SegmentRegressionError(f"invalid regression config YAML: {exc}") from exc
    return validate_regression_config(payload)


def load_segment_view(segment_view_dir: str | Path) -> dict[str, Any]:
    root = Path(segment_view_dir).expanduser().resolve()
    if not root.is_dir():
        raise SegmentRegressionError(f"segment-view directory not found: {root}")
    segments_path = root / "track_segments.jsonl"
    assigned_path = root / "segment_observations.jsonl"
    unassigned_path = root / "unassigned_observations.jsonl"
    summary_path = root / "segment_view_summary.json"
    for path in (segments_path, assigned_path, unassigned_path, summary_path):
        if not path.is_file():
            raise SegmentRegressionError(f"segment-view artifact missing: {path}")

    summary = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_json,
    )
    if not isinstance(summary, dict):
        raise SegmentRegressionError("segment_view_summary.json must be an object")
    for key, expected in (
        ("derived_view_only", True),
        ("segment_view_is_raw_track_replacement", False),
        ("raw_tracks_mutated", False),
        ("source_observations_preserved", True),
        ("automatic_split_performed", False),
        ("automatic_merge_performed", False),
        ("automatic_link_performed", False),
        ("global_id_rewrite_performed", False),
        ("team_assignment_performed", False),
        ("proven_physical_identity", False),
        ("accuracy_claimed", False),
    ):
        _require_bool(summary.get(key), field=f"segment_view_summary.{key}", expected=expected)
    for key in (
        "created_observation_count",
        "interpolated_observation_count",
        "deleted_observation_count",
        "duplicated_source_observation_count",
        "uncovered_source_observation_count",
    ):
        if int(summary.get(key, -1)) != 0:
            raise SegmentRegressionError(f"segment_view_summary.{key} must be 0")
    audit = summary.get("existing_component_audit")
    if not isinstance(audit, dict):
        raise SegmentRegressionError("segment_view_summary.existing_component_audit missing")
    if audit.get("raw_component") != [231, 635]:
        raise SegmentRegressionError("segment view raw_component must be [231, 635]")
    _require_bool(audit.get("modified"), field="existing_component_audit.modified", expected=False)

    segments = _load_jsonl(segments_path)
    assigned = _load_jsonl(assigned_path)
    unassigned = _load_jsonl(unassigned_path)
    seg_ids = [row["segment_id"] for row in segments]
    if len(seg_ids) != len(set(seg_ids)):
        raise SegmentRegressionError("duplicate segment_id in track_segments.jsonl")
    seg_id_set = set(seg_ids)
    by_id = {row["segment_id"]: row for row in segments}

    assigned_hashes: set[str] = set()
    for row in assigned:
        if row.get("segment_id") not in seg_id_set:
            raise SegmentRegressionError(
                f"assigned observation references unknown segment {row.get('segment_id')}"
            )
        nested = row.get("source_observation")
        if not isinstance(nested, dict):
            raise SegmentRegressionError("assigned source_observation must be an object")
        if "segment_id" in nested:
            raise SegmentRegressionError("source_observation must not contain segment_id")
        digest = str(row["source_observation_sha256"])
        if digest in assigned_hashes:
            raise SegmentRegressionError(f"duplicate assigned provenance hash: {digest}")
        assigned_hashes.add(digest)

    unassigned_hashes: set[str] = set()
    for row in unassigned:
        if "segment_id" in row:
            raise SegmentRegressionError("unassigned observation must not include segment_id")
        digest = str(row["source_observation_sha256"])
        if digest in unassigned_hashes:
            raise SegmentRegressionError(f"duplicate unassigned provenance hash: {digest}")
        unassigned_hashes.add(digest)
    if assigned_hashes & unassigned_hashes:
        raise SegmentRegressionError("assigned/unassigned provenance hashes intersect")

    def seg_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
        if row["segment_kind"] == SEGMENT_KIND_MANUAL:
            return (int(row["raw_track_id"]), 0, int(row["segment_index"]))
        return (int(row["raw_track_id"]), 1, 0)

    if [seg_key(r) for r in segments] != sorted(seg_key(r) for r in segments):
        raise SegmentRegressionError("track_segments.jsonl is not deterministically ordered")

    def obs_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            int(row["raw_track_id"]),
            0 if row["segment_kind"] == SEGMENT_KIND_MANUAL else 1,
            -1 if row["segment_index"] is None else int(row["segment_index"]),
            int(row["frame_index"]),
            int(row["source_row_index"]),
        )

    if [obs_key(r) for r in assigned] != sorted(obs_key(r) for r in assigned):
        raise SegmentRegressionError("segment_observations.jsonl is not deterministically ordered")

    return {
        "root": root,
        "segments": segments,
        "assigned": assigned,
        "unassigned": unassigned,
        "summary": summary,
        "by_id": by_id,
        "paths": {
            "track_segments": segments_path,
            "segment_observations": assigned_path,
            "unassigned_observations": unassigned_path,
            "summary": summary_path,
        },
    }


def validate_segment_view_against_decisions(
    view: Mapping[str, Any], decisions: Mapping[str, Any]
) -> None:
    by_id = {int(t["raw_track_id"]): t for t in decisions["tracks"]}
    planned_ids = {
        str(seg["segment_id"])
        for track in decisions["tracks"]
        for seg in (track.get("segments") or [])
    }
    for row in view["segments"]:
        kind = row["segment_kind"]
        sid = row["segment_id"]
        rid = int(row["raw_track_id"])
        if kind == SEGMENT_KIND_MANUAL:
            if sid not in planned_ids:
                raise SegmentRegressionError(
                    f"manual segment {sid} missing from decisions YAML"
                )
            if by_id[rid]["decision"] != "manual_split_candidate":
                raise SegmentRegressionError(
                    f"manual segment parent {rid} is not a split candidate"
                )
        elif kind in (SEGMENT_KIND_CONTROL, SEGMENT_KIND_PRESERVED):
            if sid != f"raw_{rid}_full":
                raise SegmentRegressionError(
                    f"full-track segment_id must be raw_{rid}_full, got {sid}"
                )
        else:
            raise SegmentRegressionError(f"unknown segment_kind {kind}")


def load_baseline_artifacts(baseline_run_dir: str | Path) -> dict[str, Any]:
    root = Path(baseline_run_dir).expanduser().resolve()
    if not root.is_dir():
        raise SegmentRegressionError(f"baseline run directory not found: {root}")
    crop_manifest = root / "crops" / "crop_manifest.jsonl"
    track_npz = root / "aggregation" / "track_embeddings.npz"
    track_index = root / "aggregation" / "track_embeddings.jsonl"
    cand_pairs = root / "candidates" / "candidate_pairs.jsonl"
    cand_summary = root / "candidates" / "candidate_summary.json"
    emb_summary = root / "embeddings" / "embedding_summary.json"
    for path in (crop_manifest, track_npz, track_index, cand_pairs, cand_summary, emb_summary):
        if not path.is_file():
            raise SegmentRegressionError(f"baseline artifact missing: {path}")

    crops = _load_jsonl(crop_manifest)
    seen_crop_ids: set[str] = set()
    for row in crops:
        if row.get("schema_version") != "reid_crop_manifest_v1":
            raise SegmentRegressionError("baseline crop manifest schema_version mismatch")
        cid = row["crop_id"]
        if cid in seen_crop_ids:
            raise SegmentRegressionError(f"duplicate baseline crop_id: {cid}")
        seen_crop_ids.add(cid)

    bundle = load_track_embedding_bundle(
        track_embeddings=track_npz, track_embeddings_index=track_index
    )
    emb_sum = json.loads(
        emb_summary.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_json,
    )
    if not isinstance(emb_sum, dict):
        raise SegmentRegressionError("embedding_summary.json must be an object")
    ckpt_sha = emb_sum.get("checkpoint_sha256")
    if not isinstance(ckpt_sha, str) or len(ckpt_sha) != 64:
        raise SegmentRegressionError("baseline embedding_summary missing checkpoint_sha256")
    if emb_sum.get("checkpoint_path") is None:
        raise SegmentRegressionError("baseline embedding_summary missing checkpoint_path")

    cand_sum = json.loads(
        cand_summary.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_json,
    )
    if not isinstance(cand_sum, dict):
        raise SegmentRegressionError("candidate_summary.json must be an object")
    if cand_sum.get("similarity_threshold") is not None:
        raise SegmentRegressionError("baseline similarity_threshold must be null")
    _require_bool(
        cand_sum.get("automatic_linking_performed"),
        field="baseline automatic_linking_performed",
        expected=False,
    )

    pairs = _load_jsonl(cand_pairs)
    pair_map: dict[tuple[int, int], dict[str, Any]] = {}
    for row in pairs:
        a = int(row["track_id_a"])
        b = int(row["track_id_b"])
        if a >= b:
            raise SegmentRegressionError(f"baseline pair not canonical: {a},{b}")
        key = (a, b)
        if key in pair_map:
            raise SegmentRegressionError(f"duplicate baseline pair: {key}")
        if row.get("decision") == "accepted_link":
            raise SegmentRegressionError("baseline automatic accept/link is forbidden")
        pair_map[key] = row

    vectors_by_track = {
        int(tid): bundle["vectors"][i].astype(np.float32, copy=False)
        for i, tid in enumerate(bundle["track_ids"])
    }
    index_by_track = {int(row["track_id"]): row for row in bundle["index_rows"]}
    return {
        "root": root,
        "crop_rows": crops,
        "track_bundle": bundle,
        "vectors_by_track": vectors_by_track,
        "index_by_track": index_by_track,
        "candidate_pairs": pairs,
        "candidate_pair_map": pair_map,
        "candidate_summary": cand_sum,
        "embedding_summary": emb_sum,
        "paths": {
            "crop_manifest": crop_manifest,
            "track_npz": track_npz,
            "track_index": track_index,
            "candidate_pairs": cand_pairs,
            "candidate_summary": cand_summary,
            "embedding_summary": emb_summary,
        },
    }


def build_representation_plan(
    *,
    segments: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    split_parents = {
        int(t["raw_track_id"])
        for t in decisions["tracks"]
        if t["decision"] == "manual_split_candidate"
    }
    vectors_by_track = baseline["vectors_by_track"]
    plan: list[dict[str, Any]] = []
    for seg in segments:
        sid = str(seg["segment_id"])
        rid = int(seg["raw_track_id"])
        kind = str(seg["segment_kind"])
        parent_has = rid in vectors_by_track
        if kind == SEGMENT_KIND_MANUAL:
            if rid not in split_parents:
                raise SegmentRegressionError(
                    f"manual segment {sid} parent {rid} is not a split candidate"
                )
            plan.append(
                {
                    "segment_id": sid,
                    "raw_track_id": rid,
                    "segment_kind": kind,
                    "segment_index": seg.get("segment_index"),
                    "status": STATUS_RECOMPUTE,
                    "parent_baseline_embedding_available": parent_has,
                    "parent_mixed_embedding_retired": True,
                }
            )
        elif kind in (SEGMENT_KIND_CONTROL, SEGMENT_KIND_PRESERVED):
            plan.append(
                {
                    "segment_id": sid,
                    "raw_track_id": rid,
                    "segment_kind": kind,
                    "segment_index": None,
                    "status": STATUS_REUSE if parent_has else STATUS_NO_BASELINE,
                    "parent_baseline_embedding_available": parent_has,
                    "parent_mixed_embedding_retired": False,
                }
            )
        else:
            raise SegmentRegressionError(f"unsupported segment_kind {kind}")
    return plan


def _observations_for_segment(
    assigned: Sequence[Mapping[str, Any]], segment_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames: set[int] = set()
    for row in assigned:
        if row["segment_id"] != segment_id:
            continue
        nested = dict(row["source_observation"])
        frame = int(nested["frame_index"])
        if frame in frames:
            raise SegmentRegressionError(
                f"duplicate frame {frame} in segment {segment_id}"
            )
        frames.add(frame)
        nested["_source_row_index"] = int(row["source_row_index"])
        nested["_source_observation_sha256"] = str(row["source_observation_sha256"])
        rows.append(nested)
    rows.sort(key=lambda r: (int(r["frame_index"]), int(r["_source_row_index"])))
    return rows


def select_segment_crops(
    *,
    segment: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    crop_config: Mapping[str, Any],
    video_width: int,
    video_height: int,
    source_video: str,
) -> tuple[list[dict[str, Any]], str | None]:
    if not observations:
        return [], "empty_segment_no_observations"
    plain = []
    meta_by_frame: dict[int, tuple[int, str]] = {}
    for obs in observations:
        frame = int(obs["frame_index"])
        meta_by_frame[frame] = (
            int(obs["_source_row_index"]),
            str(obs["_source_observation_sha256"]),
        )
        plain.append({k: v for k, v in obs.items() if not str(k).startswith("_")})
    candidates, _reasons = filter_candidate_observations(
        plain, config=crop_config, video_width=video_width, video_height=video_height
    )
    if not candidates:
        return [], "no_observations_passed_crop_filters"
    selected = select_crops_for_tracks(
        candidates,
        config=crop_config,
        source_video=source_video,
        track_ids=[int(segment["raw_track_id"])],
    )
    if not selected:
        return [], "crop_selection_returned_empty"
    enriched: list[dict[str, Any]] = []
    for row in selected:
        frame = int(row["frame_index"])
        if not (
            int(segment["configured_frame_min"])
            <= frame
            <= int(segment["configured_frame_max"])
        ):
            raise SegmentRegressionError(
                f"selected crop frame {frame} outside segment {segment['segment_id']}"
            )
        src_row, src_sha = meta_by_frame[frame]
        out = dict(row)
        out.update(
            {
                "segment_id": segment["segment_id"],
                "raw_track_id": int(segment["raw_track_id"]),
                "segment_kind": segment["segment_kind"],
                "segment_index": segment.get("segment_index"),
                "source_observation_row_index": src_row,
                "source_observation_sha256": src_sha,
                "representation_source": "recomputed_manual_segment",
                "parent_mixed_raw_embedding_reused": False,
                "ambiguous_observation_used": False,
                "schema_version": SEGMENT_CROP_SCHEMA,
            }
        )
        enriched.append(out)
    return enriched, None


def aggregate_crop_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise SegmentRegressionError(
            f"crop vectors must have shape (N, {EMBEDDING_DIM}), got {vectors.shape}"
        )
    normalized = l2_normalize_rows(vectors.astype(np.float32, copy=False))
    mean_vec = np.mean(normalized, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(mean_vec).all():
        raise SegmentRegressionError("non-finite mean segment embedding")
    norm = float(np.linalg.norm(mean_vec))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SegmentRegressionError("zero or non-finite segment mean norm")
    unit = (mean_vec / norm).astype(np.float32, copy=False)
    if abs(float(np.linalg.norm(unit)) - 1.0) > _NORM_ATOL:
        raise SegmentRegressionError("segment embedding failed unit-norm check")
    return unit


def build_segment_candidates(
    *,
    entities: Sequence[Mapping[str, Any]],
    vectors_by_segment: Mapping[str, np.ndarray],
    frames_by_segment: Mapping[str, set[int]],
) -> list[dict[str, Any]]:
    available = [
        e
        for e in entities
        if e.get("embedding_available") and e["segment_id"] in vectors_by_segment
    ]
    available.sort(key=lambda e: str(e["segment_id"]))
    pairs: list[dict[str, Any]] = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            a = available[i]
            b = available[j]
            sid_a, sid_b = str(a["segment_id"]), str(b["segment_id"])
            if sid_a > sid_b:
                a, b = b, a
                sid_a, sid_b = sid_b, sid_a
            frames_a = frames_by_segment.get(sid_a, set())
            frames_b = frames_by_segment.get(sid_b, set())
            exact_count = len(frames_a & frames_b)
            first_a = min(frames_a) if frames_a else int(a.get("first_frame") or 0)
            last_a = max(frames_a) if frames_a else int(a.get("last_frame") or 0)
            first_b = min(frames_b) if frames_b else int(b.get("first_frame") or 0)
            last_b = max(frames_b) if frames_b else int(b.get("last_frame") or 0)
            span = span_interval_overlap(
                first_a=first_a, last_a=last_a, first_b=first_b, last_b=last_b
            )
            same_parent = int(a["raw_track_id"]) == int(b["raw_track_id"])
            if exact_count > 0:
                sim = None
                exact_flag = True
            else:
                sim = cosine_similarity(
                    vectors_by_segment[sid_a], vectors_by_segment[sid_b]
                )
                exact_flag = False
            pairs.append(
                {
                    "segment_id_a": sid_a,
                    "segment_id_b": sid_b,
                    "raw_track_id_a": int(a["raw_track_id"]),
                    "raw_track_id_b": int(b["raw_track_id"]),
                    "same_parent_raw_track": same_parent,
                    "cosine_similarity": sim,
                    "rank": None,
                    "exact_same_frame_overlap": exact_flag,
                    "exact_frame_overlap_count": exact_count,
                    "span_overlap": bool(span),
                    "similarity_threshold": None,
                    "automatic_link_decision": None,
                    "automatic_reject_decision": None,
                    "component_assignment": None,
                    "manual_review_required": True,
                    "representation_source_a": a.get("representation_source"),
                    "representation_source_b": b.get("representation_source"),
                    "schema_version": SEGMENT_CANDIDATE_SCHEMA,
                }
            )
    ranked = [p for p in pairs if p["cosine_similarity"] is not None]
    rejected = [p for p in pairs if p["cosine_similarity"] is None]
    ranked.sort(
        key=lambda p: (
            -float(p["cosine_similarity"]),
            str(p["segment_id_a"]),
            str(p["segment_id_b"]),
        )
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked + rejected


def build_replacement_map(
    *,
    decisions: Mapping[str, Any],
    baseline: Mapping[str, Any],
    entity_by_segment: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    segs_by_raw: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for seg in segments:
        segs_by_raw[int(seg["raw_track_id"])].append(seg)
    split_parents = {
        int(t["raw_track_id"])
        for t in decisions["tracks"]
        if t["decision"] == "manual_split_candidate"
    }
    rows: list[dict[str, Any]] = []
    for tid in sorted(baseline["vectors_by_track"]):
        replacements = segs_by_raw.get(tid, [])
        replacement_ids = [str(s["segment_id"]) for s in replacements]
        statuses = [
            str(entity_by_segment[sid]["representation_status"])
            if sid in entity_by_segment
            else "missing"
            for sid in replacement_ids
        ]
        if tid in split_parents:
            used = False
            complete = all(
                entity_by_segment.get(sid, {}).get("embedding_available")
                or entity_by_segment.get(sid, {}).get("representation_status")
                == STATUS_NO_CROP
                for sid in replacement_ids
            )
            baseline_status = "retired_mixed_baseline_embedding"
        else:
            used = any(
                entity_by_segment.get(sid, {}).get("representation_source")
                == "reused_baseline_raw_track_embedding"
                for sid in replacement_ids
            )
            complete = used and len(replacement_ids) == 1
            baseline_status = "reused_as_full_track_segment"
        rows.append(
            {
                "raw_track_id": tid,
                "baseline_embedding_available": True,
                "baseline_entity_retained_in_baseline_artifacts": True,
                "baseline_entity_used_in_segmented_view": used,
                "baseline_representation_status": baseline_status,
                "replacement_segment_ids": replacement_ids,
                "replacement_embedding_statuses": statuses,
                "replacement_complete": complete,
                "old_embedding_used_in_segmented_view": used and tid not in split_parents,
                "proven_identity_mapping": False,
                "global_component_inherited": False,
                "schema_version": REPLACEMENT_SCHEMA,
            }
        )
    return rows


def build_pair_deltas(
    *,
    baseline_pairs: Mapping[tuple[int, int], Mapping[str, Any]],
    segment_candidates: Sequence[Mapping[str, Any]],
    entity_by_segment: Mapping[str, Mapping[str, Any]],
    split_parents: set[int],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    reused_segments = {
        sid: ent
        for sid, ent in entity_by_segment.items()
        if ent.get("representation_source") == "reused_baseline_raw_track_embedding"
        and ent.get("embedding_available")
    }
    for (a, b), brow in sorted(baseline_pairs.items()):
        if a in split_parents or b in split_parents:
            continue
        seg_a = f"raw_{a}_full"
        seg_b = f"raw_{b}_full"
        if seg_a not in reused_segments or seg_b not in reused_segments:
            continue
        sid_x, sid_y = (seg_a, seg_b) if seg_a < seg_b else (seg_b, seg_a)
        match = next(
            (
                c
                for c in segment_candidates
                if c["segment_id_a"] == sid_x and c["segment_id_b"] == sid_y
            ),
            None,
        )
        base_sim = brow.get("cosine_similarity")
        if match is None or match.get("cosine_similarity") is None or base_sim is None:
            raise SegmentRegressionError(
                f"unaffected reused pair missing segmented candidate: {a},{b}"
            )
        seg_sim = float(match["cosine_similarity"])
        if abs(seg_sim - float(base_sim)) > _SIM_ATOL:
            raise SegmentRegressionError(
                f"unaffected pair similarity mismatch for ({a},{b}): "
                f"baseline={base_sim} segmented={seg_sim}"
            )
        deltas.append(
            {
                "delta_kind": "unaffected_reused_pair",
                "baseline_track_id_a": a,
                "baseline_track_id_b": b,
                "baseline_cosine_similarity": float(base_sim),
                "segment_id_a": sid_x,
                "segment_id_b": sid_y,
                "segment_cosine_similarity": seg_sim,
                "similarity_match": True,
                "accuracy_claimed": False,
            }
        )

    for (a, b), brow in sorted(baseline_pairs.items()):
        if a not in split_parents and b not in split_parents:
            continue
        related = [
            c
            for c in segment_candidates
            if c.get("cosine_similarity") is not None
            and {
                int(c["raw_track_id_a"]),
                int(c["raw_track_id_b"]),
            }
            == {a, b}
            and not c["same_parent_raw_track"]
        ]
        sims = [float(c["cosine_similarity"]) for c in related]
        deltas.append(
            {
                "delta_kind": "affected_baseline_pair",
                "baseline_track_id_a": a,
                "baseline_track_id_b": b,
                "baseline_cosine_similarity": (
                    float(brow["cosine_similarity"])
                    if brow.get("cosine_similarity") is not None
                    else None
                ),
                "replacement_segment_pair_count": len(related),
                "replacement_segment_pairs": [
                    {
                        "segment_id_a": c["segment_id_a"],
                        "segment_id_b": c["segment_id_b"],
                        "cosine_similarity": float(c["cosine_similarity"]),
                        "rank": c.get("rank"),
                    }
                    for c in related
                ],
                "replacement_similarity_min": min(sims) if sims else None,
                "replacement_similarity_max": max(sims) if sims else None,
                "replacement_similarity_median": (
                    float(np.median(np.asarray(sims, dtype=np.float64))) if sims else None
                ),
                "baseline_similarity_overwritten": False,
                "accuracy_claimed": False,
            }
        )

    for c in segment_candidates:
        if not c["same_parent_raw_track"] or c.get("cosine_similarity") is None:
            continue
        deltas.append(
            {
                "delta_kind": "same_parent_segment_pair",
                "raw_track_id": int(c["raw_track_id_a"]),
                "segment_id_a": c["segment_id_a"],
                "segment_id_b": c["segment_id_b"],
                "cosine_similarity": float(c["cosine_similarity"]),
                "rank": c.get("rank"),
                "automatic_merge": False,
                "proven_physical_identity": False,
                "accuracy_claimed": False,
            }
        )
    return deltas


def _entity_record(
    *,
    item: Mapping[str, Any],
    representation_source: str,
    representation_status: str,
    embedding_available: bool,
    no_embedding_reason: str | None,
    crop_count: int,
    crop_ids: list[str],
    vector: np.ndarray | None,
    checkpoint_path: str,
    checkpoint_sha256: str,
    first_frame: int | None,
    last_frame: int | None,
    parent_retired: bool,
) -> dict[str, Any]:
    return {
        "segment_id": item["segment_id"],
        "raw_track_id": int(item["raw_track_id"]),
        "segment_kind": item["segment_kind"],
        "segment_index": item.get("segment_index"),
        "representation_source": representation_source,
        "representation_status": representation_status,
        "embedding_available": embedding_available,
        "no_embedding_reason": no_embedding_reason,
        "crop_count": crop_count,
        "crop_ids": crop_ids,
        "embedding_dimension": EMBEDDING_DIM if embedding_available else None,
        "embedding_sha256": (
            embedding_vector_sha256(vector) if vector is not None else None
        ),
        "embedding_row": None,
        "aggregation": AGGREGATION_NAME if embedding_available else None,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "model_name": MODEL_NAME,
        "preprocessing": PREPROCESSING,
        "parent_baseline_embedding_available": item[
            "parent_baseline_embedding_available"
        ],
        "parent_mixed_embedding_retired": parent_retired,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "proven_physical_identity": False,
        "team_assignment": None,
        "global_id": None,
        "schema_version": SEGMENT_EMB_SCHEMA,
    }


def run_segmented_reid_regression(
    *,
    video: str | Path,
    segment_view_dir: str | Path,
    baseline_run_dir: str | Path,
    segmentation_policy: str | Path,
    segment_decisions: str | Path,
    crop_config: str | Path,
    reid_config: str | Path,
    regression_config: str | Path,
    sn_reid_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    checkpoint_sha256: str | None = None,
    expected_sn_reid_commit: str | None = None,
    open_capture: Callable[[str], Any] | None = None,
    video_size: tuple[int, int] | None = None,
    model_builder: Callable[..., Any] | None = None,
    weight_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    final_dir = Path(output_dir).expanduser().resolve()
    video_path = Path(video).expanduser().resolve()
    policy_path = Path(segmentation_policy).expanduser().resolve()
    decisions_path = Path(segment_decisions).expanduser().resolve()
    crop_cfg_path = Path(crop_config).expanduser().resolve()
    reid_cfg_path = Path(reid_config).expanduser().resolve()
    reg_cfg_path = Path(regression_config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    sn_root = Path(sn_reid_root).expanduser().resolve()

    try:
        check_output_collision(final_dir, overwrite=overwrite)
    except ReIDWritersError as exc:
        raise SegmentRegressionError(str(exc)) from exc

    if not reid_cfg_path.is_file():
        raise SegmentRegressionError(f"reid-config not found: {reid_cfg_path}")

    reg_cfg = load_regression_config(reg_cfg_path)
    policy = load_segmentation_policy(policy_path)
    decisions = load_segment_decisions(decisions_path)
    crop_cfg = load_crop_selection_config(crop_cfg_path)
    view = load_segment_view(segment_view_dir)
    validate_segment_view_against_decisions(view, decisions)
    baseline = load_baseline_artifacts(baseline_run_dir)

    expected_sha = checkpoint_sha256 or str(
        baseline["embedding_summary"]["checkpoint_sha256"]
    )
    expected_commit = expected_sn_reid_commit or str(
        baseline["embedding_summary"].get("sn_reid_commit")
    )
    if model_builder is None or weight_loader is None:
        verify_checkpoint(checkpoint_path, expected_sha256=expected_sha)
        verify_sn_reid_root(sn_root, expected_commit=expected_commit)

    if video_size is None:
        width, height, _fps, _fc = probe_video_size(video_path)
    else:
        width, height = int(video_size[0]), int(video_size[1])

    try:
        source_video = str(video_path.relative_to(Path.cwd().resolve()))
    except ValueError:
        source_video = str(video_path)

    plan = build_representation_plan(
        segments=view["segments"], decisions=decisions, baseline=baseline
    )

    frames_by_segment: dict[str, set[int]] = defaultdict(set)
    for row in view["assigned"]:
        frames_by_segment[str(row["segment_id"])].add(int(row["frame_index"]))
    unassigned_frames = {
        (int(r["raw_track_id"]), int(r["frame_index"])) for r in view["unassigned"]
    }

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_regression_dir(final_dir)

        all_crop_rows: list[dict[str, Any]] = []
        recompute_targets = [p for p in plan if p["status"] == STATUS_RECOMPUTE]
        for item in recompute_targets:
            seg = view["by_id"][item["segment_id"]]
            obs = _observations_for_segment(view["assigned"], item["segment_id"])
            for o in obs:
                key = (int(o["track_id"]), int(o["frame_index"]))
                if key in unassigned_frames:
                    raise SegmentRegressionError(
                        f"ambiguous/unassigned observation used for crop plan: {key}"
                    )
            crops, reason = select_segment_crops(
                segment=seg,
                observations=obs,
                crop_config=crop_cfg,
                video_width=width,
                video_height=height,
                source_video=source_video,
            )
            if reason is not None:
                item["status"] = STATUS_NO_CROP
                item["no_embedding_reason"] = reason
            else:
                all_crop_rows.extend(crops)

        all_crop_rows.sort(
            key=lambda r: (
                int(r["raw_track_id"]),
                int(r.get("segment_index") or 0),
                int(r["selection_rank"]),
                int(r["frame_index"]),
            )
        )
        if all_crop_rows:
            extract_crops_single_pass(
                video_path=video_path,
                selected_rows=all_crop_rows,
                output_dir=temp_dir,
                open_capture=open_capture,
            )
            validate_manifest_disk_consistency(temp_dir, all_crop_rows)

        crop_vectors_by_id: dict[str, np.ndarray] = {}
        if all_crop_rows:
            if model_builder is None or weight_loader is None:
                from football_analytics.reid.embedding import (
                    build_osnet_cpu_model,
                    load_osnet_checkpoint_weights,
                )

                model = build_osnet_cpu_model(model_name=MODEL_NAME)
                load_osnet_checkpoint_weights(model, checkpoint_path)
            else:
                model = model_builder(model_name=MODEL_NAME)
                weight_loader(model, checkpoint_path)
            tensors = []
            crop_ids_order: list[str] = []
            for row in all_crop_rows:
                abs_path = temp_dir / str(row["crop_relative_path"])
                tensors.append(load_and_preprocess_crop(abs_path))
                crop_ids_order.append(str(row["crop_id"]))
            vectors = embed_tensors(model, tensors, batch_size=8)
            for cid, vec in zip(crop_ids_order, vectors):
                crop_vectors_by_id[cid] = vec.astype(np.float32, copy=False)

        crops_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_crop_rows:
            crops_by_segment[str(row["segment_id"])].append(row)

        entities: list[dict[str, Any]] = []
        vectors_by_segment: dict[str, np.ndarray] = {}
        for item in plan:
            sid = item["segment_id"]
            rid = int(item["raw_track_id"])
            status = item["status"]
            frames = frames_by_segment.get(sid, set())
            first_f = min(frames) if frames else None
            last_f = max(frames) if frames else None
            if status == STATUS_RECOMPUTE:
                crops = crops_by_segment.get(sid, [])
                if not crops:
                    raise SegmentRegressionError(
                        f"recompute segment {sid} missing crops after selection"
                    )
                mat = np.stack(
                    [crop_vectors_by_id[str(c["crop_id"])] for c in crops], axis=0
                )
                unit = aggregate_crop_vectors(mat)
                vectors_by_segment[sid] = unit
                entities.append(
                    _entity_record(
                        item=item,
                        representation_source="recomputed_manual_segment",
                        representation_status=STATUS_RECOMPUTE,
                        embedding_available=True,
                        no_embedding_reason=None,
                        crop_count=len(crops),
                        crop_ids=[str(c["crop_id"]) for c in crops],
                        vector=unit,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_sha256=expected_sha,
                        first_frame=first_f,
                        last_frame=last_f,
                        parent_retired=True,
                    )
                )
            elif status == STATUS_NO_CROP:
                entities.append(
                    _entity_record(
                        item=item,
                        representation_source="recomputed_manual_segment",
                        representation_status=STATUS_NO_CROP,
                        embedding_available=False,
                        no_embedding_reason=item.get("no_embedding_reason"),
                        crop_count=0,
                        crop_ids=[],
                        vector=None,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_sha256=expected_sha,
                        first_frame=first_f,
                        last_frame=last_f,
                        parent_retired=True,
                    )
                )
            elif status == STATUS_REUSE:
                vec = baseline["vectors_by_track"][rid].astype(np.float32, copy=True)
                vectors_by_segment[sid] = vec
                base_row = baseline["index_by_track"][rid]
                entities.append(
                    _entity_record(
                        item=item,
                        representation_source="reused_baseline_raw_track_embedding",
                        representation_status=STATUS_REUSE,
                        embedding_available=True,
                        no_embedding_reason=None,
                        crop_count=int(base_row["crop_count"]),
                        crop_ids=list(base_row.get("crop_ids", [])),
                        vector=vec,
                        checkpoint_path=str(
                            baseline["embedding_summary"].get("checkpoint_path")
                        ),
                        checkpoint_sha256=expected_sha,
                        first_frame=first_f,
                        last_frame=last_f,
                        parent_retired=False,
                    )
                )
            elif status == STATUS_NO_BASELINE:
                entities.append(
                    _entity_record(
                        item=item,
                        representation_source="no_baseline_embedding",
                        representation_status=STATUS_NO_BASELINE,
                        embedding_available=False,
                        no_embedding_reason=(
                            "parent_raw_track_absent_from_baseline_embeddings"
                        ),
                        crop_count=0,
                        crop_ids=[],
                        vector=None,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_sha256=expected_sha,
                        first_frame=first_f,
                        last_frame=last_f,
                        parent_retired=False,
                    )
                )
            else:
                raise SegmentRegressionError(f"unknown representation status {status}")

        split_parents = {
            int(t["raw_track_id"])
            for t in decisions["tracks"]
            if t["decision"] == "manual_split_candidate"
        }
        for ent in entities:
            if (
                ent["segment_kind"] != SEGMENT_KIND_MANUAL
                and int(ent["raw_track_id"]) in split_parents
                and ent["embedding_available"]
            ):
                raise SegmentRegressionError(
                    "manual-split parent must not retain mixed embedding as full entity"
                )

        entities.sort(
            key=lambda e: (
                int(e["raw_track_id"]),
                0 if e["segment_kind"] == SEGMENT_KIND_MANUAL else 1,
                -1 if e["segment_index"] is None else int(e["segment_index"]),
                str(e["segment_id"]),
            )
        )
        ordered_npz: list[np.ndarray] = []
        ordered_ids: list[str] = []
        emb_row = 0
        for ent in entities:
            if ent["embedding_available"]:
                ent["embedding_row"] = emb_row
                ordered_npz.append(vectors_by_segment[str(ent["segment_id"])])
                ordered_ids.append(str(ent["segment_id"]))
                emb_row += 1

        entity_by_segment = {str(e["segment_id"]): e for e in entities}
        candidates = build_segment_candidates(
            entities=entities,
            vectors_by_segment=vectors_by_segment,
            frames_by_segment=frames_by_segment,
        )
        available_n = sum(1 for e in entities if e["embedding_available"])
        possible_pairs = available_n * (available_n - 1) // 2
        if len(candidates) != possible_pairs:
            raise SegmentRegressionError(
                f"candidate count {len(candidates)} != possible {possible_pairs}"
            )
        ranked_count = sum(1 for c in candidates if c.get("rank") is not None)
        exact_reject = sum(1 for c in candidates if c["exact_same_frame_overlap"])
        same_parent = sum(
            1
            for c in candidates
            if c["same_parent_raw_track"] and c.get("cosine_similarity") is not None
        )

        replacement = build_replacement_map(
            decisions=decisions,
            baseline=baseline,
            entity_by_segment=entity_by_segment,
            segments=view["segments"],
        )
        deltas = build_pair_deltas(
            baseline_pairs=baseline["candidate_pair_map"],
            segment_candidates=candidates,
            entity_by_segment=entity_by_segment,
            split_parents=split_parents,
        )
        unaffected = [d for d in deltas if d["delta_kind"] == "unaffected_reused_pair"]
        affected = [d for d in deltas if d["delta_kind"] == "affected_baseline_pair"]
        same_parent_deltas = [
            d for d in deltas if d["delta_kind"] == "same_parent_segment_pair"
        ]

        component_audit = {
            "raw_component": [231, 635],
            "unchanged": True,
            "component_inheritance_performed": False,
            "automatic_segment_link_performed": False,
            "baseline_pair": {
                "track_id_a": 231,
                "track_id_b": 635,
                "present_in_baseline": (231, 635) in baseline["candidate_pair_map"],
            },
            "segmented_pairs": [],
            "raw_231_s01_inherits_component": False,
            "raw_231_s02_automatically_links_to_635": False,
            "future_manual_review_required": True,
        }
        for sid_a, sid_b in (
            ("raw_231_s01", "raw_635_full"),
            ("raw_231_s02", "raw_635_full"),
        ):
            x, y = (sid_a, sid_b) if sid_a < sid_b else (sid_b, sid_a)
            match = next(
                (
                    c
                    for c in candidates
                    if c["segment_id_a"] == x and c["segment_id_b"] == y
                ),
                None,
            )
            component_audit["segmented_pairs"].append(
                {
                    "segment_id_a": sid_a,
                    "segment_id_b": sid_b,
                    "embedding_available_a": bool(
                        entity_by_segment.get(sid_a, {}).get("embedding_available")
                    ),
                    "embedding_available_b": bool(
                        entity_by_segment.get(sid_b, {}).get("embedding_available")
                    ),
                    "cosine_similarity": (
                        None if match is None else match.get("cosine_similarity")
                    ),
                    "rank": None if match is None else match.get("rank"),
                    "exact_same_frame_overlap": (
                        None if match is None else match.get("exact_same_frame_overlap")
                    ),
                    "automatic_link_decision": None,
                    "component_assignment": None,
                }
            )

        elapsed = time.perf_counter() - started
        manual_segments = [
            s for s in view["segments"] if s["segment_kind"] == SEGMENT_KIND_MANUAL
        ]
        control_segments = [
            s for s in view["segments"] if s["segment_kind"] == SEGMENT_KIND_CONTROL
        ]
        preserved_segments = [
            s for s in view["segments"] if s["segment_kind"] == SEGMENT_KIND_PRESERVED
        ]
        summary = {
            "status": "ok",
            "schema_version": SUMMARY_SCHEMA,
            "source_video": source_video,
            "source_segment_view_dir": str(view["root"]),
            "source_segment_view_summary_sha256": sha256_file(view["paths"]["summary"]),
            "source_baseline_run_dir": str(baseline["root"]),
            "source_baseline_track_embeddings_sha256": sha256_file(
                baseline["paths"]["track_npz"]
            ),
            "source_segmentation_policy": str(policy_path),
            "source_segmentation_policy_sha256": sha256_file(policy_path),
            "source_segment_decisions": str(decisions_path),
            "source_segment_decisions_sha256": sha256_file(decisions_path),
            "source_crop_config": str(crop_cfg_path),
            "source_crop_config_sha256": sha256_file(crop_cfg_path),
            "source_reid_config": str(reid_cfg_path),
            "source_reid_config_sha256": sha256_file(reid_cfg_path),
            "source_regression_config": str(reg_cfg_path),
            "source_regression_config_sha256": sha256_file(reg_cfg_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": expected_sha,
            "sn_reid_root": str(sn_root),
            "sn_reid_commit": expected_commit,
            "raw_track_count": int(view["summary"].get("raw_track_count", 0)),
            "derived_segment_count": len(view["segments"]),
            "manual_split_segment_count": len(manual_segments),
            "no_split_control_segment_count": len(control_segments),
            "preserved_full_segment_count": len(preserved_segments),
            "unassigned_observation_count": len(view["unassigned"]),
            "baseline_embedded_raw_track_count": len(baseline["vectors_by_track"]),
            "retired_mixed_baseline_embedding_count": len(split_parents),
            "reused_baseline_segment_embedding_count": sum(
                1 for e in entities if e["representation_status"] == STATUS_REUSE
            ),
            "recomputed_manual_segment_embedding_count": sum(
                1 for e in entities if e["representation_status"] == STATUS_RECOMPUTE
            ),
            "manual_segment_without_embedding_count": sum(
                1 for e in entities if e["representation_status"] == STATUS_NO_CROP
            ),
            "total_segment_embedding_count": available_n,
            "recompute_target_segment_count": len(recompute_targets),
            "segment_with_new_crop_count": len({r["segment_id"] for r in all_crop_rows}),
            "recomputed_crop_count": len(all_crop_rows),
            "ambiguous_observation_crop_count": 0,
            "parent_raw_crop_fallback_count": 0,
            "possible_pair_count": possible_pairs,
            "exact_frame_overlap_reject_count": exact_reject,
            "ranked_candidate_count": ranked_count,
            "same_parent_raw_track_candidate_count": same_parent,
            "similarity_threshold": None,
            "automatic_link_count": 0,
            "automatic_reject_count": 0,
            "component_building_count": 0,
            "unaffected_baseline_pair_count": len(unaffected),
            "unaffected_pair_similarity_match_count": len(unaffected),
            "unaffected_pair_similarity_mismatch_count": 0,
            "affected_baseline_pair_count": len(affected),
            "affected_segment_pair_count": sum(
                int(d["replacement_segment_pair_count"]) for d in affected
            ),
            "same_parent_segment_pair_count": len(same_parent_deltas),
            "existing_component_audit": component_audit,
            "raw_tracks_mutated": False,
            "segment_view_mutated": False,
            "baseline_artifacts_mutated": False,
            "global_id_rewrite_performed": False,
            "team_assignment_performed": False,
            "automatic_split_performed": False,
            "automatic_merge_performed": False,
            "automatic_link_performed": False,
            "automatic_reject_performed": False,
            "identity_ground_truth_available": False,
            "accuracy_claimed": False,
            "limitations": [
                "manual segments are visual-review decisions, not identity GT",
                "reused embeddings preserve baseline behavior but do not prove track purity",
                "recomputed segment embeddings use selected crops, not every observation",
                "OSNet checkpoint is Market1501 general-person ReID, not a validated SoccerNet football identity model",
                "ranking changes are not accuracy improvements without labeled GT",
            ],
            "elapsed_sec": elapsed,
            "regression_config_stage_status": reg_cfg["stage_status"],
            "policy_stage_status": policy.get("stage_status"),
        }

        write_jsonl(temp_dir / SEGMENT_CROP_MANIFEST_NAME, all_crop_rows)
        write_jsonl(temp_dir / SEGMENT_EMB_INDEX_NAME, entities)
        if ordered_npz:
            np.savez(
                temp_dir / SEGMENT_EMB_NPZ_NAME,
                vectors=np.stack(ordered_npz, axis=0).astype(np.float32, copy=False),
                segment_ids=np.asarray(ordered_ids),
            )
        else:
            np.savez(
                temp_dir / SEGMENT_EMB_NPZ_NAME,
                vectors=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
                segment_ids=np.asarray([], dtype=object),
            )
        write_jsonl(temp_dir / SEGMENT_CANDIDATES_NAME, candidates)
        write_jsonl(temp_dir / REPLACEMENT_MAP_NAME, replacement)
        write_jsonl(temp_dir / PAIR_DELTAS_NAME, deltas)
        write_json(temp_dir / SUMMARY_NAME, summary)

        expected_names = {
            SEGMENT_CROP_MANIFEST_NAME,
            SEGMENT_EMB_INDEX_NAME,
            SEGMENT_EMB_NPZ_NAME,
            SEGMENT_CANDIDATES_NAME,
            REPLACEMENT_MAP_NAME,
            PAIR_DELTAS_NAME,
            SUMMARY_NAME,
            CROPS_DIRNAME,
        }
        actual_names = {p.name for p in temp_dir.iterdir()}
        if actual_names != expected_names:
            raise SegmentRegressionError(
                f"unexpected temp outputs: {sorted(actual_names)}"
            )

        finalized = finalize_regression_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        **summary,
        "output_dir": str(finalized),
        "summary_path": str(finalized / SUMMARY_NAME),
    }
