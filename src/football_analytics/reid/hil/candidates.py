"""Candidate manifest schema for HIL recovery review."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import (
    HilValidationError,
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
    reject_mutable_runtime_leak,
    require_bool,
    require_int,
    require_mapping,
    require_sha256,
    require_sha256_or_none,
    require_str,
    validate_no_path_traversal,
)
from football_analytics.reid.schema import validate_bbox_xyxy

CANDIDATE_MANIFEST_SCHEMA_VERSION = "target_recovery_candidate_manifest_v1"


class CandidateManifestError(HilValidationError):
    """Raised when a candidate manifest fails validation."""


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CandidateManifestError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateManifestError(f"{field} must be a number") from exc
    if number != number:  # NaN
        raise CandidateManifestError(f"{field} must be finite")
    return number


def validate_candidate_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    prefix = f"candidates[{index}]"
    data = require_mapping(row, field=prefix)
    reject_mutable_runtime_leak(data, field=prefix)

    # Scores are similarity metadata only — never probabilities.
    forbidden_prob_keys = {"probability", "prob", "p_target", "confidence_probability"}
    leaked = sorted(set(data) & forbidden_prob_keys)
    if leaked:
        raise CandidateManifestError(
            f"{prefix} must not store scores as probability fields: {leaked}"
        )

    candidate_id = require_str(data.get("candidate_id"), field=f"{prefix}.candidate_id")
    segment_id = require_str(data.get("segment_id"), field=f"{prefix}.segment_id")
    raw_track_id = require_str(data.get("raw_track_id"), field=f"{prefix}.raw_track_id")
    start_frame = require_int(data.get("start_frame"), field=f"{prefix}.start_frame", min_value=0)
    middle_frame = require_int(data.get("middle_frame"), field=f"{prefix}.middle_frame", min_value=0)
    end_frame = require_int(data.get("end_frame"), field=f"{prefix}.end_frame", min_value=0)
    if not (start_frame <= middle_frame <= end_frame):
        raise CandidateManifestError(
            f"{prefix} requires start_frame <= middle_frame <= end_frame"
        )

    bbox_refs = data.get("bbox_references", [])
    if not isinstance(bbox_refs, list):
        raise CandidateManifestError(f"{prefix}.bbox_references must be a list")
    normalized_bboxes: list[dict[str, Any]] = []
    for j, ref in enumerate(bbox_refs):
        ref_map = require_mapping(ref, field=f"{prefix}.bbox_references[{j}]")
        frame_index = require_int(
            ref_map.get("frame_index"),
            field=f"{prefix}.bbox_references[{j}].frame_index",
            min_value=0,
        )
        bbox = validate_bbox_xyxy(ref_map.get("bbox_xyxy"), field=f"{prefix}.bbox_references[{j}].bbox_xyxy")
        normalized_bboxes.append({"frame_index": frame_index, "bbox_xyxy": bbox})

    crop_path = data.get("crop_path")
    crop_sha256 = data.get("crop_sha256")
    if crop_path is not None:
        crop_path = validate_no_path_traversal(crop_path, field=f"{prefix}.crop_path")
        crop_sha256 = require_sha256(crop_sha256, field=f"{prefix}.crop_sha256")
    else:
        crop_sha256 = require_sha256_or_none(crop_sha256, field=f"{prefix}.crop_sha256")

    context_paths = data.get("context_paths", {})
    if not isinstance(context_paths, dict):
        raise CandidateManifestError(f"{prefix}.context_paths must be a mapping")
    for key, path in context_paths.items():
        validate_no_path_traversal(path, field=f"{prefix}.context_paths.{key}")

    context_shas = data.get("context_sha256", {})
    if not isinstance(context_shas, dict):
        raise CandidateManifestError(f"{prefix}.context_sha256 must be a mapping")
    for key, digest in context_shas.items():
        require_sha256(digest, field=f"{prefix}.context_sha256.{key}")

    short_clip_path = data.get("short_clip_path")
    short_clip_sha256 = data.get("short_clip_sha256")
    if short_clip_path is not None:
        short_clip_path = validate_no_path_traversal(
            short_clip_path, field=f"{prefix}.short_clip_path"
        )
        short_clip_sha256 = require_sha256(
            short_clip_sha256, field=f"{prefix}.short_clip_sha256"
        )

    team_evidence = require_mapping(data.get("team_evidence", {}), field=f"{prefix}.team_evidence")
    if team_evidence.get("is_identity_proof") is True:
        raise CandidateManifestError(
            f"{prefix}.team_evidence must not claim identity proof"
        )

    visibility = require_mapping(data.get("visibility", {}), field=f"{prefix}.visibility")
    quality = require_mapping(data.get("quality", {}), field=f"{prefix}.quality")
    contamination = require_mapping(
        data.get("contamination", {}), field=f"{prefix}.contamination"
    )

    model_id = data.get("sportsreid_model_id")
    checkpoint_sha = data.get("sportsreid_checkpoint_sha256")
    appearance_rank = data.get("appearance_rank")
    t_max = _optional_float(data.get("T_max"), field=f"{prefix}.T_max")
    d_max = _optional_float(data.get("D_max"), field=f"{prefix}.D_max")
    score_s = _optional_float(data.get("S"), field=f"{prefix}.S")

    if appearance_rank is not None or t_max is not None or d_max is not None or score_s is not None:
        model_id = require_str(model_id, field=f"{prefix}.sportsreid_model_id")
        if model_id != SPORTSREID_MODEL_ID:
            raise CandidateManifestError(
                f"{prefix}.sportsreid_model_id must be {SPORTSREID_MODEL_ID} "
                f"(no Market1501/R2B fallback); got {model_id!r}"
            )
        checkpoint_sha = require_sha256(
            checkpoint_sha, field=f"{prefix}.sportsreid_checkpoint_sha256"
        )
        if checkpoint_sha != SPORTSREID_CHECKPOINT_SHA256:
            raise CandidateManifestError(
                f"{prefix}.sportsreid_checkpoint_sha256 mismatch for SportsReID helper"
            )
        appearance_rank = require_int(
            appearance_rank, field=f"{prefix}.appearance_rank", min_value=1
        )

    temporal_distance = _optional_float(
        data.get("temporal_distance"), field=f"{prefix}.temporal_distance"
    )
    spatial_distance = _optional_float(
        data.get("spatial_distance"), field=f"{prefix}.spatial_distance"
    )

    eligibility = require_bool(data.get("eligibility"), field=f"{prefix}.eligibility")
    rejection_reason = data.get("rejection_reason")
    if rejection_reason is not None:
        rejection_reason = require_str(
            rejection_reason, field=f"{prefix}.rejection_reason", allow_empty=True
        )
    if not eligibility and not rejection_reason:
        raise CandidateManifestError(
            f"{prefix} ineligible candidates require rejection_reason"
        )

    display_order = require_int(
        data.get("display_order"), field=f"{prefix}.display_order", min_value=1
    )

    return {
        "candidate_id": candidate_id,
        "segment_id": segment_id,
        "raw_track_id": raw_track_id,
        "start_frame": start_frame,
        "middle_frame": middle_frame,
        "end_frame": end_frame,
        "bbox_references": normalized_bboxes,
        "crop_path": crop_path,
        "crop_sha256": crop_sha256,
        "context_paths": dict(context_paths),
        "context_sha256": {k: str(v).lower() for k, v in context_shas.items()},
        "short_clip_path": short_clip_path,
        "short_clip_sha256": short_clip_sha256,
        "team_evidence": dict(team_evidence),
        "visibility": dict(visibility),
        "quality": dict(quality),
        "contamination": dict(contamination),
        "sportsreid_model_id": model_id,
        "sportsreid_checkpoint_sha256": checkpoint_sha,
        "appearance_rank": appearance_rank,
        "T_max": t_max,
        "D_max": d_max,
        "S": score_s,
        "temporal_distance": temporal_distance,
        "spatial_distance": spatial_distance,
        "eligibility": eligibility,
        "rejection_reason": rejection_reason,
        "display_order": display_order,
        "score_semantics": "similarity_margin_not_probability",
    }


def validate_candidate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = require_mapping(payload, field="candidate_manifest")
    reject_mutable_runtime_leak(data, field="candidate_manifest")

    if (
        require_str(data.get("schema_version"), field="schema_version")
        != CANDIDATE_MANIFEST_SCHEMA_VERSION
    ):
        raise CandidateManifestError(
            f"schema_version must be {CANDIDATE_MANIFEST_SCHEMA_VERSION}"
        )

    event_id = require_str(data.get("event_id"), field="event_id")
    target_id = require_str(data.get("target_id"), field="target_id")
    candidates_raw = data.get("candidates")
    if not isinstance(candidates_raw, list):
        raise CandidateManifestError("candidates must be a list")

    candidates = [validate_candidate_row(row, index=i) for i, row in enumerate(candidates_raw)]
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise CandidateManifestError("duplicate candidate_id in manifest")

    expected_count = data.get("candidate_count")
    if expected_count is None:
        expected_count = len(candidates)
    expected_count = require_int(expected_count, field="candidate_count", min_value=0)
    if expected_count != len(candidates):
        raise CandidateManifestError(
            f"candidate_count mismatch: declared={expected_count} actual={len(candidates)}"
        )

    # Rank is display helper only; eligible universe is independent of display_order.
    eligible = [c for c in candidates if c["eligibility"]]
    display_orders = [c["display_order"] for c in candidates]
    if len(display_orders) != len(set(display_orders)):
        raise CandidateManifestError("display_order values must be unique")

    supports_direct_bbox = require_bool(
        data.get("supports_direct_bbox_selection", True),
        field="supports_direct_bbox_selection",
    )
    if not supports_direct_bbox:
        raise CandidateManifestError(
            "supports_direct_bbox_selection must be true for HIL recovery manifests"
        )

    return {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "event_id": event_id,
        "target_id": target_id,
        "candidate_count": expected_count,
        "eligible_count": len(eligible),
        "supports_direct_bbox_selection": True,
        "appearance_rank_is_helper_only": True,
        "rank_does_not_hide_candidates": True,
        "candidates": candidates,
        "metadata": dict(require_mapping(data.get("metadata", {}), field="metadata")),
    }


def candidate_ids(manifest: Mapping[str, Any]) -> set[str]:
    validated = (
        manifest
        if manifest.get("schema_version") == CANDIDATE_MANIFEST_SCHEMA_VERSION
        and "candidates" in manifest
        else validate_candidate_manifest(manifest)
    )
    return {c["candidate_id"] for c in validated["candidates"]}
