"""Stage 5B3F — non-destructive manual track segment view.

Reads raw tracking JSONL and frozen manual segment decisions, then writes a
derived segment view without mutating source observations.
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
from typing import Any, Mapping, Sequence

import yaml

from football_analytics.reid.schema import ReIDSchemaError, validate_bbox_xyxy
from football_analytics.reid.writers import ReIDWritersError, check_output_collision, cleanup_dir

SEGMENTS_NAME = "track_segments.jsonl"
SEGMENT_OBS_NAME = "segment_observations.jsonl"
UNASSIGNED_NAME = "unassigned_observations.jsonl"
SUMMARY_NAME = "segment_view_summary.json"

SEGMENT_SCHEMA = "reid_track_segment_v1"
SEGMENT_OBS_SCHEMA = "reid_segment_observation_v1"
UNASSIGNED_SCHEMA = "reid_unassigned_segment_observation_v1"
SUMMARY_SCHEMA = "reid_manual_segment_view_summary_v1"

POLICY_SCHEMA = "reid_manual_track_segmentation_policy_v1"
DECISIONS_SCHEMA = "reid_manual_track_segment_decisions_v1"

ALLOWED_DECISIONS = frozenset(
    {"manual_split_candidate", "no_split_contamination_control"}
)
ALLOWED_BOUNDARIES = frozenset(
    {"adjacent_observations", "gap_bounded", "overlap_ambiguous"}
)
SEGMENT_KIND_MANUAL = "manual_split_segment"
SEGMENT_KIND_CONTROL = "no_split_control"
SEGMENT_KIND_PRESERVED = "preserved_full_track"


class SegmentError(RuntimeError):
    """Raised when segment-view validation or writing fails."""


def _reject_non_finite_json(value: str) -> None:
    raise SegmentError(f"non-finite JSON constant is not allowed: {value}")


def _require_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise SegmentError(f"{field} must be a bool, got {value!r}")
    if expected is not None and value is not expected:
        raise SegmentError(f"{field} must be {expected}, got {value!r}")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SegmentError(f"{field} must be an int, got {value!r}")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    value = _require_int(value, field=field)
    if value <= 0:
        raise SegmentError(f"{field} must be positive, got {value!r}")
    return value


def _require_nonneg_int(value: Any, *, field: str) -> int:
    value = _require_int(value, field=field)
    if value < 0:
        raise SegmentError(f"{field} must be >= 0, got {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_observation_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_observation_sha256(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_observation_json(obj).encode("utf-8")).hexdigest()


def create_temp_segments_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_segments_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise SegmentError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_segments_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise SegmentError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise SegmentError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_segments_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_segments_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_segments_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


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


def validate_segmentation_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SegmentError("segmentation policy must be a mapping")
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise SegmentError(
            f"segmentation policy schema_version must be {POLICY_SCHEMA}"
        )
    if payload.get("stage_status") != "visually_validated_manual_segment_plan":
        raise SegmentError(
            "segmentation policy stage_status must be "
            "visually_validated_manual_segment_plan"
        )

    raw = payload.get("raw_tracks")
    if not isinstance(raw, dict):
        raise SegmentError("segmentation policy raw_tracks must be a mapping")
    _require_bool(raw.get("immutable"), field="raw_tracks.immutable", expected=True)
    _require_bool(
        raw.get("source_observations_preserved"),
        field="raw_tracks.source_observations_preserved",
        expected=True,
    )
    _require_bool(
        raw.get("in_place_mutation_allowed"),
        field="raw_tracks.in_place_mutation_allowed",
        expected=False,
    )
    _require_bool(
        raw.get("interpolation_allowed"),
        field="raw_tracks.interpolation_allowed",
        expected=False,
    )
    _require_bool(
        raw.get("deletion_allowed"), field="raw_tracks.deletion_allowed", expected=False
    )

    seg = payload.get("segmentation")
    if not isinstance(seg, dict):
        raise SegmentError("segmentation policy segmentation must be a mapping")
    if "implementation_available" not in seg:
        raise SegmentError("segmentation.implementation_available is required")
    _require_bool(
        seg.get("implementation_available"),
        field="segmentation.implementation_available",
    )
    _require_bool(
        seg.get("non_destructive_segment_view_required"),
        field="segmentation.non_destructive_segment_view_required",
        expected=True,
    )
    _require_bool(
        seg.get("automatic_split_enabled"),
        field="segmentation.automatic_split_enabled",
        expected=False,
    )
    _require_bool(
        seg.get("automatic_boundary_selection_enabled"),
        field="segmentation.automatic_boundary_selection_enabled",
        expected=False,
    )
    _require_bool(
        seg.get("manual_decisions_required"),
        field="segmentation.manual_decisions_required",
        expected=True,
    )
    _require_bool(
        seg.get("segment_ids_preserve_raw_track_provenance"),
        field="segmentation.segment_ids_preserve_raw_track_provenance",
        expected=True,
    )
    if seg.get("frame_range_semantics") != "existing_observations_only":
        raise SegmentError(
            "segmentation.frame_range_semantics must be existing_observations_only"
        )
    _require_bool(
        seg.get("ambiguous_observations_may_be_unassigned"),
        field="segmentation.ambiguous_observations_may_be_unassigned",
        expected=True,
    )
    _require_bool(
        seg.get("missing_frames_create_observations"),
        field="segmentation.missing_frames_create_observations",
        expected=False,
    )

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise SegmentError("segmentation policy evaluation must be a mapping")
    for key in (
        "manual_visual_observation_is_ground_truth",
        "identity_switch_ground_truth_available",
        "track_purity_ground_truth_available",
        "accuracy_claim_allowed",
    ):
        _require_bool(evaluation.get(key), field=f"evaluation.{key}", expected=False)

    reid = payload.get("reid")
    if not isinstance(reid, dict):
        raise SegmentError("segmentation policy reid must be a mapping")
    _require_bool(
        reid.get("automatic_segment_merge_enabled"),
        field="reid.automatic_segment_merge_enabled",
        expected=False,
    )
    _require_bool(
        reid.get("automatic_segment_link_enabled"),
        field="reid.automatic_segment_link_enabled",
        expected=False,
    )

    global_identity = payload.get("global_identity")
    if not isinstance(global_identity, dict):
        raise SegmentError("segmentation policy global_identity must be a mapping")
    _require_bool(
        global_identity.get("global_id_rewrite_enabled"),
        field="global_identity.global_id_rewrite_enabled",
        expected=False,
    )
    _require_bool(
        global_identity.get("accepted_components_automatically_modified"),
        field="global_identity.accepted_components_automatically_modified",
        expected=False,
    )

    return dict(payload)


def load_segmentation_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise SegmentError(f"segmentation policy not found: {policy_path}")
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SegmentError(f"invalid segmentation policy YAML: {exc}") from exc
    return validate_segmentation_policy(payload)


def _validate_frame_range_pair(
    pair: Any, *, field: str
) -> tuple[int, int]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise SegmentError(f"{field} must be a [min, max] pair")
    lo = _require_nonneg_int(pair[0], field=f"{field}[0]")
    hi = _require_nonneg_int(pair[1], field=f"{field}[1]")
    if lo > hi:
        raise SegmentError(f"{field} must have min <= max, got {pair!r}")
    return lo, hi


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def validate_segment_decisions(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SegmentError("segment decisions must be a mapping")
    if payload.get("schema_version") != DECISIONS_SCHEMA:
        raise SegmentError(f"segment decisions schema_version must be {DECISIONS_SCHEMA}")
    if payload.get("status") != "visually_reviewed_plan_not_applied":
        raise SegmentError(
            "segment decisions status must be visually_reviewed_plan_not_applied"
        )
    _require_bool(
        payload.get("automatic_application_enabled"),
        field="automatic_application_enabled",
        expected=False,
    )
    _require_bool(
        payload.get("manual_visual_review_is_ground_truth"),
        field="manual_visual_review_is_ground_truth",
        expected=False,
    )
    _require_bool(
        payload.get("raw_tracks_mutated"), field="raw_tracks_mutated", expected=False
    )
    _require_bool(
        payload.get("global_id_map_changed"),
        field="global_id_map_changed",
        expected=False,
    )

    tracks = payload.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise SegmentError("segment decisions tracks must be a non-empty list")

    seen_track_ids: set[int] = set()
    seen_segment_ids: set[str] = set()
    validated_tracks: list[dict[str, Any]] = []

    for idx, item in enumerate(tracks):
        source = f"tracks[{idx}]"
        if not isinstance(item, dict):
            raise SegmentError(f"{source} must be a mapping")
        raw_track_id = _require_positive_int(item.get("raw_track_id"), field=f"{source}.raw_track_id")
        if raw_track_id in seen_track_ids:
            raise SegmentError(f"duplicate raw_track_id in decisions: {raw_track_id}")
        seen_track_ids.add(raw_track_id)

        decision = item.get("decision")
        if decision not in ALLOWED_DECISIONS:
            raise SegmentError(
                f"{source}.decision must be one of {sorted(ALLOWED_DECISIONS)}"
            )

        ambiguous_raw = item.get("ambiguous_existing_observation_frames", [])
        if not isinstance(ambiguous_raw, list):
            raise SegmentError(
                f"{source}.ambiguous_existing_observation_frames must be a list"
            )
        ambiguous: list[int] = []
        for frame in ambiguous_raw:
            ambiguous.append(_require_nonneg_int(frame, field=f"{source}.ambiguous"))
        if ambiguous != sorted(ambiguous) or len(ambiguous) != len(set(ambiguous)):
            raise SegmentError(
                f"{source}.ambiguous_existing_observation_frames must be unique ascending"
            )

        gaps_raw = item.get("unobserved_gap_ranges", [])
        if not isinstance(gaps_raw, list):
            raise SegmentError(f"{source}.unobserved_gap_ranges must be a list")
        gaps: list[tuple[int, int]] = []
        for g_idx, pair in enumerate(gaps_raw):
            gaps.append(
                _validate_frame_range_pair(pair, field=f"{source}.unobserved_gap_ranges[{g_idx}]")
            )
        gaps_sorted = sorted(gaps)
        for i in range(len(gaps_sorted) - 1):
            if _ranges_overlap(gaps_sorted[i], gaps_sorted[i + 1]):
                raise SegmentError(f"{source}.unobserved_gap_ranges must not overlap")

        segments_raw = item.get("segments", [])
        if not isinstance(segments_raw, list):
            raise SegmentError(f"{source}.segments must be a list")
        boundaries_raw = item.get("boundaries", [])
        if not isinstance(boundaries_raw, list):
            raise SegmentError(f"{source}.boundaries must be a list")

        if decision == "no_split_contamination_control":
            _require_bool(
                item.get("manual_split_candidate"),
                field=f"{source}.manual_split_candidate",
                expected=False,
            )
            _require_bool(
                item.get("raw_track_preserved"),
                field=f"{source}.raw_track_preserved",
                expected=True,
            )
            if segments_raw:
                raise SegmentError(
                    f"{source}: no_split_contamination_control must have empty segments"
                )
            if boundaries_raw:
                raise SegmentError(
                    f"{source}: no_split_contamination_control must have empty boundaries"
                )
            event_count = _require_nonneg_int(
                item.get("probable_switch_event_count", 0),
                field=f"{source}.probable_switch_event_count",
            )
            if event_count != 0:
                raise SegmentError(
                    f"{source}: no-split control probable_switch_event_count must be 0"
                )
            validated_tracks.append(
                {
                    "raw_track_id": raw_track_id,
                    "decision": decision,
                    "probable_switch_event_count": 0,
                    "manual_split_candidate": False,
                    "raw_track_preserved": True,
                    "ambiguous_existing_observation_frames": ambiguous,
                    "unobserved_gap_ranges": gaps_sorted,
                    "segments": [],
                    "boundaries": [],
                    "notes": item.get("notes"),
                }
            )
            continue

        # manual_split_candidate
        _require_bool(
            item.get("manual_split_candidate", True),
            field=f"{source}.manual_split_candidate",
            expected=True,
        )
        if len(segments_raw) < 2:
            raise SegmentError(f"{source}: split candidate requires at least 2 segments")

        segments: list[dict[str, Any]] = []
        ranges: list[tuple[int, int]] = []
        for s_idx, seg in enumerate(segments_raw):
            s_src = f"{source}.segments[{s_idx}]"
            if not isinstance(seg, dict):
                raise SegmentError(f"{s_src} must be a mapping")
            segment_id = seg.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise SegmentError(f"{s_src}.segment_id must be a non-empty string")
            if segment_id in seen_segment_ids:
                raise SegmentError(f"duplicate segment_id: {segment_id}")
            seen_segment_ids.add(segment_id)
            if int(seg.get("raw_track_id")) != raw_track_id:
                raise SegmentError(
                    f"{s_src}.raw_track_id must equal parent raw_track_id {raw_track_id}"
                )
            frame_min = _require_nonneg_int(seg.get("frame_min"), field=f"{s_src}.frame_min")
            frame_max = _require_nonneg_int(seg.get("frame_max"), field=f"{s_src}.frame_max")
            if frame_min > frame_max:
                raise SegmentError(f"{s_src}: frame_min must be <= frame_max")
            _require_bool(
                seg.get("include_existing_observations_only"),
                field=f"{s_src}.include_existing_observations_only",
                expected=True,
            )
            _require_bool(
                seg.get("proven_physical_identity"),
                field=f"{s_src}.proven_physical_identity",
                expected=False,
            )
            if seg.get("team_assignment") is not None:
                raise SegmentError(f"{s_src}.team_assignment must be null")
            if seg.get("global_id") is not None:
                raise SegmentError(f"{s_src}.global_id must be null")
            for frame in ambiguous:
                if frame_min <= frame <= frame_max:
                    raise SegmentError(
                        f"{source}: ambiguous frame {frame} falls inside segment {segment_id}"
                    )
            ranges.append((frame_min, frame_max))
            segments.append(
                {
                    "segment_id": segment_id,
                    "raw_track_id": raw_track_id,
                    "frame_min": frame_min,
                    "frame_max": frame_max,
                    "include_existing_observations_only": True,
                    "proven_physical_identity": False,
                    "team_assignment": None,
                    "global_id": None,
                    "segment_index": s_idx + 1,
                }
            )

        ranges_sorted = sorted(ranges)
        for i in range(len(ranges_sorted) - 1):
            if _ranges_overlap(ranges_sorted[i], ranges_sorted[i + 1]):
                raise SegmentError(f"{source}: segment frame ranges must not overlap")

        boundaries: list[dict[str, Any]] = []
        for b_idx, boundary in enumerate(boundaries_raw):
            b_src = f"{source}.boundaries[{b_idx}]"
            if not isinstance(boundary, dict):
                raise SegmentError(f"{b_src} must be a mapping")
            btype = boundary.get("boundary_type")
            if btype not in ALLOWED_BOUNDARIES:
                raise SegmentError(
                    f"{b_src}.boundary_type must be one of {sorted(ALLOWED_BOUNDARIES)}"
                )
            _require_bool(
                boundary.get("exact_real_world_switch_frame_known"),
                field=f"{b_src}.exact_real_world_switch_frame_known",
                expected=False,
            )
            _require_bool(
                boundary.get("automatic_split_decision"),
                field=f"{b_src}.automatic_split_decision",
                expected=False,
            )
            boundaries.append(
                {
                    "last_segment_frame": _require_nonneg_int(
                        boundary.get("last_segment_frame"),
                        field=f"{b_src}.last_segment_frame",
                    ),
                    "next_segment_frame": _require_nonneg_int(
                        boundary.get("next_segment_frame"),
                        field=f"{b_src}.next_segment_frame",
                    ),
                    "boundary_type": btype,
                    "exact_real_world_switch_frame_known": False,
                    "automatic_split_decision": False,
                }
            )

        event_count = _require_nonneg_int(
            item.get("probable_switch_event_count"),
            field=f"{source}.probable_switch_event_count",
        )
        if event_count != len(boundaries):
            raise SegmentError(
                f"{source}: probable_switch_event_count ({event_count}) must equal "
                f"boundary count ({len(boundaries)})"
            )

        validated_tracks.append(
            {
                "raw_track_id": raw_track_id,
                "decision": decision,
                "probable_switch_event_count": event_count,
                "manual_split_candidate": True,
                "ambiguous_existing_observation_frames": ambiguous,
                "unobserved_gap_ranges": gaps_sorted,
                "segments": segments,
                "boundaries": boundaries,
                "notes": item.get("notes"),
            }
        )

    # collision between full IDs and manual segment IDs checked later with raw tracks
    return {
        "schema_version": DECISIONS_SCHEMA,
        "status": "visually_reviewed_plan_not_applied",
        "automatic_application_enabled": False,
        "manual_visual_review_is_ground_truth": False,
        "raw_tracks_mutated": False,
        "global_id_map_changed": False,
        "tracks": validated_tracks,
        "source_raw_tracks": payload.get("source_raw_tracks"),
        "source_visual_review": payload.get("source_visual_review"),
    }


def load_segment_decisions(path: str | Path) -> dict[str, Any]:
    decisions_path = Path(path).expanduser().resolve()
    if not decisions_path.is_file():
        raise SegmentError(f"segment decisions not found: {decisions_path}")
    try:
        payload = yaml.safe_load(decisions_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SegmentError(f"invalid segment decisions YAML: {exc}") from exc
    return validate_segment_decisions(payload)


def load_raw_track_observations(tracks_path: str | Path) -> list[dict[str, Any]]:
    """Load every person observation with provenance; preserve full source object."""
    path = Path(tracks_path).expanduser().resolve()
    if not path.is_file():
        raise SegmentError(f"tracks JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text, parse_constant=_reject_non_finite_json)
            except SegmentError:
                raise
            except json.JSONDecodeError as exc:
                raise SegmentError(
                    f"invalid JSON on tracks line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise SegmentError(f"tracks line {line_no} must be a JSON object")

            track_id = obj.get("track_id")
            frame_index = obj.get("frame_index")
            if (
                not isinstance(track_id, int)
                or isinstance(track_id, bool)
                or track_id <= 0
            ):
                raise SegmentError(
                    f"tracks line {line_no}: invalid track_id {track_id!r}"
                )
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index < 0
            ):
                raise SegmentError(
                    f"tracks line {line_no}: invalid frame_index {frame_index!r}"
                )

            if "class_id" in obj and obj.get("class_id") != 0:
                raise SegmentError(
                    f"tracks line {line_no}: only person class_id=0 is supported"
                )
            if "class_name" in obj and obj.get("class_name") != "person":
                raise SegmentError(
                    f"tracks line {line_no}: only class_name=person is supported"
                )

            try:
                bbox = validate_bbox_xyxy(obj["bbox_xyxy"])
            except ReIDSchemaError as exc:
                raise SegmentError(f"tracks line {line_no}: {exc}") from exc
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise SegmentError(
                    f"tracks line {line_no}: bbox must have positive area"
                )
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if not math.isfinite(area) or area <= 0:
                raise SegmentError(f"tracks line {line_no}: invalid bbox area")

            # Ensure nested finite serializable object for round-trip identity.
            try:
                canonical = canonical_observation_json(obj)
                parsed_again = json.loads(canonical, parse_constant=_reject_non_finite_json)
            except (TypeError, ValueError, SegmentError) as exc:
                raise SegmentError(
                    f"tracks line {line_no}: source object is not strict-JSON serializable"
                ) from exc

            key = (track_id, frame_index)
            if key in seen_keys:
                raise SegmentError(
                    f"duplicate track_id+frame_index observation: {key}"
                )
            seen_keys.add(key)
            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "source_row_index": line_no - 1,
                    "source_observation": parsed_again,
                    "source_observation_sha256": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                }
            )

    if not rows:
        raise SegmentError("tracks JSONL is empty or has no valid observations")
    return rows


def _full_segment_id(raw_track_id: int) -> str:
    return f"raw_{raw_track_id}_full"


def build_segment_view(
    *,
    observations: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_track[int(row["track_id"])].append(dict(row))
    for track_id in by_track:
        by_track[track_id].sort(
            key=lambda r: (int(r["frame_index"]), int(r["source_row_index"]))
        )

    decision_tracks = {int(t["raw_track_id"]): t for t in decisions["tracks"]}
    for raw_track_id in decision_tracks:
        if raw_track_id not in by_track:
            raise SegmentError(
                f"decision raw_track_id {raw_track_id} missing from tracks input"
            )

    # Collision: manual segment IDs must not equal any full ID that will be created.
    all_raw_ids = set(by_track)
    for track in decisions["tracks"]:
        for seg in track["segments"]:
            full_id = _full_segment_id(int(track["raw_track_id"]))
            if seg["segment_id"] == full_id:
                raise SegmentError(
                    f"manual segment_id collides with full-track id: {full_id}"
                )
            # Also forbid colliding with other tracks' full IDs
            for rid in all_raw_ids:
                if seg["segment_id"] == _full_segment_id(rid):
                    raise SegmentError(
                        f"manual segment_id {seg['segment_id']} collides with "
                        f"derived full-track id for raw_track_id={rid}"
                    )

    segment_rows: list[dict[str, Any]] = []
    assigned_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    covered_hashes: set[str] = set()

    def mark_covered(obs: Mapping[str, Any]) -> None:
        digest = str(obs["source_observation_sha256"])
        if digest in covered_hashes:
            raise SegmentError(f"duplicate source observation coverage: {digest}")
        covered_hashes.add(digest)

    for raw_track_id in sorted(by_track):
        obs_list = by_track[raw_track_id]
        decision = decision_tracks.get(raw_track_id)

        if decision is None:
            frames = [int(o["frame_index"]) for o in obs_list]
            segment_id = _full_segment_id(raw_track_id)
            segment_rows.append(
                {
                    "schema_version": SEGMENT_SCHEMA,
                    "segment_id": segment_id,
                    "raw_track_id": raw_track_id,
                    "segment_kind": SEGMENT_KIND_PRESERVED,
                    "segment_index": None,
                    "decision": "preserved_full_track",
                    "source_decision_status": decisions["status"],
                    "configured_frame_min": frames[0],
                    "configured_frame_max": frames[-1],
                    "first_observation_frame": frames[0],
                    "last_observation_frame": frames[-1],
                    "observation_count": len(obs_list),
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
            for obs in obs_list:
                mark_covered(obs)
                assigned_rows.append(
                    {
                        "schema_version": SEGMENT_OBS_SCHEMA,
                        "segment_id": segment_id,
                        "raw_track_id": raw_track_id,
                        "segment_kind": SEGMENT_KIND_PRESERVED,
                        "segment_index": None,
                        "frame_index": int(obs["frame_index"]),
                        "source_row_index": int(obs["source_row_index"]),
                        "source_observation_sha256": obs["source_observation_sha256"],
                        "source_observation": obs["source_observation"],
                    }
                )
            continue

        if decision["decision"] == "no_split_contamination_control":
            frames = [int(o["frame_index"]) for o in obs_list]
            segment_id = _full_segment_id(raw_track_id)
            segment_rows.append(
                {
                    "schema_version": SEGMENT_SCHEMA,
                    "segment_id": segment_id,
                    "raw_track_id": raw_track_id,
                    "segment_kind": SEGMENT_KIND_CONTROL,
                    "segment_index": None,
                    "decision": "no_split_contamination_control",
                    "source_decision_status": decisions["status"],
                    "configured_frame_min": frames[0],
                    "configured_frame_max": frames[-1],
                    "first_observation_frame": frames[0],
                    "last_observation_frame": frames[-1],
                    "observation_count": len(obs_list),
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
            for obs in obs_list:
                mark_covered(obs)
                assigned_rows.append(
                    {
                        "schema_version": SEGMENT_OBS_SCHEMA,
                        "segment_id": segment_id,
                        "raw_track_id": raw_track_id,
                        "segment_kind": SEGMENT_KIND_CONTROL,
                        "segment_index": None,
                        "frame_index": int(obs["frame_index"]),
                        "source_row_index": int(obs["source_row_index"]),
                        "source_observation_sha256": obs["source_observation_sha256"],
                        "source_observation": obs["source_observation"],
                    }
                )
            continue

        # manual_split_candidate
        ambiguous_set = set(decision["ambiguous_existing_observation_frames"])
        gap_ranges = list(decision["unobserved_gap_ranges"])
        frame_to_obs = {int(o["frame_index"]): o for o in obs_list}

        for frame in ambiguous_set:
            if frame not in frame_to_obs:
                raise SegmentError(
                    f"ambiguous frame {frame} for raw_track_id={raw_track_id} "
                    "is missing from source observations"
                )

        for lo, hi in gap_ranges:
            for frame, obs in frame_to_obs.items():
                if lo <= frame <= hi:
                    raise SegmentError(
                        f"source observation frame {frame} for raw_track_id="
                        f"{raw_track_id} falls inside unobserved gap [{lo}, {hi}]"
                    )

        remaining = set(frame_to_obs)
        for frame in sorted(ambiguous_set):
            obs = frame_to_obs[frame]
            mark_covered(obs)
            remaining.discard(frame)
            unassigned_rows.append(
                {
                    "schema_version": UNASSIGNED_SCHEMA,
                    "raw_track_id": raw_track_id,
                    "frame_index": frame,
                    "reason": "manual_ambiguous_existing_observation",
                    "decision_source": "manual_track_segment_decisions",
                    "source_row_index": int(obs["source_row_index"]),
                    "source_observation_sha256": obs["source_observation_sha256"],
                    "source_observation": obs["source_observation"],
                    "deleted": False,
                    "interpolated": False,
                    "assigned_to_segment": False,
                    "team_assignment": None,
                    "global_id": None,
                }
            )

        for seg in decision["segments"]:
            members = [
                frame_to_obs[frame]
                for frame in sorted(remaining)
                if seg["frame_min"] <= frame <= seg["frame_max"]
            ]
            if not members:
                raise SegmentError(
                    f"empty segment {seg['segment_id']}: no existing observations "
                    f"in [{seg['frame_min']}, {seg['frame_max']}]"
                )
            for obs in members:
                remaining.discard(int(obs["frame_index"]))
                mark_covered(obs)
                assigned_rows.append(
                    {
                        "schema_version": SEGMENT_OBS_SCHEMA,
                        "segment_id": seg["segment_id"],
                        "raw_track_id": raw_track_id,
                        "segment_kind": SEGMENT_KIND_MANUAL,
                        "segment_index": int(seg["segment_index"]),
                        "frame_index": int(obs["frame_index"]),
                        "source_row_index": int(obs["source_row_index"]),
                        "source_observation_sha256": obs["source_observation_sha256"],
                        "source_observation": obs["source_observation"],
                    }
                )
            member_frames = [int(o["frame_index"]) for o in members]
            segment_rows.append(
                {
                    "schema_version": SEGMENT_SCHEMA,
                    "segment_id": seg["segment_id"],
                    "raw_track_id": raw_track_id,
                    "segment_kind": SEGMENT_KIND_MANUAL,
                    "segment_index": int(seg["segment_index"]),
                    "decision": "manual_split_candidate",
                    "source_decision_status": decisions["status"],
                    "configured_frame_min": int(seg["frame_min"]),
                    "configured_frame_max": int(seg["frame_max"]),
                    "first_observation_frame": member_frames[0],
                    "last_observation_frame": member_frames[-1],
                    "observation_count": len(members),
                    "ambiguous_observation_count_excluded": len(ambiguous_set),
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

        if remaining:
            raise SegmentError(
                f"uncovered observations for split raw_track_id={raw_track_id}: "
                f"{sorted(remaining)}"
            )

    # Sort outputs
    def segment_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
        kind = row["segment_kind"]
        if kind == SEGMENT_KIND_MANUAL:
            return (int(row["raw_track_id"]), 0, int(row["segment_index"]))
        # full segments after manuals for same track
        return (int(row["raw_track_id"]), 1, 0)

    segment_rows.sort(key=segment_sort_key)
    assigned_rows.sort(
        key=lambda r: (
            int(r["raw_track_id"]),
            0 if r["segment_kind"] == SEGMENT_KIND_MANUAL else 1,
            -1 if r["segment_index"] is None else int(r["segment_index"]),
            int(r["frame_index"]),
            int(r["source_row_index"]),
        )
    )
    unassigned_rows.sort(
        key=lambda r: (
            int(r["raw_track_id"]),
            int(r["frame_index"]),
            int(r["source_row_index"]),
        )
    )

    # Integrity
    source_hashes = {str(o["source_observation_sha256"]) for o in observations}
    if covered_hashes != source_hashes:
        missing = source_hashes - covered_hashes
        extra = covered_hashes - source_hashes
        raise SegmentError(
            f"source coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    if len(assigned_rows) + len(unassigned_rows) != len(observations):
        raise SegmentError("assigned+unassigned count must equal source observation count")

    assigned_hashes = {r["source_observation_sha256"] for r in assigned_rows}
    unassigned_hashes = {r["source_observation_sha256"] for r in unassigned_rows}
    if assigned_hashes & unassigned_hashes:
        raise SegmentError("assigned and unassigned observation hashes intersect")

    segment_ids = {r["segment_id"] for r in segment_rows}
    for row in assigned_rows:
        if row["segment_id"] not in segment_ids:
            raise SegmentError(
                f"segment_observation references unknown segment_id {row['segment_id']}"
            )
        nested = row["source_observation"]
        if int(nested["track_id"]) != int(row["raw_track_id"]):
            raise SegmentError("nested source track_id mismatch")
        if int(nested["frame_index"]) != int(row["frame_index"]):
            raise SegmentError("nested source frame_index mismatch")
        if "segment_id" in nested:
            raise SegmentError("source_observation must not contain injected segment_id")

    for row in unassigned_rows:
        if "segment_id" in row:
            raise SegmentError("unassigned row must not include segment_id")

    split_count = sum(
        1 for t in decisions["tracks"] if t["decision"] == "manual_split_candidate"
    )
    control_count = sum(
        1
        for t in decisions["tracks"]
        if t["decision"] == "no_split_contamination_control"
    )
    manual_segment_count = sum(
        1 for r in segment_rows if r["segment_kind"] == SEGMENT_KIND_MANUAL
    )
    control_segment_count = sum(
        1 for r in segment_rows if r["segment_kind"] == SEGMENT_KIND_CONTROL
    )
    preserved_segment_count = sum(
        1 for r in segment_rows if r["segment_kind"] == SEGMENT_KIND_PRESERVED
    )

    component = policy.get("existing_component_audit", {}).get("raw_component")
    if not isinstance(component, list):
        component = []

    summary = {
        "status": "ok",
        "schema_version": SUMMARY_SCHEMA,
        "raw_track_count": len(by_track),
        "raw_observation_count": len(observations),
        "decision_track_count": len(decisions["tracks"]),
        "split_candidate_raw_track_count": split_count,
        "no_split_control_raw_track_count": control_count,
        "preserved_full_raw_track_count": len(by_track) - len(decision_tracks),
        "manual_split_segment_count": manual_segment_count,
        "no_split_control_segment_count": control_segment_count,
        "preserved_full_segment_count": preserved_segment_count,
        "total_segment_count": len(segment_rows),
        "assigned_observation_count": len(assigned_rows),
        "unassigned_observation_count": len(unassigned_rows),
        "source_coverage_observation_count": len(assigned_rows) + len(unassigned_rows),
        "created_observation_count": 0,
        "interpolated_observation_count": 0,
        "deleted_observation_count": 0,
        "duplicated_source_observation_count": 0,
        "uncovered_source_observation_count": 0,
        "raw_tracks_mutated": False,
        "source_observations_preserved": True,
        "automatic_split_performed": False,
        "automatic_merge_performed": False,
        "automatic_link_performed": False,
        "global_id_rewrite_performed": False,
        "team_assignment_performed": False,
        "frame_range_semantics": "existing_observations_only",
        "ambiguous_observations_are_unassigned": True,
        "segment_view_is_raw_track_replacement": False,
        "derived_view_only": True,
        "proven_physical_identity": False,
        "manual_visual_review_is_ground_truth": False,
        "accuracy_claimed": False,
        "existing_component_audit": {
            "raw_component": list(component),
            "modified": False,
            "segment_component_assignment_performed": False,
        },
        "limitations": [
            "manual segment boundaries are visual-review decisions, not identity-switch ground truth",
            "unassigned observations remain in raw tracking source",
            "pass-through tracks are not proven pure",
            "derived segments do not automatically inherit global identities",
            "no crop, embedding or ReID recomputation was performed",
        ],
    }

    return {
        "track_segments": segment_rows,
        "segment_observations": assigned_rows,
        "unassigned_observations": unassigned_rows,
        "summary": summary,
    }


def run_build_manual_track_segment_view(
    *,
    tracks: str | Path,
    segmentation_policy: str | Path,
    segment_decisions: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    tracks_path = Path(tracks).expanduser().resolve()
    policy_path = Path(segmentation_policy).expanduser().resolve()
    decisions_path = Path(segment_decisions).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()

    try:
        check_output_collision(final_dir, overwrite=overwrite)
    except ReIDWritersError as exc:
        raise SegmentError(str(exc)) from exc

    policy = load_segmentation_policy(policy_path)
    decisions = load_segment_decisions(decisions_path)
    observations = load_raw_track_observations(tracks_path)
    source_tracks_sha = _sha256_file(tracks_path)
    source_policy_sha = _sha256_file(policy_path)
    source_decisions_sha = _sha256_file(decisions_path)

    built = build_segment_view(
        observations=observations, decisions=decisions, policy=policy
    )
    elapsed = time.perf_counter() - started

    summary = dict(built["summary"])
    summary.update(
        {
            "source_tracks": str(tracks_path),
            "source_segmentation_policy": str(policy_path),
            "source_segment_decisions": str(decisions_path),
            "source_tracks_sha256": source_tracks_sha,
            "source_segmentation_policy_sha256": source_policy_sha,
            "source_segment_decisions_sha256": source_decisions_sha,
            "elapsed_sec": elapsed,
        }
    )

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_segments_dir(final_dir)
        write_jsonl(temp_dir / SEGMENTS_NAME, built["track_segments"])
        write_jsonl(temp_dir / SEGMENT_OBS_NAME, built["segment_observations"])
        write_jsonl(temp_dir / UNASSIGNED_NAME, built["unassigned_observations"])
        write_json(temp_dir / SUMMARY_NAME, summary)

        # Final integrity on written payloads already validated in memory.
        names = sorted(p.name for p in temp_dir.iterdir() if p.is_file())
        expected = sorted(
            [SEGMENTS_NAME, SEGMENT_OBS_NAME, UNASSIGNED_NAME, SUMMARY_NAME]
        )
        if names != expected:
            raise SegmentError(f"unexpected temp outputs: {names}")

        finalized = finalize_segments_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        **summary,
        "output_dir": str(finalized),
        "segments_path": str(finalized / SEGMENTS_NAME),
        "segment_observations_path": str(finalized / SEGMENT_OBS_NAME),
        "unassigned_path": str(finalized / UNASSIGNED_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "reid_recomputation_performed": False,
    }
