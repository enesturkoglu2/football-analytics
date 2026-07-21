"""Manually approved ReID track linking (Stage 4B-5C2).

Cosine similarity is audit/ranking only. No automatic threshold acceptance.
Components require an explicit full-clique of manual approvals; uncontrolled
Union-Find chaining is forbidden.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from football_analytics.reid.aggregate import (
    AggregateError,
    build_track_observation_stats,
    load_tracks_jsonl,
)
from football_analytics.reid.candidates import (
    DECISION_ELIGIBLE,
    DECISION_REJECTED,
    REASON_EXACT_FRAME,
    REASON_THRESHOLD_PENDING,
    span_interval_overlap,
    temporal_gap_frames,
)
from football_analytics.reid.writers import (
    check_output_collision,
    cleanup_dir,
    write_manifest_jsonl,
)

DEFAULT_POLICY = "configs/reid/linking_policy_stage4b.yaml"

POLICY_SCHEMA = "reid_linking_policy_v1"
MANUAL_SCHEMA = "reid_manual_pair_decision_v1"
ACCEPTED_EDGE_SCHEMA = "reid_accepted_edge_v1"
AUDIT_SCHEMA = "reid_linking_audit_v1"
GLOBAL_MAP_SCHEMA = "reid_global_id_map_v1"
SUMMARY_SCHEMA = "reid_linking_summary_v1"
CANDIDATE_SCHEMA = "reid_candidate_pair_v1"
TRACK_EMBED_SCHEMA = "reid_track_embedding_v1"

ACCEPTED_NAME = "accepted_edges.jsonl"
AUDIT_NAME = "linking_audit.jsonl"
GLOBAL_MAP_NAME = "global_id_map.jsonl"
SUMMARY_NAME = "linking_summary.json"

REVIEW_LABELS = frozenset(
    {
        "likely_same",
        "likely_different",
        "uncertain",
        "rejected_exact_frame_conflict",
    }
)

_NORM_ATOL = 1e-4


class LinkingError(RuntimeError):
    """Raised when controlled ReID linking fails validation or policy checks."""


def _reject_non_finite_json(value: str) -> None:
    raise LinkingError(f"NaN/Infinity forbidden in JSON ({value})")


def _ensure_finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LinkingError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise LinkingError(f"{field} must be finite, got {value!r}")
    return number


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _canonical_pair(track_a: int, track_b: int) -> tuple[int, int]:
    if track_a == track_b:
        raise LinkingError(f"self-pair is not allowed: {track_a}")
    return (track_a, track_b) if track_a < track_b else (track_b, track_a)


def load_linking_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise LinkingError(f"linking policy not found: {policy_path}")
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LinkingError(f"invalid linking policy YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise LinkingError("linking policy must be a mapping")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise LinkingError(message)

    require(payload.get("schema_version") == POLICY_SCHEMA, "policy schema_version mismatch")
    require(payload.get("automatic_linking_enabled") is False, "automatic_linking_enabled must be false")
    require(payload.get("similarity_threshold") is None, "similarity_threshold must be null")
    require(payload.get("cosine_usage") == "ranking_only", "cosine_usage must be ranking_only")

    temporal = payload.get("temporal")
    require(isinstance(temporal, dict), "policy.temporal missing")
    require(temporal.get("exact_frame_conflict_hard_reject") is True, "exact_frame hard reject required")
    require(temporal.get("span_overlap_hard_reject") is False, "span_overlap hard reject must be false")
    require(
        temporal.get("require_component_cross_member_exact_frame_check") is True,
        "component cross-member exact-frame check required",
    )

    decisions = payload.get("decisions")
    require(isinstance(decisions, dict), "policy.decisions missing")
    require(
        decisions.get("manual_acceptance_required_for_linking") is True,
        "manual acceptance required",
    )
    require(decisions.get("uncertain_pair_linking_allowed") is False, "uncertain linking forbidden")
    require(
        decisions.get("likely_different_pair_linking_allowed") is False,
        "likely_different linking forbidden",
    )
    require(
        decisions.get("exact_conflict_pair_linking_allowed") is False,
        "exact-conflict linking forbidden",
    )

    components = payload.get("component_rules")
    require(isinstance(components, dict), "policy.component_rules missing")
    require(
        components.get("accepted_edge_does_not_bypass_component_conflict_check") is True,
        "accepted edges must not bypass conflict checks",
    )
    require(
        components.get("all_cross_member_pairs_must_be_exact_frame_conflict_free") is True,
        "cross-member conflict-free requirement missing",
    )
    require(
        components.get("uncontrolled_transitive_chaining_allowed") is False,
        "uncontrolled chaining must be false",
    )

    global_id = payload.get("global_id")
    require(isinstance(global_id, dict), "policy.global_id missing")
    require(
        global_id.get("deterministic_policy") == "minimum_raw_track_id",
        "global_id policy must be minimum_raw_track_id",
    )
    require(global_id.get("preserve_raw_track_id_mapping") is True, "raw track mapping required")
    require(global_id.get("include_all_raw_tracks") is True, "all raw tracks required")
    require(
        global_id.get("no_embedding_tracks_remain_singleton") is True,
        "no-embedding tracks must remain singleton",
    )

    evidence = payload.get("evidence")
    require(isinstance(evidence, dict), "policy.evidence missing")
    require(
        _is_positive_int(evidence.get("minimum_crop_count_per_track_for_strong_review")),
        "invalid minimum crop count",
    )
    require(
        _is_positive_int(evidence.get("minimum_observation_count_per_track_for_strong_review")),
        "invalid minimum observation count",
    )

    payload["_path"] = str(policy_path)
    return payload


def _load_jsonl(path: Path, *, allow_empty: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise LinkingError(f"JSONL not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        if allow_empty:
            return []
        raise LinkingError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_non_finite_json)
        except LinkingError:
            raise
        except json.JSONDecodeError as exc:
            raise LinkingError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise LinkingError(f"line {line_no} of {path} must be a JSON object")
        rows.append(obj)
    if not rows and not allow_empty:
        raise LinkingError(f"JSONL is empty: {path}")
    return rows


def load_candidate_pairs(path: str | Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = _load_jsonl(Path(path).expanduser().resolve(), allow_empty=True)
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != CANDIDATE_SCHEMA:
            raise LinkingError(
                f"candidate schema_version must be {CANDIDATE_SCHEMA}, got {row.get('schema_version')!r}"
            )
        if row.get("decision") == "accepted_link":
            raise LinkingError("candidate_pairs must not contain accepted_link decisions")
        a = row.get("track_id_a")
        b = row.get("track_id_b")
        if not _is_positive_int(a) or not _is_positive_int(b):
            raise LinkingError(f"invalid candidate track ids: {a!r}, {b!r}")
        if a >= b:
            raise LinkingError(f"candidate requires track_id_a < track_id_b, got {a},{b}")
        key = (int(a), int(b))
        if key in pairs:
            raise LinkingError(f"duplicate candidate pair: {key}")
        cos = _ensure_finite_float(row.get("cosine_similarity"), field="cosine_similarity")
        if cos < -1.0 or cos > 1.0:
            raise LinkingError(f"cosine_similarity out of range: {cos}")
        overlap = row.get("exact_frame_overlap_count")
        if not _is_nonneg_int(overlap):
            raise LinkingError(f"invalid exact_frame_overlap_count: {overlap!r}")
        conflict = row.get("exact_frame_conflict")
        span = row.get("span_interval_overlap")
        if not isinstance(conflict, bool) or not isinstance(span, bool):
            raise LinkingError("exact_frame_conflict and span_interval_overlap must be bool")
        gap = row.get("temporal_gap_frames")
        if not _is_nonneg_int(gap):
            raise LinkingError(f"invalid temporal_gap_frames: {gap!r}")
        decision = row.get("decision")
        reason = row.get("decision_reason")
        if conflict:
            if decision != DECISION_REJECTED or reason != REASON_EXACT_FRAME:
                raise LinkingError(
                    f"exact-conflict pair {key} has inconsistent decision/reason"
                )
            if int(overlap) <= 0:
                raise LinkingError(f"exact-conflict pair {key} has overlap_count={overlap}")
        else:
            if decision != DECISION_ELIGIBLE or reason != REASON_THRESHOLD_PENDING:
                raise LinkingError(
                    f"non-conflict pair {key} has inconsistent decision/reason"
                )
            if int(overlap) != 0:
                raise LinkingError(
                    f"non-conflict pair {key} has overlap_count={overlap}"
                )
        pairs[key] = {
            "track_id_a": key[0],
            "track_id_b": key[1],
            "cosine_similarity": cos,
            "exact_frame_overlap_count": int(overlap),
            "exact_frame_conflict": conflict,
            "span_interval_overlap": span,
            "temporal_gap_frames": int(gap),
            "decision": decision,
            "decision_reason": reason,
            "schema_version": CANDIDATE_SCHEMA,
        }
    return pairs


def load_track_embeddings_index(path: str | Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(Path(path).expanduser().resolve(), allow_empty=False)
    by_id: dict[int, dict[str, Any]] = {}
    embedding_rows: list[int] = []
    for i, row in enumerate(rows):
        if row.get("schema_version") != TRACK_EMBED_SCHEMA:
            raise LinkingError(
                f"track embedding schema_version must be {TRACK_EMBED_SCHEMA}"
            )
        track_id = row.get("track_id")
        if not _is_positive_int(track_id):
            raise LinkingError(f"invalid track_id in embedding index: {track_id!r}")
        if int(track_id) in by_id:
            raise LinkingError(f"duplicate track_id in embedding index: {track_id}")
        emb_row = row.get("embedding_row")
        if not _is_nonneg_int(emb_row) or int(emb_row) != i:
            raise LinkingError(
                f"embedding_row must equal line order; expected {i}, got {emb_row!r}"
            )
        crop_count = row.get("crop_count")
        crop_ids = row.get("crop_ids")
        if not _is_positive_int(crop_count):
            raise LinkingError(f"invalid crop_count for track {track_id}")
        if not isinstance(crop_ids, list) or len(crop_ids) != int(crop_count):
            raise LinkingError(f"crop_ids length mismatch for track {track_id}")
        if len(set(map(str, crop_ids))) != len(crop_ids):
            raise LinkingError(f"duplicate crop_ids for track {track_id}")
        observation_count = row.get("observation_count")
        first_frame = row.get("first_frame")
        last_frame = row.get("last_frame")
        observed_frame_count = row.get("observed_frame_count")
        if not _is_positive_int(observation_count):
            raise LinkingError(f"invalid observation_count for track {track_id}")
        if not _is_nonneg_int(first_frame) or not _is_nonneg_int(last_frame):
            raise LinkingError(f"invalid frame span for track {track_id}")
        if int(first_frame) > int(last_frame):
            raise LinkingError(f"first_frame > last_frame for track {track_id}")
        if not _is_positive_int(observed_frame_count):
            raise LinkingError(f"invalid observed_frame_count for track {track_id}")
        l2_norm = _ensure_finite_float(row.get("l2_norm"), field="l2_norm")
        if abs(l2_norm - 1.0) > _NORM_ATOL:
            raise LinkingError(f"l2_norm not ~1.0 for track {track_id}: {l2_norm}")
        embedding_rows.append(int(emb_row))
        by_id[int(track_id)] = {
            "track_id": int(track_id),
            "embedding_row": int(emb_row),
            "crop_count": int(crop_count),
            "crop_ids": [str(x) for x in crop_ids],
            "observation_count": int(observation_count),
            "first_frame": int(first_frame),
            "last_frame": int(last_frame),
            "observed_frame_count": int(observed_frame_count),
            "l2_norm": l2_norm,
        }
    if embedding_rows != list(range(len(embedding_rows))):
        raise LinkingError("embedding_row values must be exactly 0..T-1")
    if list(by_id.keys()) != sorted(by_id.keys()):
        # Index file order is by embedding_row; track ids need not be ascending in file,
        # but we keep a sorted view separately.
        pass
    return by_id


def load_manual_decisions(path: str | Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(Path(path).expanduser().resolve(), allow_empty=True)
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if row.get("schema_version") != MANUAL_SCHEMA:
            raise LinkingError(
                f"manual decision schema_version must be {MANUAL_SCHEMA}"
            )
        a = row.get("track_id_a")
        b = row.get("track_id_b")
        if not _is_positive_int(a) or not _is_positive_int(b):
            raise LinkingError(f"invalid manual decision track ids: {a!r}, {b!r}")
        if int(a) == int(b):
            raise LinkingError(f"manual self-pair forbidden: {a}")
        key = _canonical_pair(int(a), int(b))
        if key in seen:
            raise LinkingError(f"duplicate normalized manual pair: {key}")
        seen.add(key)
        label = row.get("review_label")
        if label not in REVIEW_LABELS:
            raise LinkingError(f"invalid review_label: {label!r}")
        approved = row.get("link_approved")
        if not isinstance(approved, bool):
            raise LinkingError("link_approved must be boolean")
        note = row.get("review_note", "")
        reviewer = row.get("reviewer", "")
        if not isinstance(note, str) or not isinstance(reviewer, str):
            raise LinkingError("review_note and reviewer must be strings")
        reviewed_at = row.get("reviewed_at", None)
        if reviewed_at is not None and not isinstance(reviewed_at, str):
            raise LinkingError("reviewed_at must be string or null")

        if approved and label != "likely_same":
            raise LinkingError(
                f"link_approved=true only allowed with likely_same, got {label}"
            )
        if label in {"likely_different", "uncertain", "rejected_exact_frame_conflict"}:
            if approved:
                raise LinkingError(f"link_approved must be false for {label}")

        decisions.append(
            {
                "track_id_a": key[0],
                "track_id_b": key[1],
                "review_label": label,
                "link_approved": approved,
                "review_note": note,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "schema_version": MANUAL_SCHEMA,
            }
        )
    decisions.sort(key=lambda d: (d["track_id_a"], d["track_id_b"]))
    return decisions


def recompute_pair_temporal(
    stats_a: Mapping[str, Any], stats_b: Mapping[str, Any]
) -> dict[str, Any]:
    frames_a: set[int] = stats_a["observed_frames"]
    frames_b: set[int] = stats_b["observed_frames"]
    overlap = len(frames_a & frames_b)
    first_a = int(stats_a["first_frame"])
    last_a = int(stats_a["last_frame"])
    first_b = int(stats_b["first_frame"])
    last_b = int(stats_b["last_frame"])
    span = span_interval_overlap(
        first_a=first_a, last_a=last_a, first_b=first_b, last_b=last_b
    )
    gap = temporal_gap_frames(
        first_a=first_a, last_a=last_a, first_b=first_b, last_b=last_b
    )
    return {
        "exact_frame_overlap_count": int(overlap),
        "exact_frame_conflict": overlap > 0,
        "span_interval_overlap": bool(span),
        "temporal_gap_frames": int(gap),
    }


def verify_candidates_against_tracks(
    candidates: Mapping[tuple[int, int], Mapping[str, Any]],
    track_stats: Mapping[int, Mapping[str, Any]],
) -> None:
    for key, cand in candidates.items():
        a, b = key
        if a not in track_stats or b not in track_stats:
            raise LinkingError(f"candidate pair {key} missing from tracks.jsonl stats")
        recomputed = recompute_pair_temporal(track_stats[a], track_stats[b])
        for field in (
            "exact_frame_overlap_count",
            "exact_frame_conflict",
            "span_interval_overlap",
            "temporal_gap_frames",
        ):
            if recomputed[field] != cand[field]:
                raise LinkingError(
                    f"stale/tampered candidate temporal field {field} for {key}: "
                    f"candidate={cand[field]!r} recomputed={recomputed[field]!r}"
                )


def evidence_class_for_pair(
    *,
    meta_a: Mapping[str, Any],
    meta_b: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> str:
    min_crops = int(policy["evidence"]["minimum_crop_count_per_track_for_strong_review"])
    min_obs = int(
        policy["evidence"]["minimum_observation_count_per_track_for_strong_review"]
    )
    crop_a = int(meta_a["crop_count"])
    crop_b = int(meta_b["crop_count"])
    obs_a = int(meta_a["observation_count"])
    obs_b = int(meta_b["observation_count"])
    if crop_a < min_crops or crop_b < min_crops:
        return "low_crop_evidence"
    if obs_a < min_obs or obs_b < min_obs:
        return "short_track_evidence"
    return "strong_review_evidence"


def _connected_components(edges: Sequence[tuple[int, int]]) -> list[set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
        nodes.add(a)
        nodes.add(b)
    seen: set[int] = set()
    components: list[set[int]] = []
    for node in sorted(nodes):
        if node in seen:
            continue
        queue: deque[int] = deque([node])
        seen.add(node)
        comp: set[int] = set()
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nbr in sorted(graph[cur]):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        components.append(comp)
    components.sort(key=lambda c: (min(c), len(c), tuple(sorted(c))))
    return components


def _all_unordered_pairs(members: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted(members)
    pairs: list[tuple[int, int]] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            pairs.append((ordered[i], ordered[j]))
    return pairs


def create_temp_linking_dir(output_dir: Path) -> Path:
    final_dir = output_dir.expanduser().resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    tmp_dir = parent / f"_tmp_reid_linking_{final_dir.name}_{token}"
    if tmp_dir.exists():
        raise LinkingError(f"temporary output path already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def finalize_linking_dir(*, temp_dir: Path, final_dir: Path, overwrite: bool) -> Path:
    temp_path = temp_dir.expanduser().resolve()
    final_path = final_dir.expanduser().resolve()
    if not temp_path.is_dir():
        raise LinkingError(f"temporary output directory missing: {temp_path}")

    backup_path: Path | None = None
    try:
        if final_path.exists():
            if not overwrite:
                raise LinkingError(
                    f"output already exists: {final_path}; re-run with --overwrite"
                )
            backup_path = final_path.with_name(
                f"_backup_reid_linking_{final_path.name}_{uuid.uuid4().hex[:8]}"
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
    for stray in parent.glob(f"_tmp_reid_linking_{final_path.name}_*"):
        cleanup_dir(stray)
    for stray in parent.glob(f"_backup_reid_linking_{final_path.name}_*"):
        cleanup_dir(stray)
    return final_path


def run_link_reid_tracks(
    *,
    candidate_pairs: str | Path,
    track_embeddings_index: str | Path,
    tracks: str | Path,
    manual_decisions: str | Path,
    policy: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    final_dir = Path(output_dir).expanduser().resolve()
    check_output_collision(final_dir, overwrite=overwrite)

    policy_data = load_linking_policy(policy)
    cand_path = Path(candidate_pairs).expanduser().resolve()
    index_path = Path(track_embeddings_index).expanduser().resolve()
    tracks_path = Path(tracks).expanduser().resolve()
    decisions_path = Path(manual_decisions).expanduser().resolve()

    candidates = load_candidate_pairs(cand_path)
    embedded = load_track_embeddings_index(index_path)
    try:
        observations = load_tracks_jsonl(tracks_path)
        track_stats = build_track_observation_stats(observations)
    except AggregateError as exc:
        raise LinkingError(str(exc)) from exc

    for track_id, meta in embedded.items():
        if track_id not in track_stats:
            raise LinkingError(
                f"embedded track {track_id} missing from tracks.jsonl"
            )
        raw = track_stats[track_id]
        if int(meta["observation_count"]) != int(raw["observation_count"]):
            raise LinkingError(
                f"observation_count mismatch for track {track_id}"
            )
        if int(meta["first_frame"]) != int(raw["first_frame"]):
            raise LinkingError(f"first_frame mismatch for track {track_id}")
        if int(meta["last_frame"]) != int(raw["last_frame"]):
            raise LinkingError(f"last_frame mismatch for track {track_id}")
        if int(meta["observed_frame_count"]) != int(raw["observed_frame_count"]):
            raise LinkingError(
                f"observed_frame_count mismatch for track {track_id}"
            )

    verify_candidates_against_tracks(candidates, track_stats)
    decisions = load_manual_decisions(decisions_path)

    for decision in decisions:
        key = (decision["track_id_a"], decision["track_id_b"])
        if key not in candidates:
            raise LinkingError(f"manual decision pair missing from candidates: {key}")

    # Pair-level approval candidates (explicit only).
    approved_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for decision in decisions:
        key = (decision["track_id_a"], decision["track_id_b"])
        cand = candidates[key]
        if not decision["link_approved"]:
            continue
        if decision["review_label"] != "likely_same":
            raise LinkingError("internal error: approved non-likely_same")
        if cand["exact_frame_conflict"] or cand["decision"] != DECISION_ELIGIBLE:
            raise LinkingError(
                f"cannot approve exact-conflict or non-eligible pair {key}"
            )
        recomputed = recompute_pair_temporal(
            track_stats[key[0]], track_stats[key[1]]
        )
        if recomputed["exact_frame_conflict"]:
            raise LinkingError(
                f"cannot approve pair with recomputed exact-frame conflict: {key}"
            )
        approved_edges[key] = {
            "decision": decision,
            "candidate": cand,
        }

    # Build approval components with full-clique requirement.
    components = _connected_components(list(approved_edges.keys()))
    applied_components: list[dict[str, Any]] = []
    held_incomplete_members: set[int] = set()
    held_incomplete_edges: set[tuple[int, int]] = set()
    conflict_rejected_members: set[int] = set()
    conflict_rejected_edges: set[tuple[int, int]] = set()

    for members in components:
        ordered = sorted(members)
        required = _all_unordered_pairs(ordered)
        if any(edge not in approved_edges for edge in required):
            held_incomplete_members.update(ordered)
            for edge in required:
                if edge in approved_edges:
                    held_incomplete_edges.add(edge)
            continue

        # Full clique of approvals: re-check every cross-member pair.
        conflicted = False
        for edge in required:
            recomputed = recompute_pair_temporal(
                track_stats[edge[0]], track_stats[edge[1]]
            )
            if recomputed["exact_frame_conflict"]:
                conflicted = True
                break
        if conflicted:
            conflict_rejected_members.update(ordered)
            conflict_rejected_edges.update(required)
            continue

        sims = [
            float(approved_edges[edge]["candidate"]["cosine_similarity"])
            for edge in required
        ]
        applied_components.append(
            {
                "members": ordered,
                "edges": required,
                "global_candidate_id": int(min(ordered)),
                "similarity_min": float(min(sims)) if sims else None,
                "similarity_mean": float(sum(sims) / len(sims)) if sims else None,
            }
        )

    member_to_component: dict[int, dict[str, Any]] = {}
    for comp in applied_components:
        for tid in comp["members"]:
            member_to_component[tid] = comp

    applied_edge_keys = {
        edge for comp in applied_components for edge in comp["edges"]
    }

    # Audits for every manual decision.
    audits: list[dict[str, Any]] = []
    for decision in decisions:
        key = (decision["track_id_a"], decision["track_id_b"])
        cand = candidates[key]
        meta_a = embedded[key[0]]
        meta_b = embedded[key[1]]
        evidence = evidence_class_for_pair(
            meta_a=meta_a, meta_b=meta_b, policy=policy_data
        )
        if key in applied_edge_keys:
            outcome = "applied"
            reason = "manual_likely_same_full_clique_conflict_free"
        elif key in conflict_rejected_edges:
            outcome = "rejected_component_exact_frame_conflict"
            reason = "cross_member_exact_frame_conflict"
        elif key in held_incomplete_edges:
            outcome = "held_incomplete_component_approval"
            reason = "approval_component_not_full_clique"
        elif decision["link_approved"] and cand["exact_frame_conflict"]:
            # Should have raised earlier; keep defensive path.
            outcome = "rejected_exact_frame_conflict"
            reason = "exact_frame_conflict"
        elif not decision["link_approved"]:
            if decision["review_label"] == "rejected_exact_frame_conflict":
                outcome = "rejected_exact_frame_conflict"
                reason = "manual_rejected_exact_frame_conflict"
            else:
                outcome = "reviewed_not_approved"
                reason = f"review_label={decision['review_label']}"
        else:
            outcome = "reviewed_not_approved"
            reason = "approved_but_not_applied"
        audits.append(
            {
                "track_id_a": key[0],
                "track_id_b": key[1],
                "review_label": decision["review_label"],
                "link_approved_requested": bool(decision["link_approved"]),
                "candidate_found": True,
                "candidate_decision": cand["decision"],
                "cosine_similarity": float(cand["cosine_similarity"]),
                "exact_frame_conflict": bool(cand["exact_frame_conflict"]),
                "evidence_class": evidence,
                "final_outcome": outcome,
                "final_reason": reason,
                "schema_version": AUDIT_SCHEMA,
            }
        )

    accepted_edges: list[dict[str, Any]] = []
    for key in sorted(applied_edge_keys):
        decision = approved_edges[key]["decision"]
        cand = approved_edges[key]["candidate"]
        comp = member_to_component[key[0]]
        accepted_edges.append(
            {
                "track_id_a": key[0],
                "track_id_b": key[1],
                "cosine_similarity": float(cand["cosine_similarity"]),
                "review_label": decision["review_label"],
                "review_note": decision["review_note"],
                "reviewer": decision["reviewer"],
                "reviewed_at": decision["reviewed_at"],
                "candidate_decision": cand["decision"],
                "exact_frame_overlap_count": int(cand["exact_frame_overlap_count"]),
                "component_global_candidate_id": int(comp["global_candidate_id"]),
                "component_size": len(comp["members"]),
                "schema_version": ACCEPTED_EDGE_SCHEMA,
            }
        )

    # Global map for every raw track.
    global_rows: list[dict[str, Any]] = []
    for track_id in sorted(track_stats.keys()):
        raw = track_stats[track_id]
        has_embedding = track_id in embedded
        crop_count = int(embedded[track_id]["crop_count"]) if has_embedding else 0
        if track_id in member_to_component:
            comp = member_to_component[track_id]
            link_status = "linked_component"
            members = list(comp["members"])
            gid = int(comp["global_candidate_id"])
            edge_count = len(comp["edges"])
            sim_min = comp["similarity_min"]
            sim_mean = comp["similarity_mean"]
        else:
            members = [track_id]
            gid = int(track_id)
            edge_count = 0
            sim_min = None
            sim_mean = None
            if not has_embedding:
                link_status = "singleton_no_embedding"
            elif track_id in held_incomplete_members:
                link_status = "singleton_held_incomplete_approval"
            elif track_id in conflict_rejected_members:
                link_status = "singleton_unlinked"
            else:
                link_status = "singleton_unlinked"

        global_rows.append(
            {
                "raw_track_id": int(track_id),
                "global_candidate_id": gid,
                "component_member_track_ids": members,
                "component_size": len(members),
                "has_embedding": bool(has_embedding),
                "crop_count": crop_count,
                "observation_count": int(raw["observation_count"]),
                "first_frame": int(raw["first_frame"]),
                "last_frame": int(raw["last_frame"]),
                "observed_frame_count": int(raw["observed_frame_count"]),
                "link_status": link_status,
                "accepted_edge_count": int(edge_count),
                "component_similarity_min": sim_min,
                "component_similarity_mean": sim_mean,
                "schema_version": GLOBAL_MAP_SCHEMA,
            }
        )

    linked_track_ids = {tid for comp in applied_components for tid in comp["members"]}
    singleton_count = sum(1 for r in global_rows if r["component_size"] == 1)
    no_emb_singletons = sum(
        1 for r in global_rows if r["link_status"] == "singleton_no_embedding"
    )
    held_incomplete_component_count = len(
        {
            frozenset(comp)
            for comp in components
            if any(edge not in approved_edges for edge in _all_unordered_pairs(sorted(comp)))
        }
    )
    global_candidate_ids = {r["global_candidate_id"] for r in global_rows}

    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok",
        "policy_schema_version": POLICY_SCHEMA,
        "automatic_linking_enabled": False,
        "similarity_threshold": None,
        "cosine_usage": "ranking_only",
        "candidate_pair_count": len(candidates),
        "embedded_track_count": len(embedded),
        "raw_track_count": len(track_stats),
        "manual_decision_count": len(decisions),
        "requested_approved_edge_count": len(approved_edges),
        "applied_accepted_edge_count": len(accepted_edges),
        "linked_component_count": len(applied_components),
        "linked_track_count": len(linked_track_ids),
        "singleton_track_count": singleton_count,
        "no_embedding_singleton_count": no_emb_singletons,
        "held_incomplete_component_count": held_incomplete_component_count,
        "global_candidate_count": len(global_candidate_ids),
        "deterministic_global_id_policy": "minimum_raw_track_id",
        "uncontrolled_transitive_chaining_performed": False,
        "component_cross_member_checks_performed": True,
        "candidate_pairs": str(cand_path),
        "track_embeddings_index": str(index_path),
        "tracks_jsonl": str(tracks_path),
        "manual_decisions": str(decisions_path),
        "policy": policy_data["_path"],
        "elapsed_sec": float(elapsed),
        "schema_version": SUMMARY_SCHEMA,
    }
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise LinkingError(f"summary field {key} is not finite")

    temp_dir: Path | None = None
    try:
        temp_dir = create_temp_linking_dir(final_dir)
        write_manifest_jsonl(temp_dir / ACCEPTED_NAME, accepted_edges)
        write_manifest_jsonl(temp_dir / AUDIT_NAME, audits)
        write_manifest_jsonl(temp_dir / GLOBAL_MAP_NAME, global_rows)
        (temp_dir / SUMMARY_NAME).write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        names = sorted(p.name for p in temp_dir.iterdir())
        expected = sorted([ACCEPTED_NAME, AUDIT_NAME, GLOBAL_MAP_NAME, SUMMARY_NAME])
        if names != expected:
            raise LinkingError(f"unexpected linking output files: {names}")
        finalized = finalize_linking_dir(
            temp_dir=temp_dir, final_dir=final_dir, overwrite=overwrite
        )
        temp_dir = None
    except Exception:
        cleanup_dir(temp_dir)
        raise

    return {
        "status": "ok",
        "output_dir": str(finalized),
        "accepted_edges_path": str(finalized / ACCEPTED_NAME),
        "audit_path": str(finalized / AUDIT_NAME),
        "global_map_path": str(finalized / GLOBAL_MAP_NAME),
        "summary_path": str(finalized / SUMMARY_NAME),
        "summary": summary,
        "accepted_edges": accepted_edges,
        "audits": audits,
        "global_rows": global_rows,
    }
