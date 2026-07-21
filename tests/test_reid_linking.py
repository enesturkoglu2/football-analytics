"""Unit tests for manually approved ReID linking (no real benchmark outputs)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.candidates import (
    DECISION_ELIGIBLE,
    DECISION_REJECTED,
    REASON_EXACT_FRAME,
    REASON_THRESHOLD_PENDING,
)
from football_analytics.reid.linking import (
    ACCEPTED_NAME,
    AUDIT_NAME,
    GLOBAL_MAP_NAME,
    SUMMARY_NAME,
    LinkingError,
    load_linking_policy,
    run_link_reid_tracks,
)
from football_analytics.reid.writers import ReIDWritersError, write_manifest_jsonl

STAGE_POLICY = _PROJECT_ROOT / "configs" / "reid" / "linking_policy_stage4b.yaml"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    write_manifest_jsonl(path, rows)


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
    _write_jsonl(path, rows)


def _track_meta_from_obs(observations: list[tuple[int, int]], track_id: int) -> dict:
    frames = sorted({f for t, f in observations if t == track_id})
    return {
        "observation_count": sum(1 for t, _ in observations if t == track_id),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "observed_frame_count": len(frames),
    }


def _embed_row(
    *,
    track_id: int,
    embedding_row: int,
    observations: list[tuple[int, int]],
    crop_count: int = 5,
) -> dict:
    meta = _track_meta_from_obs(observations, track_id)
    return {
        "track_id": track_id,
        "crop_ids": [f"c{track_id}_{i}" for i in range(crop_count)],
        "crop_count": crop_count,
        "embedding_row": embedding_row,
        "aggregation": "l2_mean",
        "embedding_shape": [512],
        "embedding_dtype": "float32",
        "l2_norm": 1.0,
        "observation_count": meta["observation_count"],
        "first_frame": meta["first_frame"],
        "last_frame": meta["last_frame"],
        "observed_frame_count": meta["observed_frame_count"],
        "schema_version": "reid_track_embedding_v1",
    }


def _candidate(
    a: int,
    b: int,
    *,
    cos: float,
    observations: list[tuple[int, int]],
) -> dict:
    frames_a = {f for t, f in observations if t == a}
    frames_b = {f for t, f in observations if t == b}
    overlap = len(frames_a & frames_b)
    first_a, last_a = min(frames_a), max(frames_a)
    first_b, last_b = min(frames_b), max(frames_b)
    span = first_a <= last_b and first_b <= last_a
    if last_a < first_b:
        gap = max(0, first_b - last_a - 1)
    elif last_b < first_a:
        gap = max(0, first_a - last_b - 1)
    else:
        gap = 0
    conflict = overlap > 0
    return {
        "track_id_a": a,
        "track_id_b": b,
        "cosine_similarity": cos,
        "temporal_gap_frames": gap,
        "exact_frame_overlap_count": overlap,
        "exact_frame_conflict": conflict,
        "span_interval_overlap": span,
        "decision": DECISION_REJECTED if conflict else DECISION_ELIGIBLE,
        "decision_reason": REASON_EXACT_FRAME if conflict else REASON_THRESHOLD_PENDING,
        "schema_version": "reid_candidate_pair_v1",
    }


def _decision(
    a: int,
    b: int,
    *,
    label: str,
    approved: bool,
    note: str = "",
    reviewer: str = "tester",
) -> dict:
    return {
        "track_id_a": a,
        "track_id_b": b,
        "review_label": label,
        "link_approved": approved,
        "review_note": note,
        "reviewer": reviewer,
        "reviewed_at": "2026-07-21T00:00:00Z",
        "schema_version": "reid_manual_pair_decision_v1",
    }


def _bundle(
    tmp: Path,
    *,
    observations: list[tuple[int, int]],
    embedded_ids: list[int],
    candidates: list[dict],
    decisions: list[dict],
    crop_counts: dict[int, int] | None = None,
) -> dict[str, Path]:
    tracks = tmp / "tracks.jsonl"
    index = tmp / "track_embeddings.jsonl"
    pairs = tmp / "candidate_pairs.jsonl"
    manual = tmp / "manual_decisions.jsonl"
    _write_tracks(tracks, observations)
    crop_counts = crop_counts or {tid: 5 for tid in embedded_ids}
    embed_rows = [
        _embed_row(
            track_id=tid,
            embedding_row=i,
            observations=observations,
            crop_count=crop_counts[tid],
        )
        for i, tid in enumerate(embedded_ids)
    ]
    _write_jsonl(index, embed_rows)
    _write_jsonl(pairs, candidates)
    _write_jsonl(manual, decisions)
    return {
        "tracks": tracks,
        "index": index,
        "pairs": pairs,
        "manual": manual,
        "policy": STAGE_POLICY,
        "out": tmp / "link_out",
    }


class PolicyTests(unittest.TestCase):
    def test_valid_stage_policy(self) -> None:
        cfg = load_linking_policy(STAGE_POLICY)
        self.assertFalse(cfg["automatic_linking_enabled"])
        self.assertIsNone(cfg["similarity_threshold"])

    def test_policy_rejects(self) -> None:
        cases = [
            ("automatic_linking_enabled", True),
            ("similarity_threshold", 0.9),
            ("cosine_usage", "accept_above_threshold"),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    src = yaml.safe_load(STAGE_POLICY.read_text(encoding="utf-8"))
                    src[key] = value
                    path = Path(tmp) / "policy.yaml"
                    path.write_text(yaml.safe_dump(src), encoding="utf-8")
                    with self.assertRaises(LinkingError):
                        load_linking_policy(path)

        nested = [
            (("temporal", "exact_frame_conflict_hard_reject"), False),
            (("decisions", "manual_acceptance_required_for_linking"), False),
            (("component_rules", "uncontrolled_transitive_chaining_allowed"), True),
        ]
        for (section, key), value in nested:
            with self.subTest(section=section, key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    src = yaml.safe_load(STAGE_POLICY.read_text(encoding="utf-8"))
                    src[section][key] = value
                    path = Path(tmp) / "policy.yaml"
                    path.write_text(yaml.safe_dump(src), encoding="utf-8")
                    with self.assertRaises(LinkingError):
                        load_linking_policy(path)


class ManualDecisionTests(unittest.TestCase):
    def test_empty_decisions_all_singletons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (1, 1), (2, 10), (2, 11), (3, 20)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.99, observations=obs)],
                decisions=[],
            )
            # raw track 3 has no embedding
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["applied_accepted_edge_count"], 0)
            statuses = {r["raw_track_id"]: r["link_status"] for r in result["global_rows"]}
            self.assertEqual(statuses[1], "singleton_unlinked")
            self.assertEqual(statuses[2], "singleton_unlinked")
            self.assertEqual(statuses[3], "singleton_no_embedding")

    def test_label_approval_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10)]
            base = dict(
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.5, observations=obs)],
            )
            for label in ("likely_different", "uncertain", "rejected_exact_frame_conflict"):
                paths = _bundle(
                    tmp_path / label,
                    decisions=[_decision(1, 2, label=label, approved=True)],
                    **base,
                )
                with self.assertRaises(LinkingError):
                    run_link_reid_tracks(
                        candidate_pairs=paths["pairs"],
                        track_embeddings_index=paths["index"],
                        tracks=paths["tracks"],
                        manual_decisions=paths["manual"],
                        policy=paths["policy"],
                        output_dir=paths["out"],
                    )

    def test_duplicate_and_self_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.5, observations=obs)],
                decisions=[
                    _decision(1, 2, label="likely_same", approved=False),
                    _decision(2, 1, label="uncertain", approved=False),
                ],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths["pairs"],
                    track_embeddings_index=paths["index"],
                    tracks=paths["tracks"],
                    manual_decisions=paths["manual"],
                    policy=paths["policy"],
                    output_dir=paths["out"],
                )

    def test_missing_candidate_and_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10), (3, 20)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2, 3],
                candidates=[_candidate(1, 2, cos=0.5, observations=obs)],
                decisions=[_decision(1, 3, label="likely_same", approved=True)],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths["pairs"],
                    track_embeddings_index=paths["index"],
                    tracks=paths["tracks"],
                    manual_decisions=paths["manual"],
                    policy=paths["policy"],
                    output_dir=paths["out"],
                )
            bad = tmp_path / "bad.jsonl"
            bad.write_text("{not-json\n", encoding="utf-8")
            paths2 = _bundle(
                tmp_path / "b",
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.5, observations=obs)],
                decisions=[],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths2["pairs"],
                    track_embeddings_index=paths2["index"],
                    tracks=paths2["tracks"],
                    manual_decisions=bad,
                    policy=paths2["policy"],
                    output_dir=paths2["out"],
                )


class PairSafetyTests(unittest.TestCase):
    def test_high_cosine_without_manual_no_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.999, observations=obs)],
                decisions=[],
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["applied_accepted_edge_count"], 0)

    def test_low_cosine_with_explicit_approval_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(5, 0), (5, 1), (9, 10), (9, 11)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[5, 9],
                candidates=[_candidate(5, 9, cos=0.11, observations=obs)],
                decisions=[_decision(9, 5, label="likely_same", approved=True)],
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["applied_accepted_edge_count"], 1)
            self.assertEqual(result["accepted_edges"][0]["component_global_candidate_id"], 5)

    def test_exact_conflict_approval_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (1, 5), (2, 5), (2, 8)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.95, observations=obs)],
                decisions=[_decision(1, 2, label="likely_same", approved=True)],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths["pairs"],
                    track_embeddings_index=paths["index"],
                    tracks=paths["tracks"],
                    manual_decisions=paths["manual"],
                    policy=paths["policy"],
                    output_dir=paths["out"],
                )
            self.assertFalse(paths["out"].exists())

    def test_stale_temporal_and_accepted_link_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10)]
            cand = _candidate(1, 2, cos=0.5, observations=obs)
            cand["exact_frame_overlap_count"] = 3
            cand["exact_frame_conflict"] = True
            cand["decision"] = DECISION_REJECTED
            cand["decision_reason"] = REASON_EXACT_FRAME
            # Conflict flags disagree with raw frames → stale.
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[cand],
                decisions=[],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths["pairs"],
                    track_embeddings_index=paths["index"],
                    tracks=paths["tracks"],
                    manual_decisions=paths["manual"],
                    policy=paths["policy"],
                    output_dir=paths["out"],
                )

            cand2 = _candidate(1, 2, cos=0.5, observations=obs)
            cand2["decision"] = "accepted_link"
            cand2["decision_reason"] = "x"
            paths2 = _bundle(
                tmp_path / "c2",
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[cand2],
                decisions=[],
            )
            with self.assertRaises(LinkingError):
                run_link_reid_tracks(
                    candidate_pairs=paths2["pairs"],
                    track_embeddings_index=paths2["index"],
                    tracks=paths2["tracks"],
                    manual_decisions=paths2["manual"],
                    policy=paths2["policy"],
                    output_dir=paths2["out"],
                )


class ComponentTests(unittest.TestCase):
    def test_two_member_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(4, i) for i in range(40)] + [(682, i) for i in range(100, 150)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[4, 682],
                candidates=[_candidate(4, 682, cos=0.87, observations=obs)],
                decisions=[_decision(4, 682, label="likely_same", approved=True)],
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["linked_component_count"], 1)
            edge = result["accepted_edges"][0]
            self.assertEqual(edge["component_global_candidate_id"], 4)
            self.assertEqual(edge["cosine_similarity"], 0.87)
            rows = {r["raw_track_id"]: r for r in result["global_rows"]}
            self.assertEqual(rows[4]["global_candidate_id"], 4)
            self.assertEqual(rows[682]["global_candidate_id"], 4)
            self.assertEqual(rows[4]["component_similarity_min"], 0.87)
            self.assertEqual(rows[4]["component_similarity_mean"], 0.87)
            self.assertEqual(rows[4]["accepted_edge_count"], 1)

    def test_incomplete_chaining_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = (
                [(1, i) for i in range(0, 5)]
                + [(2, i) for i in range(20, 25)]
                + [(3, i) for i in range(40, 45)]
            )
            cands = [
                _candidate(1, 2, cos=0.8, observations=obs),
                _candidate(2, 3, cos=0.81, observations=obs),
                _candidate(1, 3, cos=0.82, observations=obs),
            ]
            decisions = [
                _decision(1, 2, label="likely_same", approved=True),
                _decision(2, 3, label="likely_same", approved=True),
                # A-C missing approval
            ]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2, 3],
                candidates=cands,
                decisions=decisions,
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["applied_accepted_edge_count"], 0)
            self.assertEqual(result["summary"]["held_incomplete_component_count"], 1)
            statuses = {r["raw_track_id"]: r["link_status"] for r in result["global_rows"]}
            self.assertEqual(statuses[1], "singleton_held_incomplete_approval")
            self.assertEqual(statuses[2], "singleton_held_incomplete_approval")
            self.assertEqual(statuses[3], "singleton_held_incomplete_approval")
            outcomes = { (a["track_id_a"], a["track_id_b"]): a["final_outcome"] for a in result["audits"]}
            self.assertEqual(outcomes[(1, 2)], "held_incomplete_component_approval")
            self.assertEqual(outcomes[(2, 3)], "held_incomplete_component_approval")

            # Deterministic under decision reorder.
            paths2 = _bundle(
                tmp_path / "reorder",
                observations=obs,
                embedded_ids=[1, 2, 3],
                candidates=cands,
                decisions=list(reversed(decisions)),
            )
            result2 = run_link_reid_tracks(
                candidate_pairs=paths2["pairs"],
                track_embeddings_index=paths2["index"],
                tracks=paths2["tracks"],
                manual_decisions=paths2["manual"],
                policy=paths2["policy"],
                output_dir=paths2["out"],
            )
            self.assertEqual(result["accepted_edges"], result2["accepted_edges"])
            self.assertEqual(
                [(a["track_id_a"], a["track_id_b"], a["final_outcome"]) for a in result["audits"]],
                [(a["track_id_a"], a["track_id_b"], a["final_outcome"]) for a in result2["audits"]],
            )

    def test_full_triangle_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = (
                [(10, i) for i in range(0, 40)]
                + [(20, i) for i in range(50, 90)]
                + [(30, i) for i in range(100, 140)]
            )
            cands = [
                _candidate(10, 20, cos=0.7, observations=obs),
                _candidate(10, 30, cos=0.8, observations=obs),
                _candidate(20, 30, cos=0.9, observations=obs),
            ]
            decisions = [
                _decision(10, 20, label="likely_same", approved=True),
                _decision(10, 30, label="likely_same", approved=True),
                _decision(20, 30, label="likely_same", approved=True),
            ]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[10, 20, 30],
                candidates=cands,
                decisions=decisions,
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["summary"]["applied_accepted_edge_count"], 3)
            self.assertEqual(result["summary"]["linked_component_count"], 1)
            rows = {r["raw_track_id"]: r for r in result["global_rows"]}
            for tid in (10, 20, 30):
                self.assertEqual(rows[tid]["global_candidate_id"], 10)
                self.assertEqual(rows[tid]["accepted_edge_count"], 3)
                self.assertEqual(rows[tid]["component_member_track_ids"], [10, 20, 30])
                self.assertAlmostEqual(rows[tid]["component_similarity_min"], 0.7)
                self.assertAlmostEqual(rows[tid]["component_similarity_mean"], 0.8)


class AuditOutputTests(unittest.TestCase):
    def test_audit_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = (
                [(1, i) for i in range(0, 40)]
                + [(2, i) for i in range(50, 90)]
                + [(3, 200)]
            )
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.6, observations=obs)],
                decisions=[
                    _decision(1, 2, label="likely_same", approved=True),
                ],
                crop_counts={1: 5, 2: 5},
            )
            # add uncertain decision requires that pair in candidates - use reviewed_not_approved via false approval
            # already have one decision
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            out = paths["out"]
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(
                names,
                sorted([ACCEPTED_NAME, AUDIT_NAME, GLOBAL_MAP_NAME, SUMMARY_NAME]),
            )
            audits = result["audits"]
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["final_outcome"], "applied")
            self.assertEqual(audits[0]["evidence_class"], "strong_review_evidence")
            summary = json.loads((out / SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertIsNone(summary["similarity_threshold"])
            self.assertFalse(summary["automatic_linking_enabled"])
            self.assertFalse(summary["uncontrolled_transitive_chaining_performed"])
            text = (out / AUDIT_NAME).read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))

            # reviewed_not_approved path
            paths2 = _bundle(
                tmp_path / "r2",
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.6, observations=obs)],
                decisions=[_decision(1, 2, label="likely_same", approved=False)],
            )
            result2 = run_link_reid_tracks(
                candidate_pairs=paths2["pairs"],
                track_embeddings_index=paths2["index"],
                tracks=paths2["tracks"],
                manual_decisions=paths2["manual"],
                policy=paths2["policy"],
                output_dir=paths2["out"],
            )
            self.assertEqual(result2["audits"][0]["final_outcome"], "reviewed_not_approved")

    def test_collision_overwrite_cleanup_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, 0), (2, 10)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.4, observations=obs)],
                decisions=[_decision(1, 2, label="likely_same", approved=True)],
            )
            first = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            with self.assertRaises(ReIDWritersError):
                run_link_reid_tracks(
                    candidate_pairs=paths["pairs"],
                    track_embeddings_index=paths["index"],
                    tracks=paths["tracks"],
                    manual_decisions=paths["manual"],
                    policy=paths["policy"],
                    output_dir=paths["out"],
                )
            second = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
                overwrite=True,
            )
            self.assertEqual(first["accepted_edges"], second["accepted_edges"])
            self.assertEqual(list(tmp_path.glob("_tmp_reid_linking_*")), [])
            self.assertEqual(list(tmp_path.glob("_backup_reid_linking_*")), [])

            with mock.patch(
                "football_analytics.reid.linking.write_manifest_jsonl",
                side_effect=LinkingError("boom"),
            ):
                with self.assertRaises(LinkingError):
                    run_link_reid_tracks(
                        candidate_pairs=paths["pairs"],
                        track_embeddings_index=paths["index"],
                        tracks=paths["tracks"],
                        manual_decisions=paths["manual"],
                        policy=paths["policy"],
                        output_dir=tmp_path / "fail_out",
                    )
            self.assertFalse((tmp_path / "fail_out").exists())
            self.assertEqual(list(tmp_path.glob("_tmp_reid_linking_*")), [])

    def test_low_crop_evidence_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs = [(1, i) for i in range(40)] + [(2, i) for i in range(100, 140)]
            paths = _bundle(
                tmp_path,
                observations=obs,
                embedded_ids=[1, 2],
                candidates=[_candidate(1, 2, cos=0.5, observations=obs)],
                decisions=[_decision(1, 2, label="uncertain", approved=False)],
                crop_counts={1: 1, 2: 5},
            )
            result = run_link_reid_tracks(
                candidate_pairs=paths["pairs"],
                track_embeddings_index=paths["index"],
                tracks=paths["tracks"],
                manual_decisions=paths["manual"],
                policy=paths["policy"],
                output_dir=paths["out"],
            )
            self.assertEqual(result["audits"][0]["evidence_class"], "low_crop_evidence")


if __name__ == "__main__":
    unittest.main()
