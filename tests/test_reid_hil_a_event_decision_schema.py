"""Tests for ReID HIL-A event/decision schema and append-only log."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_analytics.reid.hil.candidates import (  # noqa: E402
    CandidateManifestError,
    validate_candidate_manifest,
)
from football_analytics.reid.hil.common import (  # noqa: E402
    HilValidationError,
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
)
from football_analytics.reid.hil.decisions import (  # noqa: E402
    DecisionAction,
    DecisionError,
    build_decision,
    validate_decision,
)
from football_analytics.reid.hil.events import (  # noqa: E402
    EventError,
    EventType,
    validate_recovery_event,
)
from football_analytics.reid.hil.log import AppendOnlyLogError, DecisionLog  # noqa: E402
from football_analytics.reid.hil.resolve import (  # noqa: E402
    EventReviewState,
    resolve_effective_decisions,
    resolve_event_review_state,
)

VIDEO_SHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _base_event(**overrides):
    row = {
        "schema_version": "target_recovery_event_v1",
        "event_id": "evt_1",
        "project_id": "football-analytics",
        "run_id": "test",
        "target_id": "target_001",
        "event_type": EventType.TARGET_LOST.value,
        "video_id": "v1",
        "video_path": "data/video.mp4",
        "video_sha256": VIDEO_SHA,
        "created_at": "2026-07-27T00:00:00Z",
        "status": "open",
        "trigger_source": "test",
        "trigger_reason": "unit",
        "last_confirmed_segment_id": "seg_0",
        "last_confirmed_frame_index": 10,
        "review_window_start_frame": 11,
        "review_window_end_frame": 50,
        "candidate_manifest_path": None,
        "candidate_manifest_sha256": None,
        "candidate_count": 0,
        "requires_calibration": True,
        "evidence_paths": [],
        "evidence_sha256": [],
        "provenance": {},
        "metadata": {},
    }
    row.update(overrides)
    return row


def _candidate(i: int, *, eligible: bool = True, rank: int | None = None):
    rank = rank if rank is not None else i
    return {
        "candidate_id": f"c{i}",
        "segment_id": f"s{i}",
        "raw_track_id": f"r{i}",
        "start_frame": i,
        "middle_frame": i + 1,
        "end_frame": i + 2,
        "bbox_references": [{"frame_index": i + 1, "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]}],
        "crop_path": f"crops/c{i}.jpg",
        "crop_sha256": f"{i:064x}",
        "context_paths": {},
        "context_sha256": {},
        "team_evidence": {"is_identity_proof": False},
        "visibility": {},
        "quality": {},
        "contamination": {},
        "sportsreid_model_id": SPORTSREID_MODEL_ID,
        "sportsreid_checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
        "appearance_rank": rank,
        "T_max": 0.5,
        "D_max": 0.4,
        "S": 0.1,
        "eligibility": eligible,
        "rejection_reason": None if eligible else "filtered_non_player",
        "display_order": i,
    }


def _manifest(candidates=None, **overrides):
    cands = candidates if candidates is not None else [_candidate(1), _candidate(2), _candidate(3), _candidate(4)]
    row = {
        "schema_version": "target_recovery_candidate_manifest_v1",
        "event_id": "evt_re",
        "target_id": "target_001",
        "candidate_count": len(cands),
        "supports_direct_bbox_selection": True,
        "candidates": cands,
        "metadata": {},
    }
    row.update(overrides)
    return row


def _decision(**overrides):
    row = build_decision(
        decision_id="d1",
        project_id="football-analytics",
        run_id="test",
        target_id="target_001",
        event_id="evt_1",
        video_id="v1",
        video_path="data/video.mp4",
        video_sha256=VIDEO_SHA,
        reviewer="tester",
        created_at="2026-07-27T00:00:00Z",
        revision=1,
        action="DEFER",
    )
    row.update(overrides)
    return row


class EventSchemaTests(unittest.TestCase):
    def test_all_event_enums_validate(self) -> None:
        for et in EventType:
            overrides = {"event_type": et.value, "event_id": f"evt_{et.name}"}
            if et == EventType.NO_PLAUSIBLE_CANDIDATE:
                overrides["candidate_count"] = 0
            if et in {
                EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE,
                EventType.MULTIPLE_PLAUSIBLE_CANDIDATES,
            }:
                overrides.update(
                    {
                        "candidate_manifest_path": "manifest.json",
                        "candidate_manifest_sha256": "c" * 64,
                        "candidate_count": 2,
                        "review_window_end_frame": 20,
                    }
                )
            if et != EventType.TARGET_LOST and et not in {
                EventType.TARGET_REENTRY_CANDIDATES_AVAILABLE,
                EventType.MULTIPLE_PLAUSIBLE_CANDIDATES,
                EventType.NO_PLAUSIBLE_CANDIDATE,
                EventType.TARGET_TRACK_CONTINUATION_UNCERTAIN,
                EventType.TRACK_IDENTITY_SWITCH_SUSPECTED,
            }:
                # enrollment-like events still ok with window
                pass
            validate_recovery_event(_base_event(**overrides))

    def test_unknown_event_rejected(self) -> None:
        with self.assertRaises(EventError):
            validate_recovery_event(_base_event(event_type="NOT_A_REAL_EVENT"))

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(HilValidationError):
            validate_recovery_event(_base_event(video_path="../secret.mp4"))

    def test_sha_validation(self) -> None:
        with self.assertRaises(HilValidationError):
            validate_recovery_event(_base_event(video_sha256="deadbeef"))

    def test_frame_range_invalid(self) -> None:
        with self.assertRaises(EventError):
            validate_recovery_event(
                _base_event(review_window_start_frame=50, review_window_end_frame=10)
            )


class CandidateManifestTests(unittest.TestCase):
    def test_manifest_ok(self) -> None:
        m = validate_candidate_manifest(_manifest())
        self.assertEqual(m["candidate_count"], 4)
        self.assertTrue(m["supports_direct_bbox_selection"])
        self.assertEqual(m["candidates"][0]["score_semantics"], "similarity_margin_not_probability")

    def test_count_mismatch(self) -> None:
        with self.assertRaises(CandidateManifestError):
            validate_candidate_manifest(_manifest(candidate_count=99))

    def test_probability_field_rejected(self) -> None:
        c = _candidate(1)
        c["probability"] = 0.9
        with self.assertRaises(CandidateManifestError):
            validate_candidate_manifest(_manifest(candidates=[c], candidate_count=1))

    def test_no_market1501_fallback_model(self) -> None:
        c = _candidate(1)
        c["sportsreid_model_id"] = "osnet_x1_0_market1501"
        with self.assertRaises(CandidateManifestError):
            validate_candidate_manifest(_manifest(candidates=[c], candidate_count=1))

    def test_team_identity_proof_forbidden(self) -> None:
        c = _candidate(1)
        c["team_evidence"] = {"is_identity_proof": True}
        with self.assertRaises(CandidateManifestError):
            validate_candidate_manifest(_manifest(candidates=[c], candidate_count=1))


class DecisionSchemaTests(unittest.TestCase):
    def test_defaults_training_gallery_false(self) -> None:
        d = validate_decision(_decision())
        self.assertFalse(d["training_use_approved"])
        self.assertFalse(d["gallery_use_approved"])

    def test_confirm_requires_selection(self) -> None:
        with self.assertRaises(DecisionError):
            validate_decision(_decision(action="CONFIRM_TARGET"))

    def test_none_unknown_defer_ok(self) -> None:
        for action in ("NONE_OF_THESE", "UNKNOWN", "DEFER"):
            validate_decision(_decision(action=action, decision_id=f"d_{action}"))

    def test_reject_requires_candidate(self) -> None:
        with self.assertRaises(DecisionError):
            validate_decision(_decision(action="REJECT_CANDIDATE"))

    def test_direct_bbox_selection(self) -> None:
        d = validate_decision(
            _decision(
                action="CONFIRM_TARGET",
                direct_bbox_selection=True,
                selected_frame_index=12,
                selected_bbox=[1, 2, 3, 4],
                selected_segment_id="seg_x",
            )
        )
        self.assertTrue(d["direct_bbox_selection"])

    def test_listed_candidate_must_exist(self) -> None:
        manifest = validate_candidate_manifest(_manifest())
        with self.assertRaises(DecisionError):
            validate_decision(
                _decision(
                    event_id="evt_re",
                    action="CONFIRM_TARGET",
                    selected_candidate_id="missing",
                    selected_segment_id="s1",
                    candidate_manifest_sha256="a" * 64,
                    direct_bbox_selection=False,
                ),
                candidate_manifest=manifest,
            )

    def test_no_automatic_confirmation(self) -> None:
        with self.assertRaises(DecisionError):
            validate_decision(_decision(automatic_confirmation=True))
        with self.assertRaises(DecisionError):
            validate_decision(_decision(model_auto_filled=True))

    def test_sportsreid_score_metadata_only_model_id(self) -> None:
        with self.assertRaises(DecisionError):
            validate_decision(
                _decision(
                    displayed_rank=1,
                    displayed_score=0.1,
                    displayed_model_id="osnet_x1_0_market1501",
                    displayed_checkpoint_sha256=SPORTSREID_CHECKPOINT_SHA256,
                )
            )


class AppendOnlyLogTests(unittest.TestCase):
    def test_append_history_supersede_effective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path)
            event = validate_recovery_event(_base_event())
            d1 = validate_decision(
                _decision(
                    decision_id="d1",
                    revision=1,
                    action="CONFIRM_TARGET",
                    direct_bbox_selection=True,
                    selected_frame_index=12,
                    selected_bbox=[1, 2, 3, 4],
                    selected_segment_id="seg_1",
                ),
                event=event,
            )
            log.append(d1, event=event)
            before = path.read_bytes()
            log.revoke_active_decision(
                prior_decision_id="d1",
                new_decision_id="d2",
                reviewer="tester",
                created_at="2026-07-27T00:01:00Z",
                revision=2,
                event=event,
            )
            after = path.read_bytes()
            self.assertTrue(after.startswith(before))
            self.assertEqual(before, path.read_bytes()[: len(before)])
            hist = log.get_history(event_id="evt_1")
            self.assertEqual(len(hist), 2)
            effective = resolve_effective_decisions(hist)
            self.assertEqual(effective["evt_1"]["action"], DecisionAction.REVOKE.value)
            self.assertEqual(
                resolve_event_review_state(event_id="evt_1", decisions=hist),
                EventReviewState.REVOKED_NEEDS_REVIEW,
            )

    def test_existing_records_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path)
            event = validate_recovery_event(_base_event())
            log.append(validate_decision(_decision(decision_id="d1"), event=event), event=event)
            snap = path.read_bytes()
            log.append(
                validate_decision(_decision(decision_id="d2", revision=2, action="UNKNOWN"), event=event),
                event=event,
            )
            self.assertEqual(snap, path.read_bytes()[: len(snap)])

    def test_revision_monotonic_and_duplicate_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path)
            event = validate_recovery_event(_base_event())
            log.append(validate_decision(_decision(decision_id="d1", revision=1), event=event), event=event)
            with self.assertRaises(AppendOnlyLogError):
                log.append(
                    validate_decision(_decision(decision_id="d2", revision=1), event=event),
                    event=event,
                )

    def test_cycle_and_cross_target_supersede_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path)
            event = validate_recovery_event(_base_event())
            log.append(validate_decision(_decision(decision_id="d1", revision=1), event=event), event=event)
            # cross-target
            with self.assertRaises(DecisionError):
                validate_decision(
                    _decision(
                        decision_id="d2",
                        revision=2,
                        action="REVOKE",
                        supersedes_decision_id="d1",
                        target_id="target_OTHER",
                    ),
                    event=validate_recovery_event(_base_event(target_id="target_OTHER", event_id="evt_1")),
                    known_decisions={"d1": log.read_raw()[0]},
                )
            # cross-event
            with self.assertRaises(DecisionError):
                validate_decision(
                    _decision(
                        decision_id="d3",
                        revision=1,
                        event_id="evt_other",
                        action="REVOKE",
                        supersedes_decision_id="d1",
                    ),
                    known_decisions={"d1": log.read_raw()[0]},
                )

    def test_corrupt_log_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            path.write_text('{"decision_id":"x"', encoding="utf-8")  # truncated, no newline
            log = DecisionLog(path)
            with self.assertRaises(AppendOnlyLogError):
                log.read_raw()

    def test_concurrent_writer_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLog(path)
            event = validate_recovery_event(_base_event())
            log.append(validate_decision(_decision(decision_id="d1"), event=event), event=event)

            # Hold exclusive lock in another thread while append attempts.
            ready = threading.Event()
            release = threading.Event()

            def holder() -> None:
                import fcntl

                with path.open("a", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    ready.set()
                    release.wait(timeout=5)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            t = threading.Thread(target=holder)
            t.start()
            self.assertTrue(ready.wait(timeout=2))
            with self.assertRaises(AppendOnlyLogError):
                log.append(
                    validate_decision(_decision(decision_id="d2", revision=2), event=event),
                    event=event,
                )
            release.set()
            t.join(timeout=2)

    def test_deterministic_effective_state(self) -> None:
        decisions = [
            validate_decision(_decision(decision_id="d1", revision=1, action="DEFER")),
            validate_decision(
                _decision(
                    decision_id="d2",
                    revision=2,
                    action="REVOKE",
                    supersedes_decision_id="d1",
                ),
                known_decisions={"d1": validate_decision(_decision(decision_id="d1", revision=1, action="DEFER"))},
            ),
        ]
        a = resolve_effective_decisions(decisions)
        b = resolve_effective_decisions(list(reversed(decisions)))
        # Chain tip is d2 regardless of input order because superseded set is computed from links.
        self.assertEqual(a["evt_1"]["effective_decision_id"], "d2")
        self.assertEqual(b["evt_1"]["effective_decision_id"], "d2")


class HelperBoundaryTests(unittest.TestCase):
    def test_action_enum_complete(self) -> None:
        names = {a.value for a in DecisionAction}
        for required in {
            "CONFIRM_TARGET",
            "REJECT_CANDIDATE",
            "NONE_OF_THESE",
            "UNKNOWN",
            "INVALID_SEGMENT",
            "DEFER",
            "REVOKE",
            "CORRECT_PREVIOUS_DECISION",
        }:
            self.assertIn(required, names)

    def test_r2b_module_not_imported(self) -> None:
        import football_analytics.reid.hil.decisions as mod
        import football_analytics.reid.hil.log as logmod
        import football_analytics.reid.hil.candidates as candmod

        for m in (mod, logmod, candmod):
            source = Path(m.__file__).read_text(encoding="utf-8")
            self.assertNotIn("multiframe_r2b", source)
            self.assertNotIn("import torchreid", source)
            self.assertNotIn("from football_analytics.reid import multiframe_r2b", source)


if __name__ == "__main__":
    unittest.main()
