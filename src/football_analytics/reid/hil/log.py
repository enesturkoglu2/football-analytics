"""Append-only decision log storage (JSONL)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.hil.common import HilValidationError, sha256_file
from football_analytics.reid.hil.decisions import (
    DecisionAction,
    DecisionError,
    DecisionStatus,
    build_decision,
    validate_decision,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


class AppendOnlyLogError(HilValidationError):
    """Raised when append-only log integrity or locking fails."""


def compute_log_sha256(path: Path) -> str:
    if not path.is_file():
        raise AppendOnlyLogError(f"decision log not found: {path}")
    return sha256_file(path)


class DecisionLog:
    """Append-only JSONL decision log with exclusive locking."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _lock_exclusive(self, handle: Any) -> None:
        if fcntl is None:
            raise AppendOnlyLogError("fcntl locking unavailable; fail-closed")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AppendOnlyLogError(
                "concurrent writer detected; fail-closed"
            ) from exc
        except OSError as exc:
            raise AppendOnlyLogError(f"could not acquire exclusive lock: {exc}") from exc

    def _unlock(self, handle: Any) -> None:
        if fcntl is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def read_raw(self) -> list[dict[str, Any]]:
        """Read and parse the full raw log; detect truncation/corruption."""
        if not self.path.is_file():
            return []
        text = self.path.read_text(encoding="utf-8")
        if not text:
            return []
        if not text.endswith("\n") and text.strip():
            raise AppendOnlyLogError(
                "corrupt or truncated decision log: missing trailing newline"
            )
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise AppendOnlyLogError(f"corrupt decision log: empty line at {line_no}")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AppendOnlyLogError(
                    f"corrupt decision log at line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise AppendOnlyLogError(
                    f"corrupt decision log at line {line_no}: expected object"
                )
            rows.append(obj)
        return rows

    def validate_full_log(
        self,
        *,
        events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
        manifests_by_event: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        raw = self.read_raw()
        validated: list[dict[str, Any]] = []
        known: dict[str, dict[str, Any]] = {}
        max_revision_by_event: dict[str, int] = {}
        decision_ids: set[str] = set()

        for row in raw:
            event = None
            manifest = None
            if events_by_id is not None:
                event = events_by_id.get(str(row.get("event_id")))
                if event is None:
                    raise AppendOnlyLogError(
                        f"decision references unknown event_id: {row.get('event_id')!r}"
                    )
            if manifests_by_event is not None:
                manifest = manifests_by_event.get(str(row.get("event_id")))

            decision = validate_decision(
                row,
                event=event,
                candidate_manifest=manifest,
                known_decisions=known,
            )
            if decision["decision_id"] in decision_ids:
                raise AppendOnlyLogError(
                    f"duplicate decision_id in log: {decision['decision_id']}"
                )
            decision_ids.add(decision["decision_id"])

            event_id = decision["event_id"]
            rev = int(decision["revision"])
            prior_max = max_revision_by_event.get(event_id)
            if prior_max is not None and rev <= prior_max:
                raise AppendOnlyLogError(
                    f"revision not monotonic for event {event_id}: {rev} <= {prior_max}"
                )
            max_revision_by_event[event_id] = rev

            known[decision["decision_id"]] = decision
            validated.append(decision)
        return validated

    def append(
        self,
        decision: Mapping[str, Any],
        *,
        event: Mapping[str, Any] | None = None,
        candidate_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically append one validated decision. Never mutates prior rows."""
        existing = self.read_raw()
        known = {str(r["decision_id"]): r for r in existing if "decision_id" in r}
        validated = validate_decision(
            decision,
            event=event,
            candidate_manifest=candidate_manifest,
            known_decisions=known,
        )

        # Per-event revision uniqueness / monotonicity
        prior_revs = [
            int(r["revision"])
            for r in existing
            if r.get("event_id") == validated["event_id"] and "revision" in r
        ]
        if validated["revision"] in prior_revs:
            raise AppendOnlyLogError(
                f"revision {validated['revision']} already exists for event "
                f"{validated['event_id']}"
            )
        if prior_revs and validated["revision"] <= max(prior_revs):
            raise AppendOnlyLogError(
                f"revision must increase for event {validated['event_id']}: "
                f"got {validated['revision']} max_existing={max(prior_revs)}"
            )

        if validated["decision_id"] in known:
            raise AppendOnlyLogError(
                f"decision_id already exists (append-only): {validated['decision_id']}"
            )

        line = json.dumps(validated, ensure_ascii=False, allow_nan=False) + "\n"
        encoded = line.encode("utf-8")
        before = self.path.read_bytes() if self.path.exists() else b""

        try:
            with self.path.open("a", encoding="utf-8") as handle:
                self._lock_exclusive(handle)
                try:
                    current = self.path.read_bytes()
                    if current != before:
                        raise AppendOnlyLogError(
                            "concurrent writer modified log before append; fail-closed"
                        )
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    self._unlock(handle)
        except AppendOnlyLogError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppendOnlyLogError(f"atomic append failed: {exc}") from exc

        after = self.path.read_bytes()
        if not after.startswith(before):
            raise AppendOnlyLogError(
                "append-only integrity failure: prior bytes changed"
            )
        if not after.endswith(encoded):
            raise AppendOnlyLogError("append-only integrity failure: new record missing")
        return validated

    def create_superseding_decision(
        self,
        *,
        prior_decision_id: str,
        new_decision_id: str,
        action: str | DecisionAction,
        reviewer: str,
        created_at: str,
        revision: int,
        comment: str = "",
        mark_prior_status: str = DecisionStatus.SUPERSEDED.value,
        event: Mapping[str, Any] | None = None,
        candidate_manifest: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Create a superseding decision. Prior row is NOT rewritten.

        Derived views treat prior as superseded via the chain; optional
        ``mark_prior_status`` is recorded only on the new decision metadata.
        """
        raw = self.read_raw()
        prior = None
        for row in raw:
            if row.get("decision_id") == prior_decision_id:
                prior = row
                break
        if prior is None:
            raise AppendOnlyLogError(f"prior decision not found: {prior_decision_id}")

        action_value = action.value if isinstance(action, DecisionAction) else action
        payload = build_decision(
            decision_id=new_decision_id,
            project_id=str(overrides.get("project_id", prior["project_id"])),
            run_id=str(overrides.get("run_id", prior["run_id"])),
            target_id=str(overrides.get("target_id", prior["target_id"])),
            event_id=str(overrides.get("event_id", prior["event_id"])),
            video_id=str(overrides.get("video_id", prior["video_id"])),
            video_path=str(overrides.get("video_path", prior["video_path"])),
            video_sha256=str(overrides.get("video_sha256", prior["video_sha256"])),
            reviewer=reviewer,
            created_at=created_at,
            revision=revision,
            action=action_value,
            supersedes_decision_id=prior_decision_id,
            selected_candidate_id=overrides.get(
                "selected_candidate_id", prior.get("selected_candidate_id")
            ),
            selected_segment_id=overrides.get(
                "selected_segment_id", prior.get("selected_segment_id")
            ),
            selected_raw_track_id=overrides.get(
                "selected_raw_track_id", prior.get("selected_raw_track_id")
            ),
            selected_frame_index=overrides.get(
                "selected_frame_index", prior.get("selected_frame_index")
            ),
            selected_bbox_xyxy=overrides.get("selected_bbox", prior.get("selected_bbox")),
            direct_bbox_selection=bool(
                overrides.get("direct_bbox_selection", prior.get("direct_bbox_selection", False))
            ),
            candidate_manifest_path=overrides.get(
                "candidate_manifest_path", prior.get("candidate_manifest_path")
            ),
            candidate_manifest_sha256=overrides.get(
                "candidate_manifest_sha256", prior.get("candidate_manifest_sha256")
            ),
            displayed_model_id=overrides.get(
                "displayed_model_id", prior.get("displayed_model_id")
            ),
            displayed_checkpoint_sha256=overrides.get(
                "displayed_checkpoint_sha256", prior.get("displayed_checkpoint_sha256")
            ),
            displayed_rank=overrides.get("displayed_rank", prior.get("displayed_rank")),
            displayed_score=overrides.get("displayed_score", prior.get("displayed_score")),
            displayed_T_max=overrides.get("displayed_T_max", prior.get("displayed_T_max")),
            displayed_D_max=overrides.get("displayed_D_max", prior.get("displayed_D_max")),
            evidence_paths=list(
                overrides.get("evidence_paths", prior.get("evidence_paths") or [])
            ),
            evidence_sha256=list(
                overrides.get("evidence_sha256", prior.get("evidence_sha256") or [])
            ),
            comment=comment,
            confidence=str(overrides.get("confidence", prior.get("confidence", "unknown"))),
            status=DecisionStatus.ACTIVE.value,
            training_use_approved=False,
            gallery_use_approved=False,
        )
        payload["prior_status_intent"] = mark_prior_status
        return self.append(payload, event=event, candidate_manifest=candidate_manifest)

    def revoke_active_decision(
        self,
        *,
        prior_decision_id: str,
        new_decision_id: str,
        reviewer: str,
        created_at: str,
        revision: int,
        comment: str = "",
        event: Mapping[str, Any] | None = None,
        candidate_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.create_superseding_decision(
            prior_decision_id=prior_decision_id,
            new_decision_id=new_decision_id,
            action=DecisionAction.REVOKE,
            reviewer=reviewer,
            created_at=created_at,
            revision=revision,
            comment=comment,
            mark_prior_status=DecisionStatus.REVOKED.value,
            event=event,
            candidate_manifest=candidate_manifest,
            selected_candidate_id=None,
            selected_segment_id=None,
            selected_raw_track_id=None,
            selected_frame_index=None,
            selected_bbox=None,
            direct_bbox_selection=False,
            confidence="unknown",
        )

    def get_history(self, *, event_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.validate_full_log()
        if event_id is None:
            return rows
        return [r for r in rows if r["event_id"] == event_id]

    def integrity_report(self) -> dict[str, Any]:
        rows = self.read_raw()
        return {
            "path": str(self.path),
            "record_count": len(rows),
            "sha256": compute_log_sha256(self.path),
            "append_only": True,
            "trailing_newline_ok": True,
        }
