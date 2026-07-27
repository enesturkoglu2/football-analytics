"""Review session loader — events, manifests, decision log (no Streamlit)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_analytics.reid.hil.candidates import validate_candidate_manifest
from football_analytics.reid.hil.events import validate_recovery_event
from football_analytics.reid.hil.log import DecisionLog
from football_analytics.reid.hil_ui.package import load_and_validate_review_package
from football_analytics.reid.hil_ui.queue import build_queue_rows, filter_queue


@dataclass
class ReviewSession:
    package: dict[str, Any]
    package_path: Path
    events: list[dict[str, Any]]
    manifests_by_event: dict[str, dict[str, Any]]
    decision_log: DecisionLog
    event_index: int = 0
    selection: dict[str, Any] = field(default_factory=dict)

    @property
    def current_event(self) -> dict[str, Any]:
        return self.events[self.event_index]

    def set_event_index(self, index: int) -> None:
        if index < 0 or index >= len(self.events):
            raise IndexError("event index out of range")
        self.event_index = index
        self.selection = {}

    def next_event(self) -> None:
        self.set_event_index(min(self.event_index + 1, len(self.events) - 1))

    def previous_event(self) -> None:
        self.set_event_index(max(self.event_index - 1, 0))

    def current_manifest(self) -> dict[str, Any] | None:
        return self.manifests_by_event.get(self.current_event["event_id"])

    def queue(self, filter_name: str = "all") -> list[dict[str, Any]]:
        rows = build_queue_rows(self.events, self.decision_log.read_raw())
        return filter_queue(rows, filter_name=filter_name)

    def package_summary(self) -> dict[str, Any]:
        decisions = self.decision_log.read_raw()
        queue = build_queue_rows(self.events, decisions)
        unresolved = filter_queue(queue, filter_name="unresolved")
        return {
            "package_id": self.package["package_id"],
            "project_id": self.package["project_id"],
            "run_id": self.package["run_id"],
            "target_id": self.package["target_id"],
            "source_video_path": self.package["source_video_path"],
            "source_video_sha256": self.package["source_video_sha256"],
            "source_video_available": self.package.get("source_video_available"),
            "media_status": self.package.get("media_status"),
            "event_count": len(self.events),
            "unresolved_count": len(unresolved),
            "resolved_count": len(self.events) - len(unresolved),
            "decision_log_path": self.package.get("decision_log_resolved"),
            "read_only_source_roots": self.package.get("read_only_source_roots_resolved"),
            "writable_session_root": self.package.get("writable_session_root_resolved"),
        }


def _load_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict) and "events" in payload:
            rows = payload["events"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"unsupported event manifest shape: {path}")
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [validate_recovery_event(r) for r in rows]


def open_review_session(package_path: str | Path) -> ReviewSession:
    path = Path(package_path).expanduser().resolve()
    package = load_and_validate_review_package(path)
    events = _load_events(Path(package["event_manifest_resolved"]))
    manifests: dict[str, dict[str, Any]] = {}
    for rel, resolved in package["candidate_manifest_resolved"].items():
        man = validate_candidate_manifest(json.loads(Path(resolved).read_text(encoding="utf-8")))
        manifests[str(man["event_id"])] = man
        _ = rel
    decision_log = DecisionLog(package["decision_log_resolved"])
    return ReviewSession(
        package=package,
        package_path=path,
        events=events,
        manifests_by_event=manifests,
        decision_log=decision_log,
    )
