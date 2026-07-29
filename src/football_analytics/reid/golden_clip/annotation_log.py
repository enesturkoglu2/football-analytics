"""Append-only golden-clip annotation log (isolated from product decisions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.golden_clip import SCHEMA_ANNOTATION_EVENT
from football_analytics.reid.golden_clip.schema import GoldenClipError
from football_analytics.reid.hil.common import sha256_file

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class AnnotationLog:
    """Append-only JSONL annotation events; never writes product decision logs."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def sha256(self) -> str:
        return sha256_file(self.path)

    def _lock(self, handle: Any) -> None:
        if fcntl is None:
            raise GoldenClipError("fcntl locking unavailable; fail-closed")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GoldenClipError("concurrent annotation writer; fail-closed") from exc

    def _unlock(self, handle: Any) -> None:
        if fcntl is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def read_raw(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        text = self.path.read_text(encoding="utf-8")
        if not text:
            return []
        if not text.endswith("\n") and text.strip():
            raise GoldenClipError("corrupt annotation log: missing trailing newline")
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise GoldenClipError(f"corrupt annotation log: empty line {line_no}")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenClipError(f"corrupt annotation log line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise GoldenClipError(f"annotation log line {line_no} must be object")
            rows.append(obj)
        return rows

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if event.get("schema_version") != SCHEMA_ANNOTATION_EVENT:
            raise GoldenClipError("annotation event schema_version mismatch")
        if "interval" not in event or not isinstance(event["interval"], Mapping):
            raise GoldenClipError("annotation event requires interval mapping")
        payload = json.dumps(dict(event), ensure_ascii=False, allow_nan=False)
        with self.path.open("a+", encoding="utf-8") as handle:
            self._lock(handle)
            try:
                handle.seek(0, 2)
                handle.write(payload + "\n")
                handle.flush()
            finally:
                self._unlock(handle)
        return dict(event)


def assert_not_product_log(path: Path) -> None:
    """Refuse to treat product decision/approval logs as annotation logs."""
    name = path.name
    if name in {"decision_log.jsonl", "timeline_approval_log.jsonl", "gallery_approval_log.jsonl"}:
        raise GoldenClipError(
            f"refusing to write golden-clip annotations into product log: {path}"
        )
    # path heuristic
    s = str(path)
    if "product_review_package" in s and "annotation_log" not in s:
        raise GoldenClipError(
            f"refusing product_review_package contamination: {path}"
        )
