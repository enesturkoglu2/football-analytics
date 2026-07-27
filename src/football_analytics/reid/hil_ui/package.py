"""Machine-readable target recovery review package (HIL-B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from football_analytics.reid.hil.common import (
    HilValidationError,
    require_mapping,
    require_sha256,
    require_str,
    sha256_file,
    validate_no_path_traversal,
)

REVIEW_PACKAGE_SCHEMA_VERSION = "target_recovery_review_package_v1"


class ReviewPackageError(HilValidationError):
    """Raised when a review package fails validation."""


def _resolve_under_root(root: Path, relative_or_abs: str, *, field: str) -> Path:
    """Resolve a path and ensure it stays under an allowed root when relative."""
    text = validate_no_path_traversal(relative_or_abs, field=field)
    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    return resolved


def _ensure_under_any(path: Path, roots: list[Path], *, field: str) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ReviewPackageError(
        f"{field} escapes allowed roots: {resolved} not under {[str(r) for r in roots]}"
    )


def _verify_sha(path: Path, expected: str, *, field: str) -> str:
    if not path.is_file():
        raise ReviewPackageError(f"{field} file missing: {path}")
    digest = sha256_file(path)
    if digest != expected.lower():
        raise ReviewPackageError(
            f"{field} SHA-256 mismatch for {path}: expected {expected}, got {digest}"
        )
    return digest


def validate_review_package(
    payload: Mapping[str, Any],
    *,
    package_dir: Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Validate review package mapping; optionally verify on-disk SHA values."""
    row = require_mapping(payload, field="review_package")
    schema = require_str(row.get("schema_version"), field="schema_version")
    if schema != REVIEW_PACKAGE_SCHEMA_VERSION:
        raise ReviewPackageError(
            f"schema_version must be {REVIEW_PACKAGE_SCHEMA_VERSION}, got {schema!r}"
        )

    package_id = require_str(row.get("package_id"), field="package_id")
    project_id = require_str(row.get("project_id"), field="project_id")
    run_id = require_str(row.get("run_id"), field="run_id")
    target_id = require_str(row.get("target_id"), field="target_id")

    source_video_path = validate_no_path_traversal(
        row.get("source_video_path"), field="source_video_path"
    )
    source_video_sha256 = require_sha256(
        row.get("source_video_sha256"), field="source_video_sha256"
    )
    event_manifest_path = validate_no_path_traversal(
        row.get("event_manifest_path"), field="event_manifest_path"
    )
    event_manifest_sha256 = require_sha256(
        row.get("event_manifest_sha256"), field="event_manifest_sha256"
    )

    cand_paths_raw = row.get("candidate_manifest_paths")
    if not isinstance(cand_paths_raw, list) or not cand_paths_raw:
        raise ReviewPackageError("candidate_manifest_paths must be a non-empty list")
    candidate_manifest_paths = [
        validate_no_path_traversal(p, field=f"candidate_manifest_paths[{i}]")
        for i, p in enumerate(cand_paths_raw)
    ]

    cand_shas_raw = row.get("candidate_manifest_sha256")
    if not isinstance(cand_shas_raw, dict):
        raise ReviewPackageError("candidate_manifest_sha256 must be a mapping")
    candidate_manifest_sha256 = {
        validate_no_path_traversal(k, field=f"candidate_manifest_sha256.key[{k}]"): require_sha256(
            v, field=f"candidate_manifest_sha256[{k}]"
        )
        for k, v in cand_shas_raw.items()
    }
    for path in candidate_manifest_paths:
        if path not in candidate_manifest_sha256:
            raise ReviewPackageError(
                f"candidate_manifest_sha256 missing entry for {path!r}"
            )

    decision_log_path = validate_no_path_traversal(
        row.get("decision_log_path"), field="decision_log_path"
    )

    target_gallery_reference = row.get("target_gallery_reference")
    if target_gallery_reference is not None:
        if not isinstance(target_gallery_reference, dict):
            raise ReviewPackageError("target_gallery_reference must be a mapping or null")
    model_metadata = row.get("model_metadata", {})
    if not isinstance(model_metadata, dict):
        raise ReviewPackageError("model_metadata must be a mapping")

    media_root = validate_no_path_traversal(row.get("media_root"), field="media_root")
    ro_raw = row.get("read_only_source_roots")
    if not isinstance(ro_raw, list) or not ro_raw:
        raise ReviewPackageError("read_only_source_roots must be a non-empty list")
    read_only_source_roots = [
        validate_no_path_traversal(p, field=f"read_only_source_roots[{i}]")
        for i, p in enumerate(ro_raw)
    ]
    writable_session_root = validate_no_path_traversal(
        row.get("writable_session_root"), field="writable_session_root"
    )
    created_at = require_str(row.get("created_at"), field="created_at")
    provenance = row.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ReviewPackageError("provenance must be a mapping")

    media_status = require_str(
        row.get("media_status", "verified"), field="media_status", allow_empty=False
    )
    if media_status not in {"verified", "unavailable", "synthetic_fixture"}:
        raise ReviewPackageError(f"unsupported media_status: {media_status!r}")

    normalized = {
        "schema_version": REVIEW_PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "project_id": project_id,
        "run_id": run_id,
        "target_id": target_id,
        "source_video_path": source_video_path,
        "source_video_sha256": source_video_sha256,
        "event_manifest_path": event_manifest_path,
        "event_manifest_sha256": event_manifest_sha256,
        "candidate_manifest_paths": candidate_manifest_paths,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "decision_log_path": decision_log_path,
        "target_gallery_reference": target_gallery_reference,
        "model_metadata": dict(model_metadata),
        "media_root": media_root,
        "read_only_source_roots": read_only_source_roots,
        "writable_session_root": writable_session_root,
        "created_at": created_at,
        "provenance": dict(provenance),
        "media_status": media_status,
        "automatic_identity_assignment": False,
        "automatic_gallery_expansion": False,
    }

    if not verify_sources:
        return normalized
    if package_dir is None:
        raise ReviewPackageError("package_dir required when verify_sources=True")

    package_dir = package_dir.resolve()
    ro_roots = [
        _resolve_under_root(package_dir, p, field=f"read_only_source_roots[{i}]")
        if not Path(p).is_absolute()
        else Path(p).resolve()
        for i, p in enumerate(read_only_source_roots)
    ]
    writable = (
        Path(writable_session_root).resolve()
        if Path(writable_session_root).is_absolute()
        else (package_dir / writable_session_root).resolve()
    )
    writable.mkdir(parents=True, exist_ok=True)

    # Writable root must not equal / nest inside a read-only source root incorrectly
    # in reverse: write rejection for frozen sources is enforced separately.
    for root in ro_roots:
        try:
            writable.relative_to(root)
            raise ReviewPackageError(
                "writable_session_root must not be inside a read_only_source_root"
            )
        except ValueError:
            pass

    video_path = (
        Path(source_video_path)
        if Path(source_video_path).is_absolute()
        else _resolve_under_root(package_dir, source_video_path, field="source_video_path")
    )
    if media_status == "unavailable":
        normalized["source_video_resolved"] = str(video_path)
        normalized["source_video_available"] = False
    else:
        _ensure_under_any(
            video_path,
            ro_roots + [package_dir, Path(media_root).resolve() if Path(media_root).is_absolute() else (package_dir / media_root).resolve()],
            field="source_video_path",
        )
        _verify_sha(video_path, source_video_sha256, field="source_video_sha256")
        normalized["source_video_resolved"] = str(video_path.resolve())
        normalized["source_video_available"] = True

    event_path = (
        Path(event_manifest_path)
        if Path(event_manifest_path).is_absolute()
        else _resolve_under_root(package_dir, event_manifest_path, field="event_manifest_path")
    )
    _verify_sha(event_path, event_manifest_sha256, field="event_manifest_sha256")
    normalized["event_manifest_resolved"] = str(event_path.resolve())

    resolved_cands: dict[str, str] = {}
    for rel in candidate_manifest_paths:
        cpath = Path(rel) if Path(rel).is_absolute() else _resolve_under_root(
            package_dir, rel, field="candidate_manifest_path"
        )
        _verify_sha(cpath, candidate_manifest_sha256[rel], field="candidate_manifest_sha256")
        resolved_cands[rel] = str(cpath.resolve())
    normalized["candidate_manifest_resolved"] = resolved_cands

    dlog = (
        Path(decision_log_path)
        if Path(decision_log_path).is_absolute()
        else (writable / Path(decision_log_path).name)
        if not str(decision_log_path).startswith(str(writable))
        else Path(decision_log_path)
    )
    # Prefer resolving decision log under writable session root.
    if not Path(decision_log_path).is_absolute():
        dlog = (writable / decision_log_path).resolve()
    else:
        dlog = Path(decision_log_path).resolve()
        try:
            dlog.relative_to(writable)
        except ValueError as exc:
            raise ReviewPackageError(
                "decision_log_path must resolve under writable_session_root"
            ) from exc
    dlog.parent.mkdir(parents=True, exist_ok=True)
    if not dlog.exists():
        dlog.touch()
    normalized["decision_log_resolved"] = str(dlog)
    normalized["writable_session_root_resolved"] = str(writable)
    normalized["read_only_source_roots_resolved"] = [str(r) for r in ro_roots]
    return normalized


def load_and_validate_review_package(path: str | Path) -> dict[str, Any]:
    package_path = Path(path).expanduser().resolve()
    if not package_path.is_file():
        raise ReviewPackageError(f"review package not found: {package_path}")
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return validate_review_package(payload, package_dir=package_path.parent, verify_sources=True)


def assert_path_not_writable_to_frozen(path: Path, read_only_roots: list[Path]) -> None:
    """Fail-closed if a write target resolves under a frozen/read-only source root."""
    resolved = path.resolve()
    for root in read_only_roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ReviewPackageError(
            f"refusing write into read-only source root: {resolved} under {root}"
        )
