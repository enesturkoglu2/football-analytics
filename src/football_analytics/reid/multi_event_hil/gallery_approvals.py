"""Append-only match-specific gallery membership approvals (not timeline approvals)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_analytics.reid.hil.common import (
    HilValidationError,
    require_mapping,
    require_sha256,
    require_str,
    sha256_file,
    validate_no_path_traversal,
)

GALLERY_APPROVAL_SCHEMA = "match_specific_gallery_member_approval_v1"


class GalleryApprovalError(HilValidationError):
    pass


class GalleryApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_gallery_approval(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = require_mapping(payload, field="gallery_approval")
    if require_str(row.get("schema_version"), field="schema_version") != GALLERY_APPROVAL_SCHEMA:
        raise GalleryApprovalError("invalid gallery approval schema_version")
    status = require_str(row.get("approval_status"), field="approval_status")
    try:
        GalleryApprovalStatus(status)
    except ValueError as exc:
        raise GalleryApprovalError(f"invalid approval_status: {status}") from exc
    provenance = require_mapping(row.get("provenance", {}), field="provenance")
    if provenance.get("from_development_gallery") or provenance.get("from_holdout"):
        raise GalleryApprovalError("development/holdout gallery members forbidden")
    if provenance.get("automatic_gallery_expansion"):
        raise GalleryApprovalError("automatic_gallery_expansion must be false")
    return {
        "schema_version": GALLERY_APPROVAL_SCHEMA,
        "approval_id": require_str(row.get("approval_id"), field="approval_id"),
        "crop_id": require_str(row.get("crop_id"), field="crop_id"),
        "match_id": require_str(row.get("match_id"), field="match_id"),
        "analysis_run_id": require_str(row.get("analysis_run_id"), field="analysis_run_id"),
        "target_id": require_str(row.get("target_id"), field="target_id"),
        "product_package_id": require_str(
            row.get("product_package_id"), field="product_package_id"
        ),
        "segment_id": require_str(row.get("segment_id"), field="segment_id"),
        "raw_track_id": require_str(row.get("raw_track_id"), field="raw_track_id"),
        "frame_index": int(row.get("frame_index")),
        "crop_path": validate_no_path_traversal(row.get("crop_path"), field="crop_path"),
        "crop_sha256": require_sha256(row.get("crop_sha256"), field="crop_sha256"),
        "reviewer": require_str(row.get("reviewer"), field="reviewer"),
        "approved_at": require_str(row.get("approved_at"), field="approved_at"),
        "approval_status": status,
        "training_use_approved": False,
        "automatic_gallery_expansion": False,
        "comment": require_str(row.get("comment", ""), field="comment", allow_empty=True),
        "supersedes_approval_id": (
            None
            if row.get("supersedes_approval_id") is None
            else require_str(row.get("supersedes_approval_id"), field="supersedes_approval_id")
        ),
        "provenance": dict(provenance),
    }


class GalleryApprovalLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def read_raw(self) -> list[dict[str, Any]]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_gallery_approval(payload)
        ids = {r.get("approval_id") for r in self.read_raw()}
        if validated["approval_id"] in ids:
            raise GalleryApprovalError(f"duplicate approval_id: {validated['approval_id']}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
        return validated

    def sha256(self) -> str:
        return sha256_file(self.path) if self.path.is_file() else ""


def resolve_active_gallery_approvals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_crop: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = validate_gallery_approval(raw)
        by_crop.setdefault(row["crop_id"], []).append(row)
    active: dict[str, dict[str, Any]] = {}
    for crop_id, chain in by_crop.items():
        tip = chain[-1]
        if tip["approval_status"] == GalleryApprovalStatus.APPROVED.value:
            active[crop_id] = tip
    return active


def build_gallery_approval(
    *,
    approval_id: str,
    crop: Mapping[str, Any],
    match_id: str,
    analysis_run_id: str,
    target_id: str,
    product_package_id: str,
    reviewer: str,
    comment: str = "",
    approval_status: str = GalleryApprovalStatus.APPROVED.value,
    supersedes_approval_id: str | None = None,
    approved_at: str | None = None,
) -> dict[str, Any]:
    return validate_gallery_approval(
        {
            "schema_version": GALLERY_APPROVAL_SCHEMA,
            "approval_id": approval_id,
            "crop_id": crop["crop_id"],
            "match_id": match_id,
            "analysis_run_id": analysis_run_id,
            "target_id": target_id,
            "product_package_id": product_package_id,
            "segment_id": crop["segment_id"],
            "raw_track_id": str(crop["raw_track_id"]),
            "frame_index": int(crop["frame_index"]),
            "crop_path": crop["crop_path"],
            "crop_sha256": crop["crop_sha256"],
            "reviewer": reviewer,
            "approved_at": approved_at or _utc_now(),
            "approval_status": approval_status,
            "comment": comment,
            "supersedes_approval_id": supersedes_approval_id,
            "provenance": {
                "match_specific": True,
                "from_development_gallery": False,
                "from_holdout": False,
                "automatic_gallery_expansion": False,
                "explicit_user_gallery_approval": True,
            },
        }
    )
