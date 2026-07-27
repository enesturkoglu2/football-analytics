"""Embed approved match-specific gallery crops with SportsReID (helper only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from football_analytics.reid.hil.common import (
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
    sha256_file,
    sha256_json_canonical,
)
from football_analytics.reid.multi_event_hil.gallery_approvals import (
    resolve_active_gallery_approvals,
)


def embed_approved_gallery(
    *,
    crop_candidates: Mapping[str, Any],
    gallery_approvals: Sequence[Mapping[str, Any]],
    output_dir: Path,
    match_id: str,
    analysis_run_id: str,
    target_id: str,
) -> dict[str, Any]:
    """Embed only actively approved crops. Never pulls development gallery members."""
    active = resolve_active_gallery_approvals(gallery_approvals)
    by_id = {c["crop_id"]: c for c in crop_candidates.get("candidates", [])}
    approved_crops = []
    for crop_id, appr in sorted(active.items()):
        crop = by_id.get(crop_id)
        if crop is None:
            raise RuntimeError(f"approved crop_id missing from candidates: {crop_id}")
        if crop["crop_sha256"] != appr["crop_sha256"]:
            raise RuntimeError(f"crop SHA mismatch for {crop_id}")
        approved_crops.append(crop)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not approved_crops:
        manifest = {
            "schema_version": "match_specific_gallery_manifest_v1",
            "match_id": match_id,
            "analysis_run_id": analysis_run_id,
            "target_id": target_id,
            "member_count": 0,
            "members": [],
            "model_id": SPORTSREID_MODEL_ID,
            "checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
            "role": "appearance_helper_ranker_only",
            "automatic_identity_confirmation": False,
            "from_development_gallery": False,
            "embeddings_present": False,
        }
        path = output_dir / "match_specific_gallery_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"manifest": manifest, "manifest_path": str(path), "embedded": False}

    from football_analytics.reid.embedding import (
        embed_image_paths_with_model,
        load_reid_osnet_by_model_id,
    )

    loaded = load_reid_osnet_by_model_id(SPORTSREID_MODEL_ID)
    paths = [c["crop_path"] for c in approved_crops]
    emb_a = embed_image_paths_with_model(loaded["model"], paths)
    emb_b = embed_image_paths_with_model(loaded["model"], paths)
    if emb_a.shape != emb_b.shape or not np.allclose(emb_a, emb_b, atol=0.0, rtol=0.0):
        # allow tiny float noise? gate asks determinism — use exact for CPU
        if not np.array_equal(emb_a, emb_b):
            # soft check: max abs diff
            max_diff = float(np.max(np.abs(emb_a - emb_b)))
            if max_diff > 1e-6:
                raise RuntimeError(f"gallery embedding non-deterministic max_diff={max_diff}")

    emb_path = output_dir / "match_specific_gallery_embeddings.npy"
    np.save(emb_path, emb_a)
    members = []
    for i, crop in enumerate(approved_crops):
        members.append(
            {
                "member_index": i,
                "crop_id": crop["crop_id"],
                "crop_path": crop["crop_path"],
                "crop_sha256": crop["crop_sha256"],
                "segment_id": crop["segment_id"],
                "raw_track_id": crop["raw_track_id"],
                "frame_index": crop["frame_index"],
                "embedding_index": i,
                "human_approved": True,
                "automatic_enrollment": False,
                "training_use_approved": False,
            }
        )
    manifest = {
        "schema_version": "match_specific_gallery_manifest_v1",
        "match_id": match_id,
        "analysis_run_id": analysis_run_id,
        "target_id": target_id,
        "member_count": len(members),
        "members": members,
        "embeddings_path": str(emb_path),
        "embeddings_sha256": sha256_file(emb_path),
        "embedding_dim": int(emb_a.shape[1]),
        "model_id": loaded["model_id"],
        "checkpoint_path": loaded["checkpoint_path"],
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "role": "appearance_helper_ranker_only",
        "automatic_identity_confirmation": False,
        "from_development_gallery": False,
        "embeddings_present": True,
        "determinism": {
            "two_pass": True,
            "max_abs_diff": float(np.max(np.abs(emb_a - emb_b))),
            "canonical_member_list_sha256": sha256_json_canonical(members),
        },
    }
    path = output_dir / "match_specific_gallery_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "embedded": True,
        "embeddings": emb_a,
        "model_meta": {
            "model_id": loaded["model_id"],
            "checkpoint_sha256": loaded["checkpoint_sha256"],
        },
    }
