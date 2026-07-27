"""Helper-only SportsReID ranking against match-specific gallery (no auto-confirm)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from football_analytics.reid.hil.common import (
    SPORTSREID_CHECKPOINT_SHA256,
    SPORTSREID_MODEL_ID,
    sha256_file,
)


def rank_candidates_against_gallery(
    *,
    gallery_embeddings: np.ndarray,
    candidate_crop_paths: Sequence[Path],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Return helper ranks. Scores are similarity margins, not probabilities."""
    if gallery_embeddings.size == 0:
        raise RuntimeError("empty gallery embeddings")
    from football_analytics.reid.embedding import (
        embed_image_paths_with_model,
        load_reid_osnet_by_model_id,
    )

    loaded = load_reid_osnet_by_model_id(SPORTSREID_MODEL_ID)
    emb = embed_image_paths_with_model(loaded["model"], list(candidate_crop_paths))
    # cosine vs gallery mean prototype
    proto = gallery_embeddings.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-12)
    norms = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = norms @ proto
    order = list(np.argsort(-sims))
    rows = []
    for rank_i, idx in enumerate(order, start=1):
        rows.append(
            {
                "candidate_id": candidate_ids[idx],
                "appearance_rank": rank_i,
                "T_max": float(sims[idx]),
                "D_max": None,
                "S": float(sims[idx] - (float(sims[order[1]]) if len(order) > 1 else 0.0)),
                "score_semantics": "similarity_margin_not_probability",
                "sportsreid_model_id": SPORTSREID_MODEL_ID,
                "sportsreid_checkpoint_sha256": SPORTSREID_CHECKPOINT_SHA256,
                "automatic_confirm": False,
            }
        )
    return {
        "schema_version": "mehil_r1_helper_ranking_v1",
        "model_id": SPORTSREID_MODEL_ID,
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "hides_candidates": False,
        "is_probability": False,
        "ranks": rows,
    }


def write_ranking_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256_file(path)
