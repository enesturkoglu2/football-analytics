#!/usr/bin/env python3
"""CLI entry point for Stage 4B ReID candidate pair generation.

Src-layout bootstrap adds ``<project_root>/src`` so ``football_analytics``
imports resolve when the script is run directly.

No similarity threshold and no automatic linking in this stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if not _SRC_DIR.is_dir():
    print(f"error: expected src directory at {_SRC_DIR}", file=sys.stderr)
    sys.exit(2)
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.candidates import (  # noqa: E402
    CandidateError,
    run_build_reid_candidates,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build unordered track candidate pairs with cosine similarity and "
            "temporal conflict metadata. No threshold and no linking."
        )
    )
    parser.add_argument(
        "--track-embeddings",
        required=True,
        help="Path to track_embeddings.npz",
    )
    parser.add_argument(
        "--track-embeddings-index",
        required=True,
        help="Path to track_embeddings.jsonl",
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to Stage 3 tracks.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for candidate_pairs.jsonl + summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory atomically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_build_reid_candidates(
            track_embeddings=args.track_embeddings,
            track_embeddings_index=args.track_embeddings_index,
            tracks=args.tracks,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (CandidateError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['pairs_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"track_count={result['track_count']} "
        f"total_pairs={result['total_pairs']} "
        f"exact_frame_conflict_pairs={result['exact_frame_conflict_pairs']} "
        f"eligible_unthresholded_pairs={result['eligible_unthresholded_pairs']} "
        f"span_only_overlap_pairs={result['span_only_overlap_pairs']} "
        f"cosine_min={result['cosine_min']} "
        f"cosine_median={result['cosine_median']} "
        f"cosine_max={result['cosine_max']} "
        "similarity_threshold=null "
        "automatic_linking_performed=false "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
