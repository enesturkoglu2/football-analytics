#!/usr/bin/env python3
"""CLI entry point for Stage 4B track embedding aggregation.

Src-layout bootstrap adds ``<project_root>/src`` so ``football_analytics``
imports resolve when the script is run directly.
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

from football_analytics.reid.aggregate import (  # noqa: E402
    AggregateError,
    run_aggregate_reid_tracks,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate crop embeddings into per-track L2-mean embeddings. "
            "Does not compute similarity or linking."
        )
    )
    parser.add_argument(
        "--crop-embeddings",
        required=True,
        help="Path to crop_embeddings.npz",
    )
    parser.add_argument(
        "--crop-embeddings-index",
        required=True,
        help="Path to crop_embeddings_index.jsonl",
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to Stage 3 tracks.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for track embeddings + summary.",
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
        result = run_aggregate_reid_tracks(
            crop_embeddings=args.crop_embeddings,
            crop_embeddings_index=args.crop_embeddings_index,
            tracks=args.tracks,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (AggregateError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    shape = result["embedding_shape"]
    print(f"wrote {result['npz_path']}")
    print(f"wrote {result['jsonl_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"crop_embedding_count={result['crop_embedding_count']} "
        f"track_embedding_count={result['track_embedding_count']} "
        f"tracks_with_single_crop={result['tracks_with_single_crop']} "
        f"tracks_with_multiple_crops={result['tracks_with_multiple_crops']} "
        f"embeddings_shape={shape[0]}x{shape[1]} "
        f"dtype={result['embedding_dtype']} "
        f"l2_norm_min={result['l2_norm_min']:.6f} "
        f"l2_norm_max={result['l2_norm_max']:.6f} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
