#!/usr/bin/env python3
"""CLI entry point for Stage 4B ReID crop embedding extraction.

Src-layout bootstrap
--------------------
This project is not installed as an editable package yet (no ``pip install -e .``).
The block below adds only ``<project_root>/src`` to ``sys.path`` so
``football_analytics`` imports resolve when the script is run directly.

``--help`` must not load a checkpoint or import torchreid.
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

from football_analytics.reid.embedding import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    EmbeddingError,
    run_extract_reid_embeddings,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract L2-normalized OSNet ReID embeddings from a crop_manifest.jsonl. "
            "CPU-only. Does not run aggregation or linking."
        )
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to crop_manifest.jsonl from Stage 4B-2.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local OSNet checkpoint path (pretrained=False load).",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected SHA-256 of the checkpoint file (64 hex chars).",
    )
    parser.add_argument(
        "--sn-reid-root",
        required=True,
        help="Path to the sn-reid repository root containing torchreid/.",
    )
    parser.add_argument(
        "--expected-sn-reid-commit",
        required=True,
        help="Expected git HEAD commit of --sn-reid-root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for NPZ, index JSONL, and summary JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"CPU embedding batch size (default: {DEFAULT_BATCH_SIZE}).",
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
        result = run_extract_reid_embeddings(
            crop_manifest=args.crop_manifest,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            sn_reid_root=args.sn_reid_root,
            expected_sn_reid_commit=args.expected_sn_reid_commit,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except (EmbeddingError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    shape = result["embeddings_shape"]
    print(f"wrote {result['npz_path']}")
    print(f"wrote {result['index_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"crop_count={result['crop_count']} "
        f"batch_count={result['batch_count']} "
        f"batch_size={result['batch_size']} "
        f"embeddings_shape={shape[0]}x{shape[1]} "
        f"dtype={result['embedding_dtype']} "
        f"l2_norm_min={result['embedding_l2_norm_min']:.6f} "
        f"l2_norm_max={result['embedding_l2_norm_max']:.6f} "
        f"model={result['model_name']} "
        f"checkpoint_sha256={result['checkpoint_sha256']} "
        f"sn_reid_commit={result['sn_reid_commit']} "
        f"device={result['device']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
