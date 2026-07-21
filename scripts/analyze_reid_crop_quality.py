#!/usr/bin/env python3
"""CLI entry point for Stage 5A2A ReID crop quality analysis.

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

from football_analytics.reid.quality import (  # noqa: E402
    QualityError,
    run_analyze_reid_crop_quality,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ReID crop image quality and tracking-bbox contamination "
            "evidence. Measurement-only: no thresholds, no crop deletion, "
            "no embedding/linking changes."
        )
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to crop_manifest.jsonl (Stage 4B schema).",
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to Stage 3 tracks.jsonl used for bbox contamination evidence.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for quality JSONL/JSON artifacts.",
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
        result = run_analyze_reid_crop_quality(
            crop_manifest=args.crop_manifest,
            tracks=args.tracks,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (QualityError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    lap = result["laplacian_variance"]
    union = result["union_other_person_crop_coverage"]
    print(f"wrote {result['crop_quality_path']}")
    print(f"wrote {result['track_quality_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"crop_count={result['crop_count']} "
        f"track_count={result['track_count']} "
        f"crops_touching_any_frame_edge={result['crops_touching_any_frame_edge']} "
        f"crops_with_other_person_overlap="
        f"{result['crops_with_other_person_overlap']} "
        f"crops_with_other_person_center_inside="
        f"{result['crops_with_other_person_center_inside']} "
        f"laplacian_min={lap['min']} "
        f"laplacian_median={lap['median']} "
        f"laplacian_max={lap['max']} "
        f"contamination_union_min={union['min']} "
        f"contamination_union_median={union['median']} "
        f"contamination_union_max={union['max']} "
        f"quality_threshold={result['quality_threshold']} "
        f"contamination_threshold={result['contamination_threshold']} "
        f"automatic_exclusion={result['automatic_exclusion_performed']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
