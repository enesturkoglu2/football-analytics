#!/usr/bin/env python3
"""CLI entry point for Stage 5B1A ReID torso kit-descriptor analysis.

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

from football_analytics.reid.kit import (  # noqa: E402
    KitError,
    run_analyze_reid_kit_descriptors,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure torso-oriented coarse kit/color descriptors for ReID crops. "
            "Measurement-only: no team assignment, clustering, linking, "
            "crop exclusion, or quality weighting."
        )
    )
    parser.add_argument(
        "--crop-manifest",
        required=True,
        help="Path to crop_manifest.jsonl (Stage 4B schema).",
    )
    parser.add_argument(
        "--quality-signals",
        required=True,
        help="Path to crop_quality_signals.jsonl (Stage 5A schema).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to kit descriptor measurement YAML (required; no default).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for kit descriptor JSONL/JSON artifacts.",
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
        result = run_analyze_reid_kit_descriptors(
            crop_manifest=args.crop_manifest,
            quality_signals=args.quality_signals,
            config=args.config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (KitError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    chrom = result["chromatic_pixel_ratio"]
    bounds = result["torso_fraction_bounds"]
    print(f"wrote {result['crop_kit_path']}")
    print(f"wrote {result['track_kit_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"crop_count={result['crop_count']} "
        f"track_count={result['track_count']} "
        f"torso_region={result['torso_region_name']} "
        f"torso_bounds="
        f"[{bounds['x_min_fraction']},{bounds['x_max_fraction']}]x"
        f"[{bounds['y_min_fraction']},{bounds['y_max_fraction']}] "
        f"crop_dominant_families={result['crop_dominant_color_family_histogram']} "
        f"track_dominant_families={result['track_dominant_color_family_histogram']} "
        f"chromatic_ratio_min={chrom['min']} "
        f"chromatic_ratio_median={chrom['median']} "
        f"chromatic_ratio_max={chrom['max']} "
        f"quality_weighting={result['quality_weighting_performed']} "
        f"crop_exclusion={result['crop_exclusion_performed']} "
        f"team_assignment={result['automatic_team_assignment_performed']} "
        f"forced_clustering={result['forced_two_team_clustering_performed']} "
        f"kit_threshold={result['kit_similarity_threshold']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
