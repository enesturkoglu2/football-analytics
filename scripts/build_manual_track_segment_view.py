#!/usr/bin/env python3
"""CLI entry point for Stage 5B3F non-destructive manual track segment view.

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

from football_analytics.reid.segments import (  # noqa: E402
    SegmentError,
    run_build_manual_track_segment_view,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive derived track-segment view from frozen "
            "manual segment decisions. Does not mutate raw tracks, create "
            "observations, interpolate gaps, rewrite global IDs, or run ReID."
        )
    )
    parser.add_argument("--tracks", required=True, help="Raw tracks.jsonl")
    parser.add_argument(
        "--segmentation-policy",
        required=True,
        help="manual_track_segmentation_policy_stage5b3.yaml",
    )
    parser.add_argument(
        "--segment-decisions",
        required=True,
        help="manual_track_segment_decisions_stage5b3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Segment-view output directory",
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
        result = run_build_manual_track_segment_view(
            tracks=args.tracks,
            segmentation_policy=args.segmentation_policy,
            segment_decisions=args.segment_decisions,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (SegmentError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['segments_path']}")
    print(f"wrote {result['segment_observations_path']}")
    print(f"wrote {result['unassigned_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"raw_track_count={result['raw_track_count']} "
        f"raw_observation_count={result['raw_observation_count']} "
        f"split_candidate_raw_track_count="
        f"{result['split_candidate_raw_track_count']} "
        f"no_split_control_raw_track_count="
        f"{result['no_split_control_raw_track_count']} "
        f"preserved_full_raw_track_count="
        f"{result['preserved_full_raw_track_count']} "
        f"manual_split_segment_count={result['manual_split_segment_count']} "
        f"total_segment_count={result['total_segment_count']} "
        f"assigned_observation_count={result['assigned_observation_count']} "
        f"unassigned_observation_count={result['unassigned_observation_count']} "
        f"source_coverage_observation_count="
        f"{result['source_coverage_observation_count']} "
        f"created_observation_count={result['created_observation_count']} "
        f"interpolated_observation_count="
        f"{result['interpolated_observation_count']} "
        f"deleted_observation_count={result['deleted_observation_count']} "
        f"raw_tracks_mutated={result['raw_tracks_mutated']} "
        f"global_id_rewrite={result['global_id_rewrite_performed']} "
        f"reid_recomputation={result['reid_recomputation_performed']} "
        f"elapsed_sec={result['elapsed_sec']} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
