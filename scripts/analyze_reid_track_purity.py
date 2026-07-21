#!/usr/bin/env python3
"""CLI entry point for Stage 5B3A ReID track purity / kit-change audit.

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

from football_analytics.reid.purity import (  # noqa: E402
    PurityError,
    run_analyze_reid_track_purity,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit within-track kit-descriptor change and torso-region "
            "tracking-bbox contamination context. Measurement-only: no "
            "automatic split, delete, global-ID rewrite, team assignment, "
            "or composite change score."
        )
    )
    parser.add_argument("--crop-manifest", required=True, help="crop_manifest.jsonl")
    parser.add_argument(
        "--quality-signals", required=True, help="crop_quality_signals.jsonl"
    )
    parser.add_argument(
        "--crop-kit-descriptors", required=True, help="crop_kit_descriptors.jsonl"
    )
    parser.add_argument(
        "--track-kit-descriptors", required=True, help="track_kit_descriptors.jsonl"
    )
    parser.add_argument("--tracks", required=True, help="Stage 3 tracks.jsonl")
    parser.add_argument(
        "--kit-config", required=True, help="kit_descriptor_stage5b.yaml"
    )
    parser.add_argument(
        "--audit-config", required=True, help="track_purity_audit_stage5b3.yaml"
    )
    parser.add_argument("--output-dir", required=True, help="Audit output directory")
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
        result = run_analyze_reid_track_purity(
            crop_manifest=args.crop_manifest,
            quality_signals=args.quality_signals,
            crop_kit_descriptors=args.crop_kit_descriptors,
            track_kit_descriptors=args.track_kit_descriptors,
            tracks=args.tracks,
            kit_config=args.kit_config,
            audit_config=args.audit_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (PurityError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cf = result["color_family_l1"]
    lab = result["lab_mean_distance_normalized"]
    print(f"wrote {result['transition_path']}")
    print(f"wrote {result['track_purity_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"crop_count={result['crop_count']} "
        f"track_count={result['track_count']} "
        f"multi_crop_tracks={result['multi_crop_track_count']} "
        f"single_crop_tracks={result['single_crop_track_count']} "
        f"transition_count={result['transition_count']} "
        f"tracks_with_family_change={result['tracks_with_dominant_family_change']} "
        f"transitions_with_family_change="
        f"{result['transitions_with_dominant_family_change']} "
        f"torso_overlap_crops={result['crops_with_torso_other_person_overlap']} "
        f"color_family_l1_min={cf['min']} "
        f"color_family_l1_median={cf['median']} "
        f"color_family_l1_max={cf['max']} "
        f"lab_dist_min={lab['min']} "
        f"lab_dist_median={lab['median']} "
        f"lab_dist_max={lab['max']} "
        f"change_threshold={result['change_threshold']} "
        f"split_threshold={result['split_threshold']} "
        f"automatic_split={result['automatic_track_split_performed']} "
        f"global_id_rewrite={result['automatic_global_id_rewrite_performed']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
