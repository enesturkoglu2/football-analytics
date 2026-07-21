#!/usr/bin/env python3
"""CLI entry point for Stage 5B3G segmented ReID regression.

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

from football_analytics.reid.segment_regression import (  # noqa: E402
    SegmentRegressionError,
    run_segmented_reid_regression,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run segmented ReID regression against the Stage 4B raw-track "
            "baseline. Manual-split segments recompute embeddings; unaffected "
            "full-track segments reuse baseline embeddings. No global-ID "
            "rewrite, threshold, auto-link, or accuracy claim."
        )
    )
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument(
        "--segment-view-dir",
        required=True,
        help="Non-destructive segment-view output directory",
    )
    parser.add_argument(
        "--baseline-run-dir",
        required=True,
        help="Stage 4B baseline run directory (crops/aggregation/candidates)",
    )
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
        "--crop-config",
        required=True,
        help="Stage 4B crop selection YAML",
    )
    parser.add_argument(
        "--reid-config",
        required=True,
        help="Stage 4B ReID provenance/config YAML (hashed for audit)",
    )
    parser.add_argument(
        "--regression-config",
        required=True,
        help="segmented_reid_regression_stage5b3.yaml",
    )
    parser.add_argument(
        "--sn-reid-root",
        required=True,
        help="Local sn-reid repository root",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local OSNet checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help="Optional expected checkpoint SHA-256 (defaults to baseline summary)",
    )
    parser.add_argument(
        "--expected-sn-reid-commit",
        default=None,
        help="Optional expected sn-reid HEAD (defaults to baseline summary)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Segmented ReID regression output directory",
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
        result = run_segmented_reid_regression(
            video=args.video,
            segment_view_dir=args.segment_view_dir,
            baseline_run_dir=args.baseline_run_dir,
            segmentation_policy=args.segmentation_policy,
            segment_decisions=args.segment_decisions,
            crop_config=args.crop_config,
            reid_config=args.reid_config,
            regression_config=args.regression_config,
            sn_reid_root=args.sn_reid_root,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            checkpoint_sha256=args.checkpoint_sha256,
            expected_sn_reid_commit=args.expected_sn_reid_commit,
        )
    except (SegmentRegressionError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"derived_segment_count={result['derived_segment_count']} "
        f"recomputed_manual_segment_embedding_count="
        f"{result['recomputed_manual_segment_embedding_count']} "
        f"reused_baseline_segment_embedding_count="
        f"{result['reused_baseline_segment_embedding_count']} "
        f"retired_mixed_baseline_embedding_count="
        f"{result['retired_mixed_baseline_embedding_count']} "
        f"ranked_candidate_count={result['ranked_candidate_count']} "
        f"unaffected_pair_similarity_mismatch_count="
        f"{result['unaffected_pair_similarity_mismatch_count']} "
        f"raw_tracks_mutated={result['raw_tracks_mutated']} "
        f"global_id_rewrite={result['global_id_rewrite_performed']} "
        f"accuracy_claimed={result['accuracy_claimed']} "
        f"elapsed_sec={result['elapsed_sec']} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
