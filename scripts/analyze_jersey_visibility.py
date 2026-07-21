#!/usr/bin/env python3
"""CLI for Stage 5C-A selected-crop visibility/readability measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if not _SRC_DIR.is_dir():
    print(f"error: expected src directory at {_SRC_DIR}", file=sys.stderr)
    sys.exit(2)
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from football_analytics.reid.jersey_visibility import (  # noqa: E402
    JerseyVisibilityError,
    run_analyze_jersey_visibility,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure independent image and tracking-bbox context signals on "
            "existing segment-level selected crops. No recognition, automatic "
            "visibility/readability decision, crop extraction, or video input."
        )
    )
    parser.add_argument(
        "--segment-view-dir",
        required=True,
        help="Stage 5B3 non-destructive segment-view directory.",
    )
    parser.add_argument(
        "--segmented-regression-dir",
        required=True,
        help="Stage 5B3 segmented regression artifact directory.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        required=True,
        help="Stage 4B run directory containing selected-crop provenance.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Stage 5C jersey visibility measurement YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the three JSON/JSONL artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory atomically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analyze_jersey_visibility(
            segment_view_dir=args.segment_view_dir,
            segmented_regression_dir=args.segmented_regression_dir,
            baseline_run_dir=args.baseline_run_dir,
            config=args.config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except JerseyVisibilityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['crop_signals_path']}")
    print(f"wrote {result['segment_summary_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"total_derived_segment_count={result['total_derived_segment_count']} "
        f"measured_segment_count={result['measured_segment_count']} "
        f"no_selected_crop_segment_count="
        f"{result['no_selected_crop_segment_count']} "
        f"total_selected_crop_count={result['total_selected_crop_count']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
