#!/usr/bin/env python3
"""CLI for Stage 5C-A2b3 jersey manual annotation validation."""

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

from football_analytics.reid.jersey_annotation import (  # noqa: E402
    JerseyAnnotationError,
    run_validate_jersey_review_annotations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a jersey visual-review annotation CSV against immutable "
            "Stage 5C review artifacts. No OCR, recognition, gallery update, "
            "identity/team assignment, or source modification."
        )
    )
    parser.add_argument(
        "--review-dir",
        required=True,
        help="Stage 5C-A2b2 jersey review output directory.",
    )
    parser.add_argument(
        "--annotations-csv",
        required=True,
        help="Filled or blank jersey review annotation CSV to validate.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Stage 5C jersey manual review YAML config.",
    )
    parser.add_argument(
        "--report-path",
        required=True,
        help="Output path for the JSON validation report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing validation report atomically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_validate_jersey_review_annotations(
            review_dir=args.review_dir,
            annotations_csv=args.annotations_csv,
            config=args.config,
            report_path=args.report_path,
            overwrite=args.overwrite,
        )
    except JerseyAnnotationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['report_path']}")
    print(
        f"status={result['status']} "
        f"reviewed_row_count={result['reviewed_row_count']} "
        f"unreviewed_row_count={result['unreviewed_row_count']} "
        f"error_count={result['error_count']} "
        f"elapsed_sec={result['elapsed_sec']:.3f}"
    )
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
