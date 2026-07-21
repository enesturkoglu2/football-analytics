#!/usr/bin/env python3
"""CLI for Stage 5C-A2b display-only jersey review panel generation."""

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

from football_analytics.reid.jersey_review import (  # noqa: E402
    JerseyReviewError,
    run_build_jersey_review_panels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic display-only jersey review panels and a "
            "blank manual annotation template from existing Stage 5C "
            "measurement artifacts. No OCR, recognition, video input, crop "
            "extraction, enhancement, or automatic classification."
        )
    )
    parser.add_argument(
        "--measurement-dir",
        required=True,
        help="Stage 5C-A2a jersey visibility measurement output directory.",
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root that must contain every source crop path.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Stage 5C jersey review panel YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for review panels and annotation template.",
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
        result = run_build_jersey_review_panels(
            measurement_dir=args.measurement_dir,
            project_root=args.project_root,
            config=args.config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except JerseyReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['review_items_path']}")
    print(f"wrote {result['group_memberships_path']}")
    print(f"wrote {result['panel_index_path']}")
    print(f"wrote {result['annotation_template_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"canonical_review_item_count={result['canonical_review_item_count']} "
        f"group_count={result['group_count']} "
        f"total_panel_page_count={result['total_panel_page_count']} "
        f"elapsed_sec={result['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
