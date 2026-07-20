#!/usr/bin/env python3
"""CLI entry point for Stage 4B ReID crop selection and extraction.

Src-layout bootstrap
--------------------
This project is not installed as an editable package yet (no ``pip install -e .``).
The block below adds only ``<project_root>/src`` to ``sys.path`` so
``football_analytics`` imports resolve when the script is run directly.

JPEG crops are written with OpenCV quality ``95`` (see
``football_analytics.reid.crop_extract.JPEG_QUALITY``).
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

from football_analytics.reid.crop_extract import (  # noqa: E402
    CropExtractError,
    run_select_reid_crops,
)
from football_analytics.reid.crop_select import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    CropSelectError,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    default_config = _PROJECT_ROOT / DEFAULT_CROP_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Select and extract ReID player crops from tracks.jsonl using "
            "configs/reid/crop_selection_stage4b.yaml. "
            "Does not run embedding or linking."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to tracks.jsonl from Stage 3 tracking.",
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Crop selection YAML (default: {default_config}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for crop_manifest.jsonl and crops/.",
    )
    parser.add_argument(
        "--track-id",
        action="append",
        type=int,
        default=None,
        help="Optional track_id filter (repeatable). Default: all tracks.",
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
        result = run_select_reid_crops(
            video=args.video,
            tracks=args.tracks,
            config=args.config,
            output_dir=args.output_dir,
            track_ids=args.track_id,
            overwrite=args.overwrite,
        )
    except (CropSelectError, CropExtractError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reasons = result["filter_reasons"]
    reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"
    print(f"wrote {result['manifest_path']}")
    print(f"wrote crops under {result['crops_dir']}")
    print(
        "status=ok "
        f"observations_read={result['observations_read']} "
        f"tracks_examined={result['tracks_examined']} "
        f"eligible_observations={result['eligible_observations']} "
        f"tracks_with_crops={result['tracks_with_crops']} "
        f"crops_written={result['crops_written']} "
        f"jpeg_quality={result['jpeg_quality']} "
        f"filter_reasons=[{reason_text}] "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
