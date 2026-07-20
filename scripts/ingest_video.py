#!/usr/bin/env python3
"""CLI entry point for Stage 1 video ingest and metadata extraction.

Src-layout bootstrap
--------------------
This project is not installed as an editable package yet (no ``pip install -e .``).
The block below adds only ``<project_root>/src`` to ``sys.path`` so
``football_analytics`` imports resolve when the script is run directly.
Editable install is intentionally deferred to a later stage.
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

from football_analytics.ingest import IngestError, run_ingest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a football video and write ingest metadata JSON files."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ingest",
        help="Directory for video_manifest.json and ffprobe.json (default: outputs/ingest).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing video_manifest.json / ffprobe.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_ingest(
            video=args.video,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['ffprobe_path']}")
    print(f"wrote {result['manifest_path']}")
    print(f"status={result['manifest']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
