#!/usr/bin/env python3
"""CLI entry point for Stage 3 CPU person tracking (ByteTrack).

Src-layout bootstrap
--------------------
This project is not installed as an editable package yet (no ``pip install -e .``).
The block below adds only ``<project_root>/src`` to ``sys.path`` so
``football_analytics`` imports resolve when the script is run directly.

Notes
-----
- Each CLI invocation creates a fresh model/tracker state for one continuous video.
- Tracked output video is silent; source audio is not copied.
- Prefer ``--output-dir outputs/tracking/benchmark_100`` or ``outputs/tracking/full``.
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

from football_analytics.tracking import TrackingError, run_tracking  # noqa: E402
from football_analytics.tracking.pipeline import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRACKER_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track COCO person (class 0) on CPU with Ultralytics ByteTrack. "
            "One continuous video stream per run. Tracked video is silent."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory (use outputs/tracking/benchmark_100 or "
            "outputs/tracking/full)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Local YOLO weights path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER_PATH,
        help=f"Local ByteTrack YAML (default: {DEFAULT_TRACKER_PATH}).",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="Detector NMS IoU threshold (not ByteTrack match_thresh).",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on frames processed (e.g. 100 for benchmark).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing tracked.mp4 / tracks.jsonl / tracking_summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_tracking(
            video=args.video,
            output_dir=args.output_dir,
            model_path=args.model,
            tracker_path=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except TrackingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['tracked_path']}")
    print(f"wrote {result['tracks_path']}")
    print(f"wrote {result['summary_path']}")
    metrics = result["summary"]["metrics"]
    print(
        "status=ok "
        f"frames={metrics['frames_processed']} "
        f"box_obs={metrics['total_box_observations']} "
        f"unique_ids={metrics['unique_track_ids']} "
        f"avg_fps={metrics['avg_fps']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
