#!/usr/bin/env python3
"""CLI entry point for Stage 2 CPU person detection.

Src-layout bootstrap
--------------------
This project is not installed as an editable package yet (no ``pip install -e .``).
The block below adds only ``<project_root>/src`` to ``sys.path`` so
``football_analytics`` imports resolve when the script is run directly.
Editable install is intentionally deferred to a later stage.

Notes
-----
- Annotated output video is silent; source audio is not copied.
- Model weights are never downloaded by this script. Provide a local ``.pt`` file.
- Prefer ``--output-dir outputs/detection/benchmark_100`` or
  ``outputs/detection/full`` to keep runs separated.
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

from football_analytics.detection import DetectionError, run_detection  # noqa: E402
from football_analytics.detection.pipeline import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect COCO person (class 0) on CPU with Ultralytics YOLO. "
            "No tracking. Annotated video is silent."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory (use outputs/detection/benchmark_100 or "
            "outputs/detection/full)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Local YOLO weights path (default: {DEFAULT_MODEL_PATH}). No auto-download.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
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
        help="Allow replacing existing annotated.mp4 / detections.jsonl / detection_summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_detection(
            video=args.video,
            output_dir=args.output_dir,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except DetectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {result['annotated_path']}")
    print(f"wrote {result['detections_path']}")
    print(f"wrote {result['summary_path']}")
    metrics = result["summary"]["metrics"]
    print(
        "status=ok "
        f"frames={metrics['frames_processed']} "
        f"detections={metrics['total_detections']} "
        f"avg_fps={metrics['avg_fps']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
