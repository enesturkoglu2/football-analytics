"""Ground-truth overlay video for human acceptance review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from football_analytics.reid.golden_clip.schema import active_intervals
from football_analytics.reid.hil.common import sha256_file

_LABELS = {
    "TARGET_VISIBLE_ASSOCIATED": ("GT TARGET", (0, 200, 0)),
    "TARGET_VISIBLE_BUT_MISSED": ("GT TARGET — TRACKER MISSED", (0, 165, 255)),
    "WRONG_TARGET_ASSIGNED": ("GT WRONG ASSIGNMENT", (0, 0, 220)),
    "TARGET_OCCLUDED": ("GT OCCLUDED", (180, 180, 0)),
    "TARGET_OUT_OF_FRAME": ("GT OUT OF FRAME", (160, 160, 160)),
    "TARGET_UNCERTAIN": ("GT UNCERTAIN", (200, 140, 0)),
}


def _bbox_for_frame(obs: Sequence[Mapping[str, Any]], frame_index: int) -> list[float] | None:
    for row in obs:
        if int(row.get("frame_index", -1)) == int(frame_index):
            return [float(v) for v in row["bbox_xyxy"]]
    return None


def render_gt_overlay_video(
    *,
    video_path: Path,
    ground_truth: Mapping[str, Any],
    output_path: Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Burn GT state labels into diagnostic MP4 for acceptance review."""
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or ground_truth.get("fps") or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, int(max_frames))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open VideoWriter: {output_path}")

    intervals = active_intervals(ground_truth)
    written = 0
    try:
        for fi in range(total):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            hit = None
            for iv in intervals:
                if int(iv["start_frame"]) <= fi <= int(iv["end_frame"]):
                    hit = iv
                    break
            if hit is not None:
                state = str(hit["target_state"])
                label, color = _LABELS.get(state, (f"GT {state}", (255, 255, 255)))
                cv2.putText(
                    frame, label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )
                # associated / wrong assignment bbox
                bbox = _bbox_for_frame(hit.get("bbox_observations") or [], fi)
                if bbox is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    tracks = ",".join(str(t) for t in (hit.get("associated_raw_track_ids") or [])[:3])
                    cv2.putText(
                        frame,
                        f"assoc:{tracks}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                # true target bbox for WRONG_TARGET_ASSIGNED
                if state == "WRONG_TARGET_ASSIGNED":
                    true_b = _bbox_for_frame(
                        hit.get("true_target_bbox_observations") or [], fi
                    )
                    if true_b is not None:
                        x1, y1, x2, y2 = [int(round(v)) for v in true_b]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(
                            frame,
                            "TRUE TARGET",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 200, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        frame,
                        "WRONG TRACKER ASSIGNMENT",
                        (20, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 220),
                        2,
                        cv2.LINE_AA,
                    )
            tsec = fi / fps if fps else 0.0
            cv2.putText(
                frame,
                f"f={fi} t={tsec:.2f}s",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()

    return {
        "overlay_path": str(output_path),
        "overlay_sha256": sha256_file(output_path),
        "frames_written": written,
        "fps": fps,
        "note": "GT review overlay only; not model input.",
    }
