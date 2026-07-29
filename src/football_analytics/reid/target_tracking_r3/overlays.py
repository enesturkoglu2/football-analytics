"""R3 diagnostic overlays: R2 reference, bridge, switch-window slow review."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.hil.common import sha256_file


def _transcode_h264(raw_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")
    if raw_path.is_file():
        raw_path.unlink()


def _box(frame, bbox, color, label: str, thick: int = 2) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.putText(
        frame, label, (x1, max(16, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
    )


def _hud(frame, lines: Sequence[str]) -> None:
    y = 18
    for line in lines:
        cv2.putText(
            frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 16


def render_r2_reference_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_raw_track_id: str,
    r2_seed_start: int,
    r2_seed_end: int,
    r2_change_point_frame: int,
    fps: float,
    frame_count: int,
    output_path: Path,
) -> dict[str, Any]:
    """Replay R2 safe-unresolved behavior without mutating R2 artifacts."""
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")
    cap = cv2.VideoCapture(str(source_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for r in observations_by_frame.get(str(fi)) or []:
                tid = str(r.get("raw_track_id"))
                if tid != str(parent_raw_track_id):
                    continue
                if int(r2_seed_start) <= fi <= int(r2_seed_end):
                    _box(frame, r["bbox_xyxy"], (0, 220, 255), "R2 CLEAN SEED", 3)
                else:
                    _box(frame, r["bbox_xyxy"], (80, 80, 200), "R2 EXCLUDED / CONFLICT", 2)
            if fi == int(r2_change_point_frame):
                cv2.line(frame, (0, 40), (w, 40), (0, 0, 255), 2)
            status = "CLEAN SEED" if fi <= int(r2_seed_end) else "TARGET UNRESOLVED"
            _hud(frame, [f"t={fi / fps:.2f}s f={fi}", f"R2 REF: {status}", f"boundary f={r2_change_point_frame}"])
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    _transcode_h264(raw, output_path)
    return {"path": str(output_path), "sha256": sha256_file(output_path)}


def render_bridge_overlay(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    timeline: Mapping[str, Any],
    bridge_result: Mapping[str, Any],
    refined_seed_start: int,
    refined_seed_end: int,
    refined_change_point: int,
    fps: float,
    frame_count: int,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")
    by_f = {int(fr["frame_index"]): fr for fr in (bridge_result.get("frames") or [])}
    cap = cv2.VideoCapture(str(source_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # all detections faint
            for r in observations_by_frame.get(str(fi)) or []:
                _box(frame, r["bbox_xyxy"], (90, 90, 90), str(r.get("raw_track_id")), 1)
            if int(refined_seed_start) <= fi <= int(refined_seed_end):
                for r in observations_by_frame.get(str(fi)) or []:
                    # seed parent only if in refined clean
                    pass
                # paint seed from timeline parent on refined range using obs of parent in interval
            fr = by_f.get(fi)
            label = "TARGET UNRESOLVED"
            if int(refined_seed_start) <= fi <= int(refined_seed_end):
                # find parent bbox
                for r in observations_by_frame.get(str(fi)) or []:
                    # painted via seed interval — use any matching seed parent from timeline
                    pass
                for iv in timeline.get("intervals") or []:
                    if iv.get("kind") == "HUMAN_SEED_SEGMENT":
                        pid = str(iv.get("parent_raw_track_id"))
                        for r in observations_by_frame.get(str(fi)) or []:
                            if str(r.get("raw_track_id")) == pid:
                                _box(
                                    frame,
                                    r["bbox_xyxy"],
                                    (0, 220, 255),
                                    f"HUMAN SEED SEGMENT · raw{pid}/{iv.get('segment_id')}",
                                    3,
                                )
                        label = "HUMAN SEED SEGMENT"
            if fr:
                st = fr.get("status")
                if fr.get("projected_bbox") is not None:
                    _box(frame, fr["projected_bbox"], (255, 180, 0), "FLOW PREDICTION", 1)
                if st == "DETECTOR_SNAP" and fr.get("bbox_xyxy"):
                    sel = fr.get("supporting_detection") or {}
                    _box(
                        frame,
                        fr["bbox_xyxy"],
                        (0, 255, 0),
                        f"DETECTOR SNAP · cand={sel.get('raw_track_id')}",
                        3,
                    )
                    label = "DETECTOR SNAP"
                elif st == "BRIDGE_ACCEPTED" and fr.get("bbox_xyxy"):
                    sel = fr.get("supporting_detection") or {}
                    _box(
                        frame,
                        fr["bbox_xyxy"],
                        (0, 255, 80),
                        f"BRIDGE ACCEPTED · cand={sel.get('raw_track_id')}",
                        3,
                    )
                    label = "BRIDGE ACCEPTED"
                elif st in {"TARGET_UNRESOLVED", "BRIDGE_FLOW_UNRELIABLE"}:
                    label = f"TARGET UNRESOLVED ({st})"
                    # show rejected cross-team candidates in purple
                    for c in fr.get("candidates") or []:
                        if "CROSS_TEAM_KIT_MISMATCH" in (c.get("hard_rejects") or []):
                            _box(
                                frame,
                                c["bbox_xyxy"],
                                (180, 0, 180),
                                f"BRIDGE REJECTED CROSS-TEAM · {c.get('raw_track_id')}",
                                2,
                            )
                # never paint white excluded parent continuation as target red
            if fi == int(refined_change_point):
                cv2.line(frame, (0, 36), (w, 36), (0, 0, 255), 2)
            _hud(
                frame,
                [
                    f"t={fi / fps:.2f}s f={fi}",
                    f"R3: {label}",
                    f"refined_boundary f={refined_change_point}",
                ],
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    _transcode_h264(raw, output_path)
    return {"path": str(output_path), "sha256": sha256_file(output_path)}


def render_switch_window_slow_review(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    bridge_result: Mapping[str, Any],
    refined_seed_start: int,
    refined_seed_end: int,
    refined_change_point: int,
    fps: float,
    output_path: Path,
    slowdown: float = 0.25,
) -> dict[str, Any]:
    """Slow-motion crop of the local switch/bridge event only."""
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source SHA mismatch")
    frames = bridge_result.get("frames") or []
    if frames:
        f0 = min(int(refined_seed_end) - 8, int(frames[0]["frame_index"]) - 5)
        f1 = max(int(frames[-1]["frame_index"]) + 8, int(refined_change_point) + 10)
    else:
        f0 = int(refined_seed_end) - 10
        f1 = int(refined_change_point) + 20
    f0 = max(0, f0)
    by_f = {int(fr["frame_index"]): fr for fr in frames}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = output_path.with_suffix(".raw.mp4")
    out_fps = max(1.0, float(fps) * float(slowdown))
    cap = cv2.VideoCapture(str(source_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    try:
        for fi in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for r in observations_by_frame.get(str(fi)) or []:
                _box(frame, r["bbox_xyxy"], (100, 100, 100), str(r.get("raw_track_id")), 1)
            if int(refined_seed_start) <= fi <= int(refined_seed_end):
                cv2.putText(
                    frame, "SEED TARGET", (8, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA,
                )
            fr = by_f.get(fi)
            if fr:
                if fr.get("projected_bbox") is not None:
                    _box(frame, fr["projected_bbox"], (255, 180, 0), "projected", 2)
                md = fr.get("median_delta")
                if md and fr.get("projected_bbox"):
                    pc = (
                        int(0.5 * (fr["projected_bbox"][0] + fr["projected_bbox"][2])),
                        int(0.5 * (fr["projected_bbox"][1] + fr["projected_bbox"][3])),
                    )
                    pe = (int(pc[0] + md[0] * 3), int(pc[1] + md[1] * 3))
                    cv2.arrowedLine(frame, pc, pe, (0, 255, 255), 2, tipLength=0.3)
                for c in fr.get("candidates") or []:
                    col = (0, 200, 0) if c.get("eligible") else (0, 0, 200)
                    tag = f"{c.get('raw_track_id')} s={c.get('score', 0):.2f}"
                    if c.get("hard_rejects"):
                        tag += " REJECT"
                        col = (180, 0, 180) if "CROSS_TEAM" in str(c.get("hard_rejects")) else col
                    _box(frame, c["bbox_xyxy"], col, tag, 2)
                if fr.get("bbox_xyxy") and fr.get("status") in {"DETECTOR_SNAP", "BRIDGE_ACCEPTED"}:
                    _box(frame, fr["bbox_xyxy"], (0, 255, 0), fr["status"], 3)
            if fi == int(refined_change_point):
                cv2.putText(
                    frame, "PURITY BOUNDARY", (w // 2 - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )
            _hud(frame, [f"SLOW REVIEW t={fi / fps:.2f}s f={fi}", f"window [{f0},{f1}]"])
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    _transcode_h264(raw, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "window_frames": [f0, f1],
        "slowdown": slowdown,
    }


def write_review_contact_sheet(
    *,
    source_video: Path,
    frame_indices: Sequence[int],
    output_path: Path,
    cols: int = 4,
) -> Path:
    cap = cv2.VideoCapture(str(source_video))
    thumbs: list[np.ndarray] = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        thumbs.append(cv2.resize(frame, (320, 180)))
    cap.release()
    if not thumbs:
        raise RuntimeError("no contact sheet frames")
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * 180 : (r + 1) * 180, c * 320 : (c + 1) * 320] = th
        cv2.putText(
            canvas, f"f={frame_indices[i]}", (c * 320 + 6, r * 180 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
