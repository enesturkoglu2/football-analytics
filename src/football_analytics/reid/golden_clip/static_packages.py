"""Static failure-window asset packages for Target GT pilot (R1.2).

No custom interactive video component. Assets are small MP4/PNG files on disk;
UI loads one window at a time via file paths.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from football_analytics.reid.golden_clip.pilot import generate_failure_windows
from football_analytics.reid.golden_clip.window_index import track_span_lightweight
from football_analytics.reid.hil.common import sha256_file

ENROLLMENT_DISPLAY_W = 960
CLIP_RADIUS_FRAMES = 75  # ~5s @ 30fps (within 3–6s)
PACKAGE_SCHEMA = "target_failure_window_static_package_v1"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_clip_bounds(
    center: int, *, frame_count: int, radius: int = CLIP_RADIUS_FRAMES
) -> tuple[int, int]:
    start = max(0, int(center) - int(radius))
    end = min(int(frame_count) - 1, int(center) + int(radius))
    if end < start:
        end = start
    return start, end


def _decode_frame(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to decode frame {frame_index}")
        return frame
    finally:
        cap.release()


def _draw_candidates(
    frame,
    rows: Sequence[Mapping[str, Any]],
    *,
    highlight_ids: Sequence[str] | None = None,
) -> Any:
    out = frame.copy()
    hl = {str(x) for x in (highlight_ids or [])}
    for r in rows:
        tid = str(r.get("raw_track_id"))
        x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
        color = (0, 80, 255) if tid in hl else (0, 220, 220)
        thick = 3 if tid in hl else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(
            out,
            f"{tid}|{r.get('segment_id') or ''}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _write_png(path: Path, bgr) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"png write failed: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _extract_native_clip(
    *,
    source_video: Path,
    source_video_sha256: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_video) != source_video_sha256.lower():
        raise RuntimeError("source video SHA mismatch for clip extraction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and output_path.stat().st_size > 0:
        return {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "cache_hit": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
    start_s = float(start_frame) / float(fps) if fps else 0.0
    duration_s = float(end_frame - start_frame + 1) / float(fps) if fps else 0.0
    # Re-encode for reliable browser playback (avoid copy keyframe issues)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_s:.6f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"ffmpeg clip failed: {proc.stderr[-500:]}")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "cache_hit": False,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def pick_enrollment_frame(
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    frame_count: int,
    prefer_from: int = 30,
    prefer_to: int = 150,
) -> int:
    """Pick a dense early frame for static target enrollment (server-side only)."""
    best_fi = max(0, min(prefer_from, frame_count - 1))
    best_n = -1
    lo = max(0, prefer_from)
    hi = min(frame_count - 1, prefer_to)
    for fi in range(lo, hi + 1):
        n = len(observations_by_frame.get(str(fi)) or [])
        if n > best_n:
            best_n = n
            best_fi = fi
    return best_fi


def build_enrollment_package(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_index: int,
    output_dir: Path,
    display_w: int = ENROLLMENT_DISPLAY_W,
) -> dict[str, Any]:
    """Write one bbox overlay PNG + candidate metadata for static click enrollment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(observations_by_frame.get(str(int(frame_index))) or [])
    frame = _decode_frame(source_video, int(frame_index))
    h, w = frame.shape[:2]
    drawn = _draw_candidates(frame, rows)
    png_path = output_dir / f"enrollment_frame_{int(frame_index):06d}.png"
    png_meta = _write_png(png_path, drawn)
    # Display size preserving aspect (no letterbox pad)
    display_h = max(1, int(round(h * (display_w / float(w)))))
    candidates = [
        {
            "bbox_id": f"enr_{i}_{r.get('raw_track_id')}",
            "raw_track_id": str(r.get("raw_track_id")),
            "segment_id": r.get("segment_id"),
            "detection_id": r.get("detection_id"),
            "bbox_xyxy": list(r["bbox_xyxy"]),
            "candidate_id": r.get("candidate_id"),
        }
        for i, r in enumerate(rows)
    ]
    man = {
        "schema_version": "target_failure_window_enrollment_v1",
        "created_at": _utc(),
        "source_video_sha256": source_video_sha256,
        "frame_index": int(frame_index),
        "frame_w": int(w),
        "frame_h": int(h),
        "display_w": int(display_w),
        "display_h": int(display_h),
        "png": png_meta,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "custom_interactive_component": False,
    }
    _write_json(output_dir / "enrollment_manifest.json", man)
    return man


def build_static_failure_packages(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_raw_track_id: str,
    selected_segment_id: str | None,
    fps: float,
    frame_count: int,
    output_dir: Path,
    max_windows: int = 5,
) -> dict[str, Any]:
    """Materialize ≥5 candidate static packages for one selected seed track."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    span = track_span_lightweight(
        observations_by_frame,
        raw_track_id=str(selected_raw_track_id),
        segment_id=selected_segment_id,
        sample_every=30,
    )
    if span["first_frame"] is None or span["last_frame"] is None:
        raise RuntimeError("selected track has no observations")
    windows = generate_failure_windows(
        observations_by_frame=observations_by_frame,
        selected_raw_track_id=str(selected_raw_track_id),
        selected_segment_id=selected_segment_id,
        track_first=int(span["first_frame"]),
        track_last=int(span["last_frame"]),
        fps=fps,
        frame_count=frame_count,
        max_windows=max_windows,
        radius=CLIP_RADIUS_FRAMES,
    )
    # Ensure border exit/re-entry candidate if track ends near image edge
    last_rows = list(observations_by_frame.get(str(span["last_frame"])) or [])
    last_bbox = next(
        (
            list(r["bbox_xyxy"])
            for r in last_rows
            if str(r.get("raw_track_id")) == str(selected_raw_track_id)
        ),
        None,
    )
    packages: list[dict[str, Any]] = []
    for w in windows:
        pkg = _materialize_one_package(
            source_video=source_video,
            source_video_sha256=source_video_sha256,
            observations_by_frame=observations_by_frame,
            window=w,
            selected_raw_track_id=str(selected_raw_track_id),
            fps=fps,
            frame_count=frame_count,
            output_dir=output_dir,
            border_hint=last_bbox,
        )
        packages.append(pkg)

    index = {
        "schema_version": PACKAGE_SCHEMA,
        "created_at": _utc(),
        "coverage_scope": "FAILURE_WINDOW_PILOT",
        "source_video_sha256": source_video_sha256,
        "selected_raw_track_id": str(selected_raw_track_id),
        "selected_segment_id": selected_segment_id,
        "track_first_frame": span["first_frame"],
        "track_last_frame": span["last_frame"],
        "package_count": len(packages),
        "packages": packages,
        "custom_interactive_component": False,
        "full_dense_manifest_not_sent_to_browser": True,
    }
    _write_json(output_dir / "packages_index.json", index)
    return index


def _materialize_one_package(
    *,
    source_video: Path,
    source_video_sha256: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    window: Mapping[str, Any],
    selected_raw_track_id: str,
    fps: float,
    frame_count: int,
    output_dir: Path,
    border_hint: list[float] | None,
) -> dict[str, Any]:
    event_id = str(window.get("window_id") or f"fw_{uuid.uuid4().hex[:10]}")
    center = int(window["center_frame"])
    start_f, end_f = _safe_clip_bounds(center, frame_count=frame_count)
    # Prefer window-declared bounds when they fall inside the safe clip
    w_start = int(window.get("start_frame") or start_f)
    w_end = int(window.get("end_frame") or end_f)
    start_f = max(start_f, min(w_start, end_f))
    end_f = min(end_f, max(w_end, start_f))
    if end_f < start_f:
        start_f, end_f = _safe_clip_bounds(center, frame_count=frame_count)

    pkg_dir = output_dir / event_id
    pkg_dir.mkdir(parents=True, exist_ok=True)

    pre_fi = max(0, center - 15)
    fail_fi = center
    post_fi = min(frame_count - 1, center + 15)

    clip = _extract_native_clip(
        source_video=source_video,
        source_video_sha256=source_video_sha256,
        start_frame=start_f,
        end_frame=end_f,
        fps=fps,
        output_path=pkg_dir / "review_clip.mp4",
    )

    def _frame_asset(fi: int, name: str, highlight: Sequence[str] | None = None):
        rows = list(observations_by_frame.get(str(fi)) or [])
        bgr = _draw_candidates(_decode_frame(source_video, fi), rows, highlight_ids=highlight)
        return {
            "frame_index": fi,
            "timestamp": fi / fps if fps else 0.0,
            "candidates": [
                {
                    "raw_track_id": str(r.get("raw_track_id")),
                    "segment_id": r.get("segment_id"),
                    "detection_id": r.get("detection_id"),
                    "bbox_xyxy": list(r["bbox_xyxy"]),
                    "bbox_id": f"{name}_{i}_{r.get('raw_track_id')}",
                }
                for i, r in enumerate(rows)
            ],
            "png": _write_png(pkg_dir / f"{name}.png", bgr),
        }

    pre = _frame_asset(pre_fi, "pre_event", highlight=[selected_raw_track_id])
    fail = _frame_asset(fail_fi, "failure_frame", highlight=[selected_raw_track_id])
    post = _frame_asset(
        post_fi,
        "post_candidates",
        highlight=list(window.get("possible_next_raw_tracks") or [])[:6],
    )
    # All candidates on failure frame (explicit name for UI)
    all_cands = _frame_asset(
        fail_fi,
        "all_candidates",
        highlight=list(window.get("possible_next_raw_tracks") or [])[:8],
    )

    candidate_ids = []
    for block in (fail, post, all_cands):
        for c in block["candidates"]:
            tid = str(c["raw_track_id"])
            if tid != str(selected_raw_track_id) and tid not in candidate_ids:
                candidate_ids.append(tid)

    border_exit = False
    if border_hint is not None:
        x1, y1, x2, y2 = map(float, border_hint)
        # near left/right/top/bottom of 1326x750-ish frame
        frame = _decode_frame(source_video, int(window.get("center_frame") or fail_fi))
        fh, fw = frame.shape[:2]
        margin = 40
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        border_exit = cx < margin or cx > fw - margin or cy < margin or cy > fh - margin

    man = {
        "schema_version": PACKAGE_SCHEMA,
        "event_id": event_id,
        "kind": window.get("kind"),
        "previous_raw_track_id": str(selected_raw_track_id),
        "previous_segment_id": window.get("previous_segment_id"),
        "track_end_frame": int(window.get("center_frame") or fail_fi),
        "track_end_time": float(window.get("start_time") or (fail_fi / fps if fps else 0.0)),
        "clip": {
            **clip,
            "start_time": start_f / fps if fps else 0.0,
            "end_time": (end_f + 1) / fps if fps else 0.0,
            "duration_sec": (end_f - start_f + 1) / fps if fps else 0.0,
        },
        "pre_event_frame": pre,
        "failure_frame": fail,
        "post_candidate_frame": post,
        "all_candidates_frame": all_cands,
        "candidate_raw_track_ids": candidate_ids,
        "candidate_segment_ids": sorted(
            {
                str(c.get("segment_id"))
                for block in (fail, post, all_cands)
                for c in block["candidates"]
                if c.get("segment_id")
            }
        ),
        "temporal_gap_frames": window.get("time_gap_frames"),
        "spatial_displacement_px": window.get("image_position_displacement_px"),
        "detection_continuity_frames": window.get("detection_continuity_frames"),
        "bbox_overlap": window.get("bbox_overlap"),
        "border_exit_reentry_hint": border_exit,
        "tracker_variant": window.get("tracker_variant") or "A_current_bytetrack",
        "source_video_sha256": source_video_sha256,
        "auto_identity_forbidden": True,
        "possible_next_raw_tracks": list(window.get("possible_next_raw_tracks") or []),
    }
    _write_json(pkg_dir / "manifest.json", man)
    # Keep only UI-needed paths relative-friendly
    return {
        "event_id": event_id,
        "kind": man["kind"],
        "manifest_path": str(pkg_dir / "manifest.json"),
        "clip_path": clip["path"],
        "all_candidates_png": all_cands["png"]["path"],
        "failure_png": fail["png"]["path"],
        "pre_png": pre["png"]["path"],
        "post_png": post["png"]["path"],
        "track_end_frame": man["track_end_frame"],
        "track_end_time": man["track_end_time"],
        "candidate_raw_track_ids": candidate_ids,
        "temporal_gap_frames": man["temporal_gap_frames"],
        "detection_continuity_frames": man["detection_continuity_frames"],
        "spatial_displacement_px": man["spatial_displacement_px"],
        "bbox_overlap": man["bbox_overlap"],
        "border_exit_reentry_hint": border_exit,
        "clip_bytes": clip["bytes"],
    }


def load_package_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_enrollment_click(
    *,
    ui_x: float,
    ui_y: float,
    enrollment: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Map static enrollment click to a candidate using uniform scale (no pad)."""
    from football_analytics.reid.hil_ui.geometry import letterbox_params, resolve_click_to_bbox

    params = letterbox_params(
        frame_w=int(enrollment["frame_w"]),
        frame_h=int(enrollment["frame_h"]),
        display_w=int(enrollment["display_w"]),
        display_h=int(enrollment["display_h"]),
    )
    result = resolve_click_to_bbox(
        ui_x=float(ui_x),
        ui_y=float(ui_y),
        params=params,
        bboxes=list(enrollment.get("candidates") or []),
    )
    if result.status != "hit" or result.selected is None:
        return None
    # recover full candidate row
    for c in enrollment.get("candidates") or []:
        if str(c.get("bbox_id")) == str(result.selected.bbox_id):
            return dict(c)
        xy = c.get("bbox_xyxy")
        if xy and tuple(map(float, xy)) == tuple(result.selected.bbox_xyxy):
            return dict(c)
    return {
        "bbox_id": result.selected.bbox_id,
        "bbox_xyxy": list(result.selected.bbox_xyxy),
        "raw_track_id": None,
    }
