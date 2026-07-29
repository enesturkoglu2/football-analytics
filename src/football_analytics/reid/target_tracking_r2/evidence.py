"""Per-observation purity/kit evidence for a raw track (reuses Stage5 kit/quality APIs)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from football_analytics.reid.kit import compute_torso_kit_metrics, load_kit_descriptor_config
from football_analytics.reid.quality import (
    compute_image_metrics,
    compute_tracking_bbox_contamination,
)
from football_analytics.reid.target_tracking_r2.policy import R2_POLICY


def load_kit_config(project_root: Path) -> dict[str, Any]:
    path = project_root / R2_POLICY["kit_config_path"]
    return load_kit_descriptor_config(path)


def _crop_sha(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return hashlib.sha256(buf.tobytes()).hexdigest()


def _clamp_bbox(bbox: Sequence[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def classify_kit_state(
    kit: Mapping[str, Any] | None,
    *,
    quality_ok: bool,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pol = dict(policy or R2_POLICY)
    if not quality_ok or kit is None:
        return {
            "kit_state": "UNKNOWN",
            "yellow_evidence": None,
            "white_evidence": None,
            "dominant_color_family": None,
            "reliable": False,
        }
    fr = kit.get("color_family_fractions") or {}
    yellow = float(fr.get("yellow") or 0.0)
    white = float(fr.get("white") or 0.0)
    orange = float(fr.get("orange") or 0.0)
    # Football yellow kits often bleed into orange bin; count as yellow-side
    yellow_side = yellow + 0.5 * orange
    dom = str(kit.get("dominant_color_family") or "UNKNOWN")
    dom_frac = float(kit.get("dominant_color_family_fraction") or 0.0)
    chrom = float(kit.get("chromatic_pixel_ratio") or 0.0)
    reliable = dom_frac >= float(pol["min_dominant_fraction_for_reliable_kit"])
    if dom in {"yellow", "orange"} and yellow_side >= float(pol["min_dominant_fraction_for_reliable_kit"]):
        state = "YELLOW"
    elif dom == "white" and white >= float(pol["min_dominant_fraction_for_reliable_kit"]):
        state = "WHITE"
    elif yellow_side >= float(pol["yellow_frac_low"]) * 2 and yellow_side > white:
        state = "YELLOW" if reliable else "UNKNOWN"
    elif white >= float(pol["white_frac_high"]) and white > yellow_side:
        state = "WHITE" if reliable else "UNKNOWN"
    else:
        state = "UNKNOWN"
    if chrom < float(pol["min_chromatic_ratio_for_chromatic_kit"]) and state == "YELLOW":
        # very achromatic → unknown chromatic kit
        if white >= yellow_side:
            state = "WHITE" if white >= float(pol["white_frac_high"]) else "UNKNOWN"
        else:
            state = "UNKNOWN"
    return {
        "kit_state": state,
        "yellow_evidence": yellow_side,
        "white_evidence": white,
        "dominant_color_family": dom,
        "dominant_fraction": dom_frac,
        "reliable": reliable and state != "UNKNOWN",
    }


def extract_track_evidence(
    *,
    video_path: Path,
    raw_track_id: str,
    observations_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_width: int,
    frame_height: int,
    fps: float,
    kit_config: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Decode crops for one immutable raw track; reuse Stage5 kit/quality functions."""
    pol = dict(policy or R2_POLICY)
    frames = sorted(
        int(fi)
        for fi, rows in observations_by_frame.items()
        if any(str(r.get("raw_track_id")) == str(raw_track_id) for r in rows)
    )
    step = max(1, int(pol.get("sample_every_frame") or 1))
    frames = frames[::step]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    rows_out: list[dict[str, Any]] = []
    try:
        for fi in frames:
            # seek each time — short clip, acceptable for R2 audit
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            target = None
            frame_obs = []
            for r in observations_by_frame.get(str(fi)) or []:
                bbox = _clamp_bbox(r["bbox_xyxy"], frame_width, frame_height)
                frame_obs.append(
                    {
                        "track_id": int(str(r.get("raw_track_id"))),
                        "bbox_xyxy": bbox,
                    }
                )
                if str(r.get("raw_track_id")) == str(raw_track_id):
                    target = {
                        "bbox_xyxy": bbox,
                        "detection_id": r.get("detection_id"),
                        "segment_id": r.get("segment_id"),
                    }
            if target is None:
                continue
            x1, y1, x2, y2 = target["bbox_xyxy"]
            crop = frame[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)
            min_side = min(x2 - x1, y2 - y1)
            quality_ok = (
                area >= int(pol["min_crop_area_px"])
                and min_side >= int(pol["min_crop_side_px"])
                and crop.size > 0
            )
            img_metrics = None
            kit_metrics = None
            lap = 0.0
            if crop.size > 0:
                try:
                    img_metrics = compute_image_metrics(crop)
                    lap = float(img_metrics["laplacian_variance"])
                    if lap < float(pol["min_laplacian_variance"]):
                        quality_ok = False
                except Exception:  # noqa: BLE001
                    quality_ok = False
                if quality_ok:
                    try:
                        kit_metrics = compute_torso_kit_metrics(crop, config=kit_config)
                    except Exception:  # noqa: BLE001
                        kit_metrics = None
                        quality_ok = False
            try:
                cont = compute_tracking_bbox_contamination(
                    target_bbox=target["bbox_xyxy"],
                    target_track_id=int(str(raw_track_id)),
                    frame_observations=frame_obs,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            except Exception:  # noqa: BLE001
                cont = {
                    "other_person_overlap_count": 0,
                    "union_other_person_crop_coverage": 0.0,
                    "max_other_person_iou": 0.0,
                }
            kit_cls = classify_kit_state(kit_metrics, quality_ok=quality_ok, policy=pol)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            rows_out.append(
                {
                    "frame_index": fi,
                    "timestamp_sec": fi / fps if fps else 0.0,
                    "raw_track_id": str(raw_track_id),
                    "bbox_xyxy": target["bbox_xyxy"],
                    "bbox_area": area,
                    "crop_sha256": _crop_sha(crop) if crop.size else None,
                    "crop_quality_ok": quality_ok,
                    "laplacian_variance": lap,
                    "image_metrics": img_metrics,
                    "kit_metrics_summary": (
                        {
                            "dominant_color_family": kit_metrics.get("dominant_color_family"),
                            "dominant_color_family_fraction": kit_metrics.get(
                                "dominant_color_family_fraction"
                            ),
                            "color_family_fractions": kit_metrics.get("color_family_fractions"),
                            "chromatic_pixel_ratio": kit_metrics.get("chromatic_pixel_ratio"),
                        }
                        if kit_metrics
                        else None
                    ),
                    **kit_cls,
                    "contamination": cont,
                    "nearby_player_overlap": int(cont.get("other_person_overlap_count") or 0),
                    "center_xy": [cx, cy],
                    "detection_id": target.get("detection_id"),
                }
            )
    finally:
        cap.release()
    # motion discontinuity flags
    for i in range(1, len(rows_out)):
        a, b = rows_out[i - 1], rows_out[i]
        dt = max(1, int(b["frame_index"]) - int(a["frame_index"]))
        dx = b["center_xy"][0] - a["center_xy"][0]
        dy = b["center_xy"][1] - a["center_xy"][1]
        b["motion_speed_px_per_frame"] = (dx * dx + dy * dy) ** 0.5 / dt
        scale = b["bbox_area"] / max(1, a["bbox_area"])
        b["bbox_area_ratio_vs_prev"] = scale
    if rows_out:
        rows_out[0]["motion_speed_px_per_frame"] = 0.0
        rows_out[0]["bbox_area_ratio_vs_prev"] = 1.0
    return rows_out
