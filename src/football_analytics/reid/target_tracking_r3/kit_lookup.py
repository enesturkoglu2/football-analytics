"""Per-frame kit lookup helpers for R3 bridge snapping (reuses Stage5 kit APIs)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.kit import compute_torso_kit_metrics
from football_analytics.reid.quality import compute_image_metrics
from football_analytics.reid.target_tracking_r2.evidence import classify_kit_state
from football_analytics.reid.target_tracking_r3.policy import R3_POLICY


def _clamp_bbox(bbox: Sequence[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def kit_for_crop(
    bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    kit_config: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pol = dict(policy or R3_POLICY)
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox_xyxy, w, h)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"kit_state": "UNKNOWN", "reliable": False}
    area = (x2 - x1) * (y2 - y1)
    im = compute_image_metrics(crop)
    quality_ok = (
        area >= float(pol["min_crop_area_px"])
        and min(x2 - x1, y2 - y1) >= 18
        and float(im.get("laplacian_variance") or 0) >= float(pol["min_laplacian_variance"])
    )
    kit = compute_torso_kit_metrics(crop, config=kit_config) if quality_ok else None
    classified = classify_kit_state(kit, quality_ok=quality_ok, policy={
        **pol,
        "min_dominant_fraction_for_reliable_kit": 0.18,
        "min_chromatic_ratio_for_chromatic_kit": 0.08,
        "yellow_frac_low": 0.06,
        "white_frac_high": 0.22,
    })
    return {
        "kit_state": classified["kit_state"],
        "reliable": bool(classified.get("reliable")),
        "yellow_evidence": classified.get("yellow_evidence"),
        "white_evidence": classified.get("white_evidence"),
        "quality_ok": quality_ok,
    }


def build_kit_lookup_from_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keys: '{frame}:{raw_track_id}' and raw_track_id (last)."""
    out: dict[str, dict[str, Any]] = {}
    for r in evidence_rows:
        tid = str(r.get("raw_track_id") or "10")
        fi = int(r["frame_index"])
        entry = {
            "kit_state": r.get("kit_state"),
            "reliable": bool(r.get("reliable") or r.get("crop_quality_ok")),
            "yellow_evidence": r.get("yellow_evidence"),
            "white_evidence": r.get("white_evidence"),
        }
        out[f"{fi}:{tid}"] = entry
        out[tid] = entry
    return out
