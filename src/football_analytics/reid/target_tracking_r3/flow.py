"""Sparse Lucas–Kanade optical flow with forward/backward consistency."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from football_analytics.reid.target_tracking_r3.policy import R3_POLICY


def _clamp_bbox(bbox: Sequence[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def extract_quality_gated_features(
    gray: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    policy: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """goodFeaturesToTrack inside bbox; empty → None (BRIDGE_FLOW_UNRELIABLE)."""
    pol = dict(policy or R3_POLICY)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox_xyxy, w, h)
    # Prefer torso band (upper-mid crop) for kit-stable features
    band_y1 = y1 + int(0.15 * (y2 - y1))
    band_y2 = y1 + int(0.70 * (y2 - y1))
    mask = np.zeros_like(gray)
    mask[band_y1:band_y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(pol["lk_max_corners"]),
        qualityLevel=float(pol["lk_quality_level"]),
        minDistance=float(pol["lk_min_distance"]),
        blockSize=int(pol["lk_block_size"]),
        mask=mask,
    )
    if pts is None or len(pts) < int(pol["min_reliable_flow_points"]):
        # fallback full bbox
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(pol["lk_max_corners"]),
            qualityLevel=float(pol["lk_quality_level"]),
            minDistance=float(pol["lk_min_distance"]),
            blockSize=int(pol["lk_block_size"]),
            mask=mask,
        )
    if pts is None or len(pts) < int(pol["min_reliable_flow_points"]):
        return None
    return pts.astype(np.float32)


def track_flow_forward_backward(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    prev_pts: np.ndarray,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """LK forward then backward; keep FB-consistent inliers."""
    pol = dict(policy or R3_POLICY)
    win = (int(pol["lk_win_size"]), int(pol["lk_win_size"]))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, prev_pts, None, winSize=win, maxLevel=int(pol["lk_max_level"]), criteria=criteria
    )
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(
        next_gray, prev_gray, nxt, None, winSize=win, maxLevel=int(pol["lk_max_level"]), criteria=criteria
    )
    if nxt is None or back is None or st is None or st_b is None:
        return {
            "reliable": False,
            "reason": "BRIDGE_FLOW_UNRELIABLE",
            "inlier_points": None,
            "median_delta": None,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
        }

    fb_err = np.linalg.norm(prev_pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
    forward_ok = st.reshape(-1).astype(bool)
    back_ok = st_b.reshape(-1).astype(bool)
    ok = forward_ok & back_ok & (fb_err <= float(pol["fb_error_max_px"]))
    n_in = int(ok.sum())
    ratio = float(n_in) / float(len(prev_pts)) if len(prev_pts) else 0.0
    if n_in < int(pol["min_reliable_flow_points"]) or ratio < float(pol["min_flow_inlier_ratio"]):
        return {
            "reliable": False,
            "reason": "BRIDGE_FLOW_UNRELIABLE",
            "inlier_points": None,
            "median_delta": None,
            "n_inliers": n_in,
            "inlier_ratio": ratio,
            "fb_errors": fb_err[ok].tolist() if n_in else [],
        }

    src = prev_pts.reshape(-1, 2)[ok]
    dst = nxt.reshape(-1, 2)[ok]
    deltas = dst - src
    med = np.median(deltas, axis=0)
    return {
        "reliable": True,
        "reason": "FLOW_OK",
        "inlier_points": dst.astype(np.float32).reshape(-1, 1, 2),
        "src_points": src.astype(np.float32),
        "dst_points": dst.astype(np.float32),
        "median_delta": (float(med[0]), float(med[1])),
        "n_inliers": n_in,
        "inlier_ratio": ratio,
        "fb_errors": fb_err[ok].tolist(),
    }


def project_bbox(
    bbox_xyxy: Sequence[float],
    median_delta: tuple[float, float],
    *,
    scale_factor: float = 1.0,
) -> list[float]:
    """Translate (+ optional isotropic scale about center). Flow alone is not final target."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * float(scale_factor)
    h = (y2 - y1) * float(scale_factor)
    dx, dy = median_delta
    ncx, ncy = cx + dx, cy + dy
    return [ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2]
