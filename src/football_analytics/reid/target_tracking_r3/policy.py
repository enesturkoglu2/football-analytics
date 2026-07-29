"""Preregistered Target Tracking R3 short-occlusion bridge policy."""

from __future__ import annotations

from typing import Any

POLICY_SCHEMA = "target_tracking_r3_bridge_policy_v1"

# Derived from R2 kit bins, 30fps short-video geometry, and Stage5 crop gates.
# Human ≈1.6s transition is reported separately — never hard-coded as truth.
R3_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA,
    "kit_config_path": "configs/reid/kit_descriptor_stage5b.yaml",
    "human_reported_transition_sec_approx": 1.6,
    "human_reported_transition_is_algorithm_truth": False,
    # Boundary refinement (earlier than R2 confirm=5 when white already in seed)
    "refine_confirm_frames": 2,
    "refine_white_frac_high": 0.22,
    "refine_yellow_frac_collapse": 0.40,  # vs seed baseline
    "refine_contam_iou_min": 0.25,
    "refine_severe_contam_coverage": 0.40,
    # Bridge state: last clean observations only
    "bridge_history_frames": 8,
    "min_crop_area_px": 900,
    "min_laplacian_variance": 12.0,
    "max_contam_coverage_for_template": 0.20,
    # Optical flow (OpenCV LK)
    "lk_win_size": 21,
    "lk_max_level": 3,
    "lk_max_corners": 40,
    "lk_quality_level": 0.01,
    "lk_min_distance": 4,
    "lk_block_size": 7,
    "fb_error_max_px": 2.5,
    "min_reliable_flow_points": 6,
    "min_flow_inlier_ratio": 0.45,
    # Bridge window (dynamic caps — not a random fixed duration)
    "max_bridge_frames": 45,
    "long_gap_frames": 60,
    "snap_confirm_frames": 3,
    "overlap_extend_after_contam_frames": 12,
    # Detector snapping
    "max_snap_center_dist_px": 55.0,
    "min_snap_iou": 0.15,
    "max_scale_ratio": 2.0,
    "min_scale_ratio": 0.5,
    "min_score_margin": 0.12,
    "max_velocity_px_per_frame": 45.0,
    "cross_team_hard_reject": True,
    "unknown_kit_does_not_hard_reject": True,
    "appearance_reid": "UNAVAILABLE",
}
