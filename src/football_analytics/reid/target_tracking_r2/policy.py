"""Preregistered Target Tracking R2 purity / change-point / stitch policy."""

from __future__ import annotations

from typing import Any

POLICY_SCHEMA = "target_tracking_r2_purity_policy_v1"

# Derived from Stage5B kit bins + short-video 30fps geometry; not random.
# Human visual transition ≈1.6s is reported separately — not hard-coded as truth.
R2_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA,
    "kit_config_path": "configs/reid/kit_descriptor_stage5b.yaml",
    "quality_config_path": "configs/reid/crop_quality_policy_stage5a.yaml",
    "purity_config_path": "configs/reid/track_purity_audit_stage5b3.yaml",
    "source_run_id": "sv_run_20260727T234854Z",
    "human_reported_transition_sec_approx": 1.6,
    "human_reported_transition_is_algorithm_truth": False,
    # Crop reliability (measurement gates — weak crops → UNKNOWN kit)
    "min_crop_area_px": 900,
    "min_crop_side_px": 18,
    "min_laplacian_variance": 12.0,
    "min_dominant_fraction_for_reliable_kit": 0.18,
    "min_chromatic_ratio_for_chromatic_kit": 0.08,
    # Seed kit baseline window around HUMAN_SEED frame
    "seed_baseline_radius_frames": 8,
    "seed_expected_kit": "yellow",
    # Change-point temporal consistency (no one-frame splits)
    "change_confirm_frames": 5,
    "yellow_frac_low": 0.06,
    "white_frac_high": 0.22,
    "family_l1_change_min": 0.55,
    "contamination_support_min": 0.08,
    "sample_every_frame": 1,
    # Segment stitch (local, conservative)
    "max_local_gap_frames": 45,
    "max_auto_stitch_gap_frames": 30,
    "max_auto_center_displacement_px": 80,
    "max_candidate_center_displacement_px": 200,
    "min_scale_ratio": 0.5,
    "max_scale_ratio": 2.0,
    "min_candidate_duration_frames": 15,
    "min_score_margin": 0.18,
    "max_auto_cost": 0.65,
    "max_velocity_px_per_frame": 45.0,
    "cross_team_hard_reject": True,
    "unknown_kit_does_not_hard_reject": True,
    "appearance_reid": "UNAVAILABLE",
}
