"""Preregistered Target Tracking R1 conservative stitch thresholds.

Derived from short-video sv_run_20260727T234854Z track-end→birth
spatial/temporal distribution (n≈8966 pairs) and Stage-4B linking policy
(exact-frame hard reject; no automatic cosine acceptance).
"""

from __future__ import annotations

from typing import Any

POLICY_SCHEMA = "target_tracking_r1_stitch_policy_v1"

# Evidence: dist_p10≈69px, dist_p25≈163px, gap_p10=3, gap_p50=25 @ 1326×750 / 30fps.
STITCH_POLICY: dict[str, Any] = {
    "schema_version": POLICY_SCHEMA,
    "source_run_id": "sv_run_20260727T234854Z",
    "source_distribution": {
        "pair_count": 8966,
        "center_displacement_px_p10": 69.17,
        "center_displacement_px_p25": 163.49,
        "center_displacement_px_p50": 362.59,
        "gap_frames_p10": 3,
        "gap_frames_p50": 25,
        "resolution": {"width": 1326, "height": 750},
        "fps": 30.0,
    },
    "linking_policy_ref": "configs/reid/linking_policy_stage4b.yaml",
    "automatic_cosine_link_forbidden": True,
    "appearance_evidence": "UNAVAILABLE",
    # Local continuation scope
    "max_local_gap_frames": 45,  # 1.5s — beyond → LONG_GAP_REVIEW_REQUIRED
    "max_auto_stitch_gap_frames": 30,  # 1.0s
    "max_auto_center_displacement_px": 80,  # ~p10–p12
    "max_candidate_center_displacement_px": 200,  # keep in manifest, not auto
    "min_scale_ratio": 0.5,
    "max_scale_ratio": 2.0,
    "min_candidate_duration_frames": 15,
    "min_score_margin": 0.18,  # best vs 2nd normalized cost margin
    # Combined cost ceiling: within auto gap/disp windows, base cost can approach
    # ~0.45+0.45 plus small scale/motion terms; 0.65 admits mid-window clear winners
    # (e.g. gap=16/30, disp=37/80 → ~0.47) while rejecting weak combined geometry.
    "max_auto_cost": 0.65,
    "max_velocity_px_per_frame": 45.0,  # hard impossible-motion
    "border_margin_px": 40,
    "require_review_eligible": True,
    "max_chain_stitches": 25,
    "exact_frame_conflict_hard_reject": True,
}
